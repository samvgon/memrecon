"""Ingestion: turning a finished session into memories (D35, D36).

Three ways, one per kind of arm:

- **note** (`ingest_note`): the agent reads the session (never the
  `share_memory` flags) and up to 10 related existing memories found by
  retrieval, and writes notes; changes supersede the memories they replace,
  forgets retire them (D29). Prompt: prompts/ingest_note_v1.md.
- **log** (`ingest_log`): the agent chooses which of the user's words to keep,
  with the same criteria, and returns them verbatim; the harness checks each
  is an exact part of that turn and otherwise keeps the whole turn. The log is
  append-only. Prompt: prompts/ingest_log_v1.md.
- **oracle** (`ingest_oracle`): the flagged turns, verbatim, with no model call.
  An upper-bound reference that uses Memora's annotations, labelled as such.
- **two-step note** (`ingest_two_step`, D40): as in Mem0, step 1 extracts facts
  from the session without seeing any memory (prompts/extract_note_v1.md);
  step 2 decides, for each fact, against the stored memories most similar to
  it (top 5 reaching `candidate_threshold`), whether it is new, replaces some,
  asks for some to be forgotten, or is already stored (prompts/decide_note_v1.md).
  A fact with no memory reaching the threshold is added without a call, so a
  rewrite needs both a similar memory and the model choosing to change it.

Answers are one JSON object between tags, read strictly and never repaired
(as in the HW1 harness). An unusable answer is a rejected attempt, kept in the
audit log, and the model is told what was wrong and asked again, up to 3
attempts in all; an answer cut off at the output limit is retried with twice
the limit. If all attempts fail, nothing is stored for that session and the
failure is returned, to be counted and reported.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from memrecon.llm import ModelCallError, OutputTruncated
from memrecon.memory import At, MemoryStore, Source

PROMPTS = Path(__file__).resolve().parent / "prompts"
MAX_ATTEMPTS = 3
MAX_RELATED = 10
RELATED_K = 3
CANDIDATE_K = 5  # two-step: similar memories shown per fact, as in Mem0
ACTIONS = ("add", "replace", "forget", "none")


def load_prompt(name: str) -> tuple[str, str]:
    """(system, user template) from prompts/<name>.md; the leading comment is not sent."""
    text = (PROMPTS / f"{name}.md").read_text(encoding="utf-8")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.S)
    _, rest = text.split("=== SYSTEM ===", 1)
    system, user = rest.split("=== USER ===", 1)
    return system.strip(), user.strip()


def answer_format(system: str) -> str:
    """The prompt's answer-format paragraph (from "Answer with exactly one block" to the end), repeated
    in the retry message so a model that described what it would do, instead of doing it, sees the
    form again (as HW1's harness reminded the model of the tool-call format)."""
    i = system.find("Answer with exactly one block")
    return system[i:].strip() if i >= 0 else ""


def render_conversation(session: dict) -> tuple[str, dict[int, dict]]:
    """Numbered lines ("3. User: ...") and a map from line number to turn. Flags are never included."""
    lines, by_line = [], {}
    for n, t in enumerate(session.get("conversation", []), start=1):
        who = "User" if t.get("speaker") == "user_agent" else "Assistant"
        lines.append(f"{n}. {who}: {t.get('message', '')}")
        by_line[n] = t
    return "\n".join(lines), by_line


class UnusableAnswer(ValueError):
    """The model's answer could not be used; the message is what it is told."""


def extract_block(reply: str, tag: str) -> dict:
    """The JSON object between <tag> and </tag>: exactly one block, valid JSON, never repaired."""
    open_, close = f"<{tag}>", f"</{tag}>"
    if reply.count(open_) != 1 or reply.count(close) != 1:
        raise UnusableAnswer(f"Answer with exactly one {open_}...{close} block.")
    inner = reply.split(open_, 1)[1].split(close, 1)[0]
    if not inner.strip() or reply.index(open_) > reply.index(close):
        raise UnusableAnswer(f"The {open_}...{close} block is empty or out of order.")
    try:
        data = json.loads(inner)
    except json.JSONDecodeError as e:
        raise UnusableAnswer(f"The text inside {open_}...{close} is not valid JSON ({e.msg}).") from None
    if not isinstance(data, dict):
        raise UnusableAnswer("The block must hold one JSON object.")
    return data


def _ints(value, what: str) -> list[int]:
    if not isinstance(value, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in value):
        raise UnusableAnswer(f'"{what}" must be a list of numbers.')
    return value


def parse_note_answer(reply: str, labels: dict[str, str], lines: dict[int, dict], reject_duplicates: bool = True) -> dict:
    """Check a note-arm answer against the memories and lines that were shown.

    reject_duplicates (on from prompt v2): the same memory text twice makes the answer unusable.
    It is off for v1, so E16 and E17, which ran v1, replay exactly as they ran.
    """
    data = extract_block(reply, "memories")
    if set(data) != {"new", "change", "forget"}:
        raise UnusableAnswer('The object must have exactly the keys "new", "change" and "forget".')
    shown = {int(k) for k in labels}
    new = []
    if not isinstance(data["new"], list):
        raise UnusableAnswer('"new" must be a list.')
    for item in data["new"]:
        if not isinstance(item, dict) or set(item) != {"memory", "lines"} or not isinstance(item["memory"], str) or not item["memory"].strip():
            raise UnusableAnswer('Each item in "new" must be {"memory": "...", "lines": [...]}.')
        new.append((item["memory"].strip(), _check_lines(_ints(item["lines"], "lines"), lines)))
    change = []
    if not isinstance(data["change"], list):
        raise UnusableAnswer('"change" must be a list.')
    for item in data["change"]:
        if not isinstance(item, dict) or set(item) != {"replaces", "memory", "lines"} or not isinstance(item["memory"], str) or not item["memory"].strip():
            raise UnusableAnswer('Each item in "change" must be {"replaces": [...], "memory": "...", "lines": [...]}.')
        replaces = _ints(item["replaces"], "replaces")
        if not replaces or not set(replaces) <= shown:
            raise UnusableAnswer(f'"replaces" must list numbers of the existing memories shown ({sorted(shown) or "none were shown"}).')
        change.append((replaces, item["memory"].strip(), _check_lines(_ints(item["lines"], "lines"), lines)))
    forget = _ints(data["forget"], "forget")
    if not set(forget) <= shown:
        raise UnusableAnswer(f'"forget" must list numbers of the existing memories shown ({sorted(shown) or "none were shown"}).')
    texts = [t for t, _ in new] + [t for _, t, _ in change]
    if reject_duplicates and len(texts) != len(set(texts)):
        raise UnusableAnswer("The same memory appears more than once; write each memory once.")
    used = [n for r, _, _ in change for n in r] + forget
    if len(used) != len(set(used)):
        raise UnusableAnswer("Each existing memory can be replaced or forgotten only once.")
    return {"new": new, "change": change, "forget": forget}


def parse_extract_answer(reply: str, lines: dict[int, dict]) -> list[tuple[str, list[int]]]:
    """Two-step, step 1: the facts and the user lines each comes from; the same fact twice is unusable."""
    data = extract_block(reply, "facts")
    if set(data) != {"facts"} or not isinstance(data["facts"], list):
        raise UnusableAnswer('The object must be {"facts": [...]}.')
    out = []
    for item in data["facts"]:
        if not isinstance(item, dict) or set(item) != {"fact", "lines"} or not isinstance(item["fact"], str) or not item["fact"].strip():
            raise UnusableAnswer('Each item in "facts" must be {"fact": "...", "lines": [...]}.')
        out.append((item["fact"].strip(), _check_lines(_ints(item["lines"], "lines"), lines)))
    texts = [t for t, _ in out]
    if len(texts) != len(set(texts)):
        raise UnusableAnswer("The same fact appears more than once; write each fact once.")
    return out


def parse_decision(reply: str, labels: dict[str, str]) -> tuple[str, list[str], str]:
    """Two-step, step 2: (action, IDs of the memories it replaces or forgets, updated memory text)."""
    data = extract_block(reply, "decision")
    if set(data) != {"action", "replaces", "memory"}:
        raise UnusableAnswer('The object must have exactly the keys "action", "replaces" and "memory".')
    action, memory = data["action"], data["memory"]
    if action not in ACTIONS:
        raise UnusableAnswer('"action" must be one of "add", "replace", "forget" or "none".')
    if not isinstance(memory, str):
        raise UnusableAnswer('"memory" must be a string.')
    nums = _ints(data["replaces"], "replaces")
    shown = sorted(int(k) for k in labels)
    if action in ("replace", "forget"):
        if not nums or not set(nums) <= set(shown) or len(nums) != len(set(nums)):
            raise UnusableAnswer(f'For "{action}", "replaces" must list numbers of the memories shown ({shown}), each once.')
    elif nums:
        raise UnusableAnswer(f'For "{action}", "replaces" must be empty.')
    if action == "replace" and not memory.strip():
        raise UnusableAnswer('For "replace", "memory" must be the complete updated memory.')
    if action != "replace" and memory.strip():
        raise UnusableAnswer(f'For "{action}", "memory" must be empty.')
    return action, [labels[str(n)] for n in nums], memory.strip()


def _check_lines(nums: list[int], lines: dict[int, dict]) -> list[int]:
    bad = [n for n in nums if n not in lines or lines[n].get("speaker") != "user_agent"]
    if not nums or bad:
        raise UnusableAnswer('"lines" must list one or more numbers of the user\'s lines in the conversation.')
    return nums


def parse_log_answer(reply: str, lines: dict[int, dict]) -> list[tuple[int, str]]:
    data = extract_block(reply, "log")
    if set(data) != {"keep"} or not isinstance(data["keep"], list):
        raise UnusableAnswer('The object must be {"keep": [...]}.')
    out = []
    for item in data["keep"]:
        if not isinstance(item, dict) or set(item) != {"line", "words"} or not isinstance(item["words"], str) or not item["words"].strip():
            raise UnusableAnswer('Each item in "keep" must be {"line": N, "words": "..."}.')
        n = item["line"]
        if not isinstance(n, int) or n not in lines or lines[n].get("speaker") != "user_agent":
            raise UnusableAnswer('"line" must be the number of one of the user\'s lines.')
        out.append((n, item["words"].strip()))
    return out


@dataclass
class Outcome:
    """What one session's ingestion did, for counting and reporting."""

    session_id: int
    ok: bool
    attempts: int = 0
    created: list[str] = field(default_factory=list)
    superseded: list[str] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)
    span_fallbacks: int = 0  # log arm: kept words that were not an exact part of the turn
    error: str | None = None
    facts: int = 0  # two-step: facts extracted
    decisions: dict[str, int] = field(default_factory=dict)  # two-step: how each fact was handled
    decide_failures: int = 0  # two-step: facts whose decision failed 3 times (the fact is then added)


class Ingestor:
    def __init__(self, client, store: MemoryStore, retriever=None, persona: str = "", max_tokens: int = 2048,
                 related_threshold: float = 0.0, note_prompt: str = "ingest_note_v1", log_prompt: str = "ingest_log_v1",
                 tag: str | None = None, extract_prompt: str = "extract_note_v1", decide_prompt: str = "decide_note_v1",
                 candidate_k: int = CANDIDATE_K, candidate_threshold: float = 0.0) -> None:
        self.client = client
        self.store = store
        self.retriever = retriever
        self.persona = persona or store.persona
        self.max_tokens = max_tokens
        self.related_threshold = related_threshold
        self.note_prompt = note_prompt
        self.log_prompt = log_prompt
        self.tag = tag
        self.extract_prompt = extract_prompt
        self.decide_prompt = decide_prompt
        self.candidate_k = candidate_k
        self.candidate_threshold = candidate_threshold
        self.last_shown: set[str] = set()  # memories shown to the model for the last session (harness only)
        self.decision_log: list[dict] = []  # two-step: one entry per fact of the last session (harness only)
        self.calls: list[dict] = []  # one entry per answer received: session, attempt, tokens, from cache or not

    # -- shared --------------------------------------------------------------------

    def _ask(self, system: str, user: str, parse, at: At, remind: bool = False, step: str = "ingest") -> tuple[object | None, int, str | None]:
        """Call the model up to MAX_ATTEMPTS times; feed back why an answer was unusable.

        remind (on from prompt v2): the feedback also repeats the prompt's answer format. Off for v1
        prompts, so E16 and E17, whose retry messages did not include it, replay exactly as they ran."""
        reminder = (" " + answer_format(system)) if remind else ""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit = self.max_tokens
        error = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.client.chat(messages, max_tokens=limit, tag=self.tag,
                                          meta={"step": step, "persona": self.persona, "session": at.session_id, "attempt": attempt})
                usage = getattr(result, "usage", None) or {}
                self.calls.append({"session": at.session_id, "step": step, "attempt": attempt, "cached": getattr(result, "cached", False),
                                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
            except OutputTruncated:
                limit *= 2  # retry the same request with twice the room
                error = f"cut off at {limit // 2} tokens"
                continue
            except ModelCallError as e:
                error = str(e)
                if "reasoning" in str(e) or "unreadable" in str(e):
                    continue
                return None, attempt, error  # server unreachable or HTTP error: stop, the run can resume later
            try:
                return parse(result.content), attempt, None
            except UnusableAnswer as e:
                error = str(e)
                messages = messages + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"That answer could not be used: {e} Answer again, following the instructions exactly.{reminder}"},
                ]
        return None, MAX_ATTEMPTS, error

    def _related(self, session: dict, at: At) -> tuple[str, dict[str, str]]:
        self.last_shown = set()
        if self.retriever is None or not self.store.active():
            return "(none)", {}
        self.retriever.sync(self.store)
        best: dict[str, float] = {}
        for t in session.get("conversation", []):
            if t.get("speaker") != "user_agent":
                continue
            for mid, score in self.retriever.search(t.get("message", ""), k=RELATED_K, threshold=self.related_threshold):
                best[mid] = max(score, best.get(mid, -1.0))
        ids = [mid for mid, _ in sorted(best.items(), key=lambda kv: -kv[1])[:MAX_RELATED]]
        self.last_shown = set(ids)
        if not ids:
            return "(none)", {}
        return self.store.present(ids, at)

    def _candidates(self, fact: str, exclude: set[str]) -> list[tuple[str, float]]:
        """Two-step: the stored memories most similar to a fact that reach the threshold (not those in `exclude`)."""
        if self.retriever is None or not self.store.active():
            return []
        self.retriever.sync(self.store)
        hits = [(mid, s) for mid, s in self.retriever.scores(fact) if mid not in exclude and s >= self.candidate_threshold]
        return hits[: self.candidate_k]

    def _sources(self, session: dict, nums: list[int], lines: dict[int, dict]) -> list[Source]:
        return [Source(self.persona, session["session_id"], lines[n].get("turn", n)) for n in nums]

    # -- the three kinds of ingestion -------------------------------------------------------

    def ingest_note(self, session: dict) -> Outcome:
        at = At(session["session_id"], 0)
        system, template = load_prompt(self.note_prompt)
        conversation, lines = render_conversation(session)
        memories, labels = self._related(session, at)
        user = template.format(date=session.get("date", ""), memories=memories, conversation=conversation)
        strict = self.note_prompt != "ingest_note_v1"
        answer, attempts, error = self._ask(system, user, lambda r: parse_note_answer(r, labels, lines, reject_duplicates=strict), at, remind=strict)
        out = Outcome(session["session_id"], ok=answer is not None, attempts=attempts, error=error)
        if answer is None:
            return out
        date = session.get("date", "")
        for text, nums in answer["new"]:
            out.created.append(self.store.note(text, self._sources(session, nums, lines), at, date=date))
        for replaces, text, nums in answer["change"]:
            ids = [labels[str(n)] for n in replaces]
            out.created.append(self.store.note(text, self._sources(session, nums, lines), at, supersedes=ids, date=date))
            out.superseded += ids
        if answer["forget"]:
            ids = [labels[str(n)] for n in answer["forget"]]
            self.store.forget(ids, at)
            out.forgotten += ids
        return out

    def ingest_two_step(self, session: dict) -> Outcome:
        """Two-step note ingestion (D40): extract facts, then decide each against its most similar memories."""
        at = At(session["session_id"], 0)
        date = session.get("date", "")
        system, template = load_prompt(self.extract_prompt)
        conversation, lines = render_conversation(session)
        user = template.format(date=date, conversation=conversation)
        self.last_shown, self.decision_log = set(), []
        facts, attempts, error = self._ask(system, user, lambda r: parse_extract_answer(r, lines), at, remind=True, step="extract")
        out = Outcome(session["session_id"], ok=facts is not None, attempts=attempts, error=error)
        if facts is None:
            return out
        out.facts = len(facts)
        dsys, dtemplate = load_prompt(self.decide_prompt)
        added_here: set[str] = set()  # facts added from this session are not shown to its later facts
        for k, (text, nums) in enumerate(facts):
            sources = self._sources(session, nums, lines)
            others = "\n".join(f"- {t}" for j, (t, _) in enumerate(facts) if j != k) or "(none)"
            cands = self._candidates(text, added_here)
            entry = {"fact": text, "candidates": cands, "action": None, "ids": []}
            self.decision_log.append(entry)
            if not cands:
                out.created.append(self.store.note(text, sources, at, date=date))
                added_here.add(out.created[-1])
                entry["action"] = "add (no candidate)"
                out.decisions[entry["action"]] = out.decisions.get(entry["action"], 0) + 1
                continue
            shown, labels = self.store.present([mid for mid, _ in cands], at)
            self.last_shown |= set(labels.values())
            # other_facts is used from decide_note_v3 on (D43); earlier templates have no such field
            duser = dtemplate.format(date=date, fact=text, memories=shown, other_facts=others)
            decision, _, derr = self._ask(dsys, duser, lambda r: parse_decision(r, labels), at, remind=True, step="decide")
            if decision is None:
                out.decide_failures += 1
                out.error = derr
                decision = ("add", [], "")  # a fact whose decision failed is kept, never dropped
                entry["action"] = "add (decision failed)"
            action, ids, memory = decision
            if action == "replace" and len(ids) == 1 and memory == self.store.get(ids[0]).text:
                # a rewrite with no change is not a rewrite (D42): the memory stays as it was, depth unchanged
                entry["unchanged"] = ids  # harness log only: the memory the model returned unchanged
                action, ids, memory = "none", [], ""
                entry["action"] = "none (identical replace)"
            entry["action"] = entry["action"] or action
            entry["ids"] = ids
            out.decisions[entry["action"]] = out.decisions.get(entry["action"], 0) + 1
            if action == "add":
                out.created.append(self.store.note(text, sources, at, date=date))
                added_here.add(out.created[-1])
            elif action == "replace":
                out.created.append(self.store.note(memory, sources, at, supersedes=ids, date=date))
                out.superseded += ids
            elif action == "forget":
                self.store.forget(ids, at)
                out.forgotten += ids
        return out

    def ingest_log(self, session: dict) -> Outcome:
        at = At(session["session_id"], 0)
        system, template = load_prompt(self.log_prompt)
        conversation, lines = render_conversation(session)
        user = template.format(date=session.get("date", ""), conversation=conversation)
        answer, attempts, error = self._ask(system, user, lambda r: parse_log_answer(r, lines), at)
        out = Outcome(session["session_id"], ok=answer is not None, attempts=attempts, error=error)
        if answer is None:
            return out
        for n, words in answer:
            turn_text = lines[n].get("message", "")
            if words not in turn_text:  # not the user's exact words: keep the whole turn instead
                words = turn_text
                out.span_fallbacks += 1
            out.created.append(self.store.log(words, self._sources(session, [n], lines), at, date=session.get("date", "")))
        return out

    def ingest_oracle(self, session: dict) -> Outcome:
        """Upper-bound reference: Memora's flagged turns, verbatim (uses the annotations; no model call)."""
        at = At(session["session_id"], 0)
        out = Outcome(session["session_id"], ok=True)
        if session.get("operation") not in ("add", "update"):
            return out
        for t in session.get("conversation", []):
            if t.get("share_memory") and t.get("speaker") == "user_agent":
                src = [Source(self.persona, session["session_id"], t.get("turn", 0))]
                out.created.append(self.store.log(t["message"], src, at, date=session.get("date", "")))
        return out
