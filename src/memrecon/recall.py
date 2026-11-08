"""Recall at replay time and the write policies (D14, D24, D28, D37, D53).

On every replayed user turn, retrieval looks for memories relevant to what the user just said: the top
3 live memories reaching similarity 0.70 (D37). Each one found is counted (`MemoryStore.recall`).
Recall then has two steps (D54, Sam):

1. **select** (prompts/select_v1.md, one call per turn): the agent sees the memories found, listed by
   date, newest first (D59), and chooses the ones useful for responding to the user's latest message, most useful first, or
   none. Each one chosen is counted (`MemoryStore.select`).
2. **rewrite** (prompts/rewrite_v5b.md, D58; one call per chosen memory, in the order chosen, each on its own):
   the agent rewrites the memory for the conversation right now.

The **write policy** then decides what is stored:

- **discard**: nothing (the baseline every policy is compared with; Stage 0's control);
- **replace**: the rewrite takes the memory's place (depth + 1; the old text stays in its versions);
- **keep**: the rewrite is stored beside it as a new memory, its child (depth + 1).

A rewrite identical to the memory's text is no rewrite under any policy (as D42 for ingestion): the
memory stays as it was. The model is never told which policy runs, so its rewrites are the same under
every policy until the stores differ. Recall happens during a session, before the session is ingested
(ingestion runs once the session ends, D35), so a session's own memories are not recalled within it.

If a call fails (three unusable answers), the counts stand and nothing is stored for it, as under
discard: a failed select rewrites nothing that turn; a failed rewrite skips that memory. Failures are
returned to be counted and reported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date as _date

from memrecon.ingest import MAX_ATTEMPTS, UnusableAnswer, answer_format, extract_block, load_prompt, render_conversation
from memrecon.llm import ModelCallError, OutputTruncated
from memrecon.memory import At, MemoryStore

RECALL_K = 3  # D37
RECALL_THRESHOLD = 0.70  # D37
SELECT_PROMPT = "select_v1"
REWRITE_PROMPT = "rewrite_v5b"  # D58 (was rewrite_v4, D57)


# -- write policies -------------------------------------------------------------------------------

class WritePolicy:
    """What is stored after a recall-time rewrite. One small swappable component (CLAUDE.md)."""

    name = "base"

    def apply(self, store: MemoryStore, mid: str, rewritten: str, at: At) -> str | None:
        """Store (or not) one rewrite of memory `mid`; returns the ID written to, if any."""
        raise NotImplementedError


class Discard(WritePolicy):
    name = "discard"

    def apply(self, store, mid, rewritten, at):
        return None


class Replace(WritePolicy):
    name = "replace"

    def apply(self, store, mid, rewritten, at):
        if rewritten == store.get(mid).text:
            return None
        store.replace(mid, rewritten, at)
        return mid


class Keep(WritePolicy):
    name = "keep"

    def apply(self, store, mid, rewritten, at):
        if rewritten == store.get(mid).text:
            return None
        return store.keep(mid, rewritten, at)


POLICIES = {p.name: p for p in (Discard, Replace, Keep)}


def policy(name: str) -> WritePolicy:
    return POLICIES[name]()


# -- the answers ----------------------------------------------------------------------------------

def parse_select_answer(reply: str, labels: dict[str, str]) -> list[str]:
    """The labels chosen, most useful first: each a memory shown, each once; may be empty."""
    data = extract_block(reply, "useful")
    items = data.get("memories")
    if set(data) != {"memories"} or not isinstance(items, list):
        raise UnusableAnswer('The block must be {"memories": [...]}, a list of numbers (empty if none is useful).')
    out = []
    for x in items:
        label = str(x).strip().strip("[]")
        if label not in labels:
            raise UnusableAnswer(f'Each entry must be the number of a memory shown ({", ".join(labels)}); got {json.dumps(x)}.')
        if label in out:
            raise UnusableAnswer(f"Memory {label} is listed more than once.")
        out.append(label)
    return out


def parse_rewrite_answer(reply: str) -> str:
    """The rewritten memory: a non-empty string."""
    data = extract_block(reply, "memory")
    text = data.get("text")
    if set(data) != {"text"} or not isinstance(text, str) or not text.strip():
        raise UnusableAnswer('The block must be {"text": "..."} with the rewritten memory.')
    return text.strip()


# -- the recall path ---------------------------------------------------------------------------------

@dataclass
class TurnRecall:
    turn: int  # line number in the session
    found: list[str]  # memory IDs retrieval found, most similar first
    scores: list[float]
    shown: list[str] = field(default_factory=list)  # the order step 1 listed them in (newest first)
    selected: list[str] = field(default_factory=list)  # chosen as useful, most useful first
    written: list[str] = field(default_factory=list)  # IDs the policy wrote to
    unchanged: list[str] = field(default_factory=list)  # rewrites identical to the memory
    rewrites: dict[str, str] = field(default_factory=dict)  # memory ID -> its rewrite (harness only, for analysis)
    select_ok: bool = True
    rewrite_failed: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class RecallOutcome:
    session_id: int
    turns: list[TurnRecall] = field(default_factory=list)

    @property
    def found(self) -> int:
        return sum(len(t.found) for t in self.turns)

    @property
    def selected(self) -> int:
        return sum(len(t.selected) for t in self.turns)

    @property
    def failures(self) -> int:
        return sum((not t.select_ok) + len(t.rewrite_failed) for t in self.turns)


def how_long_before(memory_date: str, conversation_date: str) -> str:
    """How long before the conversation the memory's information came from, in words ("" if unknown)."""
    try:
        days = (_date.fromisoformat(conversation_date) - _date.fromisoformat(memory_date)).days
    except (TypeError, ValueError):
        return ""
    if days < 0:
        return ""
    return {0: "the same day as this one", 1: "the day before this one"}.get(days, f"{days} days before this one")


def gap_to_conversation(memory_date: str, conversation_date: str) -> str:
    """The same as `how_long_before`, worded against the conversation date and not "this one" ("" if unknown)."""
    words = how_long_before(memory_date, conversation_date)
    if not words:
        return ""
    if words.startswith("the same day"):
        return "the same date as the conversation"
    return words.replace("the day before this one", "1 day before the conversation date").replace("before this one", "before the conversation date")


def memory_slots(text: str, memory_date: str, conversation_date: str) -> dict[str, str]:
    """What a rewrite prompt may show of one memory. `memory` is the form `present` gives ("(date) text"),
    used by versions 1 to 4; `memory_text`, `memory_date`, `when` ("2 days before this one") and `gap`
    ("2 days before the conversation date") let a prompt show the date on its own line (versions 5a-5c,
    E36). A prompt uses the ones it names; the rest are ignored."""
    return {"memory": f"({memory_date}) {text}" if memory_date else text, "memory_text": text,
            "memory_date": memory_date or "an unknown date", "when": how_long_before(memory_date, conversation_date) or "date unknown",
            "gap": gap_to_conversation(memory_date, conversation_date) or "date unknown"}


def by_date(store: MemoryStore, ids: list[str]) -> list[str]:
    """Newest first (Sam): the date its information came from, then the order it was written, latest first (D54)."""
    order = {mid: n for n, mid in enumerate(store.memories)}
    return sorted(ids, key=lambda mid: (store.get(mid).date or "", order[mid]), reverse=True)


class Recaller:
    def __init__(self, client, store: MemoryStore, retriever, write_policy: WritePolicy, persona: str = "",
                 select_prompt: str = SELECT_PROMPT, rewrite_prompt: str = REWRITE_PROMPT, k: int = RECALL_K,
                 threshold: float = RECALL_THRESHOLD, max_tokens: int = 2048, tag: str | None = None,
                 call_settings: dict | None = None) -> None:
        self.client = client
        self.store = store
        self.retriever = retriever
        self.policy = write_policy
        self.persona = persona or store.persona
        self.select_prompt = select_prompt
        self.rewrite_prompt = rewrite_prompt
        self.k = k
        self.threshold = threshold
        self.max_tokens = max_tokens
        self.tag = tag
        self.call_settings = dict(call_settings or {})  # extra request settings, e.g. {"seed": 1} to ask again (E37)
        self.calls: list[dict] = []  # one entry per answer received (harness only)
        self.feedback: list[str] = []  # why answers were sent back

    def recall_session(self, session: dict) -> RecallOutcome:
        """Every user turn of a session, in order: retrieve, count, select, rewrite each chosen, apply the policy."""
        out = RecallOutcome(session["session_id"])
        conv = session.get("conversation", [])
        for n, t in enumerate(conv, start=1):
            if t.get("speaker") != "user_agent" or not self.store.active():
                continue
            self.retriever.sync(self.store)
            hits = self.retriever.search(t.get("message", ""), k=self.k, threshold=self.threshold)
            if not hits:
                continue
            at = At(session["session_id"], t.get("turn", n))
            tr = TurnRecall(n, [mid for mid, _ in hits], [s for _, s in hits])
            out.turns.append(tr)
            for mid in tr.found:
                self.store.recall(mid, at)
            conversation, _ = render_conversation({"conversation": conv[:n]})
            # step 1: which are useful, most useful first
            tr.shown = by_date(self.store, tr.found)
            shown, labels = self.store.present(tr.shown, at)
            system, template = load_prompt(self.select_prompt)
            user = template.format(date=session.get("date", ""), conversation=conversation, memories=shown)
            chosen, error = self._ask(system, user, lambda r: parse_select_answer(r, labels), at, "select")
            if chosen is None:
                tr.select_ok = False
                tr.errors.append(error or "")
                continue
            tr.selected = [labels[x] for x in chosen]
            # step 2: rewrite each chosen memory on its own, in that order
            system, template = load_prompt(self.rewrite_prompt)
            for mid in tr.selected:
                self.store.select(mid, at)
                self.store.present([mid], at)
                m = self.store.get(mid)
                user = template.format(date=session.get("date", ""), conversation=conversation,
                                       **memory_slots(m.text, m.date, session.get("date", "")))
                text, error = self._ask(system, user, parse_rewrite_answer, at, "rewrite")
                if text is None:
                    tr.rewrite_failed.append(mid)
                    tr.errors.append(error or "")
                    continue
                tr.rewrites[mid] = text
                if text == self.store.get(mid).text:
                    tr.unchanged.append(mid)
                written = self.policy.apply(self.store, mid, text, at)
                if written:
                    tr.written.append(written)
        return out

    def _ask(self, system: str, user: str, parse, at: At, step: str):
        """As ingestion's: up to MAX_ATTEMPTS, feedback with the answer format, twice the room if cut off."""
        reminder = " " + answer_format(system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit, error = self.max_tokens, None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.client.chat(messages, max_tokens=limit, tag=self.tag, **self.call_settings,
                                          meta={"step": step, "persona": self.persona, "session": at.session_id,
                                                "turn": at.turn, "policy": self.policy.name, "attempt": attempt})
                usage = getattr(result, "usage", None) or {}
                self.calls.append({"session": at.session_id, "turn": at.turn, "step": step, "attempt": attempt,
                                   "cached": getattr(result, "cached", False),
                                   "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens")})
            except OutputTruncated:
                limit *= 2
                error = f"cut off at {limit // 2} tokens"
                continue
            except ModelCallError as e:
                error = str(e)
                if "reasoning" in str(e) or "unreadable" in str(e):
                    continue
                return None, error
            try:
                return parse(result.content), None
            except UnusableAnswer as e:
                error = str(e)
                self.feedback.append(f"{step}: {e}")
                messages = messages + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"That answer could not be used: {e} Answer again, following the instructions exactly.{reminder}"},
                ]
        return None, error
