"""Answering Memora's questions from memory (D60; FAMA, D18; the judge check, D21).

Each of a persona's 15 questions is asked once, on its question date (the last day of the week), against
the store as it stands at the end of the history. Retrieval finds the `ANSWER_K` live memories most
similar to the question (no threshold; Memora's own runs use 50), which are shown newest first with their
dates (as recall shows memories, D59) and no scores. The agent answers in one block, read strictly, with
the same retries as ingestion and recall (D36). The answer is then judged against the question's yes/no
sub-questions (`evaluation_questions`): "should mention" (`memory_presence`) and "should not mention"
(`forgetting_absence`), and FAMA is computed from those (fama.py).

Nothing here changes the store except the "present" events that record what was shown (D30).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from memrecon.ingest import MAX_ATTEMPTS, UnusableAnswer, answer_format, extract_block, load_prompt
from memrecon.llm import ModelCallError, OutputTruncated
from memrecon.memory import At, MemoryStore
from memrecon.recall import by_date

ANSWER_PROMPT = "answer_v1"  # D60
ANSWER_K = 50  # D60, as Memora's runs
CATEGORIES = ("remembering", "reasoning", "recommending")


def questions(data: dict) -> list[dict]:
    """A persona's questions from its evaluation_questions_<persona>.json, in file order, each with its
    category and its yes/no sub-questions (id, text, expected answer, type)."""
    out = []
    for cat in CATEGORIES:
        for q in data["questions"].get(cat, []):
            subs = [{"id": e["evaluation_question_id"], "text": e["evaluation_question"],
                     "expected": e["expected_answer"], "type": e["evaluation_type"]}
                    for e in q.get("evaluation", {}).get("evaluation_questions", [])]
            out.append({"category": cat, "question_id": q["question_id"], "question": q["question"],
                        "date": q.get("question_date", ""), "sub_questions": subs})
    return out


def parse_answer(reply: str) -> str:
    data = extract_block(reply, "answer")
    text = data.get("text")
    if set(data) != {"text"} or not isinstance(text, str) or not text.strip():
        raise UnusableAnswer('The block must be {"text": "..."} with the answer.')
    return text.strip()


@dataclass
class Answer:
    question_id: str
    shown: list[str] = field(default_factory=list)  # memory IDs shown, in the order shown (newest first)
    scores: dict[str, float] = field(default_factory=dict)  # their similarity to the question (harness only)
    text: str | None = None  # None if no usable answer
    error: str = ""
    attempts: int = 0


class Answerer:
    def __init__(self, client, store: MemoryStore, retriever, persona: str = "", prompt: str = ANSWER_PROMPT,
                 k: int = ANSWER_K, max_tokens: int = 1024, tag: str | None = None) -> None:
        self.client = client
        self.store = store
        self.retriever = retriever
        self.persona = persona or store.persona
        self.prompt = prompt
        self.k = k
        self.max_tokens = max_tokens
        self.tag = tag

    def answer(self, q: dict, n: int = 0) -> Answer:
        """Answer one question (`questions()` form); `n` numbers it for the event file's position."""
        out = Answer(q["question_id"])
        self.retriever.sync(self.store)
        hits = self.retriever.search(q["question"], k=self.k, threshold=-1.0)
        out.scores = {mid: round(s, 4) for mid, s in hits}
        out.shown = by_date(self.store, [mid for mid, _ in hits])
        at = At(-1, n)  # after the last session; -1 marks answering in the event file
        shown, _ = self.store.present(out.shown, at)
        system, template = load_prompt(self.prompt)
        user = template.format(date=q.get("date", ""), question=q["question"], memories=shown or "(none)")
        out.text, out.error, out.attempts = self._ask(system, user, q["question_id"])
        return out

    def _ask(self, system: str, user: str, qid: str):
        reminder = " " + answer_format(system)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        limit, error = self.max_tokens, ""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                result = self.client.chat(messages, max_tokens=limit, tag=self.tag,
                                          meta={"step": "answer", "persona": self.persona, "question": qid, "attempt": attempt})
            except OutputTruncated:
                limit *= 2
                error = f"cut off at {limit // 2} tokens"
                continue
            except ModelCallError as e:
                error = str(e)
                if "reasoning" in error or "unreadable" in error:
                    continue
                return None, error, attempt
            try:
                return parse_answer(result.content), "", attempt
            except UnusableAnswer as e:
                error = str(e)
                messages = messages + [
                    {"role": "assistant", "content": result.content},
                    {"role": "user", "content": f"That answer could not be used: {e} Answer again, following the instructions exactly.{reminder}"},
                ]
        return None, error, MAX_ATTEMPTS


def to_json(a: Answer) -> str:
    return json.dumps(a.__dict__)
