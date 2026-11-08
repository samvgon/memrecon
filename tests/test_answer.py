"""Tests for answering Memora's questions from memory (D60)."""
import json

import pytest

from memrecon.answer import Answerer, parse_answer, questions
from memrecon.ingest import UnusableAnswer, load_prompt
from memrecon.memory import At, MemoryStore, Source
from memrecon.retrieval import Retriever
from test_recall import ScriptedClient, ToyEmbedder

DATA = {"questions": {
    "remembering": [{"question_id": "q1", "question": "What is my budget?", "question_date": "2025-06-07",
                     "evaluation": {"evaluation_questions": [
                         {"evaluation_question_id": "q1_0", "evaluation_question": "Does the response mention $1.7M?",
                          "expected_answer": "yes", "evaluation_type": "memory_presence"}]}}],
    "recommending": [{"question_id": "q2", "question": "Coffee?", "question_date": "2025-06-07", "evaluation": {"evaluation_questions": []}}],
}}


def ans(text):
    return "<answer>" + json.dumps({"text": text}) + "</answer>"


def test_questions_and_prompt():
    qs = questions(DATA)
    assert [q["question_id"] for q in qs] == ["q1", "q2"] and qs[0]["category"] == "remembering"
    assert qs[0]["sub_questions"][0] == {"id": "q1_0", "text": "Does the response mention $1.7M?", "expected": "yes", "type": "memory_presence"}
    system, user = load_prompt("answer_v1")
    assert all(s in user for s in ("{date}", "{question}", "{memories}")) and "share_memory" not in system + user
    assert parse_answer(ans(" x ")) == "x"
    with pytest.raises(UnusableAnswer):
        parse_answer(ans(""))


def test_answer_shows_memories_newest_first_and_changes_nothing():
    store = MemoryStore("p", mode="note")
    store.note("The budget is $1.7M. budget", [Source("p", 1, 1)], At(1, 0), date="2025-06-01")
    store.note("The new budget is $1.5M. budget", [Source("p", 2, 1)], At(2, 0), date="2025-06-03")
    store.note("Coffee was $5.", [Source("p", 3, 1)], At(3, 0), date="2025-06-02")
    ref = store.content_json()
    client = ScriptedClient("no block", ans("Your budget is $1.5M."))
    a = Answerer(client, store, Retriever(ToyEmbedder()), persona="p", k=2).answer(questions(DATA)[0], 1)
    assert a.text == "Your budget is $1.5M." and a.attempts == 2 and len(a.shown) == 2
    user = client.calls[0]["messages"][1]["content"]
    assert "Today's date: 2025-06-07" in user and user.index("(2025-06-03)") < user.index("(2025-06-01)")
    assert "Coffee" not in user  # only the top k
    assert store.content_json() == ref
