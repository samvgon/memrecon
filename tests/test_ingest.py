"""Tests for ingestion (D35, D36), with a scripted stand-in for the model."""

import json

import pytest

from memrecon.ingest import (
    MAX_ATTEMPTS,
    Ingestor,
    UnusableAnswer,
    extract_block,
    load_prompt,
    parse_log_answer,
    parse_note_answer,
    render_conversation,
)
from memrecon.llm import ModelCallError, OutputTruncated, ThinkingError
from memrecon.memory import At, MemoryStore, Source

SESSION = {
    "session_id": 13,
    "date": "2025-06-01",
    "operation": "add",
    "conversation": [
        {"turn": 1, "speaker": "user_agent", "message": "Hey there! How's it going?", "share_memory": False},
        {"turn": 2, "speaker": "ai_agent", "message": "Going well, thanks!", "share_memory": False},
        {"turn": 3, "speaker": "user_agent", "message": "Speaking of costs, I spent $11.83 on lunch today.", "share_memory": True},
        {"turn": 4, "speaker": "ai_agent", "message": "Did you get anything good?", "share_memory": False},
    ],
}


class Reply:
    def __init__(self, content):
        self.content = content


class ScriptedClient:
    """Returns (or raises) the scripted items in order and records the messages it was sent."""

    def __init__(self, *items):
        self.items = list(items)
        self.calls = []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        item = self.items.pop(0)
        if isinstance(item, Exception):
            raise item
        return Reply(item)


def note_reply(new=(), change=(), forget=()):
    return "<memories>" + json.dumps({"new": list(new), "change": list(change), "forget": list(forget)}) + "</memories>"


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def sync(self, store):
        pass

    def search(self, query, k=3, threshold=0.0):
        return self.hits


def test_prompts_load_and_flags_never_reach_the_model():
    for name in ("ingest_note_v1", "ingest_log_v1"):
        system, user = load_prompt(name)
        assert "share_memory" not in system + user and "<!--" not in system
        assert "{conversation}" in user and "{date}" in user
    text, lines = render_conversation(SESSION)
    assert text.splitlines()[2] == "3. User: Speaking of costs, I spent $11.83 on lunch today."
    assert "share_memory" not in text and "True" not in text
    assert lines[2]["speaker"] == "ai_agent"


def test_note_ingestion_creates_memories_with_sources_and_date():
    store = MemoryStore("financial_analyst")
    client = ScriptedClient(note_reply(new=[{"memory": "The user spent $11.83 on lunch on 2025-06-01.", "lines": [3]}]))
    out = Ingestor(client, store).ingest_note(SESSION)
    assert out.ok and out.attempts == 1 and len(out.created) == 1
    m = store.get(out.created[0])
    assert (m.text, m.depth, m.date) == ("The user spent $11.83 on lunch on 2025-06-01.", 1, "2025-06-01")
    assert m.sources == [Source("financial_analyst", 13, 3)]
    sent = client.calls[0]["messages"][1]["content"]
    assert "Conversation date: 2025-06-01" in sent and "(none)" in sent
    assert client.calls[0]["meta"]["step"] == "ingest"


def test_note_change_and_forget_apply_through_labels():
    store = MemoryStore("p")
    a = store.note("The user spent $10.00 on lunch.", [Source("p", 5, 3)], At(5, 0), date="2025-06-01")
    b = store.note("The user likes jazz.", [Source("p", 6, 3)], At(6, 0), date="2025-06-01")
    retriever = FakeRetriever([(b, 0.9), (a, 0.8)])  # shown as [1] = b, [2] = a
    reply = note_reply(change=[{"replaces": [2], "memory": "The user spent $11.83 on lunch.", "lines": [3]}], forget=[1])
    client = ScriptedClient(reply)
    out = Ingestor(client, store, retriever).ingest_note(SESSION)
    assert out.ok and out.superseded == [a] and out.forgotten == [b]
    new = store.get(out.created[0])
    assert new.depth == 2 and new.parents == [a]
    assert store.get(a).retired == "superseded" and store.get(b).retired == "forgotten"
    shown = client.calls[0]["messages"][1]["content"]
    assert "[1] (2025-06-01) The user likes jazz." in shown and "m0000" not in shown


def test_unusable_answer_gets_feedback_then_succeeds():
    store = MemoryStore("p")
    client = ScriptedClient("Sure! Here are the memories: none", note_reply())
    out = Ingestor(client, store).ingest_note(SESSION)
    assert out.ok and out.attempts == 2
    second = client.calls[1]["messages"]
    assert second[-2] == {"role": "assistant", "content": "Sure! Here are the memories: none"}
    assert "could not be used" in second[-1]["content"] and "<memories>" in second[-1]["content"]


def test_gives_up_after_three_attempts_and_stores_nothing():
    store = MemoryStore("p")
    client = ScriptedClient("no", "still no", "<memories>{bad json}</memories>")
    out = Ingestor(client, store).ingest_note(SESSION)
    assert not out.ok and out.attempts == MAX_ATTEMPTS and "JSON" in out.error
    assert len(store) == 0


def test_cut_off_answer_is_retried_with_twice_the_room():
    client = ScriptedClient(OutputTruncated("cut", text="", max_tokens=2048), note_reply())
    out = Ingestor(client, MemoryStore("p")).ingest_note(SESSION)
    assert out.ok and [c["max_tokens"] for c in client.calls] == [2048, 4096]


def test_reasoning_is_retried_but_a_dead_server_stops():
    client = ScriptedClient(ThinkingError("returned reasoning text"), note_reply())
    assert Ingestor(client, MemoryStore("p")).ingest_note(SESSION).ok
    dead = ScriptedClient(ModelCallError("agent at http://localhost:8080/v1 not reachable after 4 attempts"))
    out = Ingestor(dead, MemoryStore("p")).ingest_note(SESSION)
    assert not out.ok and out.attempts == 1


@pytest.mark.parametrize(
    "reply,match",
    [
        ("<memories>{}</memories>", "exactly the keys"),
        (note_reply(new=[{"memory": "x", "lines": [2]}]), "user's lines"),  # line 2 is the assistant's
        (note_reply(new=[{"memory": "x", "lines": [99]}]), "user's lines"),
        (note_reply(forget=[1]), "existing memories shown"),  # nothing was shown
        (note_reply(new=[{"memory": "", "lines": [3]}]), "new"),
        ("<memories>" + json.dumps({"new": [], "change": [], "forget": []}) + "</memories><memories></memories>", "exactly one"),
    ],
)
def test_note_answers_are_read_strictly(reply, match):
    _, lines = render_conversation(SESSION)
    with pytest.raises(UnusableAnswer, match=match):
        parse_note_answer(reply, {}, lines)


def test_same_memory_cannot_be_replaced_and_forgotten():
    _, lines = render_conversation(SESSION)
    reply = note_reply(change=[{"replaces": [1], "memory": "x", "lines": [3]}], forget=[1])
    with pytest.raises(UnusableAnswer, match="only once"):
        parse_note_answer(reply, {"1": "m00001"}, lines)


def test_extract_block_never_repairs():
    with pytest.raises(UnusableAnswer):
        extract_block('<memories>{"new": [], "change": [], "forget": [],}</memories>', "memories")  # trailing comma


def test_log_ingestion_keeps_exact_words_or_the_whole_turn():
    store = MemoryStore("p", mode="log")
    reply = '<log>{"keep": [{"line": 3, "words": "I spent $11.83 on lunch today."}]}</log>'
    out = Ingestor(ScriptedClient(reply), store).ingest_log(SESSION)
    m = store.get(out.created[0])
    assert (m.text, m.depth, m.date, out.span_fallbacks) == ("I spent $11.83 on lunch today.", 0, "2025-06-01", 0)
    # reworded words are not accepted as the user's own: the whole turn is kept
    store2 = MemoryStore("p", mode="log")
    reworded = '<log>{"keep": [{"line": 3, "words": "I spent $11.83 on my lunch."}]}</log>'
    out2 = Ingestor(ScriptedClient(reworded), store2).ingest_log(SESSION)
    assert store2.get(out2.created[0]).text == SESSION["conversation"][2]["message"] and out2.span_fallbacks == 1


def test_log_answers_must_point_at_user_lines():
    _, lines = render_conversation(SESSION)
    with pytest.raises(UnusableAnswer):
        parse_log_answer('<log>{"keep": [{"line": 2, "words": "Going well"}]}</log>', lines)


def test_oracle_uses_flagged_turns_without_the_model():
    store = MemoryStore("p", mode="log")
    out = Ingestor(ScriptedClient(), store).ingest_oracle(SESSION)
    assert [store.get(m).text for m in out.created] == ["Speaking of costs, I spent $11.83 on lunch today."]
    assert Ingestor(ScriptedClient(), MemoryStore("p", mode="log")).ingest_oracle({**SESSION, "operation": "delete"}).created == []
