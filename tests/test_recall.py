"""Tests for the recall path and the write policies (D14, D28, D37, D53)."""
import json

import pytest

from memrecon.ingest import UnusableAnswer, load_prompt
from memrecon.memory import At, MemoryStore, Source
from memrecon.recall import Discard, Keep, Recaller, Replace, how_long_before, memory_slots, parse_rewrite_answer, parse_select_answer, policy
from memrecon.retrieval import Retriever

TOPICS = ["budget", "coffee", "steps", "design"]


class ToyEmbedder:
    """Counts of a few topic words, so similarity is predictable (as in test_retrieval)."""

    def embed(self, texts, kind):
        out = []
        for t in texts:
            v = [float(t.lower().count(w)) + 0.01 for w in TOPICS]
            n = sum(x * x for x in v) ** 0.5
            out.append([x / n for x in v])
        return out


class Reply:
    def __init__(self, content):
        self.content = content


class ScriptedClient:
    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return Reply(self.items.pop(0))


def useful(*labels):
    return "<useful>" + json.dumps({"memories": list(labels)}) + "</useful>"


def memory(text):
    return "<memory>" + json.dumps({"text": text}) + "</memory>"


def store_with(*texts):
    s = MemoryStore("p", mode="note")
    for i, t in enumerate(texts):
        s.note(t, [Source("p", 1, i + 1)], At(1, 0), date="2025-06-01")
    return s


SESSION = {"session_id": 5, "date": "2025-06-05", "conversation": [
    {"turn": 1, "speaker": "user_agent", "message": "Hi there!"},
    {"turn": 2, "speaker": "ai_agent", "message": "Hello!"},
    {"turn": 3, "speaker": "user_agent", "message": "I'm worried about the budget for the project.", "share_memory": True},
]}


def test_prompts_load_and_never_mention_policies_or_flags():
    for name, slot in (("select_v1", "{memories}"), ("rewrite_v1", "{memory}"), ("rewrite_v2", "{memory}"), ("rewrite_v3", "{memory}"), ("rewrite_v4", "{memory}"),
                       ("rewrite_v5a", "{memory_text}"), ("rewrite_v5b", "{memory_text}")):
        system, user = load_prompt(name)
        text = system + user
        assert "{conversation}" in user and slot in user and "share_memory" not in text
        # v4's "Do not replace what the memory says with what the user is saying now" is about the rewrite's
        # content, not storage; the policies themselves are never described
        allowed = "do not replace what the memory says with what the user is saying now"
        assert not any(w in text.lower().replace(allowed, "") for w in ("discard", "replace", "keep", "depth", "stored"))


def test_parse_answers():
    labels = {"1": "m1", "2": "m2"}
    assert parse_select_answer(useful(2, 1), labels) == ["2", "1"]
    assert parse_select_answer(useful(), labels) == []
    with pytest.raises(UnusableAnswer, match="more than once"):
        parse_select_answer(useful(1, 1), labels)
    with pytest.raises(UnusableAnswer, match="number of a memory shown"):
        parse_select_answer(useful(3), labels)
    assert parse_rewrite_answer(memory(" a ")) == "a"
    with pytest.raises(UnusableAnswer):
        parse_rewrite_answer(memory(" "))


BUDGET = "The project budget is $1,700,000."
STEPS = "The user walked 10,151 steps."
SESSION2 = {"session_id": 5, "date": "2025-06-05", "conversation": [
    {"turn": 1, "speaker": "ai_agent", "message": "Hello!"},
    {"turn": 2, "speaker": "user_agent", "message": "Worried about the budget, and my steps and budget again.", "share_memory": True},
]}


def run(policy_obj, *replies, session=SESSION):
    store = store_with(BUDGET, STEPS)
    client = ScriptedClient(*replies)
    rc = Recaller(client, store, Retriever(ToyEmbedder()), policy_obj, persona="p")
    return store, client, rc.recall_session(session)


def test_discard_counts_and_leaves_no_trace():
    ref = store_with(BUDGET, STEPS)
    store, client, out = run(Discard(), useful(1), memory("The user's project budget is about $1.7M."))
    assert out.found == 1 and out.selected == 1 and out.turns[0].turn == 3 and len(client.calls) == 2  # "Hi there!" finds nothing
    assert store.content_json() == ref.content_json()  # D19, D26
    mid = out.turns[0].found[0]
    assert store.get(mid).times_recalled == 1 and store.get(mid).times_selected == 1
    sel, rew = (c["messages"][1]["content"] for c in client.calls)
    assert "3. User: I'm worried about the budget" in sel and "[1] (2025-06-01) The project budget is $1,700,000." in sel
    assert "Memory:\n(2025-06-01) The project budget is $1,700,000." in rew
    assert [c["meta"]["step"] for c in client.calls] == ["select", "rewrite"]


def test_nothing_chosen_rewrites_nothing():
    store, client, out = run(Replace(), useful())
    mid = out.turns[0].found[0]
    assert out.selected == 0 and len(client.calls) == 1 and store.get(mid).times_selected == 0 and store.get(mid).depth == 1


def test_memories_are_shown_by_date_and_rewritten_in_the_order_chosen():
    store = MemoryStore("p", mode="note")
    a = store.note(STEPS + " steps steps", [Source("p", 2, 1)], At(2, 0), date="2025-06-03")
    b = store.note(BUDGET + " budget", [Source("p", 1, 1)], At(1, 0), date="2025-06-01")
    client = ScriptedClient(useful(2, 1), memory("budget rewrite"), memory("steps rewrite"))
    out = Recaller(client, store, Retriever(ToyEmbedder(), k=3), Replace(), persona="p", threshold=0.0).recall_session(SESSION2)
    sel = client.calls[0]["messages"][1]["content"]
    assert sel.index("(2025-06-03)") < sel.index("(2025-06-01)")  # newest first, whatever the similarity
    assert out.turns[0].shown == [a, b] and out.turns[0].selected == [b, a]  # [2] is the older (budget) memory, chosen first
    assert store.get(a).text == "steps rewrite" and store.get(b).text == "budget rewrite"
    assert "budget" in client.calls[1]["messages"][1]["content"].split("Memory:")[1]


def test_replace_overwrites_and_keep_adds_a_child():
    store, _, out = run(Replace(), useful(1), memory("The user's project budget is about $1.7M."))
    m = store.get(out.turns[0].found[0])
    assert m.text == "The user's project budget is about $1.7M." and m.depth == 2 and len(m.versions) == 2
    store, _, out = run(Keep(), useful(1), memory("The user's project budget is about $1.7M."))
    mid = out.turns[0].found[0]
    (child,) = store.children(mid)
    assert child.depth == 2 and store.get(mid).text == BUDGET and out.turns[0].written == [child.id]
    assert child.times_selected == 0 and child.times_recalled == 0


def test_an_identical_rewrite_is_no_rewrite():
    store, _, out = run(Replace(), useful(1), memory(BUDGET))
    mid = out.turns[0].found[0]
    assert store.get(mid).depth == 1 and out.turns[0].unchanged == [mid] and out.turns[0].written == []


def test_failures_store_nothing_and_are_counted():
    ref = store_with(BUDGET, STEPS)
    store, client, out = run(Replace(), "x", "y", "z")
    assert not out.turns[0].select_ok and out.failures == 1 and len(client.calls) == 3
    assert "Answer with exactly one block" in client.calls[1]["messages"][-1]["content"]
    store, client, out = run(Replace(), useful(1), "x", "y", "z")
    assert out.turns[0].rewrite_failed == [out.turns[0].found[0]] and store.content_json() == ref.content_json()


def test_policy_lookup():
    assert isinstance(policy("discard"), Discard) and isinstance(policy("keep"), Keep)


def test_memory_slots_and_the_date_line():
    assert how_long_before("2025-06-01", "2025-06-03") == "2 days before this one"
    assert how_long_before("2025-06-02", "2025-06-03") == "the day before this one"
    assert how_long_before("2025-06-03", "2025-06-03") == "the same day as this one"
    assert how_long_before("", "2025-06-03") == "" and how_long_before("2025-06-04", "2025-06-03") == ""
    slots = memory_slots("Spent $5.", "2025-06-01", "2025-06-05")
    assert slots["memory"] == "(2025-06-01) Spent $5." and slots["when"] == "4 days before this one"
    assert memory_slots("Spent $5.", "", "2025-06-05")["memory"] == "Spent $5."
    store = store_with(BUDGET, STEPS)
    client = ScriptedClient(useful(1), memory("x"))
    Recaller(client, store, Retriever(ToyEmbedder()), Discard(), persona="p", rewrite_prompt="rewrite_v5b").recall_session(SESSION)
    rew = client.calls[1]["messages"][1]["content"]
    assert "Memory, from an earlier conversation on 2025-06-01 (4 days before this one):\nThe project budget is $1,700,000." in rew
    assert "different events of the same kind" in client.calls[1]["messages"][0]["content"]


def test_call_settings_reach_the_client():
    store = store_with(BUDGET, STEPS)
    client = ScriptedClient(useful())
    Recaller(client, store, Retriever(ToyEmbedder()), Discard(), persona="p", call_settings={"seed": 1}).recall_session(SESSION)
    assert client.calls[0]["seed"] == 1
