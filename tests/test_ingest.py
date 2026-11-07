"""Tests for ingestion (D35, D36), with a scripted stand-in for the model."""

import json

import pytest

from memrecon.ingest import (
    MAX_ATTEMPTS,
    parse_decision,
    parse_extract_answer,
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


def test_duplicate_memories_are_rejected():
    _, lines = render_conversation(SESSION)
    m = {"memory": "The user walked 10,800 steps on 2025-06-03.", "lines": [3]}
    with pytest.raises(UnusableAnswer, match="more than once"):
        parse_note_answer(note_reply(new=[m, m]), {}, lines)
    # v1 runs (E16, E17) must replay exactly as they ran, so v1 keeps accepting them
    assert len(parse_note_answer(note_reply(new=[m, m]), {}, lines, reject_duplicates=False)["new"]) == 2
    store = MemoryStore("p")
    out = Ingestor(ScriptedClient(note_reply(new=[m, m])), store).ingest_note(SESSION)
    assert out.ok and len(out.created) == 2


def test_prompt_v2_loads():
    system, user = load_prompt("ingest_note_v2")
    assert "states only what is true now" in system and "{memories}" in user and "share_memory" not in system


def test_v2_retry_repeats_the_answer_format_but_v1_does_not():
    said_not_did = "I'll now write the memories for this conversation."
    client = ScriptedClient(said_not_did, note_reply())
    out = Ingestor(client, MemoryStore("p"), note_prompt="ingest_note_v2").ingest_note(SESSION)
    assert out.ok and out.attempts == 2
    feedback = client.calls[1]["messages"][-1]["content"]
    assert "Answer with exactly one block in this form" in feedback and '"replaces"' in feedback
    v1 = ScriptedClient(said_not_did, note_reply())
    Ingestor(v1, MemoryStore("p")).ingest_note(SESSION)
    assert v1.calls[1]["messages"][-1]["content"].endswith("following the instructions exactly.")


def test_lint_comments_are_never_sent():
    for name in ("ingest_note_v1", "ingest_log_v1", "ingest_note_v2", "ingest_note_v3", "ingest_note_v4", "ingest_note_v5", "extract_note_v1", "decide_note_v1", "extract_note_v2", "decide_note_v2", "decide_note_v3"):
        system, user = load_prompt(name)
        assert "markdownlint" not in system + user


def test_prompt_v3_loads_and_is_strict():
    system, user = load_prompt("ingest_note_v3")
    assert "adds to" not in system and "calendar date" not in system and "{memories}" in user
    m = {"memory": "The user walked 10,800 steps.", "lines": [3]}
    client = ScriptedClient(note_reply(new=[m, m]), note_reply(new=[m]))
    out = Ingestor(client, MemoryStore("p"), note_prompt="ingest_note_v3").ingest_note(SESSION)
    assert out.ok and out.attempts == 2 and len(out.created) == 1


def test_prompt_v4_loads():
    system, user = load_prompt("ingest_note_v4")
    assert "One memory for each thing the user told you about" in system and "adds to" not in system


def test_prompt_v5_is_v1_with_the_decided_changes():
    v1, _ = load_prompt("ingest_note_v1")
    v5, user = load_prompt("ingest_note_v5")
    for kept in ("One memory for each thing the user told you about", "Use as many sentences as the details need",
                 "Add nothing the user did not say"):
        assert kept in v1 and kept in v5
    assert "calendar date" not in v5 and "same memory twice" in v5 and "{memories}" in user


# -- two-step ingestion (D40) -------------------------------------------------------------


def facts_reply(*facts):
    return "<facts>" + json.dumps({"facts": [{"fact": f, "lines": [3]} for f in facts]}) + "</facts>"


def decision(action, replaces=(), memory=""):
    return "<decision>" + json.dumps({"action": action, "replaces": list(replaces), "memory": memory}) + "</decision>"


class WordRetriever:
    """Similarity = share of the query's words found in the memory (stands in for embeddings)."""

    def sync(self, store):
        self.live = [(m.id, m.text) for m in store.active()]

    def scores(self, query):
        q = set(query.lower().split())
        out = [(mid, len(q & set(t.lower().split())) / len(q)) for mid, t in self.live]
        return sorted(out, key=lambda x: -x[1])


def test_two_step_prompts_load_without_flags():
    for name in ("extract_note_v1", "decide_note_v1", "extract_note_v2", "decide_note_v2", "decide_note_v3"):
        system, user = load_prompt(name)
        assert "share_memory" not in system + user and "<!--" not in system
    assert "{fact}" in load_prompt("decide_note_v1")[1] and "{memories}" not in load_prompt("extract_note_v1")[1]


def test_a_fact_with_no_similar_memory_is_added_without_a_decision_call():
    store = MemoryStore("p")
    client = ScriptedClient(facts_reply("The user spent $11.83 on lunch.", "The user likes jazz."))
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    # the second fact is not shown the first, added from the same conversation
    assert out.ok and len(client.calls) == 1 and len(out.created) == 2 and out.decisions == {"add (no candidate)": 2}
    assert client.calls[0]["meta"]["step"] == "extract" and "Existing memories" not in client.calls[0]["messages"][1]["content"]


def test_below_the_threshold_nothing_is_shown_or_rewritten():
    store = MemoryStore("p")
    old = store.note("The user spent $10.00 on lunch.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    client = ScriptedClient(facts_reply("The user spent $11.83 on lunch today."))
    out = Ingestor(client, store, WordRetriever(), candidate_threshold=0.99).ingest_two_step(SESSION)
    assert len(client.calls) == 1 and store.get(old).retired is None and len(out.created) == 1


def test_replace_supersedes_and_is_shown_only_similar_memories():
    store = MemoryStore("p")
    old = store.note("The user is working on the report 'Q3 Review', due 2025-06-10.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    client = ScriptedClient(facts_reply("The user finished the report 'Q3 Review'."),
                            decision("replace", [1], "The user has finished the report 'Q3 Review'."))
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    sent = client.calls[1]["messages"][1]["content"]
    assert "New fact: The user finished the report 'Q3 Review'." in sent and "[1] (2025-05-30)" in sent
    assert client.calls[1]["meta"]["step"] == "decide"
    new = store.get(out.created[0])
    assert out.superseded == [old] and new.depth == 2 and store.get(old).retired == "superseded"
    assert out.decisions == {"replace": 1}


def test_add_none_and_forget_decisions():
    store = MemoryStore("p")
    a = store.note("The user spent $10.00 on lunch.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    b = store.note("The user likes jazz music a lot.", [Source("p", 6, 3)], At(6, 0), date="2025-05-31")
    client = ScriptedClient(facts_reply("The user spent $11.83 on lunch.", "The user likes jazz music.", "The user wants jazz music forgotten."),
                            decision("add"), decision("none"), decision("forget", [1]))
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    assert out.decisions == {"add": 1, "none": 1, "forget": 1}
    assert store.get(a).retired is None and store.get(b).retired == "forgotten" and len(out.created) == 1


def test_a_memory_changed_earlier_in_the_session_is_shown_to_later_facts():
    store = MemoryStore("p")
    store.note("The user's meeting notes 'Sync' have the agenda: budget, hiring, travel.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    client = ScriptedClient(
        facts_reply("The user removed hiring from the agenda of the meeting notes 'Sync'.", "The user removed travel from the agenda of the meeting notes 'Sync'."),
        decision("replace", [1], "The user's meeting notes 'Sync' have the agenda: budget, travel."),
        decision("replace", [1], "The user's meeting notes 'Sync' have the agenda: budget."))
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    assert [m.text for m in store.active()] == ["The user's meeting notes 'Sync' have the agenda: budget."]
    assert store.active()[0].depth == 3 and len(out.superseded) == 2


def test_a_failed_decision_keeps_the_fact():
    store = MemoryStore("p")
    store.note("The user spent $10.00 on lunch.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    client = ScriptedClient(facts_reply("The user spent $11.83 on lunch."), "no", "still no", "<decision>{}</decision>")
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    assert out.ok and out.decide_failures == 1 and out.decisions == {"add (decision failed)": 1} and len(store.active()) == 2


@pytest.mark.parametrize(
    "reply,match",
    [
        (decision("replace", [], "x"), "must list numbers"),
        (decision("replace", [2], "x"), "must list numbers"),
        (decision("replace", [1], ""), "complete updated memory"),
        (decision("add", [1]), "must be empty"),
        (decision("add", [], "The user spent $11.83."), "must be empty"),
        (decision("merge"), "must be one of"),
    ],
)
def test_decisions_are_read_strictly(reply, match):
    with pytest.raises(UnusableAnswer, match=match):
        parse_decision(reply, {"1": "m00001"})


def test_extracted_facts_are_read_strictly():
    _, lines = render_conversation(SESSION)
    with pytest.raises(UnusableAnswer, match="more than once"):
        parse_extract_answer(facts_reply("x", "x"), lines)
    with pytest.raises(UnusableAnswer, match="user's lines"):
        parse_extract_answer('<facts>{"facts": [{"fact": "x", "lines": [2]}]}</facts>', lines)
    assert parse_extract_answer(facts_reply(), lines) == []


def test_two_step_v2_prompts():
    e, _ = load_prompt("extract_note_v2")
    d, _ = load_prompt("decide_note_v2")
    assert "for a document, give its title" in e and "not an \"add\"" in d
    client = ScriptedClient(facts_reply("The user likes jazz."))
    out = Ingestor(client, MemoryStore("p"), WordRetriever(), extract_prompt="extract_note_v2", decide_prompt="decide_note_v2").ingest_two_step(SESSION)
    assert out.ok and "for a document, give its title" in client.calls[0]["messages"][0]["content"]


def test_a_replace_with_identical_text_is_no_rewrite():
    store = MemoryStore("p")
    old = store.note("The user likes jazz music.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
    client = ScriptedClient(facts_reply("The user likes jazz music."), decision("replace", [1], "The user likes jazz music."))
    out = Ingestor(client, store, WordRetriever()).ingest_two_step(SESSION)
    assert out.decisions == {"none (identical replace)": 1} and out.superseded == [] and out.created == []
    assert store.get(old).retired is None and store.get(old).depth == 1


def test_decide_v3_shows_the_other_facts_and_v2_does_not():
    def run(prompt):
        store = MemoryStore("p")
        store.note("The user's project has a proposed budget of $1,150,000.", [Source("p", 5, 3)], At(5, 0), date="2025-05-30")
        client = ScriptedClient(facts_reply("The user is working on a proposal titled 'Atlas'.", "The user's project has a proposed budget of $750,000."),
                                decision("add"), decision("add"))
        Ingestor(client, store, WordRetriever(), decide_prompt=prompt).ingest_two_step(SESSION)
        return [c["messages"][1]["content"] for c in client.calls if c["meta"]["step"] == "decide"]
    v3 = run("decide_note_v3")
    assert any("Other facts from this conversation:\n- The user is working on a proposal titled 'Atlas'." in m for m in v3)
    assert all("Other facts" not in m for m in run("decide_note_v2"))
