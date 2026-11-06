"""Tests for the memory store (D25, D27–D30)."""

import pytest

from memrecon.memory import At, EventLog, MemoryStore, Source, UnknownMemory, first_difference, read_events

SRC = [Source("creative_designer", 89, 4)]
NEW_SRC = [Source("creative_designer", 110, 6)]
NOTE = "Budget for the localisation project: €100,000."


def note_store():
    s = MemoryStore("creative_designer", mode="note")
    mid = s.note(NOTE, SRC, At(89, 4))
    return s, mid


def test_note_is_depth_1_and_log_is_depth_0():
    s, mid = note_store()
    m = s.get(mid)
    assert (m.depth, m.written_from, m.parents, m.sources) == (1, "user", [], SRC)
    log = MemoryStore("creative_designer", mode="log")
    lid = log.log("My budget is €100,000.", SRC, At(89, 4))
    assert log.get(lid).depth == 0


def test_mode_guards():
    s, _ = note_store()
    with pytest.raises(ValueError):
        s.log("x", SRC, At(1, 1))
    log = MemoryStore("p", mode="log")
    with pytest.raises(ValueError):
        log.note("x", SRC, At(1, 1))
    with pytest.raises(ValueError):
        MemoryStore("p", mode="summary")


def test_discard_only_counts():
    s, mid = note_store()
    before = s.content_json()
    for _ in range(3):
        s.recall(mid)
    assert s.get(mid).times_recalled == 3
    assert s.content_json() == before


def test_keep_makes_a_child_one_deeper():
    s, mid = note_store()
    s.recall(mid)
    child = s.keep(mid, "Roughly €100k, already tight.", At(95, 2))
    c, p = s.get(child), s.get(mid)
    assert (c.depth, c.written_from, c.parents, c.times_recalled) == (2, "memory", [mid], 0)
    assert c.sources == p.sources
    assert p.times_recalled == 1 and p.text == NOTE
    grandchild = s.keep(child, "About 100k.", At(96, 1))
    assert s.get(grandchild).depth == 3
    assert [m.id for m in s.children(mid)] == [child]
    assert {m.id for m in s.active()} == {mid, child, grandchild}


def test_replace_overwrites_in_place_and_keeps_history():
    s, mid = note_store()
    for k, text in enumerate(["Roughly €100k budget.", "About 100k, tight."], start=1):
        s.recall(mid)
        s.replace(mid, text, At(100 + k, 2))
    m = s.get(mid)
    assert m.text == "About 100k, tight."
    assert m.depth == 3
    assert m.times_recalled == 2  # same memory, its counter carries on
    assert [(v.text, v.depth) for v in m.versions] == [(NOTE, 1), ("Roughly €100k budget.", 2), ("About 100k, tight.", 3)]


def test_policies_not_run_on_log_memories():
    log = MemoryStore("p", mode="log")
    lid = log.log("My budget is €100,000.", SRC, At(89, 4))
    log.recall(lid)  # discard is fine
    with pytest.raises(ValueError):
        log.keep(lid, "x", At(90, 1))
    with pytest.raises(ValueError):
        log.replace(lid, "x", At(90, 1))


def test_user_change_supersedes_what_it_was_written_from():
    s, mid = note_store()
    s.recall(mid)
    child = s.keep(mid, "Roughly €100k, tight.", At(95, 2))  # depth 2
    # the user changes the budget; retrieval found both; the model writes a merged note replacing both
    new = s.note("Budget is now €120,000.", NEW_SRC, At(110, 6), supersedes=[mid, child])
    n = s.get(new)
    assert n.depth == 3  # deepest of what it was written from (2) + 1
    assert n.written_from == "both"
    assert n.parents == [mid, child]
    assert n.sources == SRC + NEW_SRC
    for old in (mid, child):
        o = s.get(old)
        assert (o.retired, o.superseded_by, o.retired_at) == ("superseded", new, At(110, 6))
    assert [m.id for m in s.active()] == [new]


def test_stale_child_survives_if_retrieval_missed_it():
    s, mid = note_store()
    child = s.keep(mid, "Roughly €100k.", At(95, 2))
    new = s.note("Budget is now €120,000.", NEW_SRC, At(110, 6), supersedes=[mid])
    assert {m.id for m in s.active()} == {child, new}  # the stale copy is still retrievable


def test_forget_is_superseding_with_nothing():
    s, mid = note_store()
    s.forget([mid], At(109, 3))
    m = s.get(mid)
    assert (m.retired, m.superseded_by) == ("forgotten", None)
    assert s.active() == []


def test_retired_memories_cannot_be_used_or_retired_twice():
    s, mid = note_store()
    s.forget([mid], At(109, 3))
    with pytest.raises(ValueError):
        s.recall(mid)
    with pytest.raises(ValueError):
        s.forget([mid], At(110, 1))
    with pytest.raises(ValueError):
        s.note("x", SRC, At(111, 1), supersedes=[mid])
    with pytest.raises(UnknownMemory):
        s.recall("m99999")


def test_log_mode_supersede_keeps_depth_0():
    log = MemoryStore("p", mode="log")
    a = log.log("My budget is €100,000.", SRC, At(89, 4))
    b = log.log("Actually it's €120,000 now.", NEW_SRC, At(110, 6), supersedes=[a])
    assert log.get(b).depth == 0
    assert log.get(b).written_from == "user"  # a log entry is the user's words, not written from the old one
    assert log.get(a).superseded_by == b


def test_model_sees_text_only():
    s, mid = note_store()
    child = s.keep(mid, "Roughly €100k.", At(95, 2))
    text, labels = s.present([child, mid])
    assert text == f"[1] Roughly €100k.\n[2] {NOTE}"
    assert labels == {"1": child, "2": mid}
    for hidden in ("m0000", "depth", "parent", "creative_designer", "superseded"):
        assert hidden not in text


def test_ids_and_snapshots_are_deterministic(tmp_path):
    def run():
        s, mid = note_store()
        s.recall(mid)
        s.note("Prefers modernist design.", [Source("creative_designer", 29, 2)], At(29, 2))
        return s

    run().save(tmp_path / "a.json")
    run().save(tmp_path / "b.json")
    assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes()


def test_stage0_check_ingest_only_vs_discard():
    a, _ = note_store()
    b, mid_b = note_store()
    b.recall(mid_b)
    b.recall(mid_b)
    assert first_difference(a.content_json(), b.content_json()) is None
    assert a.to_json() != b.to_json()  # full snapshots differ only by the recall counters
    b.replace(mid_b, "Roughly €100k.", At(95, 1))  # a leak from the recall path
    assert first_difference(a.content_json(), b.content_json()) is not None


def test_snapshot_round_trip(tmp_path):
    s, mid = note_store()
    s.recall(mid)
    child = s.keep(mid, "Roughly €100k.", At(95, 1))
    s.replace(child, "About 100k.", At(96, 1))
    s.note("Budget is now €120,000.", NEW_SRC, At(110, 6), supersedes=[mid])
    s.save(tmp_path / "snap.json")
    loaded = MemoryStore.load(tmp_path / "snap.json")
    assert loaded.to_json() == s.to_json()
    assert loaded.note("next", SRC, At(111, 1)) == "m00004"  # ID counter survives
    assert not list(tmp_path.glob("*.tmp*"))


# --- event file (audit log) -------------------------------------------------------


def test_every_operation_is_an_event_in_order(tmp_path):
    ev = EventLog(tmp_path / "events.jsonl")
    s = MemoryStore("creative_designer", mode="note", events=ev)
    mid = s.note(NOTE, SRC, At(89, 4))
    s.present([mid], At(95, 2))
    s.recall(mid, At(95, 2))
    child = s.keep(mid, "Roughly €100k.", At(95, 2))
    s.replace(child, "About 100k.", At(96, 1))
    new = s.note("Budget is now €120,000.", NEW_SRC, At(110, 6), supersedes=[mid, child])
    s.forget([new], At(120, 1))
    events = read_events(tmp_path / "events.jsonl")
    assert [e["op"] for e in events] == ["note", "present", "recall", "keep", "replace", "note", "forget"]
    assert [e["n"] for e in events] == list(range(1, 8))
    assert events[3] == {"n": 4, "op": "keep", "at": {"session_id": 95, "turn": 2}, "id": child, "parent": mid, "text": "Roughly €100k.", "depth": 2, "since_user": 2}
    assert events[5]["supersedes"] == [mid, child] and events[5]["depth"] == 4
    assert events[1]["labels"] == {"1": mid}


def test_events_continue_after_resume_and_skip_a_cut_off_line(tmp_path):
    path = tmp_path / "events.jsonl"
    s = MemoryStore("p", events=EventLog(path))
    mid = s.note("one", SRC, At(1, 1))
    s.recall(mid, At(2, 1))
    s.save(tmp_path / "snap.json")
    with open(path, "a") as f:
        f.write('{"n": 3, "op": "rec')  # a crash mid-line
    resumed = MemoryStore.load(tmp_path / "snap.json", events=EventLog(path))
    resumed.recall(mid, At(3, 1))
    assert [e["n"] for e in read_events(path)] == [1, 2, 3]


def test_event_count_is_not_part_of_the_stage0_comparison():
    a, _ = note_store()
    b, mid_b = note_store()
    b.present([mid_b])
    b.recall(mid_b)
    assert a.content_json() == b.content_json()


# --- the second depth view: since_user (D33) -----------------------------------------


def test_since_user_follows_depth_until_a_user_change():
    s, mid = note_store()
    child = s.keep(mid, "Roughly €100k.", At(95, 2))
    s.replace(child, "About 100k.", At(96, 1))
    c = s.get(child)
    assert (c.depth, c.since_user) == (3, 3)
    new = s.note("Budget is now €120,000.", NEW_SRC, At(110, 6), supersedes=[mid, child])
    n = s.get(new)
    assert (n.depth, n.since_user) == (4, 1)  # primary: deepest + 1; second view: as though new
    grandchild = s.keep(new, "About 120k.", At(115, 1))
    assert (s.get(grandchild).depth, s.get(grandchild).since_user) == (5, 2)
    assert [(v.depth, v.since_user) for v in c.versions] == [(2, 2), (3, 3)]


def test_since_user_in_log_mode_and_events(tmp_path):
    log = MemoryStore("p", mode="log", events=EventLog(tmp_path / "e.jsonl"))
    a = log.log("My budget is €100,000.", SRC, At(89, 4))
    assert log.get(a).since_user == 0
    s = MemoryStore("p", events=EventLog(tmp_path / "f.jsonl"))
    mid = s.note(NOTE, SRC, At(89, 4))
    s.keep(mid, "Roughly €100k.", At(95, 2))
    assert [e["since_user"] for e in read_events(tmp_path / "f.jsonl")] == [1, 2]
