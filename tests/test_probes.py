"""Tests for the probe answer key (drift measure): extraction checks, derived columns, the table."""
import json

import pytest

from memrecon.ingest import UnusableAnswer, load_prompt
from memrecon.probes import ProbeExtractor, Row, build_table, graded_qualifier, grading_view, parse_probe_answer, resolve_date, split_number, true_at

SESSION = {
    "session_id": 7,
    "date": "2025-06-04",
    "operation": "update",
    "conversation": [
        {"turn": 1, "speaker": "user_agent", "message": "Hi!", "share_memory": False},
        {"turn": 2, "speaker": "ai_agent", "message": "Hello!", "share_memory": False},
        {"turn": 3, "speaker": "user_agent", "message": "Please update the budget from $1,050,000 to about $900,000.", "share_memory": True},
    ],
}


class Reply:
    def __init__(self, content):
        self.content = content


class ScriptedClient:
    def __init__(self, *items):
        self.items, self.calls = list(items), []

    def chat(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return Reply(self.items.pop(0))


def answer(*details):
    return "<details>" + json.dumps({"details": list(details)}) + "</details>"


def d(detail, status="new", field="budget", kind="number", qualifier="", who="", line=3):
    return {"line": line, "detail": detail, "qualifier": qualifier, "field": field, "who": who, "kind": kind, "status": status}


def test_prompt_loads_without_flags():
    system, user = load_prompt("probe_extract_v1")
    assert "share_memory" not in system + user and "<!--" not in system
    assert "{fields}" in user and "{lines}" in user and '"qualifier"' in system


@pytest.mark.parametrize("detail,expected", [
    ("10k steps", (10000.0, "steps")), ("$1.2 million", (1200000.0, "$")), ("15%", (15.0, "%")),
    ("€750,000", (750000.0, "€")), ("$50 per week", (50.0, "$ per week")), ("Count Basie", (None, None)),
    ("18-month timeline", (18.0, "month timeline")),
])
def test_split_number(detail, expected):
    assert split_number(detail) == expected


@pytest.mark.parametrize("detail,expected", [  # 2025-06-04 is a Wednesday
    ("next Tuesday", "2025-06-10"), ("this Friday", "2025-06-06"), ("Friday", "2025-06-06"), ("today", "2025-06-04"),
    ("this morning", "2025-06-04"), ("tomorrow", "2025-06-05"), ("in 10 days", "2025-06-14"), ("August 21st", "2025-08-21"),
    ("21 August", "2025-08-21"), ("June 5, 2025", "2025-06-05"), ("2025-07-01", "2025-07-01"), ("next week", None), ("soon", None),
    ("due in 10 days", "2025-06-14"), ("due by the end of next week", None), ("within the next three months", None),
    ("on August 21st this year", "2025-08-21"), ("may", None), ("last night", "2025-06-03"),
])
def test_resolve_date(detail, expected):
    assert resolve_date(detail, "2025-06-04") == expected


def test_details_and_qualifiers_must_be_exact_copies_of_a_line_to_read():
    readable = {3: SESSION["conversation"][2]["message"]}
    rows = parse_probe_answer(answer(d("$1,050,000", "replaced"), d("$900,000", "updated", qualifier="about")), readable)
    assert [r["status"] for r in rows] == ["replaced", "updated"]
    with pytest.raises(UnusableAnswer, match="copied exactly"):
        parse_probe_answer(answer(d("$900K", "updated")), readable)
    with pytest.raises(UnusableAnswer, match="qualifier"):
        parse_probe_answer(answer(d("$900,000", "updated", qualifier="roughly")), readable)
    with pytest.raises(UnusableAnswer, match="got 1"):
        parse_probe_answer(answer(d("Hi", line=1)), readable)
    with pytest.raises(UnusableAnswer, match='without quotes.*got "3"'):
        parse_probe_answer(answer(d("$900,000", "updated", line="3")), readable)
    with pytest.raises(UnusableAnswer, match="status"):
        parse_probe_answer(answer(d("$900,000", "changed")), readable)
    with pytest.raises(UnusableAnswer, match="more than once"):
        parse_probe_answer(answer(d("$900,000", "updated"), d("$900,000", "updated")), readable)


def test_extractor_derives_columns_and_shows_fields_so_far():
    client = ScriptedClient(answer(d("$1,050,000", "replaced"), d("$900,000", "updated", qualifier="about")),
                            answer(d("$900,000", "new")))
    ex = ProbeExtractor(client, persona="p")
    out = ex.extract(SESSION, ("content", "proposal_1"))
    assert out.ok and [(r.number, r.unit, r.qualifier) for r in out.rows] == [(1050000.0, "$", ""), (900000.0, "$", "about")]
    sent = client.calls[0]["messages"][1]["content"]
    assert "Lines to read: 3" in sent and "Fields so far: (none)" in sent and client.calls[0]["meta"]["step"] == "probe_extract"
    ex.extract({**SESSION, "session_id": 8}, ("content", "proposal_1"))
    assert "Fields so far: budget" in client.calls[1]["messages"][1]["content"]


def test_a_bad_answer_is_sent_back_with_the_reason():
    client = ScriptedClient(answer(d("$900K", "updated")), answer(d("$900,000", "updated")))
    ex = ProbeExtractor(client)
    out = ex.extract(SESSION, ("content", "p"))
    assert out.ok and out.attempts == 2 and "copied exactly" in client.calls[1]["messages"][-1]["content"]


def R(detail, status, session, field="budget", kind="number", record=("content", "p")):
    return Row(record, field, detail, "", "", kind, status, session, 3, f"2025-06-{session:02d}")


def test_table_updated_closes_the_field_added_does_not():
    rows = [R("$1,050,000", "new", 1), R("$900,000", "updated", 5),
            R("Cost review", "new", 1, field="agenda item", kind="item"), R("Hiring", "added", 5, field="agenda item", kind="item")]
    t = build_table(rows)
    now = {(e["detail"], e["state"]) for e in true_at(t, 5)}
    assert now == {("$900,000", "true"), ("Cost review", "true"), ("Hiring", "true")}
    assert {(e["detail"], e["state"]) for e in true_at(t, 4)} == {("$1,050,000", "true"), ("Cost review", "true")}


def test_table_replaced_removed_done_and_no_longer():
    rows = [R("Hiring", "new", 1, field="agenda item", kind="item"), R("Hiring", "removed", 3, field="agenda item", kind="item"),
            R("Buy groceries", "new", 1, field="task", kind="task", record=("todo_list", "buy groceries")),
            R("Buy groceries", "done", 4, field="task", kind="task", record=("todo_list", "buy groceries")),
            R("Count Basie", "new", 1, field="liked artist", kind="preference", record=("preference", "count basie")),
            R("Count Basie", "no longer", 6, field="liked artist", kind="preference", record=("preference", "duke ellington")),
            R("not this Friday", "replaced", 2, field="date", kind="date")]
    t = true_at(build_table(rows), 6)
    states = {(e["detail"], e["state"]) for e in t}
    assert ("Hiring", "true") not in states and ("Buy groceries", "done") in states
    assert ("Count Basie", "no longer") in states and ("Count Basie", "true") not in states  # matched across records
    assert ("not this Friday", "false") in states  # nothing to close: kept as a value that should not be


@pytest.mark.parametrize("q,expected", [("about", "about"), ("at least", "at least"), ("every day", "every day"), ("under", "under"),
                                         ("from", ""), ("also", ""), ("just", ""), ("used to", ""), ("just haven't resonated with me", "")])
def test_only_real_hedges_are_graded(q, expected):
    assert graded_qualifier(q) == expected


def test_grading_view_splits_code_and_judge_probes():
    rows = [R("$900,000", "new", 1), R("plan a dream vacation", "new", 1, field="task", kind="task"),
            R("Count Basie", "new", 1, field="liked artist", kind="name", record=("preference", "count basie")),
            R("Count Basie", "no longer", 2, field="liked artist", kind="name", record=("preference", "count basie"))]
    rows[0].who, rows[0].qualifier = "user", "from"
    view = {(e["detail"], e["state"]): e for e in grading_view(build_table(rows))}
    assert view[("$900,000", "true")]["grade"] == "code" and view[("$900,000", "true")]["qualifier"] == "" and view[("$900,000", "true")]["who"] == ""
    assert view[("plan a dream vacation", "true")]["grade"] == "judge" and view[("Count Basie", "no longer")]["grade"] == "judge"


# -- version 2: ids, attached_to, named who (D46) ---------------------------------------------------

LINE = "The action items include Legal & Compliance providing a cost projection by the end of next month."


def d2(i, detail, field, kind="item", status="new", who="", attached=None, line=3):
    return {"id": i, "line": line, "detail": detail, "qualifier": "", "field": field, "who": who, "kind": kind, "status": status, "attached_to": attached}


def test_prompt_v2_loads():
    system, user = load_prompt("probe_extract_v2")
    assert '"attached_to"' in system and "meeting notes title" in system and "someone else" not in system.split("Give each detail a status")[0]


def test_v2_checks_ids_links_and_named_who():
    readable = {3: LINE}
    ok = answer(d2(1, "Legal & Compliance providing a cost projection", "action item", who="Legal & Compliance"),
                d2(2, "by the end of next month", "action item deadline", kind="date", attached=1))
    rows = parse_probe_answer(ok, readable, version=2)
    assert rows[1]["attached_to"] == 1 and rows[0]["who"] == "Legal & Compliance"
    with pytest.raises(UnusableAnswer, match="name copied exactly"):
        parse_probe_answer(answer(d2(1, "a cost projection", "action item", who="Legal team")), readable, version=2)
    with pytest.raises(UnusableAnswer, match="attached_to"):
        parse_probe_answer(answer(d2(1, "a cost projection", "action item", attached=7)), readable, version=2)
    with pytest.raises(UnusableAnswer, match="different whole number"):
        parse_probe_answer(answer(d2(1, "a cost projection", "action item"), d2(1, "by the end of next month", "deadline", kind="date")), readable, version=2)
    with pytest.raises(UnusableAnswer, match="exactly the keys"):  # a version 1 answer to the version 2 prompt
        parse_probe_answer(answer(d("a cost projection", field="action item", kind="item")), readable, version=2)


def test_extractor_v2_gives_global_row_ids():
    s = {**SESSION, "conversation": SESSION["conversation"][:2] + [{"turn": 3, "speaker": "user_agent", "message": LINE, "share_memory": True}]}
    client = ScriptedClient(answer(d2(1, "a cost projection", "action item", who="Legal & Compliance"),
                                   d2(2, "by the end of next month", "action item deadline", kind="date", attached=1)))
    out = ProbeExtractor(client, persona="p", prompt="probe_extract_v2").extract(s, ("content", "notes_1"))
    assert [(r.row_id, r.attached_to) for r in out.rows] == [("p:7:1", None), ("p:7:2", "p:7:1")]


def test_v2_table_updates_only_the_deadline_of_the_same_item():
    def row(rid, detail, field, status, session, attached=None, kind="item"):
        r = R(detail, status, session, field=field, kind=kind, record=("content", "notes_1"))
        r.row_id, r.attached_to = rid, attached
        return r
    rows = [row("p:1:1", "sensitivity analysis", "action item", "new", 1), row("p:1:2", "in 10 days", "action item deadline", "new", 1, "p:1:1", "date"),
            row("p:1:3", "revised forecast", "action item", "new", 1), row("p:1:4", "end of next week", "action item deadline", "new", 1, "p:1:3", "date"),
            row("p:5:1", "sensitivity analysis", "action item", "new", 5), row("p:5:2", "Friday", "action item deadline", "updated", 5, "p:5:1", "date")]
    now = {e["detail"] for e in true_at(build_table(rows), 5) if e["field"] == "action item deadline"}
    assert now == {"Friday", "end of next week"}  # the other item's deadline stays


def test_grading_view_recomputes_derived_columns():
    r = R("18-month timeline", "new", 3, field="project timeline")
    r.number, r.unit = 18.0, "-month timeline"  # as stored before the parser fix
    e = grading_view(build_table([r]))[0]
    assert (e["number"], e["unit"]) == (18.0, "month timeline")


def test_v2_unlinked_update_closes_a_linked_old_value():
    def row(rid, detail, field, status, session, attached=None, kind="number"):
        r = R(detail, status, session, field=field, kind=kind)
        r.row_id, r.attached_to = rid, attached
        return r
    rows = [row("p:1:1", "Atlas", "proposal title", "new", 1, kind="title"), row("p:1:2", "$1,050,000", "project budget", "new", 1, "p:1:1"),
            row("p:5:1", "$900,000", "project budget", "updated", 5)]  # "update the budget to $900,000", title not restated
    now = {e["detail"] for e in true_at(build_table(rows), 5) if e["field"] == "project budget"}
    assert now == {"$900,000"}


# -- version 3 (D47) -------------------------------------------------------------------------------

@pytest.mark.parametrize("detail,expected", [
    ("ten thousand steps", (10000.0, "steps")), ("five days", (5.0, "days")), ("twenty-five dollars", (25.0, "dollars")),
    ("one hundred and twenty steps", (120.0, "steps")), ("a hundred", (100.0, None)), ("and so on", (None, None)),
])
def test_split_number_reads_number_words(detail, expected):
    assert split_number(detail) == expected


def d3(i, detail, field, kind="item", status="new", who="", attached=None, replaces=None, qualifier="", line=3):
    return {**d2(i, detail, field, kind, status, who, attached, line), "qualifier": qualifier, "replaces": replaces}


def test_prompt_v3_loads():
    system, user = load_prompt("probe_extract_v3")
    assert '"replaces"' in system and '"unknown"' in system and "{recorded}" in user and "share_memory" not in system + user


def test_v3_checks_root_unknown_e_labels_and_replaces():
    readable = {3: "Please update the budget from $1,050,000 to about $900,000."}
    existing = {"E1": "p:1:2"}
    ok = answer(d3(1, "$900,000", "project budget", "number", "updated", attached="R", replaces="E1", qualifier="about"))
    assert parse_probe_answer(ok, readable, 3, existing)[0]["replaces"] == "E1"
    assert parse_probe_answer(answer(d3(1, "$900,000", "project budget", "number", attached="unknown")), readable, 3, existing)
    with pytest.raises(UnusableAnswer, match="E label shown"):
        parse_probe_answer(answer(d3(1, "$900,000", "project budget", "number", attached="E4")), readable, 3, existing)
    with pytest.raises(UnusableAnswer, match='"replaces"'):  # only replacing statuses name a row
        parse_probe_answer(answer(d3(1, "$900,000", "project budget", "number", "new", replaces="E1")), readable, 3, existing)
    with pytest.raises(UnusableAnswer, match="exactly the keys"):  # a version 2 answer to the version 3 prompt
        parse_probe_answer(answer(d2(1, "$900,000", "project budget", "number")), readable, 3, existing)


def test_extractor_v3_shows_what_is_recorded_and_maps_labels():
    first = {**SESSION, "session_id": 1, "operation": "add", "conversation": SESSION["conversation"][:2] + [
        {"turn": 3, "speaker": "user_agent", "message": "The project budget is $1,050,000.", "share_memory": True}]}
    client = ScriptedClient(answer(d3(1, "$1,050,000", "project budget", "number", attached="R")),
                            answer(d3(1, "$900,000", "project budget", "number", "updated", attached="R", replaces="E1", qualifier="about")))
    ex = ProbeExtractor(client, persona="p", prompt="probe_extract_v3")
    a = ex.extract(first, ("content", "notes_1"))
    assert "nothing recorded yet" in client.calls[0]["messages"][1]["content"]
    assert a.rows[0].attached_to == "p:R:content/notes_1"
    b = ex.extract(SESSION, ("content", "notes_1"))
    assert "E1 (project budget): $1,050,000" in client.calls[1]["messages"][1]["content"]
    assert b.rows[0].replaces == "p:1:1"
    now = true_at(build_table(ex.rows), 7)
    assert [(e["detail"], e["attached_detail"]) for e in now] == [("$900,000", "p:R:content/notes_1")]


def test_v3_replaces_closes_exactly_the_named_row():
    def row(rid, detail, field, status, session, attached=None, replaces=None, kind="item"):
        r = R(detail, status, session, field=field, kind=kind, record=("content", "notes_1"))
        r.row_id, r.attached_to, r.replaces = rid, attached, replaces
        return r
    rows = [row("p:1:1", "draft the memo", "action item", "new", 1), row("p:1:2", "send the deck", "action item", "new", 1),
            row("p:4:1", "draft the memo", "action item", "done", 4, replaces="p:1:1")]
    states = {(e["detail"], e["state"]) for e in true_at(build_table(rows), 4)}
    assert states == {("draft the memo", "done"), ("send the deck", "true")}


def test_v3_linked_update_on_the_root_closes_the_roots_value():
    def row(rid, detail, status, session, attached):
        r = R(detail, status, session, field="post platform", kind="name", record=("content", "post_1"))
        r.row_id, r.attached_to = rid, attached
        return r
    root = "p:R:content/post_1"
    rows = [row("p:1:1", "LinkedIn", "new", 1, root), row("p:3:1", "Twitter", "updated", 3, root)]
    assert {e["detail"] for e in true_at(build_table(rows), 3)} == {"Twitter"}


def test_qualifiers_are_scored_only_on_numbers_and_dates():
    a = R("10k steps", "new", 1, field="step count", kind="number")
    a.qualifier = "about"
    b = R("Atlas", "new", 1, field="project title", kind="title")
    b.qualifier = "about"
    view = {e["detail"]: e["qualifier"] for e in grading_view(build_table([a, b]))}
    assert view == {"10k steps": "about", "Atlas": ""}


def test_v3_accepts_an_id_written_in_quotes():
    readable = {3: LINE}
    rows = parse_probe_answer(answer(d3(1, "a cost projection", "action item", attached="R"),
                                     d3(2, "by the end of next month", "action item deadline", "date", attached="1")), readable, 3, {})
    assert rows[1]["attached_to"] == 1
    with pytest.raises(UnusableAnswer, match="attached_to"):  # a quoted id that is not in the answer is still wrong
        parse_probe_answer(answer(d3(1, "a cost projection", "action item", attached="7")), readable, 3, {})


def test_v3_same_words_attached_to_different_things_are_two_details():
    line = "The risks are talent availability, which is a medium risk, and explainability, also a medium risk."
    readable = {3: line}
    two = answer(d3(1, "talent availability", "key risk", attached="R"), d3(2, "explainability", "key risk", attached="R"),
                 d3(3, "medium risk", "risk level", "name", attached=1), d3(4, "medium risk", "risk level", "name", attached=2))
    assert len(parse_probe_answer(two, readable, 3, {})) == 4
    with pytest.raises(UnusableAnswer, match="attach each"):
        parse_probe_answer(answer(d3(1, "medium risk", "risk level", "name", attached="R"), d3(2, "medium risk", "risk level", "name", attached="R")), readable, 3, {})
