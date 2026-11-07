"""Tests for the probe answer key (drift measure): extraction checks, derived columns, the table."""
import json

import pytest

from memrecon.ingest import UnusableAnswer, load_prompt
from memrecon.probes import ProbeExtractor, Row, build_table, parse_probe_answer, resolve_date, split_number, true_at

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
])
def test_split_number(detail, expected):
    assert split_number(detail) == expected


@pytest.mark.parametrize("detail,expected", [  # 2025-06-04 is a Wednesday
    ("next Tuesday", "2025-06-10"), ("this Friday", "2025-06-06"), ("Friday", "2025-06-06"), ("today", "2025-06-04"),
    ("this morning", "2025-06-04"), ("tomorrow", "2025-06-05"), ("in 10 days", "2025-06-14"), ("August 21st", "2025-08-21"),
    ("21 August", "2025-08-21"), ("June 5, 2025", "2025-06-05"), ("2025-07-01", "2025-07-01"), ("next week", None), ("soon", None),
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
    with pytest.raises(UnusableAnswer, match="lines to read"):
        parse_probe_answer(answer(d("Hi", line=1)), readable)
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
