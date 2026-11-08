"""Tests for Stage 0's control check (D19, D26): discard leaves no trace; a leak is caught."""
import json

from memrecon import stage0
from memrecon.recall import Replace
from memrecon.stage0 import choice_pattern, control_run, finished, next_attempt

from test_recall import ToyEmbedder


class Reply:
    def __init__(self, content):
        self.content, self.cached, self.usage = content, False, {}


class FakeAgent:
    """Answers by step: step 1 extracts each user line as a fact; step 2 adds it; select chooses all;
    rewrite rewords the memory."""

    def chat(self, messages, meta=None, **kw):
        step = (meta or {}).get("step")
        user = messages[1]["content"]
        if step == "extract":
            facts = [{"fact": line.split("User: ", 1)[1], "lines": [int(line.split(".")[0])]}
                     for line in user.splitlines() if ". User: " in line and any(w in line for w in ("budget", "steps"))]
            return Reply("<facts>" + json.dumps({"facts": facts}) + "</facts>")
        if step == "decide":
            return Reply('<decision>{"action": "add", "replaces": [], "memory": ""}</decision>')
        if step == "select":
            n = sum(1 for line in user.split("Memories found:", 1)[1].splitlines() if line.startswith("["))
            return Reply("<useful>" + json.dumps({"memories": list(range(1, n + 1))}) + "</useful>")
        if step == "rewrite":
            return Reply('<memory>{"text": "Reworded for the discussion."}</memory>')
        raise AssertionError(step)


def sessions():
    def conv(*msgs):
        return [{"turn": i + 1, "speaker": "user_agent", "message": m, "share_memory": True} for i, m in enumerate(msgs)]
    return [{"session_id": 1, "date": "2025-06-01", "conversation": conv("The budget is $1,700,000.")},
            {"session_id": 2, "date": "2025-06-02", "conversation": conv("Back to the budget again.", "I walked 9,000 steps.")},
            {"session_id": 3, "date": "2025-06-03", "conversation": conv("How is the budget?")}]


def test_discard_passes_and_writes_its_files(tmp_path):
    out = next_attempt(tmp_path / "p" / "note")
    res = control_run("p", sessions(), "note", FakeAgent(), ToyEmbedder(), out)
    assert res.passed and res.summary()["found"] >= 2 and res.summary()["rewrites"] >= 2
    assert (out / "sessions" / "session_003.json").exists() and (out / "B" / "snapshots" / "session_002.json").exists()
    assert finished(tmp_path / "p" / "note") == out / "summary.json"
    assert next_attempt(tmp_path / "p" / "note").name == "run_2"  # a new attempt never reuses a folder
    events = [json.loads(x) for x in (out / "B" / "events.jsonl").read_text().splitlines()]
    assert {"recall", "select", "present"} <= {e["op"] for e in events}


def test_a_leak_from_the_recall_path_is_caught(tmp_path, monkeypatch):
    monkeypatch.setattr(stage0, "Discard", Replace)  # a "discard" that writes the rewrite back
    res = control_run("p", sessions(), "note", FakeAgent(), ToyEmbedder(), tmp_path / "run_1")
    assert not res.passed
    first = next(s for s in res.sessions if not s.equal)
    assert first.session == 2 and first.difference


def test_choice_patterns():
    assert choice_pattern(["a", "b", "c"], []) == "none"
    assert choice_pattern(["a"], ["a"]) == "one shown"
    assert choice_pattern(["a", "b"], ["a", "b"]) == "all, in listed order"
    assert choice_pattern(["a", "b"], ["b", "a"]) == "all, reversed"
    assert choice_pattern(["a", "b", "c"], ["c", "a"]) == "some"
