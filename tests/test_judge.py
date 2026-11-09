"""Tests for the judge (D21, D61)."""
import json

import pytest

from memrecon.ingest import UnusableAnswer, load_prompt
from memrecon.judge import Judge, cohen_kappa, parse_judgement
from test_recall import ScriptedClient


def block(answer, reason="r"):
    return "<judgement>" + json.dumps({"reason": reason, "answer": answer}) + "</judgement>"


def test_prompt_and_parse():
    system, user = load_prompt("judge_v1")
    assert "{response}" in user and "{question}" in user and "share_memory" not in system + user
    assert parse_judgement(block("no", " because ")) == ("no", "because")
    for bad in (block("maybe"), "<judgement>{\"answer\": \"yes\", \"x\": 1}</judgement>", "yes"):
        with pytest.raises(UnusableAnswer):
            parse_judgement(bad)


def test_judge_retries_and_passes_settings():
    client = ScriptedClient("nope", block("yes"))
    out = Judge(client, call_settings={"seed": 1}).judge("The budget is $60.", "Does the response mention $60?", {"item": "J01"})
    assert out["answer"] == "yes" and out["attempts"] == 2 and client.calls[0]["seed"] == 1
    assert "The response:\nThe budget is $60." in client.calls[0]["messages"][1]["content"]


def test_kappa():
    assert cohen_kappa(["yes", "no", "yes", "no"], ["yes", "no", "yes", "no"]) == 1.0
    assert abs(cohen_kappa(["yes", "yes", "no", "no"], ["yes", "no", "yes", "no"])) < 1e-9
    assert cohen_kappa(["yes"] * 3, ["yes"] * 3) == 1.0
