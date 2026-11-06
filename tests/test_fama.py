"""Tests for fama_score: hand-worked cases, plus a check against Memora's own code."""

import ast
from itertools import product
from pathlib import Path

import pytest

from memrecon.fama import fama_score, table_cell

ROOT = Path(__file__).resolve().parents[1]
MEMORA_EVAL = ROOT / "data" / "memora" / "evals" / "agent_eval" / "memory_to_answer.py"


def test_hand_worked_mixed():
    # 3 of 4 "should mention" passed, 1 of 2 "should not mention" passed:
    # MPA = 0.75, FAA = 0.5, lambda = 2/6, FAMA = 0.75 - (1/3)(0.5) = 0.58333...
    assert fama_score(3, 4, 1, 2) == pytest.approx(0.75 - (2 / 6) * 0.5)


def test_reasoning_question_has_no_forget_items():
    # No "should not mention" sub-questions: lambda = 0, FAMA = MPA.
    assert fama_score(2, 5, 0, 0) == pytest.approx(0.4)


def test_perfect_and_clamped():
    assert fama_score(4, 4, 2, 2) == 1.0
    # MPA = 0, FAA = 0, lambda = 0.5: 0 - 0.5 < 0, clamped to 0.
    assert fama_score(0, 2, 0, 2) == 0.0


def test_remembering_forgotten_items_costs_points():
    # All "should mention" passed but every "should not mention" failed:
    # 1 - (3/7)(1 - 0) = 4/7.
    assert fama_score(4, 4, 0, 3) == pytest.approx(4 / 7)


def test_memora_edge_cases():
    assert fama_score(0, 0, 0, 0) == 0.0  # no sub-questions at all
    assert fama_score(0, 0, 2, 2) == 0.0  # no "should mention": MPA = 0


@pytest.mark.parametrize("args", [(-1, 2, 0, 0), (3, 2, 0, 0), (0, 2, 3, 2)])
def test_bad_counts_rejected(args):
    with pytest.raises(ValueError):
        fama_score(*args)


def test_table_cell():
    assert table_cell([1.0, 0.5, 0.0]) == pytest.approx(50.0)
    assert table_cell([]) == 0.0


def _memora_fama_score():
    """Load fama_score from Memora's evaluator source without importing its dependencies."""
    tree = ast.parse(MEMORA_EVAL.read_text(encoding="utf-8"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "fama_score")
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(MEMORA_EVAL), "exec"), namespace)
    return namespace["fama_score"]


@pytest.mark.skipif(not MEMORA_EVAL.exists(), reason="Memora's evaluation code is not downloaded (data/memora)")
def test_matches_memora_code():
    theirs = _memora_fama_score()
    for n_p, n_f in product(range(0, 9), range(0, 9)):
        for c_p, c_f in product(range(n_p + 1), range(n_f + 1)):
            assert fama_score(c_p, n_p, c_f, n_f) == pytest.approx(theirs(c_p, n_p, c_f, n_f)), (c_p, n_p, c_f, n_f)
