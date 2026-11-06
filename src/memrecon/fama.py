"""Memora's forgetting-aware memory accuracy (FAMA).

Per question:

    FAMA = max(0, MPA - lambda * (1 - FAA))
    MPA    = share of "should mention" (memory_presence) sub-questions passed
    FAA    = share of "should not mention" (forgetting_absence) sub-questions passed
    lambda = N_forget / (N_presence + N_forget)

A Table 3 cell is the mean per-question FAMA times 100.

This mirrors `fama_score` in Memora's evaluation code
(evals/agent_eval/memory_to_answer.py and evals/model_eval/model_based_evaluator.py,
which have the same logic), including its edge cases: a question with no
"should mention" sub-questions gets MPA = 0, one with no "should not mention"
sub-questions gets FAA = 1 (so lambda = 0 and FAMA = MPA), and a question with
neither scores 0. The only addition is input checking. tests/test_fama.py
compares this function with Memora's own code when data/memora is present.
"""

from __future__ import annotations

from collections.abc import Iterable


def fama_score(presence_correct: int, presence_total: int, forget_correct: int, forget_total: int) -> float:
    """Per-question FAMA in [0, 1]."""
    for name, value in (
        ("presence_correct", presence_correct),
        ("presence_total", presence_total),
        ("forget_correct", forget_correct),
        ("forget_total", forget_total),
    ):
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if presence_correct > presence_total or forget_correct > forget_total:
        raise ValueError("correct count exceeds total")

    n_p, n_f = presence_total, forget_total
    if n_p == 0 and n_f == 0:
        return 0.0
    mpa = presence_correct / n_p if n_p > 0 else 0.0
    faa = forget_correct / n_f if n_f > 0 else 1.0
    lam = n_f / (n_p + n_f)
    return max(0.0, mpa - lam * (1.0 - faa))


def table_cell(per_question: Iterable[float]) -> float:
    """Mean per-question FAMA times 100, as in Memora's Table 3. Empty input gives 0."""
    scores = list(per_question)
    return sum(scores) / len(scores) * 100 if scores else 0.0
