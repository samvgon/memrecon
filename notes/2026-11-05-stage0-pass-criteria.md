# Stage 0 pass criteria: options and reasoning

*Written by Claude, Nov 5, 2026; section 1 revised Nov 6 after discussion with Sam · Proposed by Both. Decided as D19 (control check), D20 (reported quantities) and D21 (grader check), reviewed by Sam Nov 5–6; where this file and the decision log differ, the log is authoritative. Nothing here has been run.*

Stage 0 has to show, before Stage 1 starts, that the measurement is trustworthy. The plan names two checks (the judge agrees with ~50 hand labels; the no-rewrite control shows drift ≈ 0) but gives no numbers. These should be fixed before anything runs, so that the result can't steer the threshold.

## 1. No-rewrite control: "discard leaves no trace" (revised Nov 6)

*The Nov 5 draft offered options A–C here (store unchanged; plus a repeat run; plus a 2-point bound on the repeat run) and recommended C. Revised after discussion with Sam, for three reasons:*

- *"The store is unchanged" only means something against a reference. Memora's update and delete sessions legitimately change memories, so the reference is a run that ingests the same sessions but never recalls anything.*
- *A threshold on two repeat runs would pass almost automatically at temperature 0, and the call cache would answer an unchanged rerun from disk.*
- *Ingestion loss (what the note leaves out of what the user said) is shared by every write-policy condition, so it lowers all of them equally rather than favouring one. It is reported, not gated (Sam).*

**Pass/fail.** For each development persona:

1. Run A, ingest only: replay the sessions and write memories, including Memora's updates and deletes, but never recall.
2. Run B, the discard condition: the same replay, with recall on every turn (retrieve, rewrite for the turn, discard).
3. After every session, the two stores must be byte-for-byte identical.

Exact comparison works because both runs make the same ingestion requests, which the call cache answers with the same saved text. Any difference means the recall path changed memory, or changed what ingestion was asked: a bug, and "discard" would not really be discard. Run for both the model-note + discard arm (the baseline every write policy is compared with) and the verbatim arm (D24).

**Reported, no threshold:**

- How much a repeat run differs with the cache bypassed (a determinism check).
- Ingestion loss: how much of each probe fact the model note keeps, compared with verbatim.
- How many probe facts survive ingestion. This is the number of facts available to compare, so it sets the sample size and the power.

## 2. Grader agreement with hand labels

What gets checked: the judge (Gemma 4 26B-A4B, D16) and the code graders. The plan says to grade with code where possible; code graders can also be wrong ("€100,000" vs "100k"), so they should face the same hand labels.

Which judge jobs: (i) Memora's yes/no sub-questions for FAMA, and (ii) probe answers that code can't grade. Hedge judging belongs to Stage 1b and is checked there, on the synthetic set, if Stage 1b runs.

| Option | Pass if | Notes |
|---|---|---|
| A | Raw agreement ≥ 90% | Easy to read; can look good when most items are easy "yes" cases. |
| B | Cohen's kappa ≥ 0.7 | Corrects for chance agreement; with ~50 items its uncertainty is large (roughly ±0.15). |
| C | Both A and B, plus flip rate ≤ 5% when the judge scores every item twice | Covers both, and the plan's "score each item twice" rule. |

**Recommendation: C.**

**Labels.** Label about 70 items rather than 50: about 20 for writing and adjusting the judge prompt, and 50 kept back for the pass/fail check. Using the same items to tune the prompt and to test it would overstate agreement. Items come from the development personas only, drawn from the control run's real agent answers, mixed across probe types and Memora sub-question types, with some deliberately borderline. Sam labels before seeing any judge output.

**If the judge fails:** try the fallback judge (Gemma 4 31B, D16) on the same 50 held-back items; if that also fails, the judge prompt or the task needs rethinking, and that is logged before anything else runs.

## 3. Decisions that were needed (made: D19–D21)

1. ~~Control criterion: A, B or C (and the bound, if C).~~ Replaced by the revised section 1 (Nov 6); confirm it.
2. Grader criterion: A, B or C (and the numbers).
3. Number of labelled items and the tune/check split.
4. Whether the judge's jobs in Stage 0 are (i) and (ii) above, with hedges left to Stage 1b.
