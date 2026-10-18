"""E11: what does LongMemEval contain, and could it supply probes for repeated recall?

Question: how many questions of each type, how long are the histories, and which turns hold the answer?
Method: read whichever of these files exist in data/longmemeval/ (from Hugging Face
`xiaowu0162/longmemeval-cleaned`): longmemeval_oracle.json, longmemeval_s_cleaned.json,
longmemeval_m_cleaned.json. Fields used, as described in the LongMemEval README: question_id, question_type,
question, answer, question_date, haystack_session_ids, haystack_dates, haystack_sessions (lists of turns with
role, content and, on evidence turns, has_answer), answer_session_ids. Abstention questions have a question_id
ending in `_abs`. Tokens are estimated as characters / 4. Prints each file's sha256.
This script was written from the README before seeing the real files; if a field differs it will say so.
"""
import hashlib
import statistics as st
from collections import Counter

from common import ROOT, load

D = ROOT / "data" / "longmemeval"
NAMES = ["longmemeval_oracle.json", "longmemeval_s_cleaned.json", "longmemeval_m_cleaned.json"]
found = [D / n for n in NAMES if (D / n).exists()]
if not found:
    raise SystemExit(f"No LongMemEval files in {D}. Download them first (see the decision log, section 5).")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def summary(xs):
    xs = sorted(xs)
    return f"min {xs[0]}, median {int(st.median(xs))}, mean {int(st.mean(xs))}, max {xs[-1]}"


for path in found:
    print("=" * 70)
    print(f"file: {path.name}  {path.stat().st_size} bytes  sha256 {sha(path)}")
    data = load(path)
    print(f"questions: {len(data)}; keys of first item: {sorted(data[0].keys())}")
    types = Counter(q["question_type"] for q in data)
    abst = sum(1 for q in data if str(q["question_id"]).endswith("_abs"))
    print("by question_type:", dict(types), f"| abstention (_abs): {abst}")
    n_sess, n_turns, toks, ans_sess, ans_turns, no_flag = [], [], [], [], [], 0
    for q in data:
        sessions = q["haystack_sessions"]
        n_sess.append(len(sessions))
        n_turns.append(sum(len(s) for s in sessions))
        toks.append(sum(len(t.get("content", "")) for s in sessions for t in s) // 4)
        ans_sess.append(len(q.get("answer_session_ids", [])))
        flagged = sum(1 for s in sessions for t in s if t.get("has_answer"))
        ans_turns.append(flagged)
        if flagged == 0:
            no_flag += 1
    print("haystack sessions per question:", summary(n_sess))
    print("haystack turns per question:   ", summary(n_turns))
    print("haystack ~tokens per question: ", summary(toks))
    print("answer sessions per question:  ", summary(ans_sess))
    print("turns flagged has_answer per question:", summary(ans_turns), f"| questions with none: {no_flag}")
    distinct_sessions = {sid for q in data for sid in q["haystack_session_ids"]}
    print(f"distinct session ids across all questions: {len(distinct_sessions)} (histories are shared between questions if this is far below the total)")
    print("by type: mean answer sessions / flagged turns:")
    for t in sorted(types):
        qs = [i for i, q in enumerate(data) if q["question_type"] == t]
        print(f"  {t:28} n={len(qs):3d}  answer sessions {st.mean(ans_sess[i] for i in qs):.1f}  flagged turns {st.mean(ans_turns[i] for i in qs):.1f}")
    print("examples (first of each type):")
    seen = set()
    for q in data:
        if q["question_type"] not in seen:
            seen.add(q["question_type"])
            print(f"  [{q['question_type']}] {q['question']} -> {str(q['answer'])[:80]!r}  (asked {q['question_date']})")
