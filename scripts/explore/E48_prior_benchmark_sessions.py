"""E48: how many sessions does each LoCoMo and LongMemEval question need? (checks a claim in the Memora paper)

Question: Memora's introduction says 94% of LoCoMo's questions and 85% of LongMemEval's need evidence from
no more than two earlier sessions, and its Table 1 gives the average and maximum number of sessions a
question needs ("memory consolidation"): LoCoMo 1.3 and 15, LongMemEval 1.9 and 6. The paper does not say
how these were computed. Do the released annotations give the same numbers?
Method (no model calls): count the distinct sessions behind each question's evidence annotation.
  LoCoMo (data/locomo/locomo10.json): `evidence` lists dialogue ids; the session is the number after "D"
  (e.g. "D1:3" = session 1, turn 3). Some evidence strings are malformed or hold several ids (E10), so every
  "D<n>:" in each string is counted. Questions with no evidence are reported separately.
  LongMemEval (data/longmemeval/, the oracle file if present, else S): `answer_session_ids`, documented as
  "the evidence sessions". Abstention questions (id ending in _abs) are reported with and without.
For each: the share of questions needing at most 1 and at most 2 sessions, the mean and maximum, by
category or question type, under the variants of which questions count, so the paper's figures can be
matched to a definition. Prints file hashes so the data version is recorded.
Memora itself (Table 1, weekly: consolidation mean 5.3, max 26; mutation mean 2.7, max 11), on all 10
weekly personas and on the 3 development personas. Counting annotations is descriptive and tunes nothing
(Sam, Nov 13, as with E01 and E08); no model sees the test personas. Each question's `memory_evidence` and
`forgetting_evidence` blocks name the sessions behind it (`session_id` fields at any depth, and
`session_history` lists). Reported: sessions behind the presence evidence, behind all evidence, behind the
forgetting evidence, and the number of forgotten items, so the paper's definitions can be matched.
"""
import hashlib
import re
import statistics as st
from collections import Counter, defaultdict

from common import ROOT, WEEKLY, load, personas
from memrecon.memora import dev_personas

PAPER = {"LoCoMo": (94, 1.3, 15), "LongMemEval": (85, 1.9, 6), "Memora weekly": (None, 5.3, 26)}  # % needing <= 2 sessions, mean, max


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def describe(name, counts):
    """counts: list of session counts (>= 1)."""
    if not counts:
        return f"  {name:44} n=   0"
    n = len(counts)
    le1 = sum(c <= 1 for c in counts) / n
    le2 = sum(c <= 2 for c in counts) / n
    return (f"  {name:44} n={n:4d}  <=1 session {100 * le1:5.1f}%  <=2 sessions {100 * le2:5.1f}%  "
            f"mean {st.mean(counts):4.2f}  max {max(counts):2d}")


def dist(counts):
    c = Counter(counts)
    return "  distribution (sessions: questions): " + ", ".join(f"{k}: {c[k]}" for k in sorted(c))


# -- LoCoMo ---------------------------------------------------------------------------------------------
path = ROOT / "data" / "locomo" / "locomo10.json"
print("== LoCoMo ==")
if not path.exists():
    print(f"  missing {path.relative_to(ROOT)}")
else:
    print(f"  file {path.relative_to(ROOT)}  sha256 {sha(path)}…")
    rows = []  # (category, sessions count, had answer)
    empty = Counter()
    odd = 0
    for conv in load(path):
        for q in conv["qa"]:
            ev = q.get("evidence") or []
            sess = set()
            for e in ev:
                found = re.findall(r"D(\d+)\s*:", str(e))
                if not found:
                    odd += 1
                sess.update(int(x) for x in found)
            if not sess:
                empty[q["category"]] += 1
                continue
            rows.append((q["category"], len(sess), "answer" in q))
    total = sum(empty.values()) + len(rows)
    print(f"  {total} questions; {sum(empty.values())} with no usable evidence (by category: {dict(sorted(empty.items()))}); "
          f"{odd} evidence strings with no session id")
    allc = [c for _, c, _ in rows]
    print(describe("all questions with evidence", allc))
    print(describe("excluding category 5", [c for k, c, _ in rows if k != 5]))
    print(describe("only questions with an answer field", [c for _, c, a in rows if a]))
    print(dist(allc))
    for k in sorted({k for k, _, _ in rows}):
        print(describe(f"category {k}", [c for kk, c, _ in rows if kk == k]))
    p = PAPER["LoCoMo"]
    print(f"  paper: {p[0]}% need <= 2 sessions; mean {p[1]}, max {p[2]}")
print()

# -- LongMemEval ----------------------------------------------------------------------------------------
D = ROOT / "data" / "longmemeval"
files = [D / n for n in ("longmemeval_oracle.json", "longmemeval_s_cleaned.json", "longmemeval_s.json") if (D / n).exists()]
print("== LongMemEval ==")
if not files:
    print(f"  no LongMemEval file in {D.relative_to(ROOT)}")
else:
    path = files[0]
    print(f"  file {path.relative_to(ROOT)}  sha256 {sha(path)}…")
    data = load(path)
    rows = []  # (type, sessions, abstention)
    none = 0
    for q in data:
        ids = q.get("answer_session_ids") or []
        if not ids:
            none += 1
            continue
        rows.append((q["question_type"], len(set(ids)), str(q["question_id"]).endswith("_abs")))
    print(f"  {len(data)} questions; {none} with no answer_session_ids")
    allc = [c for _, c, _ in rows]
    print(describe("all questions", allc))
    print(describe("excluding abstention", [c for _, c, a in rows if not a]))
    print(dist(allc))
    by = defaultdict(list)
    for t, c, _ in rows:
        by[t].append(c)
    for t in sorted(by):
        print(describe(t, by[t]))
    p = PAPER["LongMemEval"]
    print(f"  paper: {p[0]}% need <= 2 sessions; mean {p[1]}, max {p[2]}")
print()

# -- Memora (development personas only) ---------------------------------------------------------------
def sessions_in(x, acc):
    if isinstance(x, dict):
        for k, v in x.items():
            if k == "session_id" and isinstance(v, int):
                acc.add(v)
            elif k == "session_history" and isinstance(v, list):
                acc.update(i for i in v if isinstance(i, int))
            else:
                sessions_in(v, acc)
    elif isinstance(x, list):
        for v in x:
            sessions_in(v, acc)
    return acc


rows = []  # (persona, category, presence sessions, all sessions, forgetting sessions, forgotten items)
for persona in personas():
    f = WEEKLY / persona / f"evaluation_questions_{persona}.json"
    for cat, qs in load(f)["questions"].items():
        for q in qs:
            sp = sessions_in(q.get("memory_evidence") or {}, set())
            sf = sessions_in(q.get("forgetting_evidence") or {}, set())
            items = (q.get("forgetting_evidence") or {}).get("forgotten_items") or []
            rows.append((persona, cat, len(sp), len(sp | sf), len(sf), len(items)))

for label, keep in (("all 10 weekly personas (as the paper)", set(personas())), ("development personas only", set(dev_personas()))):
    sub = [r for r in rows if r[0] in keep]
    print(f"== Memora, weekly, {label}: {len(sub)} questions ==")
    print(describe("sessions behind the presence evidence", [r[2] for r in sub if r[2]]))
    print(describe("sessions behind all evidence", [r[3] for r in sub if r[3]]))
    for cat in ("remembering", "reasoning", "recommending"):
        print(describe(f"  all evidence, {cat}", [r[3] for r in sub if r[1] == cat and r[3]]))
    fs = [r[4] for r in sub]
    fi = [r[5] for r in sub]
    print(f"  sessions behind the forgetting evidence: mean {st.mean(fs):.2f}, max {max(fs)}; forgotten items: mean {st.mean(fi):.2f}, max {max(fi)}")
    print()
print("  paper (all 10 personas): consolidation mean 5.3, max 26; mutation mean 2.7, max 11")
