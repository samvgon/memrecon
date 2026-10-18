"""E10: what does LoCoMo (locomo10.json) contain, and could it supply probes for repeated recall?

Question: how long are the conversations, what kinds of questions are there, and does each question
point to the dialogue turns that hold its answer (`evidence`)?
Method: read data/locomo/locomo10.json (downloaded from github.com/snap-research/locomo, data/locomo10.json),
count sessions, turns, characters (tokens estimated as characters / 4, not a tokenizer count), questions by
`category`, evidence references, and which questions have an `answer`. Prints the file's sha256 so the
exact data version is recorded (the data folder is not a git clone, and the file on `main` can change).
The meaning of the category numbers is NOT documented in the README we read, so they are reported as numbers.
"""
import hashlib
import re
from collections import Counter

from common import ROOT, load

PATH = ROOT / "data" / "locomo" / "locomo10.json"
if not PATH.exists():
    raise SystemExit(f"Missing {PATH}. Download it first (see the decision log, section 5).")

print(f"file: {PATH.relative_to(ROOT)}  {PATH.stat().st_size} bytes  sha256 {hashlib.sha256(PATH.read_bytes()).hexdigest()}")
data = load(PATH)
print(f"conversations: {len(data)}")
print()
print(f"{'sample_id':10} {'sessions':>8} {'turns':>6} {'~tokens':>8} {'questions':>9}")
cat = Counter()
no_answer = Counter()
ev_empty = 0
ev_total = 0
ev_refs = Counter()
bad_ref = 0
qtot = 0
tot_tokens = 0
for conv in data:
    c = conv["conversation"]
    sess = sorted(k for k in c if re.fullmatch(r"session_\d+", k))
    turns = chars = 0
    ids = set()
    for k in sess:
        for t in c[k]:
            turns += 1
            chars += len(t.get("text", ""))
            ids.add(t.get("dia_id"))
    qa = conv["qa"]
    print(f"{conv['sample_id']:10} {len(sess):8d} {turns:6d} {chars // 4:8d} {len(qa):9d}")
    tot_tokens += chars // 4
    for q in qa:
        qtot += 1
        cat[q["category"]] += 1
        if "answer" not in q:
            no_answer[q["category"]] += 1
        ev = q.get("evidence", [])
        ev_total += 1
        if not ev:
            ev_empty += 1
        ev_refs[len(ev)] += 1
        for e in ev:
            # evidence strings look like 'D1:3'; some are malformed or hold several ids
            if e not in ids:
                bad_ref += 1
print()
print(f"all conversations: about {tot_tokens} tokens (characters / 4); mean about {tot_tokens // len(data)}")
print(f"questions: {qtot}")
print("by category (number only; meaning not documented in the README we read):")
for k in sorted(cat):
    print(f"  category {k}: {cat[k]}   without an 'answer' field: {no_answer[k]}")
print(f"questions with no evidence listed: {ev_empty} of {ev_total}")
print("number of evidence references per question:", dict(sorted(ev_refs.items())))
print(f"evidence references that do not match any turn id exactly: {bad_ref}")
print()
print("example questions (first of each category):")
seen = set()
for conv in data:
    for q in conv["qa"]:
        if q["category"] not in seen:
            seen.add(q["category"])
            print(f"  [cat {q['category']}] {q['question']} -> {q.get('answer', q.get('adversarial_answer'))!r}  evidence={q.get('evidence')}")
