"""Every live memory's similarity to one user turn, from the embedding cache (for the walkthrough's retrieval step).

A recall run saves only the memories that pass the threshold (top 3 at >= 0.70). This rebuilds the store
just before the given turn from the run's event file, looks up the cached vectors for every live memory
and for the turn's message, and scores them the way retrieval does (dot product of unit vectors, D14).
It makes no model calls: if a vector is missing from the cache it stops and says so.

Usage (where the run was made, so results/embeddings.sqlite has the vectors):
  uv run python scripts/turn_scores.py results/stage0/control/73328e6d45/financial_analyst/note/run_1 --session 36 --turn 14
Writes notes/evidence/turn-scores-<persona>-<arm>-s<session>-t<turn>.json and prints a summary.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from memrecon.embed import Embedder
from memrecon.memora import ROOT, sessions as load_sessions

ap = argparse.ArgumentParser()
ap.add_argument("run", help="a run directory: .../<persona>/<arm>/run_<n>")
ap.add_argument("--session", type=int, required=True)
ap.add_argument("--turn", type=int, required=True)
ap.add_argument("--store", default="B", help="which store's event file (B: recall + discard + ingest)")
args = ap.parse_args()

run = Path(args.run)
persona, arm = run.parent.parent.name, run.parent.name
target = {"session_id": args.session, "turn": args.turn}

# the store just before this turn's recall: replay the event file up to the turn's first event
live: dict[str, str] = {}
for line in open(run / args.store / "events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("at") == target:
        break
    op = e["op"]
    if op == "note":
        live[e["id"]] = e["text"]
        for old in e.get("supersedes", []):
            live.pop(old, None)
    elif op == "forget":
        for old in e.get("ids", []):
            live.pop(old, None)
    elif op in ("replace", "keep"):
        live[e["id"]] = e["text"]
else:
    sys.exit(f"no event at session {args.session}, turn {args.turn} in {run / args.store / 'events.jsonl'}")

# the turn's message, numbered the way recall numbers turns
session = next((s for s in load_sessions(persona) if s["session_id"] == args.session), None)
if session is None:
    sys.exit(f"no session {args.session} for {persona}")
message = None
for n, t in enumerate(session.get("conversation", []), start=1):
    if t.get("speaker") == "user_agent" and t.get("turn", n) == args.turn:
        message = t.get("message", "")
if message is None:
    sys.exit(f"no user turn {args.turn} in session {args.session}")

emb = Embedder()
ids = list(live)
docs = [emb.prefixed(live[i], "document") for i in ids]
query = emb.prefixed(message, "query")
keys = [emb.key(t) for t in docs] + [emb.key(query)]
found = emb._lookup(keys)
missing = [k for k in keys if k not in found]
if missing:
    sys.exit(f"{len(missing)} of {len(keys)} vectors are not in the cache; run this where the run was made")
matrix = np.asarray([found[emb.key(t)] for t in docs], dtype=np.float32)
q = np.asarray(found[emb.key(query)], dtype=np.float32)
sims = matrix @ q
order = np.argsort(-sims, kind="stable")
scores = [{"id": ids[i], "score": round(float(sims[i]), 4), "text": live[ids[i]]} for i in order]

# check against what the run logged for this turn
logged = None
summary = run / "sessions" / f"session_{args.session:03d}.json"
if summary.exists():
    for t in json.loads(summary.read_text()).get("turns", []):
        if t.get("turn") == args.turn:
            logged = dict(zip(t.get("found", []), t.get("scores", [])))
out = {"persona": persona, "arm": arm, "session": args.session, "turn": args.turn, "message": message,
       "embed_model": emb.model, "live": len(ids), "logged_hits": logged, "scores": scores}
dest = ROOT / "notes" / "evidence" / f"turn-scores-{persona}-{arm}-s{args.session}-t{args.turn}.json"
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"{persona} {arm} session {args.session} turn {args.turn}: {len(ids)} live memories")
for s in scores[:8]:
    print(f"  {s['score']:.4f}  {s['id']}  {s['text'][:70]}")
if logged:
    ok = all(abs(next(s['score'] for s in scores if s['id'] == m) - v) < 1e-3 for m, v in logged.items())
    print("matches the run's logged scores" if ok else f"DOES NOT match the run's logged scores: {logged}")
print(f"wrote {dest.relative_to(ROOT)}")
