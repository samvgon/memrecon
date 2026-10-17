"""Recreate or check the frozen dev/held-out persona split (seed 329).

Usage, from anywhere:  python3 scripts/make_split.py
If notes/split.json already exists, this only checks that it matches the recomputation
and never overwrites it. The split is frozen: do not change it after seeing results.
"""
import json
import random
import sys

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "explore"))
from common import ROOT, personas  # noqa: E402

SEED = 329
ps = personas()
rng = random.Random(SEED)
dev = sorted(rng.sample(ps, 3))
new = {"seed": SEED, "dev": dev, "test": sorted(set(ps) - set(dev))}
out = ROOT / "notes" / "split.json"
if out.exists():
    old = json.loads(out.read_text())
    if old == new:
        print("OK: notes/split.json matches the seed-329 recomputation.")
    else:
        print("MISMATCH: notes/split.json differs from the recomputation. Do not overwrite; investigate.")
        print("file:", old)
        print("recomputed:", new)
        sys.exit(1)
else:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(new, indent=1))
    print("Wrote", out, new)
