"""E03: what does the dialogue actually say, compared with the log?

Question: when the log records an add, update or delete, is the content spoken in the conversation?
Method: print the log metadata and every turn (flagged turns marked) for three sessions of
one persona: the add (0089), the delete (0109) and the update (0110) for `project_proposal_2`.
This is a three-session example, not a systematic check (see E06).
"""
import json

from common import WEEKLY, load

PERSONA = "creative_designer"
for num in (89, 109, 110):
    path = WEEKLY / PERSONA / "conversations" / f"session_{num:04d}.json"
    if not path.exists():
        print(f"--- {path.name}: missing")
        continue
    s = load(path)
    meta = {k: v for k, v in s.items() if k != "conversation"}
    print(f"=== {PERSONA}/{path.name}")
    print(json.dumps(meta, indent=1, ensure_ascii=False)[:2500])
    print("--- turns ([F] = share_memory true)")
    for t in s["conversation"]:
        mark = "[F]" if t.get("share_memory") else "   "
        print(f"{mark} {t['turn']:2d} {t['speaker']:10s} | {t['message']}")
    print()
