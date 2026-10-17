"""E09: how long is each persona's full conversation history?

Question: can the full history be put in one model context (Memora's Track 1 does this, and it is our
"re-read the raw history" baseline)? How big is the history the memory agent must ingest?
Method: count turns and characters of all messages per persona and estimate tokens as characters / 4
(a rough rule of thumb for English text; this is not a tokenizer count, so check with the real tokenizer
before sizing a context window). Message text only: speaker labels, dates and JSON are not counted.
"""
from common import iter_sessions, personas

rows = {}
for persona, path, s in iter_sessions():
    r = rows.setdefault(persona, {"sessions": 0, "turns": 0, "chars": 0, "max_session_chars": 0})
    chars = sum(len(t["message"]) for t in s["conversation"])
    r["sessions"] += 1
    r["turns"] += len(s["conversation"])
    r["chars"] += chars
    r["max_session_chars"] = max(r["max_session_chars"], chars)

print(f"{'persona':24} {'sessions':>8} {'turns':>6} {'chars':>9} {'~tokens':>8} {'longest session (~tokens)':>26}")
for p in personas():
    r = rows[p]
    print(f"{p:24} {r['sessions']:8d} {r['turns']:6d} {r['chars']:9d} {r['chars'] // 4:8d} {r['max_session_chars'] // 4:26d}")
tot = sum(r["chars"] for r in rows.values())
print()
print(f"all personas: {tot} characters, about {tot // 4} tokens; mean per persona about {tot // 4 // len(rows)} tokens")
print(f"largest persona: about {max(r['chars'] for r in rows.values()) // 4} tokens (estimate = characters / 4)")
