"""Every model call behind one recall turn and the session's ingestion, for the walkthrough (and to fill its gaps).

For one turn of a Stage 0 control run, this sends the turn's calls again exactly as recall sent them, so the
call cache answers them: the select call, and the rewrite of each selected memory with each prompt version
asked for. Ingestion's calls for the same session (extract, then decide) are found in the call cache by their
prompt and the session's content. Only a request that is not in the cache would reach a model, and then only
with --allow-calls (local agent server; never a paid model unless the models file says so and Sam has
allowed it, see CLAUDE.md).

Usage (on the EVO-X2, where results/raw_calls has the run's calls):
  uv run python scripts/turn_trace.py results/stage0/control/73328e6d45/financial_analyst/note/run_1 \
      --session 36 --turn 14 --rewrite-prompts rewrite_v4 rewrite_v5b --find '$6.96 on a coffee'
  add --allow-calls (agent server up: scripts/serve.sh agent) for requests not in the cache.
Writes notes/evidence/turn-trace-<persona>-<arm>-s<session>-t<turn>.json and prints a summary.
"""
import argparse
import json
import sys
from pathlib import Path

from memrecon.ingest import load_prompt, render_conversation
from memrecon.llm import ModelCallError, cache_key, client as model_client
from memrecon.memora import ROOT, sessions as load_sessions
from memrecon.memory import At, MemoryStore
from memrecon.recall import SELECT_PROMPT, Discard, Recaller, memory_slots, parse_rewrite_answer

ap = argparse.ArgumentParser()
ap.add_argument("run", help="a run directory: .../<persona>/<arm>/run_<n>")
ap.add_argument("--session", type=int, required=True)
ap.add_argument("--turn", type=int, required=True)
ap.add_argument("--rewrite-prompts", nargs="+", default=["rewrite_v4"])
ap.add_argument("--find", default=None, help="text from the session that identifies its ingestion calls")
ap.add_argument("--ingest-prompts", nargs="+", default=["extract_note_v2", "decide_note_v3"])
ap.add_argument("--allow-calls", action="store_true", help="send requests that are not in the cache to the agent server")
args = ap.parse_args()

run = Path(args.run)
persona, arm = run.parent.parent.name, run.parent.name
record = json.loads((run / "sessions" / f"session_{args.session:03d}.json").read_text())
turn = next((t for t in record["turns"] if t["turn"] == args.turn), None)
if turn is None:
    sys.exit(f"recall did not run on turn {args.turn} of session {args.session}")


class NotCached(Exception):
    pass


class Gate:
    """The agent client, but a request that is not in the cache stops unless --allow-calls."""

    def __init__(self, inner, allow):
        self.inner, self.allow, self.sent = inner, allow, 0

    def chat(self, messages, *, use_cache=True, tag=None, meta=None, **settings):
        path = self.inner.path_for(cache_key(self.inner.request_body(messages, **settings)))
        if not path.exists():
            if not self.allow:
                raise NotCached("not in the call cache (rerun with --allow-calls to ask the agent)")
            self.sent += 1
        return self.inner.chat(messages, use_cache=use_cache, tag=tag, meta=meta, **settings)


agent = Gate(model_client("agent"), args.allow_calls)

# the store as it was before this turn: text and date of each memory, from the run's event file
mem = {}
for line in open(run / "B" / "events.jsonl", encoding="utf-8"):
    e = json.loads(line)
    if e.get("at") == {"session_id": args.session, "turn": args.turn}:
        break
    if e.get("text") is not None and e.get("id"):
        mem[e["id"]] = (e["text"], e.get("date", mem.get(e["id"], ("", ""))[1]))

session = next(s for s in load_sessions(persona) if s["session_id"] == args.session)
conv = session["conversation"]
n = next(i for i, t in enumerate(conv, start=1) if t.get("speaker") == "user_agent" and t.get("turn", i) == args.turn)
conversation, _ = render_conversation({"conversation": conv[:n]})
date = session.get("date", "")
at = At(args.session, args.turn)
out = {"persona": persona, "arm": arm, "session": args.session, "turn": args.turn, "date": date,
       "message": conv[n - 1]["message"], "shown": turn["shown"], "selected": turn["selected"]}


def ask(rc, system, user, parse, step):
    try:
        value, error = rc._ask(system, user, parse, at, step)
    except NotCached as e:
        return {"error": str(e)}
    except ModelCallError as e:
        return {"error": f"model call failed: {e}"}
    return {"answer": value, "error": error, "cached": all(c["cached"] for c in rc.calls)}


# select: the numbered lines exactly as present() wrote them
lines = "\n".join(f"[{k}] ({mem[m][1]}) {mem[m][0]}" if mem[m][1] else f"[{k}] {mem[m][0]}" for k, m in enumerate(turn["shown"], start=1))
system, template = load_prompt(SELECT_PROMPT)
rc = Recaller(agent, MemoryStore(persona, mode=arm), None, Discard(), persona=persona, tag="turn-trace")
out["select"] = {"prompt": SELECT_PROMPT, "memories": lines,
                 **ask(rc, system, template.format(date=date, conversation=conversation, memories=lines), lambda r: r, "select")}

# rewrites: each selected memory, with each prompt version
out["rewrites"] = {}
for mid in turn["selected"]:
    text, mdate = mem[mid]
    out["rewrites"][mid] = {"memory": text, "memory_date": mdate, "run": turn["rewrites"].get(mid)}
    for version in args.rewrite_prompts:
        system, template = load_prompt(version)
        rc = Recaller(agent, MemoryStore(persona, mode=arm), None, Discard(), persona=persona, rewrite_prompt=version, tag="turn-trace")
        user = template.format(date=date, conversation=conversation, **memory_slots(text, mdate, date))
        out["rewrites"][mid][version] = ask(rc, system, user, parse_rewrite_answer, "rewrite")

# ingestion: the session's calls, found in the cache by prompt and content (no calls are sent)
out["ingestion"] = {}
if args.find:
    wanted = {load_prompt(p)[0]: p for p in args.ingest_prompts}
    for f in sorted((ROOT / "results" / "raw_calls" / "agent").glob("*/*.json")):
        if ".attempt" in f.name:
            continue
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            msgs = rec["request"]["messages"]
        except (json.JSONDecodeError, KeyError):
            continue
        name = wanted.get(msgs[0].get("content", "")) if msgs else None
        if not name or not any(args.find in m.get("content", "") for m in msgs[1:2]):
            continue
        try:
            answer = json.loads(rec["response_text"])["choices"][0]["message"]["content"]
        except (json.JSONDecodeError, KeyError, IndexError, TypeError):
            answer = None
        out["ingestion"].setdefault(name, []).append({"key": rec["key"], "user": msgs[1]["content"], "answer": answer,
                                                      "finished_at": rec.get("finished_at")})

dest = ROOT / "notes" / "evidence" / f"turn-trace-{persona}-{arm}-s{args.session}-t{args.turn}.json"
dest.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

print(f"{persona} {arm} session {args.session} turn {args.turn}")
print(f"  select: {out['select'].get('answer') or out['select'].get('error')}")
for mid, r in out["rewrites"].items():
    for version in args.rewrite_prompts:
        x = r[version]
        print(f"  {mid} {version}: {x.get('answer') or x.get('error')}" + ("" if x.get("cached", True) else "   (new call)"))
for name, calls in out["ingestion"].items():
    print(f"  {name}: {len(calls)} call(s) found in the cache")
print(f"calls sent to the agent: {agent.sent}")
print(f"wrote {dest.relative_to(ROOT)}")
