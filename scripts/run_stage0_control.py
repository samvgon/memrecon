"""Stage 0 control check: discard leaves no trace (D19, D26). See src/memrecon/stage0.py.

For each development persona and arm (note, log), replays the sessions twice side by side, ingest only
(A) and recall + discard + ingest (B), and checks after every session that the two stores' contents are
identical. Pass: every session identical, for every persona and arm. Also reports how often recall found,
selected and rewrote memories (D54) and every failure.

Usage (on the EVO-X2, agent and embedding servers up; inside tmux for the full run):
  uv run python scripts/run_stage0_control.py                     # all development personas, both arms
  uv run python scripts/run_stage0_control.py --sessions 20 --personas financial_analyst --arms note   # pilot
Results: results/stage0/control/<config id>/<persona>/<arm>/run_<n>/ (sessions/, A/ and B/ snapshots and
event files, summary.json), and report.md beside them; progress in results/stage0/control/progress.log.
A finished persona and arm is skipped; an interrupted one starts a fresh attempt, replaying from the cache.
A pilot (--sessions) goes to its own config id, so it never counts as the full run.
"""
import argparse
import json
import sys
import time

from memrecon.embed import Embedder
from memrecon.llm import client as model_client, endpoint
from memrecon.memora import ROOT, dev_personas, sessions as load_sessions
from memrecon.monitor import wait_for_memory
from memrecon.recall import RECALL_K, RECALL_THRESHOLD, REWRITE_PROMPT, SELECT_PROMPT
from memrecon.stage0 import ARMS, ControlResult, config_id, control_run, finished, next_attempt

ap = argparse.ArgumentParser()
ap.add_argument("--personas", nargs="*", default=None, help="development personas only (default: all three)")
ap.add_argument("--arms", nargs="*", default=list(ARMS), choices=list(ARMS))
ap.add_argument("--sessions", type=int, default=0, help="pilot: only the first N sessions")
args = ap.parse_args()

dev = dev_personas()
personas = args.personas or dev
if any(p not in dev for p in personas):
    sys.exit(f"Stage 0 runs on the development personas only ({', '.join(dev)}).")
config = {"stage": "0-control", "arms": {a: ARMS[a] for a in args.arms}, "select_prompt": SELECT_PROMPT, "rewrite_prompt": REWRITE_PROMPT,
          "recall_k": RECALL_K, "recall_threshold": RECALL_THRESHOLD, "agent": endpoint("agent").model, "sessions": args.sessions or "all"}
root = ROOT / "results" / "stage0" / "control" / config_id(config)
root.mkdir(parents=True, exist_ok=True)
(root / "config.json").write_text(json.dumps(config, indent=1, sort_keys=True))
log_path = ROOT / "results" / "stage0" / "control" / "progress.log"


def progress(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


agent = model_client("agent")
embedder = Embedder()
summaries = []
for persona in personas:
    sessions = load_sessions(persona)
    if args.sessions:
        sessions = sessions[:args.sessions]
    for arm in args.arms:
        unit = root / persona / arm
        done = finished(unit)
        if done:
            progress(f"{persona} {arm}: already finished ({done.parent.name}), skipped")
            summaries.append(json.loads(done.read_text()))
            continue
        out = next_attempt(unit)
        progress(f"{persona} {arm}: {len(sessions)} sessions -> {out.relative_to(ROOT)}")
        res: ControlResult = control_run(persona, sessions, arm, agent, embedder, out, tag="stage0-control", progress=progress,
                                         guard=lambda: wait_for_memory(say=progress))
        summaries.append(res.summary())

lines = [f"# Stage 0 control check (config {root.name})", "", f"Config: `{json.dumps(config, sort_keys=True)}`", "",
         "| Persona | Arm | Sessions | Identical | Pass | First difference | Turns with recall | Found | Selected | Rewrites | Identical rewrites | Recall failures | Ingest failures (A/B) | Seconds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for s in summaries:
    fd = s["first_difference"]
    lines.append(f"| {s['persona']} | {s['arm']} | {s['sessions']} | {s['equal']} | {'yes' if s['passed'] else 'NO'} | "
                 f"{'—' if fd is None else 'session ' + str(fd['session'])} | {s['turns_with_recall']} | {s['found']} | {s['selected']} | "
                 f"{s['rewrites']} | {s['unchanged_rewrites']} | {s['failures']} | {s['ingest_failures']['A']}/{s['ingest_failures']['B']} | {s['seconds']} |")
overall = all(s["passed"] for s in summaries)
lines += ["", f"**Overall: {'PASS' if overall else 'FAIL'}**" + (" (pilot: not the gate)" if args.sessions else ""), "",
          "Step 1's answers (D54; to see whether the answer format steers them): " + "; ".join(
              f"{s['persona']} {s['arm']}: " + ", ".join(f"{k} {v}" for k, v in s["choices"].items()) for s in summaries)]
for s in summaries:
    if s["first_difference"]:
        lines += ["", f"First difference, {s['persona']} {s['arm']} session {s['first_difference']['session']}:", "```", s["first_difference"]["where"], "```"]
(root / "report.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
