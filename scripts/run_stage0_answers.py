"""Answer Memora's questions from the end-of-history memory of a Stage 0 control run (D60).

For each development persona and arm, loads store B's last snapshot from the control run (recall with
discard, then ingestion; identical to ingestion alone, E38) and answers the persona's 15 questions with
`answer_v1` (src/memrecon/answer.py). These answers are what the judge is checked on (D21; the labelling
sample is made from them by scripts/make_judge_sample.py) and, once the judge passes, the discard
baseline's FAMA.

Usage (on the EVO-X2, agent and embedding servers up; a few minutes):
  uv run python scripts/run_stage0_answers.py                  # the control run for the current prompts
  uv run python scripts/run_stage0_answers.py --config a0018ce663
Results: results/stage0/answers/<config id>/<persona>/<arm>/answers.jsonl (one line per question:
question, sub-questions, memories shown, answer), written when the unit finishes; a finished unit is skipped.
"""
import argparse
import json
import os
import sys
import time

from memrecon.answer import ANSWER_K, ANSWER_PROMPT, Answerer, questions
from memrecon.embed import Embedder
from memrecon.llm import client as model_client
from memrecon.memora import ROOT, WEEKLY, dev_personas
from memrecon.memory import MemoryStore
from memrecon.monitor import wait_for_memory
from memrecon.recall import REWRITE_PROMPT, SELECT_PROMPT
from memrecon.retrieval import Retriever

CONTROL = ROOT / "results" / "stage0" / "control"
OUT = ROOT / "results" / "stage0" / "answers"

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None, help="control run config id (default: the full run with the current prompts)")
args = ap.parse_args()


def current_config():
    for p in sorted(CONTROL.iterdir()):
        f = p / "config.json"
        if f.exists():
            c = json.loads(f.read_text())
            if c.get("sessions") == "all" and c.get("rewrite_prompt") == REWRITE_PROMPT and c.get("select_prompt") == SELECT_PROMPT \
                    and (p / "report.md").exists():
                return p.name
    sys.exit(f"No finished full control run with {SELECT_PROMPT} and {REWRITE_PROMPT} in {CONTROL}.")


config = args.config or current_config()
control = CONTROL / config
print(f"Control run {config}; answer prompt {ANSWER_PROMPT}, top {ANSWER_K} memories")
agent = model_client("agent")
embedder = Embedder()
for persona in dev_personas():
    qs = questions(json.loads((WEEKLY / persona / f"evaluation_questions_{persona}.json").read_text()))
    for arm in ("note", "log"):
        out = OUT / config / persona / arm / "answers.jsonl"
        if out.exists():
            print(f"{persona} {arm}: already answered, skipped")
            continue
        run = sorted((control / persona / arm).glob("run_*"), key=lambda p: int(p.name[4:]))[-1]
        snaps = sorted((run / "B" / "snapshots").glob("session_*.json"))
        snaps = [s for s in snaps if not s.name.endswith(".summary.json")]
        store = MemoryStore.load(snaps[-1])
        answerer = Answerer(agent, store, Retriever(embedder), persona=persona, tag="stage0-answers")
        t0 = time.monotonic()
        lines = []
        for n, q in enumerate(qs, start=1):
            wait_for_memory(say=print)
            a = answerer.answer(q, n)
            lines.append(json.dumps({**q, "persona": persona, "arm": arm, "snapshot": snaps[-1].name, "active": len(store.active()),
                                     "shown": a.shown, "scores": a.scores, "shown_text": [store.get(m).text for m in a.shown],
                                     "shown_dates": [store.get(m).date for m in a.shown],
                                     "answer": a.text, "error": a.error, "attempts": a.attempts}))
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.replace(tmp, out)
        failed = sum(json.loads(x)["answer"] is None for x in lines)
        print(f"{persona} {arm}: {len(qs)} questions from {snaps[-1].name} ({len(store.active())} live memories), "
              f"{failed} failed, {time.monotonic() - t0:.0f} s -> {out.relative_to(ROOT)}")
