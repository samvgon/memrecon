"""Stage 0's control check: discard leaves no trace (D19, D20, D26, D54).

For one persona and one arm, two stores replay the same sessions side by side:

- **A, ingest only**: each session is ingested once it ends; nothing is ever recalled;
- **B, discard**: during each session every user turn goes through the recall path (retrieve, count,
  select, rewrite each chosen memory, discard the rewrite), then the session is ingested as in A.

After every session the two stores' `content_json()` (texts, depths, versions, sources, IDs; not the
recall counters) must be identical. Both make the same ingestion requests, which the call cache answers
with the same saved text, so any difference means the recall path changed memory, or changed what
ingestion was asked: a bug. Arms: **note** (two-step v3 ingestion, frozen in D52) and **log** (the user's
own words, `ingest_log_v1`; D24's verbatim tier; write policies never run on it, discard does).

Durability (CLAUDE.md, D30): each session's result is written atomically to its own file, both stores
are snapshotted after every session, and each store's operations go to its own event file, all inside a
fresh `run_<n>/` folder per attempt, so nothing an earlier attempt wrote is overwritten. A finished
attempt (its `summary.json` exists) is not run again; an interrupted one is left as it is and the next
attempt replays from the start, where the calls already made come from the cache.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from memrecon.ingest import Ingestor
from memrecon.memory import EventLog, MemoryStore, first_difference
from memrecon.monitor import save_summary
from memrecon.recall import Discard, Recaller
from memrecon.retrieval import Retriever

INGEST_K = 3  # retrieval for ingestion's related memories (as in E23)
ARMS = {
    "note": {"mode": "note", "method": "ingest_two_step", "extract_prompt": "extract_note_v2", "decide_prompt": "decide_note_v3",
             "note_prompt": "ingest_note_v5", "candidate_threshold": 0.0},
    "log": {"mode": "log", "method": "ingest_log", "log_prompt": "ingest_log_v1"},
}


CHOICE_PATTERNS = ("none", "all, in listed order", "all, reversed", "some", "one shown")


def choice_pattern(shown: list[str], selected: list[str]) -> str:
    """How step 1's answer relates to the order it was shown (to see whether the answer format steers it)."""
    if not selected:
        return "none"
    if len(shown) == 1:
        return "one shown"
    if selected == shown:
        return "all, in listed order"
    if selected == shown[::-1]:
        return "all, reversed"
    return "some"


def write_json(path: Path, data) -> None:
    """Atomically: temp file, fsync, rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def config_id(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:10]


@dataclass
class SessionCheck:
    session: int
    equal: bool
    difference: str | None
    turns_found: int
    found: int
    selected: int
    rewrites: int
    unchanged: int
    failures: int
    choices: dict  # step 1's answers by pattern (D54): none, all in listed order, all reversed, some
    ingest_ok_a: bool
    ingest_ok_b: bool
    seconds: float


@dataclass
class ControlResult:
    persona: str
    arm: str
    sessions: list[SessionCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return bool(self.sessions) and all(s.equal for s in self.sessions)

    def summary(self) -> dict:
        s = self.sessions
        first = next((x for x in s if not x.equal), None)
        return {"persona": self.persona, "arm": self.arm, "sessions": len(s), "equal": sum(x.equal for x in s),
                "passed": self.passed, "first_difference": None if first is None else {"session": first.session, "where": first.difference},
                "turns_with_recall": sum(x.turns_found for x in s), "found": sum(x.found for x in s),
                "selected": sum(x.selected for x in s), "rewrites": sum(x.rewrites for x in s),
                "unchanged_rewrites": sum(x.unchanged for x in s), "failures": sum(x.failures for x in s),
                "choices": {k: sum(x.choices.get(k, 0) for x in s) for k in CHOICE_PATTERNS},
                "ingest_failures": {"A": sum(not x.ingest_ok_a for x in s), "B": sum(not x.ingest_ok_b for x in s)},
                "seconds": round(sum(x.seconds for x in s), 1)}


def finished(unit_dir: Path) -> Path | None:
    """The summary of a finished attempt in this unit's folder, if any."""
    done = sorted(Path(unit_dir).glob("run_*/summary.json"))
    return done[-1] if done else None


def next_attempt(unit_dir: Path) -> Path:
    n = 1
    while (Path(unit_dir) / f"run_{n}").exists():
        n += 1
    return Path(unit_dir) / f"run_{n}"


def control_run(persona: str, sessions: list[dict], arm: str, agent, embedder, out_dir: Path, tag: str = "stage0",
                progress=None, guard=None) -> ControlResult:
    """Run A (ingest only) and B (recall + discard, then ingest) side by side; compare after every session.
    `out_dir` is one attempt's folder (see next_attempt)."""
    spec = ARMS[arm]
    out_dir = Path(out_dir)
    stores, ingestors = {}, {}
    for name in ("A", "B"):
        store = MemoryStore(persona, mode=spec["mode"], events=EventLog(out_dir / name / "events.jsonl"))
        kwargs = {k: v for k, v in spec.items() if k not in ("mode", "method")}
        ingestors[name] = Ingestor(agent, store, Retriever(embedder, k=INGEST_K), persona=persona, tag=tag, **kwargs)
        stores[name] = store
    recaller = Recaller(agent, stores["B"], Retriever(embedder), Discard(), persona=persona, tag=tag)
    result = ControlResult(persona, arm)
    for s in sessions:
        if guard:
            guard()  # the memory guard before each unit of work (D31)
        t0 = time.monotonic()
        sid = s["session_id"]
        # A: ingest only
        a_ing = ingestors["A"]
        if stores["A"].active():
            a_ing.retriever.sync(stores["A"])
        out_a = getattr(a_ing, spec["method"])(s)
        # B: recall on every user turn (discard), then ingest
        rec = recaller.recall_session(s)
        b_ing = ingestors["B"]
        if stores["B"].active():
            b_ing.retriever.sync(stores["B"])
        out_b = getattr(b_ing, spec["method"])(s)
        ja, jb = stores["A"].content_json(), stores["B"].content_json()
        check = SessionCheck(
            session=sid, equal=ja == jb, difference=None if ja == jb else first_difference(ja, jb),
            turns_found=len(rec.turns), found=rec.found, selected=rec.selected,
            rewrites=sum(len(t.rewrites) for t in rec.turns), unchanged=sum(len(t.unchanged) for t in rec.turns),
            failures=rec.failures,
            choices=dict(Counter(choice_pattern(t.shown, t.selected) for t in rec.turns if t.select_ok)), ingest_ok_a=out_a.ok, ingest_ok_b=out_b.ok, seconds=round(time.monotonic() - t0, 2))
        result.sessions.append(check)
        stores["A"].save(out_dir / "A" / "snapshots" / f"session_{sid:03d}.json")
        stores["B"].save(out_dir / "B" / "snapshots" / f"session_{sid:03d}.json")
        save_summary(stores["B"], out_dir / "B" / "snapshots" / f"session_{sid:03d}.json")  # for the monitor (D31)
        write_json(out_dir / "sessions" / f"session_{sid:03d}.json", {
            **check.__dict__, "turns": [{"turn": t.turn, "found": t.found, "shown": t.shown, "scores": [round(x, 4) for x in t.scores],
                                         "selected": t.selected, "rewrites": t.rewrites, "unchanged": t.unchanged,
                                         "select_ok": t.select_ok, "rewrite_failed": t.rewrite_failed, "errors": t.errors}
                                        for t in rec.turns]})
        if progress:
            progress(f"{persona} {arm} session {sid}: equal={check.equal} found={check.found} selected={check.selected} "
                     f"failures={check.failures} ({check.seconds:.1f} s)")
    write_json(out_dir / "summary.json", result.summary())
    return result
