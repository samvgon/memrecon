"""Live view of a run on the EVO-X2: machine memory, model servers, token use, agent memory, disk (D31).

    uv run python scripts/monitor.py              # refresh every 5 s; Ctrl-C to stop
    uv run python scripts/monitor.py --once       # print once and exit
    uv run python scripts/monitor.py --raw-metrics  # also list every server metric

Run it in its own tmux session (`tmux new -s monitor`) next to a run. It only
reads: system and GPU memory, each local server's /metrics (start servers with
scripts/serve.sh, which turns metrics on), the ledger, the newest
*.summary.json in each run folder under results/, and the size of results/.
"""

import argparse
import json
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

from memrecon.llm import REPO_ROOT, roles
from memrecon.monitor import (
    GiB,
    dir_size,
    gpu_memory,
    latest_summaries,
    ledger_rates,
    now_iso,
    server_metrics,
    system_memory,
)

RESULTS = REPO_ROOT / "results"
LEDGER = RESULTS / "raw_calls" / "ledger.jsonl"
WINDOW_S = 300

SHOWN = [  # llama-server metric -> label; whichever exist on this build are shown
    ("llamacpp:prompt_tokens_seconds", "prompt tok/s"),
    ("llamacpp:predicted_tokens_seconds", "gen tok/s"),
    ("llamacpp:requests_processing", "busy"),
    ("llamacpp:requests_deferred", "queued"),
    ("llamacpp:kv_cache_usage_ratio", "KV used"),
    ("llamacpp:kv_cache_tokens", "KV tokens"),
]


class LedgerTail:
    """Reads only new ledger lines on each refresh and keeps the last WINDOW_S seconds."""

    def __init__(self, path):
        self.path, self.offset, self.rows = Path(path), 0, deque()

    def update(self):
        if not self.path.exists():
            return list(self.rows)
        with open(self.path, encoding="utf-8") as f:
            f.seek(self.offset)
            for line in f:
                if not line.endswith("\n"):
                    break  # a line still being written; read it next time
                self.offset += len(line.encode("utf-8"))
                try:
                    self.rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        while len(self.rows) > 100_000:
            self.rows.popleft()
        return list(self.rows)


def healthy(port):
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/health", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def fmt_gb(b):
    return "?" if b is None else f"{b / GiB:.1f}"


def render(tail, disk_cache, raw_metrics):
    out = [f"memrecon monitor   {now_iso()}   (Ctrl-C to stop)", ""]

    out.append("MACHINE")
    mem = system_memory()
    guard = float(os.environ.get("MEMRECON_MIN_FREE_GB", 8))
    if mem:
        flag = "   << BELOW GUARD" if mem["available"] < guard * GiB else ""
        out.append(f"  system memory: {fmt_gb(mem['available'])} GB available of {fmt_gb(mem['total'])} GB (guard pauses work below {guard:g} GB){flag}")
    else:
        out.append("  system memory: not available here (no /proc/meminfo)")
    for g in gpu_memory():
        out.append(
            f"  GPU {g['card']}: VRAM {fmt_gb(g['vram_used'])}/{fmt_gb(g['vram_total'])} GB, "
            f"GTT (system memory used by the GPU) {fmt_gb(g['gtt_used'])}/{fmt_gb(g['gtt_total'])} GB"
        )

    out += ["", "MODEL SERVERS"]
    for role, s in roles().items():
        if s["PROVIDER"] != "llamacpp" or not s["PORT"]:
            continue
        port = int(s["PORT"])
        if not healthy(port):
            out.append(f"  {role:<12} port {port}: not answering")
            continue
        m = server_metrics(port)
        if m is None:
            out.append(f"  {role:<12} port {port}: healthy (no metrics: restart it with scripts/serve.sh)")
            continue
        parts = []
        for name, label in SHOWN:
            if name in m:
                v = m[name]
                parts.append(f"{label} {v * 100:.0f}%" if name.endswith("ratio") else f"{label} {v:,.0f}" if v >= 10 else f"{label} {v:.1f}")
        out.append(f"  {role:<12} port {port}: healthy | " + (" | ".join(parts) or "metrics on, none of the usual names"))
        if raw_metrics:
            out += [f"      {k} = {v:g}" for k, v in sorted(m.items()) if k.startswith("llamacpp:")]

    out += ["", f"TOKENS (last {WINDOW_S // 60} min, from the ledger; cache hits not counted)"]
    rates = ledger_rates(tail.update(), window_s=WINDOW_S)
    if not rates:
        out.append("  no calls in this window")
    for role, r in sorted(rates.items()):
        out.append(
            f"  {role:<12} {r['calls']:>6} calls | prompt {r['prompt_tok_per_s']:,.0f} tok/s | "
            f"output {r['output_tok_per_s']:,.1f} tok/s | memory in prompts {r['memory_tokens']:,} tok"
            + (f" | {r['rejected']} rejected" if r["rejected"] else "")
        )

    out += ["", "AGENT MEMORY (as in real use: live memories only; newest summary per run folder)"]
    summaries = latest_summaries(RESULTS) if RESULTS.exists() else []
    if not summaries:
        out.append("  no run summaries yet")
    for path, s in summaries[-12:]:
        where = path.parent.relative_to(RESULTS)
        out.append(
            f"  {str(where):<40} {s['live_memories']:>5} live | ~{s['live_text_tokens_est']:,} tok (est.) | "
            f"max depth {s['max_depth']} | depths {s['depths']} | retired {s['retired'] or '-'}"
        )

    out += ["", "EXPERIMENT OVERHEAD (audit log, cache, snapshots on disk)"]
    if time.time() - disk_cache.get("t", 0) > 60:
        disk_cache.update(t=time.time(), size=dir_size(RESULTS))
    size, files = disk_cache["size"]
    out.append(f"  results/: {size / GiB:.2f} GB in {files:,} files (rechecked every 60 s)")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--raw-metrics", action="store_true")
    args = ap.parse_args()
    tail, disk = LedgerTail(LEDGER), {}
    try:
        while True:
            text = render(tail, disk, args.raw_metrics)
            if args.once:
                print(text)
                return 0
            sys.stdout.write("\033[2J\033[H" + text + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
