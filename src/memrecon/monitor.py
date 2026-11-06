"""Watching a run: machine memory, model servers, token use, and the agent's memory (D31).

Two kinds of memory are watched, and kept apart:

- **the machine's memory**, so a run never crashes for lack of it: system
  memory (/proc/meminfo), GPU memory (amdgpu's sysfs counters), each local
  model server's own metrics (llama-server started with --metrics), and the
  size of results/ on disk. `wait_for_memory` is the guard a runner calls
  before each unit of work: below the threshold it pauses instead of
  starting work that could run out of memory;
- **the agent's memory as it would be in real use**, without the experiment's
  bookkeeping: `store_summary` (number of live memories, their text size,
  depths), written next to each snapshot, and the memory tokens each prompt
  carries (recorded in the ledger by the caller as `meta`). The audit log,
  attempts, cache and snapshots are experiment overhead and are reported
  separately (`results/` on disk).

`scripts/monitor.py` shows all of this, refreshed every few seconds.
Everything here only reads; nothing is ever shown to the model.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

GiB = 1024**3


class MemoryPressure(RuntimeError):
    """Free memory stayed below the threshold for longer than the guard was allowed to wait."""


# -- machine memory -----------------------------------------------------------------


def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo as bytes, e.g. {"MemTotal": ..., "MemAvailable": ...}."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+):\s+(\d+)(?:\s+kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if line.rstrip().endswith("kB") else 1)
    return out


def system_memory(meminfo: str | Path = "/proc/meminfo") -> dict[str, int] | None:
    """Total and available system memory in bytes, or None where /proc/meminfo doesn't exist (e.g. macOS)."""
    p = Path(meminfo)
    if not p.exists():
        return None
    info = parse_meminfo(p.read_text())
    return {"total": info.get("MemTotal", 0), "available": info.get("MemAvailable", 0)}


def gpu_memory(drm: str | Path = "/sys/class/drm") -> list[dict]:
    """amdgpu memory counters per card, in bytes: dedicated (VRAM) and system memory mapped for the GPU (GTT).

    On the EVO-X2's integrated GPU, VRAM is the share of memory set aside in the
    BIOS and GTT is ordinary system memory the GPU uses, which also shows up as
    used system memory. Empty list if there is no amdgpu device.
    """
    cards = []
    for dev in sorted(Path(drm).glob("card*/device")):
        def read(name: str) -> int | None:
            f = dev / f"mem_info_{name}"
            try:
                return int(f.read_text().strip())
            except (OSError, ValueError):
                return None

        vals = {k: read(k) for k in ("vram_used", "vram_total", "gtt_used", "gtt_total")}
        if any(v is not None for v in vals.values()):
            cards.append({"card": dev.parent.name, **vals})
    return cards


def wait_for_memory(
    min_available_gb: float | None = None,
    poll_s: float = 30.0,
    max_wait_s: float = 3600.0,
    meminfo: str | Path = "/proc/meminfo",
    sleep=time.sleep,
    say=print,
) -> None:
    """Guard before each unit of work: wait while available memory is below the threshold.

    Threshold from the argument, else MEMRECON_MIN_FREE_GB, else 8 GB. Raises
    MemoryPressure if memory doesn't recover within max_wait_s, so the runner
    stops cleanly between units (a resumed run picks up where it stopped).
    Does nothing where /proc/meminfo doesn't exist.
    """
    limit = float(min_available_gb if min_available_gb is not None else os.environ.get("MEMRECON_MIN_FREE_GB", 8)) * GiB
    waited = 0.0
    while True:
        mem = system_memory(meminfo)
        if mem is None or mem["available"] >= limit:
            return
        if waited >= max_wait_s:
            raise MemoryPressure(
                f"only {mem['available'] / GiB:.1f} GB available (need {limit / GiB:.1f}) after waiting {waited:.0f} s"
            )
        say(f"memory guard: {mem['available'] / GiB:.1f} GB available, below {limit / GiB:.1f} GB; waiting {poll_s:.0f} s")
        sleep(poll_s)
        waited += poll_s


# -- model servers ---------------------------------------------------------------------


def parse_prometheus(text: str) -> dict[str, float]:
    """Prometheus text format (llama-server's /metrics) as {name: value}; labels are dropped."""
    out = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        m = re.match(r"([A-Za-z_:][\w:]*)(?:\{[^}]*\})?\s+([-+eE0-9.naNinf]+)", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def server_metrics(port: int, timeout: float = 2.0) -> dict[str, float] | None:
    """A local llama-server's metrics, or None if it isn't answering or wasn't started with --metrics."""
    try:
        with urllib.request.urlopen(f"http://localhost:{port}/metrics", timeout=timeout) as r:
            return parse_prometheus(r.read().decode("utf-8", errors="replace"))
    except Exception:
        return None


# -- token use (from the ledger) ---------------------------------------------------------


def _parse_time(value: str) -> float | None:
    try:
        return datetime.fromisoformat(value).timestamp()
    except (TypeError, ValueError):
        return None


def ledger_rates(rows: list[dict], now: float | None = None, window_s: float = 300.0) -> dict[str, dict]:
    """Per role over the last `window_s` seconds: calls, prompt and output tokens per second, memory tokens, rejections."""
    now = time.time() if now is None else now
    out: dict[str, dict] = {}
    for row in rows:
        t = _parse_time(row.get("at", ""))
        if t is None or now - t > window_s:
            continue
        r = out.setdefault(
            row.get("role", "?"), {"calls": 0, "prompt_tokens": 0, "output_tokens": 0, "memory_tokens": 0, "rejected": 0}
        )
        r["calls"] += 1
        r["prompt_tokens"] += row.get("prompt_tokens") or 0
        r["output_tokens"] += row.get("completion_tokens") or 0
        r["memory_tokens"] += (row.get("meta") or {}).get("memory_tokens") or 0
        if row.get("outcome") not in (None, "ok", "empty"):
            r["rejected"] += 1
    for r in out.values():
        r["prompt_tok_per_s"] = r["prompt_tokens"] / window_s
        r["output_tok_per_s"] = r["output_tokens"] / window_s
    return out


# -- the agent's memory ---------------------------------------------------------------------


def approx_tokens(text: str) -> int:
    """Rough token count (characters / 4), labelled as an estimate wherever it is shown."""
    return (len(text) + 3) // 4


def store_summary(store) -> dict:
    """Size of the agent's memory as it would be in use: live memories only, no audit-log bookkeeping."""
    live = store.active()
    retired = Counter(m.retired for m in store.memories.values() if m.retired)
    return {
        "persona": store.persona,
        "mode": store.mode,
        "live_memories": len(live),
        "live_text_chars": sum(len(m.text) for m in live),
        "live_text_tokens_est": sum(approx_tokens(m.text) for m in live),
        "depths": dict(sorted(Counter(m.depth for m in live).items())),
        "max_depth": max((m.depth for m in live), default=0),
        "retired": dict(sorted(retired.items())),
        "total_recalls": sum(m.times_recalled for m in store.memories.values()),
    }


def save_summary(store, snapshot_path: str | Path) -> Path:
    """Write store_summary next to a snapshot: <snapshot>.summary.json (atomic)."""
    snapshot_path = Path(snapshot_path)
    out = snapshot_path.with_name(snapshot_path.stem + ".summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(store_summary(store), sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return out


def latest_summaries(results: str | Path) -> list[tuple[Path, dict]]:
    """The newest *.summary.json in each folder under results/ (one per persona and arm, as runs write them)."""
    newest: dict[Path, Path] = {}
    for p in Path(results).rglob("*.summary.json"):
        if p.parent not in newest or p.stat().st_mtime > newest[p.parent].stat().st_mtime:
            newest[p.parent] = p
    out = []
    for p in sorted(newest.values()):
        try:
            out.append((p, json.loads(p.read_text(encoding="utf-8"))))
        except (OSError, json.JSONDecodeError):
            continue
    return out


# -- experiment overhead on disk ---------------------------------------------------------------


def dir_size(path: str | Path) -> tuple[int, int]:
    """(bytes, files) under a folder; (0, 0) if it doesn't exist."""
    total = files = 0
    stack = [Path(path)]
    while stack:
        try:
            with os.scandir(stack.pop()) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    elif e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
                        files += 1
        except OSError:
            continue
    return total, files


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
