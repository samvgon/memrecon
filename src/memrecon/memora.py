"""Reading Memora (read-only). The same layout and order as scripts/explore/common.py (A6: file-name
order is time order), for runners that live outside the exploration scripts."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WEEKLY = ROOT / "data" / "memora" / "data" / "weekly"


def personas() -> list[str]:
    return sorted(p.name for p in WEEKLY.iterdir() if p.is_dir())


def sessions(persona: str) -> list[dict]:
    """A persona's sessions in file-name order (chronological, A6)."""
    out = []
    for path in sorted((WEEKLY / persona / "conversations").glob("session_*.json")):
        with open(path, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def dev_personas() -> list[str]:
    """The development personas of the frozen split (notes/split.json)."""
    return json.loads((ROOT / "notes" / "split.json").read_text())["dev"]
