"""Shared helpers for the data-exploration scripts.

These scripts only read the dataset; they never modify it.
Paths are computed from this file's location, so they work from any directory.
"""
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MEMORA = ROOT / "data" / "memora"
WEEKLY = MEMORA / "data" / "weekly"


def personas():
    return sorted(p.name for p in WEEKLY.iterdir() if p.is_dir())


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def iter_sessions(persona=None):
    """Yield (persona, path, session_dict) in file-name order (the README says this is chronological)."""
    for name in ([persona] if persona else personas()):
        for path in sorted((WEEKLY / name / "conversations").glob("session_*.json")):
            yield name, path, load(path)


def iter_question_files():
    """Yield (persona, path, question_file_dict) for every evaluation question file."""
    for name in personas():
        for path in sorted((WEEKLY / name).glob("evaluation_questions_*.json")):
            yield name, path, load(path)


def flagged_turns(session):
    """Turns marked share_memory: True."""
    return [t for t in session.get("conversation", []) if t.get("share_memory")]


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=True).stdout.strip()


def code_version():
    """Short commit of this repo, plus a flag if the scripts folder has uncommitted changes."""
    try:
        rev = _git(ROOT, "rev-parse", "--short", "HEAD")
        dirty = _git(ROOT, "status", "--porcelain", "--", "scripts")
        return rev + (" (scripts have uncommitted changes)" if dirty else " (scripts clean)")
    except Exception:
        return "unknown"


def data_version():
    """Short commit of the Memora clone."""
    try:
        return _git(MEMORA, "rev-parse", "--short", "HEAD")
    except Exception:
        return "unknown"
