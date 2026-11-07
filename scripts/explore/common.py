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


# -- shared by E16 and later scripts (E14 and E15 keep their own copies, unchanged) --------------

import re as _re

_UNIT = (
    r"(?:%|k\b|K\b|M\b|million|billion|thousand|steps?|days?|weeks?|months?|years?|hours?|minutes?|mins?"
    r"|languages?|locales?|people|participants|users|KB|MB|GB|TB|km|miles?|kg|lbs?|calories|pages?|words?"
    r"|slides?|screens?|countries|regions?|phases?|am|pm|AM|PM)"
)
_PATTERNS = {
    "number": _re.compile(r"(?<![\w#])(?:[$€£¥]\s?)?\d+(?:[.,]\d+)*\+?(?:\s?-?\s?" + _UNIT + r")?"),
    "quoted": _re.compile(r"[\"“*]([^\"”*]{3,80})[\"”*]|(?<=[\s:])'([^']{3,120})'(?=[\s.,;!?]|$)"),
    "hashtag": _re.compile(r"#\w+"),
    "qualifier": _re.compile(
        r"\b(?:maximum|minimum|at least|at most|up to|no more than|no later than|exactly|only"
        r"|(?:high|medium|low)(?:-|\s)(?:risk|priority))\b",
        _re.I,
    ),
}
_MULTI_NAME = _re.compile(r"\b[A-Z][\w&'’.-]*(?:\s+(?:of|the|for|de|on|in|&)?\s*[A-Z][\w&'’.-]*)+")
_SINGLE_NAME = _re.compile(r"(?<=[a-z,;:]\s)[A-Z][\w&'’.-]{2,}")
_NOT_NAMES = {"I", "I'm", "I've", "I'd", "I'll", "I’m", "I’ve", "Oh", "OK", "Okay"}


def _bare(x):
    return _re.sub(r"[\"“”*'.,!?:;]", "", x).strip().lower()


def probe_elements(text, record_value=None):
    """Code-checkable candidate elements in a turn, as defined in E14: [(kind, text), ...]."""
    found = []
    for kind, pat in _PATTERNS.items():
        for m in pat.finditer(text):
            found.append((kind, m.group(0).strip()))
    quoted = {_bare(v) for k, v in found if k == "quoted"}
    names = set()
    for m in _MULTI_NAME.finditer(text):
        names.add(m.group(0).rstrip(".,;:!?"))
    for m in _SINGLE_NAME.finditer(text):
        w = m.group(0).rstrip(".,;:!?")
        if w not in _NOT_NAMES and not any(w in n for n in names):
            names.add(w)
    found += [("name", n) for n in sorted(names) if _bare(n) not in quoted and n not in _NOT_NAMES]
    if record_value and record_value.strip().lower() in text.lower():
        found.append(("record value", record_value.strip()))
    return found


def record_key(s):
    """Identity of the record an add/update/delete session is about (assumption A10, as in E12, E14, E15)."""
    d = s.get("operation_details") or {}
    t = s.get("session_type")
    if t == "preference":
        item = d.get("item")
        return ("preference", item.strip().lower()) if isinstance(item, str) else None
    if t != "activity":
        return None
    cat, it = d.get("category"), d.get("item")
    if cat is None and isinstance(it, str):
        return ("content", it)
    if not isinstance(it, dict):
        return None
    if cat == "todo_list" and it.get("description"):
        return (cat, it["description"].strip().lower())
    if cat == "calendar_event" and it.get("event_name"):
        return (cat, it["event_name"].strip().lower())
    if cat == "food_expenses" and it.get("amount") is not None:
        return (cat, f"{float(it['amount']):.2f}")
    if cat == "step_tracker" and it.get("step_count") is not None:
        return (cat, str(int(it["step_count"])))
    return None


def record_value(s):
    """The log's own key string for a record (preference item, todo description, event name), or None."""
    d = s.get("operation_details") or {}
    it = d.get("item")
    if s.get("session_type") == "preference" and isinstance(it, str):
        return it
    if isinstance(it, dict):
        v = it.get("description") or it.get("event_name")
        return v if isinstance(v, str) else None
    return None
