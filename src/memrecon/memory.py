"""The memory store (D25, D27–D30).

Operations (D28):

| operation | written from         | stored                        | retrieval counter      | depth                    | the memory it came from |
|-----------|----------------------|-------------------------------|------------------------|--------------------------|-------------------------|
| log       | user's words         | the user's words              | new: 0                 | 0                        | n/a                     |
| note      | user's words         | model note                    | new: 0                 | 1                        | n/a                     |
| discard   | a memory             | nothing                       | n/a                    | n/a                      | counter +1 (`recall`)   |
| keep      | a memory             | the rewrite, as a new memory  | new: 0                 | that memory's depth + 1  | counter +1, unchanged   |
| replace   | a memory             | the rewrite, in its place     | same memory: +1        | that memory's depth + 1  | old text kept only in its version history |

Every recall-time policy starts with `recall` (counter +1); discard stops
there, keep then calls `keep`, replace calls `replace`. Write policies are not
run on logged memories (D24).

A user's change to a fact is a `note` that **supersedes** the memories it was
written from (D29): the new note is written from those memories' text plus the
user's new words, so its depth is the deepest of them + 1, and they are
retired from retrieval but kept for scoring (like Zep/Graphiti, which
invalidate old facts rather than delete them). A user's request to forget is
the same with nothing new: `forget` retires the memories. Memories retrieval
did not find are untouched, so stale copies can survive, as in real systems.

**Depth** (D27, D33) counts generations of rewriting between a memory's text
and what the user said: log 0, note 1, anything written from other memories
the deepest of them + 1. It never resets, and it is the primary depth for the
analysis (pre-registered, D33). **`since_user`** is the second view: the same
count, except that a user's change restarts it as though the note were new
(1; 0 for the log). A third counter, generations since the last check against
the source, is added with Stage 2's verify policy. How far a text has
actually drifted is measured (probes, embedding distance), never counted.

Lineage, kept for the harness and **never shown to the model** (D28):
`written_from` ("user", "memory" or "both"), `parents` (memory IDs it was
written from), `sources` (every user turn in its ancestry), `superseded_by`,
`retired` (None, "superseded" or "forgotten"), `versions`, `times_recalled`,
`times_selected` (D54).
The model sees memories only through `present()`: their text and the date of
the conversation their information came from (D36: relative dates count from
it), under labels numbered afresh for each call and mapped back by the
harness. A rewrite keeps its parent's date; a user's change takes the date of
the conversation it was made in.

Snapshots are canonical JSON (sorted keys, no clock times, IDs assigned in
insertion order), so two runs that did the same things give byte-identical
snapshots. The Stage 0 check (D19, D26) compares `content_json()`, which
leaves out `times_recalled`, `times_selected` and the event count: an ingest-only run never
recalls, while everything retrieval and the write policies can see must
match exactly.

**Event file** (part of the audit log, D30): given an `EventLog`, the store
appends one line per operation, in order, as it happens (log, note, forget,
recall, select, keep, replace, present), with what it touched and, for writes, the
text and depth. Snapshots show the state after each session; the event file
shows how it got there. Lines are flushed and synced as they are written; a
line cut off by a crash is skipped when reading. Event numbers continue
across a resumed run because the count is saved in the snapshot.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SNAPSHOT_FORMAT = 6
MODES = ("note", "log")


@dataclass(frozen=True)
class Source:
    """A user turn a memory came from."""

    persona: str
    session_id: int
    turn: int


@dataclass(frozen=True)
class At:
    """A logical position in the replay: the session and turn where something happened."""

    session_id: int
    turn: int


@dataclass
class Version:
    text: str
    depth: int
    at: At
    since_user: int = 0


@dataclass
class Memory:
    id: str
    text: str
    depth: int
    since_user: int
    written_from: str  # "user", "memory" or "both"
    parents: list[str]
    sources: list[Source]
    versions: list[Version]
    created_at: At
    date: str = ""  # date of the conversation its information came from (D36); shown to the model
    times_recalled: int = 0  # retrieval found it for a turn (D25)
    times_selected: int = 0  # the agent chose it as useful for the turn, so it was rewritten (D54)
    superseded_by: str | None = None
    retired: str | None = None  # None, "superseded" or "forgotten"
    retired_at: At | None = None


class UnknownMemory(KeyError):
    pass


class EventLog:
    """Append-only JSONL file of store operations (part of the audit log; never shown to the model)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # After a crash the last line may be cut off; start the next event on a fresh line.
        if self.path.exists() and self.path.stat().st_size > 0:
            with open(self.path, "rb+") as f:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    f.write(b"\n")

    def append(self, event: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_events(path: str | Path) -> list[dict]:
    """Events in order; a final line cut off by a crash is skipped."""
    p = Path(path)
    if not p.exists():
        return []
    events = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return events


@dataclass
class MemoryStore:
    persona: str
    mode: str = "note"  # "note": the model writes notes; "log": memories are the user's own words
    memories: dict[str, Memory] = field(default_factory=dict)
    _next: int = 1
    events: EventLog | None = field(default=None, repr=False, compare=False)
    _event_count: int = 0

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")

    # -- ingestion: what the user did --------------------------------------------

    def log(self, text: str, sources: list[Source], at: At, supersedes: list[str] = (), date: str = "") -> str:
        """Store the user's own words (log mode). Depth 0. May supersede earlier log entries."""
        self._require_mode("log", "log")
        mid = self._new(text, 0, 0, sources, at, supersedes, uses_parent_text=False, date=date)
        self._emit("log", at, id=mid, text=text, depth=0, since_user=0, date=date, supersedes=list(supersedes), sources=[asdict(x) for x in sources])
        return mid

    def note(self, text: str, sources: list[Source], at: At, supersedes: list[str] = (), date: str = "") -> str:
        """Store a model note on the user's words (note mode).

        With `supersedes`, the note records a user's change: it was written from
        those memories plus the user's new words, so its depth is the deepest of
        them + 1, and they are retired as superseded. Without, depth 1.
        Either way `since_user` is 1: the user just supplied input.
        """
        self._require_mode("note", "note")
        parents = [self._get(p) for p in supersedes]
        depth = max(p.depth for p in parents) + 1 if parents else 1
        mid = self._new(text, depth, 1, sources, at, supersedes, uses_parent_text=True, date=date)
        self._emit("note", at, id=mid, text=text, depth=depth, since_user=1, date=date, supersedes=list(supersedes), sources=[asdict(x) for x in sources])
        return mid

    def forget(self, ids: list[str], at: At) -> None:
        """The user asked to forget: retire these memories (superseded by nothing). Kept for scoring."""
        for mid in ids:
            self._retire(self._get(mid), "forgotten", at)
        self._emit("forget", at, ids=list(ids))

    # -- recall-time write policies ------------------------------------------------

    def recall(self, mid: str, at: At | None = None) -> str:
        """Retrieval handed this memory to the agent (counter +1). Discard stops here."""
        m = self._live(mid)
        m.times_recalled += 1
        self._emit("recall", at, id=mid, times_recalled=m.times_recalled)
        return m.text

    def select(self, mid: str, at: At | None = None) -> None:
        """The agent chose this recalled memory as useful for the turn (counter +1), so it is rewritten (D54)."""
        m = self._live(mid)
        m.times_selected += 1
        self._emit("select", at, id=mid, times_selected=m.times_selected)

    def keep(self, mid: str, rewritten: str, at: At) -> str:
        """Keep the rewrite as a new memory beside the one it came from (depth + 1)."""
        self._no_policies_on_log()
        p = self._live(mid)
        new = self._new_id()
        self.memories[new] = Memory(
            id=new,
            text=rewritten,
            depth=p.depth + 1,
            since_user=p.since_user + 1,
            written_from="memory",
            parents=[mid],
            sources=list(p.sources),
            versions=[Version(rewritten, p.depth + 1, at, p.since_user + 1)],
            created_at=at,
            date=p.date,  # a rewrite carries the date of the information it came from
        )
        self._emit("keep", at, id=new, parent=mid, text=rewritten, depth=p.depth + 1, since_user=p.since_user + 1)
        return new

    def replace(self, mid: str, rewritten: str, at: At) -> None:
        """The rewrite takes the memory's place (depth + 1); the old text stays only in its version history."""
        self._no_policies_on_log()
        m = self._live(mid)
        m.text = rewritten
        m.depth += 1
        m.since_user += 1
        m.versions.append(Version(rewritten, m.depth, at, m.since_user))
        self._emit("replace", at, id=mid, text=rewritten, depth=m.depth, since_user=m.since_user)

    # -- what the model may see ---------------------------------------------------

    def present(self, ids: list[str], at: At | None = None) -> tuple[str, dict[str, str]]:
        """The model's view of some memories: numbered text, with the date its information came from; no IDs or lineage.

        Returns the text to put in a prompt and a map from the labels shown
        ("1", "2", ...) back to memory IDs, so a model's answer ("memory 2 is
        out of date") can be turned into an action on the right memory.
        """
        labels: dict[str, str] = {}
        lines = []
        for k, mid in enumerate(ids, start=1):
            m = self._live(mid)
            labels[str(k)] = mid
            lines.append(f"[{k}] ({m.date}) {m.text}" if m.date else f"[{k}] {m.text}")
        self._emit("present", at, labels=labels)
        return "\n".join(lines), labels

    # -- reading (harness only) -----------------------------------------------------

    def active(self) -> list[Memory]:
        """Memories retrieval may return (not retired), in ID order."""
        return [m for m in self.memories.values() if m.retired is None]

    def get(self, mid: str) -> Memory:
        return self._get(mid)

    def children(self, mid: str) -> list[Memory]:
        return [m for m in self.memories.values() if mid in m.parents]

    def __len__(self) -> int:
        return len(self.memories)

    # -- snapshots ---------------------------------------------------------------

    def to_json(self) -> str:
        """Canonical JSON: the same store state always gives the same bytes."""
        return self._dump(include_recall_counts=True)

    def content_json(self) -> str:
        """Canonical JSON without the recall counters: what the Stage 0 check compares (D19, D26)."""
        return self._dump(include_recall_counts=False)

    def _dump(self, include_recall_counts: bool) -> str:
        memories = []
        for m in self.memories.values():
            d = asdict(m)
            if not include_recall_counts:
                del d["times_recalled"]
                del d["times_selected"]
            memories.append(d)
        data = {
            "format": SNAPSHOT_FORMAT,
            "persona": self.persona,
            "mode": self.mode,
            "next_id": self._next,
            "memories": memories,
        }
        if include_recall_counts:
            data["event_count"] = self._event_count
        return json.dumps(data, sort_keys=True, ensure_ascii=False, indent=1) + "\n"

    @classmethod
    def from_json(cls, text: str, events: EventLog | None = None) -> MemoryStore:
        data = json.loads(text)
        if data.get("format") != SNAPSHOT_FORMAT:
            raise ValueError(f"unknown snapshot format {data.get('format')!r}")
        store = cls(
            persona=data["persona"],
            mode=data["mode"],
            _next=data["next_id"],
            events=events,
            _event_count=data.get("event_count", 0),
        )
        for d in data["memories"]:
            d["sources"] = [Source(**s) for s in d["sources"]]
            d["versions"] = [Version(v["text"], v["depth"], At(**v["at"]), v["since_user"]) for v in d["versions"]]
            d["created_at"] = At(**d["created_at"])
            d["retired_at"] = At(**d["retired_at"]) if d["retired_at"] else None
            store.memories[d["id"]] = Memory(**d)
        return store

    def save(self, path: str | Path) -> None:
        """Write a snapshot atomically (temp file, fsync, rename)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(self.to_json())
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str | Path, events: EventLog | None = None) -> MemoryStore:
        """Load a snapshot; pass the run's EventLog to keep appending to it when resuming."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"), events=events)

    # -- internals -------------------------------------------------------------

    def _new(self, text, depth, since_user, sources, at, supersedes, uses_parent_text, date="") -> str:
        parents = [self._live(p) for p in supersedes]
        mid = self._new_id()
        inherited = [s for p in parents for s in p.sources]
        all_sources = list(dict.fromkeys(inherited + list(sources)))  # keep order, drop repeats
        self.memories[mid] = Memory(
            id=mid,
            text=text,
            depth=depth,
            since_user=since_user,
            written_from="both" if (parents and uses_parent_text) else "user",
            parents=[p.id for p in parents],
            sources=all_sources,
            versions=[Version(text, depth, at, since_user)],
            created_at=at,
            date=date,
        )
        for p in parents:
            self._retire(p, "superseded", at, by=mid)
        return mid

    def _emit(self, op: str, at: At | None, **details) -> None:
        self._event_count += 1
        if self.events is not None:
            self.events.append({"n": self._event_count, "op": op, "at": asdict(at) if at else None, **details})

    def _retire(self, m: Memory, how: str, at: At, by: str | None = None) -> None:
        if m.retired is not None:
            raise ValueError(f"{m.id} is already {m.retired}")
        m.retired = how
        m.retired_at = at
        m.superseded_by = by

    def _require_mode(self, mode: str, op: str) -> None:
        if self.mode != mode:
            raise ValueError(f"`{op}` is for {mode}-mode stores; this store is in {self.mode} mode")

    def _no_policies_on_log(self) -> None:
        if self.mode == "log":
            raise ValueError("write policies are not run on logged memories (D24)")

    def _new_id(self) -> str:
        mid = f"m{self._next:05d}"
        self._next += 1
        return mid

    def _get(self, mid: str) -> Memory:
        try:
            return self.memories[mid]
        except KeyError:
            raise UnknownMemory(mid) from None

    def _live(self, mid: str) -> Memory:
        m = self._get(mid)
        if m.retired is not None:
            raise ValueError(f"{mid} is {m.retired}; retired memories can't be used")
        return m


def first_difference(a: str, b: str, context: int = 60) -> str | None:
    """Where two snapshots first differ (for the Stage 0 check's error message), or None if identical."""
    if a == b:
        return None
    i = next((k for k, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    line = a.count("\n", 0, i) + 1
    return f"first difference at line {line}: {a[max(0, i - context):i + context]!r} vs {b[max(0, i - context):i + context]!r}"
