"""E13: is file order the same as date order, and are session and turn IDs unique?

Question (assumption A6): ingestion and probe extraction apply Memora's
add/update/delete operations in file order. That is only right if file order
matches the sessions' dates, and if each session (and each turn in it) can be
identified unambiguously, since memories will point back to their source turns.
Method: for each weekly persona, read the sessions in file-name order (as
iter_sessions does) and check:
  - the file number (session_0007.json -> 7) equals the session's `session_id`,
    and session_ids are unique and consecutive from 1;
  - the `persona` field matches the folder;
  - every `date` parses as YYYY-MM-DD and dates never go backwards in file order
    (days with several sessions are counted: their order within the day can
    only come from the file order);
  - within each session, `turn` numbers are unique and run 1, 2, 3, ...
Limits: checks order and identity only; it does not check that an update or
delete comes after the add of the same record (that needs the record keys).
"""
import collections
import datetime
import re

from common import iter_sessions, personas

problems = collections.Counter()
examples = collections.defaultdict(list)


def note(kind, text):
    problems[kind] += 1
    if len(examples[kind]) < 5:
        examples[kind].append(text)


print(f"{'persona':<24} {'sessions':>8} {'first date':>11} {'last date':>11} {'shared-date days':>17} {'max per day':>12}")
for persona in personas():
    ids, dates = [], []
    for _, path, s in iter_sessions(persona):
        where = f"{persona}/{path.name}"
        m = re.fullmatch(r"session_(\d+)\.json", path.name)
        file_no = int(m.group(1)) if m else None
        sid = s.get("session_id")
        ids.append(sid)
        if file_no != sid:
            note("file number != session_id", f"{where}: session_id={sid!r}")
        if s.get("persona") != persona:
            note("persona field != folder", f"{where}: persona={s.get('persona')!r}")
        try:
            d = datetime.date.fromisoformat(s.get("date", ""))
        except (TypeError, ValueError):
            note("date not YYYY-MM-DD", f"{where}: date={s.get('date')!r}")
            d = None
        if d and dates and dates[-1] and d < dates[-1]:
            note("date goes backwards in file order", f"{where}: {dates[-1]} -> {d}")
        dates.append(d)
        turns = [t.get("turn") for t in s.get("conversation", [])]
        if turns != list(range(1, len(turns) + 1)):
            dup = len(turns) - len(set(turns))
            note("turn numbers not 1..n", f"{where}: {len(turns)} turns, {dup} duplicates, first {turns[:6]}")

    if len(set(ids)) != len(ids):
        note("duplicate session_id", f"{persona}: {len(ids) - len(set(ids))} duplicates")
    clean_ids = [i for i in ids if isinstance(i, int)]
    if sorted(clean_ids) != list(range(1, len(ids) + 1)):
        note("session_ids not 1..n", f"{persona}: min {min(clean_ids, default=None)}, max {max(clean_ids, default=None)}, n {len(ids)}")
    per_day = collections.Counter(d for d in dates if d)
    shared = sum(1 for c in per_day.values() if c > 1)
    valid = [d for d in dates if d]
    print(f"{persona:<24} {len(ids):>8} {str(min(valid)) if valid else '-':>11} {str(max(valid)) if valid else '-':>11} "
          f"{shared:>17} {max(per_day.values(), default=0):>12}")

print()
if not problems:
    print("No problems found: file order matches date order, and session and turn IDs are unique and consecutive.")
else:
    print("Problems found:")
    for kind, n in problems.most_common():
        print(f"  {n:5d}  {kind}")
        for e in examples[kind]:
            print(f"         e.g. {e}")
