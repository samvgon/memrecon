"""E12: after a fact is introduced, how often does it come up again in later conversations?

Question: if the agent recalls memories while the sessions are replayed (retrieval on each user turn),
how many natural recall opportunities does a fact get? This sets how deep rewrites can get without us
injecting extra queries.
Method: for every fact introduced in an `add` session (preferences and activities; goals are skipped because
their values are bare numbers), build match strings from the log: preference -> the item name; todo -> the
description; calendar event -> the event name; food expense -> the amount ("3.66" or "$3.66"); step
tracker -> the step count ("6200" or "6,200"); content records (project proposal, meeting notes, email) ->
the first of project_title / meeting_title / email_subject / email_purpose found in `content_data`.
Then count the LATER sessions (by file order; assumption A6, order untested) whose USER turns contain a match
string (case-insensitive), skipping sessions that update or delete that same fact (they mention it by design).
This is a strict string match, so paraphrases are missed: every count is a LOWER BOUND on natural recalls.
It measures only whether the topic comes up again; it does not run any retrieval.
"""
import statistics as st
from collections import Counter, defaultdict

from common import iter_sessions, personas

CONTENT_KEYS = ["project_title", "meeting_title", "email_subject", "email_purpose"]


def facts_in(s):
    """Return (group, [match strings]) for an add/update/delete session, or None if not measurable."""
    d = s.get("operation_details") or {}
    t = s.get("session_type")
    if t == "preference":
        item = d.get("item")
        return ("preference", [item]) if isinstance(item, str) and len(item) >= 3 else None
    if t != "activity":
        return None
    cat, it = d.get("category"), d.get("item")
    if cat == "todo_list" and isinstance(it, dict):
        return ("activity/todo_list", [it.get("description")])
    if cat == "calendar_event" and isinstance(it, dict):
        return ("activity/calendar_event", [it.get("event_name")])
    if cat == "food_expenses" and isinstance(it, dict) and it.get("amount") is not None:
        a = f"{float(it['amount']):.2f}"
        return ("activity/food_expenses", [a])
    if cat == "step_tracker" and isinstance(it, dict) and it.get("step_count") is not None:
        n = int(it["step_count"])
        return ("activity/step_tracker", [str(n), f"{n:,}"])
    if cat is None and isinstance(it, str):
        cd = d.get("content_data") or {}
        for k in CONTENT_KEYS:
            if isinstance(cd.get(k), str) and cd[k].strip():
                return ("activity/content_record", [cd[k]])
    return None


def canon(strings):
    return (strings[0] or "").strip().lower()


by_persona = defaultdict(list)
for persona, path, s in iter_sessions():
    by_persona[persona].append(s)

groups = defaultdict(list)  # group -> list of (later_mention_count, mentions_in_no_memory, changed_later)
unmeasured = Counter()
for persona in personas():
    sessions = by_persona[persona]
    user_text = []
    for s in sessions:
        user_text.append(" ".join(t["message"] for t in s["conversation"] if t.get("speaker") == "user_agent").lower())
    fact_keys = [facts_in(s) if s.get("operation") in ("add", "update", "delete") else None for s in sessions]
    for i, s in enumerate(sessions):
        if s.get("operation") != "add":
            continue
        if s.get("session_type") == "goal":
            unmeasured["goal (bare numbers)"] += 1
            continue
        fk = fact_keys[i]
        if fk is None:
            unmeasured[f"{s.get('session_type')} (no match string)"] += 1
            continue
        group, strings = fk
        strings = [x.strip().lower() for x in strings if isinstance(x, str) and x.strip()]
        if not strings:
            unmeasured[f"{group} (empty match string)"] += 1
            continue
        me = canon(strings)
        count = nomem = 0
        changed = False
        for j in range(i + 1, len(sessions)):
            other = fact_keys[j]
            if other and canon([x for x in other[1] if isinstance(x, str)] or [""]) == me:
                changed = True
                continue  # an update or delete of this same fact
            if any(x in user_text[j] for x in strings):
                count += 1
                if sessions[j].get("session_type") == "no_memory":
                    nomem += 1
        groups[group].append((count, nomem, changed))

print(f"{'group':26} {'facts':>6} {'>=1 later':>10} {'0':>5} {'1':>5} {'2':>5} {'3+':>5} {'mean':>6} {'max':>4} {'in filler':>9} {'later changed':>14}")
allrows = []
for g in sorted(groups):
    rows = groups[g]
    allrows += rows
    c = [r[0] for r in rows]
    n = len(rows)
    ge1 = sum(1 for x in c if x >= 1)
    total_m = sum(c)
    filler = sum(r[1] for r in rows)
    print(f"{g:26} {n:6d} {ge1:5d} ({100 * ge1 // n:2d}%) {sum(1 for x in c if x == 0):5d} {sum(1 for x in c if x == 1):5d} {sum(1 for x in c if x == 2):5d} {sum(1 for x in c if x >= 3):5d} {st.mean(c):6.2f} {max(c):4d} {filler:9d} {sum(1 for r in rows if r[2]):14d}")
n = len(allrows)
c = [r[0] for r in allrows]
print()
print(f"all measured facts: {n}; with at least one later mention: {sum(1 for x in c if x >= 1)} ({100 * sum(1 for x in c if x >= 1) // n}%); mean later mentions {st.mean(c):.2f}")
print("('in filler' = how many of those mentions were in no_memory sessions; 'later changed' = facts later updated or deleted)")
print("not measured:", dict(unmeasured))
print()
print("Strict string matching misses paraphrases, so these are lower bounds on how often a fact comes up again.")
