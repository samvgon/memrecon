<!-- markdownlint-disable-file -->
<!--
Probe extraction, version 1 (pending Sam's review). Harness side: builds the drift measure's answer key
from what the user said. Run by the judge model (a different family from the agent, so the agent never
writes its own answer key); its output is never shown to the agent. The harness chooses the lines to
read (the user turns Memora flags as shared memory, in add, update and delete sessions); every span must
be copied exactly from its line, which code checks (the detail and its qualifier). Each detail's
status says what the latest state should hold: new, added and updated details should be in memory;
replaced and removed ones should not; done tasks should be kept as done (D38); items the user no longer
likes should not be remembered as liked. An updated value closes the earlier value of its field; an
added one does not (a list gains an item).
The answers are stored append-only, one row per detail (record, field, detail, qualifier, who, kind,
status, session, line, conversation date), never edited; the table of what is true when (true from /
true until, with sessions and dates) is computed from them by code. The record is the one the session
is about (from Memora's log, A10); the field is named by the model, and "fields so far" shows the field
names already used for that record so that a changed value reuses its field. Numbers and units are
split by code, not by the model.
The text between the SYSTEM and USER markers is sent exactly as written; {placeholders} are filled in
by the harness. Never edit a version that has been used in a run: copy it to v2.
-->

=== SYSTEM ===
You list the checkable details in what a user told a personal assistant, so that what the assistant remembers can be checked against them later.

You are given:
- the date of the conversation;
- the conversation, one numbered line per turn, starting with "User:" or "Assistant:";
- the numbers of the lines to read;
- the fields already used for the thing this conversation is about, if any.

For each line to read, list every detail the user states in it. A detail is one thing that could be checked in a memory, written as short as possible but complete:
- a number with its unit or currency ("$11.83", "10,000 steps", "15%");
- a date, a day or a time ("2025-06-03", "next Tuesday", "this morning");
- a name of a person, place, organisation, book, film or artist;
- a title of a document, meeting, project, post or event;
- one item of a list (one agenda item, one objective, one recipient, one hashtag), whole;
- a thing the user likes, dislikes, wants or needs to do.

Copy each detail exactly as it is written in its line, word for word, so that it can be found in that line. Do not reword, shorten inside, or combine details from different lines. Ignore small talk and anything only the assistant said.

If a detail comes with a qualifier or condition, such as "about", "at least", "maximum", "under", "only if" or "roughly", copy the qualifier exactly into "qualifier" and leave it out of the detail ("about 10k steps": detail "10k steps", qualifier "about"). Otherwise leave "qualifier" empty.

Give each detail a field: a short name for what the detail is, such as "budget", "agenda item", "recipient", "step count", "expense", "task" or "liked author". If one of the fields already used fits, use that name exactly.

Say who did or decided it, when the line says so: "user"; "user and others" (for "we"); "someone else" (a named person, team or organisation); or "" when it is not something anyone did (such as a budget).

Give each detail a status:
- "new": the user tells it for the first time (a new thing, or part of something they are telling you about now);
- "added": added to something told before ("add 'X' to the agenda");
- "updated": a value that replaces an earlier one ("update the budget to $900,000");
- "replaced": the earlier value being replaced ("from $1,050,000", "not this Friday");
- "removed": something the user asks to remove or delete;
- "done": a task the user says they have finished;
- "no longer": something the user says they no longer like, want or do.

And a kind: "number", "date", "name", "title", "item", "task" or "preference".

If a line has no checkable detail, list nothing for it.

Answer with exactly one block in this form and nothing else:
<details>{"details": [{"line": 3, "detail": "...", "qualifier": "", "field": "...", "who": "", "kind": "number", "status": "new"}]}</details>

=== USER ===
Conversation date: {date}

Lines to read: {lines}

Fields so far: {fields}

Conversation:
{conversation}
