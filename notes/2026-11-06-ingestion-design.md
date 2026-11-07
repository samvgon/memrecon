# Ingestion: how memories get written (draft for Sam's review)

*Written by Claude, Nov 6, 2026 · Proposed by Both · Review: pending. Logged as D35; Sam's answers to the questions below are D36 (one memory per item; notes convert dates, the log keeps the wording; log arm = the kept words verbatim, checked as an exact part of the turn; an oracle arm of flagged turns as a labelled reference; tagged JSON with 3 attempts instead of the server's JSON-schema option). Nothing here has been run.*

The prompt itself is `src/memrecon/prompts/ingest_note_v1.md`. These are the choices around it.

## Settled by the draft (change any of them)

1. **When.** Once per session, after it ends. A session's memories become retrievable from the next session on; recall during a session (D14) uses memories from earlier sessions. One ingestion call per session also fits the unit of work.
2. **What the model reads.** The whole session, user and assistant lines, with the session date. **Not the `share_memory` flags:** they are Memora's annotation of which turns matter, and showing them, or using them to choose what to store, would tell the agent what the evaluation checks. Flags stay on the harness side, for building probes.
3. **Which existing memories it sees, for changes and forgets.** Retrieval with each user turn of the session as a query; the union of the top 3 per turn, the 10 best overall, shown as numbered text only (`present()`, D28). The relevance cut for this step is set with the main threshold (E16).
4. **Output.** JSON in a fixed form (new memories; changes with the memories they replace; forgets), enforced by the server's JSON-schema option so the answer always parses. The harness maps the numbers back to memory IDs and applies `note(..., supersedes=...)` and `forget()` (D29).
5. **Settings.** As every call: thinking off, temperature 0, seed 0. Output limit 2,048 tokens (a cut-off answer is an error, D23).
6. **Versioning.** The prompt file is never edited once used in a run; a change is a new version (`v2`). Each saved call records the full prompt anyway.

## Questions for Sam

1. **One memory per item, or one per detail?** The draft writes one memory per thing the user told about (a project proposal with its budget, timeline and deliverables in one memory). The alternative, one memory per detail ("Project X budget: €100,000"), is how some systems (Mem0) store facts: retrieval is more precise and a rewrite touches less, but a project becomes a dozen memories. One per item is closer to Memora's own records, and long memories are where rewriting is most likely to lose detail, which is what the project measures.
2. **Relative dates.** The draft asks the model to add the calendar date ("next Tuesday" also becomes "2025-06-10"). That adds text the user didn't say, but it is derived from the conversation date, and without it "next Tuesday" is meaningless a week later. Keep, or store only what was said?
3. **The log arm (D24 needs revisiting).** D24 says a log memory is "exactly what the user said (the flagged turn)". The agent can't know which turns are flagged (point 2 above), so the log arm needs another rule:
   - **(a) every user turn, verbatim:** a true log. Retrieval then searches all the small talk too (E15 suggests chatter already causes false matches).
   - **(b) the user turns the ingestion model chooses to keep, verbatim:** the same prompt picks what is worth remembering, but returns turn numbers instead of rewriting. Selection is then identical to the note arm's, so "log vs note + discard" isolates what *writing* a note does. Depth stays 0.
   - **(c) keep the flagged turns:** an oracle upper bound, labelled as such, not a realistic arm.

   I'd suggest (b) as the log arm, possibly with (a) as an extra reference if time allows. Whatever is chosen gets a new log entry, since D24 is reviewed.
