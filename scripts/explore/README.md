# Data exploration scripts

Every finding about the data that a decision relies on must come from a script in this folder, not from an ad hoc command typed into a terminal. That way each finding can be re-run, and if we later find we misread a field, we can see exactly which code and which assumption produced the decision.

## Rules

1. **One question per script**, named `E<number>_<short_name>.py` (E01, E02, ...). The docstring states the question, the method and the limits.
2. **Run through the runner**, from the repo root: `uv run python scripts/explore/run.py E01`. It saves `notes/evidence/E01.txt` with a header recording the date, the memrecon commit, the Memora commit and the exit status, followed by the output.
3. **Commit the script and its evidence file together**, before citing it in a decision. The header's code version is only meaningful if the scripts are committed (it says if they are not).
4. **Never edit a script after a decision cites it.** If the question or method changes, write a new script with the next number and say in the log which one supersedes the old one.
5. **Log it** in the decision log: an entry in the evidence register (what it answers, what it showed, which assumptions it uses) and, for every interpretation of a field that it relies on, an entry in the assumptions register. Decisions cite evidence as `E03` and assumptions as `A2`.
6. **Scripts only read the dataset.** Nothing here writes to `data/`.

## Current scripts

| ID | Question |
|---|---|
| E01 | What do `session_type` and `operation` contain, and how many of each? |
| E02 | How many remembered and forgotten items do Memora's evaluation questions list? Does the counter miss differently shaped evidence? |
| E03 | For one project's add, delete and update sessions, what does the dialogue say compared with the log? |
| E04 | How many turns are flagged `share_memory: True`, and how long are they? |
| E05 | How often do flagged turns contain hedging language (loose and tight searches)? |
| E06 | How much of what the log records for adds and updates is spoken in the dialogue, and how much of it is in flagged turns? |
| E07 | Which flagged turns are facts to remember (add, update sessions) and which are instructions to forget (delete sessions)? |
| E08 | How big is Memora's scored rubric (the `evaluation` yes/no sub-questions)? Supersedes E02 for sizing. |
| E09 | How long is each persona's full history (turns, characters, estimated tokens)? |
| E10 | LoCoMo (`data/locomo/locomo10.json`): conversation lengths, question categories, evidence references. Prints the file's sha256 (the data is not a git clone). |
| E11 | LongMemEval (`data/longmemeval/*.json`): question types, history lengths, answer sessions and `has_answer` turns. Prints each file's sha256. Written from the README before the data was seen. |
| E12 | After a fact is introduced, how often do later sessions mention it again (strict string match, user turns, excluding update/delete of the same fact)? Lower bound on natural recall opportunities. |
| E13 | Is file order the same as date order (assumption A6), and are session IDs and turn numbers unique and consecutive? Needed before applying operations in order and pointing memories at source turns. |
| E14 | How many code-gradable probe candidates (numbers, quoted titles, names, hashtags, qualifiers, the record's own value) do the flagged turns of add and update sessions give, per persona, and how many belong to facts never changed later? Sizes the probe pool for the power estimate (D10). Candidates found by pattern, not checked probes. |
| E15 | With embedding retrieval (top 3, a grid of thresholds), how often are facts recalled by later user turns on the development personas, and where should the relevance threshold sit (own update/delete sessions as hits, small-talk turns as likely false recalls)? Needs the `embed` server running. |
| E16 | On the development personas, how well does ingestion keep each fact's checkable details (note and log arms vs the oracle), does it invent values or store small talk, how long are its answers and how long does it take; and how often does retrieval find a fact when the user talks about it, vs small talk recalling something? Runs the real ingestion with the agent model (about 950 calls); needs the agent and embed servers. `MEMRECON_E16_PILOT=N` runs only N sessions, for timing (not evidence). |
| E17 | With note memories (E16's note ingestion, replayed from the cache) and threshold 0.70, how often is each item recalled naturally on the development personas, and what do split items and notes from delete sessions look like? No new model calls; needs the agent server (for cache lookups through the client) and the embed server. |
| E18 | On the development personas, does ingestion prompt v2 fix what E16/E17 found in v1 (splits, duplicates, working-out, merged history, deletes, finished tasks), without losing details? Also update recognition: were a record's earlier memories shown at its update and replaced, and how many stale copies remain. v1 replays E16 from the cache; v2 makes about 476 new calls (about 25 min); needs the agent and embed servers. `MEMRECON_E18_PILOT=N` runs only N sessions, for timing (not evidence). |
| E19 | Follow-ups on E18, from E18's cached calls (no new calls): why second attempts were needed, retrieval vs small talk over thresholds for v1 and v2 notes (and memory length found vs missed), changes that rewrite a memory of another record, add/update sessions that saved nothing, relative dates in document contents vs other memories, and samples for "we" and change descriptions. Needs the agent and embed servers. |
| E20 | Ingestion prompt v5 (v1's text with the changes decided in D38 and D39: v2's update and duplicate rules, no folding of new events) against v1 and v2 on the development personas: E18's measures, retrieval vs small talk across thresholds, changes replacing another record's memory (preference sessions vs the rest), sessions that saved nothing, and retention (are never-changed records' details still in an active memory at the end?). v1 and v2 replay from the cache; v5 makes about 480 new calls (about 18 min); needs the agent and embed servers. `MEMRECON_E20_PILOT=N` runs only N sessions, for timing (not evidence). |
| E21 | Two-step (Mem0-style) note ingestion against one-call v1, v2 and v5 on the development personas: E20's measures, plus how each extracted fact was handled and how similar the candidate memories of the fact's own record are compared with other records' (to set the candidate threshold, `MEMRECON_E21_THRESHOLD`, default 0.0). v1, v2 and v5 replay from the cache; two-step makes about 476 extraction calls plus one per fact with a candidate (about 30 min); needs the agent and embed servers. `MEMRECON_E21_PILOT=N` runs only N sessions, for timing (not evidence). |
| E22 | Two-step ingestion with prompts v2 (every fact names its document; a fact describing a change is a replace or forget) against one-call v1 and v5 and two-step v1 on the development personas: E21's sections, with the candidate similarity of the fact's own record vs other records for both two-step versions. v1, v5 and two-step v1 replay from the cache; two-step v2 makes about 1,450 new calls (about 26 min); needs the agent and embed servers. `MEMRECON_E22_PILOT=N` runs only N sessions, for timing (not evidence). |
| E23 | Two-step ingestion with step 2 shown the other facts of the same conversation (decide v3, D43) against one-call v1 and v5 and two-step v2 on the development personas: E22's sections, mainly cross-document folding. v1 and v5 replay from the cache; two-step v2 nearly so (D42 applies); two-step v3 makes about 1,350 new calls (about 26 min); needs the agent and embed servers. `MEMRECON_E23_PILOT=N` runs only N sessions, for timing (not evidence). |

`common.py` holds the shared loaders and the version helpers. `../make_split.py` recreates or checks the frozen persona split.
