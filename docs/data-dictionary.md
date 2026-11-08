# Data dictionary

What the words in this project mean, and what every table and file holds: each column, where its value
comes from, and who writes it. It is a map, kept short. Behaviour is defined by the module docstrings
(`src/memrecon/*.py`) and decisions by `notes/decision-log.md`; where this file and those disagree, they
win, and this file should be fixed. Decision (D), evidence (E) and assumption (A) numbers point into
the decision log.

Add to this file whenever a new term, table or column appears.

## Contents

1. [Terms](#1-terms)
2. [Input data: Memora](#2-input-data-memora)
3. [The memory store](#3-the-memory-store)
4. [The audit log](#4-the-audit-log)
5. [Ingestion records](#5-ingestion-records)
6. [The drift measure's answer key (probes)](#6-the-drift-measures-answer-key-probes)
7. [Run summaries and monitoring](#7-run-summaries-and-monitoring)
8. [Configuration](#8-configuration)
9. [Notes and registers](#9-notes-and-registers)

## 1. Terms

### The experiment

| Term | Meaning here |
|---|---|
| **write policy** | What happens to the agent's recall-time rewrite of a memory. The only thing that varies between conditions; everything else (model, prompts, retrieval, budget) is held fixed. |
| **discard** | Write policy: the rewrite is used for the task and thrown away; the memory is unchanged (its recall counter goes up). The no-rewrite control. |
| **replace** | Write policy: the rewrite overwrites the memory (same ID; the old text is kept only in its version history). |
| **keep** | Write policy: the rewrite is stored as a new memory and the original stays. The proposal calls this "added (both kept)". |
| **verify** | Stage 2 write policy (not built yet): the rewrite is written back only after a check against the original. |
| **arm** | One condition of a run: a memory mode (note, log, oracle) combined with a write policy. |
| **note arm** | Memories are notes the agent model writes from what the user said. |
| **log arm** | Memories are the user's own words, verbatim: spans of the turns the model picks, checked to be exact substrings (whole turn kept if not). Append-only; write policies do not run on it (D24). Not to be confused with the **audit log**. |
| **oracle arm** | Upper-bound reference: the turns Memora flags (`share_memory`), verbatim, with no model call. The only arm that uses Memora's annotations; labelled as such. |
| **Stage 0 / 1 / 1b / 2 / 3** | The staged plan in `CLAUDE.md`: 0 checks the measurement, 1 asks Q1 (persist vs discard), 1b repeats Stage 1 on synthetic hedges, 2 asks Q2 (verification), 3 asks Q3 (when to verify). |
| **development / test personas** | `notes/split.json` (seed 329): 3 development personas for tuning prompts and thresholds, 7 test personas for reported results only. Frozen. |

### Memory

| Term | Meaning here |
|---|---|
| **memory** | One stored entry in the memory store (text plus lineage). What the agent sees of it is only its text and date. |
| **active** | Not retired: findable by retrieval and shown to the model. |
| **retired** | Kept for scoring but never shown or retrieved again: `superseded` (a newer memory replaced it) or `forgotten` (the user asked). Nothing is ever deleted. |
| **supersede** | A user's change: a new note is written from the old memories plus the user's new words; the old ones are retired as superseded (D29). Also how a finished task is recorded (D38). |
| **forget** | A user's request to remove something entirely: the memories are retired as forgotten, nothing new is written. |
| **depth** | Generations of rewriting between a memory's text and what the user said: log 0, note 1, anything written from memories the deepest of them + 1. Never resets. The primary depth for the analysis (D27, D33). |
| **since_user** | The same count, except that a user's change restarts it (note 1, log 0). A secondary view. |
| **lineage** | `written_from`, `parents`, `sources`, `superseded_by`, `retired`, `versions`: kept for the harness, **never shown to the model** (D28). |
| **date** (of a memory) | The date of the conversation its information came from; shown to the model as "(2025-06-04)". A rewrite keeps its parent's date; a user's change takes the date of its conversation (D36). |
| **present** | The only way the model sees memories: numbered `[1]`, `[2]`… afresh for each call, with text and date; the harness maps the numbers back. |

### Saving (ingestion)

| Term | Meaning here |
|---|---|
| **ingestion** (saving) | Writing a finished session into memory. Runs once per session, after it ends. |
| **one-call ingestion** | One model call per session that both extracts what the user said and decides what it changes (prompts `ingest_note_v1`–`v5`, `ingest_log_v1`). Kept for replaying earlier runs. |
| **two-step ingestion** | As in Mem0 (D40): **step 1, extract** writes facts from the session without seeing memory (`extract_note_v*`); **step 2, decide** takes each fact with its most similar memories and answers add, replace, forget or none (`decide_note_v*`). |
| **fact** | One item step 1 writes: a self-contained sentence about the user, with the line numbers it came from. Becomes a memory if added. |
| **related memories** | One-call ingestion: the memories shown with a session, the top 3 per user turn by similarity, up to 10, no threshold. |
| **candidates** | Two-step: the memories shown with one fact, its top 5 by similarity reaching the **candidate threshold** (`candidate_threshold`, default 0.0, as in Mem0). A fact with none is added without a call. |
| **identical replace** | A replace whose text equals the one memory it replaces. Recorded as `none (identical replace)` and not a rewrite: depth unchanged (D42). |
| **folding** | A replace that overwrites a memory of a *different* record (another coffee purchase, another project's budget): a rewrite of something the user never changed. Measured in E19–E23. |
| **stale copy** | An active memory still holding a value the user has since changed. |
| **retention** | For records the user never changed: the share of their details still in an active memory at the end of the replay (E20 on). Losses mean ingestion rewrote something the user did not change. |

### Data and records

| Term | Meaning here |
|---|---|
| **persona** | One simulated user in Memora; one weekly history of about 155 sessions. |
| **session** | One conversation (`session_*.json`), with a date and at most one memory operation. |
| **turn / line** | A turn is one message in Memora's `conversation`. A line is the same turn as numbered for the model ("3. User: …", counting all turns from 1). |
| **flagged turn** | A user turn with `share_memory: true`: Memora's mark that it carries memory content. Used on the harness side only (probes, the oracle arm), never shown to the model. |
| **operation** | Memora's add, update or delete for a session. Not the same as the store's operations (log, note, forget, recall, keep, replace, present). |
| **record** | The thing a session's operation is about, identified by a key (A10): a preference's item, a to-do's description, a calendar event's name, an expense's amount, a step count, a document's ID (`meeting_notes_1`); goals by subcategory (probes). |
| **item** | In E13–E24, a record told about in an add or update session (counted for recall and survival). In probes, `kind: item` means one list item (an agenda item, a recipient). |
| **document** | An activity record with no category, whose log holds full `content_data`: meeting notes, emails, posts, project proposals. |

### Recall and measurement

| Term | Meaning here |
|---|---|
| **recall** | A memory retrieved for a user turn (top 3 reaching the relevance threshold, 0.70, D37). Under keep and replace, each recall is a rewrite. |
| **natural recall** | Recall by the replayed conversation's own user turns (D14). |
| **injected query** | Planned: extra recall queries about specific details, to build depth where natural recall is too rare (E17). |
| **drift** | How far a memory's content has moved from the latest legitimate state of what the user said. Measured by probes, never counted. |
| **probe** | One checkable detail of the answer key (a number, date, name, title, list item…) with its expected state at a point in time. |
| **answer key** | The probe table built from the user's words by the extractor (section 6). |
| **primary / secondary probe** | Primary: graded by code (kinds number, date, name, title, item). Secondary: graded by the judge (tasks, preferences, done, no longer) (D45). |
| **FAMA, MPA, FAA** | Memora's forgetting-aware score: MPA and FAA are the shares of "should mention" and "should not mention" sub-questions answered correctly; FAMA = max(0, MPA − λ(1 − FAA)) (`src/memrecon/fama.py`). |
| **judge** | The judge model role (a different family from the agent): extracts the answer key and grades what code cannot, checked against Sam's hand labels (D21). |
| **hand labels / hand check** | Sam's own marks on samples, the reference the judge and the extractor are checked against. Samples are stratified (`CLAUDE.md`). |

### The harness

| Term | Meaning here |
|---|---|
| **role** | What code asks for: `agent`, `judge`, `embed`, `agent-small`. Which model serves a role is set only in `config/models.env` (D18). |
| **thinking off** | Every chat call sends `chat_template_kwargs: {"enable_thinking": false}` (or the provider's equivalent); a reply with reasoning text is rejected. |
| **audit log** | The harness's append-only records: raw calls, attempts, ledger, snapshots, event files (D30). Never overwritten or deleted. Not the log arm. |
| **raw call** | One request and its raw response, saved before parsing (section 4). |
| **attempt** | An earlier try at the same request, renamed `<key>.attempt<N>.json` before a retry writes. |
| **ledger** | One row per call that reached a model: tokens, time, cost, outcome (section 4). |
| **cache key** | sha256 of the full request (model, messages, every setting); a repeated request is answered from its saved file. A resume aid, not proof of reproducibility. |
| **tag / meta** | A run label (`tag`, e.g. "E21") and facts about a call (`meta`: step, persona, session, attempt) recorded in the ledger. Neither is part of the cache key. |
| **unusable answer** | A reply that fails the strict parse (wrong block, bad JSON, wrong keys, a span not copied exactly…). Sent back with the reason and the answer format, up to 3 attempts; never repaired. |
| **pilot** | A run of the first N sessions (`MEMRECON_E<n>_PILOT=N`) for timing; never evidence. |

## 2. Input data: Memora

Downloaded by `scripts/download_data.sh` (commit `a6493188`) to `data/memora/` (git-ignored). Weekly split:
`data/memora/data/weekly/<persona>/`.

**`conversations/session_<n>.json`**, one per session, in time order by file name (A6, E13):

| Field | What it is |
|---|---|
| `session_id` | Integer, unique within a persona. |
| `persona` | The persona's name. |
| `date` | Conversation date (`YYYY-MM-DD`). Several sessions share a day. |
| `session_type` | `activity`, `preference`, `goal` or `no_memory` (small talk) (A1). |
| `operation` | `add`, `update`, `delete`, or none for small talk (A1). |
| `operation_details` | The log of what changed (E25). Activity: `category` (`todo_list`, `calendar_event`, `food_expenses`, `step_tracker`, or none for documents) and `item` (a dict, or a document ID with `content_data`, the whole document after the change). Preference: `item`, `preference` (like/dislike), `subcategory`; updates add `old_item`, `old_preference`, `update_type`. Goal: `subcategory`, `item` (amount). |
| `conversation` | List of turns: `turn`, `speaker` (`user_agent` or `ai_agent`), `message`, `share_memory` (the flag). |

**`evaluation_questions_<persona>.json`**: `persona`, `date_range`, and `questions` grouped by category; each
has `question_id`, `question`, `question_date`, `memory_evidence`, `forgetting_evidence` and an `evaluation`
block of yes/no sub-questions ("should mention" / "should not mention"), which FAMA scores (A7, A8). The
evidence blocks are for auditing only.

**`notes/split.json`**: `seed`, `dev` (development personas), `test` (held-out). Frozen.

## 3. The memory store

`src/memrecon/memory.py`. One store per persona and arm.

**Memory** (one per memory, in snapshots):

| Field | What it is | Written by |
|---|---|---|
| `id` | `m00001`, … in insertion order. Never shown to the model. | store |
| `text` | The memory's current text. Shown to the model. | model (note arm), user's words (log, oracle) |
| `depth` | Generations of rewriting (section 1). | store |
| `since_user` | Depth restarted at a user's change. | store |
| `written_from` | `user`, `memory` or `both`. | store |
| `parents` | IDs of the memories it was written from. | store |
| `sources` | Every user turn in its ancestry: `persona`, `session_id`, `turn`. | harness |
| `versions` | Text history: `text`, `depth`, `since_user`, `at`. | store |
| `created_at` | `At`: `session_id`, `turn` when written. | harness |
| `date` | Date of the conversation its information came from. Shown to the model. | harness |
| `times_recalled` | Recall counter. Not in `content_json` (D26). | store |
| `superseded_by` | ID of the memory that replaced it, if any. | store |
| `retired`, `retired_at` | `None`, `superseded` or `forgotten`, and when. | store |

**Snapshot**: canonical JSON of the store after each session (sorted keys, no clock times; format
`SNAPSHOT_FORMAT`). **`content_json()`**: the same without `times_recalled` and the event count, compared
byte for byte in the Stage 0 no-trace check (D19, D26).

**Event file** (part of the audit log): JSONL, one line per store operation, in order. Every line has
`n` (event number, continuing across resumes), `op` and `at`; then by `op`:

| `op` | Fields |
|---|---|
| `log`, `note` | `id`, `text`, `depth`, `since_user`, `date`, `supersedes`, `sources` |
| `forget` | `ids` |
| `recall` | `id`, `times_recalled` |
| `keep` | `id` (new), `parent`, `text`, `depth`, `since_user` |
| `replace` | `id`, `text`, `depth`, `since_user` |
| `present` | `labels` (number shown → memory ID) |

**Embedding cache**: `results/embeddings.sqlite`, table `vectors` (`key` = hash of the model name and the
exact prefixed text; `model`; `dim`; `vec`, normalised to length 1, so a dot product is the cosine
similarity). Queries take the prefix `search_query:`, memories `search_document:` (nomic-embed-text).
Embedding calls that reach the server are ledger rows too (role `embed`). Embeddings are never shown to
the model.

## 4. The audit log

All under `results/raw_calls/` (git-ignored), written by `src/memrecon/llm.py`. Append-only (D30).

**Raw call**, `results/raw_calls/<role>/<key[:2]>/<key>.json`:

| Field | What it is |
|---|---|
| `key` | Cache key: sha256 of the request. |
| `role`, `provider`, `host`, `base_url` | Which role, which kind of server (`llamacpp`, `openrouter`), where. |
| `request` | The full request body as sent (messages and every setting, thinking off). |
| `started_at`, `finished_at` | UTC timestamps. |
| `http_status`, `response_text` | The raw response, saved before any parsing. |
| `attempts` | HTTP tries inside this call (connection retries). |
| `latency_s` | Seconds. |

**Attempt**: `<key>.attempt<N>.json`, an earlier answer to the same request kept as it was (cut off, empty,
failed or rejected) before a retry wrote the new one.

**Ledger**, `results/raw_calls/ledger.jsonl`, one row per call that reached a model (cache hits are not
rows):

| Column | What it is |
|---|---|
| `at` | When it finished. |
| `role`, `model`, `served_by` | Role, model name, and the provider that served it. |
| `tag` | Run label (argument, or `MEMRECON_TAG`). |
| `key` | The raw call's key. |
| `http_status`, `finish_reason` | From the response. |
| `outcome` | `ok`, `empty`, or why it was rejected: `reasoning`, `truncated`, `http_error`, `unreadable`. Rejected calls count towards the budget but are left out of cost evaluations of the systems. |
| `prompt_tokens`, `completion_tokens` | From the response's usage. |
| `cost_usd` | 0 for local models; reported cost for paid ones (`None` if not reported). |
| `latency_s` | Seconds. |
| `meta` | Facts the caller added: `step` (`ingest`, `extract`, `decide`, `probe_extract`), `persona`, `session`, `attempt`. |

`uv run python -m memrecon.llm` prints calls, tokens and spending by role, model and tag.

## 5. Ingestion records

`src/memrecon/ingest.py`. Harness side only; never shown to the model.

**Outcome** (returned for each session ingested):

| Field | What it is |
|---|---|
| `session_id`, `ok`, `attempts`, `error` | Whether the answer was usable, after how many attempts, and the last reason if not. |
| `created`, `superseded`, `forgotten` | Memory IDs written, replaced and forgotten. |
| `span_fallbacks` | Log arm: kept words that were not an exact part of the turn (the whole turn was kept). |
| `facts` | Two-step: facts extracted. |
| `decisions` | Two-step: count per outcome (`add`, `add (no candidate)`, `add (decision failed)`, `replace`, `forget`, `none`, `none (identical replace)`). |
| `decide_failures` | Two-step: decisions that failed 3 times (the fact is then added). |

**Ingestor attributes** read by exploration scripts: `last_shown` (memories shown for the last session),
`decision_log` (two-step, one entry per fact: `fact`, `candidates` as (ID, similarity), `action`, `ids`,
and `unchanged` for an identical replace), `calls` (one per model call: `session`, `step`, `attempt`,
`cached`, tokens).

**Prompt files**, `src/memrecon/prompts/<name>.md`: a leading comment (never sent; why the version exists),
then `=== SYSTEM ===` and `=== USER ===` sent exactly as written, with `{placeholders}` filled by the
harness. A version used in a run is never edited; changes go in the next version.

## 6. The drift measure's answer key (probes)

`src/memrecon/probes.py`, prompts `probe_extract_v1.md` (E26), `probe_extract_v2.md` (E28, D46) and
`probe_extract_v3.md` (E29, D47) and `probe_extract_v4.md` (E30, D48), run by the judge (D44–D48), then
the split pass, `probe_split_v1.md` (E31, D49). Harness side only.

**Row** (one per extracted detail; stored append-only, never edited; E26, E28, E29 and E30 write them to, and E31 writes the split pass's parts to
`results/probes/E<n>_<persona>.jsonl`):

| Column | What it is | Comes from |
|---|---|---|
| `row_id` | Unique across the answer key: `persona:session:id`, from the number the model gives the detail (version 2; empty for version 1). | harness |
| `record` | The record the session is about (A10; goals by subcategory). | harness (Memora's log) |
| `field` | What the detail is, in terms of what it belongs to (`meeting notes title`, `agenda item`, `action item deadline`, `liked author`); reuses the record's earlier field names, shown as "fields so far". | extractor |
| `detail` | The detail, copied exactly from the line (checked by code). | extractor |
| `qualifier` | A word or phrase that limits, hedges or conditions the detail, copied exactly ("about", "at least", "only if approved"), or empty. Collected for every kind; scored only for numbers and dates (D47). Not "about" meaning "on the subject of", nor "from"/"to" in a change (version 3's definition). | extractor |
| `who` | Who did, decided or is to do the thing: `user`, `user and others` ("we"), or, from version 2, the person, team or organisation named in the line, copied exactly (version 1: `someone else`); empty otherwise. From version 4 the user's own activities ("I spent", "I walked") get `user`, and the contents of a document or list get none even when the user asks for them to be changed. | extractor |
| `kind` | `number` (amount, count, percentage, score, level such as "medium"), `date` (date, day, time, deadline, duration, timeframe), `name` (a proper name only), `title` (a thing's own title), `item` (any other single thing), `task` (something to do or done), `preference` (the liking itself). Defined one by one from version 4. | extractor |
| `status` | What the statement did when it was made, fixed (a row is never re-labelled when a later session changes it): `new` (first told; from version 4 also something added to a thing told before), `added` (versions 1–3: added to something told before; dropped in version 4, D48), `updated` (replaces an earlier value), `replaced` (the earlier value, when not already a row; version 4 refuses it for a value shown as an E row), `removed`, `done`, `no longer` (the field's claim no longer holds). | extractor |
| `session`, `line`, `date` | Where it was said, and the conversation date ("said on"). | harness |
| `number`, `unit` | For kind number: value and unit ("10k steps" → 10000, "steps"). From version 4 also for kind date when it holds a duration ("within seven days" → 7, "day"; `split_duration`). | code |
| `resolved` | For kind date: the calendar day, from the conversation date, when it names one day ("next Tuesday"); empty for ranges ("next week"). | code |
| `attached_to` | What the detail belongs to: the `row_id` of another detail (a deadline, timeframe, owner or amount of an action item, decision or task); from version 3 also an earlier session's row (an E label), the record itself (a root id, `persona:R:<record>`, "R" to the model: "This post is for LinkedIn"), or `unknown` (it clearly belongs to something the line does not name). Empty: it stands on its own. Version 2's links reach only the same session. | extractor (an id or label), harness (the row_id) |
| `replaces` | Version 3: for `updated`, `removed`, `done` and `no longer`, the `row_id` of the earlier row it replaces, removes, finishes or no longer holds (from an E label), or empty when that row is not recorded. | extractor (a label), harness (the row_id) |
| `origin` | `extract` (the extractor's own row) or `split` (a part the split pass found in a long detail, D49: `row_id` `<whole>.<k>`, attached to the whole, status `new`, other columns from the whole except the qualifier). | harness |

**Table entry** (computed from all rows by `build_table`, never stored): the row's columns plus `state`
(`true` should be in memory, `false` should not, `done`, `no longer`), `from_session` / `from_date` and
`until_session` / `until_date` (empty while it still holds), and `row_id`, `attached_to` and
`attached_detail` (the text of the row it belongs to, or the root id or `unknown`), and `replaces`. A row
with `replaces` closes exactly that row (an `updated` one then holds in its place); without it, a linked `updated` closes the field's earlier values
linked to the same thing (one action item's deadline, not another's); an unlinked one closes all the field's
earlier values; `added` does not; `replaced`, `removed`, `done` and `no longer` close or mark the matching value.
When an entry closes, the open entries attached to it close with it, in the same state (D49): an action
item's deadline when the item is done, a removed item's parts. `true_at(entries, session)` gives what holds
just after a session.

**Split pass** (`ProbeSplitter`, D49): a row is a candidate (`split_candidate`) when its detail is at least 6
words, contains "and", is not a number, date or name, and its status is `new`, `added` or `updated`. The
judge answers with no parts (one thing) or two or more, each copied exactly from the detail (code checks).
Decisions are kept in `results/probes/E31_decisions_<persona>.jsonl` (`row_id`, `detail`, `parts`, `ok`).

**Grading view** (`grading_view`, D45): each entry with `grade` (`code` for kinds number, date, name, title,
item in states true or false; `judge` otherwise), its `qualifier` only for numbers and dates and only if on
the hedge list (`HEDGES`), and
`who` only for kind `task` and only if not `user` (D50; judge-graded). Number, unit and resolved day are recomputed from the detail by code when
grading, so rows extracted before a parser fix are graded with the current parser.

**Code grading of memories** (`code_match`, D50): whether one memory holds a code-graded probe.
- *Number*: by value (`numbers_in` reads "$1.55M", "10k", "seven days"; ISO dates left out), and the unit's
  word when it has one besides a currency ("10,151 steps" needs "steps").
- *Date*: the text as told, the resolved day in any common form (`date_forms`: "2025-06-03", "June 3",
  "3rd of June"), or a duration's number and unit ("within seven days" = "within 7 days").
- *Name, title, item*: under `LONG_WORDS` (5) words, any alias (`aliases`: without a leading article or final
  stop, without a title such as "Dr.", a bracket dropped, or an abbreviation in brackets standing for what
  precedes it) as whole words; 5 words or more, the share of its content words (`content_words`: stop words
  dropped, plural "s" dropped, hyphens split) the memory holds, at least `OVERLAP` (0.8).
- *Qualifier* (`qualifier_kept`): a graded qualifier must be in the same memory as its detail.
- *With its parent* (`match_with_parent`, `parent_entry`, D51): a detail attached to another detail (a
  deadline, a timeframe, an amount, a risk level; not one attached to the record itself, and not a split
  part) counts only in a memory that also holds that detail, found at `PARENT_OVERLAP` (0.5).

**A preference no longer liked** (`is_record_item`, D51): when a preference record's own item (the thing it
is named after) is marked `no longer` or `removed`, every other open entry of that record told before
closes with it, in the same state (its reasons and examples no longer hold as current likes).

**Probe terms used in scoring** (E32): *recall*, the share of probes that should hold (state `true`) found
in some active memory; *found when told*, found right after the session the detail was told in; *lost*,
found when told and not at the end; *old value*, a value that held and was later updated or removed,
found in an active memory (may be history, not an error).

**What version 3 is shown as already recorded** (`ProbeExtractor.recorded`): `R`, the record itself, then
`E1`, `E2` ... for the record's entries that still hold and for other records' entries whose detail appears
in the lines to read (up to 15; a liked item the user now says they no longer like is another record under
A10), each with its field and, if not `true`, its state. The labels are local to one call; the harness maps
them to `row_id`s.

**Hand-check store** (the Probe Hand Check page's database): collection `labels`, one document per row
(`v3~persona~session~line~index` for extractor version 3, `v2~...` for version 2; version 1 marks had no
prefix: `verdict` ok/wrong, `note`, and the row's `detail`, `field`, `status`, `kind`, `row`, `attached_to`,
`replaces` (version 3), `version`); collection `missing`, one per line (`v3~persona~session~line`: `text`). Read back into
`notes/labels/` when the check is done.

## 7. Run summaries and monitoring

`src/memrecon/monitor.py`.

**Run summary**, `<snapshot>.summary.json`, next to each snapshot: `persona`, `mode`, `live_memories`,
`live_text_chars`, `live_text_tokens_est` (characters / 4), `depths` (count per depth), `max_depth`,
`retired` (count per reason), `total_recalls`. The agent's memory as in use, without audit-log overhead.

`scripts/monitor.py` shows system and GPU memory, servers, ledger rates, the newest summaries and the size
of `results/`. Progress logs: `results/e<n>_progress.log` for exploration runs that write one.

## 8. Configuration

**`config/models.env`** (local) and **`config/models.openrouter.env`** (hosted), per role (`AGENT`, `JUDGE`,
`EMBED`, `AGENT_SMALL`): `_KIND` (chat or embed), `_FILE` (local `.gguf`), `_NAME` (model name recorded in
every call), `_PORT`, `_CTX`; for embed, `EMBED_QUERY_PREFIX` and `EMBED_DOC_PREFIX`.

**Environment** (`.env`, git-ignored; see `.env.example`): `MEMRECON_MODELS` (which models file),
`MEMRECON_ALLOW_PAID=1` (needed for any paid call; never set without Sam's say-so), `MEMRECON_BUDGET_USD`,
`MEMRECON_MIN_FREE_GB` (memory guard, default 8), `MEMRECON_TAG`, `MEMRECON_<ROLE>_URL` and
`MEMRECON_<ROLE>_MODEL` (per-machine overrides), `MEMRECON_E<n>_PILOT` and `MEMRECON_E<n>_THRESHOLD`
(exploration scripts).

## 9. Notes and registers

**`notes/decision-log.md`**:

| Register | Columns |
|---|---|
| Decision log (section 2) | ID (D#), Date, Decision, Why, Proposed by (Sam, Claude, Both), Written by (Claude, date), Review (`pending`, or `Sam, <date>`, which also confirms the decision), Evidence |
| Evidence register (section 9) | ID (E#), Script, Question, Result, Assumptions, Run (who, where, commit), Used by, Status |
| Assumptions register (section 10) | ID (A#), Assumption, Basis, Evidence, Could be wrong if, Decisions, Status (`revised` reopens the decisions citing it) |

**`notes/evidence/E<n>.txt`**: an exploration script's output, saved by `scripts/explore/run.py` with a
header: `script`, `run at`, `memrecon` (commit, and whether scripts were clean), `Memora data` (commit),
`exit status`. Committed with the script.

**`scripts/explore/E<n>_<name>.py`**: one script per question about the data; its docstring states the
question, method and limits. `scripts/explore/README.md` lists them.
