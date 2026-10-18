# memrecon decision log

Running record of decisions, findings and deviations from the submitted proposal, kept so the midway and final reports can be written from it. **Append new entries as they happen, with the date. Do not rewrite reviewed entries; if a decision changes, add a new entry that says so.**

Proposal: *Keep, Discard, or Verify: Write Policies for Recall-Time Memory Rewrites in LLM Agents* (submitted Oct 9, 2026). Solo project, CS329Z, Stanford.

## How provenance works

Every entry records who proposed it, who wrote it down, and whether Sam has reviewed it.

- **ID:** decisions are numbered `D01`, `D02`, … in the order they were logged, so they can be discussed by number. Never renumber; new decisions get the next number.
- **Proposed by:** `Sam`, `Claude` or `Both`.
- **Written by:** who wrote the entry, with the date, e.g. `Claude, Oct 16` or `Sam, Oct 16`.
- **Review:** `pending` until Sam checks the entry, then `Sam, Oct 16` (name and date, same style as Written by). **Reviewing means Sam has checked that the record is accurate and confirms the decision stands.** Only Sam changes an entry to reviewed. Claude never marks its own entries reviewed.
- **A Claude-proposed entry with `pending` review is a recommendation, not a decision.** Do not build on it as settled; ask Sam first, unless it has already been acted on (and then say so in the entry).
- **Evidence:** the evidence (`E01`, …) and assumptions (`A1`, …) a decision rests on, defined in sections 9 and 10. Decisions that rest on a data finding must cite it.
- **Reopening:** if an assumption turns out to be wrong (its status in section 10 becomes `revised`), change the Review cell of every decision that cites it to `reopened (A#, date)` and add a line to section 5. This is the one change allowed to a reviewed entry, and Claude may make it; Sam decides what to do about it.
- New entries in sections 4–10 end with a tag in this form: `[Proposed: Claude · Written: Claude, 2026-10-17 · Review: pending]`.
- Once an entry is reviewed, do not edit it otherwise. If it turns out to be wrong, add a new dated entry that corrects it.
- To list everything waiting for review: `grep -n "review: pending\|Review: pending\|| pending |" notes/decision-log.md`
- To find the decisions that depend on an assumption or a piece of evidence: `grep -n "A2" notes/decision-log.md` (or `E06`, …).

This also gives a ready record for the AI-use disclosure in the reports: what Claude proposed, what Sam decided, and what Sam checked.

---

## 1. Where the project stands

- Stage 0 (measurement gate): not started.
- Stage 1 (Q1: keep vs discard the rewrite): not started.
- Data: Memora weekly split downloaded and inspected (Oct 15). Exploration scripts E01–E12 written, committed and run on the EVO-X2 (Oct 16, code commit `bb38eda`, scripts clean); stamped outputs are in `notes/evidence/` (commit `72e7b8f`).
- Compute: local models only; no API keys set up yet.

## 2. Decision log (newest last)

Evidence (`E#`) and assumptions (`A#`) are defined in sections 9 and 10.

| ID | Date | Decision | Why | Proposed by | Written by | Review | Evidence |
|---|---|---|---|---|---|---|---|
| D01 | Oct 8 | Committed to the memory-reconsolidation topic; solo project over the full quarter. | Lecturer found it interesting and said data, evals and metrics had to be thought through in the proposal. | Sam | Claude, Oct 16 | Sam, Oct 16 | — |
| D02 | Oct 8–9 | Narrowed the question to whether an agent should persist a recall-time rewrite rather than discard it. Staged design (Stages 0–3). | Gives a clean Q1 → Q2 → Q3 chain where each stage is informative whether or not drift appears. | Both | Claude, Oct 16 | Sam, Oct 16 | — |
| D03 | Oct 9 | Proposal submitted. Citation-chain tracer dropped as a testbed. | Not needed for the staged design. | Sam | Claude, Oct 16 | Sam, Oct 16 | — |
| D04 | Oct 15 | New repo `memrecon` (private), not a rename of `citation-tracer`. | Different project and history; a rename would carry stale commits and notes. | Both | Claude, Oct 16 | Sam, Oct 16 | — |
| D05 | Oct 15 | No paid API calls and no API keys for now; local llama.cpp server only. Model endpoint set via environment variables. | Protect the credits; switching later should be a `.env` change only. | Sam | Claude, Oct 16 | Sam, Oct 16 | — |
| D06 | Oct 15 | Durability rules: one result file per unit of work, atomic writes, resume by skipping, raw outputs saved before parsing, `tmux` for long jobs, backups of `results/`. | Runs can be interrupted (SSH drop, power, network). | Both | Claude, Oct 16 | Sam, Oct 16 | — |
| D07 | Oct 15 | Docker deferred until the project first moves to another machine. Portability via `uv.lock`, `.env` endpoint and a data download script. | llama.cpp with Vulkan in Docker needs GPU passthrough; the harness only needs an endpoint URL. | Both | Claude, Oct 16 | Sam, Oct 16 | — |
| D08 | Oct 15 | Dev/held-out split fixed: seed 329, 3 development personas (`financial_analyst`, `management_consultant`, `sales_manager`), 7 test personas. Saved in `notes/split.json`. | Fixed before any results were seen. **Frozen.** Never retune on test personas. `scripts/make_split.py` recreates and checks it (reproduced Oct 16). | Claude | Claude, Oct 16 | Sam, Oct 16 | `make_split.py` |
| D09 | Oct 15 | Hedge probes: start without a separate hedge probe (option A); add a synthetic hedge subset as **Stage 1b** (option B). | Natural hedges are almost absent in Memora: about 7 real hedges in 1,613 flagged turns. | Both | Claude, Oct 16 | Sam, Oct 16 | E05 · A4 |
| D10 | Oct 15 | Probes are built from the dialogue (`share_memory: True` turns), not from `operation_details`. | For activity updates the log holds the whole record but the dialogue states only the change (about a third of logged values appear verbatim); preference and goal values are almost always spoken. | Claude | Claude, Oct 16 | pending | E03 · E04 · E06 · E07 · A1 · A2 · A3 |
| D11 | Oct 16 | Provenance and review convention added to this log: Proposed by, Written by, Review (with name and date). | Sam wants to see who added each item and whether she has checked it. | Sam | Claude, Oct 16 | Sam, Oct 16 | — |
| D12 | Oct 16 | Evidence trail: every data finding comes from a numbered script in `scripts/explore/`, run through `run.py` (output saved in `notes/evidence/`), and is logged in sections 9–10 with the assumptions it uses. Decisions carry IDs and cite evidence; if an assumption proves wrong, dependent decisions are marked reopened. | If we later find we misread a field, we need to know which decisions were made on that basis and re-evaluate them. | Sam | Claude, Oct 16 | Sam, Oct 16 | — |
| D13 | Oct 17 | Memora stays the main dataset for now. LongMemEval and LoCoMo are not adopted; a LongMemEval subset remains a possible external check. | Sam's reasons: work already done on Memora; it is smaller and more manageable; and, given how it is annotated, she expects it to be easy to inject extra recall events later to increase rewrite depth (an expectation, not yet tested). Facts that bear on it: Memora has an operations log and small histories (about 683k tokens for all ten weekly personas, versus about 61 million for LongMemEval S ingested per question); LoCoMo has no update information. | Sam | Claude, Oct 17 | pending | E09 · E10 · E11 |
| D14 | Oct 17 | Recall happens throughout the replay: on each replayed user turn the agent retrieves related memories, rewrites them for the current discussion, and the write policy decides what is stored (natural recall). Memora's end-of-timeline questions measure the resulting state. Our own queries about specific facts are a separate, clearly labelled arm, used only if natural recalls are too rare to build rewrite depth. | Reconsolidation happens during ongoing experience, not only when tested (Sam). How often a fact gets recalled naturally is unmeasured (E12, A9), so the injected arm is a fallback. Scripted user turns do not react to the agent's replies (see section 7). | Both | Claude, Oct 17 | pending | A9 · E12 |

## 3. Deviations from the submitted proposal

State these openly in the reports. *Written by Claude, Oct 16 · review: pending (whole section).*

1. **Probe source.** The proposal says Memora's log gives per-fact ground truth for repeated recall. In practice we build probes from the dialogue (flagged turns) and use the log only to find the latest legitimate state of each fact (D10; E03, E06).
2. **Recall events.** Memora asks each of its 15 questions per persona once, so its questions alone cannot build rewrite depth. D14 (pending review) proposes that recall happens on every replayed user turn (natural recall), with Memora's questions measuring the end state and our own recall queries as a separate fallback arm if natural recalls are too rare.
3. **Hedges.** The proposal lists hedges as a probe type. Natural data has almost none, so hedge results come only from the synthetic subset (Stage 1b), reported separately and labelled synthetic (D09; E05).
4. **Probe count.** The proposal's power estimate assumes about 500 probed facts. The real count of probe-able facts is not yet known. Two sources exist: the dialogue (1,354 flagged turns in add and update sessions, E07, but many mix a fact with chatter) and Memora's own rubric (479 "should mention" sub-questions, E08). See section 5.
5. **External check.** Stretch goals now also include PersonaMem (from Lecture 7) alongside MemoryAgentBench FactConsolidation and a LongMemEval subset.
6. **Scoring pipeline.** Memora's own evaluation scripts need an OpenRouter key for their judges (a three-judge majority vote) and score answers from specific models (R02). We run local and keyless for now (D05), so we implement the FAMA formula and the yes/no sub-question protocol ourselves with our own judge. Our FAMA is comparable with Memora's Table 3 in formula only, not in judge or answering model (A8).

## 4. What we know about the data (Memora, weekly split)

*Written by Claude, Oct 16, from scripts E01–E06 · review: pending (whole section). Each bullet cites its evidence.*

- 10 personas, 1,551 sessions. `session_type` is a content category (activity, preference, goal, no_memory); the add/update/delete log is in `operation` and `operation_details`. The README's example `session_type` values (`memory_introduction`, `memory_update`) do not match the data. (E01, A1)
- Counts over all 10 personas: no_memory 519; activity add 514, update 58, delete 148; preference add 159, update 78, delete 45; goal add 30. (E01)
- About one third of sessions are `no_memory` filler, which works as haystack. (E01)
- 1,613 flagged turns (`share_memory: True`); each is one short sentence (median 106 characters, mean 117, max 401); at most 10 per session. Every activity, preference and goal session has at least one flagged turn and no `no_memory` session has any. (E04, A2)
- **The flag also marks instructions to forget** (README: "or, for forgetting sessions, to forget"). Flagged turns by session kind: add sessions 1,125 (activity 857, preference 238, goal 30), update sessions 229, delete sessions 259. So about 1,354 flagged turns can carry facts to remember, and 259 are things like "I no longer need to track X". Many fact turns mix the fact with chatter ("I just spent $3.66 on a coffee, and it got me thinking…"); update turns state changes ("update the budget from $800,000 to $850,000"). (E07, A2)
- The add session for `project_proposal_2` (creative_designer, session 0089) states a record richly: budget "€100,000" (the log stores 100000 with no currency), "12+ languages", "12 European locales", "maximum file size of 150KB per screen", phased timeline, named lead and stakeholders. Good material for numeric, unit and qualifier probes. (E03)
- Update session 0110 says "I have some changes to the timeline" and never states the change; only an added deliverable is spoken. This is one session; E06 checks it across all sessions. (E03, A3)
- Log values found verbatim in the dialogue (strict matching, so a lower bound): preference add 93%, preference update 100%, goal add 100%, activity add 52%, activity update 33%. Activity records are mostly paraphrased in the dialogue, and updates state only what changed. (E06, A2, A3)
- Memora's evaluation questions: 150 in total (50 per task type). The scored rubric is the `evaluation` block, not the evidence blocks (README): **735 yes/no sub-questions, 479 "should mention" (`memory_presence`) and 256 "should not mention" (`forgetting_absence`)**, 65–81 per persona. Reasoning questions have only "should mention" sub-questions (90); remembering has 260 + 150; recommending 129 + 106. The stated totals match the counts in every question. Sub-questions test presence of whole fields ("Does the response include the project title: …?"). (E08, A7)
- **FAMA** (Memora's headline metric, per the evals README, R02): per question, `FAMA = max(0, MPA − λ·(1 − FAA))`, where MPA is the share of "should mention" sub-questions answered correctly, FAA the share of "should not mention" sub-questions answered correctly, and λ = N_forget / (N_presence + N_forget). A Table 3 cell is the mean per-question FAMA × 100. Reasoning questions have no forget sub-questions, so λ = 0 for them. Memora's judging is a majority vote of three OpenRouter judges (`openai/gpt-4.1`, `anthropic/claude-haiku-4.5`, `google/gemini-2.5-flash`). Track 1 puts the whole history in the model's context (`qwen/qwen3-32b`, `anthropic/claude-sonnet-4.5`, `google/gemini-3-pro-preview`, `openai/gpt-5.2`, each with reasoning on and off); Track 2 evaluates memory agents (a_mem, langmem, mem_0, memobase, memos, nemori) answered by `gpt-4o-mini`. Read, not run. (R02, A8)
- **History length:** one weekly persona's full history is about 62–74k tokens (estimate: characters / 4; mean about 68k), over 145–163 sessions of at most about 900 tokens each. A full-history baseline therefore needs a model context comfortably above about 74k tokens plus the prompt; check the tokenizer and the `llama-server` context setting. (E09)
- The first count of evidence items (34 remembered, 252 forgotten) was an undercount: the evidence blocks take many shapes (`memory_items`, `content_data`, `session_history`, `goal_value`, …) and the counter matched only one of them. (E02, A5)
- **Other benchmarks considered as the main dataset (Oct 17; not a decision):** LoCoMo: 10 conversations of about 18k tokens, 1,986 questions, no add/update/delete information, 444 questions without an answer field (E10). LongMemEval: 500 questions, each with its own mostly separate history (about 122k tokens in the S file, about 6.6k in the oracle file), 78 knowledge-update questions, and `has_answer` turns that locate the evidence; answers are free-form text. Ingesting every S history separately would be about 500 × 122k ≈ 61 million tokens, roughly 90 times Memora's 683k tokens for all ten weekly personas (E11, E09). Both ask each question once, like Memora. (E10, E11, R03, R04) [Proposed: Claude · Written: Claude, 2026-10-17 · Review: pending]
- Full write-up: the project doc `memora-findings.md`.

## 5. Open questions and next steps

*Written by Claude, Oct 16 · review: pending (whole section).*

- [x] Run E01–E09 on the EVO-X2, commit the scripts with the stamped evidence files, and update the "Run" column in section 9 (done Oct 16: scripts `bb38eda`, evidence `72e7b8f`).
- [ ] Decide how to use Memora's own rubric (735 sub-questions, 479 "should mention"): at least run it once at the end of the timeline for comparability (FAMA), and decide whether its "should mention" items can also serve as probes.
- [ ] Extract facts from flagged turns of add and update sessions (not delete sessions) into probe facts (persona, session, date); apply the operations log in order; count usable facts per persona. Decides whether about 500 probes is realistic.
- [ ] Check A6 (file order matches `date`; session IDs unique per persona) before applying operations in order.
- [ ] Run E12 on the EVO-X2 and decide from it whether natural recall gives enough rewrite depth (D14, A9). If not, design the injected-query arm (how many recalls, at which points, query templates). A retrieval-based count (embedding model on the EVO-X2) would replace the string-match lower bound.
- [ ] Choose the agent model and the independent-family judge (both local for now); check what is installed.
- [ ] Measure real token counts with the tokenizer and set the `llama-server` context (`-c`) above the largest persona history (E09 estimates about 74k tokens) plus prompts; check the KV cache memory this needs on the EVO-X2.
- [ ] Implement `fama()` from the formula in R02 with a unit test, and implement the yes/no sub-question judging protocol against a local judge (A8).
- [ ] Stage 0: build the no-rewrite harness (drift should be near zero) and hand-label about 50 items to validate the judge.
- [ ] Decide the role of each benchmark: Memora as the main dataset (operations log, small histories), a LongMemEval subset (oracle or S) as main or as the external check, LoCoMo or neither. Needs: how many LongMemEval questions can be code-graded (numbers, names) rather than judge-graded; why 21 questions have no `has_answer` turn; what ingesting 500 separate histories would cost on the EVO-X2. Log the choice as a new decision.
- [ ] Check whether the `goal` sessions (30 in total) are worth using.
- [ ] Wait for the AWS credit details (Lecture 7) and decide whether to use any paid API.

## 6. Resources

*Written by Claude, Oct 16, from amounts Sam reported · review: pending (whole section).*

| Resource | Amount | Notes |
|---|---|---|
| Local: GMKtec EVO-X2 | 128 GB unified memory | llama.cpp (`llama-server`, port 8080), accessed over SSH |
| OpenRouter (Laude) | $200 | Course lifetime allowance; not set up |
| Tinker | $250 | Expires in one year; not set up |
| Gemini | $100 | Not set up |
| AWS (Laude) | $200 | Announced Oct 15; details pending |
| Google Cloud Free Trial | $300 | Ends Jan 7, 2027; no GPUs, cannot pay for AI Studio |

## 7. Limitations to state in the reports

*Written by Claude, Oct 16 · review: pending (whole section).*

- Memora's conversations are synthetic and short; facts are single sentences, so fidelity of long, nuanced records is not tested.
- Only 10 personas (3 development, 7 test), so persona-level variation is hard to estimate.
- Natural hedges are almost absent; the hedge result relies on a synthetic subset.
- Power is the main risk: differences below about 6 points may not be detectable at roughly 500 probes.
- The conversations are scripted: the user's turns do not react to what our agent says, so a rewritten memory can only affect later answers and the final questions, not the course of the conversation.
- Our FAMA uses our own judge and answering model, so it is comparable with Memora's published Table 3 in formula only.
- Local open-weight models may behave differently from frontier models; a second model size or one frontier check is a stretch goal.

## 8. Course references to cite

*Written by Claude, Oct 16 · review: pending (whole section).*

- Lecture 7 (Agent Memory Architectures): evaluation desiderata (realistic, long-context, challenging, verifiable); benchmarks LoCoMo, PersonaMem, LongMemEval-v2, MemoryArena. Verify details in the papers before citing.

## 9. Evidence register

Each piece of evidence is a script in `scripts/explore/` whose output is saved in `notes/evidence/` (see `scripts/explore/README.md`). *Written by Claude, Oct 16 · review: pending (whole section).*

**Run column:** E01–E06 began as ad hoc commands Sam ran on the EVO-X2 on Oct 15 (E06 is new). Claude re-created them as saved scripts and ran all nine on Oct 16 against a fresh shallow clone of Memora (commit `a6493188`), where the numbers matched Sam's Oct 15 output for E01, E02, E04 and E05. Sam then ran E01–E09 on the EVO-X2 with the scripts committed (code `bb38eda`, header "scripts clean"; outputs committed in `72e7b8f`). The stamped files in `notes/evidence/` are the record; the numbers below match them.

| ID | Script | Question | Result | Assumptions | Run | Used by | Status |
|---|---|---|---|---|---|---|---|
| E01 | `E01_session_operations.py` | What do `session_type` and `operation` contain? | 1,551 sessions; add/update/delete counts as in section 4; README's `session_type` examples do not match the data | A1 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D10 | Reproduced on EVO-X2; matches Claude's test-clone run |
| E02 | `E02_eval_question_evidence.py` | How many items do Memora's questions list? | 150 questions; the `{value, session_id}` counter found 34 remembered and 252 forgotten items but evidence has many other shapes, so the count is an undercount | A5 | Sam, Oct 16 (EVO-X2, `bb38eda`) | — (earlier claim withdrawn) | Reproduced on EVO-X2; shows A5 wrong; superseded by E08 for sizing |
| E03 | `E03_session_text_examples.py` | For one project, what does the dialogue say compared with the log? | Add session states the whole record in about 10 flagged turns; update session states only an added deliverable | A2, A3 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D10 | Three-session example only |
| E04 | `E04_flagged_turn_stats.py` | How many flagged turns, how long? | 1,613; median 106, mean 117 characters; max 10 per session; flags only in memory-bearing sessions | A2 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D10 | Reproduced on EVO-X2 |
| E05 | `E05_hedge_search.py` | How often do flagged turns hedge? | Loose search 151 (9.4%), mostly the preposition "about"; tight search about 7 real hedges ("estimated" 6, "approximately" 1) | A4 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D09 | Reproduced on EVO-X2; keyword search only |
| E06 | `E06_log_vs_dialogue.py` | How much of the log's content is spoken? | Verbatim in dialogue: preference add 93%, update 100%, goal 100%, activity add 52%, activity update 33% (lower bounds; paraphrases missed) | A2, A3 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D10 | Run on EVO-X2 |
| E07 | `E07_flagged_by_operation.py` | Which flagged turns are facts to remember, and which are instructions to forget? | Of 1,613: add sessions 1,125, update sessions 229, delete sessions 259, none in `no_memory`; update turns state changes ("from $800,000 to $850,000") | A2 | Sam, Oct 16 (EVO-X2, `bb38eda`) | D10 | Run on EVO-X2 |
| E08 | `E08_eval_rubric_counts.py` | How big is Memora's scored rubric? | 735 sub-questions (479 memory_presence, 256 forgetting_absence); reasoning questions presence only; stated totals match in all 150 questions | A7 | Sam, Oct 16 (EVO-X2, `bb38eda`) | — | Run on EVO-X2 |
| E09 | `E09_history_length.py` | How long is each persona's full history? | About 62–74k tokens per persona (characters / 4 estimate), 145–163 sessions each, longest session about 900 tokens | — | Sam, Oct 16 (EVO-X2, `bb38eda`) | — | Run on EVO-X2; token counts are estimates |
| E10 | `E10_locomo_structure.py` | What does LoCoMo contain? | 10 conversations of 11–22k tokens (mean about 18k), 19–32 sessions each; 1,986 questions in 5 numbered categories (meaning not documented in the README we read); 444 of the 446 category-5 questions have no `answer` field; 1,982 questions list evidence turn IDs, 9 references match no turn ID; no add/update/delete information | R04 | Sam, Oct 17 (EVO-X2, `59c3d81`) | — | Run on EVO-X2; data sha256 `79fa87e9…` (matches Claude's earlier download) |
| E11 | `E11_longmemeval_structure.py` | What does LongMemEval contain? | 500 questions: temporal-reasoning 133, multi-session 133, knowledge-update 78, single-session-user 70, single-session-assistant 56, single-session-preference 30 (30 abstention). `oracle` file: about 6.6k tokens of evidence-only history per question. `S` file: about 122k tokens per question (mean 122,324; range 113–128k), 38–62 sessions (mean 47.7), about 494 turns; 19,195 distinct session IDs across 500 questions, so histories are mostly separate. Overall mean 1.9 answer sessions and 1.8 `has_answer` turns per question; knowledge-update questions have 2.0 and 1.8. 21 questions have no `has_answer` turn (not yet explained) | R03 | Sam, Oct 17 (EVO-X2, `66e5d60`) | — | Run on EVO-X2; sha256 `821a2034…` (oracle), `d6f21ea9…` (S). Bug fixed Oct 17: the first run (`59c3d81`) truncated means and medians to integers; fixed in place and re-run, counts unchanged. M file (2.7 GB) not downloaded |
| E12 | `E12_later_mentions.py` | After a fact is introduced, how often do later sessions mention it again? | Not yet run on the EVO-X2. Claude's test run on a fresh clone (not stamped, preliminary): of 656 measurable facts, 41 (6%) are mentioned again in a later user turn outside update/delete sessions (preferences 25%; todos, calendar events, content records, expenses and step counts about 0%); 216 facts are later updated or deleted | A9 | Claude, Oct 17 (test clone; Sam's run pending) | D14 | Written, tested on a clone; strict string match, so counts are lower bounds |

**Reference documents** (read, not run; they can be wrong too, so claims that rest on them are assumptions):

| ID | Document | Used for |
|---|---|---|
| R01 | Memora `data/README.md` (pasted by Sam, Oct 15 and Oct 16) | Layout; meaning of `share_memory` (including "forgetting sessions"); that the `evaluation` block is the scored rubric |
| R02 | Memora `evals/README.md` (pasted by Sam, Oct 16) | FAMA formula; judges and answering models; what the evaluation scripts need (an OpenRouter key) |
| R03 | LongMemEval README (github.com/xiaowu0162/LongMemEval, summarised by Claude via web fetch, Oct 17; data from Hugging Face `xiaowu0162/longmemeval-cleaned`) | File variants (oracle, S, M), field names, question types, `has_answer`, MIT license |
| R04 | LoCoMo README (github.com/snap-research/locomo, summarised by Claude via web fetch, Oct 17) | Data path `data/locomo10.json`, fields (`qa`, `evidence`, `event_summary`); license not read |

## 10. Assumptions register

Interpretations of the data that decisions rely on. If one is revised, reopen the decisions that cite it (see "Reopening" above). *Written by Claude, Oct 16 · review: pending (whole section).*

| ID | Assumption | Basis | Evidence | Could be wrong if | Decisions | Status |
|---|---|---|---|---|---|---|
| A1 | `session_type` is a content category (activity, preference, goal, no_memory) and `operation` (add, update, delete, none) is the memory operation. | Values in the data; the README's examples (`memory_introduction`, `memory_update`) disagree with the data, so we trust the data. | E01 | Some sessions carry more than one operation in a way `operation` doesn't show. | D10 | Holds on current data; README inconsistent |
| A2 | Turns flagged `share_memory: True` are the facts told to the agent. | README: the flag marks information the user later expects the agent to remember, or in forgetting sessions to forget. | E03, E04, E06, E07 | Unflagged turns also state facts (they do: in session 0110 an unflagged turn names the project title), and flagged turns in delete sessions (259) are instructions to forget, not facts. | D10 | Partly supported; not "all and only". Use flagged turns from add and update sessions only. |
| A3 | In activity updates the log holds the whole record, while the dialogue states only what changed. | One example (E03) plus a coverage count (E06: about a third of logged values appear verbatim). | E03, E06 | Strict matching undercounts paraphrases; more of the record may be implied. | D10 | Supported; lower bound |
| A4 | Natural hedges are rare in flagged turns. | Keyword search for hedge words. | E05 | Hedges are expressed in words the search does not list. | D09 | Open |
| A5 | The count of evidence items that matched `{value, session_id}` reflects how many facts Memora's questions list. | First counter run Oct 15. | E02, E08 | — | none (the Oct 15 claim that the questions were too thin on this basis is withdrawn) | **Revised: wrong** (Oct 16). E02 shows many other evidence shapes; E08 gives the real rubric size (735 sub-questions). |
| A6 | `session_NNNN` file order is chronological, matches `date`, and session IDs are unique per persona. | README says sessions are stored in chronological order. | none yet | Order or IDs are inconsistent in some persona. | future: applying operations in order | Untested |
| A7 | The `evaluation` block (yes/no sub-questions) is what Memora actually scores; the `memory_evidence` and `forgetting_evidence` blocks are for auditing only. | R01 states this; R02 describes the same scoring by per-question yes/no sub-questions. | E08 (stated totals match the sub-question counts in all 150 questions) | The released evaluator scores something other than the `evaluation` block. | none yet | Supported by R01, R02 and E08; evaluator code not read |
| A8 | Memora's FAMA is the per-question formula in R02, averaged over questions; our implementation, with our own judge, follows the same formula. | R02. | none yet | The released code computes FAMA differently from the README. | none yet (needed for Stage 0 metrics) | Open; code not read |
| A9 | Facts come up again in later sessions often enough that natural recall during replay gives meaningful rewrite depth. | Sam's reconsolidation argument; frequency not measured. | E12 (pending Sam's run; Claude's test run suggests it is false for most activity facts) | Most facts are never mentioned again outside their own update or delete session. | D14 | Open; preliminary test run points against it |
