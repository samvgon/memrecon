# memrecon decision log

Running record of decisions, findings and deviations from the submitted proposal, kept so the midway and final reports can be written from it. **Append new entries as they happen, with the date. Do not rewrite old entries; if a decision changes, add a new entry that says so.**

Proposal: *Keep, Discard, or Verify: Write Policies for Recall-Time Memory Rewrites in LLM Agents* (submitted Oct 9, 2026). Solo project, CS329Z, Stanford.

---

## 1. Where the project stands

- Stage 0 (measurement gate): not started.
- Stage 1 (Q1: keep vs discard the rewrite): not started.
- Data: Memora weekly split downloaded and inspected (Oct 15).
- Compute: local models only; no API keys set up yet.

## 2. Decision log (newest last)

| Date | Decision | Why |
|---|---|---|
| Oct 8 | Committed to the memory-reconsolidation topic; solo project over the full quarter. | Lecturer found it interesting and said data, evals and metrics had to be thought through in the proposal. |
| Oct 8–9 | Narrowed the question to whether an agent should persist a recall-time rewrite rather than discard it. Staged design (Stages 0–3). | Gives a clean Q1 → Q2 → Q3 chain where each stage is informative whether or not drift appears. |
| Oct 9 | Proposal submitted. Citation-chain tracer dropped as a testbed. | Not needed for the staged design. |
| Oct 15 | New repo `memrecon` (private), not a rename of `citation-tracer`. | Different project and history; a rename would carry stale commits and notes. |
| Oct 15 | No paid API calls and no API keys for now; local llama.cpp server only. Model endpoint set via environment variables. | Protect the credits; switching later should be a `.env` change only. |
| Oct 15 | Durability rules: one result file per unit of work, atomic writes, resume by skipping, raw outputs saved before parsing, `tmux` for long jobs, backups of `results/`. | Runs can be interrupted (SSH drop, power, network). |
| Oct 15 | Docker deferred until the project first moves to another machine. Portability via `uv.lock`, `.env` endpoint and a data download script. | llama.cpp with Vulkan in Docker needs GPU passthrough; the harness only needs an endpoint URL. |
| Oct 15 | Dev/held-out split fixed: seed 329, 3 development personas (`financial_analyst`, `management_consultant`, `sales_manager`), 7 test personas. Saved in `notes/split.json`. | Fixed before any results were seen. **Frozen.** Never retune on test personas. |
| Oct 15 | Hedge probes: start without a separate hedge probe (option A); add a synthetic hedge subset as **Stage 1b** (option B). | Natural hedges are almost absent in Memora (about 7 of 1,613 flagged turns). |
| Oct 15 | Probes are built from `share_memory: True` turns, not from `operation_details`. | The log holds details the agent was never told. |

## 3. Deviations from the submitted proposal

State these openly in the reports.

1. **Probe source.** The proposal says Memora's log gives per-fact ground truth for repeated recall. In practice we build probes from the flagged turns and use the log only to find the latest legitimate state of each fact.
2. **Recall schedule.** Memora has 15 questions per persona, each asked once. We generate our own recall queries from the flagged facts and re-ask them at several points in the timeline, so rewrite depth can accumulate.
3. **Hedges.** The proposal lists hedges as a probe type. Natural data has almost none, so hedge results come only from the synthetic subset (Stage 1b), reported separately and labelled synthetic.
4. **Probe count.** The proposal's power estimate assumes about 500 probed facts. The real count after applying updates and deletes is not yet known (see section 5).
5. **External check.** Stretch goals now also include PersonaMem (from Lecture 7) alongside MemoryAgentBench FactConsolidation and a LongMemEval subset.

## 4. What we know about the data (Memora, weekly split)

- 10 personas, about 1,550 sessions. `session_type` is a category (activity, preference, goal, no_memory); the add/update/delete log is in `operation` and `operation_details`.
- Counts over all 10 personas: no_memory 519; activity add 514, update 58, delete 148; preference add 159, update 78, delete 45; goal add 30.
- About one third of sessions are `no_memory` filler, which works as haystack.
- 1,613 flagged turns (`share_memory: True`); each is one short sentence (median 106 characters, mean 117); at most 10 per session.
- The add session for `project_proposal_2` (creative_designer, session 0089) states a record richly: budget "€100,000" (the log stores 100000 with no currency), "12+ languages", "12 European locales", "maximum file size of 150KB per screen", phased timeline, named lead and stakeholders. Good material for numeric, unit and qualifier probes.
- Update session 0110 says "I have some changes to the timeline" and never states the change; only an added deliverable is spoken. Hence the rule to probe from flagged turns.
- Memora's evaluation questions: 150 in total; a first count found 34 memory items and 252 forgotten items, but the counter may undercount.
- Full write-up: the project doc `memora-findings.md`.

## 5. Open questions and next steps

- [ ] Extract flagged turns into probe facts (persona, session, date); apply the operations log in order; count usable facts per persona. Decides whether about 500 probes is realistic.
- [ ] Decide the recall schedule (how many recalls, at which points, query templates).
- [ ] Choose the agent model and the independent-family judge (both local for now); check what is installed.
- [ ] Stage 0: build the no-rewrite harness (drift should be near zero) and hand-label about 50 items to validate the judge.
- [ ] Check whether the `goal` sessions (30 in total) are worth using.
- [ ] Re-count the evaluation-question evidence with a looser counter.
- [ ] Wait for the AWS credit details (Lecture 7) and decide whether to use any paid API.

## 6. Resources

| Resource | Amount | Notes |
|---|---|---|
| Local: GMKtec EVO-X2 | 128 GB unified memory | llama.cpp (`llama-server`, port 8080), accessed over SSH |
| OpenRouter (Laude) | $200 | Course lifetime allowance; not set up |
| Tinker | $250 | Expires in one year; not set up |
| Gemini | $100 | Not set up |
| AWS (Laude) | $200 | Announced Oct 15; details pending |
| Google Cloud Free Trial | $300 | Ends Jan 7, 2027; no GPUs, cannot pay for AI Studio |

## 7. Limitations to state in the reports

- Memora's conversations are synthetic and short; facts are single sentences, so fidelity of long, nuanced records is not tested.
- Only 10 personas (3 development, 7 test), so persona-level variation is hard to estimate.
- Natural hedges are almost absent; the hedge result relies on a synthetic subset.
- Power is the main risk: differences below about 6 points may not be detectable at roughly 500 probes.
- Local open-weight models may behave differently from frontier models; a second model size or one frontier check is a stretch goal.

## 8. Course references to cite

- Lecture 7 (Agent Memory Architectures): evaluation desiderata (realistic, long-context, challenging, verifiable); benchmarks LoCoMo, PersonaMem, LongMemEval-v2, MemoryArena. Verify details in the papers before citing.
