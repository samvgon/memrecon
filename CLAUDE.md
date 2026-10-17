# memrecon

CS329Z (Stanford, Engineering AI Agents) solo project by Sam Goncalves. Full title: *Keep, Discard, or Verify: Write Policies for Recall-Time Memory Rewrites in LLM Agents*. The proposal was submitted Oct 9, 2026. The midway report and final report use ICLR format, and the course wraps in December 2026.

## The question

Some agent-memory systems rewrite a memory at recall time so it suits the task at hand (JitMem, MemHarness, ReMe). None of the ones we found writes that rewrite back unchecked; TierMem writes back only after verifying against an immutable raw log. This project tests what happens when the agent **keeps its own rewrite unchecked**. It resembles human memory reconsolidation: retrieval makes a memory labile, which can keep it relevant but can also distort it.

**Hypothesis.** Persisting unchecked rewrites erodes fidelity, increasingly with rewrite depth, and verifying against the original recovers it at lower cost than re-reading.

- **Q1.** Does persisting the agent's unchecked rewrite cause more drift than discarding it?
- **Q2.** If so, does a check against the source make write-back safe?
- **Q3.** Can rewrite depth, or how much a memory has changed from its original, tell the agent when to run that check?

Each question is conditioned on the last. Stage 1 is informative whether or not drift appears.

## System design

One memory agent: it ingests sessions into memory, retrieves memories to answer each query, and rewrites what it retrieves for the task. **Only the write policy varies**; everything else (model, prompts, retrieval, budget) is held fixed. The write policy decides what is stored.

Every memory carries a **rewrite counter** and an **archived original**, kept for scoring whether or not the policy uses them. Each recall rewrites the latest version, so rewrite depth accumulates.

## Staged plan

| Stage | Conditions compared | Runs if |
|---|---|---|
| 0 (gate) | LLM judge vs. ~50 hand labels; no-rewrite control (drift ≈ 0) | Always; gates the rest |
| 1 (Q1) | Rewrite *discarded* (original only), *replaced* (overwrites original), or *added* (both kept); append-only; periodic summarization (no archive); verbatim duplicate-original control; full and source-filtered re-reading | Always |
| 1b (hedges) | The Stage 1 conditions repeated on a small **synthetic hedge subset**: 100–150 numeric facts rewritten to carry a hedge ("approximately", "estimated", "tentatively"), with the unhedged originals kept as a control. Reported separately and labelled synthetic. | Stage 1 works on natural facts; time permitting |
| 2 (Q2) | Write back always; only on a detected conflict (no source check); after verifying against the original (TierMem-style) | Stage 1 finds drift |
| 3 (Q3) | Verify at depth k; when distance from original passes a threshold; always; periodically (full-pipeline cost) | Drift is gradual; Stage 2 works |

Stages 0–1 target the midway report and Stages 2–3 the final. Stretch goals: a second model size, MemoryAgentBench's FactConsolidation task, a LongMemEval subset, and PersonaMem as an external check (code-graded multiple choice with evolving preferences). Verify the details of any benchmark before citing it.

**Do Stage 0 first.** Nothing after it means anything if the measurement is wrong. Do not start Stage 1 until the harness shows near-zero drift with no rewriting and the judge agrees with the hand labels.

## Data and evaluation

- **Core data: Memora** (Uddin et al., 2026). Simulated persona conversations; a weekly persona averages 155 sessions with a per-session add/update/delete log, giving per-fact ground truth for repeated recall. Check this structure on the real download before building on it.
- **What Memora actually contains (checked Oct 15, 2026; see `notes/` and the project's `memora-findings.md`):**
  - About 1,550 weekly sessions over 10 personas. The add/update/delete log is in each session's `operation` and `operation_details`. About a third of sessions are `no_memory` filler.
  - For activity updates the log's `operation_details` holds the whole record while the dialogue states only what changed (about a third of logged values appear verbatim; E06). **Build probes from the dialogue, mainly the turns flagged `share_memory: True`** (1,613 of them; short single sentences), not from `operation_details`. The flag is not strictly "all and only" the told facts: an unflagged turn can state one, and flagged turns in delete sessions (259 of the 1,613) are instructions to forget, so use flagged turns from add and update sessions only (E07; A2).
  - Apply the add/update/delete operations in order to get the latest legitimate state, and score drift against that. Facts never updated or deleted are the cleanest probes.
  - Memora asks each of its 15 questions per persona once, so rewrite depth can't build up from them. We generate our own recall queries from the flagged facts and re-ask them at several points in the timeline. Memora's scored rubric is the `evaluation` block: 735 yes/no sub-questions over the 10 weekly personas, 479 "should mention" and 256 "should not mention" (E08, A7). The evidence blocks are for auditing only; an early count of 34 evidence items was an undercount (E02, A5).
  - Natural hedges are almost absent (about 7 of 1,613 flagged turns). The main probe types are therefore numbers and units, qualifiers ("12+", "maximum", currency), named entities and conditions, all code-gradable. Hedge probes come from the synthetic subset in Stage 1b.
- **Scoring (Memora's evals README, read not run; A8):** per question, `FAMA = max(0, MPA − λ·(1 − FAA))`, where MPA and FAA are the shares of "should mention" and "should not mention" yes/no sub-questions answered correctly and λ = N_forget / (N_presence + N_forget). Memora's own scripts need an OpenRouter key and use a three-judge vote, so **do not run them**; we implement the formula and the sub-question protocol ourselves with a local judge, and our FAMA is comparable with Memora's Table 3 in formula only.
- **Persona split:** `notes/split.json` (seed 329, 3 development and 7 test personas). Frozen once committed; never change it after seeing results.
- **Metrics:** benchmark accuracy (including Memora's forgetting-aware FAMA); fidelity via probes against the original fact (numbers and units, qualifiers and conditions, hedges via Stage 1b); tokens per use; a failure taxonomy (lost detail, lost hedge, invented detail).
- **Grading:** code where possible. Hedges go to an LLM judge from a *different model family* than the agent, scoring each item twice (re-judging can flip labels).
- **Split personas into development and held-out test sets on day one**, before looking at any results. Prompts and the choice of k are tuned on development only. Report test-set results only.
- **Pre-registered test:** compare each persisted state with discarded on fidelity and accuracy at matched context tokens, paired. About 500 probed facts detects roughly a 6-point difference. Power is the main risk.
- **Success:** a write policy that matches the fidelity of discarding at lower cost than always verifying or re-reading.

## Compute and costs

- Runs on the GMKtec EVO-X2 (Ubuntu, 128 GB unified memory, reached over SSH). Local inference is llama.cpp's `llama-server` (OpenAI-compatible, port 8080). Check which models are actually installed before assuming; earlier plans were Qwen3-30B-A3B-Instruct-2507 Q4_K_M as the default, Qwen3-8B Q4_K_M, and nomic-embed-text-v1.5 Q8_0 for embeddings.
- **Local models only for now. Never make a paid API call unless Sam explicitly says so in that session.** No API keys are set up yet. If a key is ever added, it goes in `.env` (git-ignored), never in code or the repo.
- One weekly persona's full history is about 62–74k tokens (E09 estimate, characters / 4). The full-history baseline needs the `llama-server` context (`-c`) comfortably above that plus the prompt; check the real tokenizer count and the memory the KV cache needs.
- Make the endpoint configurable: read base URL, model name and key from environment variables, defaulting to `http://localhost:8080/v1` with a dummy key. Switching to OpenRouter or Gemini later should be a `.env` change only.
- Credits that exist (for later): OpenRouter $200 (course lifetime), Tinker $250, Gemini $100, AWS $200 (pending), plus a Google Cloud $300 trial that ends Jan 7, 2027 and can't pay for AI Studio or use GPUs.

## Repo layout and conventions

- Python with `uv`. Add dependencies with `uv add`; run scripts with `uv run`.
- `data/` and `results/` are git-ignored. Never commit datasets, run outputs, model files or `.env`.
- `scripts/` for runnable entry points; keep the experiment logic in the package so it is testable.
- **Reproducibility:** pin and log model names and versions, seeds, prompts, and the write-policy config for every run. Save raw agent and judge outputs, not only the scores.
- Keep the write policy a small swappable component with one interface, so adding a condition from the table above doesn't touch the rest of the agent.
- Run multiple seeds and report paired comparisons with confidence intervals; differences may be small.

## Durability: never lose work to an interruption

Assume any run can be killed at any moment (SSH drop, power cut, network loss, a crash). Design so that costs at most one unit of work.

- **Unit of work = one persona × policy × seed × session.** Each unit writes its result to its own file as soon as it finishes. Write to a temp file, flush, then rename, so a crash never leaves a half-written file.
- **Resumable by default.** On startup, skip any unit whose valid output already exists. Unit IDs come from a hash of the config, so reruns are idempotent. Every runner must support this.
- **Save raw output before parsing.** Write each agent and judge response (request and response) to disk before any parsing or scoring, so a parsing bug never forces a rerun.
- **Snapshot memory state after every session**, including rewrite counters and archived originals.
- **Cache model calls** by a hash of prompt, model, and settings. This is a resume aid, not proof of reproducibility, because local models are not perfectly deterministic.
- **Long jobs run inside `tmux`** (or a systemd service), never in a bare SSH session. Log to a file as well as the terminal.
- **Research and long Claude responses:** write findings to `notes/` as they are found, one short dated file per topic, not at the end of the task. Do the same for long literature checks or deep-research runs.
- **Commit small and often**, and push. Commit summary files (aggregate metrics, notes), not raw `results/`.
- **Back up `results/`** (git-ignored) by rsync to another machine or a bucket, and after any long run.
- **Portability:** keep `uv.lock` committed, keep the model endpoint in `.env`, and have `scripts/download_data.sh` fetch every dataset. A Dockerfile for the harness (not the model server) can wait until the project first moves to another machine.

## Working rules for Claude

- Don't invent results, numbers, citations or dataset details. If something hasn't been run or checked, say so.
- Check the real structure of a downloaded dataset before writing code that depends on it.
- Before running anything long or expensive, say what it will do and roughly how long it will take.
- Don't tune anything on the held-out personas.
- **Any finding about the data that a decision will rely on must come from a numbered script in `scripts/explore/`, not an ad hoc terminal command.** Read `scripts/explore/README.md` first. Write one script per question (`E<next number>_<name>.py`), run it with `uv run python scripts/explore/run.py E<number>` so the output is saved to `notes/evidence/` with the code and data versions, commit both together, and add it to the evidence register (section 9 of `notes/decision-log.md`). List every interpretation of a field it depends on in the assumptions register (section 10). Never edit a script a decision already cites; write the next-numbered script and say which one supersedes it. Decisions cite evidence and assumptions by ID. If an assumption is found wrong, mark it `revised`, set the Review of every decision that cites it to `reopened (A#, date)`, and tell Sam.
- Write findings to `notes/` as short dated markdown files when something is learned that the next session would otherwise have to rediscover.
- **Keep `notes/decision-log.md` current.** Whenever a decision is made, something is found out about the data, or the plan deviates from the submitted proposal, append a dated entry (decision log table, deviations list, data facts, open questions). Do not rewrite old entries; add a new one if a decision changes. This is the source for the midway and final reports. **Follow the provenance rules at the top of the log:** give every new entry a Proposed by (`Sam`, `Claude` or `Both`), Written by (`Claude, <date>`) and Review `pending`. Never mark an entry as reviewed and never edit an entry Sam has reviewed; only Sam sets Review to `Sam, <date>`, and that also means she confirms the decision. Do not record Proposed by as `Sam` unless the idea was hers. **A Claude-proposed entry that is still pending review is a recommendation, not a decision: don't build on it as settled without asking Sam, unless it has already been acted on, and say so in the entry.** When Sam asks what is waiting for review, run `grep -n "review: pending\|Review: pending\|| pending |" notes/decision-log.md`.
