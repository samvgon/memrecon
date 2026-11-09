# memrecon

**Keep, Discard, or Verify: Write Policies for Recall-Time Memory Rewrites in LLM Agents.** CS329Z (Stanford, Engineering AI Agents), solo project by Sam Goncalves, Autumn 2026.

Some agent-memory systems rewrite a memory when they recall it, so it fits the task at hand. This project tests what happens when the agent **keeps that rewrite unchecked**: one memory agent replays simulated conversations from the Memora benchmark, and only its *write policy* changes between conditions (discard the rewrite, replace the original, keep both, verify against the original, ...). Everything else (models, prompts, retrieval, settings) is held fixed.

**Status (Nov 8, 2026, midway):** Stage 0 is complete. The agent runs end to end (ingestion in two arms, recall with select-then-rewrite, the write policies discard / replace / keep, answering Memora's questions, the judge and FAMA), and every measurement has passed its pre-set check on the development personas: discard leaves memory byte-identical to ingestion alone (E38), the judge agrees with hand labels (E39: 96%, kappa 0.92, no flips), and the code grader agrees with hand labels on a held-out sample (E42: 97%, kappa 0.95). Stage 1 (keep vs. discard) on the 7 test personas is next. Every decision, with who proposed it, its evidence and whether it has been reviewed, is in [`notes/decision-log.md`](notes/decision-log.md).

## Contents

| Path | What it is |
|---|---|
| `config/models.env` | Which model serves each role (agent, judge, ...) when running locally. The default. |
| `config/models.openrouter.env` | The same models hosted on OpenRouter, for running without the local setup. |
| `scripts/serve.sh` | Starts, stops and checks the local model servers. |
| `scripts/check_models.py` | Sends one test question to each model. |
| `scripts/monitor.py` | Live view of a run: machine and GPU memory, model servers, token use, the agent's memory, disk. |
| `scripts/download_data.sh` | Downloads the Memora dataset at the pinned commit. |
| `scripts/make_split.py` | Recreates or checks the frozen development/test persona split (`notes/split.json`). |
| `scripts/run_stage0_control.py` | Stage 0's control check: discard leaves no trace (ingest only vs recall + discard, compared after every session). |
| `scripts/run_stage0_answers.py` | Answers Memora's questions from the end-of-history memory of the Stage 0 control run (D60), for the judge check and FAMA. |
| `scripts/make_judge_sample.py` | Makes the frozen, stratified sample of 70 judge items for hand labelling (D21, D60). |
| `scripts/run_stage0_judge_check.py` | Stage 0's judge check (D21, D61): the judge against the hand labels, each item judged twice; `--set tune` to adjust the prompt, `--set check` once with it frozen. |
| `scripts/make_grader_sample.py` | Makes the frozen, stratified sample of code-grader items (a probe and five memories from the end-of-history store) for hand labelling (D21). |
| `scripts/run_stage0_grader_check.py` | Stage 0's code-grader check (D21): the code's found / not found against the hand labels. |
| `scripts/make_grader_recheck.py` | Lists every code-grader verdict that D62's fixes change in the end-of-history stores, for hand labelling. |
| `scripts/explore/` | Numbered evidence scripts (E01–E43, except those produced by the runners above); their saved outputs are in `notes/evidence/`. See its README. |
| `src/memrecon/llm.py` | The only way experiment code calls a model. |
| `src/memrecon/embed.py`, `src/memrecon/retrieval.py` | Embeddings (cached) and retrieval: which memories are relevant to a turn. |
| `src/memrecon/memory.py` | The memory store: memories, their lineage and depth, snapshots and the event file. |
| `src/memrecon/ingest.py` | Saving a finished session into memory: one-call (note, log, oracle arms) and two-step (extract, then decide per fact). |
| `src/memrecon/prompts/` | Every prompt sent to a model, one file per version (never edited once used in a run; older versions stay because the evidence that ran them cites them). **In use:** note ingestion `extract_note_v2` + `decide_note_v3` (D52), log ingestion `ingest_log_v1`, recall `select_v1` + `rewrite_v5b` (D58), answering `answer_v1` (D60), judge `judge_v1` (D61), answer key `probe_extract_v4` + `probe_split_v1` (D50). |
| `src/memrecon/recall.py` | Recall at replay time (retrieve, select, rewrite) and the write policies (discard, replace, keep). |
| `src/memrecon/stage0.py`, `src/memrecon/memora.py` | Stage 0's control check; reading Memora for runners. |
| `src/memrecon/probes.py` | The drift measure's answer key: the probe extractor, the table of what is true when, and what gets graded. |
| `src/memrecon/monitor.py` | Readers for the monitor, and the memory guard runs call before each unit of work. |
| `src/memrecon/answer.py` | Answering Memora's questions from memory (top 50 memories, newest first). |
| `src/memrecon/judge.py` | The judge for Memora's yes/no sub-questions, and Cohen's kappa. |
| `src/memrecon/fama.py` | Memora's FAMA score. |
| `tests/` | Tests (`uv run pytest`). |
| `docs/data-dictionary.md` | What the project's terms mean (arm, log arm, audit log, depth, record, probe…) and what every table, file and column holds. |
| `notes/` | Decision log, evidence, notes. |
| `data/`, `results/` | Datasets and run outputs. Not in git. |

## Setup

Needs [uv](https://docs.astral.sh/uv/) and git. uv installs the right Python (3.14) itself.

```bash
git clone https://github.com/samvgon/memrecon.git
cd memrecon
uv sync                        # installs the package and the test tools
scripts/download_data.sh       # Memora into data/memora, pinned to commit a6493188
uv run pytest                  # should pass without any model running
```

Then choose **one** way to run the models: locally (A) or hosted (B).

### A. Local models (how all of the project's own runs are done)

The project's runs use a GMKtec EVO-X2 (128 GB unified memory, Ubuntu) with llama.cpp. The two model files are about 39 GB together, and the agent's long context needs more memory on top of that, so plan for a machine with well over 48 GB of GPU-usable memory.

1. **Build llama.cpp** at the commit used here (`9e2a04a`, build 717), with the GPU backend for your machine (this project uses Vulkan):

   ```bash
   git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
   cd ~/llama.cpp && git checkout 9e2a04a
   cmake -B build -DGGML_VULKAN=ON && cmake --build build -j
   ```

   `scripts/serve.sh` expects `~/llama.cpp/build/bin/llama-server`; set `LLAMA_SERVER` to use another path.
2. **Download the model files** into the Hugging Face cache, at the exact revisions used. `config/models.env` points at these paths.

   ```bash
   uvx --from huggingface_hub hf download unsloth/Qwen3.6-35B-A3B-GGUF Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --revision a483e9e6cbd595906af30beda3187c2663a1118c
   uvx --from huggingface_hub hf download unsloth/gemma-4-26B-A4B-it-GGUF gemma-4-26B-A4B-it-UD-Q4_K_M.gguf --revision c099eb48e663fd284577b04978a94ffccb261841
   uvx --from huggingface_hub hf download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf --local-dir ~/models
   ```

   Check each file against its sha256 (`sha256sum <file>`): the two chat models in the table below; the embedding model `3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`. The small agent (stretch goal) is listed in `config/models.env` and not needed yet.
3. **Start the servers** (each runs in its own tmux session and logs to `~/logs/<role>.log`):

   ```bash
   scripts/serve.sh agent
   scripts/serve.sh judge
   scripts/serve.sh embed         # retrieval (nomic-embed-text-v1.5, port 8082)
   scripts/serve.sh status        # wait until all three say "healthy" (a minute or two)
   uv run python scripts/check_models.py
   ```

### B. Hosted models (OpenRouter)

For anyone without the local setup. **These calls cost money** (the two models are cheap, but a full run makes many calls), so the client refuses them unless you opt in.

```bash
cp .env.example .env
```

Then in `.env`, set:

```text
MEMRECON_MODELS=config/models.openrouter.env
OPENROUTER_API_KEY=<your key>
MEMRECON_ALLOW_PAID=1
```

and check: `uv run python scripts/check_models.py`. No servers are needed. `.env` is git-ignored; never commit a key. To cap spending, also set `MEMRECON_BUDGET_USD` (e.g. `5`): the client stops making paid calls once the recorded spend reaches it. `uv run python -m memrecon.llm` shows what has been spent.

Hosted results will be **close to but not identical with** local ones: OpenRouter's hosts don't run the same 4-bit files (and don't state their precision), the serving software differs, and seeds may be ignored. Each role is pinned to one hosting company (DeepInfra) with no fallback, so a run doesn't switch hosts midway; if it is down, change `_HOST` in `config/models.openrouter.env`. Every saved call records which host served it.

## How the models are run

Code never names a model; it asks for a **role**. The settings below apply to **every call**, in both setups, unless a row says otherwise.

| | Agent | Judge |
|---|---|---|
| Job | Ingests sessions, selects and rewrites recalled memories, answers Memora's questions | Builds the drift measure's answer key (probe extraction) and judges Memora's yes/no sub-questions (different model family from the agent; checked against hand labels, E39) |
| Model | Qwen3.6-35B-A3B (Alibaba, Apr 2026; mixture of experts, ~3B active) | Gemma 4 26B-A4B-it (Google, Apr 2026; mixture of experts, ~3.8B active) |
| Weights (local) | `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (unsloth, 4-bit), sha256 `ac0e2c1189e055faa36eff361580e79c5bd6f8e76bffb4ce547f167d53e31a61` | `gemma-4-26B-A4B-it-UD-Q4_K_M.gguf` (unsloth, 4-bit), sha256 `f2c28b3dc4776931ac6f879e11f203dec637ea0f14267a86ec8f6165f63f293f` |
| Hosted ID | `qwen/qwen3.6-35b-a3b` via DeepInfra | `google/gemma-4-26b-a4b-it` via DeepInfra |
| **Reasoning ("thinking")** | **Off** | **Off** |
| Temperature | 0 (greedy) | 0 (greedy) |
| Seed | 0 | 0 |
| Max output tokens | 1024 unless a step sets its own; each saved call records the value used | same |
| Context (local) | 131,072 tokens (a whole persona history, about 62–74k tokens, fits with room for prompts) | 16,384 tokens |
| Requests at a time | 1 per server (`-np 1`) | 1 |

**Reasoning is off for both models, on every call, and this is enforced, not just requested.** Both models think by default. The client turns thinking off in the way each provider expects (llama.cpp: `chat_template_kwargs: {"enable_thinking": false}`; OpenRouter: `reasoning: {"effort": "none"}`), callers can't override it, and the client **rejects any response that still contains reasoning text**. We checked this by hand on Nov 5: with thinking off, both models answer directly; without the setting, the judge spent its whole output budget on reasoning and gave no answer (decision D17).

**Other sampling settings** (top-p, top-k, penalties) are left at the server's defaults; at temperature 0 llama.cpp picks the most likely token, so they have no effect locally. Hosted providers may treat temperature 0 slightly differently.

**Local server settings** (all from `config/models.env` via `scripts/serve.sh`): llama.cpp build 717 (`9e2a04a`), Vulkan backend, all layers on the GPU (`-ngl 99`), the model's own chat template (`--jinja`), no vision encoder (`--no-mmproj`), and no carrying of earlier reasoning into later turns (`--no-reasoning-preserve`).

**Every call is saved.** The exact request (model name, messages and every setting above) and the raw response are written to `results/raw_calls/<role>/` before anything reads them, together with the provider, host and timing.

**Identical requests are answered from that store** instead of calling the model again, at no cost (the same idea as the HW1 harness's cache). This lets an interrupted run resume and makes reruns of unchanged code free, but it is not a guarantee of identical answers on a fresh run, since models are not perfectly deterministic even at temperature 0. Answers that were cut off, empty, failed or rejected for reasoning are never reused, and never overwritten: before a retry, the earlier attempt is renamed `<key>.attempt<N>.json`.

**Cut-off answers are errors.** If a model stops at its output limit, the client raises `OutputTruncated` instead of returning the partial text, since a cut-off rewrite would look like memory drift. The step that hit the limit needs a larger `max_tokens`.

**Ledger.** Every call that reaches a model (not cache hits) is appended to `results/raw_calls/ledger.jsonl` with its role, model, host, tag, token counts, time, cost (0 for local calls; OpenRouter's reported cost for hosted ones) and outcome (`ok`, `empty`, or why it was rejected: `reasoning`, `truncated`, `http_error`, `unreadable`). Rejected calls count towards the budget, since they were paid for; cost comparisons between systems should leave them out or report them separately. `uv run python -m memrecon.llm` prints calls, tokens and spend by role and tag; the token counts also give the "tokens per use" metric.

**Audit log.** The raw calls, attempts and ledger above, the memory store's snapshot after every session, and an append-only **event file** recording every memory operation in order (note, log, recall, keep, replace, forget, and which memories were shown to the model) together make up the audit log. None of it is ever shown to the model, which sees only memory text (decision D28). (The "log" arm is different: there, the agent's memories are the user's own words.)

## What's frozen

Everything fixed before results were looked at, in one place. Each row's decision in `notes/decision-log.md` gives the reasons and evidence.

| What | Value | Where | Decision |
|---|---|---|---|
| Persona split | Seed 329. Development (all tuning): `financial_analyst`, `management_consultant`, `sales_manager`. Test (reported results only): the other 7 | `notes/split.json` (`scripts/make_split.py` recreates or checks it) | D08 |
| Data | Memora, weekly split, commit `a6493188` | `scripts/download_data.sh` | — |
| Models | Agent Qwen3.6-35B-A3B and judge Gemma 4 26B-A4B-it (both `UD-Q4_K_M`), embeddings nomic-embed-text-v1.5 `Q8_0`; files checked by sha256; llama.cpp build 717 | `config/models.env`, [How the models are run](#how-the-models-are-run) | D15–D17 |
| Model calls | Temperature 0, seed 0, thinking off. Checks that ask again use seed 1 (judge flip rate, E37, E43's fresh run) | `src/memrecon/llm.py` | D17, D18 |
| Answer key | Extractor `probe_extract_v4`, then the split pass `probe_split_v1`; grading rules as amended | `src/memrecon/probes.py` | D50, D51, D62 |
| Note ingestion | Two-step v3: `extract_note_v2`, then `decide_note_v3` | `src/memrecon/stage0.py` | D52 |
| Retrieval at recall | Top 3 live memories, similarity ≥ 0.70 | `src/memrecon/recall.py` | D37 |
| Recall prompts | `select_v1` (memories listed newest first), `rewrite_v5b` | `src/memrecon/recall.py` | D58, D59 |
| Answering and judging | `answer_v1` (top 50 memories, newest first), `judge_v1` | `src/memrecon/answer.py`, `src/memrecon/judge.py` | D60, D61 |
| Stage 0 configuration | Hash `a0018ce663` (the configuration E38 ran) | `src/memrecon/stage0.py` | D19, D26 |
| Stage 0 pass criteria | Control: memory identical to ingestion alone. Graders: agreement ≥ 90%, Cohen's kappa ≥ 0.7; judge flips ≤ 5% | `notes/2026-11-05-stage0-pass-criteria.md` | D19, D21 |
| Hand-label samples | Judge: seed 21, 70 items (20 tune, 50 check). Code grader: seed 40 (v2), seed 63 (v3, held out after D62) | `notes/labels/` and its README | D21 |

## Reproducing Stage 0

Run from the repository root with the agent, judge and embedding servers up (setup A or B). Each step writes to `results/` (not in git) and the steps marked *evidence* save their output to `notes/evidence/`. Long steps belong in `tmux`. Times are on the EVO-X2; a cached rerun is much faster.

```bash
# 1. The answer key for the drift measure (probe extraction on the judge model, then the split pass)
uv run python scripts/explore/run.py E30            # extractor v4 on the development personas
uv run python scripts/explore/run.py E31            # split pass
# 2. The ingestion comparison that froze the note arm's ingestion (D52)
uv run python scripts/explore/run.py E32            # E33, E34 likewise
# 3. Stage 0 control: recall + discard leaves memory identical to ingestion alone (D19, D26)
uv run python scripts/run_stage0_control.py         # ~30 min fresh, ~12 min on cached ingestion; report in results/stage0/control/<config>/report.md (E38)
# 4. Answers to Memora's questions from the end-of-history memory, and the judge check (D21, D60, D61)
uv run python scripts/run_stage0_answers.py
uv run python scripts/run_stage0_judge_check.py --set check      # E39; uses the hand labels in notes/labels/
# 5. The code-grader checks (D21, D62)
uv run python scripts/run_stage0_grader_check.py                 # E40
uv run python scripts/run_stage0_grader_check.py --sample 2026-11-08-grader-sample-v3.jsonl --labels 2026-11-08-grader-labels-v3-resolved.jsonl   # E42
# 6. Stage 0's reported numbers: probe survival, ingestion loss, determinism (D20)
uv run python scripts/explore/run.py E43            # MEMRECON_E43_FRESH=0 skips the slow fresh re-ingestion
```

The samples and hand labels used by steps 4 and 5 are committed in `notes/labels/` (with how each was made in its README), so the checks rerun exactly; drawing new samples (`scripts/make_judge_sample.py`, `scripts/make_grader_sample.py`) refuses to overwrite a frozen one. Because local models are not perfectly deterministic, fresh calls can differ from ours (E37); the call cache makes reruns on the same machine exact.

## Watching a run

```bash
tmux new -s monitor
uv run python scripts/monitor.py      # refreshes every 5 s; --once prints once; --raw-metrics lists every server metric
```

It shows, separately: **the machine** (system memory, GPU memory, each model server's throughput and load from its `/metrics` page, which `scripts/serve.sh` turns on); **token use** over the last 5 minutes from the ledger, including how many tokens of memory the prompts carried; **the agent's memory as it would be in real use** (live memories, their size and depths, from the summary written next to each snapshot); and **experiment overhead** (the size of `results/`: audit log, cache, snapshots). The model servers reserve their memory when they start, so memory is most at risk when a server starts, not mid-run. Runs also call a memory guard before each unit of work: if available memory is below `MEMRECON_MIN_FREE_GB` (default 8 GB), they pause until it recovers, and stop cleanly after an hour (a resumed run picks up where it stopped).

## Rules this project follows

- **No paid API calls by default** (decision D05): the client refuses to call a paid provider unless `MEMRECON_ALLOW_PAID=1` is set in `.env`, and `MEMRECON_BUDGET_USD` caps the spend.
- **Frozen persona split:** 3 development personas for building and tuning, 7 test personas for reporting (`notes/split.json`, seed 329). Nothing is tuned on test personas.
- **Every data finding comes from a numbered script** in `scripts/explore/`, with its output saved in `notes/evidence/` and logged.
- **Every decision is logged** in `notes/decision-log.md` with who proposed it and whether it has been reviewed.

## Data

[Memora](https://github.com/geniesinc/Memora) (Uddin et al., 2026), weekly split: 10 simulated personas, about 1,550 sessions with a per-session add/update/delete log, and 150 evaluation questions with 735 yes/no sub-questions. Pinned to commit `a6493188`. We implement Memora's FAMA score ourselves (`src/memrecon/fama.py`, tested against Memora's own code) and grade with our own judge, so our FAMA numbers are comparable with Memora's published table in formula only.

## AI use

Claude (Anthropic), through Claude Code, wrote most of the code under Sam's direction, proposed parts of the design, analysed outputs and keeps the decision log; the log records, for each decision, who proposed it and whether Sam has reviewed it. Sam made the decisions, ran every experiment and made all hand labels. The full statement is in the midway report.
