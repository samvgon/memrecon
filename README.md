# memrecon

**Keep, Discard, or Verify: Write Policies for Recall-Time Memory Rewrites in LLM Agents.** CS329Z (Stanford, Engineering AI Agents), solo project by Sam Goncalves, Autumn 2026.

Some agent-memory systems rewrite a memory when they recall it, so it fits the task at hand. This project tests what happens when the agent **keeps that rewrite unchecked**: one memory agent replays simulated conversations from the Memora benchmark, and only its *write policy* changes between conditions (discard the rewrite, replace the original, keep both, verify against the original, ...). Everything else (models, prompts, retrieval, settings) is held fixed.

**Status (Nov 5, 2026):** Stage 0 (checking the measurement) is in progress. The model setup, model client and scoring function exist and are tested; no experiment has been run yet. Every decision, with who proposed it, its evidence and whether Sam has reviewed it, is in [`notes/decision-log.md`](notes/decision-log.md).

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
| `scripts/explore/` | Numbered data-exploration scripts (E01–E12); their saved outputs are in `notes/evidence/`. See its README. |
| `src/memrecon/llm.py` | The only way experiment code calls a model. |
| `src/memrecon/memory.py` | The memory store: memories, their lineage and depth, snapshots and the event file. |
| `src/memrecon/monitor.py` | Readers for the monitor, and the memory guard runs call before each unit of work. |
| `src/memrecon/fama.py` | Memora's FAMA score. |
| `tests/` | Tests (`uv run pytest`). |
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
   ```
   Check each file against the sha256 in the table below (`sha256sum <file>`). The embedding model and the small agent are only needed later; their files are listed in `config/models.env`.
3. **Start the servers** (each runs in its own tmux session and logs to `~/logs/<role>.log`):
   ```bash
   scripts/serve.sh agent
   scripts/serve.sh judge
   scripts/serve.sh status        # wait until both say "healthy" (a minute or two)
   uv run python scripts/check_models.py
   ```

### B. Hosted models (OpenRouter)

For anyone without the local setup. **These calls cost money** (the two models are cheap, but a full run makes many calls), so the client refuses them unless you opt in.

```bash
cp .env.example .env
```
Then in `.env`, set:
```
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
| Job | Ingests sessions, retrieves and rewrites memories, answers questions | Grades answers that code can't grade (different model family from the agent) |
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

## Watching a run

```bash
tmux new -s monitor
uv run python scripts/monitor.py      # refreshes every 5 s; --once prints once; --raw-metrics lists every server metric
```

It shows, separately: **the machine** (system memory, GPU memory, each model server's throughput and load from its `/metrics` page, which `scripts/serve.sh` turns on); **token use** over the last 5 minutes from the ledger, including how many tokens of memory the prompts carried; **the agent's memory as it would be in real use** (live memories, their size and depths, from the summary written next to each snapshot); and **experiment overhead** (the size of `results/`: audit log, cache, snapshots). The model servers reserve their memory when they start, so memory is most at risk when a server starts, not mid-run. Runs also call a memory guard before each unit of work: if available memory is below `MEMRECON_MIN_FREE_GB` (default 8 GB), they pause until it recovers, and stop cleanly after an hour (a resumed run picks up where it stopped).

## Rules this project follows

- **No paid API calls** unless Sam explicitly approves them (decision D05); the client enforces this with `MEMRECON_ALLOW_PAID`, and `MEMRECON_BUDGET_USD` caps the spend.
- **Frozen persona split:** 3 development personas for building and tuning, 7 test personas for reporting (`notes/split.json`, seed 329). Nothing is tuned on test personas.
- **Every data finding comes from a numbered script** in `scripts/explore/`, with its output saved in `notes/evidence/` and logged.
- **Every decision is logged** in `notes/decision-log.md` with who proposed it and whether Sam has reviewed it.

## Data

[Memora](https://github.com/geniesinc/Memora) (Uddin et al., 2026), weekly split: 10 simulated personas, about 1,550 sessions with a per-session add/update/delete log, and 150 evaluation questions with 735 yes/no sub-questions. Pinned to commit `a6493188`. We implement Memora's FAMA score ourselves (`src/memrecon/fama.py`, tested against Memora's own code) and grade with our own judge, so our FAMA numbers are comparable with Memora's published table in formula only.

## AI use

Claude (Anthropic) helped design the experiment, write the code and keep the decision log; the log records, for each decision, who proposed it and whether Sam has reviewed it.
