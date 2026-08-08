# RQ1/RQ3 HPC Batch Command Sheet

Self-contained run sheet for a cluster agent that knows the HPC but not this
project. Every flag, env var, and config name below was verified against the code
(`contexto_solver/experiment.py`, `config.py`, `llm_client.py`, `methods/llm_only.py`,
`game_api.py`, `rank_cache.py`, and the `scripts/` verifiers). Syntax is bash/Linux.

## 0. Ground rules (read first)

- **Entry point is the batch runner, not `main.py`:**
  `python -m contexto_solver.experiment`. Run it from the repo root (the working
  directory must contain the `contexto_solver/` package). Python 3.10+.
- **Config is read from the environment at process start.** `config.py` calls
  `os.getenv(...)` at import time and `load_dotenv()` uses `setdefault` (it never
  overrides a variable already exported). So **all `export`s must be set in the job
  script before `python` launches** — they cannot be passed as CLI flags. Each arm's
  env block below is the job-script preamble; prepend it verbatim to that arm's
  command.
- **One command = one job = one game number.** Each command runs all 5 repeats for a
  single game (`--runs-per-target 5`), sequentially, inside one job. This mapping is
  what makes the concurrency rule in §2 trivial to enforce.
- **Seeds:** `--random-seed 0` means repeat *k* (0-indexed) uses seed `0 + k`, i.e.
  seeds 0,1,2,3,4 within each command. Reproducible and identical across arms.
- **No LLM API key needed:** with `--provider ollama` the client sets its key to the
  literal `"ollama"` automatically. Do **not** set `OPENAI_API_KEY` / `ANTHROPIC_API_KEY`.

---

## 1. Per-arm environment blocks and commands

Common flags on every command: `--game api --provider ollama --runs-per-target 5 --random-seed 0`.

Each arm writes its raw per-run traces and its batch summary into an **arm-specific
`TRACE_DIR`** so arms are separable at analysis time even though A1/A3/A4 share the
method name and game numbers (see §1.5 for the naming/collision details).

### Arm A1 — `ea_llm_self_adaptive`, self-report ON, rationale inheritance ON, frozen-uniform sigma, qwen3:14b

Env block (job-script preamble):

```bash
export SELF_REPORT=1
export RATIONALE_INHERITANCE=1
export SELF_ADAPTIVE_SIGMA_MODE=frozen_uniform
export TRACE_DIR=traces/rq1_A1
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1   # point at the node-local Ollama; must end in /v1
# export RANK_CACHE_DIR=data/rank_cache            # default; shared per-game cache (see §2)
```

Commands (10 games → 10 jobs):

```bash
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1303 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1303.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1307 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1307.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1319 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1319.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1327 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1327.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1335 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1335.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1352 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1352.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1364 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1364.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1365 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1365.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1372 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1372.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1384 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A1/A1_ea_llm_self_adaptive_game1384.json
```

### Arm A2 — `llm_only`, self-report ON only, qwen3:14b, guess budget capped at 350

**Guess-budget knob:** `llm_only` has no separate guess cap — the loop in
`LLMOnlyMethod.solve()` runs `while not solved and self.generation < max_generations`,
incrementing `generation` once per guess attempt. So **`--max-generations` IS the
guess cap** for `llm_only`; set it to `350`. (`RATIONALE_INHERITANCE` and
`SELF_ADAPTIVE_SIGMA_MODE` do not apply to `llm_only`; leave them unset.)

Env block:

```bash
export SELF_REPORT=1
unset RATIONALE_INHERITANCE
unset SELF_ADAPTIVE_SIGMA_MODE
export TRACE_DIR=traces/rq1_A2
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1   # must end in /v1
```

Commands (5 games → 5 jobs):

```bash
python -m contexto_solver.experiment --game api --method llm_only --provider ollama --ollama-model qwen3:14b --game-numbers 1303 --runs-per-target 5 --random-seed 0 --max-generations 350 --output traces/rq1_A2/A2_llm_only_game1303.json
python -m contexto_solver.experiment --game api --method llm_only --provider ollama --ollama-model qwen3:14b --game-numbers 1307 --runs-per-target 5 --random-seed 0 --max-generations 350 --output traces/rq1_A2/A2_llm_only_game1307.json
python -m contexto_solver.experiment --game api --method llm_only --provider ollama --ollama-model qwen3:14b --game-numbers 1319 --runs-per-target 5 --random-seed 0 --max-generations 350 --output traces/rq1_A2/A2_llm_only_game1319.json
python -m contexto_solver.experiment --game api --method llm_only --provider ollama --ollama-model qwen3:14b --game-numbers 1335 --runs-per-target 5 --random-seed 0 --max-generations 350 --output traces/rq1_A2/A2_llm_only_game1335.json
python -m contexto_solver.experiment --game api --method llm_only --provider ollama --ollama-model qwen3:14b --game-numbers 1372 --runs-per-target 5 --random-seed 0 --max-generations 350 --output traces/rq1_A2/A2_llm_only_game1372.json
```

### Arm A3 — `ea_llm_self_adaptive`, self-report OFF, rationale inheritance OFF, frozen-uniform sigma, qwen3:14b

Both flags are booleans parsed by `config._env_bool`: **on** = one of `1/true/yes/on`
(case-insensitive); **off** = unset, empty, `0`, `false`, `no`, or `off`. Explicit `0`
is used below (equivalent to unsetting). `SELF_ADAPTIVE_SIGMA_MODE` stays
`frozen_uniform` (same substrate as A1 for a clean A1-vs-A3 contrast).

Env block:

```bash
export SELF_REPORT=0
export RATIONALE_INHERITANCE=0
export SELF_ADAPTIVE_SIGMA_MODE=frozen_uniform
export TRACE_DIR=traces/rq1_A3
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1   # must end in /v1
```

Commands (same 5 games → 5 jobs):

```bash
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1303 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A3/A3_ea_llm_self_adaptive_game1303.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1307 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A3/A3_ea_llm_self_adaptive_game1307.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1319 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A3/A3_ea_llm_self_adaptive_game1319.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1335 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A3/A3_ea_llm_self_adaptive_game1335.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model qwen3:14b --game-numbers 1372 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A3/A3_ea_llm_self_adaptive_game1372.json
```

### Arm A4 — `ea_llm_self_adaptive`, flags as A1, Mistral-class model (tag as a parameter)

Same env as A1. The Mistral tag is a parameter: set `MISTRAL_TAG` in the job script
(confirm the exact tag later) and reference it in `--ollama-model`. Note:
`SUPPORTED_OLLAMA_MODELS` in `config.py` lists only `qwen3:14b, qwen3:30b`, but that
tuple is **advisory** (used only in `--help` text) — it is **not enforced anywhere**,
so any tag your Ollama server has pulled (including a Mistral tag) is accepted and
passed straight through to the server.

Env block:

```bash
export SELF_REPORT=1
export RATIONALE_INHERITANCE=1
export SELF_ADAPTIVE_SIGMA_MODE=frozen_uniform
export TRACE_DIR=traces/rq1_A4
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1   # must end in /v1
export MISTRAL_TAG=MISTRAL_TAG_PLACEHOLDER          # <-- set the real tag before launch
```

Commands (same 5 games → 5 jobs):

```bash
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model "$MISTRAL_TAG" --game-numbers 1303 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A4/A4_ea_llm_self_adaptive_game1303.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model "$MISTRAL_TAG" --game-numbers 1307 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A4/A4_ea_llm_self_adaptive_game1307.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model "$MISTRAL_TAG" --game-numbers 1319 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A4/A4_ea_llm_self_adaptive_game1319.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model "$MISTRAL_TAG" --game-numbers 1335 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A4/A4_ea_llm_self_adaptive_game1335.json
python -m contexto_solver.experiment --game api --method ea_llm_self_adaptive --provider ollama --ollama-model "$MISTRAL_TAG" --game-numbers 1372 --runs-per-target 5 --random-seed 0 --max-generations 50 --output traces/rq1_A4/A4_ea_llm_self_adaptive_game1372.json
```

### 1.5 Where trace files land and their naming pattern

For every command:

- **Batch summary** (written to `--output`): the JSON you named, plus a sibling CSV
  with the same stem (`..._game1303.json` → `..._game1303.csv`). The JSON's
  `runs[].trace_path` lists the raw per-run trace for each repeat.
- **Raw per-run solver traces** (one per repeat) land in `$TRACE_DIR` with the pattern:

  ```
  $TRACE_DIR/<method>_api_<game_number>_run<k>_<YYYYmmdd_HHMMSS>.json
  ```

  where `<method>` is `ea_llm_self_adaptive` or `llm_only`, and `<k>` = repeat index
  1..5. Example (A1, game 1303): `traces/rq1_A1/ea_llm_self_adaptive_api_1303_run1_20260721_HHMMSS.json`.

**Why the per-arm `TRACE_DIR` matters:** the raw filename encodes method + game +
repeat but **not** the arm. A1, A3, and A4 all use method `ea_llm_self_adaptive` and
reuse games 1303/1307/1319/1335/1372, so their raw filenames would otherwise collide
in a single directory. Routing each arm to `traces/rq1_A1|A2|A3|A4` keeps them
separable by directory; additionally each arm is distinguishable inside the trace via
`RUN_CONFIG` (`self_report`, `rationale_inheritance`, `self_adaptive_sigma_mode`,
`llm_model`). Do **not** collapse arms into one `TRACE_DIR`.

---

## 2. Concurrency rules (state to the scheduler plainly)

1. **The unit of parallelism is the game number.** Different game numbers may run
   fully in parallel across nodes/jobs.
2. **Never run two jobs on the same game number at the same time — across *all* arms.**
   Each game number has one shared per-game rank-cache file:
   ```
   data/rank_cache/game_<N>_https_api.contexto.me_machado_en_game.json
   ```
   `RankCache` reads this file once at construction and rewrites the whole file
   (atomic `os.replace`) on every stored rank. Two concurrent jobs on the same `<N>`
   would clobber each other's cache writes and race the per-process rate limiter.
   Games 1303/1307/1319/1335/1372 appear in A1 **and** A2/A3/A4 — so, e.g.,
   `A1/game1303` and `A3/game1303` must be serialized relative to each other, not just
   within an arm.
   - *Optional relaxation:* if you must run same-game jobs from different arms
     concurrently, give each arm its own `export RANK_CACHE_DIR=data/rank_cache_A1`
     (etc.) so no file is shared. This costs redundant API calls (the rank of a word
     for a given game is deterministic, so a shared cache is otherwise a pure win).
3. **Everything inside one job runs sequentially.** The API batch runner is strictly
   sequential across repeats (and across game numbers if you ever pass more than one),
   by design — do not attempt intra-job parallelism.
4. **Be gentle on the shared public API.** All jobs hit the same public host from your
   cluster's egress IP. Each process self-limits to ~1 request / 0.5 s
   (`API_RATE_LIMIT`), but keep the number of simultaneously-running game jobs modest
   to avoid tripping server-side throttling.

---

## 3. External requirements

- **Outbound HTTPS to `api.contexto.me` (port 443) from compute nodes.**
  `config.API_BASE_URL = https://api.contexto.me/machado/en/game`
  (override via env `API_BASE_URL` only if you must). The client issues
  `GET {API_BASE_URL}/<game_number>/<word>` with a 15 s timeout and a ~0.5 s
  pre-request sleep (`API_RATE_LIMIT`). Cache hits do not touch the network.
- **An Ollama server reachable by the job.** The code locates it **only** via the
  `OLLAMA_BASE_URL` env var (`config.py`: `OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL",
  "http://localhost:11434/v1")`). The client POSTs to
  `${OLLAMA_BASE_URL}/chat/completions`, so the value **must include the `/v1`
  suffix**. Point it at a node-local server, e.g.
  `export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1`. Per-call timeout is
  `OLLAMA_REQUEST_TIMEOUT_SECONDS` (default `900`); raise it if a node's generations
  are slow. No API key is required for Ollama.
- **Model tags to pull on each Ollama server before launch:**
  - `ollama pull qwen3:14b`  (arms A1, A2, A3)
  - `ollama pull <MISTRAL_TAG>`  (arm A4; the confirmed Mistral tag)
  Verify with `ollama list`. A missing tag surfaces as
  `Model <tag> not found. Run ollama pull <tag>.`

---

## 4. First-trace verification (run after the first A1 job completes)

Point these at the **raw per-run trace** (not the `--output` summary). Use the first
completed A1 repeat, e.g. `traces/rq1_A1/ea_llm_self_adaptive_api_1303_run1_*.json`.
All three scripts are analysis-only (no network, no LLM, no writes to traces/caches).

**a) Primary gate — instrumentation verifier:**

```bash
python scripts/verify_self_report_pilot.py traces/rq1_A1/ea_llm_self_adaptive_api_1303_run1_*.json
```

Pass criteria (from its printed report):
- **Parse failures near 0** — `parse failures: <n> (rate=...)`; expect ~0, and
  `parse_failure_within_hard_max: True`. The script's hard limit is 10% and it flags
  >2% for discussion, but for a healthy run this should be essentially 0.
- **Non-degenerate bucket distribution** — `bucket counts: {...}` should span multiple
  buckets (not all in one), the closeness `histogram [0-1]` should be spread, and
  `degenerate clustering: False` (equivalently acceptance check
  `closeness_degenerate_cluster: False`). Also expect `all in [0,1]: True`.
- Sanity: `all normalized key sets canonical: True` and
  `schema version consistent across files: True`.

**b) Per-run tidy count (individual extraction sanity):**

```bash
python scripts/rq1_tidy_table.py traces/rq1_A1/ea_llm_self_adaptive_api_1303_run1_*.json --output traces/rq1_A1/first_trace_tidy.csv
```

Pass criterion:
- **Per-run tidy count roughly 19–42** — the printed `individuals: <n>` (one row per
  self-reported individual). Local A1-equivalent pilot runs produced 19, 24, 37, 42,
  42 individuals, so 19–42 is the expected band for a single 50-generation run.

**c) Calibration/bucket pipeline (confirms the RQ1 metrics stage runs on the trace):**

```bash
python scripts/rq1_metrics.py traces/rq1_A1/ea_llm_self_adaptive_api_1303_run1_*.json --output traces/rq1_A1/first_trace_metrics.json
```

Pass criteria:
- `per_run[...]` shows a `metrics` block with a `count` in the same **19–42** band.
- `pooled` bucket/reliability fields are populated (predicted-bucket distribution is
  non-degenerate — spread across buckets), confirming the pipeline consumes the arm's
  traces end-to-end.

If (a) shows high parse failures, an empty `total proposals`, or degenerate
clustering, **stop and diagnose before launching the rest of the batch** (most likely
`SELF_REPORT` was not exported into the job's environment).

---

## 5. Run counts, timing, and disk footprint

**Run counts (125 total):**

| Arm | Method | Games | Repeats/game | Runs | Jobs |
|-----|--------|-------|--------------|------|------|
| A1  | ea_llm_self_adaptive | 10 | 5 | 50 | 10 |
| A2  | llm_only             |  5 | 5 | 25 |  5 |
| A3  | ea_llm_self_adaptive |  5 | 5 | 25 |  5 |
| A4  | ea_llm_self_adaptive |  5 | 5 | 25 |  5 |
| **Total** | | | | **125** | **25** |

**Wall-clock per run (extrapolated from local runs, qwen3:14b on local Ollama):**

- **EA arms (A1/A3/A4), 50 generations:** measured local `ea_llm_self_adaptive` API
  runs took ≈ **2,400–4,300 s (~40–72 min), ~1 h typical** per run. Wall time is
  dominated by Ollama generation, not the Contexto API (summed API latency was only
  ~300–800 s per run). HPC GPU nodes should be comparable or faster depending on the
  GPU. → An EA job (5 sequential repeats) ≈ **~3.5–6 h**.
- **A2 `llm_only`, capped at 350 guesses:** no separate local 350-cap timing; each
  generation is one `next_guess` LLM call (+ up to 2 clean-word retries), and runs
  terminate early on solve. Estimate ≈ **~20–60 min per run**, so an A2 job ≈
  **~2–5 h**. Treat as a rough upper bound tied to the 350 cap.
- **A4 Mistral-class:** budget like an EA run (~1 h/run); revise once the tag/GPU are
  fixed.
- **Whole batch wall-clock:** if all 25 game-jobs run in parallel (respecting §2),
  total wall time ≈ the longest single job ≈ **~5–6 h**. Fully sequential it would be
  on the order of **~90–120 h** of compute.

**Disk footprint of traces:**

- Local `ea_llm_self_adaptive` API traces with `SELF_REPORT=1` measured **~0.28–0.66
  MB** each (max observed ~1 MB for very long runs). A3 (self-report OFF) traces are
  smaller (~0.1–0.2 MB, no self-report payload). `llm_only` traces are small
  (~0.1–0.3 MB).
- Estimate: A1+A4 (75 self-report runs × ~0.5 MB) ≈ **~38 MB**; A3 (25 × ~0.2 MB) ≈
  **~5 MB**; A2 (25 × ~0.2 MB) ≈ **~5 MB**; plus per-arm summary JSON/CSV (small) and
  the shared rank caches in `data/rank_cache/` (~tens of KB per game, ~1 MB total).
- **Total ≈ 50–100 MB; provision ~200 MB of headroom** to absorb long-run spikes.
