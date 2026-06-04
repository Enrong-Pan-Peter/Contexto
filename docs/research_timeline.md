# Research Timeline

This document reconstructs the project timeline from repository documentation,
trace names, and recorded experiment notes. It is intended to help recover the
sequence of implementation decisions and experiments for a future research
report. Dates are only included where supported by existing files, trace names,
or logs.

## Project Goal

The project studies automated strategies for solving Contexto-like word guessing
games. A game returns a semantic rank for each guessed word; lower ranks indicate
higher semantic closeness to the hidden target. The current codebase compares:

- LLM-generated semantic hypotheses and candidate words.
- Embedding-neighbor search baselines.
- Local games where the ranking model is known (`glove.6B.300d`).
- Real Contexto API games where the ranking model is unknown.

## Timeline

### 2026-04-28 — Early Local Embedding Validation

Evidence: early local embedding traces such as
`traces/embedding_local_cat_20260428_130247.json`.

Milestones:
- Loaded GloVe vectors.
- Built an offline Contexto-style local game.
- Validated the shared rank behavior for simple words such as `cat`, `dog`, and
  unknown guesses.

Research value: this established a deterministic local backend for unrestricted
experiments without real API rate limits.

### 2026-05-04 — LLM Evolutionary Solver and Early Failure Analysis

Evidence: `traces/llm_local_cat_20260504_121840.json` and early API/local traces
listed in `docs/experiment_log.md`.

Milestones:
- The LLM evolutionary solver solved local target `cat` in 468 guesses over 13
  generations.
- Trace inspection showed hypothesis bloat and near-duplicate mutation paths.
- Real API rank normalization was validated through later API solves.
- The shared game interface was treated as an invariant: solvers see rank `1` as
  solved regardless of backend.

Design response:
- Cap active hypotheses.
- Deduplicate near-identical hypotheses.
- Encourage mutation toward divergent interpretations of strong clues.

### 2026-05-04 — Batch Experiment Runner and Embedding Baseline

Evidence: `traces/experiment_smoke.json`, `traces/experiment_smoke.csv`, and
`contexto_solver/experiment.py`.

Milestones:
- Added a local batch experiment runner that writes JSON and CSV summaries.
- Added support for repeated runs over target sets.
- Added scaffolding for aligned and non-aligned embedding experiments.
- Added an embedding-neighbor solver baseline.

Research value: this created the infrastructure needed for repeated local
comparisons rather than isolated single-run anecdotes.

### 2026-05-04 — Local Search and Generation-Budget Observations

Evidence: `notorious` runs documented in `docs/experiment_log.md`.

Milestones:
- `notorious` remained unsolved at 15 generations with best word `crime`, rank
  19.
- Increasing the budget to 20 generations improved to `gang`, rank 4, but did
  not solve.
- Broader local-search prompts and global avoid lists improved exploration but
  did not reliably escape the crime/group noun neighborhood.

Research value: this exposed local semantic stagnation near strong but
incomplete clues.

### 2026-05-05 — Same-Target Variance on `herbaceous`

Evidence:
- `traces/llm_local_herbaceous_20260505_122620.json`
- `traces/llm_local_herbaceous_20260505_130240.json`
- `traces/llm_local_herbaceous_20260505_130555.json`

Milestones:
- Same target and setup produced a fast solve, a rank-3 stall, and a later solve.
- Reaching `shrub` at rank 3 did not guarantee the solver would discover
  `herbaceous`.

Research value: this showed that reaching the right neighborhood is not enough;
the solver must also identify the right semantic relation.

### 2026-05-06 — Singular/Plural Redundancy and `superficial` Failure Modes

Evidence:
- `traces/llm_local_herbaceous_20260506_141417.json`
- `traces/llm_local_superficial_20260506_150023.json`
- `traces/llm_local_superficial_20260506_151008.json`
- `traces/llm_local_superficial_20260506_151506.json`

Milestones:
- `herbaceous` exposed `shrub -> shrubs` as a redundant but high-ranking move.
- `superficial` exposed misleading neighborhoods around `subtle`, `obvious`,
  and `visceral`.

Design response:
- Added prompt instructions to avoid singular/plural variants.
- Added lightweight singular/plural family filtering.
- Strengthened the motivation for explicit stall-pivot behavior.

### 2026-05-06 to 2026-05-07 — Stall Pivot Mechanism and Ollama Backend

Evidence:
- `docs/architecture.md`
- `contexto_solver/solver_llm.py`
- `contexto_solver/llm_client.py`
- commit messages including `Mitigation for stalling around a word` and `Add
  local LLM backend for stall experiments`.

Milestones:
- Added a stall detector.
- Added LLM-backed pivot operators for morphology, register shift, adjacent
  category jumps, and fresh diversity.
- Added Ollama provider support with `--provider ollama` and `--ollama-model`.
- Validated an Ollama `qwen3:14b` local run on `superficial`.

Research value: this created a local, quota-free evaluation path for long LLM
experiments and introduced a concrete intervention for near-target stagnation.

### 2026-05-08 to 2026-05-11 — Pivot Matrix Evaluation In Progress

Evidence:
- `traces/pivot_matrix_off.json`
- `traces/pivot_matrix_off.csv`
- terminal output in the active experiment terminal
- `contexto_solver/analyze_pivot_matrix.py`

Milestones:
- Began a paired evaluation matrix using Ollama `qwen3:14b`, targets
  `notorious`, `herbaceous`, and `superficial`, five repeats per target, and
  max generation budget 50.
- The pivot-off condition was observed in progress/completion during this
  interval.
- At this point in the timeline, the pivot-on condition and final analysis
  outputs were not yet available.

Research value: this started the first planned batch-level evaluation of the
pivot mechanism. The completed 2026-05-13 milestone below records the resulting
batch-level conclusion.

### 2026-05-13 — Pivot Evaluation Matrix Completed

Evidence:
- [`traces/pivot_matrix_off.json`](../traces/pivot_matrix_off.json)
- [`traces/pivot_matrix_on.json`](../traces/pivot_matrix_on.json)
- [`traces/pivot_matrix_off.csv`](../traces/pivot_matrix_off.csv)
- [`traces/pivot_matrix_on.csv`](../traces/pivot_matrix_on.csv)
- [`traces/pivot_matrix_analysis.json`](../traces/pivot_matrix_analysis.json)
- [`traces/pivot_matrix_condition_stats.csv`](../traces/pivot_matrix_condition_stats.csv)
- [`traces/pivot_matrix_paired_stats.csv`](../traces/pivot_matrix_paired_stats.csv)
- [`traces/pivot_matrix_combined_runs.csv`](../traces/pivot_matrix_combined_runs.csv)
- [`docs/experiment_log.md`](experiment_log.md#2026-05-13--pivot-evaluation-matrix-qwen3-14b)
- [`docs/findings.md`](findings.md#2026-05-13--pivot-matrix-shows-faster-stall-recovery-but-not-a-complete-unblock)

Milestones:
- Completed the paired pivot evaluation matrix: three targets, five repeats per
  target, pivot off/on, Ollama `qwen3:14b`, aligned local GloVe game, and a
  50-generation cap.
- Confirmed batch-level evidence that pivots improve solver speed and narrow
  failed-run outcomes on the tested matrix.
- Confirmed the limitation that `notorious` remains hard: both conditions solved
  1/5 and hit the 50-generation median cap.
- Decision pending on continuous pivot direction; supervisor question has been
  sent.

Research value: this moves pivoting from single-run and repeated-run evidence to
batch-level evidence, while preserving a clear open decision about whether the
next step should continue pivot work or shift to another diversity mechanism.

### 2026-05-19 — HPC Pivot Matrix Replication

Evidence:
- [`traces/pivot_matrix_20260519_hpc_analysis.json`](../traces/pivot_matrix_20260519_hpc_analysis.json)
- [`traces/pivot_matrix_20260519_hpc_condition_stats.csv`](../traces/pivot_matrix_20260519_hpc_condition_stats.csv)
- [`traces/pivot_matrix_20260519_hpc_paired_stats.csv`](../traces/pivot_matrix_20260519_hpc_paired_stats.csv)
- [`traces/pivot_matrix_20260519_hpc_combined_runs.csv`](../traces/pivot_matrix_20260519_hpc_combined_runs.csv)
- [`docs/experiment_log.md`](experiment_log.md#2026-05-19--hpc-pivot-evaluation-matrix-qwen3-14b)
- [`docs/findings.md`](findings.md#2026-05-19--hpc-pivot-replication-strengthens-aggregate-speed-claim-but-weakens-per-target-certainty)

Milestones:
- Re-ran the pivot off/on matrix on cloud compute resources for the same target
  set and 50-generation cap.
- Replicated the aggregate direction of the local matrix: pivots reduced
  solved-run guesses and generations, while narrowing failed-run variance.
- Strengthened the aggregate solved-guess evidence from borderline local
  evidence to a statistically significant HPC paired result.
- Identified instability in per-target conclusions: `herbaceous` strengthened,
  `superficial` weakened, and `notorious` remained difficult.

Research value: this replication makes the aggregate pivot-speed claim more
defensible while clarifying that per-target effects at n=5 should remain
illustrative rather than definitive.

### 2026-05-21 — Trace Trajectory Visualization Tools

Evidence:
- [`contexto_solver/plot_trajectory.py`](../contexto_solver/plot_trajectory.py)
- [`figures/llm_local_superficial_20260507_133325_rank.png`](../figures/llm_local_superficial_20260507_133325_rank.png)
- [`figures/llm_local_superficial_20260507_133325_distance.png`](../figures/llm_local_superficial_20260507_133325_distance.png)
- [`figures/llm_local_superficial_20260507_133325_pacmap.png`](../figures/llm_local_superficial_20260507_133325_pacmap.png)
- [`figures/llm_local_superficial_20260506_151008_rank.png`](../figures/llm_local_superficial_20260506_151008_rank.png)
- [`figures/llm_local_superficial_20260506_151008_distance.png`](../figures/llm_local_superficial_20260506_151008_distance.png)
- [`figures/llm_local_superficial_20260506_151008_pacmap.png`](../figures/llm_local_superficial_20260506_151008_pacmap.png)

Milestones:
- Added a standalone trajectory plotting module for existing trace JSON files.
- Added target-neighborhood variance checks, single-run 2D projections, rank
  trajectories, cosine-distance trajectories, and PCA/UMAP/PaCMAP projection
  options.
- Verified the visualization pipeline on one solved and one unsolved
  `superficial` trace generated with the local GloVe game.

Research value: these plots improve qualitative trace diagnosis and help compare
individual trajectories, but they are not a substitute for repeated-run or
batch-level evidence.

### 2026-05-22 — Self-Adaptive Mutation Method

Evidence:
- [`contexto_solver/operators.py`](../contexto_solver/operators.py)
- [`contexto_solver/methods/ea_llm_self_adaptive.py`](../contexto_solver/methods/ea_llm_self_adaptive.py)
- [`tests/test_self_adaptive_operators.py`](../tests/test_self_adaptive_operators.py)
- [`scripts/inspect_self_adaptive_trace.py`](../scripts/inspect_self_adaptive_trace.py)
- [`traces/ea_llm_self_adaptive_local_notorious_20260522_035806.json`](../traces/ea_llm_self_adaptive_local_notorious_20260522_035806.json)
- [`docs/design_decisions.md`](design_decisions.md#self-adaptive-mutation-operators)

Milestones:
- Supervisor design discussion selected four mutation operators and `mu=5`;
  the initial log-normal self-adaptation framing was replaced by Dirichlet
  perturbation because sigma is a probability vector on the four-simplex.
- Added `ea_llm_self_adaptive` as a separate EA+LLM method.
- Added `operators.py` with four operator IDs, prompt mapping, operator
  sampling, sigma perturbation, and uniform initial sigma.
- Added four operator prompts in `llm_client.py`.
- Single-run observation: completed the first available end-to-end
  self-adaptive trace:
  [`ea_llm_self_adaptive_local_herbaceous_20260521_214416.json`](../traces/ea_llm_self_adaptive_local_herbaceous_20260521_214416.json).
- Added prompt-leakage checks to preserve the invariant that sigma remains
  backend-only metadata.
- Added a standalone self-adaptive trace inspection script for sigma drift,
  parent lineage, operator usage, and best-lineage analysis.
- Single-run observation: ran the `notorious` self-adaptive trace to rank 4
  (`gangster`); the lineage inspection exposed a crossover uniform-sigma reset
  and opaque adaptive lineage.
- Added Fix 1 (`child_sigma` in `OPERATOR_SAMPLED`) and Fix 2 (full mutation
  child records in `MUTATE.children`) to make mutation lineage inspectable.

Research value: this adds an adaptive exploration-scale mechanism for future
experiments. The current evidence is implementation and smoke-run validation,
not a repeated-run performance result.

### 2026-05-25 — Self-Adaptive Trace Diagnosis on `superficial`

Evidence:
- [`traces/ea_llm_self_adaptive_local_superficial_20260525_194148.json`](../traces/ea_llm_self_adaptive_local_superficial_20260525_194148.json)
- [`docs/experiment_log.md`](experiment_log.md#self-adaptive-runs)
- [`docs/findings.md`](findings.md#2026-05-26--self-adaptive-sigma-telemetry-shows-trace-level-adaptation-signals)

Milestones:
- Single-run observation: the `superficial` run stayed at `sharpness` rank 98
  for 27 generations and later reached `medium` rank 42 through local search.
- Uncertain / needs more data: trace inspection identified local-search
  uniform-sigma injection as a mechanism that can enter the adaptive population;
  the mechanism is confirmed in code, but the empirical effect needs
  post-Fix-6 comparison.

Research value: this separated implementation telemetry from an algorithmic
confound that needed a method-local mitigation.

### 2026-05-26 — Self-Adaptive Crossover and Local-Search Fixes

Evidence:
- [`contexto_solver/methods/ea_llm_self_adaptive.py`](../contexto_solver/methods/ea_llm_self_adaptive.py)
- [`traces/ea_llm_self_adaptive_local_superficial_20260526_011632.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_011632.json)
- [`traces/ea_llm_self_adaptive_local_superficial_20260526_082930.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_082930.json)
- [`traces/ea_llm_self_adaptive_local_superficial_20260526_135302.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_135302.json)

Milestones:
- Fix 5: implemented self-adaptive crossover sigma blending; the five-generation
  smoke trace verified blended crossover metadata and non-uniform crossover
  child sigma.
- Single-run observation: the full post-Fix-5 `superficial` run reached `thin`
  rank 5 and verified that crossover blending fired on every crossover event.
- Fix 6: disabled local search by default in adaptive mode and added one-shot
  `LOCAL_SEARCH_DISABLED` trace logging.
- Single-run observation: the 15-generation post-Fix-6 verification run on
  `superficial` verified 0 `LOCAL_SEARCH` events, 1 `LOCAL_SEARCH_DISABLED`
  event, and non-uniform crossover child sigma for all 15 crossover events; it
  also identified population diversity collapse as a new open issue.

Research value: this made self-adaptive lineage telemetry more continuous and
removed the default uniform-sigma local-search injection path. The post-Fix-6
verification trace adds a new open question about diversity maintenance under
fully adaptive selection.

### 2026-05-29 — Self-Adaptive Population Increase

Evidence:
- [`contexto_solver/config.py`](../contexto_solver/config.py)
- [`contexto_solver/main.py`](../contexto_solver/main.py)
- [`contexto_solver/experiment.py`](../contexto_solver/experiment.py)
- [`docs/design_decisions.md`](design_decisions.md#self-adaptive-mutation-operators)

Milestones:
- Raised the default self-adaptive population from `mu=5` to `mu=15`.
- Added `SELF_ADAPTIVE_INITIAL_CATEGORIES=15` so
  `ea_llm_self_adaptive` starts from 15 initial hypotheses without changing
  `ea_llm` or `ea_llm_pivot`, which continue to use `INITIAL_CATEGORIES=6` and
  `MAX_ACTIVE_HYPOTHESES=5`.
- Kept the `mu=5` baseline reproducible through environment overrides for
  comparison runs.

Research value: this follows Ting's recommendation and gives sigma adaptation
more selection pressure across competing lineages. It remains a design change
pending repeated-run evidence, not a performance result.

### 2026-06-01 — MAP-Elites Archive Method

Evidence:
- [`contexto_solver/methods/ea_llm_map_elites.py`](../contexto_solver/methods/ea_llm_map_elites.py)
- [`contexto_solver/llm_client.py`](../contexto_solver/llm_client.py)
- [`contexto_solver/config.py`](../contexto_solver/config.py)
- [`docs/design_decisions.md`](design_decisions.md#map-elites-archive-selection-ea_llm_map_elites)

Milestones:
- Added `ea_llm_map_elites`, inheriting from `ea_llm_self_adaptive` and
  replacing top-mu selection with a `5x5` behavior archive over two LLM-placed
  axes: concreteness and specificity.
- Placement is a single anchored-scale LLM call (`LLMClient.place_word`), cached
  to disk keyed by `(model, anchors_hash, word)` so anchor changes invalidate
  stale entries; no embedding centroid math, so the method works against the
  real Contexto API.
- Inherited the sigma self-adaptation unchanged; each hypothesis is created with
  exactly one immutable `best_word` and placed by per-cell competition.
- Added trace events `AXIS_DEFINITION`, `PLACEMENT`, and `ARCHIVE_*` to make
  placements re-derivable and support later quality-diversity analysis.
- Left all existing methods (`ea_llm`, `ea_llm_pivot`, `ea_llm_self_adaptive`,
  `llm_only`, `embedding`) behaviorally unchanged.

Research value: this targets the selection-layer diversity problem identified in
earlier batches. It is a design change pending repeated-run evidence, not yet a
performance result. Post-analysis visualization (sigma heatmaps, archive scatter
plots) is deferred as follow-up tooling.

### 2026-06-02 — MAP-Elites Visualization

Evidence:
- [`contexto_solver/plot_map_elites.py`](../contexto_solver/plot_map_elites.py)
- [`docs/architecture.md`](architecture.md#contexto_solverplot_map_elites)

Milestones:
- Added `contexto_solver.plot_map_elites`, a standalone analysis script that
  renders seven static figures from a MAP-Elites trace: cell-occupancy and
  hit-count heatmaps, archive growth, continuous placement scatter, per-component
  sigma heatmaps (final and over-time snapshots), and the winning-lineage sigma
  trajectory, plus an optional combined summary.
- All figures are reconstructed from existing trace events (`AXIS_DEFINITION`,
  `PLACEMENT`, `ARCHIVE_*`); no new events or solver changes were needed.
- Output is grouped per run under `figures/<run_label>/`; non-MAP-Elites traces
  exit cleanly. `plot_trajectory.py` and `inspect_self_adaptive_trace.py` are
  unchanged.

Research value: makes the quality-diversity behavior of the archive legible
(spatial sigma structure, occupancy growth, empty-vs-contested cells) for
diagnosing runs. Diagnostic tooling, not a performance result.

## Current Open Questions

- Should work continue directly on pivot direction selection, given the completed
  matrix's speed gains but persistent `notorious` failures?
- Are pivot effects stable beyond `notorious`, `herbaceous`, and `superficial`
  under a larger target set?
- Are misleading neighborhoods caused mainly by GloVe geometry, LLM proposal
  bias, or the evolutionary selection loop?
- Would an archive or MAP-Elites-inspired diversity mechanism improve over the
  current active-set plus reactive-pivot design? An initial implementation now
  exists (`ea_llm_map_elites`); whether it improves solve rate, generation
  count, or diversity over the current design remains open pending repeated-run
  evidence.
- How do LLM-guided search, aligned embedding search, and non-aligned embedding
  search compare under repeated local benchmarks?
- Does self-adaptive operator selection improve solve rate, generation count, or
  failed-run stability compared with fixed mutation and pivot-only methods?
