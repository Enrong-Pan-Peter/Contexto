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

### 2026-06-06 — First MAP-Elites `superficial` Test Run Analysis

Evidence:
- [`traces/ea_llm_map_elites_local_superficial_20260606_015531.json`](../traces/ea_llm_map_elites_local_superficial_20260606_015531.json)
- [`docs/experiment_log.md`](experiment_log.md#2026-06-06--superficial-seed-4-qwen3-14b)
- [`docs/findings.md`](findings.md#2026-06-06--first-map-elites-superficial-run-suggests-generation-layer-exhaustion)

Milestones:
- Recorded the first analyzed MAP-Elites `superficial` run as single-run
  evidence only: seed 4, Ollama `qwen3:14b`, 70-generation cap, MiniLM local-game
  backend, LLM-driven placement.
- Observed generation-layer exhaustion: successful children/gen fell from 18.15
  in generations 1-20 to 7.20 in generations 41-70, while the inferred
  duplicate-proposal/drop rate reached 68.33% in generations 50-70.
- Recorded archive saturation: 19/25 final cells occupied, no placements ever
  landed in the six empty cells, occupancy last increased at generation 45, and
  best rank last improved at generation 37 (`thin`/5).
- Recorded sigma diagnostics as suggestive but confounded: final incumbents had
  elevated `sigma_l` mean (0.447 vs 0.25 uniform), and the winning lineage also
  drifted toward large mutation, but controls are needed before interpreting
  this as selection-layer validation.

Research value: surfaces a new hypothesis that MAP-Elites solved the
selection-layer diversity problem but exposed a generation-layer exhaustion and
endgame-focus problem. This is diagnostic single-run evidence, not a performance
claim.

### 2026-06-08 — Pooled MAP-Elites Sigma-Fitness Coupling Analysis

Evidence:
- [`scripts/measure_sigma_fitness_coupling.py`](../scripts/measure_sigma_fitness_coupling.py)
- [`traces/`](../traces/) filtered to MAP-Elites traces with `AXIS_DEFINITION`
- [`docs/experiment_log.md`](experiment_log.md#2026-06-08--pooled-sigma-fitness-coupling-for-map-elites)
- [`docs/findings.md`](findings.md#2026-06-08--pooled-map-elites-sigma-fitness-coupling-favors-small-mutation)

Milestones:
- Pooled 21 MAP-Elites traces and 10,581 linked mutation children, excluding
  crossover because it has no single sampled operator.
- Found a clean operator-fitness gradient favoring small mutation:
  `REPLACE` rate fell from `s_mutation` 14.1% to `l_mutation` 4.8%, and median
  log-rank delta fell from `s=-1.34` to `l=-2.69`.
- Confirmed the gradient survived early/late phase splitting and parent-rank
  tercile control.
- Step-0 sigma confirmation showed that the earlier `sigma_l ~= 0.45` archive
  elevation was not pooled: final archive sigma averaged across runs was
  `[s=0.271, m=0.234, ml=0.232, l=0.263]`.
- Identified inheritance decoupling as the mechanism: child sigma depends on
  parent sigma, not on which operator produced the child word.

Research value: turns the sigma question from a single-run artifact into a
pooled operator-fitness diagnostic. It motivates testing reward-windowed
adaptive operator selection and upgrades the frozen-sigma / random-sigma control
to a causal test of whether sigma adaptation misallocates effort. The origin of
the single-run `sigma_l` elevation remains explicitly uncertain pending those
controls.

### 2026-06-15 — Sigma-Control Arm-Comparison Tooling (no batch results yet)

Evidence:
- [`scripts/compare_sigma_control_arms.py`](../scripts/compare_sigma_control_arms.py)
- [`scripts/run_sigma_control.py`](../scripts/run_sigma_control.py) (produces the batch this consumes)
- [`docs/architecture.md`](architecture.md#scriptscompare_sigma_control_armspy)
- [`docs/design_decisions.md`](design_decisions.md) (arm-comparison analysis design)

Milestones:
- Added the arm-comparison analysis script ahead of the sigma-control batch
  finishing, so the read-out is ready when results land. It groups runs by
  `MAPELITES_SIGMA_MODE`, pairs by `(target, seed)`, and reports per-arm
  `best_rank`, solve rate, archive occupancy, and the per-operator archive sigma
  from the last `ARCHIVE_SNAPSHOT`, with the three highlighted contrasts
  (`adaptive` vs `frozen_uniform`, `adaptive` vs `frozen_fixed`, `random` vs
  `adaptive`).
- Distinguished it from `measure_sigma_fitness_coupling.py`, which pools runs for
  an operator-fitness gradient and does not separate the arms.
- Verified the parser/table on existing MAP-Elites traces only (trace mode);
  those traces predate `MAPELITES_SIGMA_MODE`, so they group as a single
  `unknown` arm and the three contrasts correctly report their arms as absent.

Research value at the time: tooling readiness, not a result. This entry records
the pre-batch state; the completed corrected batch is recorded in the 2026-06-24
milestone below.

### 2026-06-15 — Embedding-vs-LLM Closeness Diagnostic

Evidence:
- [`scripts/compare_embedding_llm_closeness.py`](../scripts/compare_embedding_llm_closeness.py)
- [`docs/experiment_log.md`](experiment_log.md#2026-06-15--embedding-vs-llm-closeness-diagnostic-tool--single-run-illustration)
- [`docs/design_decisions.md`](design_decisions.md) (closeness diagnostic rationale)

Milestones:
- Added a per-target diagnostic that lists the top-N closest words by the local
  game's embedding (`nearest_neighbors`) next to the LLM's ordered list (via the
  public `complete_json_prompt()` path), and reports overlap, exact-position
  matches, Spearman, embedding blind spots, and LLM-only-far words.
- Verified it on real embeddings and the live `qwen3:14b` path; a single
  illustrative run on `chicken` (top-8) showed 1/8 overlap and concrete blind
  spots (see experiment log).

Research value: gives a measurable lens on why LLM guesses may miss
embedding-close words (the solver's blind spots), motivating an embedding/LLM
closeness sweep over hard targets. This is a single-run illustration plus tooling,
not a batch result; the blind-spot-causes-stalls explanation is an untested
hypothesis and reflects MiniLM geometry, not real Contexto.

### 2026-06-15 — Self-Adaptive Operator -> Selection/Fitness Coupling Run

Evidence:
- [`scripts/measure_self_adaptive_selection_coupling.py`](../scripts/measure_self_adaptive_selection_coupling.py)
- [`selcoupling.json`](../selcoupling.json)
- [`docs/experiment_log.md`](experiment_log.md#2026-06-15--self-adaptive-operator---selectionfitness-coupling)
- [`docs/findings.md`](findings.md#2026-06-15--self-adaptive-operator-fitness-gradient-favors-small-mutation-partial)

Milestones:
- Ran the self-adaptive coupling analysis over 30 existing traces and 6,186
  mutation children; gradient generality was confirmed as a partial result:
  small mutation had the strongest operator -> fitness signal under the logged
  top-`max_active`+elite selection step, while sigma-drift and any causal claim
  remain pending.

Research value: upgrades the operator-fitness gradient from MAP-Elites-specific
evidence to a batch-level pooled observational pattern also visible in plain
self-adaptive traces. It does not measure sigma trajectories in those
self-adaptive runs. At the time, it motivated a frozen/random-sigma control; the
completed MAP-Elites control batch is recorded in the next milestone.

### 2026-06-24 — Sigma-Control Batch Corrected and Interpreted

Evidence:
- [`traces/sigma_arms_batch/`](../traces/sigma_arms_batch/)
- [`sigma_control_report_corrected.json`](../sigma_control_report_corrected.json)
- [`scripts/compare_sigma_control_arms.py`](../scripts/compare_sigma_control_arms.py)
- [`docs/experiment_log.md`](experiment_log.md#2026-06-24--corrected-sigma-control-arm-comparison)
- [`docs/findings.md`](findings.md#2026-06-24--sigma-control-batch-inherited-sigma-does-not-beat-controls)

Milestones:
- Completed and analyzed the four-arm MAP-Elites sigma-control batch on the
  MiniLM local game: `adaptive`, `frozen_fixed`, `frozen_uniform`, and `random`.
- Recovered 59 completed runs from 60 planned, with one missing
  `frozen_uniform` run.
- Corrected `best_rank` accounting so solved runs count as rank 1 even when the
  `SOLVED` event occurs after the final `ARCHIVE_SNAPSHOT`.
- Found that inherited sigma self-adaptation did not beat the controls in this
  batch: `adaptive` had the lowest solve rate and worst corrected median
  `best_rank`; `frozen_fixed` performed best on both metrics.
- Retired the broad "sigma drifts toward large" framing. The large-sigma signal
  remains a single-run observation from the older K-off `superficial` trace, not
  a batch-level pattern.

Research value: this converts the sigma question from an observational
operator-fitness gradient into a direct causal-control result on the MiniLM
proxy. The result argues against the current inheritance-based credit assignment
and motivates static informed sigma or adaptive operator selection. Individual
arm comparisons remain underpowered and non-significant at the current n.

### 2026-06-24 — Real-Contexto Top-300 Closeness Diagnostic

Evidence:
- [`closeness_contexto_300_3.json`](../closeness_contexto_300_3.json)
- [`closeness_reports/closeness_contexto_300_3_summary.txt`](../closeness_reports/closeness_contexto_300_3_summary.txt)
- [`closeness_reports/closeness_contexto_300_3_metrics.json`](../closeness_reports/closeness_contexto_300_3_metrics.json)
- [`scripts/compare_embedding_llm_closeness.py`](../scripts/compare_embedding_llm_closeness.py)
- [`scripts/analyze_closeness.py`](../scripts/analyze_closeness.py)
- [`docs/experiment_log.md`](experiment_log.md#2026-06-24--contexto-anchored-top-300-closeness-comparison)
- [`docs/findings.md`](findings.md#2026-06-24--contexto-anchored-closeness-real-neighbors-are-mostly-associative)

Milestones:
- Extended the closeness diagnostic from MiniLM-vs-LLM to a three-way comparison:
  MiniLM local-game neighbors, qwen3:14b LLM neighbors, and manually collected
  real Contexto top-300 ranks.
- Added and ran the offline analyzer over seven targets:
  `blade`, `loyalty`, `otter`, `blackboard`, `arrow`, `rhythm`, and `safe`.
- Recorded denominator-aware aggregate health: `loyalty` failed the LLM branch;
  `otter` is excluded from embedding means as degenerate.
- Found that real Contexto top-50 neighborhoods in this diagnostic set are mostly
  associative words not captured by either proxy, while MiniLM and qwen3 fail in
  different ways.

Research value: this downgrades broad claims that the MiniLM local game is a
strong proxy for real Contexto. Solver results on the local game remain valid as
MiniLM-proxy results, but claims about real Contexto behavior now need direct
real-rank evidence or an explicit uncertainty label.

### 2026-07-06 — Ollama JSON-Shape Fixes and RQ1 Self-Report Shared Layer

Evidence:
- [`contexto_solver/llm_client.py`](../contexto_solver/llm_client.py)
  (`_normalize_json_list`, `_request_json_list`, object-wrapped prompt schemas)
- [`contexto_solver/self_report.py`](../contexto_solver/self_report.py)
- [`contexto_solver/methods/ea_core.py`](../contexto_solver/methods/ea_core.py)
  (`_complete_proposal`, `_crossover_request`, init fail-fast)
- [`tests/test_initial_categories_parsing.py`](../tests/test_initial_categories_parsing.py),
  [`tests/test_list_prompts_parsing.py`](../tests/test_list_prompts_parsing.py),
  [`tests/test_legacy_prompt_snapshots.py`](../tests/test_legacy_prompt_snapshots.py),
  [`tests/test_self_report_prompt.py`](../tests/test_self_report_prompt.py),
  [`tests/test_prompt_isolation.py`](../tests/test_prompt_isolation.py)
- [`scripts/verify_self_report_pilot.py`](../scripts/verify_self_report_pilot.py)
- Single-run pilot (bug exposure, not calibration):
  [`traces/ea_llm_map_elites_aligned_ivory_run1_20260706_151900.json`](../traces/ea_llm_map_elites_aligned_ivory_run1_20260706_151900.json)
- Design notes:
  [`docs/design_decisions.md`](design_decisions.md#ollama-json_object-response-shape-normalization),
  [`docs/design_decisions.md`](design_decisions.md#rq1-operator-self-report-instrumentation-logged-only)

Milestones:
- Fixed Ollama `json_object` mismatches for initial categories and five
  array-shaped list prompts (`propose_words`, `specialize`, `local_search`,
  pivot word lists) via object wrappers plus shared normalization with
  retry-once-then-raise validation.
- Added fail-fast guards so empty initialization no longer runs silent empty
  generations.
- Consolidated RQ1 self-report request/parse routing across all five live LLM
  modes through `contexto_solver/self_report.py` and shared EA helpers; locked
  flag-off prompt bytes with snapshot fixtures for operator and legacy proposal
  paths.
- Extended information-isolation audits to legacy non-operator prompts; no target,
  unguessed-rank, or sigma leaks found in the audited fixture population.
- Documented `SELF_REPORT` usage in `README.md`; detailed structure in
  `docs/architecture.md`.

Research value: unblocks Ollama-backed solver runs that previously could start
with empty archives, and prepares RQ1 calibration data collection. The empty
`ivory` pilot confirms the pre-fix failure mode but does not yet provide
multi-generation self-report calibration evidence. Batch audit runs remain open.

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
- Would adaptive operator selection, which credits the operator that actually
  fired, outperform the inherited sigma mechanism or the informed fixed profile?
- Does the sigma-control result replicate with ranked context off, with more
  targets, or against the real Contexto API rather than the MiniLM proxy?
- How large and diverse must the real-Contexto closeness target set be before the
  associative-neighborhood decomposition supports a general claim?
- After the Ollama JSON-shape fixes, do previously observed empty-archive MAP-Elites
  traces disappear under repeated local pilots, and do instrumented audit runs
  yield analyzable `self_report` fields across generations?
- Does operator `predicted_closeness` correlate with realized game rank once
  dedicated `SELF_REPORT=1` solver traces exist, or is that still untested?
