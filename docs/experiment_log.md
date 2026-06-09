# Experiment Log

This document is a compact register of completed non-smoke experiments and their
evidence artifacts. It is not a findings document. Paper-facing interpretations
belong in `docs/findings.md`; algorithmic rationale belongs in
`docs/design_decisions.md`.

## How To Read This Log

- `Solved` means the solver reached rank `1` through the shared game interface.
- `Best rank` is the closest rank reached when a run did not solve.
- Local-game experiments use `data/glove.6B.300d.txt` unless otherwise noted.
- Single LLM runs are stochastic and should be treated as qualitative evidence
  unless repeated or included in a batch analysis.
- Smoke tests and zero-generation plumbing checks are intentionally omitted.

## Single-Run Experiments

| Date/time | Backend | Solver/model | Target/game | Result | Guesses | Generations | Best word/rank | Evidence |
|---|---|---|---|---|---:|---:|---|---|
| 2026-05-04 12:12 | local GloVe | LLM | `cat` | solved | 468 | 13 | `cat` / 1 | `traces/llm_local_cat_20260504_121840.json` |
| 2026-05-04 12:49 | real API | LLM | game `1314` | not solved | 290 | 15 | `rug` / 8 | `traces/llm_api_1314_20260504_124906.json` |
| 2026-05-04 14:00 | local GloVe | LLM | `cat` | solved | 2 | 0 | `cat` / 1 | `traces/llm_local_cat_20260504_140013.json` |
| 2026-05-04 14:01 | local GloVe | LLM | `house` | solved | 107 | 4 | `house` / 1 | `traces/llm_local_house_20260504_140109.json` |
| 2026-05-04 14:15 | real API | LLM | game `1314` | solved | 165 | 7 | `ivory` / 1 | `traces/llm_api_1314_20260504_141552.json` |
| 2026-05-04 14:38 | local GloVe | LLM | `notorious` | not solved | 291 | 15 | `crime` / 19 | `traces/llm_local_notorious_20260504_143848.json` |
| 2026-05-04 15:22 | local GloVe | LLM | `notorious` | not solved | 281 | 20 | `gang` / 4 | `traces/llm_local_notorious_20260504_152234.json` |
| 2026-05-04 15:38 | local GloVe | LLM | `notorious` | not solved | 380 | 20 | `gang` / 4 | `traces/llm_local_notorious_20260504_153816.json` |
| 2026-05-04 15:54 | real API | LLM | game `1323` | solved | 254 | 15 | `sponges` / 1 | `traces/llm_api_1323_20260504_155456.json` |
| 2026-05-04 16:26 | local GloVe | LLM | `notorious` | not solved | 387 | 20 | `gang` / 4 | `traces/llm_local_notorious_20260504_162627.json` |
| 2026-05-05 12:26 | local GloVe | LLM | `herbaceous` | solved | 101 | 4 | `herbaceous` / 1 | `traces/llm_local_herbaceous_20260505_122620.json` |
| 2026-05-05 13:02 | local GloVe | LLM | `herbaceous` | not solved | 311 | 20 | `shrub` / 3 | `traces/llm_local_herbaceous_20260505_130240.json` |
| 2026-05-05 13:05 | local GloVe | LLM | `herbaceous` | solved | 215 | 10 | `herbaceous` / 1 | `traces/llm_local_herbaceous_20260505_130555.json` |
| 2026-05-06 14:14 | local GloVe | LLM | `herbaceous` | not solved | 274 | 20 | `shrubs` / 2 | `traces/llm_local_herbaceous_20260506_141417.json` |
| 2026-05-06 14:20 | local GloVe | LLM | `herbaceous` | solved | 219 | 10 | `herbaceous` / 1 | `traces/llm_local_herbaceous_20260506_142033.json` |
| 2026-05-06 14:23 | local GloVe | LLM | `herbaceous` | not solved | 345 | 20 | `shrub` / 3 | `traces/llm_local_herbaceous_20260506_142357.json` |
| 2026-05-06 15:00 | local GloVe | LLM | `superficial` | not solved | 383 | 20 | `obvious` / 8 | `traces/llm_local_superficial_20260506_150023.json` |
| 2026-05-06 15:10 | local GloVe | LLM | `superficial` | not solved | 316 | 20 | `visceral` / 41 | `traces/llm_local_superficial_20260506_151008.json` |
| 2026-05-06 15:15 | local GloVe | LLM | `superficial` | not solved | 348 | 20 | `subtle` / 13 | `traces/llm_local_superficial_20260506_151506.json` |
| 2026-05-07 13:33 | local GloVe | Ollama `qwen3:14b` | `superficial` | solved | 388 | 15 | `superficial` / 1 | `traces/llm_local_superficial_20260507_133325.json` |

## Self-Adaptive Runs

Columns: target, date, method, max_generations, seed, llm_model, best_rank,
generation_reached, total_guesses, notes, trace_path.

| Target | Date | Method | Max generations | Seed | LLM model | Best rank | Generation reached | Total guesses | Notes | Trace path |
|---|---|---|---:|---:|---|---:|---:|---:|---|---|
| `herbaceous` | 2026-05-21 | `ea_llm_self_adaptive` | 50 | 42 | `qwen3:14b` | 1 | 9 | 284 | Single-run observation. First available end-to-end self-adaptive trace in the repo; solved with `herbaceous`. Pre-crossover-blending and pre-local-search-disable. | [`traces/ea_llm_self_adaptive_local_herbaceous_20260521_214416.json`](../traces/ea_llm_self_adaptive_local_herbaceous_20260521_214416.json) |
| `notorious` | 2026-05-22 | `ea_llm_self_adaptive` | 50 | 42 | `qwen3:14b` | 4 | 50 | 921 | Single-run observation. Reached `gangster` rank 4; inspection showed the best lineage terminating at a crossover child with uniform sigma before crossover blending was added. | [`traces/ea_llm_self_adaptive_local_notorious_20260522_035806.json`](../traces/ea_llm_self_adaptive_local_notorious_20260522_035806.json) |
| `superficial` | 2026-05-25 | `ea_llm_self_adaptive` | 50 | 42 | `qwen3:14b` | 42 | 50 | 827 | Single-run observation. Post mutation-child trace fixes, pre-crossover-blending and pre-local-search-disable. Best rank stayed at `sharpness`/98 from generations 7-33 and later reached `medium`/42 via local search. | [`traces/ea_llm_self_adaptive_local_superficial_20260525_194148.json`](../traces/ea_llm_self_adaptive_local_superficial_20260525_194148.json) |
| `superficial` | 2026-05-26 | `ea_llm_self_adaptive` | 5 | 42 | `qwen3:14b` | 18 | 5 | 175 | Single-run smoke observation. Post-crossover-blending verification trace; all 5 crossover events included `child_sigma_pre_perturbation` and no crossover child had exactly uniform sigma. | [`traces/ea_llm_self_adaptive_local_superficial_20260526_011632.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_011632.json) |
| `superficial` | 2026-05-26 | `ea_llm_self_adaptive` | 50 | 42 | `qwen3:14b` | 5 | 50 | 761 | Single-run observation. Post-crossover-blending, pre-local-search-disable. Reached `thin` rank 5; all 50 crossover events included blended sigma metadata and no crossover child had exactly uniform sigma. | [`traces/ea_llm_self_adaptive_local_superficial_20260526_082930.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_082930.json) |
| `superficial` | 2026-05-26 | `ea_llm_self_adaptive` | 15 | 42 | `qwen3:14b` | 42 | 15 | 387 | Single-run smoke observation. Post-Fix 5 + Fix 6 verification: 0 `LOCAL_SEARCH` events, 1 `LOCAL_SEARCH_DISABLED`, and all 15 crossover children had non-uniform sigma. Best rank `medium`/42 reached at generation 2, with no improvement for 13 generations; the same two parents appeared in every crossover from generation 4 onward. | [`traces/ea_llm_self_adaptive_local_superficial_20260526_135302.json`](../traces/ea_llm_self_adaptive_local_superficial_20260526_135302.json) |

## MAP-Elites Runs

### 2026-06-06 — `superficial`, Seed 4, Qwen3 14B

- Evidence level: single-run observation only. Source trace:
  [`traces/ea_llm_map_elites_local_superficial_20260606_015531.json`](../traces/ea_llm_map_elites_local_superficial_20260606_015531.json).
- Configuration verified from `RUN_CONFIG`: `method=ea_llm_map_elites`,
  local game, target `superficial`, seed `4`, `max_generations=70`,
  Ollama `qwen3:14b`, game backend embeddings
  `data/embeddings/all-MiniLM-L6-v2.npz`. Placement was LLM-driven from
  `AXIS_DEFINITION`/`PLACEMENT` events, not embedding-driven.
- Outcome: NOT SOLVED (`FAILED`). Final `best_word=thin`, `best_rank=5`,
  `total_guesses=845` from the final event. The valid `GUESS` count is `841`;
  the remaining four attempts are `SKIP_INVALID_GUESS` events.
- Best-rank trajectory from `GUESS` events:
  `teacher`/22153 -> `shirt`/10667 -> `crisp`/216 -> `sharp`/170 ->
  `plump`/18 -> `appearance`/14 -> `thin`/5. `thin` first reached rank 5 at
  generation 37.
- Archive state: final `ARCHIVE_SNAPSHOT` at generation 70 has 19/25 cells
  occupied. Empty cells were `(0,4)`, `(1,4)`, `(2,0)`, `(2,4)`, `(3,4)`,
  `(4,4)`; four of the six empty cells are in the top specificity row.
- Saturation: archive occupancy last increased at generation 45 (last
  `ARCHIVE_PLACE`). Best rank last improved at generation 37. Generations
  46-70 were nearly pure churn: 166 `ARCHIVE_REJECT` events and one
  `ARCHIVE_REPLACE`, with no new occupied cells and no best-rank improvement.
- Pipeline counts: successful children are 616 mutation children
  (`OPERATOR_SAMPLED`) + 210 crossover children (`CROSSOVER`) + 15 initial
  children = 841. These match exactly the 841 valid `GUESS` events, 841
  `PLACEMENT` events, and 841 archive outcomes. Archive outcome split:
  19 `ARCHIVE_PLACE`, 48 `ARCHIVE_REPLACE`, and 774 `ARCHIVE_REJECT`. All 841
  valid guessed words are distinct.
- Budget collapse: against a 20-child/generation mutation+crossover budget,
  observed children averaged 18.15/gen in generations 1-20, 12.35/gen in
  generations 21-40, and 7.20/gen in generations 41-70.
- Duplicate-proposal rate is inferred from the budget gap because already-seen
  proposals are skipped before logging. Generations 1-20 created 363/400
  possible children, so 37 attempts (9.25%) produced no child. Generations
  50-70 created 133/420 possible children, so 287 attempts (68.33%) produced no
  child.

## Analysis Artifacts

### 2026-06-08 — Pooled Sigma-Fitness Coupling for MAP-Elites

- Evidence level: pooled / batch-level trace analysis. Reproducer:
  [`scripts/measure_sigma_fitness_coupling.py`](../scripts/measure_sigma_fitness_coupling.py)
  over all MAP-Elites traces in [`traces/`](../traces/) filtered by
  `AXIS_DEFINITION`; fresh command:
  `python scripts/measure_sigma_fitness_coupling.py --report-json _sigma_fitness_coupling_report.json`.
- Scope: 21 MAP-Elites traces, 10,581 linked mutation children. Crossover
  children were excluded because they do not sample a single operator. Phases
  were split per run at the occupancy-freeze generation, defined as the final
  `ARCHIVE_PLACE` generation.
- Linkage method: `OPERATOR_SAMPLED.details.sampled_op` -> child archive
  outcome by child hypothesis ID (`ARCHIVE_PLACE.hypothesis_id`,
  `ARCHIVE_REPLACE.new_hypothesis_id`, or
  `ARCHIVE_REJECT.child_hypothesis_id`) -> child rank from the archive outcome
  -> parent rank via `parent_id`.
- Measurement A, outcome counts and rates:
  `s_mutation` N=2274, PLACE 60 (2.6%), REPLACE 321 (14.1%), REJECT 1893
  (83.2%); `m_mutation` N=2574, PLACE 53 (2.1%), REPLACE 236 (9.2%), REJECT
  2285 (88.8%); `ml_mutation` N=2388, PLACE 48 (2.0%), REPLACE 221 (9.3%),
  REJECT 2119 (88.7%); `l_mutation` N=3345, PLACE 32 (1.0%), REPLACE 160
  (4.8%), REJECT 3153 (94.3%).
- Measurement B, pooled `delta = log(parent_rank) - log(child_rank)`:
  median delta `s=-1.34`, `m=-2.07`, `ml=-1.70`, `l=-2.69`; improvement rates
  `s=23%`, `m=19%`, `ml=20%`, `l=13%`; jackpot rates (`delta >= ln(10)`)
  `s=4%`, `m=3%`, `ml=3%`, `l=2%`; at-least-10x-worse rates
  `s=35%`, `m=47%`, `ml=41%`, `l=56%`.
- Phase split: in early/growth, median delta was `s=-1.30`, `m=-1.85`,
  `ml=-1.46`, `l=-2.35`; in late/post-freeze, median delta was `s=-1.67`,
  `m=-2.87`, `ml=-2.57`, `l=-3.58`.
- Parent-rank confound: `l_mutation` drew from better-ranked parents on average
  (median parent rank 4,332) than `s_mutation` (median parent rank 8,846), but
  the operator ordering survived parent-rank tercile control. Low parent-rank
  tercile median delta: `s=-3.90`, `m=-4.92`, `ml=-4.62`, `l=-5.38`; high
  tercile median delta: `s=-0.28`, `m=-0.37`, `ml=-0.35`, `l=-0.61`.
- Step-0 sigma confirmation: final archive-incumbent sigma averaged per run over
  the same 21 traces was `[s=0.271, m=0.234, ml=0.232, l=0.263]`, so the earlier
  `sigma_l ~= 0.45` elevation is not a pooled final-archive pattern; it remains
  a single-run observation from the `superficial` seed-4 run.
- Finding link:
  [`docs/findings.md`](findings.md#2026-06-08--pooled-map-elites-sigma-fitness-coupling-favors-small-mutation).

### 2026-05-21 — Trajectory Visualizations for `superficial` Traces

- Evidence level: qualitative single-run visualization, not a repeated-run or
  batch-level performance result.
- Code: [`contexto_solver/plot_trajectory.py`](../contexto_solver/plot_trajectory.py).
- Compared traces:
  [`llm_local_superficial_20260507_133325.json`](../traces/llm_local_superficial_20260507_133325.json)
  solved with Ollama `qwen3:14b`, and
  [`llm_local_superficial_20260506_151008.json`](../traces/llm_local_superficial_20260506_151008.json)
  failed with best word `visceral`, rank 41.
- Generated rank plots:
  [`llm_local_superficial_20260507_133325_rank.png`](../figures/llm_local_superficial_20260507_133325_rank.png)
  and
  [`llm_local_superficial_20260506_151008_rank.png`](../figures/llm_local_superficial_20260506_151008_rank.png).
- Generated distance plots using `data/glove.6B.300d.txt`:
  [`llm_local_superficial_20260507_133325_distance.png`](../figures/llm_local_superficial_20260507_133325_distance.png)
  and
  [`llm_local_superficial_20260506_151008_distance.png`](../figures/llm_local_superficial_20260506_151008_distance.png).
- Generated PaCMAP trajectory plots using `data/glove.6B.300d.txt`:
  [`llm_local_superficial_20260507_133325_pacmap.png`](../figures/llm_local_superficial_20260507_133325_pacmap.png)
  and
  [`llm_local_superficial_20260506_151008_pacmap.png`](../figures/llm_local_superficial_20260506_151008_pacmap.png).
- Visual checks: the solved rank plot reaches rank 1, the solved distance plot
  reaches cosine distance 0, and the solved PaCMAP run reported a target/guess
  coordinate delta of 0 for `superficial`. These are consistency checks for the
  visualization pipeline and the two selected traces, not general claims about
  solver performance.

## Batch Experiments

### 2026-05-19 — HPC Pivot Evaluation Matrix, Qwen3 14B

- Evidence level: replicated batch-level comparison on cloud compute resources,
  15 paired runs across `notorious`, `herbaceous`, and `superficial`, pivot
  off/on, Ollama `qwen3:14b`, aligned local game, and a 50-generation cap.
- Analysis outputs:
  [`pivot_matrix_20260519_hpc_analysis.json`](../traces/pivot_matrix_20260519_hpc_analysis.json),
  [`pivot_matrix_20260519_hpc_condition_stats.csv`](../traces/pivot_matrix_20260519_hpc_condition_stats.csv),
  [`pivot_matrix_20260519_hpc_paired_stats.csv`](../traces/pivot_matrix_20260519_hpc_paired_stats.csv),
  and [`pivot_matrix_20260519_hpc_combined_runs.csv`](../traces/pivot_matrix_20260519_hpc_combined_runs.csv).
- Aggregate result from `pivot_matrix_20260519_hpc_condition_stats.csv`: pivot
  off solved 9/15 runs (60%), median 633 solved-run guesses, and median 41
  generations. Pivot on solved 10/15 runs (67%), median 270 solved-run guesses,
  and median 18 generations.
- Paired statistics from `pivot_matrix_20260519_hpc_paired_stats.csv`: solved-run
  guesses improved with Wilcoxon p=0.03125 and Cliff's delta -0.796 over seven
  paired solved comparisons. Generations improved with Wilcoxon p=0.04977 and
  Cliff's delta -0.356 over all 15 pairs.
- Per-target result from `pivot_matrix_20260519_hpc_condition_stats.csv`:
  `herbaceous` strengthened under pivot on (4/5 -> 5/5 solved; median
  generations 41 -> 8). `notorious` remained hard in both conditions (1/5 solved
  and median 50 generations), but failed-run best-rank spread narrowed
  substantially (unsolved best-rank IQR 60.5 -> 2.5). `superficial` was more
  mixed than in the earlier local matrix (both conditions solved 4/5; median
  generations 29 -> 12).
- Finding link:
  [`docs/findings.md`](findings.md#2026-05-19--hpc-pivot-replication-strengthens-aggregate-speed-claim-but-weakens-per-target-certainty).

### 2026-05-13 — Pivot Evaluation Matrix, Qwen3 14B

- Evidence level: batch-level repeated-run comparison, 15 paired runs across
  three targets (`notorious`, `herbaceous`, `superficial`) with pivot off/on,
  Ollama `qwen3:14b`, aligned local GloVe game, and a 50-generation cap.
- Command pattern: `python -m contexto_solver.experiment --solver llm --provider ollama --ollama-model qwen3:14b --mode aligned --targets notorious,herbaceous,superficial --runs-per-target 5 --max-generations 50 --llm-workers 1 --output traces/pivot_matrix_<condition>.json --resume`.
- Condition summaries: [`pivot_matrix_off.json`](../traces/pivot_matrix_off.json),
  [`pivot_matrix_off.csv`](../traces/pivot_matrix_off.csv),
  [`pivot_matrix_on.json`](../traces/pivot_matrix_on.json),
  [`pivot_matrix_on.csv`](../traces/pivot_matrix_on.csv).
- Analysis outputs: [`pivot_matrix_analysis.json`](../traces/pivot_matrix_analysis.json),
  [`pivot_matrix_condition_stats.csv`](../traces/pivot_matrix_condition_stats.csv),
  [`pivot_matrix_paired_stats.csv`](../traces/pivot_matrix_paired_stats.csv),
  [`pivot_matrix_combined_runs.csv`](../traces/pivot_matrix_combined_runs.csv).
- Aggregate result from `pivot_matrix_condition_stats.csv`: pivot off solved
  8/15 runs (53%), median 582 solved-run guesses, and median 41 generations.
  Pivot on solved 10/15 runs (67%), median 247 solved-run guesses, and median
  12 generations.
- Per-target result from `pivot_matrix_condition_stats.csv`: `herbaceous` solved
  4/5 in both conditions, but pivot on was roughly 3x faster by medians
  (220 vs 645 solved-run guesses, 9 vs 34 generations). `superficial` improved
  from 3/5 solved with pivot off to 5/5 solved with pivot on, with median
  generations dropping from 38 to 9. `notorious` stayed difficult: both
  conditions solved 1/5 and had median 50 generations, while failed pivot-off
  runs had median best rank 35 (IQR 60.5) and failed pivot-on runs had median
  best rank 7 (IQR 0.5).
- Paired statistics from `pivot_matrix_paired_stats.csv`: generations improved
  with Wilcoxon p=0.0497 and Cliff's delta -0.42, a medium effect. Solved-run
  guesses improved with Wilcoxon p=0.09375 and Cliff's delta -0.72, a large
  effect estimate, but only six paired solved-run comparisons were available.
- Finding link: [`docs/findings.md`](findings.md#2026-05-13--pivot-matrix-shows-faster-stall-recovery-but-not-a-complete-unblock).

## Finding Links

- Near-target stagnation and misleading neighborhoods: `docs/findings.md`.
- Design rationale for local search, deduplication, and pivoting:
  `docs/design_decisions.md`.
- Chronological context: `docs/research_timeline.md`.

