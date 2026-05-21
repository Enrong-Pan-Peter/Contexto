# Findings

This document records research-facing findings for the Contexto solver project.
It is intentionally selective: raw run details belong in `docs/experiment_log.md`,
technical structure belongs in `docs/architecture.md`, and design rationale
belongs in `docs/design_decisions.md`.

Entries are in reverse chronological order. Claims are phrased according to the
current evidence level; unresolved or incomplete results are marked explicitly.

## 2026-05-19 — HPC Pivot Replication Strengthens Aggregate Speed Claim But Weakens Per-Target Certainty

The cloud-compute replication of the pivot matrix supports the same aggregate
direction as the earlier local matrix: enabling pivots reduces solved-run guess
counts, reduces generation counts, and narrows failed-run variance. The strongest
paper-level claim is therefore aggregate and batch-level: on these two small
matrices, the stall-pivot method appears to improve speed and stabilize failures,
but it does not reliably solve the hardest target.

Evidence:
- Experiment log entry:
  [`2026-05-19 HPC Pivot Evaluation Matrix`](experiment_log.md#2026-05-19--hpc-pivot-evaluation-matrix-qwen3-14b).
- HPC analysis outputs:
  [`pivot_matrix_20260519_hpc_analysis.json`](../traces/pivot_matrix_20260519_hpc_analysis.json),
  [`pivot_matrix_20260519_hpc_condition_stats.csv`](../traces/pivot_matrix_20260519_hpc_condition_stats.csv),
  [`pivot_matrix_20260519_hpc_paired_stats.csv`](../traces/pivot_matrix_20260519_hpc_paired_stats.csv), and
  [`pivot_matrix_20260519_hpc_combined_runs.csv`](../traces/pivot_matrix_20260519_hpc_combined_runs.csv).
- Earlier local comparison outputs:
  [`pivot_matrix_analysis.json`](../traces/pivot_matrix_analysis.json),
  [`pivot_matrix_condition_stats.csv`](../traces/pivot_matrix_condition_stats.csv),
  [`pivot_matrix_paired_stats.csv`](../traces/pivot_matrix_paired_stats.csv), and
  [`pivot_matrix_combined_runs.csv`](../traces/pivot_matrix_combined_runs.csv).
- Aggregate HPC condition stats: pivot off solved 9/15 runs with median 633
  solved-run guesses and median 41 generations; pivot on solved 10/15 runs with
  median 270 solved-run guesses and median 18 generations.
- Aggregate HPC paired stats: solved-run guesses improved with Wilcoxon p=0.03125
  and Cliff's delta -0.796 over seven paired solved comparisons. Generations
  improved with Wilcoxon p=0.04977 and Cliff's delta -0.356 over all 15 paired
  runs.
- Compared with the earlier local matrix, solve rates stayed within one run
  (`8/15 -> 10/15` locally, `9/15 -> 10/15` on HPC), and the generations effect
  stayed directionally consistent (median difference -19 locally, -11 on HPC;
  both p approximately 0.05).

Robust patterns across the local and HPC matrices:
- Aggregate speed improvement is the most stable result. The solved-guess paired
  test strengthened from borderline local evidence (p=0.09375, Cliff's delta
  -0.72, six pairs) to statistically significant HPC evidence (p=0.03125,
  Cliff's delta -0.796, seven pairs).
- `notorious` remains unsolved by the pivot mechanism as a hard case: both
  matrices show 1/5 solved in both conditions and median 50 generations. The more
  stable observation is variance collapse among failed `notorious` runs, not
  reliable solving.
- `herbaceous` is the cleanest per-target illustration of the pivot effect in the
  HPC data: solve rate improved from 4/5 to 5/5, median generations dropped from
  41 to 8, and the generation Cliff's delta was -1.0.

Unstable or weaker patterns:
- `superficial` is not a robust per-target success claim. In the earlier local
  matrix it was the strongest pivot result; in the HPC matrix both conditions
  solved 4/5, and the generation effect attenuated to median difference -4 with
  Cliff's delta -0.40.
- `notorious` solved-run guess counts are too unstable to quote as a target-level
  performance estimate. The pivot-on condition has only one solved run in each
  matrix, and the solved count changed from 159 locally to 657 on HPC.
- Per-target solve rates are still noisy at n=5 per target. They should be used
  as illustrative evidence, not as standalone claims.

Interpretation: the HPC matrix strengthens the aggregate claim that pivots help
speed and failure stability, and it gives one aggregate metric with p<0.05 where
the local matrix was borderline. At the same time, the `superficial` swing shows
that narrow per-target claims are underpowered. The writeup should emphasize the
aggregate paired result and use target-level outcomes as explanatory examples
rather than definitive target-specific conclusions.

## 2026-05-13 — Pivot Matrix Shows Faster Stall Recovery But Not a Complete Unblock

The completed pivot evaluation matrix provides batch-level evidence that the
stall-pivot mechanism improves solver speed and reduces the spread of failed-run
outcomes on the tested local GloVe targets. The evidence comes from 15 paired
runs across three targets (`notorious`, `herbaceous`, `superficial`), comparing
`ENABLE_PIVOT=false` and `ENABLE_PIVOT=true` with Ollama `qwen3:14b` and a
50-generation cap.

Evidence:
- Experiment log entry:
  [`2026-05-13 Pivot Evaluation Matrix`](experiment_log.md#2026-05-13--pivot-evaluation-matrix-qwen3-14b).
- Raw condition summaries:
  [`pivot_matrix_off.json`](../traces/pivot_matrix_off.json),
  [`pivot_matrix_on.json`](../traces/pivot_matrix_on.json),
  [`pivot_matrix_off.csv`](../traces/pivot_matrix_off.csv), and
  [`pivot_matrix_on.csv`](../traces/pivot_matrix_on.csv).
- Analysis outputs:
  [`pivot_matrix_analysis.json`](../traces/pivot_matrix_analysis.json),
  [`pivot_matrix_condition_stats.csv`](../traces/pivot_matrix_condition_stats.csv),
  [`pivot_matrix_paired_stats.csv`](../traces/pivot_matrix_paired_stats.csv), and
  [`pivot_matrix_combined_runs.csv`](../traces/pivot_matrix_combined_runs.csv).
- Aggregate condition stats: pivot off solved 8/15 runs (53%) with median
  582 solved-run guesses and 41 generations; pivot on solved 10/15 runs (67%)
  with median 247 solved-run guesses and 12 generations.
- Paired stats: generations improved with Wilcoxon p=0.0497 and Cliff's delta
  -0.42, a medium effect. Solved-run guesses improved with Wilcoxon p=0.09375
  and Cliff's delta -0.72, a large effect estimate, but this metric only had
  six paired solved-run comparisons and is underpowered.
- Failed-run variance collapsed most clearly on `notorious`: failed pivot-off
  runs had median best rank 35 with IQR 60.5, while failed pivot-on runs had
  median best rank 7 with IQR 0.5.

Interpretation: this is stronger than prior single-run or repeated same-target
evidence because it is a paired batch-level comparison. Pivots reliably improve
runtime efficiency on this small matrix, especially for `herbaceous` and
`superficial`, and make failed runs land in a narrower near-target band.

Limitation: the same evidence also shows that pivots do not unblock genuinely
hard targets by themselves. `notorious` remained 1/5 solved in both conditions
and both conditions hit the 50-generation median cap, so further direction
selection or diversity mechanisms are still needed before claiming robust
solution of the hardest stagnation mode.

## 2026-05-08 — Pivot Matrix Evaluation Criteria

The stall-pivot mechanism is being evaluated with a paired local-game matrix
rather than isolated single runs. The design compares `ENABLE_PIVOT=false` and
`ENABLE_PIVOT=true` using the same targets (`notorious`, `herbaceous`,
`superficial`), five repeats per target, Ollama `qwen3:14b`, and a 50-generation
budget.

Evaluation criteria:
- Pair runs by `(target, run_index)` so each pivot-enabled run is compared with
  the corresponding no-pivot run.
- Report solve rate per condition and per target.
- Report median and IQR of guesses-to-solution using solved runs only.
- For unsolved runs, report the best rank reached rather than treating the run
  as if it had a solved guess count.
- Use Wilcoxon signed-rank tests for paired differences and Cliff's delta as a
  nonparametric effect-size estimate.

Current status: superseded by the completed 2026-05-13 matrix finding above.

## 2026-05-08 — Failure Modes Across Stagnation-Prone Targets

The difficult local targets studied so far (`notorious`, `herbaceous`, and
`superficial`) repeatedly produce strong but incomplete clues under the local
GloVe game.

Observed failure modes:
- **Neighborhood lock-in:** For `notorious`, several runs moved into crime or
  organized-group language (`crime`, `gang`, `gangster`, `mafia`) but did not
  reliably pivot to the reputational adjective.
- **Descriptor miss after a close noun:** For `herbaceous`, the solver can reach
  `shrub` or `shrubs` at rank 2-3, but still fail to identify the botanical
  descriptor relation needed for `herbaceous`.
- **Misleading contrast or association:** For `superficial`, good ranks can come
  from contrastive or associated words such as `subtle`, `obvious`, and
  `visceral`, which pull the LLM into plausible but incorrect explanatory
  frames.

Interpretation: stagnation is not a single failure mode. A high-ranking word may
be close by class membership, descriptor relation, antonymy/contrast, or loose
association. The pivot mechanism should therefore be evaluated as an intervention
for relation-shifting rather than only as a generic exploration boost.

## 2026-05-07 — Ollama `qwen3:14b` Validates Local LLM Experimentation

Local Ollama support was validated on the local GloVe game with the LLM
evolutionary solver, target `superficial`, and a 50-generation budget.

Evidence:
- Trace: `traces/llm_local_superficial_20260507_133325.json`
- Provider/model: Ollama `qwen3:14b`
- Result: solved in 388 guesses over 15 generations.
- Path: the run moved through body/surface clues including `skin`, `epidermis`,
  and `vein`; local search around `vein` eventually proposed `superficial`.

Interpretation: this run shows that local Ollama execution is operationally
viable for long experiments and can solve a previously difficult target. It is
not a stable performance estimate because it is a single stochastic run. Its
main value is methodological: it reduces dependence on cloud API quota and makes
overnight local evaluation practical.

## 2026-05-06 — `superficial` Shows Misleading Close Neighborhoods

Three local LLM runs on `superficial` failed despite reaching apparently useful
rank neighborhoods.

Evidence:
- `traces/llm_local_superficial_20260506_150023.json`: not solved after 383
  guesses over 20 generations; best word `obvious`, rank 8.
- `traces/llm_local_superficial_20260506_151008.json`: not solved after 316
  guesses over 20 generations; best word `visceral`, rank 41.
- `traces/llm_local_superficial_20260506_151506.json`: not solved after 348
  guesses over 20 generations; best word `subtle`, rank 13.

Observation: the local GloVe neighborhood exposes multiple misleading routes:
abstract contrast words such as `subtle` and `obvious`, and body/depth contrast
words such as `visceral`.

Interpretation: good rank feedback can still pull the LLM into an incorrect
explanatory frame when the embedding relation reflects contrast or association
rather than the exact target descriptor. A useful pivot for this target likely
needs to move toward surface/depth descriptors rather than generic subtlety or
deeper anatomy terms.

## 2026-05-06 — `herbaceous` Exposes Singular/Plural Redundancy

Repeated local LLM runs on `herbaceous` showed that close singular/plural forms
can consume search effort without producing a meaningful semantic pivot.

Evidence:
- `traces/llm_local_herbaceous_20260506_141417.json`: not solved after 274
  guesses over 20 generations; best word `shrubs`, rank 2. The run followed
  `tree -> plant -> vegetation -> shrub -> shrubs` and then stalled.
- `traces/llm_local_herbaceous_20260506_142033.json`: solved in 219 guesses over
  10 generations after moving from `shrub` toward `plant forms`.
- `traces/llm_local_herbaceous_20260506_142357.json`: not solved after 345
  guesses over 20 generations; best word `shrub`, rank 3.

Interpretation: Contexto-like gameplay treats singular/plural variants as
effectively redundant. The solver should avoid treating `shrub -> shrubs` as a
useful search direction. This finding motivated explicit prompt instructions and
lightweight singular/plural family filtering in candidate acceptance.

## 2026-05-05 — Same-Target Variance on `herbaceous`

Three local LLM runs on the same target (`herbaceous`) produced a fast solve, a
rank-3 stall, and a later solve under the same local game and LLM setup.

Evidence:
- `traces/llm_local_herbaceous_20260505_122620.json`: solved in 101 guesses over
  4 generations.
- `traces/llm_local_herbaceous_20260505_130240.json`: not solved after 311
  guesses over 20 generations; best word `shrub`, rank 3.
- `traces/llm_local_herbaceous_20260505_130555.json`: solved in 215 guesses over
  10 generations.

Interpretation: this target demonstrates that the remaining variance is not only
about reaching a good neighborhood. The solver also needs a reliable way to move
from a close clue (`shrub`) to the exact target relation or descriptor
(`herbaceous`).

## 2026-05-04 — Active Cap, Deduplication, and Divergent Mutation Reduced Early Search Bloat

Early LLM runs showed hypothesis bloat and redundant mutation paths. The first
documented local `cat` run solved in 468 guesses over 13 generations, but by
generation 7 the solver had accumulated many active or near-duplicate
hypotheses. The word `bite` scored well but mutation stayed mostly in food and
measurement interpretations before eventually reaching the animal neighborhood.

Changes applied afterward:
- Capped active hypotheses at `MAX_ACTIVE_HYPOTHESES`.
- Added deduplication for near-identical hypotheses.
- Changed mutation prompts to request divergent interpretations of strong clues,
  not only narrower subcategories.

Post-change evidence:
- `traces/llm_local_cat_20260504_140013.json`: solved during initialization in 2
  guesses. This confirms immediate stopping, but is not a useful convergence
  benchmark.
- `traces/llm_local_house_20260504_140109.json`: solved in 107 guesses over 4
  generations; mutation found a legislative-body interpretation via `senate`.

Interpretation: trace inspection supports that capping and deduplication reduced
redundant hypothesis growth, and divergent mutation can help sense shifts.
However, the evidence is still small-sample and should be framed as
implementation validation rather than a robust performance estimate.

## 2026-05-04 — Local vs Online Performance Gap Is Plausible But Unconfirmed

Early runs suggested that the LLM evolutionary solver sometimes performed better
against the online Contexto API than against the local GloVe game, even though
the LLM solver does not directly inspect either backend's embedding model.

Evidence at the time included real API solves such as `ivory` in 165 guesses and
`sponges` in 254 guesses, while several local GloVe targets remained difficult
or unsolved within the generation budget.

Interpretation: this may indicate a mismatch between LLM-generated semantic
neighborhoods and GloVe's local ranking geometry, or a closer implicit alignment
between the LLM and the real Contexto backend. This claim is currently weak
because the compared targets and conditions differ. It needs controlled repeated
experiments before being used as a paper-level result.
