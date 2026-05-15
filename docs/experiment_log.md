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

## Batch Experiments

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

