# Camera-ready analysis (NeurIPS 2026 workshop, submission 227)

What the reviews asked for and what each file here does. Everything runs from the repository root.

## The one rule change

Report accuracy is scored against the event the self-report prompt names: whether the **best of the
operator's own proposed words** reaches the top 100 (and the bucket of that best word). The submitted
paper scored the first graded word. Policy: a word the run had already guessed keeps the rank the
run returned (the solver re-grades it from its cache and counts it toward the child); a proposal
whose every word was invalid counts as outside the top 100; a proposal with no graded word for any
other reason is excluded. Proposed words come from the operator's raw JSON and are matched to their
own graded outcomes, which also removes the same-name pooling of the old join (34 / 14 / 27 records
for Qwen / Gemma / Ministral). Code: `contexto_solver/rq1/reader.py` (word-level join),
`contexto_solver/rq1/metrics.py` (`which="best_proposed"`, now the default), `scripts/rq3_lineage.py`
(same rule, plus run-resampled 95 % intervals).

## Regenerate every number

```
python scripts/camera_ready.py --out camera_ready_outputs_2026-10-27
```

Takes about 15 minutes (three selection analyses with 10,000 run resamples each). Add
`--resample traces/rq2_resample` once the second genuine sample has been run on the cluster (below).
Output layout is the one the paper's figure scripts read:

| file | read by |
|---|---|
| `A1_augmented.csv`, `A2_…`, `A4_…`, `A5_…` | `fig2_reliability.py` (column `realized_rank_used`) |
| `report_A1/per_run_metrics.csv` (and A2, A4, A5) | `fig3_independence.py`, `figS_auroc.py`, `fig3_dissociation_shared5.py` |
| `rq3_A1_3games/rq3_summary.json`, `rq3_random/rq3_summary.json` | `fig5_selection.py` (now with `ci95`) |
| `rq2_hl_bootstrap.json` | `fig4_replay.py` (rows incl. `genuine_resample` when present) |
| `table2.md` / `table2.json` | Table 2 (medians, paired Wilcoxon, Holm over eight metrics) |
| `rq3_A1/rq3_summary.json` | Table 3, first column |
| `direct_qwen_reweight.json` | the Direct-Qwen comparison, strata stated explicitly |
| `submitted_rule/…` | the same outputs under the submitted first-word rule |
| `CHANGES.md` | submitted versus camera-ready numbers, for the appendix erratum |

Copy the output folder next to the paper's `figures/` folder as `camera_ready_outputs`, or set
`CONTEXTO_ART_DIR` to its path, and run the patched figure scripts in `camera_ready/figures/`
(`fig2_reliability.py`, `fig4_replay.py`, `fig5_selection.py` replace the paper's copies; the others
are unchanged).

## The second genuine sample (cluster)

```
sbatch run_rq2_resample.sh          # 10 jobs, one per game, about an hour each
```

One fresh call per stored intervention event with the genuine rationale (754 calls), written to
`traces/rq2_resample/<game>/`. Then rerun `scripts/camera_ready.py` with `--resample
traces/rq2_resample`: Figure 4 gains the "Genuine, resampled" row and a shaded band (sampling
variation alone), and the paper reads every replacement against it.

## Cross-check

`camera_ready/rescore_crosscheck.py` re-derives the calibration table without the reader
(independent parsing and matching); its numbers match the pipeline's to the last digit.

## Tests

```
python -m pytest -q tests/test_rq1_analysis.py tests/test_rq3_lineage.py tests/test_rq2_replay.py
```
