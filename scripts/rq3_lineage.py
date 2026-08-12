"""RQ3 lineage analysis: survivor vs culled agreement over batch traces (offline).

Units are the self-reported mutation children (``OPERATOR_SAMPLED`` records) of
instrumented self-adaptive runs. For each individual the script computes two
per-individual agreement measures (from the RQ1 semantics):

- ``binary_error``: |predicted_closeness - 1{realized_rank <= 100}|;
- ``bucket_distance``: |ordinal(predicted_bucket) - ordinal(realized bucket)|.

Survivor labels come from the trace's ``SELECT`` events (name-based kept /
discarded lists). A child born in generation g is labeled by the FIRST SELECT
after its birth (generation g+1) only; later resurrections of a discarded name
are counted separately and never relabel. Excluded from labels (all counted):
children dedup-merged away before their first SELECT, final-generation births
(no subsequent SELECT), and name-collision-ambiguous matches.

Analyses:

- survivor-vs-culled gap in each agreement measure within matched parent-rank
  bins, with a within-(generation x parent-rank-bin) permutation null (predicted
  values permuted across individuals; B seeded permutations) - the guard against
  selection-on-outcome mechanically shifting agreement among survivors;
- transmission: parent-child correlation of agreement along parent_id ->
  child_id lineage links, against the same within-cell permutation baseline;
- generation trend of agreement at matched parent-rank bins.

No network, no LLM, no writes to traces or caches.

Usage (PowerShell):

    python scripts/rq3_lineage.py traces/rq1_A1/ea_llm_self_adaptive_api_*.json `
        --output traces/rq3_lineage_A1 --seed 0 --permutations 1000
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from scipy import stats

from contexto_solver.rq1.metrics import (
    POSITIVE_RANK_THRESHOLD,
    _parent_rank_bin,
    bucket_ordinal,
    realized_bucket,
)
from contexto_solver.rq1.reader import (
    _first_valid,
    _named_guess_events,
    _proposed_from_events,
    load_trace,
    run_config,
)
from contexto_solver.self_report import read_self_report

MEASURES = ("binary_error", "bucket_distance")

EXCLUSION_DEDUP = "dedup_merged_before_select"
EXCLUSION_FINAL_GEN = "no_subsequent_select"
EXCLUSION_COLLISION = "name_collision"
EXCLUSION_MISSING = "missing_from_select"


@dataclass
class LineageIndividual:
    """One self-reported mutation child with realized outcome and survivor label."""

    trace_file: str
    child_id: str
    child_name: str | None
    generation: int
    parent_id: str | None
    parent_rank: int | None
    parent_rank_bin: str
    sampled_op: str | None
    predicted_closeness: float | None
    predicted_bucket: str | None
    self_report_parse_failed: bool
    realized_rank_first: int | None
    child_best_rank: int | None
    realized_rank_used: int | None  # child_best policy with first-proposed fallback
    proposed_word_invalid: bool
    label: str | None = None  # "survivor" | "culled" | None (excluded)
    exclusion_reason: str | None = None
    resurrected: bool = False

    @property
    def binary_error(self) -> float | None:
        if self.predicted_closeness is None or self.realized_rank_used is None:
            return None
        label = 1.0 if self.realized_rank_used <= POSITIVE_RANK_THRESHOLD else 0.0
        return abs(float(self.predicted_closeness) - label)

    @property
    def bucket_distance(self) -> float | None:
        predicted = bucket_ordinal(self.predicted_bucket)
        actual = bucket_ordinal(realized_bucket(self.realized_rank_used))
        if predicted is None or actual is None:
            return None
        return float(abs(predicted - actual))

    def measure(self, name: str) -> float | None:
        if name == "binary_error":
            return self.binary_error
        if name == "bucket_distance":
            return self.bucket_distance
        raise ValueError(f"unknown measure: {name!r}")

    def cell(self) -> tuple[str, int, str]:
        """Permutation cell: (trace, generation, parent-rank bin)."""
        return (self.trace_file, self.generation, self.parent_rank_bin)


# --- extraction ------------------------------------------------------------------


def extract_lineage(events: list[dict[str, Any]], trace_file: str) -> tuple[list[LineageIndividual], dict[str, Any]]:
    """All self-reported mutation children with survivor labels for one trace."""
    trace_config = run_config(events)
    individuals: list[LineageIndividual] = []

    select_by_gen: dict[int, tuple[list[str], list[str]]] = {}
    dedup_discarded_by_gen: dict[int, list[str]] = {}
    for event in events:
        generation = event.get("generation")
        details = event.get("details", {}) or {}
        if event.get("event") == "SELECT":
            select_by_gen[int(generation)] = (
                [str(name) for name in details.get("kept", [])],
                [str(name) for name in details.get("discarded", [])],
            )
        elif event.get("event") == "DEDUPLICATE":
            names = dedup_discarded_by_gen.setdefault(int(generation), [])
            for merge in details.get("merged", []) or []:
                if isinstance(merge, dict) and merge.get("discarded"):
                    names.append(str(merge["discarded"]))

    birth_names: dict[tuple[int, str], int] = {}
    for event in events:
        if event.get("event") != "OPERATOR_SAMPLED":
            continue
        details = event.get("details", {}) or {}
        if not isinstance(details.get("self_report"), dict):
            continue
        generation = int(event.get("generation", -1))
        child_name = details.get("child_hypothesis_name")
        report = read_self_report({"self_report": details["self_report"]})
        guesses = _named_guess_events(events, child_name, generation)
        first_word, first_rank, invalid = _proposed_from_events(guesses)
        _best_word, best_rank = _first_valid(guesses)
        parent_rank = details.get("parent_rank") if isinstance(details.get("parent_rank"), int) else None
        individuals.append(
            LineageIndividual(
                trace_file=trace_file,
                child_id=str(details.get("child_id")),
                child_name=str(child_name) if child_name else None,
                generation=generation,
                parent_id=details.get("parent_id"),
                parent_rank=parent_rank,
                parent_rank_bin=_parent_rank_bin(parent_rank),
                sampled_op=details.get("sampled_op"),
                predicted_closeness=report.get("predicted_closeness"),
                predicted_bucket=report.get("predicted_bucket"),
                self_report_parse_failed=bool(report.get("self_report_parse_failed")),
                realized_rank_first=first_rank,
                child_best_rank=best_rank,
                realized_rank_used=best_rank if best_rank is not None else first_rank,
                proposed_word_invalid=bool(invalid),
            )
        )
        if child_name:
            key = (generation, str(child_name))
            birth_names[key] = birth_names.get(key, 0) + 1

    # --- labeling: first SELECT after birth only ---
    counts = {
        "individuals": len(individuals),
        "survivor": 0,
        "culled": 0,
        EXCLUSION_DEDUP: 0,
        EXCLUSION_FINAL_GEN: 0,
        EXCLUSION_COLLISION: 0,
        EXCLUSION_MISSING: 0,
        "resurrections": 0,
    }
    for individual in individuals:
        generation = individual.generation
        name = individual.child_name
        if name is None:
            individual.exclusion_reason = EXCLUSION_MISSING
            counts[EXCLUSION_MISSING] += 1
            continue
        if birth_names.get((generation, name), 0) > 1:
            individual.exclusion_reason = EXCLUSION_COLLISION
            counts[EXCLUSION_COLLISION] += 1
            continue
        if name in dedup_discarded_by_gen.get(generation, []):
            individual.exclusion_reason = EXCLUSION_DEDUP
            counts[EXCLUSION_DEDUP] += 1
            continue
        select = select_by_gen.get(generation + 1)
        if select is None:
            individual.exclusion_reason = EXCLUSION_FINAL_GEN
            counts[EXCLUSION_FINAL_GEN] += 1
            continue
        kept, discarded = select
        occurrences = kept.count(name) + discarded.count(name)
        if occurrences > 1:
            individual.exclusion_reason = EXCLUSION_COLLISION
            counts[EXCLUSION_COLLISION] += 1
            continue
        if occurrences == 0:
            individual.exclusion_reason = EXCLUSION_MISSING
            counts[EXCLUSION_MISSING] += 1
            continue
        if name in kept:
            individual.label = "survivor"
            counts["survivor"] += 1
        else:
            individual.label = "culled"
            counts["culled"] += 1
            # Resurrection: the culled name reappears in a LATER kept list.
            for later_gen, (later_kept, _later_discarded) in select_by_gen.items():
                if later_gen > generation + 1 and name in later_kept:
                    individual.resurrected = True
                    counts["resurrections"] += 1
                    break

    meta = {
        "trace_file": trace_file,
        "game_number": trace_config.game_number,
        "provenance_hash": trace_config.provenance_hash,
        "counts": counts,
    }
    return individuals, meta


# --- matched survivor-vs-culled gap -----------------------------------------------


def matched_gap(individuals: list[LineageIndividual], measure: str) -> tuple[float | None, list[dict[str, Any]]]:
    """(weighted gap over parent-rank bins, per-bin rows).

    Gap per bin = mean(measure | survivor) - mean(measure | culled); the overall
    gap is the bin-size-weighted mean over bins where BOTH sides are non-empty.
    Positive gap = survivors have LARGER error (worse agreement) than culled.
    """
    by_bin: dict[str, dict[str, list[float]]] = {}
    for individual in individuals:
        if individual.label not in ("survivor", "culled"):
            continue
        value = individual.measure(measure)
        if value is None:
            continue
        by_bin.setdefault(individual.parent_rank_bin, {"survivor": [], "culled": []})[individual.label].append(value)

    rows: list[dict[str, Any]] = []
    weighted_sum = 0.0
    weight_total = 0
    for bin_label in sorted(by_bin):
        survivors = by_bin[bin_label]["survivor"]
        culled = by_bin[bin_label]["culled"]
        gap = None
        if survivors and culled:
            gap = sum(survivors) / len(survivors) - sum(culled) / len(culled)
            weight = len(survivors) + len(culled)
            weighted_sum += gap * weight
            weight_total += weight
        rows.append(
            {
                "measure": measure,
                "parent_rank_bin": bin_label,
                "n_survivor": len(survivors),
                "n_culled": len(culled),
                "mean_survivor": (sum(survivors) / len(survivors)) if survivors else None,
                "mean_culled": (sum(culled) / len(culled)) if culled else None,
                "gap": gap,
            }
        )
    overall = (weighted_sum / weight_total) if weight_total else None
    return overall, rows


def _permutation_pool(individuals: list[LineageIndividual], measure: str) -> list[LineageIndividual]:
    """Individuals eligible for the within-cell permutation of predicted values."""
    pool = []
    for individual in individuals:
        if individual.label not in ("survivor", "culled"):
            continue
        if individual.realized_rank_used is None:
            continue
        if measure == "binary_error" and individual.predicted_closeness is None:
            continue
        if measure == "bucket_distance" and bucket_ordinal(individual.predicted_bucket) is None:
            continue
        pool.append(individual)
    return pool


def _permuted_errors(
    pool: list[LineageIndividual], measure: str, rng: np.random.Generator
) -> list[tuple[LineageIndividual, float]]:
    """Errors after permuting predicted values within (trace, generation, bin) cells."""
    cells: dict[tuple[str, int, str], list[int]] = {}
    for index, individual in enumerate(pool):
        cells.setdefault(individual.cell(), []).append(index)

    predictions: list[tuple[float | None, str | None]] = [
        (individual.predicted_closeness, individual.predicted_bucket) for individual in pool
    ]
    permuted = list(predictions)
    for indices in cells.values():
        shuffled = [predictions[i] for i in rng.permutation(indices)]
        for slot, value in zip(indices, shuffled):
            permuted[slot] = value

    results: list[tuple[LineageIndividual, float]] = []
    for individual, (closeness, bucket) in zip(pool, permuted):
        if measure == "binary_error":
            label = 1.0 if individual.realized_rank_used <= POSITIVE_RANK_THRESHOLD else 0.0
            results.append((individual, abs(float(closeness) - label)))
        else:
            predicted = bucket_ordinal(bucket)
            actual = bucket_ordinal(realized_bucket(individual.realized_rank_used))
            results.append((individual, float(abs(predicted - actual))))
    return results


def _gap_from_values(values: list[tuple[LineageIndividual, float]]) -> float | None:
    by_bin: dict[str, dict[str, list[float]]] = {}
    for individual, value in values:
        by_bin.setdefault(individual.parent_rank_bin, {"survivor": [], "culled": []})[individual.label].append(value)
    weighted_sum = 0.0
    weight_total = 0
    for sides in by_bin.values():
        survivors, culled = sides["survivor"], sides["culled"]
        if survivors and culled:
            gap = sum(survivors) / len(survivors) - sum(culled) / len(culled)
            weight = len(survivors) + len(culled)
            weighted_sum += gap * weight
            weight_total += weight
    return (weighted_sum / weight_total) if weight_total else None


def permutation_gap_test(
    individuals: list[LineageIndividual], measure: str, permutations: int, rng: np.random.Generator
) -> dict[str, Any]:
    """Two-sided empirical p for the matched gap under within-cell permutation."""
    pool = _permutation_pool(individuals, measure)
    observed = _gap_from_values([(individual, individual.measure(measure)) for individual in pool])
    if observed is None:
        return {"observed_gap": None, "pvalue": None, "n_pool": len(pool), "permutations": permutations}
    at_least_as_extreme = 0
    completed = 0
    for _draw in range(permutations):
        permuted_gap = _gap_from_values(_permuted_errors(pool, measure, rng))
        if permuted_gap is None:
            continue
        completed += 1
        if abs(permuted_gap) >= abs(observed):
            at_least_as_extreme += 1
    pvalue = (1 + at_least_as_extreme) / (1 + completed) if completed else None
    return {"observed_gap": observed, "pvalue": pvalue, "n_pool": len(pool), "permutations": completed}


# --- transmission -----------------------------------------------------------------


def lineage_pairs(individuals: list[LineageIndividual]) -> list[tuple[LineageIndividual, LineageIndividual]]:
    """(parent individual, child individual) pairs along parent_id -> child_id links."""
    by_key: dict[tuple[str, str], LineageIndividual] = {
        (individual.trace_file, individual.child_id): individual for individual in individuals
    }
    pairs: list[tuple[LineageIndividual, LineageIndividual]] = []
    for child in individuals:
        if not child.parent_id:
            continue
        parent = by_key.get((child.trace_file, str(child.parent_id)))
        if parent is not None:
            pairs.append((parent, child))
    return pairs


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2 or len(set(x)) < 2 or len(set(y)) < 2:
        return None
    rho, _p = stats.spearmanr(x, y)
    return float(rho)


def transmission_test(
    individuals: list[LineageIndividual],
    measure: str,
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Parent-child Spearman of agreement vs the within-cell permutation baseline.

    Pairs require both endpoints to have a defined measure. The permutation
    reuses the survivor/culled pool cells; endpoints outside the pool (e.g. an
    excluded parent) keep their observed values under the null, which is
    conservative for the transmission correlation.
    """
    pairs = [
        (parent, child)
        for parent, child in lineage_pairs(individuals)
        if parent.measure(measure) is not None and child.measure(measure) is not None
    ]
    observed = _spearman(
        [parent.measure(measure) for parent, _child in pairs],
        [child.measure(measure) for _parent, child in pairs],
    )
    if observed is None:
        return {"n_pairs": len(pairs), "observed_rho": None, "pvalue": None, "permutations": permutations}

    pool = _permutation_pool(individuals, measure)
    pool_ids = {(individual.trace_file, individual.child_id) for individual in pool}
    at_least_as_extreme = 0
    completed = 0
    for _draw in range(permutations):
        permuted = {
            (individual.trace_file, individual.child_id): value
            for individual, value in _permuted_errors(pool, measure, rng)
        }

        def value_of(individual: LineageIndividual) -> float:
            key = (individual.trace_file, individual.child_id)
            if key in pool_ids:
                return permuted[key]
            return individual.measure(measure)

        rho = _spearman(
            [value_of(parent) for parent, _child in pairs],
            [value_of(child) for _parent, child in pairs],
        )
        if rho is None:
            continue
        completed += 1
        if abs(rho) >= abs(observed):
            at_least_as_extreme += 1
    pvalue = (1 + at_least_as_extreme) / (1 + completed) if completed else None
    return {"n_pairs": len(pairs), "observed_rho": observed, "pvalue": pvalue, "permutations": completed}


# --- generation trend ---------------------------------------------------------------


def generation_trend(individuals: list[LineageIndividual]) -> list[dict[str, Any]]:
    """Mean agreement per (generation, parent-rank bin) - matching is explicit."""
    groups: dict[tuple[int, str], list[LineageIndividual]] = {}
    for individual in individuals:
        groups.setdefault((individual.generation, individual.parent_rank_bin), []).append(individual)
    rows: list[dict[str, Any]] = []
    for (generation, bin_label), members in sorted(groups.items()):
        binary = [m.binary_error for m in members if m.binary_error is not None]
        buckets = [m.bucket_distance for m in members if m.bucket_distance is not None]
        rows.append(
            {
                "generation": generation,
                "parent_rank_bin": bin_label,
                "n": len(members),
                "n_binary": len(binary),
                "mean_binary_error": (sum(binary) / len(binary)) if binary else None,
                "n_bucket": len(buckets),
                "mean_bucket_distance": (sum(buckets) / len(buckets)) if buckets else None,
            }
        )
    return rows


# --- outputs -------------------------------------------------------------------------


def individual_row(individual: LineageIndividual) -> dict[str, Any]:
    return {
        "trace_file": individual.trace_file,
        "child_id": individual.child_id,
        "child_name": individual.child_name,
        "generation": individual.generation,
        "parent_id": individual.parent_id,
        "parent_rank": individual.parent_rank,
        "parent_rank_bin": individual.parent_rank_bin,
        "sampled_op": individual.sampled_op,
        "predicted_closeness": individual.predicted_closeness,
        "predicted_bucket": individual.predicted_bucket,
        "self_report_parse_failed": individual.self_report_parse_failed,
        "realized_rank_first": individual.realized_rank_first,
        "child_best_rank": individual.child_best_rank,
        "realized_rank_used": individual.realized_rank_used,
        "proposed_word_invalid": individual.proposed_word_invalid,
        "binary_error": individual.binary_error,
        "bucket_distance": individual.bucket_distance,
        "label": individual.label,
        "exclusion_reason": individual.exclusion_reason,
        "resurrected": individual.resurrected,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def analyze_traces(
    trace_paths: list[str], *, seed: int, permutations: int
) -> dict[str, Any]:
    all_individuals: list[LineageIndividual] = []
    trace_metas: list[dict[str, Any]] = []
    for path in trace_paths:
        events = load_trace(path)
        individuals, meta = extract_lineage(events, Path(path).name)
        all_individuals.extend(individuals)
        trace_metas.append(meta)

    rng = np.random.default_rng(seed)
    gap_rows: list[dict[str, Any]] = []
    transmission_rows: list[dict[str, Any]] = []
    summary_tests: dict[str, Any] = {}

    for measure in MEASURES:
        # per-run rows
        for meta in trace_metas:
            run_individuals = [i for i in all_individuals if i.trace_file == meta["trace_file"]]
            _overall, per_bin = matched_gap(run_individuals, measure)
            for row in per_bin:
                gap_rows.append({"scope": "run", "trace_file": meta["trace_file"], **row})
        # pooled rows + permutation tests
        pooled_overall, pooled_bins = matched_gap(all_individuals, measure)
        for row in pooled_bins:
            gap_rows.append({"scope": "pooled", "trace_file": None, **row})
        gap_test = permutation_gap_test(all_individuals, measure, permutations, rng)
        trans_test = transmission_test(all_individuals, measure, permutations, rng)
        transmission_rows.append({"scope": "pooled", "trace_file": None, "measure": measure, **trans_test})
        summary_tests[measure] = {
            "pooled_matched_gap": pooled_overall,
            "gap_permutation": gap_test,
            "transmission": trans_test,
        }

    return {
        "individuals": all_individuals,
        "trace_metas": trace_metas,
        "gap_rows": gap_rows,
        "transmission_rows": transmission_rows,
        "trend_rows": generation_trend(all_individuals),
        "summary_tests": summary_tests,
        "seed": seed,
        "permutations": permutations,
    }


def write_outputs(result: dict[str, Any], output_dir: str | Path) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(out_dir / "rq3_individuals.csv", [individual_row(i) for i in result["individuals"]])
    _write_csv(out_dir / "rq3_gap_by_bin.csv", result["gap_rows"])
    _write_csv(out_dir / "rq3_generation_trend.csv", result["trend_rows"])
    _write_csv(out_dir / "rq3_transmission.csv", result["transmission_rows"])

    counts_total: dict[str, int] = {}
    for meta in result["trace_metas"]:
        for key, value in meta["counts"].items():
            counts_total[key] = counts_total.get(key, 0) + value
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": result["seed"],
        "permutations": result["permutations"],
        "traces": result["trace_metas"],
        "counts_total": counts_total,
        "tests": result["summary_tests"],
        "note": (
            "Gap = mean agreement error (survivor) - mean agreement error (culled), "
            "weighted over matched parent-rank bins; generation and parent distance "
            "are confounded, so all comparisons are within (generation x parent-rank-bin) "
            "cells for the permutation null and within parent-rank bins for the gap."
        ),
    }
    (out_dir / "rq3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="RQ3 lineage analysis over batch traces (offline).")
    parser.add_argument("traces", nargs="+", help="Raw trace JSON path(s) or glob(s).")
    parser.add_argument("--output", required=True, help="Output directory for CSVs + summary JSON.")
    parser.add_argument("--seed", type=int, default=0, help="Permutation RNG seed (default 0).")
    parser.add_argument("--permutations", type=int, default=1000, help="Permutation draws B (default 1000).")
    args = parser.parse_args()

    paths: list[str] = []
    for pattern in args.traces:
        matches = glob.glob(pattern)
        paths.extend(sorted(matches) if matches else ([pattern] if Path(pattern).exists() else []))
    if not paths:
        raise SystemExit("No trace files matched the given path(s).")

    result = analyze_traces(paths, seed=args.seed, permutations=args.permutations)
    out_dir = write_outputs(result, args.output)

    counts: dict[str, int] = {}
    for meta in result["trace_metas"]:
        for key, value in meta["counts"].items():
            counts[key] = counts.get(key, 0) + value
    print(f"Traces: {len(result['trace_metas'])}  individuals: {counts.get('individuals', 0)}")
    print(
        f"  survivor={counts.get('survivor', 0)} culled={counts.get('culled', 0)} "
        f"resurrections={counts.get('resurrections', 0)} "
        f"excluded: dedup={counts.get(EXCLUSION_DEDUP, 0)} final_gen={counts.get(EXCLUSION_FINAL_GEN, 0)} "
        f"collision={counts.get(EXCLUSION_COLLISION, 0)} missing={counts.get(EXCLUSION_MISSING, 0)}"
    )
    for measure, tests in result["summary_tests"].items():
        gap = tests["gap_permutation"]
        trans = tests["transmission"]
        print(
            f"  {measure}: matched gap = {gap['observed_gap']} (perm p = {gap['pvalue']}), "
            f"transmission rho = {trans['observed_rho']} (perm p = {trans['pvalue']}, pairs = {trans['n_pairs']})"
        )
    print(f"Wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
