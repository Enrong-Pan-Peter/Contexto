"""Camera-ready pipeline: regenerate every analysis output of the paper under the corrected rule.

    python scripts/camera_ready.py --out camera_ready_outputs_2026-10-26
    python scripts/camera_ready.py --out camera_ready_outputs_2026-10-26 --resample traces/rq2_resample
    python scripts/camera_ready.py --out ... --quick            # 200 bootstrap draws, for a smoke test

What it runs (all offline; the second genuine sample must already exist if --resample is given):

1. RQ1, per configuration (A1 EA-Qwen, A2 Direct-Qwen, A4 EA-Gemma, A5 EA-Ministral): the
   per-run metrics and the per-individual rows under the camera-ready rule
   ``best_proposed`` (the event the report prompt names: best of the operator's own proposed
   words; a re-proposed word keeps its earlier rank; an all-invalid proposal counts as a miss)
   -> ``report_<arm>/per_run_metrics.csv`` and ``<arm>_augmented.csv`` (what the figure scripts
   read), plus ``rq1_metrics_<arm>.json``. The same under the submitted rule ``first_proposed``
   into ``submitted_rule/`` so the erratum can show both.
2. Table 2: medians over scored runs per configuration and the paired Wilcoxon tests of
   EA-Qwen against EA-Gemma and against Direct-Qwen (pairs = same game and run index), Holm
   over the eight metrics -> ``table2.json`` / ``table2.md``.
3. RQ3 (selection): ``rq3_A1`` (10 games), ``rq3_A1_3games`` (the control games 1303 1319
   1372), ``rq3_random`` (random-selection control), each with the permutation p and the
   run-resampled 95 % intervals -> ``rq3_*/rq3_summary.json`` (what fig5 reads).
4. RQ2 (rationale intervention): the stored 754-event replay, plus the second genuine sample
   when ``--resample`` points at its outputs -> ``rq2_analysis/`` and ``rq2_hl_bootstrap.json``
   (what fig4 reads: Hodges-Lehmann rank effect and interval per condition, first-word
   agreement with interval, and the ``genuine_resample`` row when available).
5. The Direct-Qwen difficulty reweighting with explicit strata -> ``direct_qwen_reweight.json``.
6. ``CHANGES.md``: submitted versus camera-ready numbers side by side.

Paths are relative to the repository root (the study traces live under traces/batch/traces/).
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from contexto_solver.rq1 import metrics as rq1_metrics  # noqa: E402
from contexto_solver.rq1.reader import read_run  # noqa: E402

TRACES = ROOT / "traces" / "batch" / "traces"
LOG = Path("camera_ready.log")  # set in main() to <out>/camera_ready.log
ARMS = {
    "A1": ("EA-Qwen", "rq1_A1/ea_llm_self_adaptive_api_*.json"),
    "A2": ("Direct-Qwen", "rq1_A2/llm_only_api_*.json"),
    "A4": ("EA-Gemma", "rq1_A4/ea_llm_self_adaptive_api_*.json"),
    "A5": ("EA-Ministral", "rq1_A5/ea_llm_self_adaptive_api_*.json"),
}
CONTROL_GAMES = ("1303", "1319", "1372")
TABLE2_METRICS = (
    ("brier", "Brier score"),
    ("bucket_accuracy", "Bucket accuracy"),
    ("mean_signed_bucket_error", "Mean signed bucket error"),
    ("off_by_one_rate", "Off-by-one bucket rate"),
    ("spearman_rho", "Rank correlation rho"),
    ("auroc", "Area under the ROC curve"),
    ("ece", "ECE"),
    ("positive_rate", "Observed top-100 rate"),
)
RQ2_CONDITIONS = (("wrong", "Unrelated"), ("filler", "Filler"), ("absent", "Removed"), ("genuine_resample", "Genuine, second sample"))


def run(cmd: list[str]) -> None:
    """Run one analysis script; its (often long) stdout is kept in a log next to the outputs."""
    print("$", cmd[0], "...", cmd[-1], flush=True)
    completed = subprocess.run([sys.executable, "-I", *cmd], cwd=ROOT, capture_output=True, text=True)
    LOG.open("a", encoding="utf-8").write(f"\n$ {' '.join(str(c) for c in cmd)}\n{completed.stdout}\n{completed.stderr}\n")
    if completed.returncode:
        print(completed.stdout[-2000:]); print(completed.stderr[-4000:])
        raise SystemExit(f"{cmd[0]} failed with status {completed.returncode}")


def trace_paths(pattern: str, games: tuple[str, ...] | None = None) -> list[str]:
    paths = sorted(glob.glob(str(TRACES / pattern)))
    if games:
        paths = [p for p in paths if any(f"_{game}_run" in Path(p).name for game in games)]
    return paths


# ---------------------------------------------------------------- 1. RQ1 per configuration


def rq1(out: Path, which: str, label: str) -> None:
    for arm, (_name, pattern) in ARMS.items():
        paths = trace_paths(pattern)
        if not paths:
            print(f"  no traces for {arm}; skipped")
            continue
        run(["scripts/rq1_report.py", *paths, "--output-dir", str(out / f"report_{arm}"), "--which", which])
        run(["scripts/rq1_metrics.py", *paths, "--which", which, "--output", str(out / f"rq1_metrics_{arm}.json"),
             "--augmented-csv", str(out / f"{arm}_augmented.csv")])
    print(f"  RQ1 ({label}) written under {out}")


# ---------------------------------------------------------------- 2. Table 2


def _holm(pvalues: list[float | None]) -> list[float | None]:
    indexed = [(p, i) for i, p in enumerate(pvalues) if p is not None]
    adjusted: list[float | None] = [None] * len(pvalues)
    m = len(indexed)
    running = 0.0
    for rank, (p, i) in enumerate(sorted(indexed)):
        running = max(running, (m - rank) * p)
        adjusted[i] = min(1.0, running)
    return adjusted


def table2(out: Path, which: str) -> dict[str, Any]:
    """Medians per configuration and paired Wilcoxon of EA-Qwen vs EA-Gemma / Direct-Qwen (Holm over eight)."""
    from scipy import stats

    per_run: dict[str, dict[tuple[Any, Any], dict[str, Any]]] = {}
    for arm, (_name, pattern) in ARMS.items():
        bundle_path = out / f"report_{arm}" / "report.json"
        if not bundle_path.is_file():
            continue
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        scalars = bundle["per_run_scalars"]
        keyed: dict[tuple[Any, Any], dict[str, Any]] = {}
        for path in trace_paths(pattern):
            config, _individuals = read_run(path)
            name = Path(path).name
            if name in scalars:
                keyed[(config.game_number, config.raw.get("run_index"))] = scalars[name]
        per_run[arm] = keyed

    result: dict[str, Any] = {"which_realized": which, "metrics": {}}
    for arm_b in ("A4", "A2"):
        comparisons = []
        for metric, _label in TABLE2_METRICS:
            a_runs = per_run.get("A1", {})
            b_runs = per_run.get(arm_b, {})
            keys = [k for k in a_runs if k in b_runs and a_runs[k].get(metric) is not None and b_runs[k].get(metric) is not None]
            if len(keys) >= 2:
                x = [float(a_runs[k][metric]) for k in keys]
                y = [float(b_runs[k][metric]) for k in keys]
                try:
                    p = float(stats.wilcoxon(x, y).pvalue) if any(a != b for a, b in zip(x, y)) else 1.0
                except ValueError:
                    p = None
            else:
                p = None
            comparisons.append({"metric": metric, "n_pairs": len(keys), "pvalue": p})
        for comparison, p_holm in zip(comparisons, _holm([c["pvalue"] for c in comparisons])):
            comparison["pvalue_holm"] = p_holm
        result[f"A1_vs_{arm_b}"] = comparisons

    for metric, label in TABLE2_METRICS:
        entry: dict[str, Any] = {"label": label}
        for arm in ARMS:
            values = [float(s[metric]) for s in per_run.get(arm, {}).values() if s.get(metric) is not None]
            entry[arm] = {"median": float(np.median(values)) if values else None, "n_runs": len(values)}
        result["metrics"][metric] = entry

    (out / "table2.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["| metric | EA-Qwen | EA-Gemma | EA-Ministral | Direct-Qwen | n vs Gemma | p Holm | n vs Direct | p Holm |", "|---|---|---|---|---|---|---|---|---|"]
    for metric, label in TABLE2_METRICS:
        row = result["metrics"][metric]
        gem = next(c for c in result["A1_vs_A4"] if c["metric"] == metric)
        direct = next(c for c in result["A1_vs_A2"] if c["metric"] == metric)
        fmt = lambda v: "–" if v is None else f"{v:.3f}"  # noqa: E731
        lines.append(f"| {label} | {fmt(row['A1']['median'])} | {fmt(row['A4']['median'])} | {fmt(row['A5']['median'])} | {fmt(row['A2']['median'])} | "
                     f"{gem['n_pairs']} | {fmt(gem['pvalue_holm'])} | {direct['n_pairs']} | {fmt(direct['pvalue_holm'])} |")
    (out / "table2.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


# ---------------------------------------------------------------- 3. RQ3


def rq3(out: Path, which: str, bootstrap: int, permutations: int) -> None:
    jobs = {
        "rq3_A1": trace_paths(ARMS["A1"][1]),
        "rq3_A1_3games": trace_paths(ARMS["A1"][1], CONTROL_GAMES),
        "rq3_random": trace_paths("rq3_random/ea_llm_self_adaptive_api_*.json"),
    }
    for name, paths in jobs.items():
        if not paths:
            print(f"  no traces for {name}; skipped")
            continue
        run(["scripts/rq3_lineage.py", *paths, "--output", str(out / name), "--seed", "0",
             "--permutations", str(permutations), "--bootstrap", str(bootstrap), "--which", which])


# ---------------------------------------------------------------- 4. RQ2


def rq2(out: Path, resample_dir: Path | None, bootstrap: int) -> None:
    inputs = sorted(glob.glob(str(TRACES / "rq2_replay" / "*" / "rq2_replay_records.json")))
    if resample_dir is not None:
        extra = sorted(glob.glob(str(resample_dir / "*" / "rq2_replay_records.json")))
        if not extra:
            print(f"  WARNING: no rq2_replay_records.json under {resample_dir}; the second genuine sample is not included")
        inputs += extra
    if not inputs:
        print("  no stored intervention records; RQ2 skipped")
        return
    run(["scripts/rq2_analysis.py", *inputs, "--output", str(out / "rq2_analysis"), "--bootstrap", str(bootstrap), "--seed", "0"])
    summary = json.loads((out / "rq2_analysis" / "rq2_summary.json").read_text(encoding="utf-8"))
    agreement = {row["contrast"]: row for row in csv.DictReader((out / "rq2_analysis" / "rq2_word_agreement.csv").open(encoding="utf-8"))}
    diffs: dict[str, list[float]] = {}
    for row in csv.DictReader((out / "rq2_analysis" / "rq2_paired_differences.csv").open(encoding="utf-8")):
        diffs.setdefault(row["contrast"], []).append(float(row["diff_genuine_minus_variant"]))
    hl: dict[str, Any] = {"identity": {}}
    for key, label in RQ2_CONDITIONS:
        tests = summary["tests"].get(key)
        if not tests:
            continue
        boot = tests.get("cluster_bootstrap", {})
        hl_entry = boot.get("hl_diff", {})
        identity = boot.get("word_identity_rate", {})
        hl[key] = {
            "label": label,
            "n_events": tests["pooled_descriptives"]["n"],
            "hl": hl_entry.get("observed"),
            "ci_lo": hl_entry.get("ci95_low"),
            "ci_hi": hl_entry.get("ci95_high"),
            "identity_rate": identity.get("observed"),
            "identity_ci_lo": identity.get("ci95_low"),
            "identity_ci_hi": identity.get("ci95_high"),
            "jaccard": float(agreement[key]["mean_jaccard_overlap"]) if key in agreement and agreement[key]["mean_jaccard_overlap"] else None,
            "n_identity_events": int(agreement[key]["n_events"]) if key in agreement else None,
            "diffs": diffs.get(key, []),   # panel A of fig4: every paired difference
        }
        hl["identity"][key] = {"rate": identity.get("observed"), "ci_lo": identity.get("ci95_low"), "ci_hi": identity.get("ci95_high")}
    (out / "rq2_hl_bootstrap.json").write_text(json.dumps(hl, indent=2), encoding="utf-8")
    print(f"  wrote {out / 'rq2_hl_bootstrap.json'} with conditions {[k for k in hl if k != 'identity']}")


# ---------------------------------------------------------------- 5. Direct-Qwen reweighting


def _state_stratum(rank: int | None) -> str:
    """Difficulty stratum of a search state = the best rank the run had reached before the call."""
    return rq1_metrics._parent_rank_bin(rank)


def _best_rank_before(events: list[dict[str, Any]]) -> list[int | None]:
    """For every self-reported call in trace order, the run's best rank before that call."""
    best: int | None = None
    out: list[int | None] = []
    for event in events:
        name = event.get("event")
        details = event.get("details", {}) or {}
        if name in ("OPERATOR_SAMPLED", "CROSSOVER") and (isinstance(details.get("self_report"), dict) or isinstance((details.get("child") or {}).get("self_report"), dict)):
            out.append(best)
        if name == "GUESS":
            if isinstance(details.get("self_report"), dict):  # llm_only: the report precedes this guess's outcome
                out.append(best)
            rank = details.get("rank")
            if isinstance(rank, int) and rank > 0 and (best is None or rank < best):
                best = rank
    return out


def direct_qwen_reweight(out: Path, which: str) -> dict[str, Any]:
    """ECE of Direct-Qwen reweighted to EA-Qwen's distribution over search-state strata.

    Stratum = the best rank the run had reached before the call (bins 1-10, 11-100,
    101-500, 501-1000, 1000+, "unknown" before any graded guess). Reported descriptively.
    """
    from contexto_solver.rq1.reader import load_trace

    per_arm: dict[str, dict[str, list[tuple[float, int]]]] = {}
    for arm in ("A1", "A2"):
        strata: dict[str, list[tuple[float, int]]] = {}
        for path in trace_paths(ARMS[arm][1]):
            events = load_trace(path)
            _config, individuals = read_run(path)
            before = _best_rank_before(events)
            if len(before) != len(individuals):
                before = (before + [None] * len(individuals))[: len(individuals)]
            for individual, state in zip(individuals, before):
                rank = rq1_metrics.realized_rank(individual, which)
                if individual.predicted_closeness is None or rank is None:
                    continue
                strata.setdefault(_state_stratum(state), []).append((float(individual.predicted_closeness), 1 if rank <= 100 else 0))
        per_arm[arm] = strata

    def ece(pairs: list[tuple[float, int]]) -> float | None:
        if not pairs:
            return None
        scores = np.array([p for p, _ in pairs]); labels = np.array([y for _, y in pairs])
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(scores, edges[1:-1]), 0, 9)
        total = 0.0
        for b in range(10):
            mask = idx == b
            if mask.any():
                total += mask.mean() * abs(labels[mask].mean() - scores[mask].mean())
        return float(total)

    ea, direct = per_arm["A1"], per_arm["A2"]
    n_ea = sum(len(v) for v in ea.values())
    rows = []
    standardized = 0.0
    covered = 0.0
    for stratum in sorted(set(ea) | set(direct)):
        share = len(ea.get(stratum, [])) / n_ea if n_ea else 0.0
        ece_direct = ece(direct.get(stratum, []))
        rows.append({"stratum": stratum, "ea_share": share, "n_ea": len(ea.get(stratum, [])), "n_direct": len(direct.get(stratum, [])),
                     "ece_ea": ece(ea.get(stratum, [])), "ece_direct": ece_direct})
        if ece_direct is not None:
            standardized += share * ece_direct
            covered += share
    ece_ea_all = ece([p for v in ea.values() for p in v])
    ece_direct_all = ece([p for v in direct.values() for p in v])
    standardized = standardized / covered if covered else None
    gap = (ece_direct_all - ece_ea_all) if (ece_direct_all is not None and ece_ea_all is not None) else None
    composition = ((ece_direct_all - standardized) / gap) if (standardized is not None and gap) else None
    result = {
        "which_realized": which,
        "stratum_definition": "best rank the run had reached before the call: 1-10, 11-100, 101-500, 501-1000, 1000+; 'unknown' before any graded guess",
        "ece_ea_qwen": ece_ea_all, "ece_direct_qwen": ece_direct_all,
        "ece_direct_qwen_standardized_to_ea_states": standardized,
        "ea_share_covered_by_direct_strata": covered,
        "gap": gap, "composition_share": composition,
        "residual_share": (1 - composition) if composition is not None else None,
        "strata": rows,
    }
    (out / "direct_qwen_reweight.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


# ---------------------------------------------------------------- 6. CHANGES.md


def _pooled(out: Path, arm: str) -> dict[str, Any] | None:
    path = out / f"rq1_metrics_{arm}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))["pooled"]["overall"]


def changes(out: Path, submitted: Path, reweight: dict[str, Any] | None) -> None:
    lines = [f"# Submitted versus camera-ready numbers ({date.today().isoformat()})", "",
             "Rule change: report accuracy is scored against the best of the operator's own proposed words (the event the report prompt names); "
             "the submitted paper scored the first graded word. Records are matched to their own words (the submitted join pooled same-name children). "
             "An all-invalid proposal counts as outside the top 100.", "",
             "| configuration | rule | n | mean stated | observed top-100 | stated/observed | ECE | Brier | AUROC | bucket acc. | mean signed bucket error |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for arm, (name, _pattern) in ARMS.items():
        for label, folder in (("submitted (first graded word)", submitted), ("camera-ready (best proposed word)", out)):
            pooled = _pooled(folder, arm)
            if pooled is None:
                continue
            n = pooled["brier"]["n"]; obs = pooled["brier"]["positive_rate"]
            stated = pooled.get("mean_predicted_closeness")
            if stated is None:
                bins = pooled["reliability"]["bins"]
                stated = sum(b["count"] * b["mean_predicted"] for b in bins if b["count"]) / max(1, n)
            fmt = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
            lines.append(f"| {name} | {label} | {n} | {fmt(stated)} | {fmt(obs)} | {fmt(stated / obs if obs else None, 2)} | {fmt(pooled['reliability']['ece'])} | "
                         f"{fmt(pooled['brier']['brier'])} | {fmt(pooled['auroc']['auroc'])} | {fmt(pooled['buckets']['accuracy'])} | {fmt(pooled['buckets']['mean_signed_bucket_error'], 2)} |")
    if reweight:
        lines += ["", f"Direct-Qwen reweighting (strata: {reweight['stratum_definition']}): ECE EA-Qwen {reweight['ece_ea_qwen']:.3f}, Direct-Qwen {reweight['ece_direct_qwen']:.3f}, "
                  f"Direct-Qwen standardised to EA-Qwen's states {reweight['ece_direct_qwen_standardized_to_ea_states']:.3f}; "
                  f"composition share {reweight['composition_share']:.3f}, residual {reweight['residual_share']:.3f}."]
    for name in ("rq3_A1", "rq3_A1_3games", "rq3_random"):
        path = out / name / "rq3_summary.json"
        if not path.is_file():
            continue
        s = json.loads(path.read_text(encoding="utf-8"))
        lines += ["", f"Selection ({name}, {s.get('n_runs')} runs, rule {s.get('which_realized')}):"]
        for measure, tests in s["tests"].items():
            g = tests["gap_permutation"]; t = tests["transmission"]
            gci = g.get("ci95") or {}; tci = t.get("ci95") or {}
            lines.append(f"- {measure}: gap {g['observed_gap']:+.3f} [{gci.get('low', float('nan')):+.3f}, {gci.get('high', float('nan')):+.3f}] (perm p {g['pvalue']:.3f}); "
                         f"transmission rho {t['observed_rho']:+.3f} [{tci.get('low', float('nan')):+.3f}, {tci.get('high', float('nan')):+.3f}] (perm p {t['pvalue']:.3f}, pairs {t['n_pairs']})")
    hl_path = out / "rq2_hl_bootstrap.json"
    if hl_path.is_file():
        hl = json.loads(hl_path.read_text(encoding="utf-8"))
        lines += ["", "Rationale intervention (genuine minus condition, Hodges-Lehmann rank effect; positive = genuine worse):"]
        for key, item in hl.items():
            if key == "identity":
                continue
            lines.append(f"- {item['label']}: n {item['n_events']}, HL {item['hl']:+.0f} [{item['ci_lo']:+.0f}, {item['ci_hi']:+.0f}], "
                         f"first-word agreement {item['identity_rate']:.3f} [{item['identity_ci_lo']:.3f}, {item['identity_ci_hi']:.3f}], Jaccard {item['jaccard']:.3f}")
    (out / "CHANGES.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


# ---------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=f"camera_ready_outputs_{date.today().isoformat()}")
    parser.add_argument("--resample", default=None, help="directory holding <game>/rq2_replay_records.json of the second genuine sample")
    parser.add_argument("--which", default="best_proposed", choices=list(rq1_metrics.REALIZED_RANK_RULES))
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--permutations", type=int, default=1000)
    parser.add_argument("--quick", action="store_true", help="200 bootstrap draws and 100 permutations (smoke test)")
    parser.add_argument("--skip", default="", help="comma-separated steps to skip: rq1,table2,rq3,rq2,reweight")
    args = parser.parse_args(argv)
    if args.quick:
        args.bootstrap, args.permutations = 200, 100
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    out = (ROOT / args.out) if not Path(args.out).is_absolute() else Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    global LOG
    LOG = out / "camera_ready.log"
    LOG.write_text("", encoding="utf-8")
    submitted = out / "submitted_rule"
    submitted.mkdir(exist_ok=True)

    if "rq1" not in skip:
        rq1(out, args.which, "camera-ready rule")
        rq1(submitted, "first_proposed", "submitted rule")
    if "table2" not in skip:
        table2(out, args.which)
        table2(submitted, "first_proposed")
    if "rq3" not in skip:
        rq3(out, "best_proposed" if args.which.startswith("best_proposed") else "child_best", args.bootstrap, args.permutations)
    if "rq2" not in skip:
        rq2(out, Path(args.resample) if args.resample else None, args.bootstrap)
    reweight = direct_qwen_reweight(out, args.which) if "reweight" not in skip else None
    changes(out, submitted, reweight)
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
