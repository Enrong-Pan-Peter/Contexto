"""Fig 2 - reliability, signed gap, and bin mass, one column per configuration.

Argues: the overconfidence is systematic and large, not a tail artifact.

Three rows share each column's x-axis, which is the stated top-100 probability:

  row 1  reliability curve against the diagonal, with the wedge between them
         shaded. That wedge is what Equation 1 integrates. The axes are EQUAL
         and are not rescaled to enlarge the wedge: the gap has to be read at
         true scale or the figure argues by zoom. Uncertainty is a ribbon, not
         an error bar: a 95 percent cluster bootstrap over runs, so a bin resting
         on nine observations visibly carries more uncertainty than one resting
         on four hundred.
  row 2  the same gap as a signed quantity, stated minus realized per bin,
         against a horizontal zero. Every bar is positive in every panel, which
         is the finding: the operator is overconfident everywhere, not in a tail.
  row 3  each bin's share of that configuration's proposals, as its own strip
         rather than behind the curve, because a count drawn under a line reads
         as part of the line.

Every value is recomputed here from the per-guess records; no pooled summary is
read. The recomputation reproduces each arm's frozen pooled ECE exactly (A1
0.3769, A2 0.4210, A4 0.6296, A5 0.6056), which is the check that the binning
here is the binning the artifact used: 10 equal-width bins on [0, 1], each bin
contributing |mean outcome - mean stated| weighted by its share of proposals.

A proposal enters only if it has both a stated probability and a realized rank.
Curve markers sit at the MEAN stated probability inside the bin, not at the bin
center, because that is the quantity the calibration error differences. Bins
holding fewer than 30 proposals are drawn hollow in row 1.

Run: .venv/bin/python paper/figures/fig2_reliability.py
Out: paper/fig/fig2_reliability.pdf
"""
from __future__ import annotations
import csv, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import matplotlib.pyplot as plt
from style import FULL_WIDTH, INK, INK_MUTED, PAPER_DIR, REPO_ROOT, save, assert_no_text_overlap

import os
# Camera-ready: the package written by scripts/camera_ready.py (Contexto repo); CONTEXTO_ART_DIR
# overrides the default location next to the paper.
ART = Path(os.environ.get("CONTEXTO_ART_DIR", PAPER_DIR / "camera_ready_outputs"))
NBINS, THIN = 10, 30
N_BOOT, SEED = 2_000, 20260819   # cluster bootstrap over runs, for the ribbons
# Family colors as in Fig 3. Direct-Qwen is the same model without the loop, so
# it takes a neutral gray rather than a fourth family hue.
PANELS = [("A1", "EA-Qwen", "#E69F00", "^"), ("A4", "EA-Gemma", "#0072B2", "o"),
          ("A5", "EA-Ministral", "#009E73", "s"), ("A2", "Direct-Qwen", "#595959", "D")]
RC = {"axes.linewidth": 0.9, "axes.spines.top": False, "axes.spines.right": False,
      "axes.grid": False, "xtick.major.width": 0.9, "ytick.major.width": 0.9,
      "xtick.major.size": 2.2, "ytick.major.size": 2.2}


def load(arm):
    rows = list(csv.DictReader((ART / f"{arm}_augmented.csv").open(newline="")))
    # realized_rank_used follows the rule the package was built with (camera-ready: the best of
    # the operator's own proposed words); realized_rank is always the first graded word.
    ok = [r for r in rows if r["predicted_closeness"] not in ("", "None")
          and r["realized_rank_used"] not in ("", "None")]
    p = np.array([float(r["predicted_closeness"]) for r in ok])
    y = np.array([1.0 if float(r["realized_rank_used"]) <= 100 else 0.0 for r in ok])
    run = np.array([r["trace_file"] for r in ok])
    return p, y, run


def ribbon(p, y, run):
    """95% cluster bootstrap of the per-bin realized rate, resampling RUNS.

    Runs are the unit of replication here as everywhere else in the paper; a
    per-proposal bootstrap would understate the width. A bin missing from a
    resample contributes nothing to that replicate rather than being imputed."""
    edges = np.linspace(0, 1, NBINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, NBINS - 1)
    keys = np.unique(run)
    pos = {k: np.flatnonzero(run == k) for k in keys}
    rng = np.random.default_rng(SEED)
    reps = np.full((N_BOOT, NBINS), np.nan)
    for i in range(N_BOOT):
        sel = np.concatenate([pos[k] for k in rng.choice(keys, keys.size)])
        bi, yi = idx[sel], y[sel]
        for b in range(NBINS):
            m = bi == b
            if m.any():
                reps[i, b] = yi[m].mean()
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():          # empty bins yield all-NaN columns
            warnings.simplefilter("ignore", RuntimeWarning)
            lo = np.nanpercentile(reps, 2.5, axis=0)
            hi = np.nanpercentile(reps, 97.5, axis=0)
    return lo, hi


def reliability(p, y):
    edges = np.linspace(0, 1, NBINS + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, NBINS - 1)
    xs, ys, ns, ece = [], [], [], 0.0
    for b in range(NBINS):
        m = idx == b
        n = int(m.sum())
        ns.append(n)
        if n:
            xs.append(p[m].mean()); ys.append(y[m].mean())
            ece += (n / p.size) * abs(y[m].mean() - p[m].mean())
        else:
            xs.append(np.nan); ys.append(np.nan)
    return np.array(xs), np.array(ys), np.array(ns), ece


def main():
    with plt.rc_context(RC):
        fig = plt.figure(figsize=(FULL_WIDTH, 2.72))
        gs = fig.add_gridspec(3, 4, height_ratios=[2.72, 1.02, 0.60],
                              left=0.093, right=0.978, bottom=0.175, top=0.925,
                              wspace=0.17, hspace=0.30)
        report = []
        for col, (arm, name, colour, marker) in enumerate(PANELS):
            p, y, run = load(arm)
            xs, ys, ns, ece = reliability(p, y)
            rlo, rhi = ribbon(p, y, run)
            fat, ok = ns >= THIN, ns > 0
            share = 100 * ns / ns.sum()
            centers = np.linspace(0.05, 0.95, NBINS)

            ax = fig.add_subplot(gs[0, col])
            ax.set_aspect("equal", adjustable="box")
            # Neutral gray for the wedge and family color for the ribbon, so the
            # quantity being reported and the uncertainty on it never read as one
            # shape. The wedge is what Equation 1 integrates.
            ax.fill_between(xs[ok], ys[ok], xs[ok], color=INK_MUTED, alpha=0.15,
                            linewidth=0, zorder=2)
            ax.fill_between(xs[ok], rlo[ok], rhi[ok], color=colour, alpha=0.34,
                            linewidth=0, zorder=3)
            ax.plot([0, 1], [0, 1], linestyle=(0, (3, 2.5)), linewidth=0.8,
                    color=INK_MUTED, zorder=3)
            ax.plot(xs, ys, linewidth=1.0, color=colour, zorder=4)
            ax.scatter(xs[fat], ys[fat], s=14, marker=marker, color=colour,
                       edgecolor="white", linewidth=0.4, zorder=5)
            ax.scatter(xs[ok & ~fat], ys[ok & ~fat], s=14, marker=marker,
                       facecolor="white", edgecolor=colour, linewidth=0.9, zorder=5)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.set_xticks([0, 0.5, 1.0]); ax.set_yticks([0, 0.5, 1.0])
            ax.set_xticklabels([])
            ax.set_title(f"({chr(65 + col)}) {name}", pad=3,
                         color=colour, fontweight="bold")
            ax.annotate(f"ECE {ece:.3f}\n$n$={p.size:,}", xy=(0.185, 0.985),
                        xycoords="axes fraction", ha="left", va="top",
                        fontsize=7, color=INK)
            if col:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel("realized\ntop-100 rate", labelpad=2)

            axg = fig.add_subplot(gs[1, col])
            axg.axhline(0, color=INK_MUTED, linewidth=0.8, zorder=1)
            axg.fill_between(xs[ok], (xs - rhi)[ok], (xs - rlo)[ok], color=colour,
                             alpha=0.34, linewidth=0, zorder=2)
            axg.plot(xs[ok], (xs - ys)[ok], color=colour, linewidth=1.1, zorder=3)
            axg.scatter(xs[ok], (xs - ys)[ok], s=9, marker=marker, color=colour,
                        edgecolor="white", linewidth=0.3, zorder=4)
            axg.set_xlim(0, 1); axg.set_ylim(-0.12, 0.92)
            axg.set_xticks([0, 0.5, 1.0]); axg.set_xticklabels([])
            axg.set_yticks([0, 0.4, 0.8])
            if col:
                axg.set_yticklabels([])
            else:
                axg.set_ylabel("stated $-$\nrealized", labelpad=2)

            axn = fig.add_subplot(gs[2, col])
            axn.bar(centers[ok], share[ok], width=0.082, color=colour, alpha=0.42,
                    linewidth=0, zorder=2)
            axn.set_xlim(0, 1); axn.set_ylim(0, 60)
            axn.set_xticks([0, 0.5, 1.0]); axn.set_yticks([0, 50])
            axn.set_xlabel("stated $P$(top-100)", labelpad=1.5)
            if col:
                axn.set_yticklabels([])
            else:
                axn.set_ylabel("proposals\nper bin (%)", labelpad=2)
            report.append((name, arm, p.size, ece, p.mean(), y.mean(),
                           int((ns[ok] < THIN).sum()), int(ok.sum()),
                           float(np.nanmin((xs - ys)[ok]))))
    assert_no_text_overlap(fig)
    out = save(fig, "fig2_reliability"); plt.close(fig)
    print(f"wrote {out.relative_to(REPO_ROOT)}")
    for name, arm, n, ece, ms, my, thin, filled, mingap in report:
        print(f"  {name:14} {arm}  n={n:5,}  ECE={ece:.4f}  stated mean={ms:.4f}  "
              f"realized={my:.4f}  bins<{THIN}: {thin}/{filled}  min signed gap={mingap:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
