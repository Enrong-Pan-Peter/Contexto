# Academic Figure Skill Asset Confirmation
# Panel A, paired rank differences: assets/GroupedViolin and assets/Violin were
# inspected. Native reuse was rejected because the exact zero mass and long two
# sided tails make a density silhouette misleading. The restrained distribution
# styling is inherited, while every observation is plotted directly.
# Panel B, interval summary: no bundled asset matched a horizontal forest plot.
# The design inherits the typography, palette, and interval conventions.
# Panel C, categorical agreement: no bundled asset matched a three condition
# proportion plot. The design inherits the same palette and marker redundancy.

"""Generate the controlled rationale intervention figure with TikZ.

This implementation needs only the Python standard library and the manuscript's
existing LaTeX toolchain. It writes a vector PDF and a 300 dpi PNG preview.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import subprocess
import tempfile
from pathlib import Path


import os

PAPER_DIR = Path(__file__).resolve().parent.parent
# Camera-ready: the analysis package written by scripts/camera_ready.py (Contexto repo);
# point CONTEXTO_ART_DIR at it, or drop it next to this folder under the default name.
ART = Path(os.environ.get("CONTEXTO_ART_DIR", PAPER_DIR / "camera_ready_outputs"))
SRC = ART / "rq2_hl_bootstrap.json"
PDF_OUT = PAPER_DIR / "fig" / "fig4_replay.pdf"
PNG_OUT = PAPER_DIR / "fig" / "fig4_replay.png"

CONDITIONS = [
    ("wrong", "Unrelated", "0072B2", "circle"),
    ("filler", "Filler", "E69F00", "square"),
    ("absent", "Removed", "CC79A7", "diamond"),
]
# Camera-ready: the second genuine sample of the same prompts is the reference every
# replacement is read against; drawn as a fourth row and as a shaded band when present.
RESAMPLE = ("genuine_resample", "Genuine, resampled", "595959", "circle")
SEED = 20260819
MARGIN = 250.0   # the submitted paper's dashed band; drawn only when no resample row exists


def symlog(value: float) -> float:
    """Signed logarithm with a linear neighborhood around zero."""
    if abs(value) <= 100:
        return value / 100
    return math.copysign(1 + math.log10(abs(value) / 100), value)


def marker(x: float, y: float, shape: str, color: str, size: float,
           opacity: float = 1.0) -> str:
    if shape == "circle":
        return (f"\\fill[plot{color},opacity={opacity:.2f}] "
                f"({x:.4f},{y:.4f}) circle ({size:.3f}cm);")
    if shape == "square":
        return (f"\\fill[plot{color},opacity={opacity:.2f}] "
                f"({x-size:.4f},{y-size:.4f}) rectangle "
                f"({x+size:.4f},{y+size:.4f});")
    return (f"\\fill[plot{color},opacity={opacity:.2f}] "
            f"({x:.4f},{y+size:.4f}) -- ({x+size:.4f},{y:.4f}) -- "
            f"({x:.4f},{y-size:.4f}) -- ({x-size:.4f},{y:.4f}) -- cycle;")


def map_linear(value: float, lo: float, hi: float, out_lo: float, out_hi: float) -> float:
    return out_lo + (value - lo) * (out_hi - out_lo) / (hi - lo)


def build_tex(data: dict) -> str:
    rng = random.Random(SEED)
    conditions = list(CONDITIONS) + ([RESAMPLE] if RESAMPLE[0] in data else [])
    has_resample = len(conditions) == 4
    lines: list[str] = []
    a_x0, a_x1 = 0.95, 5.05
    b_x0, b_x1 = 6.45, 10.05
    c_x0, c_x1 = 11.20, 13.55
    y0, y1 = 1.12, 5.05

    lines.extend([
        r"\documentclass{article}",
        r"\usepackage[paperwidth=13.97cm,paperheight=5.97cm,margin=0pt]{geometry}",
        r"\usepackage{tikz}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathptmx}",
        r"\usetikzlibrary{patterns}",
        r"\definecolor{ink}{HTML}{222222}",
        r"\definecolor{muted}{HTML}{666666}",
        r"\definecolor{grid}{HTML}{D9D9D9}",
        r"\definecolor{plot0072B2}{HTML}{0072B2}",
        r"\definecolor{plotE69F00}{HTML}{E69F00}",
        r"\definecolor{plotCC79A7}{HTML}{CC79A7}",
        r"\definecolor{plot222222}{HTML}{222222}",
        r"\definecolor{plot595959}{HTML}{595959}",
        r"\definecolor{band}{HTML}{EBEBEB}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]",
        r"\useasboundingbox (0,0) rectangle (13.97,5.97);",
        r"\tikzset{tick/.style={font=\fontsize{6.8}{7.4}\selectfont,text=ink},"
        r" title/.style={font=\fontsize{8.2}{9}\selectfont\bfseries,text=ink},"
        r" axislabel/.style={font=\fontsize{7.2}{8}\selectfont,text=ink},"
        r" xtick/.style={font=\fontsize{6.2}{6.8}\selectfont,text=ink},"
        r" note/.style={font=\fontsize{6.2}{7}\selectfont,text=muted},"
        r" panel/.style={font=\fontsize{9}{9}\selectfont\bfseries,text=ink}}",
    ])

    # Panel A axes and labels.
    lines += [
        f"\\draw[muted,line width=.45pt] ({a_x0},{y0}) -- ({a_x0},{y1});",
        f"\\draw[muted,line width=.45pt] ({a_x0},{y0}) -- ({a_x1},{y0});",
        f"\\node[title,anchor=south] at ({(a_x0+a_x1)/2},{y1+0.24}) {{(A) Paired rank differences}};",
        f"\\node[axislabel,rotate=90] at ({a_x0-0.84},{(y0+y1)/2}) {{Rank difference}};",
        f"\\node[note,anchor=north east] at ({a_x1-0.06},{y1-0.04}) {{Genuine worse}};",
        f"\\node[note,anchor=south east] at ({a_x1-0.06},{y0+0.04}) {{Genuine better}};",
    ]
    # Keep the labelled ticks unchanged while reserving unlabelled headroom
    # beyond the observed 10^5 tails for the directional annotations.
    sy_lo, sy_hi = -4.7, 4.7
    for value, label in [(-10000, r"$-10^4$"), (-100, r"$-10^2$"),
                         (0, "0"), (100, r"$10^2$"), (10000, r"$10^4$")]:
        yy = map_linear(symlog(value), sy_lo, sy_hi, y0, y1)
        lines.append(f"\\draw[grid,line width=.35pt] ({a_x0},{yy:.4f}) -- ({a_x1},{yy:.4f});")
        lines.append(f"\\node[tick,anchor=east] at ({a_x0-0.10},{yy:.4f}) {{{label}}};")
    x_positions = [1.60, 3.00, 4.40] if not has_resample else [1.45, 2.50, 3.55, 4.60]
    for x, (key, label, color, shape) in zip(x_positions, conditions):
        values = [float(v) for v in data[key]["diffs"]]
        for value in values:
            yy = map_linear(max(sy_lo, min(sy_hi, symlog(value))), sy_lo, sy_hi, y0, y1)
            lines.append(marker(x + rng.uniform(-0.24, 0.24), yy, shape, color, 0.018, 0.26))
        median = sorted(values)[len(values) // 2]
        my = map_linear(symlog(median), sy_lo, sy_hi, y0, y1)
        lines.append(f"\\draw[ink,line width=1.1pt] ({x-0.30},{my:.4f}) -- ({x+0.30},{my:.4f});")
        lines.append(marker(x, my, "diamond", "222222", 0.060, 1.0))
        lines.append(f"\\node[xtick,anchor=north,align=center] at ({x},{y0-0.14}) {{{label.replace(', ', chr(92) * 2)}}};")

    # Panel B forest plot.
    lines += [
        f"\\node[title,anchor=south] at ({(b_x0+b_x1)/2},{y1+0.24}) {{(B) Estimated intervention effect}};",
    ]
    bx = lambda v: map_linear(v, -500, 650, b_x0, b_x1)
    if has_resample:
        # Reference band: the interval of genuine-minus-genuine (sampling variation alone), drawn first.
        ref = data[RESAMPLE[0]]
        lines.append(f"\\fill[band] ({bx(max(-500, ref['ci_lo'])):.4f},{y0}) rectangle ({bx(min(650, ref['ci_hi'])):.4f},{y1});")
    for value in (-250, 0, 250, 500):
        xx = bx(value)
        lines.append(f"\\draw[grid,line width=.35pt] ({xx:.4f},{y0}) -- ({xx:.4f},{y1});")
        lines.append(f"\\node[tick,anchor=north] at ({xx:.4f},{y0-0.12}) {{{value}}};")
    lines.append(f"\\draw[muted,line width=.55pt] ({bx(0):.4f},{y0}) -- ({bx(0):.4f},{y1});")
    if not has_resample:
        lines.append(f"\\draw[muted,dashed,line width=.55pt] ({bx(-MARGIN):.4f},{y0}) -- ({bx(-MARGIN):.4f},{y1});")
        lines.append(f"\\draw[muted,dashed,line width=.55pt] ({bx(MARGIN):.4f},{y0}) -- ({bx(MARGIN):.4f},{y1});")
    y_positions = [4.20, 3.05, 1.90] if not has_resample else [4.40, 3.50, 2.60, 1.70]
    for yy, (key, label, color, shape) in zip(y_positions, conditions):
        item = data[key]
        lo, hi, estimate = item["ci_lo"], item["ci_hi"], item["hl"]
        value_text = "0" if estimate == 0 else f"{estimate:+.0f}"
        label_x = (bx(hi) + bx(MARGIN)) / 2 if estimate == 0 else bx(hi) + 0.08
        label_anchor = "center" if estimate == 0 else "west"
        lines += [
            f"\\node[tick,anchor=east,align=right] at ({b_x0-0.12},{yy}) {{{label.replace(', ', chr(92) * 2)}}};",
            f"\\draw[plot{color},line width=1.1pt] ({bx(lo):.4f},{yy}) -- ({bx(hi):.4f},{yy});",
            f"\\draw[plot{color},line width=.8pt] ({bx(lo):.4f},{yy-0.11}) -- ({bx(lo):.4f},{yy+0.11});",
            f"\\draw[plot{color},line width=.8pt] ({bx(hi):.4f},{yy-0.11}) -- ({bx(hi):.4f},{yy+0.11});",
            marker(bx(estimate), yy, shape, color, 0.075, 1.0),
            f"\\node[note,anchor={label_anchor},text=ink] at ({label_x:.4f},{yy}) {{{value_text}}};",
        ]
    lines += [
        f"\\draw[muted,line width=.45pt] ({b_x0},{y0}) -- ({b_x1},{y0});",
        f"\\node[axislabel,anchor=north] at ({(b_x0+b_x1)/2},{y0-0.48}) {{Rank effect with 95\\% interval}};",
    ]

    # Panel C agreement.
    lines += [
        f"\\node[title,anchor=south] at ({(c_x0+c_x1)/2},{y1+0.24}) {{(C) Proposal agreement}};",
    ]
    cx = lambda v: map_linear(v, 10, 21, c_x0, c_x1)
    if has_resample:
        ref = data["identity"][RESAMPLE[0]]
        lines.append(f"\\fill[band] ({cx(max(10, 100 * ref['ci_lo'])):.4f},{y0}) rectangle ({cx(min(21, 100 * ref['ci_hi'])):.4f},{y1});")
    for value in (12, 16, 20):
        xx = cx(value)
        lines.append(f"\\draw[grid,line width=.35pt] ({xx:.4f},{y0}) -- ({xx:.4f},{y1});")
        lines.append(f"\\node[tick,anchor=north] at ({xx:.4f},{y0-0.12}) {{{value}}};")
    for yy, (key, label, color, shape) in zip(y_positions, conditions):
        item = data["identity"][key]
        lo, hi, rate = 100 * item["ci_lo"], 100 * item["ci_hi"], 100 * item["rate"]
        lines += [
            f"\\draw[plot{color},line width=1.1pt] ({cx(lo):.4f},{yy}) -- ({cx(hi):.4f},{yy});",
            f"\\draw[plot{color},line width=.8pt] ({cx(lo):.4f},{yy-0.11}) -- ({cx(lo):.4f},{yy+0.11});",
            f"\\draw[plot{color},line width=.8pt] ({cx(hi):.4f},{yy-0.11}) -- ({cx(hi):.4f},{yy+0.11});",
            marker(cx(rate), yy, shape, color, 0.075, 1.0),
        ]
    lines += [
        f"\\draw[muted,line width=.45pt] ({c_x0},{y0}) -- ({c_x1},{y0});",
        f"\\node[axislabel,anchor=north] at ({(c_x0+c_x1)/2},{y0-0.48}) {{Identical first word (\\%)}};",
        r"\end{tikzpicture}",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    data = json.loads(SRC.read_text())
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="rq2_figure_") as tmp_name:
        tmp = Path(tmp_name)
        tex = tmp / "fig4_replay.tex"
        tex.write_text(build_tex(data))
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(tmp), str(tex)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout)
        shutil.copy2(tmp / "fig4_replay.pdf", PDF_OUT)
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "300",
             str(PDF_OUT), str(PNG_OUT.with_suffix(""))],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    for key, label, _, _ in list(CONDITIONS) + ([RESAMPLE] if RESAMPLE[0] in data else []):
        item = data[key]
        print(f"{label}: n={item['n_events']}, HL={item['hl']:+.0f}, "
              f"95% interval [{item['ci_lo']:+.1f}, {item['ci_hi']:+.1f}]")
    print(f"wrote {PDF_OUT}")
    print(f"wrote {PNG_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
