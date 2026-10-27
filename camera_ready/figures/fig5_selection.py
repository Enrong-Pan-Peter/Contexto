# Academic Figure Skill Asset Confirmation
# GroupedBarChart and BarComparison assets were inspected. Native reuse was
# rejected because Panels A and B are signed point estimates without intervals,
# while Panels C and D mix paired counts and a zero-inflated count contrast.
# Their restrained axes, direct values, and shared categorical encoding are
# inherited without manufacturing uncertainty that the artifacts lack.

"""Generate the selection response figure as vector TikZ line art.

The local manuscript runtime does not include matplotlib. This implementation
maps the Academic Figure Skill typography, palette, and export specifications
onto TikZ and uses only the Python standard library plus the existing LaTeX
toolchain. It writes a vector PDF and a 300 dpi PNG preview.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path


# Academic Figure Skill Nature/Cell/Science Color Palette -- COPY VERBATIM
CATEGORICAL = ["#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666"]
CATEGORICAL_EXTENDED = [
    "#2166AC", "#B2182B", "#1B7837", "#F1A340", "#762A83", "#666666",
    "#4393C3", "#D6604D", "#5AAE61", "#B35806", "#9970AB", "#999999",
]
DIVERGING = ["#2166AC", "#F7F7F7", "#B2182B"]
SEQUENTIAL = ["#F7FBFF", "#6BAED6", "#08306B"]
ACCENT_RED = "#B2182B"
GREY = "#999999"
BLACK = "#222222"

# Candidate semantic pairs for direct visual comparison. Shape and fill remain
# redundant encodings so the comparison does not depend on color alone.
PALETTE_VARIANTS = {
    "aubergine_copper": ("#5A3D6E", "#AD5E32"),
    "plum_azure": ("#754F7C", "#2878A5"),
}
GRID_COLOR = "#D9D9D9"     # shared neutral rule
AXIS_COLOR = "#666666"


import os

PAPER_DIR = Path(__file__).resolve().parent.parent
# Camera-ready: the package written by scripts/camera_ready.py (Contexto repo); CONTEXTO_ART_DIR
# overrides the default location next to the paper.
ART_DIR = Path(os.environ.get("CONTEXTO_ART_DIR", PAPER_DIR / "camera_ready_outputs"))
FIG_DIR = PAPER_DIR / "fig"

SOURCES = {
    "fitness": ART_DIR / "rq3_A1_3games" / "rq3_summary.json",
    "random": ART_DIR / "rq3_random" / "rq3_summary.json",
}

# Locked behavioral audit from final_batch_report_2026-08-17.md and the
# manuscript's verified RQ3 summary. Each target has five runs per mechanism.
RUNS_SOLVED = {
    1303: {"fitness": 5, "random": 5},
    1319: {"fitness": 5, "random": 1},
    1372: {"fitness": 5, "random": 5},
}


def map_linear(value: float, low: float, high: float,
               output_low: float, output_high: float) -> float:
    return output_low + (value - low) * (output_high - output_low) / (high - low)


def load_results() -> dict:
    output = {}
    for key, source in SOURCES.items():
        with source.open() as handle:
            payload = json.load(handle)
        def with_interval(test: dict, key: str) -> tuple[float, float | None, float | None]:
            # Camera-ready: every point carries its 95 % run-resampled interval (rq3_lineage --bootstrap).
            ci = test.get("ci95") or {}
            return test[key], ci.get("low"), ci.get("high")

        output[key] = {
            "probability": [
                with_interval(payload["tests"]["binary_error"]["gap_permutation"], "observed_gap"),
                with_interval(payload["tests"]["binary_error"]["transmission"], "observed_rho"),
            ],
            "bucket": [
                with_interval(payload["tests"]["bucket_distance"]["gap_permutation"], "observed_gap"),
                with_interval(payload["tests"]["bucket_distance"]["transmission"], "observed_rho"),
            ],
            "reselections": payload["counts_total"]["resurrections"],
        }
    return output


def marker(x: float, y: float, mechanism: str, size: float = 0.075) -> str:
    if mechanism == "fitness":
        return f"\\fill[fitness] ({x:.4f},{y:.4f}) circle ({size:.3f}cm);"
    return (
        f"\\draw[random,line width=1pt,fill=white] "
        f"({x-size:.4f},{y-size:.4f}) rectangle ({x+size:.4f},{y+size:.4f});"
    )


def build_tex(results: dict, fitness_color: str, random_color: str) -> str:
    lines = [
        r"\documentclass{article}",
        r"\usepackage[paperwidth=13.97cm,paperheight=5.45cm,margin=0pt]{geometry}",
        r"\usepackage{tikz}",
        r"\usepackage[T1]{fontenc}",
        r"\usepackage{mathptmx}",
        r"\definecolor{ink}{HTML}{222222}",
        rf"\definecolor{{muted}}{{HTML}}{{{AXIS_COLOR[1:]}}}",
        rf"\definecolor{{grid}}{{HTML}}{{{GRID_COLOR[1:]}}}",
        rf"\definecolor{{fitness}}{{HTML}}{{{fitness_color[1:]}}}",
        rf"\definecolor{{random}}{{HTML}}{{{random_color[1:]}}}",
        r"\pagestyle{empty}",
        r"\setlength{\parindent}{0pt}",
        r"\begin{document}",
        r"\begin{tikzpicture}[x=1cm,y=1cm,line cap=round,line join=round]",
        r"\useasboundingbox (0,0) rectangle (13.97,5.45);",
        r"\tikzset{tick/.style={font=\fontsize{6.8}{7.4}\selectfont,text=ink},"
        r" axislabel/.style={font=\fontsize{7.2}{8}\selectfont,text=ink},"
        r" rowlabel/.style={font=\fontsize{6.5}{7.0}\selectfont,text=ink,align=right},"
        r" group/.style={font=\fontsize{8}{8.6}\selectfont\bfseries,text=ink},"
        r" panel/.style={font=\fontsize{9}{9}\selectfont\bfseries,text=ink}}",
        r"\node[group,anchor=center] at (3.68,4.78) {No detectable response in report accuracy (95\% intervals)};",
        r"\node[group,anchor=center] at (10.55,4.78) {Search outcomes and dynamics differ};",
        r"\fill[fitness] (5.13,5.19) circle (0.072cm);",
        r"\node[tick,anchor=west] at (5.29,5.19) {Fitness selection};",
        r"\draw[random,line width=1pt,fill=white] (7.34,5.12) rectangle (7.48,5.26);",
        r"\node[tick,anchor=west] at (7.60,5.19) {Random selection};",
    ]

    y_rows = [3.35, 1.80]

    # Panels A and B: reporting diagnostics.
    # Camera-ready: the axes are zoomed from the full ranges (+-1, +-3) so the 95 % intervals
    # are readable; the caption states the full range.
    metric_panels = [
        ("A", "probability", 0.3, 2.08, 4.20, "Probability error", True),
        ("B", "bucket", 0.6, 5.18, 7.18, "Bucket distance", False),
    ]
    panel_letter_x = {"A": 0.18, "B": 4.65}
    for panel, metric, limit, x0, x1, label, show_rows in metric_panels:
        lines.append(
            f"\\node[panel,anchor=west] at ({panel_letter_x[panel]:.2f},4.26) {{{panel}}};"
        )
        if show_rows:
            lines += [
                f"\\node[rowlabel,anchor=east] at ({x0-0.16:.2f},{y_rows[0]:.2f}) "
                r"{selection\\differential};",
                f"\\node[rowlabel,anchor=east] at ({x0-0.16:.2f},{y_rows[1]:.2f}) "
                r"{parent--offspring\\correlation};",
            ]
        zero_x = map_linear(0, -limit, limit, x0, x1)
        lines.append(f"\\draw[muted,line width=.55pt] ({zero_x:.4f},1.25) -- ({zero_x:.4f},3.88);")
        for value in (-limit, 0, limit):
            x = map_linear(value, -limit, limit, x0, x1)
            lines.append(f"\\draw[grid,line width=.35pt] ({x:.4f},1.25) -- ({x:.4f},3.88);")
            text = "0" if value == 0 else f"{'+' if value > 0 else '$-$'}{abs(value):g}"
            lines.append(f"\\node[tick,anchor=north] at ({x:.4f},1.10) {{{text}}};")
        for mechanism, offset in (("fitness", 0.12), ("random", -0.12)):
            for row_y, (value, lo, hi) in zip(y_rows, results[mechanism][metric]):
                x = map_linear(value, -limit, limit, x0, x1)
                if lo is not None and hi is not None:
                    xl = map_linear(max(-limit, lo), -limit, limit, x0, x1)
                    xh = map_linear(min(limit, hi), -limit, limit, x0, x1)
                    yy = row_y + offset
                    lines += [
                        f"\\draw[{mechanism},line width=1.0pt] ({xl:.4f},{yy:.4f}) -- ({xh:.4f},{yy:.4f});",
                        f"\\draw[{mechanism},line width=.7pt] ({xl:.4f},{yy-0.08:.4f}) -- ({xl:.4f},{yy+0.08:.4f});",
                        f"\\draw[{mechanism},line width=.7pt] ({xh:.4f},{yy-0.08:.4f}) -- ({xh:.4f},{yy+0.08:.4f});",
                    ]
                lines.append(marker(x, row_y + offset, mechanism))
        lines += [
            f"\\draw[muted,line width=.45pt] ({x0:.4f},1.25) -- ({x1:.4f},1.25);",
            f"\\node[axislabel,anchor=north] at ({(x0+x1)/2:.4f},0.66) {{{label}}};",
        ]

    # Panel C: paired solved counts by target.
    c_x0, c_x1 = 8.35, 10.25
    c_rows = [3.55, 2.55, 1.55]
    lines += [
        r"\node[panel,anchor=west] at (7.75,4.26) {C};",
    ]
    for value in (0, 1, 3, 5):
        x = map_linear(value, 0, 5, c_x0, c_x1)
        lines.append(f"\\draw[grid,line width=.35pt] ({x:.4f},1.25) -- ({x:.4f},3.88);")
        lines.append(f"\\node[tick,anchor=north] at ({x:.4f},1.10) {{{value}}};")
    for target, row_y in zip(RUNS_SOLVED, c_rows):
        lines += [
            f"\\node[tick,anchor=east] at ({c_x0-0.12:.4f},{row_y:.4f}) {{{target}}};",
            f"\\draw[grid,line width=.35pt] ({c_x0:.4f},{row_y:.4f}) -- ({c_x1:.4f},{row_y:.4f});",
        ]
        for mechanism, offset in (("fitness", 0.12), ("random", -0.12)):
            x = map_linear(RUNS_SOLVED[target][mechanism], 0, 5, c_x0, c_x1)
            lines.append(marker(x, row_y + offset, mechanism))
    lines += [
        f"\\draw[muted,line width=.45pt] ({c_x0},1.25) -- ({c_x1},1.25);",
        f"\\node[axislabel,anchor=north] at ({(c_x0+c_x1)/2:.4f},0.66) {{Runs solved (of 5)}};",
    ]

    # Panel D: archive reselections as a vertical count plot. A shared baseline
    # makes the zero count visible and lets the 542 count read at a glance.
    d_x0, d_x1 = 11.35, 13.35
    d_y0, d_y1 = 1.25, 3.88
    d_positions = {"fitness": 11.85, "random": 12.85}
    lines.append(r"\node[panel,anchor=west] at (10.92,4.26) {D};")
    for value in (0, 250, 500):
        y = map_linear(value, 0, 600, d_y0, d_y1)
        lines.append(f"\\draw[grid,line width=.35pt] ({d_x0:.4f},{y:.4f}) -- ({d_x1:.4f},{y:.4f});")
        lines.append(f"\\node[tick,anchor=east] at ({d_x0-0.10:.4f},{y:.4f}) {{{value}}};")
    for mechanism, x in d_positions.items():
        value = results[mechanism]["reselections"]
        y = map_linear(value, 0, 600, d_y0, d_y1)
        label = "Fitness" if mechanism == "fitness" else "Random"
        lines += [
            f"\\draw[{mechanism},line width=1.15pt] ({x:.4f},{d_y0:.4f}) -- ({x:.4f},{y:.4f});",
            marker(x, y, mechanism, 0.082),
            f"\\node[tick,anchor=south] at ({x:.4f},{y+0.10:.4f}) {{{value}}};",
            f"\\node[tick,anchor=north] at ({x:.4f},1.10) {{{label}}};",
        ]
    lines += [
        f"\\draw[muted,line width=.45pt] ({d_x0},{d_y0}) -- ({d_x1},{d_y0});",
        f"\\node[axislabel,anchor=north] at ({(d_x0+d_x1)/2:.4f},0.49) {{Archive reselections}};",
        r"\end{tikzpicture}",
        r"\end{document}",
    ]
    return "\n".join(lines) + "\n"


def render_variant(results: dict, variant: str,
                   fitness_color: str, random_color: str) -> tuple[Path, Path]:
    pdf_out = FIG_DIR / f"fig5_selection_{variant}.pdf"
    png_out = FIG_DIR / f"fig5_selection_{variant}.png"
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fig5_selection_") as temp_name:
        temp = Path(temp_name)
        tex = temp / "fig5_selection_codex.tex"
        tex.write_text(build_tex(results, fitness_color, random_color))
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error",
             "-output-directory", str(temp), str(tex)],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout)
        shutil.copy2(temp / "fig5_selection_codex.pdf", pdf_out)
        subprocess.run(
            ["pdftoppm", "-png", "-singlefile", "-r", "300",
             str(pdf_out), str(png_out.with_suffix(""))],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    print(f"wrote {pdf_out}")
    print(f"wrote {png_out}")
    return pdf_out, png_out


def main() -> int:
    results = load_results()
    rendered = {}
    for variant, (fitness_color, random_color) in PALETTE_VARIANTS.items():
        rendered[variant] = render_variant(results, variant, fitness_color, random_color)
    selected_pdf, selected_png = rendered["aubergine_copper"]
    shutil.copy2(selected_pdf, FIG_DIR / "fig5_selection_codex.pdf")
    shutil.copy2(selected_png, FIG_DIR / "fig5_selection_codex.png")
    print(f"selected {selected_pdf.name} as fig5_selection_codex.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
