"""Batch runner for the MAP-Elites sigma-mode control.

Orchestrates the sigma-control experiment: for each sigma mode arm it launches a
single ``python -m contexto_solver.experiment`` subprocess (with the arm selected
via the ``MAPELITES_SIGMA_MODE`` environment variable), then renders MAP-Elites
figures for every successful trace produced by that arm.

The four arms isolate the effect of self-adaptive sigma by holding everything
else constant (anchors, ranked-context K, targets, seeds, generations):
  * adaptive       - current behavior (Dirichlet perturbation of parent sigma)
  * frozen_uniform - operator probabilities pinned to the uniform prior
  * frozen_fixed   - operator probabilities pinned to MAPELITES_FROZEN_SIGMA
  * random         - operator probabilities redrawn from Dirichlet(1) each child

This script does not reimplement the experiment or the plotter; it only sets the
environment per arm and shells out to the existing entrypoints.

Usage:
    python scripts/run_sigma_control.py [--targets superficial,notorious,house]
        [--runs-per-target 5] [--random-seed 1] [--max-generations 70]
        [--ranked-context-k 20] [--provider ollama] [--ollama-model qwen3:14b]
        [--modes adaptive,frozen_uniform,frozen_fixed,random]
        [--output-dir traces] [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ALL_MODES = ("adaptive", "frozen_uniform", "frozen_fixed", "random")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the MAP-Elites sigma-mode control batch.")
    parser.add_argument("--targets", default="superficial,notorious,house",
                        help="Comma-separated local target words, constant across arms.")
    parser.add_argument("--runs-per-target", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=1,
                        help="Base random seed, held constant across arms for paired comparison.")
    parser.add_argument("--max-generations", type=int, default=70)
    parser.add_argument("--ranked-context-k", type=int, default=20,
                        help="MAPELITES_RANKED_CONTEXT_K, held constant across all arms.")
    parser.add_argument("--provider", default="ollama", choices=["openai", "anthropic", "ollama"])
    parser.add_argument("--ollama-model", default="qwen3:14b")
    parser.add_argument("--mode", default="aligned", choices=["aligned", "non_aligned"])
    parser.add_argument("--modes", default=",".join(ALL_MODES),
                        help="Comma-separated sigma modes (arms) to run.")
    parser.add_argument("--output-dir", default="traces",
                        help="Directory for the per-arm summary JSON files.")
    parser.add_argument("--combined-figures", action="store_true",
                        help="Also write the combined MAP-Elites summary PNG per trace.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the commands and environment overrides without running them.")
    return parser.parse_args()


def _selected_modes(raw: str) -> list[str]:
    modes = [piece.strip() for piece in raw.split(",") if piece.strip()]
    unknown = [mode for mode in modes if mode not in ALL_MODES]
    if unknown:
        raise SystemExit(f"Unknown sigma modes {unknown}; valid modes are {list(ALL_MODES)}.")
    return modes


def _experiment_command(args: argparse.Namespace, output_path: Path) -> list[str]:
    command = [
        sys.executable, "-m", "contexto_solver.experiment",
        "--method", "ea_llm_map_elites",
        "--mode", args.mode,
        "--targets", args.targets,
        "--runs-per-target", str(args.runs_per_target),
        "--max-generations", str(args.max_generations),
        "--provider", args.provider,
        "--output", str(output_path),
    ]
    if args.random_seed is not None:
        command += ["--random-seed", str(args.random_seed)]
    if args.provider == "ollama":
        command += ["--ollama-model", args.ollama_model]
    return command


def _plot_command(args: argparse.Namespace, trace_path: str) -> list[str]:
    command = [sys.executable, "-m", "contexto_solver.plot_map_elites", "--trace", trace_path]
    if args.combined_figures:
        command.append("--combined")
    return command


def _arm_env(mode: str, ranked_context_k: int) -> dict[str, str]:
    env = os.environ.copy()
    env["MAPELITES_SIGMA_MODE"] = mode
    env["MAPELITES_RANKED_CONTEXT_K"] = str(ranked_context_k)
    return env


def _trace_paths(output_path: Path) -> list[str]:
    if not output_path.exists():
        return []
    data = json.loads(output_path.read_text(encoding="utf-8"))
    paths: list[str] = []
    for row in data.get("runs", []):
        trace_path = row.get("trace_path")
        if trace_path:
            paths.append(trace_path)
    return paths


def main() -> int:
    args = _parse_args()
    modes = _selected_modes(args.modes)
    output_dir = Path(args.output_dir)

    for mode in modes:
        output_path = output_dir / f"sigma_control_{mode}.json"
        env = _arm_env(mode, args.ranked_context_k)
        command = _experiment_command(args, output_path)
        overrides = f"MAPELITES_SIGMA_MODE={mode} MAPELITES_RANKED_CONTEXT_K={args.ranked_context_k}"

        if args.dry_run:
            print(f"[{mode}] {overrides}")
            print(f"[{mode}] {' '.join(command)}")
            print(f"[{mode}] then plot each successful trace from {output_path}")
            continue

        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"=== arm: {mode} ({overrides}) ===", flush=True)
        result = subprocess.run(command, env=env)
        if result.returncode != 0:
            print(f"[{mode}] experiment exited with code {result.returncode}; skipping plots for this arm.")
            continue

        for trace_path in _trace_paths(output_path):
            plot_command = _plot_command(args, trace_path)
            print(f"[{mode}] plotting {trace_path}", flush=True)
            plot_result = subprocess.run(plot_command, env=env)
            if plot_result.returncode != 0:
                print(f"[{mode}] plot failed for {trace_path} (code {plot_result.returncode}); continuing.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
