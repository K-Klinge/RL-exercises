"""
Aggregate actor-critic CSV logs and plot baseline comparisons with RLiable.

Expected layout (matches the Hydra multirun in the README recipe):
    <root>/<env>/<baseline>/seed_<i>/ppo.csv
CSV rows: step,mean_return,std_return (no header).

Usage:
    python -m rl_exercises.week_6.plotting --root outputs/ac_sweep --out plots
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

BASELINES = ("actor_critic", "vanilla", "improved")


def discover_runs(root: Path) -> dict[str, dict[str, list[Path]]]:
    """Return {env: {baseline: [csv_paths]}} discovered under root."""
    runs: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    for csv_path in root.rglob("ppo.csv"):
        parts = csv_path.relative_to(root).parts
        # Expect <env>/<baseline>/seed_<i>/ppo.csv
        if len(parts) < 4:
            continue
        env, baseline = parts[0], parts[1]
        if baseline not in BASELINES:
            continue
        runs[env][baseline].append(csv_path)
    return runs


def load_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    return data[:, 0].astype(int), data[:, 1].astype(float)


def stack_seeds(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    """Load all seeds for one (env, baseline). Returns (steps, scores[n_seeds, n_evals])."""
    steps_ref: np.ndarray | None = None
    rows: list[np.ndarray] = []
    for p in sorted(paths):
        steps, returns = load_csv(p)
        if steps_ref is None:
            steps_ref = steps
        # Truncate to the shortest run so all seeds align.
        n = min(len(steps_ref), len(returns))
        steps_ref = steps_ref[:n]
        rows = [r[:n] for r in rows]
        rows.append(returns[:n])
    assert steps_ref is not None
    return steps_ref, np.stack(rows, axis=0)


def plot_env(env: str, by_baseline: dict[str, list[Path]], out_dir: Path) -> None:
    score_dict: dict[str, np.ndarray] = {}
    steps_ref: np.ndarray | None = None
    for baseline in BASELINES:
        if baseline not in by_baseline:
            continue
        steps, scores = stack_seeds(by_baseline[baseline])
        if steps_ref is None or len(steps) < len(steps_ref):
            steps_ref = steps
        score_dict[baseline] = scores

    if not score_dict:
        print(f"[skip] {env}: no runs found")
        return

    # Align all baselines to the shortest eval grid.
    n = min(arr.shape[1] for arr in score_dict.values())
    score_dict = {k: v[:, :n] for k, v in score_dict.items()}
    steps_ref = steps_ref[:n]

    # IQM with bootstrap CIs, evaluated independently at each eval step.
    iqm_curves: dict[str, np.ndarray] = {k: np.empty(n) for k in score_dict}
    iqm_cis: dict[str, np.ndarray] = {k: np.empty((2, n)) for k in score_dict}
    for i in range(n):
        sub = {k: v[:, i : i + 1] for k, v in score_dict.items()}
        point, ci = rly.get_interval_estimates(sub, metrics.aggregate_iqm, reps=2000)
        for k in score_dict:
            iqm_curves[k][i] = point[k]
            iqm_cis[k][:, i] = ci[k].squeeze()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    plot_utils.plot_sample_efficiency_curve(
        frames=steps_ref,
        point_estimates=iqm_curves,
        interval_estimates=iqm_cis,
        algorithms=list(score_dict.keys()),
        xlabel="Environment steps",
        ylabel="IQM return",
        ax=ax,
    )
    ax.set_title(f"{env}: actor-critic baselines")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', env)}.png"
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    n_seeds = {k: v.shape[0] for k, v in score_dict.items()}
    print(f"[ok]   {env}: saved {out_path} (seeds per baseline: {n_seeds})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Sweep root directory")
    parser.add_argument(
        "--out", type=Path, default=Path("plots"), help="Output dir for PNGs"
    )
    args = parser.parse_args()

    runs = discover_runs(args.root)
    if not runs:
        raise SystemExit(f"No ppo.csv files found under {args.root}")
    for env, by_baseline in runs.items():
        plot_env(env, by_baseline, args.out)


if __name__ == "__main__":
    main()
