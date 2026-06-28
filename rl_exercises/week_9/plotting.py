"""Disclosure: The plotting script was generated with Claude Code (Opus-4.8)
Plot Dyna-PPO sweep results with RLiable.

Loads the per-run CSV files produced by the Hydra sweeps in ``run11.sh`` and
``run21.sh`` and draws an IQM sample-efficiency curve (average eval reward vs.
real environment steps) with stratified bootstrap confidence intervals.

Both run scripts share the same directory layout::

    <sweep_dir>/<env.name>/<variant>/seed_<seed>/out.csv

where each ``out.csv`` has the columns ``real_steps,avg_reward,std_reward``.
The ``<variant>`` level is whatever the sweep varies:

* ``run11.sh``  -> ``use_model`` (True/False), labelled as Dyna-PPO / PPO
* ``run21.sh``  -> ``imag_horizon`` (21a), a named config (21b) or
  ``max_buffer_size`` (21c).

Usage:
    # Model-based vs. model-free (default labelling)
    python plotting.py

    # Hyperparameter sweeps: label each curve with the swept value
    python plotting.py --sweep-dir outputs/ppo_sweep_21a \\
        --param imag_horizon --out 21a.png
    python plotting.py --sweep-dir outputs/ppo_sweep_21b \\
        --param config --out 21b.png
    python plotting.py --sweep-dir outputs/ppo_sweep_21c \\
        --param max_buffer_size --out 21c.png
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rliable import library as rly
from rliable import metrics, plot_utils


def make_label(env_name: str, variant: str, param: str | None) -> str:
    """
    Build the legend label for a run.

    Args:
        env_name (str): Environment name (sweep subdir level just above variant).
        variant (str): The varied value (e.g. ``True``/``False`` or ``imag_horizon``).
        param (str | None): Name of the swept hyperparameter. If ``None`` the
            run is assumed to be a ``use_model`` sweep and labelled Dyna-PPO/PPO.

    Returns:
        str: Legend label.
    """
    if param is None:
        algo = "Dyna-PPO" if variant == "True" else "PPO"
        return f"{env_name} | {algo}"
    return f"{env_name} | {param}={variant}"


def _variant_sort_key(variant: str) -> tuple[int, float | str]:
    """Sort numeric variants numerically, falling back to alphabetical order."""
    try:
        return (0, float(variant))
    except ValueError:
        return (1, variant)


def load_scores(
    sweep_dir: Path, param: str | None
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    runs: dict[str, list[np.ndarray]] = defaultdict(list)
    variant_of: dict[str, str] = {}
    frames: np.ndarray | None = None

    csv_paths = sorted(sweep_dir.glob("**/out.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No out.csv files found under {sweep_dir!r}")

    for csv_path in csv_paths:
        # Path: <sweep_dir>/<env>/<variant>/seed_<seed>/out.csv
        variant = csv_path.parent.parent.name
        env_name = csv_path.parent.parent.parent.name
        algo = make_label(env_name, variant, param)
        variant_of[algo] = variant

        df = pd.read_csv(csv_path).sort_values("real_steps")
        if frames is None:
            frames = df["real_steps"].to_numpy()
        elif not np.array_equal(frames, df["real_steps"].to_numpy()):
            raise ValueError(
                f"Inconsistent real_steps axis in {csv_path}; "
                "all runs must share the same eval schedule."
            )
        runs[algo].append(df["avg_reward"].to_numpy())

    # Stack into (num_runs, num_tasks=1, num_frames) as expected by rliable,
    # ordering curves by their (numeric when possible) variant value.
    ordered = sorted(runs, key=lambda algo: _variant_sort_key(variant_of[algo]))
    scores = {algo: np.stack(runs[algo], axis=0)[:, None, :] for algo in ordered}
    assert frames is not None
    return scores, frames


def plot(
    scores: dict[str, np.ndarray],
    frames: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """
    Compute IQM sample-efficiency curves and save the RLiable figure.

    Args:
        scores (dict): Algorithm name -> array of shape ``(num_runs, num_tasks, num_frames)``.
        frames (np.ndarray): The ``real_steps`` axis.
        out_path (Path): Where to save the resulting figure.
        title (str): Figure title.
    """
    # IQM across runs x tasks at every frame, with stratified bootstrap CIs.
    iqm = lambda x: np.array(  # noqa: E731
        [metrics.aggregate_iqm(x[..., i]) for i in range(x.shape[-1])]
    )
    iqm_scores, iqm_cis = rly.get_interval_estimates(scores, iqm, reps=2000)

    plt.rcParams.update({"figure.figsize": (8, 5)})
    plot_utils.plot_sample_efficiency_curve(
        frames,
        iqm_scores,
        iqm_cis,
        algorithms=list(scores.keys()),
        xlabel="Real environment steps",
        ylabel="IQM eval reward",
    )
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-dir",
        type=Path,
        default=Path(__file__).parent / "outputs" / "ppo_sweep_11",
        help="Root directory of the Hydra sweep output.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).parent / "11.png",
        help="Path to save the generated figure.",
    )
    parser.add_argument(
        "--param",
        type=str,
        default=None,
        help=(
            "Name of the swept hyperparameter (e.g. imag_horizon, max_buffer_size). "
            "When given, curves are labelled '<param>=<value>'. When omitted, the "
            "variant level is treated as use_model and labelled Dyna-PPO / PPO."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Figure title (defaults to a sensible description).",
    )
    args = parser.parse_args()

    scores, frames = load_scores(args.sweep_dir, args.param)

    for algo, arr in scores.items():
        print(f"{algo}: {arr.shape[0]} runs, {arr.shape[-1]} eval points")

    if args.title is not None:
        title = args.title
    elif args.param is None:
        title = "Sample efficiency: average reward vs. real steps"
    else:
        title = f"Sample efficiency vs. {args.param}"

    plot(scores, frames, args.out, title)


if __name__ == "__main__":
    main()
