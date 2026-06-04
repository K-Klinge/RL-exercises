"""
Disclosure: I used claude (sonnet 4.7) to get the plotting functions.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import pathlib

import gymnasium as gym
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from rl_exercises.week_4.dqn import DQNAgent
from rliable import metrics
from rliable.library import get_interval_estimates
from rliable.plot_utils import plot_interval_estimates, plot_sample_efficiency_curve

# CartPole-v1 max episode return — used to normalize for the optimality gap.
CARTPOLE_MAX_RETURN = 500.0

SEEDS = (0, 1, 2, 3, 4)
NUM_FRAMES: int = 20_000
EVAL_POINTS: int = 100
FINAL_WINDOW_EPISODES: int = 10
BOOTSTRAP_REPS: int = 2000

BASE_DIR = pathlib.Path(__file__).parent.resolve()
RUNS_DIR = BASE_DIR / "l2_runs"


def train_one_seed(seed: int, num_frames: int) -> Tuple[List[int], List[float]]:
    env = gym.make("CartPole-v1")
    agent = DQNAgent(
        env,
        buffer_capacity=10_000,
        batch_size=64,
        lr=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.01,
        epsilon_decay=500,
        target_update_freq=1000,
        seed=seed,
    )
    frames, rewards = agent.train(num_frames=num_frames, eval_interval=50)
    env.close()
    return frames, rewards


def resample_to_grid(
    frames: List[int],
    rewards: List[float],
    grid: np.ndarray,
    smoothing_window: int = 10,
) -> np.ndarray:
    if len(rewards) == 0:
        return np.zeros_like(grid, dtype=float)
    f = np.asarray(frames, dtype=float)
    r = np.asarray(rewards, dtype=float)
    if smoothing_window > 1:
        kernel = np.ones(smoothing_window) / smoothing_window
        smoothed = np.convolve(r, kernel, mode="full")[: len(r)]
        denom = np.minimum(np.arange(1, len(r) + 1), smoothing_window)
        smoothed = smoothed * smoothing_window / denom
    else:
        smoothed = r
    return np.interp(grid, f, smoothed, left=smoothed[0], right=smoothed[-1])


def run_all_seeds() -> Dict[int, pd.DataFrame]:
    RUNS_DIR.mkdir(exist_ok=True)
    per_seed: Dict[int, pd.DataFrame] = {}
    for seed in SEEDS:
        csv_path = RUNS_DIR / f"seed_{seed}.csv"
        if csv_path.exists():
            print(f"[seed {seed}] loading cached results from {csv_path.name}")
            per_seed[seed] = pd.read_csv(csv_path)
            continue
        print(f"[seed {seed}] training {NUM_FRAMES} frames…")
        frames, rewards = train_one_seed(seed, NUM_FRAMES)
        df = pd.DataFrame({"frame": frames, "reward": rewards})
        df.to_csv(csv_path, index=False)
        per_seed[seed] = df
    return per_seed


def build_score_matrix(
    per_seed: Dict[int, pd.DataFrame], grid: np.ndarray
) -> np.ndarray:
    rows = []
    for seed in SEEDS:
        df = per_seed[seed]
        rows.append(resample_to_grid(df["frame"].tolist(), df["reward"].tolist(), grid))
    return np.stack(rows, axis=0)


def plot_training_curve(grid: np.ndarray, scores: np.ndarray) -> None:
    train_scores = {"DQN": scores}
    iqm = lambda s: np.array(  # noqa: E731
        [metrics.aggregate_iqm(s[:, t]) for t in range(s.shape[-1])]
    )
    iqm_scores, iqm_cis = get_interval_estimates(train_scores, iqm, reps=BOOTSTRAP_REPS)
    plot_sample_efficiency_curve(
        grid,
        iqm_scores,
        iqm_cis,
        algorithms=["DQN"],
        xlabel="Number of Frames",
        ylabel="IQM Episode Return",
    )
    plt.legend()
    plt.tight_layout()
    out = BASE_DIR / "l2_training_curve.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"saved {out}")


def plot_final_interval_estimates(per_seed: Dict[int, pd.DataFrame]) -> None:
    final_scores = np.array(
        [per_seed[seed]["reward"].tail(FINAL_WINDOW_EPISODES).mean() for seed in SEEDS]
    )
    norm = (final_scores / CARTPOLE_MAX_RETURN).reshape(-1, 1)

    aggregate_fn = lambda s: np.array(  # noqa: E731
        [
            metrics.aggregate_median(s),
            metrics.aggregate_iqm(s),
            metrics.aggregate_mean(s),
            metrics.aggregate_optimality_gap(s, gamma=1.0),
        ]
    )
    point_estimates, interval_estimates = get_interval_estimates(
        {"DQN": norm}, aggregate_fn, reps=BOOTSTRAP_REPS
    )

    plot_interval_estimates(
        point_estimates,
        interval_estimates,
        metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
        algorithms=["DQN"],
        xlabel="Normalized CartPole-v1 Score",
        subfigure_width=3.4,
        row_height=3.0,
        xlabel_y_coordinate=-0.06,
    )
    out = BASE_DIR / "l2_interval_estimates.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"saved {out}")
    print(
        "Final-window mean returns per seed: "
        + ", ".join(f"{s}={v:.1f}" for s, v in zip(SEEDS, final_scores))
    )


def main() -> None:
    per_seed = run_all_seeds()
    grid = np.linspace(0, NUM_FRAMES, EVAL_POINTS)
    scores = build_score_matrix(per_seed, grid)
    plot_training_curve(grid, scores)
    plot_final_interval_estimates(per_seed)


if __name__ == "__main__":
    main()
