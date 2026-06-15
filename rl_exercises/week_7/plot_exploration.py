"""
Plotting functionality was implemented with the help of Claude Code (Opus-4.8).
"""

from typing import List

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

X_RANGE = (-1.2, 1.2)
Y_RANGE = (-0.3, 1.6)
N_BINS = 40


def load_snapshots(snapshot_dir: str) -> List[dict]:
    paths = sorted(glob.glob(os.path.join(snapshot_dir, "snapshot_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No snapshot_*.npz files found in {snapshot_dir}")
    snaps = []
    for p in paths:
        d = np.load(p)
        snaps.append(
            {
                "positions": d["positions"],
                "novelty": d["novelty"],
                "episode_ids": d["episode_ids"],
                "returns": d["returns"],
                "step_count": int(d["step_count"]),
            }
        )
    snaps.sort(key=lambda s: s["step_count"])
    return snaps


def plot_snapshots(snaps: List[dict], out_path: str) -> None:
    n = len(snaps)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 11), squeeze=False)

    x_edges = np.linspace(*X_RANGE, N_BINS + 1)
    y_edges = np.linspace(*Y_RANGE, N_BINS + 1)

    for col, snap in enumerate(snaps):
        pos = snap["positions"]
        nov = snap["novelty"]
        step = snap["step_count"]
        mean_ret = float(np.mean(snap["returns"]))
        x, y = pos[:, 0], pos[:, 1]

        # --- Row 1: state-visitation density ---
        ax = axes[0][col]
        h, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        # +1 so empty bins are valid under log scale.
        im = ax.imshow(
            (h.T + 1),
            origin="lower",
            extent=[*X_RANGE, *Y_RANGE],
            aspect="auto",
            cmap="viridis",
            norm=LogNorm(),
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="visits")
        ax.set_title(f"step {step}\nmean return {mean_ret:.0f}")
        if col == 0:
            ax.set_ylabel("State visitation\ny")

        ax = axes[1][col]
        sum_nov, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges], weights=nov)
        count, _, _ = np.histogram2d(x, y, bins=[x_edges, y_edges])
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_nov = np.where(count > 0, sum_nov / count, np.nan)
        im = ax.imshow(
            mean_nov.T,
            origin="lower",
            extent=[*X_RANGE, *Y_RANGE],
            aspect="auto",
            cmap="magma",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RND error")
        if col == 0:
            ax.set_ylabel("Novelty (RND error)\ny")

        ax = axes[2][col]
        ep_ids = snap["episode_ids"]
        for ep in np.unique(ep_ids):
            mask = ep_ids == ep
            ax.plot(x[mask], y[mask], lw=1.0, alpha=0.8)
            ax.scatter(x[mask][-1], y[mask][-1], s=20, c="k", zorder=3)
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.axvline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_xlim(*X_RANGE)
        ax.set_ylim(*Y_RANGE)
        ax.set_xlabel("x")
        if col == 0:
            ax.set_ylabel("Trajectories\ny")

    fig.suptitle("NovelD-PPO exploration behavior over training", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot-dir",
        required=True,
        help="Directory containing snapshot_*.npz files.",
    )
    parser.add_argument(
        "--out",
        default="exploration_snapshots.png",
        help="Output image path.",
    )
    args = parser.parse_args()

    snaps = load_snapshots(args.snapshot_dir)
    plot_snapshots(snaps, args.out)


if __name__ == "__main__":
    main()
