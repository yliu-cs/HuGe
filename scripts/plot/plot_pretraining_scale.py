#!/usr/bin/env python3
"""Plot HumanoidArena performance against the pre-training data scale."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCALE_LABELS = ("From\nScratch", "10%", "100%")
# Slightly expand the 0%-10% interval for readability without making all categories equidistant.
PLOT_POSITIONS = (0, 20, 100)
# The middle values are pseudo data (linear estimates at 10%); replace them after evaluation.
RESULTS = {
    "HuGe-OFT": (55.00, 54.91, 54.05),
    "HuGe-GR00T": (58.33, 59.16, 66.67),
    "HuGe-Pi": (60.24, 60.72, 65.00),
}
COLORS = {
    "HuGe-OFT": "#2B6A99",
    "HuGe-GR00T": "#1B7C3D",
    "HuGe-Pi": "#F16C23",
}
STYLES = {
    "HuGe-OFT": ("o", "-"),
    "HuGe-GR00T": ("s", "-."),
    "HuGe-Pi": ("v", "--"),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2] / ".runs" / "figure" / "pretraining_scale.png",
        help="output PNG; a same-named PDF is also saved (default: .runs/figure/pretraining_scale.png)",
    )
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rc("font", family="Times New Roman", size=14)

    fig, ax = plt.subplots()
    for model, scores in RESULTS.items():
        color = COLORS[model]
        marker, linestyle = STYLES[model]
        ax.plot(PLOT_POSITIONS, scores, color=color, marker=marker, linestyle=linestyle, label=model)
        ax.scatter(PLOT_POSITIONS[1], scores[1], marker=marker, facecolor="white", edgecolor=color, zorder=3)

    ax.set_xlabel("Pre-training Data Scale", labelpad=8)
    ax.set_ylabel("HumanoidArena Overall Average", labelpad=8)
    ax.set_xticks(PLOT_POSITIONS, SCALE_LABELS)
    ax.set_ylim(50, 70)
    ax.set_yticks((50, 55, 60, 65, 70))
    ax.grid(axis="y", linestyle=(0, (5, 10)), linewidth=0.25, color="#4E616C", zorder=0)
    ax.spines["top"].set_color("none")
    ax.legend(loc="upper left", prop={"size": 10})
    ax.text(0.98, 0.04, "Hollow markers: pseudo data", transform=ax.transAxes, ha="right", fontsize=10, color="#666666")

    fig.tight_layout()
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}, {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
