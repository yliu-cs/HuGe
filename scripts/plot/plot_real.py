#!/usr/bin/env python3
"""Save two separate HuGe-Pi real-robot evaluation figures as PNG and PDF.

stage_success shows HuGe-Pi's R110 + E400 Syn stage outcomes. raw_vs_syn
compares Real only, Real + Raw, and Real + Syn as three bars.

Fill STAGE_RESULTS with (total, approached, picked_up, crouched, thrown_in).
Each count is the number of trials that completed that stage and every
preceding stage. Counts must be nonincreasing, e.g. (20, 19, 17, 15, 13).
Each failed trial exits once, at its first failed stage. Stage 4 supplies
the Real + Syn bar in the Raw/Syn comparison; do not enter it twice.

Fill RESULTS for Real only and Real + Raw with (successful trials, total
trials), e.g. (17, 20).
None means unmeasured, not zero success. Use the same evaluation protocol
and checkpoint-selection rule across settings. Bar labels show the success
percentage and successful/total trials.

All three settings use 110 Real demonstrations. Raw and Syn each add 400
Ego demonstrations. The x-axis represents categories, not a numeric scale.

The stage figure follows the success/failure flow in TWIST2 Fig. 14:
https://arxiv.org/html/2511.02832v1#S4.F14
Font, palette, hatching and grid style follow CVLA/plot/theme.py and like.py.

"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as PlotPath
from matplotlib.transforms import Affine2D


Result = tuple[int, int] | None
StageResult = tuple[int, int, int, int, int] | None

# R110 + E400 Syn: total trials, then cumulative successes for the four stages.
# Trials use no reset or intervention between stages; later stages require all
# earlier stages to pass. Record a trial under its first failed stage only.
STAGE_RESULTS: StageResult = (30, 29, 24, 21, 20)
STAGE_LABELS = ("1. Approach table", "2. Pick up cola", "3. Crouch to bin", "4. Throw into bin")

# Fill Pure Real and Real + Raw. Real + Syn comes from STAGE_RESULTS, above.
RESULTS: dict[str, Result] = {
    "R110_E0_none": (13, 30),
    "R110_E400_raw": (18, 30),
}
# CVLA/plot/Theme.md: orange / blue / green three-color theme.
PALETTE = ("#F16C23", "#2B6A99", "#1B7C3D")
BAR_BACKGROUND = "#F9F8F4"
HATCHES = ("xxx", "///", "\\\\\\")
REFERENCE_COLOR = "#4E616C"


def _new_figure() -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle=(0, (5, 10)), linewidth=0.25, color=REFERENCE_COLOR, zorder=0)
    ax.spines["top"].set_color("none")
    ax.spines["right"].set_visible(False)
    return fig, ax


def _ribbon(ax: Axes, x0: float, x1: float, y0: float, y1: float, height: float, color: str) -> None:
    """One constant-width band between two stage nodes."""
    if height == 0:
        return
    middle = (x0 + x1) / 2
    vertices = [(x0, y0), (middle, y0), (middle, y1), (x1, y1),
                (x1, y1 + height), (middle, y1 + height), (middle, y0 + height),
                (x0, y0 + height), (x0, y0)]
    codes = [PlotPath.MOVETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4,
             PlotPath.LINETO, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CURVE4, PlotPath.CLOSEPOLY]
    ax.add_patch(PathPatch(PlotPath(vertices, codes), facecolor=color, edgecolor="none", alpha=0.35))


def _plot_stage_flow(ax: Axes) -> None:
    counts = STAGE_RESULTS
    color = PALETTE[0]
    # Rising node positions curve both edges without changing trial counts.
    tops = tuple(1 + 0.22 * stage for stage in range(5))
    ax.set_xlim(-0.65, 4.55)
    ax.set_ylim(-0.85, 2.45)
    ax.axis("off")
    labels = ("Total trials", *STAGE_LABELS)
    for column, label in enumerate(labels):
        count_text = "Pending" if counts is None else (
            str(counts[0]) if column == 0 else f"{counts[column]}/{counts[0]} ({100 * counts[column] / counts[0]:.1f}%)"
        )
        ax.text(column, tops[column] + 0.08, f"{label}\n{count_text}", ha="center", va="bottom", fontsize=11)
    if counts is None:
        ax.text(2, 0.3, "Stage outcomes pending", color=REFERENCE_COLOR, ha="center", fontsize=12)
        return

    total = counts[0]
    node_width = 0.08
    ax.add_patch(Rectangle((-node_width / 2, 0), node_width, 1, facecolor=color))
    for stage in range(1, 5):
        previous, passed = counts[stage - 1], counts[stage]
        failed = previous - passed
        success_height, failure_height = passed / total, failed / total
        left, right = stage - 1 + node_width / 2, stage - node_width / 2
        failure_start = tops[stage - 1] - previous / total
        # Equal downward bends keep branch lengths consistent; endpoints follow
        # their source nodes instead of sharing an artificial horizontal floor.
        failure_end = failure_start - 0.3
        _ribbon(ax, left, right, tops[stage - 1] - success_height, tops[stage] - success_height,
                success_height, color)
        _ribbon(ax, left, right, failure_start, failure_end, failure_height, REFERENCE_COLOR)
        ax.add_patch(Rectangle((right, tops[stage] - success_height), node_width, success_height, facecolor=color))
        ax.add_patch(Rectangle((right, failure_end), node_width, failure_height, facecolor=BAR_BACKGROUND,
                               edgecolor=REFERENCE_COLOR, linewidth=0.8, hatch="///"))
        ax.text(stage, failure_end - 0.1, f"Failed at stage {stage}\n{failed}/{total} ({100 * failed / total:.1f}%)",
                ha="center", va="top", fontsize=10, color=REFERENCE_COLOR)


def _plot_measurement(ax: Axes, position: float, result: Result, *,
                      color: str = REFERENCE_COLOR) -> None:
    if result is None:
        ax.annotate("Pending", xy=(position, 0.28), xycoords=ax.get_xaxis_transform(),
                    ha="center",
                    color=color, fontsize=10)
        return
    rate = 100 * result[0] / result[1]
    ax.annotate(f"{rate:.1f}% ({result[0]}/{result[1]})", xy=(position, rate),
                xytext=(0, 6), textcoords="offset points", ha="center", va="bottom", fontsize=10)


def _expand_hatches(fig: Figure) -> None:
    """Replace diagonal hatches with vector lines to avoid PDF tiling patterns."""
    for ax in fig.axes:
        for patch in ax.patches:
            hatch = patch.get_hatch()
            if not hatch:
                continue
            patch.set_hatch(None)
            bounds = patch.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
            width, height = bounds.width, bounds.height
            if not math.isfinite(width + height) or width <= 0 or height <= 0:
                continue  # Pending bars and zero-count failure bands have no fill.
            segments = []
            for slope, symbol in ((1, "/"), (-1, "\\")):
                count = hatch.count(symbol) + hatch.count("x") + hatch.count("X")
                if not count:
                    continue
                # Matplotlib's diagonal hatches use density 6 in a one-inch tile.
                spacing = 2 / (6 * count)
                low, high = (-width, height) if slope == 1 else (0, height + width)
                for index in range(math.floor(low / spacing), math.ceil(high / spacing) + 1):
                    offset = index * spacing
                    segments.append(((0, offset), (width, offset + slope * width)))
            # Anchor physical spacing to the patch so DPI changes and tight-bbox
            # export move/scale the lines together with their clipping rectangle.
            transform = Affine2D().scale(1 / width, 1 / height) + patch.get_transform()
            lines = LineCollection(segments, colors=[patch.get_edgecolor()],
                                   linewidths=patch.get_hatch_linewidth(), transform=transform,
                                   capstyle="butt", zorder=patch.get_zorder() + 0.01)
            ax.add_collection(lines, autolim=False)
            lines.set_clip_path(patch)
            lines.set_in_layout(False)


def _save(fig: Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout(h_pad=0.5)
    _expand_hatches(fig)
    output = output_dir / f"{name}.png"
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}, {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / ".runs" / "figure" / "real",
                        help="directory for two PNG/PDF pairs (default: .runs/figure/real)")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    # Validate entered measurements before saving any figures.
    counts = STAGE_RESULTS
    if counts is not None and (
        len(counts) != 5 or any(type(count) is not int or count < 0 for count in counts)
        or counts[0] <= 0 or any(later > earlier for earlier, later in zip(counts, counts[1:]))
    ):
        raise ValueError("expected total >= approach >= pick up >= crouch >= throw in >= 0, total > 0")
    for result in RESULTS.values():
        if result is not None:
            successes, trials = result
            if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials <= 0:
                raise ValueError(f"expected integer counts 0 <= successes <= trials, trials > 0; got {result!r}")
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rc("font", family="Times New Roman", size=14)
    plt.rcParams["pdf.fonttype"] = 42

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    _plot_stage_flow(ax)
    _save(fig, output_dir, "stage_success")

    syn_result = (counts[-1], counts[0]) if counts is not None else None

    results = [RESULTS["R110_E0_none"], RESULTS["R110_E400_raw"], syn_result]
    fig, ax = _new_figure()
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks((0, 1, 2), ("Real only\nR110/E0", "Real + Raw\nR110/E400", "Real + Syn\nR110/E400"))
    rates = [100 * result[0] / result[1] if result is not None else math.nan for result in results]
    bars = ax.bar((0, 1, 2), rates, color=BAR_BACKGROUND, edgecolor=PALETTE, width=0.55, zorder=3)
    for position, (bar, hatch, result) in enumerate(zip(bars, HATCHES, results, strict=True)):
        bar.set_hatch(hatch)
        _plot_measurement(ax, position, result)
    _save(fig, output_dir, "raw_vs_syn")

    print("Unmeasured RESULTS entries are marked Pending; no success rates are fabricated.")


if __name__ == "__main__":
    main()
