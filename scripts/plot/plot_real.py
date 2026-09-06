#!/usr/bin/env python3
"""Save three separate real-robot evaluation figures as PNG and PDF.

Fill STAGE_RESULTS with (total, approached, picked_up, crouched, thrown_in).
Each count is the number of trials that completed that stage and every
preceding stage. Counts must be nonincreasing, e.g. (20, 19, 17, 15, 13).
Each failed trial exits once, at its first failed stage. Stage 4 supplies
the Real-only baseline for the other figures; do not enter it twice.

Fill mixed-data RESULTS with (successful trials, total trials), e.g. (17, 20).
None means unmeasured, not zero success. Use the same evaluation protocol
and checkpoint-selection rule across settings. Ablation error bars are 95%
Wilson intervals over trials, not variation across training seeds.

data_scale combines the Ego and Real sweeps on paired categorical ticks:
11/50, 55/200, 110/400 (Real/Ego counts). Ego varies with Real fixed at 110;
Real varies with Ego Syn fixed at 400. A tick does not name a joint training
configuration. Both curves share the single R110_E400_syn endpoint.

The architecture figure follows the success/failure flow in TWIST2 Fig. 14:
https://arxiv.org/html/2511.02832v1#S4.F14
Font, palette, hatching and grid style follow CVLA/plot/theme.py and like.py.

Run without --architecture for the architecture comparison and pending
ablation figures. After selecting an architecture, pass --architecture
qwenpi_v3 or --architecture qwengr00t to draw its two ablation figures.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from statistics import NormalDist

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as PlotPath


Result = tuple[int, int] | None
StageResult = tuple[int, int, int, int, int] | None

# Total trials, then cumulative successes: approach, pick up, crouch, throw in.
# Trials use no reset or intervention between stages; later stages require all
# earlier stages to pass. Record a trial under its first failed stage only.
STAGE_RESULTS: dict[str, StageResult] = {"qwenpi_v3": None, "qwengr00t": None}
STAGE_LABELS = ("1. Approach the table", "2. Pick up the cola", "3. Crouch toward the bin", "4. Throw into the bin")

# Fill the six mixtures for the chosen model. Reuse each experiment's result
# across figures; Pure Real comes from STAGE_RESULTS, above.
RESULTS: dict[str, dict[str, Result]] = {
    "qwenpi_v3": {
        "R110_E400_raw": None,
        "R110_E400_syn": None,
        "R110_E50_syn": None,
        "R110_E200_syn": None,
        "R55_E400_syn": None,
        "R11_E400_syn": None,
    },
    "qwengr00t": {
        "R110_E400_raw": None,
        "R110_E400_syn": None,
        "R110_E50_syn": None,
        "R110_E200_syn": None,
        "R55_E400_syn": None,
        "R11_E400_syn": None,
    },
}
MODEL_LABELS = {"qwenpi_v3": "HuGe-Pi", "qwengr00t": "HuGe-GR00T"}
# CVLA/plot/Theme.md: orange / blue / green three-color theme.
PALETTE = ("#F16C23", "#2B6A99", "#1B7C3D")
COLORS = {"qwenpi_v3": PALETTE[0], "qwengr00t": PALETTE[2]}
BAR_BACKGROUND = "#F9F8F4"
HATCHES = ("xxx", "///", "\\\\\\")
REFERENCE_COLOR = "#4E616C"


def _interval(result: Result) -> tuple[float, float, float] | None:
    """Return success percentage and the lower/upper Wilson bounds."""
    if result is None:
        return None
    successes, trials = result
    if type(successes) is not int or type(trials) is not int or not 0 <= successes <= trials or trials <= 0:
        raise ValueError(f"expected integer counts 0 <= successes <= trials, trials > 0; got {result!r}")
    probability = successes / trials
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / trials
    center = (probability + z**2 / (2 * trials)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / trials + z**2 / (4 * trials**2)) / denominator
    return 100 * probability, 100 * max(0.0, center - radius), 100 * min(1.0, center + radius)


def _new_figure(xlabel: str) -> tuple[Figure, Axes]:
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlabel(xlabel, labelpad=8)
    ax.set_ylabel("Success rate (%)", labelpad=8)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", linestyle=(0, (5, 10)), linewidth=0.25, color=REFERENCE_COLOR, zorder=0)
    ax.spines["top"].set_color("none")
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


def _plot_stage_flow(ax: Axes, architecture: str) -> None:
    counts = STAGE_RESULTS[architecture]
    color = COLORS[architecture]
    # Rising node positions curve both edges without changing trial counts.
    tops = tuple(1 + 0.22 * stage for stage in range(5))
    ax.set_xlim(-0.65, 4.55)
    ax.set_ylim(-0.85, 2.45)
    ax.axis("off")
    ax.text(-0.48, 0.5, MODEL_LABELS[architecture], rotation=90, ha="center", va="center",
            color=color, fontsize=12, fontweight="bold")
    labels = ("Total trials", *STAGE_LABELS)
    for column, label in enumerate(labels):
        count_text = "Pending" if counts is None else (
            str(counts[0]) if column == 0 else f"{counts[column]}/{counts[0]} ({100 * counts[column] / counts[0]:.0f}%)"
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
        ax.text(stage, failure_end - 0.1, f"Failed at stage {stage}\n{failed}/{total} ({100 * failed / total:.0f}%)",
                ha="center", va="top", fontsize=10, color=REFERENCE_COLOR)


def _plot_measurement(ax: Axes, position: float, result: Result, *,
                      color: str = REFERENCE_COLOR, text_offset: float = 0) -> None:
    alignment = "right" if text_offset < 0 else "left" if text_offset > 0 else "center"
    interval = _interval(result)
    if interval is None:
        ax.annotate("Pending", xy=(position, 0.28), xycoords=ax.get_xaxis_transform(),
                    xytext=(text_offset, 0), textcoords="offset points", ha=alignment,
                    color=color, fontsize=10)
        return
    rate, lower, upper = interval
    ax.errorbar(position, rate, yerr=[[max(0.0, rate - lower)], [max(0.0, upper - rate)]],
                fmt="none", ecolor=color, capsize=3, linewidth=1, zorder=4)
    ax.annotate(f"{rate:.1f}% ({result[0]}/{result[1]})", xy=(position, upper),
                xytext=(text_offset, 6), textcoords="offset points", ha=alignment, va="bottom", fontsize=10)


def _plot_bars(ax: Axes, results: list[Result]) -> None:
    rates = [100 * result[0] / result[1] if result is not None else math.nan for result in results]
    ax.set_xlim(-0.6, 2.6)
    rectangles = ax.bar((0, 1, 2), rates, color=BAR_BACKGROUND, edgecolor=PALETTE, width=0.55, zorder=3)
    for position, (rectangle, hatch, result) in enumerate(zip(rectangles, HATCHES, results, strict=True)):
        rectangle.set_hatch(hatch)
        _plot_measurement(ax, position, result)


def _save(fig: Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout(h_pad=0.5)
    output = output_dir / f"{name}.png"
    fig.savefig(output, dpi=600, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {output}, {output.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--architecture", choices=tuple(MODEL_LABELS),
                        help="chosen model for the three ablations; never selected automatically from scores")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[2] / ".runs" / "figure" / "real",
                        help="directory for three PNG/PDF pairs (default: .runs/figure/real)")
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()

    # Validate entered measurements before saving any figures.
    for architecture, counts in STAGE_RESULTS.items():
        if counts is not None and (
            len(counts) != 5 or any(type(count) is not int or count < 0 for count in counts)
            or counts[0] <= 0 or any(later > earlier for earlier, later in zip(counts, counts[1:]))
        ):
            raise ValueError(f"{architecture}: expected total >= approach >= pick up >= crouch >= throw in >= 0, total > 0")
    for measurements in RESULTS.values():
        for result in measurements.values():
            _interval(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.rc("font", family="Times New Roman", size=14)
    plt.rcParams["pdf.fonttype"] = 42

    fig, axes = plt.subplots(2, 1, figsize=(8.8, 5.2))
    for ax, architecture in zip(axes, MODEL_LABELS, strict=True):
        _plot_stage_flow(ax, architecture)
    _save(fig, output_dir, "architecture_comparison")

    selected = RESULTS[args.architecture] if args.architecture else {}
    stages = STAGE_RESULTS[args.architecture] if args.architecture else None
    baseline = (stages[-1], stages[0]) if stages is not None else None

    fig, ax = _new_figure("Training data (R110)")
    _plot_bars(ax, [baseline, selected.get("R110_E400_raw"), selected.get("R110_E400_syn")])
    ax.set_xticks((0, 1, 2), ("Real only\nE0", "Real + Raw\nE400", "Real + Syn\nE400"))
    _save(fig, output_dir, "raw_vs_syn")

    scale_specs = (
        ("Ego scale (Real=110)", ("R110_E50_syn", "R110_E200_syn", "R110_E400_syn"),
         PALETTE[0], "-", "o", -0.04),
        ("Real scale (Ego=400)", ("R11_E400_syn", "R55_E400_syn", "R110_E400_syn"),
         PALETTE[1], "--", "s", 0.04),
    )
    # Retain every confidence interval and the baseline in the merged y range.
    lower_bounds = [interval[1] for spec in scale_specs for experiment in spec[1]
                    if (interval := _interval(selected.get(experiment))) is not None]
    baseline_interval = _interval(baseline)
    if lower_bounds and baseline_interval is not None:
        lower_bounds.append(baseline_interval[0])
    ymin = max(0, 10 * math.floor((min(lower_bounds) - 5) / 10)) if lower_bounds else 0
    fig, ax = _new_figure("Real / Ego demonstrations")
    ax.set_ylim(ymin, 100)
    ax.set_yticks(range(ymin, 101, 10 if ymin >= 30 else 20))
    ax.set_xlim(-0.6, 2.6)
    ax.set_xticks((0, 1, 2), ("11/50", "55/200", "110/400"))
    for label, experiments, color, linestyle, marker, offset in scale_specs:
        results = [selected.get(experiment) for experiment in experiments]
        # Small offsets separate the two intervals within each categorical tick.
        # Both curves meet at the unchanged shared endpoint; NaN breaks missing data.
        positions = (offset, 1 + offset, 2)
        rates = [100 * result[0] / result[1] if result is not None else math.nan for result in results]
        ax.plot(positions, rates, color=color, linestyle=linestyle, marker=marker,
                markersize=6 if marker == "o" else 8, markerfacecolor=color if marker == "o" else "none",
                linewidth=1.5, label=label, zorder=3)
        for position, result in zip(positions[:2], results[:2], strict=True):
            _plot_measurement(ax, position, result, color=color, text_offset=-6 if offset < 0 else 6)
    _plot_measurement(ax, 2, selected.get("R110_E400_syn"))
    if baseline_interval is not None:
        ax.axhline(baseline_interval[0], color=REFERENCE_COLOR, linestyle=":", linewidth=1,
                   label="Real only (R110)")
        ax.text(0.98, baseline_interval[0], f"{baseline_interval[0]:.1f}%",
                transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=10,
                color=REFERENCE_COLOR, bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    ax.legend(loc="lower right", prop={"size": 10}, ncol=1, labelspacing=0.25, borderpad=0.3)
    _save(fig, output_dir, "data_scale")

    if args.architecture is None:
        print("Pass --architecture qwenpi_v3 or qwengr00t after choosing the model for the ablations.")
    print("Unmeasured RESULTS entries are marked Pending; no success rates are fabricated.")


if __name__ == "__main__":
    main()
