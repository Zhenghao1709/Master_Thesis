from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


COLORS = {
    "input": "#DCE6F1",
    "input_edge": "#4472A8",
    "shared": "#ECEFF1",
    "shared_edge": "#65737E",
    "development": "#E1F0E4",
    "development_edge": "#3E8E5B",
    "test": "#FBE5D6",
    "test_edge": "#D06B32",
    "output": "#E8E2F2",
    "output_edge": "#7560A8",
    "text": "#222222",
    "arrow": "#5A5A5A",
}


def add_box(
    ax,
    x: float,
    y: float,
    width: float,
    height: float,
    text: str,
    facecolor: str,
    edgecolor: str,
    fontsize: float = 11.5,
    weight: str = "normal",
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.008",
        linewidth=1.25,
        facecolor=facecolor,
        edgecolor=edgecolor,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["text"],
        linespacing=1.22,
    )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    connectionstyle: str = "arc3",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=COLORS["arrow"],
            connectionstyle=connectionstyle,
            shrinkA=2,
            shrinkB=2,
        )
    )


def main() -> None:
    project_root = PROJECT_ROOT
    output_dir = project_root / "results" / "kelmarsh" / "figures" / "thesis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Keep the canvas close to the final thesis text width so labels remain
    # readable when the vector PDF is inserted at 0.98\textwidth.
    fig, ax = plt.subplots(figsize=(7.1, 5.7))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x, right_x, column_w = 0.035, 0.515, 0.45
    add_box(
        ax, left_x, 0.80, column_w, 0.13,
        "SCADA measurements\nsix turbines, 2016–2024",
        COLORS["input"], COLORS["input_edge"], weight="semibold",
    )
    add_box(
        ax, right_x, 0.80, column_w, 0.13,
        "Context records\nstatus, reviewed events\nand curtailment",
        COLORS["input"], COLORS["input_edge"], fontsize=10.7, weight="semibold",
    )
    add_box(
        ax, 0.14, 0.58, 0.72, 0.14,
        "Align on each turbine's 10-minute timeline\ncheck signal quality; attach operating and event flags",
        COLORS["shared"], COLORS["shared_edge"], fontsize=11.3,
    )
    add_arrow(ax, (0.26, 0.80), (0.40, 0.72))
    add_arrow(ax, (0.74, 0.80), (0.60, 0.72))

    add_box(
        ax, left_x, 0.31, column_w, 0.18,
        "Development: 2016–2022\nselect healthy continuous segments\nsplit into training and validation",
        COLORS["development"], COLORS["development_edge"], fontsize=11.1,
    )
    add_box(
        ax, right_x, 0.31, column_w, 0.18,
        "Test: 2023–2024\nretain the flagged timeline,\nincluding event-related periods",
        COLORS["test"], COLORS["test_edge"], fontsize=11.1,
    )
    add_arrow(ax, (0.43, 0.58), (0.26, 0.49))
    add_arrow(ax, (0.57, 0.58), (0.74, 0.49))

    add_box(
        ax, left_x, 0.06, column_w, 0.16,
        "Healthy train / validation sequences\n12-step inputs → next-step targets",
        COLORS["output"], COLORS["output_edge"], fontsize=11.0,
    )
    add_box(
        ax, right_x, 0.06, column_w, 0.16,
        "Independent test sequences\nflags retained for alarm evaluation",
        COLORS["output"], COLORS["output_edge"], fontsize=11.0,
    )
    add_arrow(ax, (0.26, 0.31), (0.26, 0.22))
    add_arrow(ax, (0.74, 0.31), (0.74, 0.22))

    stem = output_dir / "kelmarsh_preprocessing_dataset_pipeline"
    output_specs = {
        ".png": {"dpi": 300},
        ".pdf": {},
        ".svg": {},
    }
    saved_paths = []
    for suffix, extra_options in output_specs.items():
        output_path = stem.with_suffix(suffix)
        try:
            fig.savefig(
                output_path,
                bbox_inches="tight",
                facecolor="white",
                **extra_options,
            )
            saved_paths.append(output_path)
        except PermissionError:
            print(f"Skipped locked output: {output_path}")
    plt.close(fig)

    for output_path in saved_paths:
        print(f"{output_path.suffix[1:].upper()}: {output_path}")


if __name__ == "__main__":
    main()
