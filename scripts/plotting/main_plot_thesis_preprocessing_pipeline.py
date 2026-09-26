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
    fontsize: float = 9.0,
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

    fig, ax = plt.subplots(figsize=(15.5, 11.2))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.025, top=0.95)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Raw inputs.
    input_y, input_h = 0.865, 0.064
    input_specs = [
        (0.075, 0.38, "SCADA measurements\n2016–2024, six turbines"),
        (0.545, 0.38, "Operational and event records\nstatus, manual events, and curtailment"),
    ]
    for x, width, text in input_specs:
        add_box(
            ax,
            x,
            input_y,
            width,
            input_h,
            text,
            COLORS["input"],
            COLORS["input_edge"],
            fontsize=9.0,
            weight="semibold",
        )

    # Input-specific ingestion.
    ingest_y, ingest_h = 0.745, 0.082
    add_box(
        ax,
        0.055,
        ingest_y,
        0.40,
        ingest_h,
        "SCADA preparation\nSelect the required signals\nStandardise timestamps and remove duplicates",
        COLORS["shared"],
        COLORS["shared_edge"],
        fontsize=8.8,
    )
    add_box(
        ax,
        0.545,
        ingest_y,
        0.40,
        ingest_h,
        "Operational-record preparation\nClean status records and parse labelled intervals",
        COLORS["shared"],
        COLORS["shared_edge"],
        fontsize=8.8,
    )
    add_arrow(ax, (0.265, input_y), (0.255, ingest_y + ingest_h))
    add_arrow(ax, (0.735, input_y), (0.745, ingest_y + ingest_h))

    # Shared timeline construction and flagging.
    align_y, shared_h = 0.635, 0.072
    add_box(
        ax,
        0.19,
        align_y,
        0.62,
        shared_h,
        "Timestamp alignment on the 10-minute SCADA timeline\nAttach status, manual-event, extended fault-window, and curtailment flags",
        COLORS["shared"],
        COLORS["shared_edge"],
        fontsize=9.1,
    )
    add_arrow(ax, (0.255, ingest_y), (0.39, align_y + shared_h))
    add_arrow(ax, (0.745, ingest_y), (0.61, align_y + shared_h))

    flags_y = 0.525
    add_box(
        ax,
        0.19,
        flags_y,
        0.62,
        shared_h,
        "Quality and operating-context flags\nIdentify usable measurements, operating conditions, and event-related intervals",
        COLORS["shared"],
        COLORS["shared_edge"],
        fontsize=8.8,
    )
    add_arrow(ax, (0.5, align_y), (0.5, flags_y + shared_h))

    split_y = 0.425
    add_box(
        ax,
        0.24,
        split_y,
        0.52,
        0.060,
        "Chronological role assignment: 2016–2022 development  |  2023–2024 independent test",
        COLORS["shared"],
        COLORS["shared_edge"],
        fontsize=9.1,
        weight="semibold",
    )
    add_arrow(ax, (0.5, flags_y), (0.5, split_y + 0.060))

    # Development branch.
    left_x, right_x = 0.025, 0.525
    branch_w = 0.45
    branch_header_y, branch_header_h = 0.342, 0.052
    add_box(
        ax,
        left_x,
        branch_header_y,
        branch_w,
        branch_header_h,
        "Development branch: selected healthy operation",
        COLORS["development"],
        COLORS["development_edge"],
        fontsize=9.3,
        weight="semibold",
    )
    add_box(
        ax,
        right_x,
        branch_header_y,
        branch_w,
        branch_header_h,
        "Test branch: complete flagged timeline",
        COLORS["test"],
        COLORS["test_edge"],
        fontsize=9.3,
        weight="semibold",
    )
    add_arrow(ax, (0.45, split_y), (0.25, branch_header_y + branch_header_h))
    add_arrow(ax, (0.55, split_y), (0.75, branch_header_y + branch_header_h))

    dev_boxes = [
        (
            0.260,
            "Healthy-operation selection\nApply data-quality, physical-validity, and operating-context criteria",
        ),
        (
            0.165,
            "Continuous healthy-segment construction\nLinearly interpolate internal gaps within each turbine + segment only\nRemove values that remain missing after interpolation",
        ),
        (
            0.070,
            "Pooled chronological train/validation split (approximately 80:20)\nGenerate 12-step sequences: 8 inputs → 3 next-step targets",
        ),
    ]
    test_boxes = [
        (
            0.260,
            "Retain the complete 2023–2024 flagged timeline\nAbnormal and non-operational periods remain available",
        ),
        (
            0.165,
            "Construct continuous prediction segments\nNever construct a sequence across a time gap or turbine boundary",
        ),
        (
            0.070,
            "Generate the same 12-step input sequences\nPreserve timestamps and flags for later alarm evaluation",
        ),
    ]
    box_h = 0.075
    for x, items, face, edge in [
        (left_x, dev_boxes, COLORS["development"], COLORS["development_edge"]),
        (right_x, test_boxes, COLORS["test"], COLORS["test_edge"]),
    ]:
        for y, text in items:
            add_box(ax, x, y, branch_w, box_h, text, face, edge, fontsize=8.25)
        add_arrow(ax, (x + branch_w / 2, branch_header_y), (x + branch_w / 2, 0.260 + box_h))
        add_arrow(ax, (x + branch_w / 2, 0.260), (x + branch_w / 2, 0.165 + box_h))
        add_arrow(ax, (x + branch_w / 2, 0.165), (x + branch_w / 2, 0.070 + box_h))

    # Final model-ready outputs.
    output_y, output_h = 0.012, 0.045
    add_box(
        ax,
        left_x + 0.045,
        output_y,
        branch_w - 0.09,
        output_h,
        "GRU-ready training and validation datasets",
        COLORS["output"],
        COLORS["output_edge"],
        fontsize=8.8,
        weight="semibold",
    )
    add_box(
        ax,
        right_x + 0.045,
        output_y,
        branch_w - 0.09,
        output_h,
        "Independent test sequences with evaluation context",
        COLORS["output"],
        COLORS["output_edge"],
        fontsize=8.8,
        weight="semibold",
    )
    add_arrow(ax, (left_x + branch_w / 2, 0.070), (left_x + branch_w / 2, output_y + output_h))
    add_arrow(ax, (right_x + branch_w / 2, 0.070), (right_x + branch_w / 2, output_y + output_h))

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
