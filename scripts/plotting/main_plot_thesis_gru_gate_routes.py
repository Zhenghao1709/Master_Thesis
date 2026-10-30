from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "thesis"

NODE_EDGE = "#286B97"
NODE_FILL = "#F7FBFD"
ARROW = "#49606F"
INK = "#24323B"

NODES = {
    "previous": ((1.00, 2.25), r"$h_{t-1}$"),
    "input": ((1.00, 0.75), r"$x_t$"),
    "reset": ((3.60, 2.25), r"$r_t$"),
    "update": ((3.60, 0.75), r"$z_t$"),
    "candidate": ((6.40, 2.25), r"$\tilde{h}_t$"),
    "h_t": ((9.20, 2.25), r"$h_t$"),
}


def straight(ax, source: str, target: str) -> None:
    start = NODES[source][0]
    end = NODES[target][0]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = (dx * dx + dy * dy) ** 0.5
    radius = 0.45
    start_edge = (start[0] + radius * dx / length,
                  start[1] + radius * dy / length)
    end_edge = (end[0] - radius * dx / length,
                end[1] - radius * dy / length)
    ax.add_patch(FancyArrowPatch(
        start_edge, end_edge, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=ARROW, shrinkA=0, shrinkB=0,
        joinstyle="round", zorder=1,
    ))


def curve(ax, points):
    path = MplPath(points, [MplPath.MOVETO] + [MplPath.CURVE4] * 3)
    ax.add_patch(FancyArrowPatch(
        path=path, arrowstyle="-|>", mutation_scale=14,
        linewidth=1.6, color=ARROW, shrinkA=0, shrinkB=0,
        joinstyle="round", zorder=1,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    fig.subplots_adjust(left=0.025, right=0.975, bottom=0.035, top=0.975)
    ax.set_xlim(-0.25, 10.45)
    ax.set_ylim(0.05, 5.00)
    ax.set_aspect("equal")
    ax.axis("off")

    # The two bypass paths remain curved, now above the horizontal chain.
    curve(ax, [(0.82, 2.66), (1.15, 5.35), (8.45, 5.35), (9.00, 2.64)])
    curve(ax, [(1.18, 2.66), (1.65, 4.20), (5.65, 4.20), (6.16, 2.63)])
    for source, target in (
        ("previous", "reset"),
        ("previous", "update"),
        ("input", "reset"),
        ("input", "update"),
        ("input", "candidate"),
        ("reset", "candidate"),
        ("update", "h_t"),
        ("candidate", "h_t"),
    ):
        straight(ax, source, target)

    for key, ((x, y), label) in NODES.items():
        ax.add_patch(Circle((x, y), 0.45, facecolor=NODE_FILL,
                            edgecolor=NODE_EDGE, linewidth=1.7, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=16 if key == "previous" else 18,
                color=INK, zorder=4)

    ax.text(0.12, 2.25, r"$\cdots$", ha="center", va="center",
            fontsize=19, color=INK)
    ax.text(10.10, 2.25, r"$\cdots$", ha="center", va="center",
            fontsize=19, color=INK)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "gru_gate_update_schematic"
    for suffix, options in {".png": {"dpi": 300}, ".pdf": {}, ".svg": {}}.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, facecolor="white", bbox_inches="tight", **options)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
