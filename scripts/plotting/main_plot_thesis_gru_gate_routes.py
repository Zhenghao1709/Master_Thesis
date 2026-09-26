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
    "h_t": ((4.15, 10.80), r"$h_t$"),
    "candidate": ((4.15, 8.45), r"$\tilde{h}_t$"),
    "reset": ((4.25, 5.65), r"$r_t$"),
    "update": ((7.25, 5.65), r"$z_t$"),
    "previous": ((4.25, 2.95), r"$h_{t-1}$"),
    "input": ((7.25, 2.95), r"$x_t$"),
}


def straight(ax, start, end):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=14,
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
    fig, ax = plt.subplots(figsize=(6.3, 7.4))
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.035, top=0.98)
    ax.set_xlim(0.55, 9.45)
    ax.set_ylim(1.45, 12.20)
    ax.set_aspect("equal")
    ax.axis("off")

    # Keep only the direct dependencies in the GRU update equations.
    curve(ax, [(3.90, 2.70), (0.90, 4.10), (1.30, 8.00), (3.78, 10.54)])
    curve(ax, [(3.92, 2.72), (2.15, 4.05), (2.10, 6.80), (3.77, 8.22)])
    straight(ax, (4.25, 3.38), (4.25, 5.19))
    straight(ax, (4.58, 3.22), (6.91, 5.38))
    straight(ax, (6.92, 3.25), (4.60, 5.40))
    straight(ax, (7.25, 3.38), (7.25, 5.19))
    straight(ax, (6.97, 3.28), (4.48, 8.17))
    straight(ax, (4.25, 6.09), (4.17, 8.01))
    straight(ax, (4.15, 8.89), (4.15, 10.36))
    straight(ax, (6.99, 5.99), (4.48, 10.54))

    for key, ((x, y), label) in NODES.items():
        ax.add_patch(Circle((x, y), 0.45, facecolor=NODE_FILL,
                            edgecolor=NODE_EDGE, linewidth=1.7, zorder=3))
        ax.text(x, y, label, ha="center", va="center",
                fontsize=16 if key == "previous" else 18,
                color=INK, zorder=4)

    ax.text(4.15, 11.65, r"$\vdots$", ha="center", va="center",
            fontsize=19, color=INK)
    ax.text(4.25, 1.95, r"$\vdots$", ha="center", va="center",
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
