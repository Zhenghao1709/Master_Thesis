from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "results" / "kelmarsh" / "figures" / "thesis"

INK = "#26323B"
MUTED = "#53636D"
BLUE = "#316B94"
BLUE_FILL = "#EAF2F8"
TEAL = "#247B70"
TEAL_FILL = "#E6F3EF"
ORANGE = "#BA6841"
ORANGE_FILL = "#FCF0E8"


def box(ax, center_y, label, *, edge=TEAL, fill=TEAL_FILL, width=0.50):
    x = 0.25
    height = 0.085
    ax.add_patch(FancyBboxPatch(
        (x, center_y - height / 2), width, height,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.35, edgecolor=edge, facecolor=fill,
    ))
    ax.text(x + width / 2, center_y, label, ha="center", va="center",
            fontsize=10.3, color=INK, linespacing=1.3)


def arrow(ax, start, end, *, color=MUTED):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=13,
        linewidth=1.45, color=color, shrinkA=2, shrinkB=2,
    ))


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=0.015, top=0.985)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.18, 0.99)
    ax.axis("off")

    rows = [0.915, 0.785, 0.655, 0.525, 0.395, 0.265]
    box(ax, rows[0], r"Training input  $\mathbf{X}_{t}\in\mathbb{R}^{12\times 8}$",
        edge=BLUE, fill=BLUE_FILL)
    box(ax, rows[1], "GRU layer\ninput size = 8, hidden size = 64, layers = 1")
    box(ax, rows[2], r"Last hidden state  $\mathbf{h}_{t}\in\mathbb{R}^{64}$")
    box(ax, rows[3],
        r"Linear (64, 3)  $\hat{\mathbf{y}}_{t+1}=W_{o}\mathbf{h}_{t}+\mathbf{b}_{o}$")
    box(ax, rows[4], r"Output  $\hat{\mathbf{y}}_{t+1}\in\mathbb{R}^{3}$",
        edge=ORANGE, fill=ORANGE_FILL)
    box(ax, rows[5], "MSE loss", edge=ORANGE, fill=ORANGE_FILL)

    for upper, lower in zip(rows, rows[1:]):
        arrow(ax, (0.50, upper - 0.047), (0.50, lower + 0.047))

    ax.add_patch(FancyBboxPatch(
        (0.80, rows[5] - 0.0425), 0.19, 0.085,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.35, edgecolor=ORANGE, facecolor=ORANGE_FILL,
    ))
    ax.text(0.895, rows[5], "True target\n" +
            r"$\mathbf{y}_{t+1}\in\mathbb{R}^{3}$", ha="center", va="center",
            fontsize=9.1, color=INK, linespacing=1.3)
    arrow(ax, (0.797, rows[5]), (0.753, rows[5]))

    ax.add_patch(FancyBboxPatch(
        (0.015, rows[4] - 0.05), 0.205, 0.10,
        boxstyle="round,pad=0.004,rounding_size=0.01",
        linewidth=1.35, edgecolor=BLUE, facecolor=BLUE_FILL,
    ))
    ax.text(0.1175, rows[4], "Backpropagation\n(Adam)", ha="center",
            va="center", fontsize=8.7, color=INK, linespacing=1.3)
    ax.plot([0.247, 0.1175], [rows[5], rows[5]],
            color=BLUE, linewidth=1.45)
    arrow(ax, (0.1175, rows[5]), (0.1175, rows[4] - 0.054), color=BLUE)

    # The return path is schematic: Adam updates GRU and output-layer weights.
    ax.plot([0.1175, 0.1175], [rows[4] + 0.054, rows[3]],
            color=BLUE, linewidth=1.45)
    arrow(ax, (0.1175, rows[3]), (0.247, rows[3]), color=BLUE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = OUTPUT_DIR / "gru_nbm_architecture"
    for suffix, options in {".png": {"dpi": 300}, ".pdf": {}, ".svg": {}}.items():
        path = stem.with_suffix(suffix)
        fig.savefig(path, bbox_inches="tight", facecolor="white", **options)
        print(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
