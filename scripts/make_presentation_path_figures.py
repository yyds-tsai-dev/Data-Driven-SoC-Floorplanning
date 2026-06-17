#!/usr/bin/env python3
"""Generate slide-ready architecture diagrams for the FloorSet presentation."""

from __future__ import annotations

from pathlib import Path
from textwrap import fill

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "presentation" / "figures"


COLORS = {
    "ink": "#172033",
    "muted": "#5C667A",
    "line": "#748095",
    "panel": "#F6F8FB",
    "input": "#E8F0FA",
    "graph": "#E7F4EE",
    "model": "#EEE7F8",
    "solver": "#FFF0DD",
    "rank": "#FBE2DF",
    "output": "#DDEFE7",
    "block": "#F8FAFD",
    "block_edge": "#AAB3C2",
    "white": "#FFFFFF",
}


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def make_canvas(title: str | None) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(13.33, 7.5), dpi=180)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    if title:
        ax.text(
            0.45,
            8.62,
            title,
            fontsize=15,
            fontweight="bold",
            color=COLORS["ink"],
            va="center",
        )
    return fig, ax


def add_band(ax: plt.Axes, x: float, y: float, w: float, h: float, label: str) -> None:
    ax.add_patch(
        Rectangle(
            (x, y),
            w,
            h,
            linewidth=0.7,
            edgecolor="#D4D9E2",
            facecolor=COLORS["panel"],
            zorder=0,
        )
    )
    ax.text(
        x + 0.15,
        y + h - 0.28,
        label,
        fontsize=8.5,
        fontweight="bold",
        color=COLORS["muted"],
        va="top",
    )


def add_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    color: str,
    *,
    fontsize: float = 8.2,
    weight: str = "normal",
    wrap: int | None = None,
    edge: str = "#AAB3C2",
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.09",
        linewidth=0.9,
        edgecolor=edge,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    if wrap:
        text = "\n".join(fill(part, wrap) for part in text.split("\n"))
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        fontsize=fontsize,
        fontweight=weight,
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.18,
        zorder=3,
    )
    return (x, y, w, h)


def add_labeled_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    description: str,
    *,
    color: str = COLORS["block"],
    edge: str = COLORS["block_edge"],
    title_size: float = 8.8,
    desc_size: float = 7.15,
    title_y: float = 0.64,
    desc_y: float = 0.36,
) -> tuple[float, float, float, float]:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.09",
        linewidth=0.95,
        edgecolor=edge,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * title_y,
        title,
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.08,
        zorder=3,
    )
    ax.text(
        x + w / 2,
        y + h * desc_y,
        description,
        fontsize=desc_size,
        fontweight="normal",
        color=COLORS["ink"],
        ha="center",
        va="center",
        linespacing=1.12,
        zorder=3,
    )
    return (x, y, w, h)


def center_right(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w, y + h / 2)


def center_left(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, _w, h = box
    return (x, y + h / 2)


def center_top(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2, y + h)


def center_bottom(box: tuple[float, float, float, float]) -> tuple[float, float]:
    x, y, w, _h = box
    return (x + w / 2, y)


def arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    rad: float = 0.0,
    color: str = COLORS["line"],
    lw: float = 1.25,
    style: str = "-|>",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=11,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=4,
            shrinkB=4,
            zorder=1,
        )
    )


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("svg", "pdf", "png"):
        path = OUT_DIR / f"{stem}.{ext}"
        if ext == "png":
            fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        else:
            fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def draw_inference_path() -> None:
    fig, ax = make_canvas(None)

    add_band(ax, 0.35, 0.65, 3.0, 7.55, "Input and Shared Instance")
    add_band(ax, 3.75, 0.65, 5.1, 7.55, "Neural Guidance")
    add_band(ax, 9.15, 0.65, 3.0, 7.55, "Deterministic Solver")
    add_band(ax, 12.55, 0.65, 3.1, 7.55, "Score-Aware Selection")

    evaluator = add_labeled_box(
        ax,
        0.65,
        6.35,
        2.4,
        1.05,
        "Evaluation Input",
        "block areas, nets\npins, constraints",
        color=COLORS["input"],
    )
    parser = add_labeled_box(
        ax,
        0.65,
        4.75,
        2.4,
        1.05,
        "Input Normalization",
        "build shared\nproblem view",
        color=COLORS["input"],
    )
    instance = add_labeled_box(
        ax,
        0.65,
        3.05,
        2.4,
        1.05,
        "Shared Problem",
        "normalized\nrepresentation",
        color=COLORS["input"],
    )
    contract = add_labeled_box(
        ax,
        0.65,
        1.35,
        2.4,
        1.0,
        "Stable Contract",
        "encoder can change",
        color=COLORS["input"],
    )

    hetero = add_labeled_box(
        ax,
        4.05,
        6.15,
        2.6,
        1.35,
        "Heterogeneous Graph",
        "blocks, pins, clusters\nMIB groups, boundaries",
        color=COLORS["model"],
    )
    view = add_labeled_box(
        ax,
        4.05,
        4.05,
        2.6,
        1.55,
        "Encoder-Specific View",
        "MPNN block graph\nTransformer context edges\nHGT typed relations",
        color=COLORS["model"],
    )
    model = add_labeled_box(
        ax,
        4.05,
        1.95,
        2.6,
        1.55,
        "Neural Encoder",
        "predicts shared guidance\nlocation, order\nshape, pair direction",
        color=COLORS["model"],
    )
    guidance = add_labeled_box(
        ax,
        7.0,
        3.25,
        1.65,
        1.45,
        "Placement\nGuidance",
        "placement priors\npriorities\nshape hints\npair directions",
        color=COLORS["model"],
        title_size=7.9,
        desc_size=6.45,
        title_y=0.74,
        desc_y=0.33,
    )

    specs = add_labeled_box(
        ax,
        9.45,
        6.25,
        2.3,
        1.05,
        "Candidate Policy",
        "runtime budget\nand profiles",
        color=COLORS["solver"],
    )
    decoder = add_labeled_box(
        ax,
        9.45,
        4.55,
        2.3,
        1.1,
        "Constructive Decoder",
        "relative ordering\noptional beam search",
        color=COLORS["solver"],
    )
    repair = add_labeled_box(
        ax,
        9.45,
        2.55,
        2.3,
        1.35,
        "Repair",
        "hard legality first\nthen boundary, grouping\nand MIB refinement",
        color=COLORS["solver"],
    )

    metrics = add_labeled_box(
        ax,
        12.85,
        5.55,
        2.45,
        1.05,
        "Placement Diagnostics",
        "wirelength estimate\noutline area, soft violations",
        color=COLORS["rank"],
        desc_size=6.95,
    )
    rank = add_labeled_box(
        ax,
        12.85,
        3.45,
        2.45,
        1.45,
        "Score-Aware Selection",
        "hard legality first\nquality estimate\nsoft penalties\nfinal tie-break",
        color=COLORS["rank"],
    )
    output = add_labeled_box(
        ax,
        12.85,
        1.45,
        2.45,
        1.15,
        "Final Floorplan",
        "rectangle list\nposition and size",
        color=COLORS["rank"],
    )

    arrow(ax, center_bottom(evaluator), center_top(parser))
    arrow(ax, center_bottom(parser), center_top(instance))
    arrow(ax, center_bottom(instance), center_top(contract))
    arrow(ax, center_right(instance), center_left(hetero))
    arrow(ax, center_bottom(hetero), center_top(view))
    arrow(ax, center_bottom(view), center_top(model))
    arrow(ax, center_right(model), center_left(guidance))
    arrow(ax, center_right(guidance), center_left(specs), rad=-0.08)
    arrow(ax, center_bottom(specs), center_top(decoder))
    arrow(ax, center_bottom(decoder), center_top(repair))
    arrow(ax, center_right(repair), center_left(metrics), rad=-0.06)
    arrow(ax, center_bottom(metrics), center_top(rank))
    arrow(ax, center_bottom(rank), center_top(output))

    ax.plot([8.95, 8.95], [0.9, 7.95], color="#B8C0CC", lw=1.0, ls=(0, (4, 3)))

    save_figure(fig, "inference_path_block_architecture")


def draw_training_path() -> None:
    fig, ax = make_canvas("Training path: learn guidance, promote by evaluator evidence")

    add_band(ax, 0.35, 0.65, 3.35, 7.55, "Data")
    add_band(ax, 4.0, 0.65, 3.35, 7.55, "Target construction")
    add_band(ax, 7.65, 0.65, 3.35, 7.55, "Encoder training")
    add_band(ax, 11.3, 0.65, 4.2, 7.55, "Checkpoint selection")

    sample = add_box(
        ax,
        0.65,
        6.0,
        2.75,
        1.45,
        "FloorSet-Lite sample\narea_target, b2b, p2b\npins, constraints\nfp_sol, tree_sol, metrics_sol",
        COLORS["input"],
        fontsize=7.25,
    )
    unpack = add_box(
        ax,
        0.65,
        3.75,
        2.75,
        1.55,
        "unpack_batch_sample()\ncurrent path uses fp_sol\nmetrics as metadata\ntree_sol for future work",
        COLORS["input"],
        fontsize=7.25,
    )

    audit = add_box(
        ax,
        4.3,
        5.75,
        2.65,
        1.25,
        "Soft-constraint audit\nboundary, grouping,\nMIB violations",
        COLORS["solver"],
        fontsize=7.6,
    )
    clean = add_box(
        ax,
        4.1,
        3.55,
        1.45,
        1.25,
        "Clean fp_sol\nuse direct\ngeometry target",
        COLORS["output"],
        fontsize=7.0,
    )
    dirty = add_box(
        ax,
        5.85,
        3.2,
        1.3,
        1.95,
        "Dirty sample\nskip strict-clean\nor repair into\npseudo target\nand down-weight",
        COLORS["rank"],
        fontsize=6.7,
    )
    target = add_box(
        ax,
        4.35,
        1.35,
        2.55,
        1.35,
        "build_training_target_record()\nanchor center, priority\nlog_aspect, pair axis",
        COLORS["solver"],
        fontsize=7.1,
        weight="bold",
    )

    features = add_box(
        ax,
        7.95,
        5.75,
        2.7,
        1.25,
        "build_anchor_node_features()\n+ heterogeneous graph",
        COLORS["graph"],
        fontsize=7.4,
    )
    graph_view = add_box(
        ax,
        7.95,
        3.45,
        2.7,
        1.75,
        "Encoder-specific graph view\nMPNN block graph\nGraph Transformer context edges\nHGT typed relations",
        COLORS["graph"],
        fontsize=7.0,
    )
    forward = add_box(
        ax,
        7.95,
        1.45,
        2.7,
        1.25,
        "FloorplanGNN.forward()\nshared prediction heads",
        COLORS["model"],
        fontsize=7.5,
        weight="bold",
    )

    losses = add_box(
        ax,
        11.65,
        5.45,
        3.35,
        1.7,
        "Losses\nanchor geometry, aspect\npriority/order, pairwise\nweighted dirty targets",
        COLORS["model"],
        fontsize=7.35,
    )
    checkpoint = add_box(
        ax,
        11.65,
        3.45,
        3.35,
        1.2,
        "save_anchor_checkpoint()\nmanifest records config\nand relation gates",
        COLORS["white"],
        fontsize=7.45,
    )
    evaluator = add_box(
        ax,
        11.65,
        1.25,
        3.35,
        1.55,
        "Training-owned evaluator\nfull validation per epoch\npromote by evaluator evidence\nruntime remains a gate",
        COLORS["rank"],
        fontsize=7.2,
        weight="bold",
    )

    arrow(ax, center_bottom(sample), center_top(unpack))
    arrow(ax, center_right(unpack), center_left(audit))
    arrow(ax, center_bottom(audit), center_top(clean), rad=0.18)
    arrow(ax, center_bottom(audit), center_top(dirty), rad=-0.18)
    arrow(ax, center_bottom(clean), center_top(target), rad=-0.05)
    arrow(ax, center_bottom(dirty), center_top(target), rad=0.05)
    arrow(ax, center_right(target), center_left(features), rad=-0.05)
    arrow(ax, center_bottom(features), center_top(graph_view))
    arrow(ax, center_bottom(graph_view), center_top(forward))
    arrow(ax, center_right(forward), center_left(losses), rad=-0.06)
    arrow(ax, center_bottom(losses), center_top(checkpoint))
    arrow(ax, center_bottom(checkpoint), center_top(evaluator))
    arrow(
        ax,
        (13.35, 1.25),
        (13.35, 3.45),
        rad=0.0,
        color="#9C6B73",
        lw=1.0,
        style="<|-",
    )

    ax.text(
        0.45,
        0.28,
        "Key point: training learns guidance targets; inference still uses deterministic legality repair and v10-aware candidate ranking.",
        fontsize=8.1,
        color=COLORS["muted"],
        va="center",
    )
    save_figure(fig, "training_path_flowchart")


def main() -> None:
    draw_inference_path()
    draw_training_path()


if __name__ == "__main__":
    main()
