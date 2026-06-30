from __future__ import annotations

from pathlib import Path

import torch

from floorset_arch.diagnostics import placement_metrics, repair_delta
from floorset_arch.geometry import bbox
from floorset_arch.models import Instance, Placement, Rect


def xywh_from_fp_sol(fp_sol: torch.Tensor, block_count: int) -> torch.Tensor:
    sol = torch.as_tensor(fp_sol).detach().cpu().float()
    if sol.dim() >= 3 and sol.shape[-1] == 2:
        polygons = sol.reshape(-1, sol.shape[-2], 2)[:block_count]
        min_xy = polygons.min(dim=1).values
        max_xy = polygons.max(dim=1).values
        width_height = (max_xy - min_xy).clamp_min(1.0)
        return torch.cat([min_xy, width_height], dim=1)
    if sol.dim() == 1:
        sol = sol.reshape(-1, 4)
    sol = sol.reshape(-1, sol.shape[-1])[:block_count, :4]
    width = sol[:, 0].clamp_min(1.0)
    height = sol[:, 1].clamp_min(1.0)
    x = sol[:, 2]
    y = sol[:, 3]
    return torch.stack([x, y, width, height], dim=1)


def placement_from_fp_sol(fp_sol: torch.Tensor, block_count: int) -> Placement:
    xywh = xywh_from_fp_sol(fp_sol, block_count)
    rects: dict[int, Rect] = {}
    for block in range(min(block_count, int(xywh.shape[0]))):
        x, y, width, height = [float(value) for value in xywh[block].tolist()]
        rects[block] = Rect(x, y, max(1.0, width), max(1.0, height))
    return Placement(rects)


def placement_from_positions(
    positions: list[tuple[float, float, float, float]] | tuple[tuple[float, float, float, float], ...],
    block_count: int | None = None,
) -> Placement:
    count = len(positions) if block_count is None else min(int(block_count), len(positions))
    rects: dict[int, Rect] = {}
    for block in range(count):
        x, y, width, height = [float(value) for value in positions[block]]
        rects[block] = Rect(x, y, max(1.0, width), max(1.0, height))
    return Placement(rects)


def _boundary_label(code: int) -> str:
    parts = []
    if code & 8:
        parts.append("B")
    if code & 1:
        parts.append("L")
    if code & 2:
        parts.append("R")
    if code & 4:
        parts.append("T")
    return "".join(parts)


def constraint_annotation_summary(inst: Instance) -> dict[int, dict[str, object]]:
    cluster_by_block = {
        block: int(cluster_id)
        for cluster_id, members in inst.cluster_groups.items()
        for block in members
    }
    mib_by_block = {
        block: int(mib_id)
        for mib_id, members in inst.mib_groups.items()
        for block in members
    }
    return {
        block: {
            "cluster": cluster_by_block.get(block, 0),
            "mib": mib_by_block.get(block, 0),
            "fixed": block in inst.fixed,
            "preplaced": block in inst.preplaced,
            "boundary": _boundary_label(int(inst.boundary.get(block, 0))),
        }
        for block in range(inst.block_count)
    }


def placement_panel_title(
    title: str,
    metrics: dict[str, float | int],
    cost: float | None = None,
) -> str:
    soft = (
        int(metrics["boundary_violations"])
        + int(metrics["group_violations"])
        + int(metrics["mib_violations"])
    )
    cost_text = "n/a" if cost is None else f"{float(cost):.4f}"
    return (
        f"{title}\n"
        f"overlaps={int(metrics['overlap_count'])} "
        f"soft={soft} bbox={float(metrics['bbox_area']):.0f} cost={cost_text}"
    )


def build_diffusion_diagnostic_report(
    inst: Instance,
    raw: Placement,
    repaired: Placement,
    golden: Placement,
    case_id: int | None = None,
) -> dict:
    raw_metrics = placement_metrics(inst, raw)
    repaired_metrics = placement_metrics(inst, repaired)
    golden_metrics = placement_metrics(inst, golden)
    return {
        "case_id": case_id,
        "block_count": inst.block_count,
        "placements": {
            "raw": {"metrics": raw_metrics},
            "repaired": {"metrics": repaired_metrics},
            "golden": {"metrics": golden_metrics},
        },
        "repair_delta": repair_delta(raw, repaired, raw_metrics, repaired_metrics),
        "golden_delta": repair_delta(repaired, golden, repaired_metrics, golden_metrics),
    }


def _draw_placement(
    ax,
    inst: Instance,
    placement: Placement,
    title: str,
    metrics: dict[str, float | int],
    cost: float | None = None,
) -> None:
    import matplotlib.patches as patches
    from matplotlib.lines import Line2D

    rects = list(placement.rects.values())
    bounds = bbox(rects)
    annotations = constraint_annotation_summary(inst)
    cluster_ids = sorted(
        {int(item["cluster"]) for item in annotations.values() if int(item["cluster"]) > 0}
    )
    palette = [
        "#f59e0b",
        "#0ea5e9",
        "#22c55e",
        "#a855f7",
        "#ef4444",
        "#14b8a6",
        "#eab308",
        "#6366f1",
    ]
    cluster_color = {
        cluster_id: palette[index % len(palette)]
        for index, cluster_id in enumerate(cluster_ids)
    }
    for block, rect in placement.rects.items():
        annotation = annotations.get(block, {})
        cluster_id = int(annotation.get("cluster", 0) or 0)
        facecolor = cluster_color.get(cluster_id, "#d1d5db")
        edgecolor = "#111827" if annotation.get("fixed") or annotation.get("preplaced") else "#6b7280"
        linewidth = 1.4 if annotation.get("fixed") or annotation.get("preplaced") else 0.7
        patch = patches.Rectangle(
            (rect.x, rect.y),
            rect.width,
            rect.height,
            linewidth=linewidth,
            edgecolor=edgecolor,
            facecolor=facecolor,
            alpha=0.58 if cluster_id else 0.36,
        )
        ax.add_patch(patch)
        if len(placement.rects) <= 160:
            ax.text(
                rect.center_x,
                rect.center_y,
                str(block),
                ha="center",
                va="center",
                fontsize=6,
                color="#111827",
            )
        if cluster_id:
            ax.text(
                rect.x,
                rect.top,
                f"C{cluster_id}",
                ha="left",
                va="bottom",
                fontsize=5,
                color="#2563eb",
            )
        mib_id = int(annotation.get("mib", 0) or 0)
        if mib_id:
            ax.scatter(
                [rect.center_x],
                [rect.top],
                marker="D",
                s=18,
                facecolors="none",
                edgecolors="#d946ef",
                linewidths=0.8,
            )
            ax.text(
                rect.center_x,
                rect.top,
                f"M{mib_id}",
                ha="center",
                va="bottom",
                fontsize=5,
                color="#d946ef",
            )
        boundary = str(annotation.get("boundary", ""))
        if boundary:
            ax.text(
                rect.center_x,
                rect.top + max(rect.height * 0.05, 0.3),
                boundary,
                ha="center",
                va="bottom",
                fontsize=5,
                color="#dc2626",
            )
        if annotation.get("fixed") or annotation.get("preplaced"):
            ax.scatter(
                [rect.x],
                [rect.y],
                marker="s",
                s=16,
                color="#111827",
            )
            label = "P" if annotation.get("preplaced") else "F"
            ax.text(
                rect.x,
                rect.y,
                label,
                ha="right",
                va="top",
                fontsize=5,
                color="#111827",
            )
    pad = max(bounds.width, bounds.height, 1.0) * 0.04
    ax.set_xlim(bounds.x - pad, bounds.right + pad)
    ax.set_ylim(bounds.y - pad, bounds.top + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(placement_panel_title(title, metrics, cost=cost), fontsize=9)
    ax.set_xlabel("X", fontsize=8)
    ax.set_ylabel("Y", fontsize=8)
    ax.tick_params(axis="both", labelsize=7)
    legend_handles = [
        Line2D([0], [0], marker="o", color="none", markeredgecolor="#2563eb", label="cluster: same color", markersize=5),
        Line2D([0], [0], marker="D", color="none", markeredgecolor="#d946ef", label="MIB", markersize=5),
        Line2D([0], [0], marker="s", color="#111827", label="fixed/preplaced", markersize=5),
        Line2D([0], [0], marker="*", color="#dc2626", label="boundary", markersize=6),
    ]
    ax.legend(handles=legend_handles, fontsize=5.5, loc="upper right", framealpha=0.75)


def plot_diffusion_diagnostic(
    inst: Instance,
    raw: Placement,
    repaired: Placement,
    golden: Placement,
    output_path: str | Path,
    case_id: int | None = None,
    costs: dict[str, float | None] | None = None,
    panel_titles: dict[str, str] | None = None,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    report = build_diffusion_diagnostic_report(inst, raw, repaired, golden, case_id)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    costs = costs or {}
    panel_titles = panel_titles or {}
    labels = (
        ("raw", panel_titles.get("raw", "Diffusion raw"), raw),
        ("repaired", panel_titles.get("repaired", "Repaired"), repaired),
        ("golden", panel_titles.get("golden", "Official golden"), golden),
    )
    for ax, (key, name, placement) in zip(axes, labels, strict=True):
        metrics = report["placements"][key]["metrics"]
        _draw_placement(ax, inst, placement, name, metrics, cost=costs.get(key))
    if case_id is not None:
        fig.suptitle(f"Diffusion diagnostic case {case_id}", fontsize=11)
    fig.savefig(output, dpi=150)
    plt.close(fig)
    return output
