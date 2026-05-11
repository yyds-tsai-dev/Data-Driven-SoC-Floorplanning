from __future__ import annotations

from floorset_arch.geometry import bbox, overlaps
from floorset_arch.models import Instance, Placement
from floorset_arch.repair import soft_violation_counts
from floorset_arch.scoring import hpwl_proxy


def overlap_count(placement: Placement) -> int:
    rects = list(placement.rects.values())
    count = 0
    for idx, rect in enumerate(rects):
        for other in rects[idx + 1 :]:
            if overlaps(rect, other):
                count += 1
    return count


def placement_metrics(inst: Instance, placement: Placement) -> dict[str, float | int]:
    bounds = bbox(list(placement.rects.values()))
    boundary, grouping, mib = soft_violation_counts(inst, placement)
    return {
        "overlap_count": overlap_count(placement),
        "boundary_violations": boundary,
        "group_violations": grouping,
        "mib_violations": mib,
        "hpwl_proxy": float(hpwl_proxy(inst, placement.rects)),
        "bbox_area": float(bounds.area),
    }


def repair_delta(
    before: Placement,
    after: Placement,
    before_metrics: dict[str, float | int],
    after_metrics: dict[str, float | int],
) -> dict[str, float]:
    moved = 0.0
    count = 0
    for block, before_rect in before.rects.items():
        after_rect = after.rects.get(block)
        if after_rect is None:
            continue
        moved += abs(after_rect.x - before_rect.x) + abs(after_rect.y - before_rect.y)
        count += 1
    return {
        "avg_moved_manhattan": moved / max(count, 1),
        "bbox_area_delta": float(after_metrics["bbox_area"]) - float(before_metrics["bbox_area"]),
        "hpwl_proxy_delta": float(after_metrics["hpwl_proxy"]) - float(before_metrics["hpwl_proxy"]),
    }
