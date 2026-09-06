"""Same-shape sparse-topology distillation for the frozen Direct denoiser.

The trainer consumes only receipt-bound sanitized training corpora and sparse
topology labels.  It never imports contest validation loaders and never uses
dense solution coordinates as a target.

This is a fork of ``train_topology_prior.py`` (frozen, do not edit) that adds
optional on-the-fly P2B/B2B endpoint-resampling augmentation ("v4"), so the
student is trained to be robust to the connectivity shift used by the
``alpha_1`` / ``p2b_1__b2b_1`` shadow-suite probes: pins/nets re-attached to
blocks proportional to ``(1 - z)^alpha``, where ``z`` is a per-pin (P2B) or
per-layout (B2B) min-max-normalized Manhattan distance computed against the
*golden* layout for that instance.  See ``artifacts/shadow_hidden_suites/alpha_1/alpha_1/
proxy_manifest.json`` and ``artifacts/shadow_hidden_suites/p2b_1__b2b_1/p2b_1__b2b_1/
proxy_manifest.json`` for the reference semantics this mirrors.

Only training-time conditioning is perturbed; the sparse topology labels
(edges/contacts/pin_paths, derived from the frozen teacher) are never
touched, and heldout evaluation always runs unaugmented so checkpoint
selection stays comparable to the v2 baseline.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import torch


_PARTNER = Path(__file__).resolve().parents[1] / "solver"
_REPO = _PARTNER.parent
for _import_root in (_PARTNER, _REPO / "FloorSet/iccad2026contest"):
    if str(_import_root) not in sys.path:
        sys.path.insert(0, str(_import_root))

from . import data as batch_data
from . import energy
from . import engine
from .checkpoint_identity import (
    IDENTITY_SCHEMA,
    canonical_checkpoint_identity,
    canonical_state_sha256,
)
from .data import LITE_ROOT, BandFileSampler
from .sampler import expand_cond, sample_differentiable
from .topology_data import (
    ContactLabel,
    CorpusSourceReceipt,
    SparseEdge,
    TopologyLabel,
    _load_receipt_source,
    _sanitize,
    collate_labels,
    fingerprint_case,
    source_instance_id,
)
from .topology_prior import topology_losses


SOURCE_FILE_SHA256 = "508f5fce594ba3b5aeca93ce5e8db417cb256b5e409634acf8bd837add606659"
CONTROL_FILE_SHA256 = "2b9ce827aed93443e442a002d178e8e6282cb4c6148818c9122a6ff0411c8a02"
FLOW_FILE_SHA256 = "110c1d84d74ee88d94cf8d3be9ac464602747db8c301d95b3ca69a2cb8bd2f09"
EXPECTED_CONFIG_SHA256 = "4c6a1e19f0574af348efa81a758c05524522ad3501d46f18e839774fa933971b"
EXPECTED_KEYSET_SHA256 = "79a51975d9b9f583143259198d244554c8a4e97122fc5e5cf150ec64f4429ba7"
EXPECTED_STATE_SHA256 = "92838740993a697a56f3afdfba4402eb83c8dc095fe43462f8bdaffdb4ef5ecb"


@dataclass(frozen=True)
class TrainingRecord:
    receipt: CorpusSourceReceipt
    case: Mapping[str, Any]
    label: TopologyLabel


# ---------------------------------------------------------------------------
# P2B / B2B endpoint-resampling augmentation ("v4")
# ---------------------------------------------------------------------------
class GoldenLayoutCache:
    """Loads golden (x, y, w, h) rectangles for a record, hash-verified.

    Reuses ``topology_data._load_receipt_source`` (sha256-checked against the
    receipt) and ``BandFileSampler._instance`` (the same reconstruction path
    the corpus itself was built from) so augmentation never trusts an
    unverified file.  Loads are cached per source file (each file holds 112
    layouts), matching ``BandFileSampler``'s own amortization strategy.
    """

    def __init__(self, root: Path = LITE_ROOT):
        self._root = root
        self._source_cache: dict[Path, tuple[str, Any]] = {}
        self._instance_cache: dict[str, Optional[torch.Tensor]] = {}

    def centers(self, receipt: CorpusSourceReceipt, n: int) -> Optional[torch.Tensor]:
        """Golden block centers as a float64 (n, 2) tensor, or None on any
        verification / shape mismatch (augmentation is skipped for that
        record rather than risking a silently wrong conditioning signal)."""
        key = source_instance_id(receipt)
        if key not in self._instance_cache:
            self._instance_cache[key] = self._load(receipt, n)
        centers = self._instance_cache[key]
        if centers is None or int(centers.shape[0]) != n:
            return None
        return centers

    def _load(self, receipt: CorpusSourceReceipt, n: int) -> Optional[torch.Tensor]:
        try:
            source = _load_receipt_source(self._root, receipt, self._source_cache)
            instance = BandFileSampler._instance(source, receipt.layout_index)
        except Exception:
            return None
        if not isinstance(instance, Mapping) or int(instance.get("n", -1)) != n:
            return None
        golden = instance.get("golden")
        if not isinstance(golden, Sequence) or len(golden) != n:
            return None
        try:
            centers = torch.tensor(
                [[float(x) + float(w) / 2.0, float(y) + float(h) / 2.0]
                 for (x, y, w, h) in golden],
                dtype=torch.float64,
            )
        except (TypeError, ValueError):
            return None
        if not torch.isfinite(centers).all():
            return None
        return centers


def _weighted_rank_without_replacement(
    rng: random.Random, weights: Sequence[float]
) -> list[int]:
    """Efraimidis-Spirakis weighted sampling without replacement.

    Returns a permutation of ``range(len(weights))`` ranked from most to
    least likely to be drawn first under weights ``w_i``: each index gets key
    ``u_i ** (1 / w_i)`` for ``u_i ~ Uniform(0, 1)``, and the permutation
    sorts by key descending.  Taking a prefix of this permutation is
    equivalent to iterated weighted sampling without replacement.
    """
    keyed = []
    for index, weight in enumerate(weights):
        w = max(float(weight), 1e-12)
        u = max(rng.random(), 1e-12)
        keyed.append((u ** (1.0 / w), index))
    keyed.sort(key=lambda item: item[0], reverse=True)
    return [index for _, index in keyed]


def _minmax_z(distances: Sequence[float]) -> list[float]:
    if not distances:
        return []
    d_min = min(distances)
    d_max = max(distances)
    if d_max <= d_min:
        return [0.0 for _ in distances]
    span = d_max - d_min
    return [(d - d_min) / span for d in distances]


def _resample_p2b(
    p2b: Sequence[Sequence[float]],
    pins: Sequence[Sequence[float]],
    centers: torch.Tensor,
    n: int,
    rng: random.Random,
    alpha: float,
) -> list[list[float]]:
    """Alpha_1 semantics: per pin, z = min-max Manhattan distance from the
    pin to each block's golden center; new block endpoints drawn
    P(block) ~ (1 - z)^alpha, without replacement per pin.  P2B row count,
    per-pin degree, and row weights are preserved exactly."""
    if alpha <= 0 or not p2b or n <= 0:
        return [list(row) for row in p2b]
    by_pin: dict[int, list[int]] = {}
    for row_index, row in enumerate(p2b):
        by_pin.setdefault(int(row[0]), []).append(row_index)
    new_p2b = [list(row) for row in p2b]
    cx = centers[:, 0].tolist()
    cy = centers[:, 1].tolist()
    for pin_index, row_indices in by_pin.items():
        if not 0 <= pin_index < len(pins):
            continue
        px, py = float(pins[pin_index][0]), float(pins[pin_index][1])
        distances = [abs(cx[block] - px) + abs(cy[block] - py) for block in range(n)]
        weights = [(1.0 - z) ** alpha for z in _minmax_z(distances)]
        ranked = _weighted_rank_without_replacement(rng, weights)
        degree = min(len(row_indices), n)
        chosen = ranked[:degree]
        for row_index, block in zip(row_indices, chosen):
            new_p2b[row_index][1] = int(block)
    return new_p2b


def _resample_b2b(
    b2b: Sequence[Sequence[float]],
    centers: torch.Tensor,
    n: int,
    rng: random.Random,
    alpha: float,
) -> list[list[float]]:
    """p2b_1__b2b_1 semantics: z is normalized per layout over the unique
    unordered block pairs (not per-row), P(pair) ~ (1 - z)^alpha; new
    endpoints drawn without replacement from that shared per-layout pool
    (wrapping back to the top of the weighted ranking if the row count
    exceeds the number of unique pairs).  B2B row count and row weights are
    preserved exactly."""
    if alpha <= 0 or not b2b or n < 2:
        return [list(row) for row in b2b]
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n)]
    cx = centers[:, 0].tolist()
    cy = centers[:, 1].tolist()
    distances = [abs(cx[a] - cx[b]) + abs(cy[a] - cy[b]) for a, b in pairs]
    weights = [(1.0 - z) ** alpha for z in _minmax_z(distances)]
    ranked = _weighted_rank_without_replacement(rng, weights)
    new_b2b = [list(row) for row in b2b]
    for offset, row in enumerate(new_b2b):
        pair_index = ranked[offset % len(ranked)]
        a, b = pairs[pair_index]
        row[0], row[1] = int(a), int(b)
    return new_b2b


def augment_case(
    case: Mapping[str, Any],
    receipt: CorpusSourceReceipt,
    golden: GoldenLayoutCache,
    rng: random.Random,
    *,
    p2b_alpha: float,
    b2b_alpha: float,
) -> Optional[dict[str, Any]]:
    """Return a copy of ``case`` with P2B/B2B endpoints resampled against the
    golden layout, or None if the golden layout cannot be verified/loaded
    (caller should fall back to the unaugmented record)."""
    if p2b_alpha <= 0 and b2b_alpha <= 0:
        return None
    n = int(case["n"])
    centers = golden.centers(receipt, n)
    if centers is None:
        return None
    augmented = dict(case)
    if p2b_alpha > 0:
        augmented["p2b"] = _resample_p2b(
            case["p2b"], case["pins"], centers, n, rng, p2b_alpha
        )
    if b2b_alpha > 0:
        augmented["b2b"] = _resample_b2b(case["b2b"], centers, n, rng, b2b_alpha)
    return augmented


def _canonical_bytes(value: Any, *, newline: bool = False) -> bytes:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return payload + (b"\n" if newline else b"")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("temporary output")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def _atomic_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise ValueError("temporary checkpoint")
    try:
        with temporary.open("xb") as stream:
            torch.save(dict(payload), stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists() and not temporary.is_symlink():
            temporary.unlink()


def frozen_portfolio_contract(*, flow_slots: int = 3, nref: int = 6) -> dict[str, Any]:
    payload = {
        "schema": "icdc_topology_3d3f_v1",
        "nref": nref,
        "direct_slots": nref - flow_slots,
        "flow_slots": flow_slots,
        "direct_sampler": "dpmpp",
        "direct_steps": 2,
        "flow_sampler": "euler",
        "flow_steps": 8,
        "oversample": 1,
    }
    if type(flow_slots) is not int or type(nref) is not int:
        raise ValueError("portfolio")
    if nref != 6 or flow_slots != 3 or payload["direct_slots"] != 3:
        raise ValueError("portfolio")
    return {**payload, "sha256": hashlib.sha256(_canonical_bytes(payload)).hexdigest()}


def _state_shape_dtype(state: Any) -> Optional[dict[str, tuple[tuple[int, ...], str]]]:
    if not isinstance(state, Mapping) or not state:
        return None
    result: dict[str, tuple[tuple[int, ...], str]] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key or not isinstance(value, torch.Tensor):
            return None
        result[key] = (tuple(value.shape), str(value.dtype))
    return result


def checkpoint_contract(
    base: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    sampler_method: str = "dpmpp",
    sampler_steps: int = 2,
    candidate_count: int = 6,
    portfolio_contract_sha256: str,
) -> dict[str, Any]:
    failed: list[str] = []
    try:
        portfolio = frozen_portfolio_contract()
    except ValueError:
        portfolio = {"sha256": ""}
    if sampler_method != "dpmpp" or sampler_steps != 2:
        failed.append("direct_sampler")
    if type(candidate_count) is not int or candidate_count != 6:
        failed.append("candidate_count")
    if portfolio_contract_sha256 != portfolio["sha256"]:
        failed.append("portfolio_contract")
    if not isinstance(base, Mapping) or not isinstance(candidate, Mapping):
        failed.append("checkpoint_mapping")
    else:
        if base.get("model_config") != candidate.get("model_config"):
            failed.append("model_config")
        for field in ("model", "ema"):
            base_contract = _state_shape_dtype(base.get(field))
            candidate_contract = _state_shape_dtype(candidate.get(field))
            if base_contract is None or candidate_contract != base_contract:
                failed.append(f"{field}_shape_dtype")
    return {
        "ok": not failed,
        "failed": failed,
        "sampler": sampler_method,
        "sampler_steps": sampler_steps,
        "production_candidates": candidate_count,
        "portfolio_contract_sha256": portfolio_contract_sha256,
    }


def verify_source_contract(
    source: Mapping[str, Any], control: Mapping[str, Any]
) -> dict[str, Any]:
    failed: list[str] = []
    try:
        source_ema = canonical_state_sha256(source["ema"])
        control_model = canonical_state_sha256(control["model"])
        control_ema = canonical_state_sha256(control["ema"])
    except (KeyError, TypeError, ValueError):
        return {"ok": False, "failed": ["checkpoint_state"]}
    if not (source_ema == control_model == control_ema):
        failed.append("source_ema_equals_c0")
    if source.get("model_config") != control.get("model_config"):
        failed.append("model_config")
    return {
        "ok": not failed,
        "failed": failed,
        "source_ema_sha256": source_ema,
        "control_model_sha256": control_model,
        "control_ema_sha256": control_ema,
    }


def _receipt(value: Any) -> CorpusSourceReceipt:
    if not isinstance(value, Mapping) or set(value) != {
        "relative_path", "file_sha256", "layout_index", "fingerprint"
    }:
        raise ValueError("receipt")
    receipt = CorpusSourceReceipt(
        relative_path=value["relative_path"],
        file_sha256=value["file_sha256"],
        layout_index=value["layout_index"],
        fingerprint=value["fingerprint"],
    )
    source_instance_id(receipt)
    for field in (receipt.file_sha256, receipt.fingerprint):
        if not isinstance(field, str) or len(field) != 64:
            raise ValueError("receipt")
        try:
            int(field, 16)
        except ValueError as exc:
            raise ValueError("receipt") from exc
    return receipt


def _sparse_edge(value: Any, n: int) -> SparseEdge:
    if not isinstance(value, Mapping) or set(value) != {
        "src", "dst", "axis", "margin", "kind", "weight"
    }:
        raise ValueError("edge")
    edge = SparseEdge(**value)
    if (
        type(edge.src) is not int
        or type(edge.dst) is not int
        or not 0 <= edge.src < n
        or not 0 <= edge.dst < n
        or edge.src == edge.dst
        or type(edge.axis) is not int
        or edge.axis not in (0, 1)
        or not isinstance(edge.kind, str)
        or not edge.kind
        or not math.isfinite(float(edge.margin))
        or float(edge.margin) < 0
        or not math.isfinite(float(edge.weight))
        or float(edge.weight) <= 0
    ):
        raise ValueError("edge")
    return edge


def _contact(value: Any, n: int) -> ContactLabel:
    if not isinstance(value, Mapping) or set(value) != {
        "a", "b", "axis", "a_before_b", "perp_margin", "weight"
    }:
        raise ValueError("contact")
    contact = ContactLabel(**value)
    if (
        type(contact.a) is not int
        or type(contact.b) is not int
        or not 0 <= contact.a < n
        or not 0 <= contact.b < n
        or contact.a == contact.b
        or type(contact.axis) is not int
        or contact.axis not in (0, 1)
        or type(contact.a_before_b) is not bool
        or not math.isfinite(float(contact.perp_margin))
        or float(contact.perp_margin) <= 0
        or not math.isfinite(float(contact.weight))
        or float(contact.weight) <= 0
    ):
        raise ValueError("contact")
    return contact


def _topology_label(value: Any) -> TopologyLabel:
    if not isinstance(value, Mapping) or set(value) != {
        "instance_id", "n", "sample_seed", "teacher_cost", "base_cost",
        "record_weight", "edges", "contacts", "pin_paths"
    }:
        raise ValueError("label")
    n = value["n"]
    if type(n) is not int or n <= 0:
        raise ValueError("label")
    edges = tuple(_sparse_edge(item, n) for item in value["edges"])
    contacts = tuple(_contact(item, n) for item in value["contacts"])
    pin_paths = tuple(tuple(path) for path in value["pin_paths"])
    label = TopologyLabel(
        instance_id=value["instance_id"],
        n=n,
        sample_seed=value["sample_seed"],
        teacher_cost=value["teacher_cost"],
        base_cost=value["base_cost"],
        record_weight=value["record_weight"],
        edges=edges,
        contacts=contacts,
        pin_paths=pin_paths,
    )
    collate_labels([label], torch.device("cpu"), torch.float32)
    return label


def _selection_bucket(instance_id: str, modulus: int, namespace: str) -> int:
    if type(modulus) is not int or modulus <= 0:
        raise ValueError("selection modulus")
    payload = namespace.encode("ascii") + b"\0" + instance_id.encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % modulus


def load_paired_records(
    corpus_path: str | Path,
    labels_path: str | Path,
    *,
    selection_mod: int = 1,
    selection_namespace: str = "icdc-topology-records-v1",
    max_records: Optional[int] = None,
) -> list[TrainingRecord]:
    if max_records is not None and (type(max_records) is not int or max_records <= 0):
        raise ValueError("max records")
    records: list[TrainingRecord] = []
    with Path(corpus_path).open("r", encoding="ascii") as corpus_stream, Path(
        labels_path
    ).open("r", encoding="ascii") as label_stream:
        line_number = 0
        while True:
            corpus_line = corpus_stream.readline()
            label_line = label_stream.readline()
            if not corpus_line and not label_line:
                break
            line_number += 1
            if not corpus_line or not label_line:
                raise ValueError("record coverage")
            try:
                corpus_row = json.loads(corpus_line)
                label_row = json.loads(label_line)
            except (json.JSONDecodeError, UnicodeError) as exc:
                raise ValueError("record json") from exc
            if (
                not isinstance(corpus_row, Mapping)
                or set(corpus_row) != {"receipt", "case"}
                or not isinstance(label_row, Mapping)
                or set(label_row) != {"receipt", "label"}
                or corpus_row["receipt"] != label_row["receipt"]
            ):
                raise ValueError("record binding")
            receipt = _receipt(corpus_row["receipt"])
            case = _sanitize(corpus_row["case"], artifact=True)
            label = _topology_label(label_row["label"])
            instance_id = source_instance_id(receipt)
            if (
                case["instance_id"] != instance_id
                or label.instance_id != instance_id
                or label.n != case["n"]
                or fingerprint_case(case) != receipt.fingerprint
            ):
                raise ValueError("record binding")
            if _selection_bucket(instance_id, selection_mod, selection_namespace) != 0:
                continue
            records.append(TrainingRecord(receipt, case, label))
            if max_records is not None and len(records) >= max_records:
                break
    if not records:
        raise ValueError("empty records")
    ids = [record.label.instance_id for record in records]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate records")
    return records


def _tensor_case(case: Mapping[str, Any]) -> dict[str, Any]:
    def matrix(name: str, width: int) -> torch.Tensor:
        value = case[name]
        if not value:
            return torch.empty((0, width), dtype=torch.float64)
        return torch.tensor(value, dtype=torch.float64)

    return {
        "instance_id": case["instance_id"],
        "n": case["n"],
        "area": torch.tensor(case["area"], dtype=torch.float64),
        "cons": torch.tensor(case["cons"], dtype=torch.float64),
        "tp": torch.tensor(case["tp"], dtype=torch.float64),
        "b2b": matrix("b2b", 3),
        "p2b": matrix("p2b", 3),
        "pins": matrix("pins", 2),
        "hpwl_ref": float(case["hpwl_ref"]),
        "area_ref": float(case["area_ref"]),
    }


def _batch(records: Sequence[TrainingRecord], device: torch.device) -> dict[str, torch.Tensor]:
    return batch_data.collate(
        [_tensor_case(record.case) for record in records],
        device=device,
        dtype=torch.float32,
    )


def _fixed_noise(
    labels: Sequence[TopologyLabel],
    samples: int,
    n: int,
    z_dim: int,
    device: torch.device,
) -> torch.Tensor:
    rows = []
    for label in labels:
        for trajectory in range(samples):
            generator = torch.Generator(device="cpu").manual_seed(
                (label.sample_seed + 0x9E3779B97F4A7C15 * trajectory)
                % (2**63 - 1)
            )
            rows.append(torch.randn((n, z_dim), generator=generator))
    return torch.stack(rows).to(device=device, dtype=torch.float32)


def _trajectory_loss(
    model: torch.nn.Module,
    schedule: Any,
    records: Sequence[TrainingRecord],
    *,
    samples: int,
    steps: int,
    device: torch.device,
    grad_steps: int,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    batch = _batch(records, device)
    labels = [record.label for record in records]
    cond = engine.build_cond(batch, model.config)
    known_z, known_mask = engine.known_channels(batch)
    cond_k = expand_cond(cond, samples)
    known_z_k = known_z.repeat_interleave(samples, dim=0)
    known_mask_k = known_mask.repeat_interleave(samples, dim=0)
    batch_k = {
        key: value.repeat_interleave(samples, dim=0)
        for key, value in batch.items()
    }
    noise = _fixed_noise(
        labels,
        samples,
        int(batch["area"].shape[1]),
        int(model.config.z_dim),
        device,
    )
    anchor_seed = int.from_bytes(
        hashlib.sha256(
            b"\0".join(str(label.sample_seed).encode("ascii") for label in labels)
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)
    generator = torch.Generator(device=device).manual_seed(anchor_seed)
    z = sample_differentiable(
        model,
        cond_k,
        schedule,
        steps=steps,
        generator=generator,
        z_known=known_z_k,
        known_mask=known_mask_k,
        grad_steps=grad_steps,
        noise=noise,
    )
    rects = energy.decode_rects(
        z, batch_k["area"], batch_k["cons"], batch_k["tp"], batch_k["scale"]
    )
    expanded_labels = [label for label in labels for _ in range(samples)]
    sparse = collate_labels(expanded_labels, device, rects.dtype)
    losses = topology_losses(rects, sparse, batch_k["scale"])
    centres = rects[..., :2].view(len(records), samples, rects.shape[1], 2)
    mask = batch["area"].gt(0).unsqueeze(1).unsqueeze(-1)
    centred = torch.where(mask, centres, torch.zeros_like(centres))
    spread = centred.var(dim=1, unbiased=False).mean()
    return losses, spread


@torch.no_grad()
def _evaluate(
    model: torch.nn.Module,
    schedule: Any,
    records: Sequence[TrainingRecord],
    *,
    batch_size: int,
    samples: int,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"total": 0.0, "separation": 0.0, "contact": 0.0, "spread": 0.0}
    groups = 0
    for start in range(0, len(records), batch_size):
        group = records[start : start + batch_size]
        losses, spread = _trajectory_loss(
            model,
            schedule,
            group,
            samples=samples,
            steps=steps,
            device=device,
            grad_steps=0,
        )
        for key in ("total", "separation", "contact"):
            totals[key] += float(losses[key]) * len(group)
        totals["spread"] += float(spread) * len(group)
        groups += len(group)
    model.train()
    return {key: value / max(groups, 1) for key, value in totals.items()}


@torch.no_grad()
def _evaluate_ema(
    model: torch.nn.Module,
    ema: Any,
    schedule: Any,
    records: Sequence[TrainingRecord],
    *,
    batch_size: int,
    samples: int,
    steps: int,
    device: torch.device,
) -> dict[str, float]:
    restore = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    try:
        ema.copy_to(model)
        return _evaluate(
            model,
            schedule,
            records,
            batch_size=batch_size,
            samples=samples,
            steps=steps,
            device=device,
        )
    finally:
        model.load_state_dict(restore, strict=True)


def _checkpoint_payload(
    model: torch.nn.Module,
    ema: Any,
    model_config: Mapping[str, Any],
    *,
    step: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "ema": {key: value.detach().cpu() for key, value in ema.state_dict().items()},
        "model_config": dict(model_config),
        "step": step,
        "topology_prior_contract": dict(contract),
    }


def _preflight(
    checkpoint: Path,
    control_checkpoint: Path,
    flow_checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    identities = {
        "source_file_sha256": _sha256(checkpoint),
        "control_file_sha256": _sha256(control_checkpoint),
        "flow_file_sha256": _sha256(flow_checkpoint),
    }
    expected_files = {
        "source_file_sha256": SOURCE_FILE_SHA256,
        "control_file_sha256": CONTROL_FILE_SHA256,
        "flow_file_sha256": FLOW_FILE_SHA256,
    }
    if identities != expected_files:
        raise ValueError("checkpoint file identity")
    source = torch.load(checkpoint, map_location="cpu", weights_only=False)
    control = torch.load(control_checkpoint, map_location="cpu", weights_only=False)
    source_contract = verify_source_contract(source, control)
    if not source_contract["ok"]:
        raise ValueError("source contract")
    source_identity = canonical_checkpoint_identity(source)
    expected_identity = {
        "identity_schema": IDENTITY_SCHEMA,
        "model_config_sha256": EXPECTED_CONFIG_SHA256,
        "model_keyset_sha256": EXPECTED_KEYSET_SHA256,
        "ema_keyset_sha256": EXPECTED_KEYSET_SHA256,
        "ema_state_sha256": EXPECTED_STATE_SHA256,
    }
    if source_identity != expected_identity:
        raise ValueError("source canonical identity")
    if any(
        source_contract[key] != EXPECTED_STATE_SHA256
        for key in (
            "source_ema_sha256", "control_model_sha256", "control_ema_sha256"
        )
    ):
        raise ValueError("source canonical state")
    return source, control, {
        **identities,
        **source_contract,
        "source_identity": source_identity,
    }


def train(args: argparse.Namespace) -> dict[str, Any]:
    if args.sampler_steps != 2 or args.student_samples != 3:
        raise ValueError("sampler contract")
    if args.production_candidates != 6:
        raise ValueError("candidate contract")
    if args.max_steps <= 0 or args.batch <= 0 or args.eval_every <= 0:
        raise ValueError("training config")
    if args.aug_p2b_alpha < 0 or args.aug_b2b_alpha < 0 or not 0.0 <= args.aug_prob <= 1.0:
        raise ValueError("augmentation config")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda unavailable")
    out_dir = Path(args.out_dir)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise ValueError("nonempty output")
    out_dir.mkdir(parents=True, exist_ok=True)

    source, control, source_contract = _preflight(
        Path(args.checkpoint),
        Path(args.control_checkpoint),
        Path(args.flow_checkpoint),
    )
    source_contract.update(
        {
            "schema": "icdc_topology_source_contract_v1",
            "train_corpus_sha256": _sha256(Path(args.train_corpus)),
            "train_labels_sha256": _sha256(Path(args.train_labels)),
            "heldout_corpus_sha256": _sha256(Path(args.heldout_corpus)),
            "heldout_labels_sha256": _sha256(Path(args.heldout_labels)),
        }
    )
    _atomic_bytes(
        out_dir / "source_contract.json",
        _canonical_bytes(source_contract, newline=True),
    )

    train_records = load_paired_records(args.train_corpus, args.train_labels)
    heldout_records = load_paired_records(
        args.heldout_corpus,
        args.heldout_labels,
        selection_mod=args.heldout_selection_mod,
        selection_namespace="icdc-topology-heldout-selection-v1",
        max_records=args.heldout_max_records,
    )

    from direct_diffusion_model import DirectDenoiser, DirectModelConfig, EMA
    from diffusion_model import DiffusionSchedule

    config_fields = DirectModelConfig.__dataclass_fields__
    config_payload = dict(source["model_config"])
    config = DirectModelConfig(
        **{key: value for key, value in config_payload.items() if key in config_fields}
    )
    model = DirectDenoiser(config).to(device)
    model.load_state_dict(source["ema"], strict=True)
    ema = EMA(model, decay=args.ema_decay)
    ema.load_state_dict(source["ema"])
    # The shared EMA codec clones the loaded shadow on its source device (CPU).
    # `EMA.update` mutates the shadow in place against live model tensors, so the
    # shadow must live on the training device.  Device placement only; values,
    # dtypes and key order are unchanged (no-op when device is CPU).
    ema.shadow = {key: value.to(device) for key, value in ema.shadow.items()}
    base_anchor = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }
    schedule = DiffusionSchedule(config.timesteps, device=device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=0.0,
    )
    del source

    portfolio = frozen_portfolio_contract()
    contract = {
        "schema": "icdc_same_shape_topology_prior_v1",
        "sampler": "dpmpp",
        "sampler_steps": 2,
        "student_samples": 3,
        "production_candidates": 6,
        "portfolio_contract_sha256": portfolio["sha256"],
        "seed": args.seed,
        "train_records": len(train_records),
        "heldout_records": len(heldout_records),
        "heldout_selection_mod": args.heldout_selection_mod,
        "aug_p2b_alpha": args.aug_p2b_alpha,
        "aug_b2b_alpha": args.aug_b2b_alpha,
        "aug_prob": args.aug_prob,
        "source_contract_sha256": hashlib.sha256(
            _canonical_bytes(source_contract)
        ).hexdigest(),
    }
    _atomic_bytes(out_dir / "contract.json", _canonical_bytes(contract, newline=True))

    baseline = _evaluate_ema(
        model,
        ema,
        schedule,
        heldout_records,
        batch_size=args.batch,
        samples=3,
        steps=2,
        device=device,
    )
    best_loss = baseline["total"]
    best_step = 0
    log_path = out_dir / "train_log.jsonl"
    _atomic_checkpoint(
        out_dir / "best.pt",
        _checkpoint_payload(model, ema, config_payload, step=0, contract=contract),
    )

    rng = random.Random(args.seed)
    aug_enabled = args.aug_p2b_alpha > 0 or args.aug_b2b_alpha > 0
    aug_rng = random.Random(args.seed ^ 0xA55A5A5A_1CD0_1CD0)
    golden_cache = GoldenLayoutCache() if aug_enabled else None
    aug_steps = 0
    aug_records_total = 0
    order = list(range(len(train_records)))
    cursor = len(order)
    model.train()
    for step in range(1, args.max_steps + 1):
        if cursor + args.batch > len(order):
            rng.shuffle(order)
            cursor = 0
        indices = order[cursor : cursor + args.batch]
        cursor += len(indices)
        group = [train_records[index] for index in indices]
        step_augmented = 0
        if aug_enabled and aug_rng.random() < args.aug_prob:
            aug_group = []
            for record in group:
                augmented_case = augment_case(
                    record.case,
                    record.receipt,
                    golden_cache,
                    aug_rng,
                    p2b_alpha=args.aug_p2b_alpha,
                    b2b_alpha=args.aug_b2b_alpha,
                )
                if augmented_case is None:
                    aug_group.append(record)
                    continue
                aug_group.append(dataclasses.replace(record, case=augmented_case))
                step_augmented += 1
            group = aug_group
            if step_augmented:
                aug_steps += 1
                aug_records_total += step_augmented
        losses, spread = _trajectory_loss(
            model,
            schedule,
            group,
            samples=3,
            steps=2,
            device=device,
            grad_steps=2,
        )
        anchor = torch.zeros((), dtype=torch.float32, device=device)
        anchor_elements = 0
        for name, parameter in model.named_parameters():
            anchor = anchor + (parameter - base_anchor[name]).square().sum()
            anchor_elements += parameter.numel()
        anchor = anchor / max(anchor_elements, 1)
        loss = losses["total"] + args.anchor_weight * anchor
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip))
        optimizer.step()
        ema.update(model)

        if step == 1 or step % args.log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach()),
                "topology": float(losses["total"].detach()),
                "separation": float(losses["separation"].detach()),
                "contact": float(losses["contact"].detach()),
                "anchor": float(anchor.detach()),
                "spread": float(spread.detach()),
                "gradient": gradient,
                "aug_applied": bool(step_augmented) if aug_enabled else False,
                "aug_steps_so_far": aug_steps,
                "aug_records_so_far": aug_records_total,
            }
            with log_path.open("ab") as stream:
                stream.write(_canonical_bytes(row, newline=True))

        if step % args.eval_every == 0 or step == args.max_steps:
            heldout = _evaluate_ema(
                model,
                ema,
                schedule,
                heldout_records,
                batch_size=args.batch,
                samples=3,
                steps=2,
                device=device,
            )
            evaluation = {
                "step": step,
                "heldout": heldout,
                "baseline_heldout": baseline,
                "constraint_gain_fraction": (
                    1.0 - heldout["total"] / baseline["total"]
                    if baseline["total"] > 0
                    else 0.0
                ),
            }
            with log_path.open("ab") as stream:
                stream.write(_canonical_bytes(evaluation, newline=True))
            if math.isfinite(heldout["total"]) and heldout["total"] < best_loss:
                best_loss = heldout["total"]
                best_step = step
                _atomic_checkpoint(
                    out_dir / "best.pt",
                    _checkpoint_payload(
                        model, ema, config_payload, step=step, contract=contract
                    ),
                )
            _atomic_checkpoint(
                out_dir / "latest.pt",
                _checkpoint_payload(
                    model, ema, config_payload, step=step, contract=contract
                ),
            )

    candidate = torch.load(out_dir / "best.pt", map_location="cpu", weights_only=False)
    final_contract = checkpoint_contract(
        control,
        candidate,
        sampler_method="dpmpp",
        sampler_steps=2,
        candidate_count=6,
        portfolio_contract_sha256=portfolio["sha256"],
    )
    if not final_contract["ok"]:
        raise ValueError("candidate checkpoint contract")
    result = {
        "schema": "icdc_topology_training_result_v1",
        "best_step": best_step,
        "best_heldout_topology_loss": best_loss,
        "baseline_heldout_topology_loss": baseline["total"],
        "constraint_gain_fraction": (
            1.0 - best_loss / baseline["total"] if baseline["total"] > 0 else 0.0
        ),
        "candidate_sha256": _sha256(out_dir / "best.pt"),
        "checkpoint_contract": final_contract,
        "aug_p2b_alpha": args.aug_p2b_alpha,
        "aug_b2b_alpha": args.aug_b2b_alpha,
        "aug_prob": args.aug_prob,
        "aug_steps": aug_steps,
        "aug_records_total": aug_records_total,
    }
    _atomic_bytes(out_dir / "result.json", _canonical_bytes(result, newline=True))
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--control-checkpoint", required=True)
    parser.add_argument("--flow-checkpoint", required=True)
    parser.add_argument("--train-corpus", required=True)
    parser.add_argument("--train-labels", required=True)
    parser.add_argument("--heldout-corpus", required=True)
    parser.add_argument("--heldout-labels", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--sampler-steps", type=int, default=2)
    parser.add_argument("--student-samples", type=int, default=3)
    parser.add_argument("--production-candidates", type=int, default=6)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--anchor-weight", type=float, default=100.0)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--heldout-selection-mod", type=int, default=64)
    parser.add_argument("--heldout-max-records", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--aug-p2b-alpha", type=float, default=0.0,
        help="alpha_1-semantics P2B endpoint resampling strength; 0 disables it",
    )
    parser.add_argument(
        "--aug-b2b-alpha", type=float, default=0.0,
        help="p2b_1__b2b_1-semantics B2B endpoint resampling strength; 0 disables it",
    )
    parser.add_argument(
        "--aug-prob", type=float, default=0.5,
        help="fraction of training steps that see an augmented instance; "
        "heldout evaluation is always unaugmented",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    result = train(_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
