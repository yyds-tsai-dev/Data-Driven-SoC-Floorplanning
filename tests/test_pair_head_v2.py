"""Pair-head v2 tests.

Verifies the order-faithful 4-class label helper matches the decoder's exact
``build_order_dags`` rule bit-for-bit, that the loss/accuracy plumbing works,
and that the v2 model produces both the 4-class output and a backward-compatible
2-logit ``pair_logits`` contract that the live relative_order.py consumer reads.
"""

from __future__ import annotations

import math
import random

import torch

from floorset_arch.nn.model import FloorplanGNN
from floorset_arch.training.losses import (
    build_order_axis4_targets,
    order_axis4_loss,
    pair_edge_weights,
)


# ---------------------------------------------------------------------------
# Reference: the EXACT per-pair class the decoder derives from build_order_dags
# (scripts/probes/gen_decoder_probe.py). Copied verbatim so the test is the
# contract: any drift in the label helper fails here.
#   class 0 = i left-of j (x, dx>0), 1 = j left-of i (x, dx<0),
#   class 2 = i below j   (y, dy>0), 3 = j below i   (y, dy<0)
# ---------------------------------------------------------------------------
def _decoder_class(hints, i, j):
    cx = [h[0] + h[2] / 2.0 for h in hints]
    cy = [h[1] + h[3] / 2.0 for h in hints]
    ws = [h[2] for h in hints]
    hs = [h[3] for h in hints]
    dx = cx[j] - cx[i]
    dy = cy[j] - cy[i]
    hxi, hxj = ws[i] / 2.0, ws[j] / 2.0
    hyi, hyj = hs[i] / 2.0, hs[j] / 2.0
    sep_x = abs(dx) / (hxi + hxj) if (hxi + hxj) > 0 else 0.0
    sep_y = abs(dy) / (hyi + hyj) if (hyi + hyj) > 0 else 0.0
    use_x = sep_x >= sep_y  # tie -> x-axis (deterministic)
    if use_x:
        return 0 if (dx > 0 or (dx == 0 and i < j)) else 1
    return 2 if (dy > 0 or (dy == 0 and i < j)) else 3


def _random_layout(n, rng):
    """Random non-degenerate (x, y, w, h) rects -> fp_sol rows (w, h, x, y)."""
    hints = []
    fp_rows = []
    for _ in range(n):
        w = rng.uniform(0.5, 4.0)
        h = rng.uniform(0.5, 4.0)
        x = rng.uniform(0.0, 20.0)
        y = rng.uniform(0.0, 20.0)
        hints.append((x, y, w, h))
        fp_rows.append((w, h, x, y))  # fp_sol column order
    return hints, torch.tensor(fp_rows, dtype=torch.float32)


def test_axis4_labels_match_decoder_rule_exactly():
    rng = random.Random(20260705)
    total_mismatch = 0
    total_pairs = 0
    for _ in range(40):
        n = rng.randint(3, 25)
        hints, fp_sol = _random_layout(n, rng)
        pairs = torch.tensor(
            [(i, j) for i in range(n) for j in range(i + 1, n)], dtype=torch.long
        )
        got = build_order_axis4_targets(fp_sol, pairs)["label"].tolist()
        want = [_decoder_class(hints, int(i), int(j)) for i, j in pairs.tolist()]
        for g, w in zip(got, want, strict=True):
            total_pairs += 1
            if g != w:
                total_mismatch += 1
    assert total_pairs > 0
    assert total_mismatch == 0, f"{total_mismatch}/{total_pairs} label mismatches vs decoder"


def test_axis4_labels_tie_breaks_to_x():
    # Equal normalized separation on both axes -> tie -> x-axis (class 0/1).
    # Same-size blocks placed on the perfect diagonal: sep_x == sep_y.
    fp_sol = torch.tensor(
        [
            [2.0, 2.0, 0.0, 0.0],   # block 0 center (1,1)
            [2.0, 2.0, 3.0, 3.0],   # block 1 center (4,4): dx=dy=3, equal halves
        ],
        dtype=torch.float32,
    )
    pairs = torch.tensor([(0, 1)], dtype=torch.long)
    label = build_order_axis4_targets(fp_sol, pairs)["label"]
    assert int(label[0]) == 0  # x-axis, i-left-of-j


def test_axis4_loss_and_accuracy_plumbing():
    fp_sol = torch.tensor(
        [[1.0, 1.0, 0.0, 0.0], [1.0, 1.0, 5.0, 0.0], [1.0, 1.0, 0.0, 5.0]],
        dtype=torch.float32,
    )
    pairs = torch.tensor([(0, 1), (0, 2), (1, 2)], dtype=torch.long)
    targets = build_order_axis4_targets(fp_sol, pairs)
    # Perfect logits -> accuracy 1.0 and near-zero loss.
    logits = torch.full((3, 4), -10.0)
    for row, cls in enumerate(targets["label"].tolist()):
        logits[row, cls] = 10.0
    loss, comp_acc, axis_acc = order_axis4_loss(logits, targets)
    assert comp_acc == 1.0
    assert axis_acc == 1.0
    assert float(loss) < 1e-3


def test_pair_edge_weights_reflect_netlist():
    pairs = torch.tensor([(0, 1), (0, 2)], dtype=torch.long)
    valid_b2b = torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float32)  # edge 0-1, w=4
    w = pair_edge_weights(pairs, valid_b2b, block_count=3, alpha=0.5)
    assert w[0] > 1.0  # connected pair up-weighted
    assert float(w[1]) == 1.0  # unconnected pair unchanged
    # alpha<=0 disables.
    w0 = pair_edge_weights(pairs, valid_b2b, block_count=3, alpha=0.0)
    assert torch.allclose(w0, torch.ones(2))


def test_v2_model_emits_axis4_and_legacy_pair_logits():
    torch.manual_seed(0)
    n, feat = 6, 5
    model = FloorplanGNN(node_feat_dim=feat, hidden_dim=16, num_layers=2, pair_head_version=2)
    model.eval()
    node_feat = torch.randn(n, feat)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.empty((0, 1), dtype=torch.float32)
    pairs = torch.tensor([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=torch.long)
    with torch.no_grad():
        out = model(node_feat, edge_index, edge_attr, pairs=pairs)
    assert "pair_axis4_logits" in out
    assert out["pair_axis4_logits"].shape == (pairs.shape[0], 4)
    # Legacy 2-logit contract still present for relative_order.py.
    assert out["pair_logits"].shape == (pairs.shape[0], 2)
    x_logit, y_logit = out["pair_logits"][0].tolist()
    assert math.isfinite(x_logit) and math.isfinite(y_logit)


def test_v1_model_unchanged_2logit():
    torch.manual_seed(0)
    model = FloorplanGNN(node_feat_dim=5, hidden_dim=16, num_layers=2)  # default v1
    assert model.pair_head_v2 is None
    pairs = torch.tensor([(0, 1), (0, 2)], dtype=torch.long)
    with torch.no_grad():
        out = model(torch.randn(3, 5), torch.empty((2, 0), dtype=torch.long),
                    torch.empty((0, 1)), pairs=pairs)
    assert out["pair_logits"].shape == (2, 2)
    assert "pair_axis4_logits" not in out


def test_v2_checkpoint_roundtrip_loads_through_probe_style():
    """Save a v2 model's payload and reload it the way _probe_hints does."""
    from floorset_arch.training.checkpoint import anchor_checkpoint_payload
    from types import SimpleNamespace

    model = FloorplanGNN(node_feat_dim=5, hidden_dim=16, num_layers=2, pair_head_version=2)
    args = SimpleNamespace(num_samples=1, epochs=1, hidden_dim=16, layers=2,
                           accumulation_steps=1, encoder="mpnn", selection_metric="x")
    payload = anchor_checkpoint_payload(model, args, epoch=1, train_stats={}, val_stats={})
    assert payload["has_pair_head"] is True
    assert payload["pair_head_version"] == 2
    assert payload["pair_head_classes"] == 4
    # Reconstruct like GnnHintProvider / optimizer loader.
    reloaded = FloorplanGNN(
        node_feat_dim=int(payload["node_feat_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
        num_layers=int(payload["layers"]),
        pair_head_version=int(payload.get("pair_head_version", 1)),
    )
    missing, unexpected = reloaded.load_state_dict(payload["model_state_dict"], strict=False)
    assert not unexpected  # v2 loader must accept every saved key
    assert reloaded.pair_head_v2 is not None
