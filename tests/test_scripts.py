from pathlib import Path


def test_train_transformer_script_defaults_to_graph_transformer():
    script = Path("scripts/train_transformer.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENCODER="${ENCODER:-graph-transformer}"' in text
    assert 'NUM_HEADS="${NUM_HEADS:-8}"' in text
    assert 'CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn_transformer}"' in text
    assert '--encoder "$ENCODER"' in text
    assert '--num-heads "$NUM_HEADS"' in text
    assert 'train_arch_v5_transformer_${LOG_TAG}.log' in text


def test_train_hgt_script_defaults_to_hgt_encoder():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENCODER="${ENCODER:-hgt}"' in text
    assert 'NUM_HEADS="${NUM_HEADS:-4}"' in text
    assert 'CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn_hgt}"' in text
    assert 'HIGH_RISK_ORDER_MULTIPLIER="${HIGH_RISK_ORDER_MULTIPLIER:-1.5}"' in text
    assert 'HIGH_RISK_PAIRWISE_MULTIPLIER="${HIGH_RISK_PAIRWISE_MULTIPLIER:-1.5}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in text
    assert '_bs${BATCH_SIZE}' in text
    assert '--encoder "$ENCODER"' in text
    assert '--num-heads "$NUM_HEADS"' in text
    assert '--batch-size "$BATCH_SIZE"' in text
    assert '--high-risk-order-multiplier "$HIGH_RISK_ORDER_MULTIPLIER"' in text
    assert '--high-risk-pairwise-multiplier "$HIGH_RISK_PAIRWISE_MULTIPLIER"' in text
    assert 'train_arch_v5_hgt_${LOG_TAG}.log' in text


def test_train_hgt_script_exposes_pseudo_target_knobs():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENABLE_REPAIRED_PSEUDO_TARGETS="${ENABLE_REPAIRED_PSEUDO_TARGETS:-0}"' in text
    assert 'DIRTY_PSEUDO_ORDER_WEIGHT="${DIRTY_PSEUDO_ORDER_WEIGHT:-0.20}"' in text
    assert 'DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT="${DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT:-0.35}"' in text
    assert '--dirty-pseudo-order-weight "$DIRTY_PSEUDO_ORDER_WEIGHT"' in text
