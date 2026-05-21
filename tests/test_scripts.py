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
