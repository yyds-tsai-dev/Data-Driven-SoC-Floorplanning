from pathlib import Path


def test_train_transformer_script_defaults_to_graph_transformer():
    script = Path("scripts/train_transformer.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENCODER="${ENCODER:-graph-transformer}"' in text
    assert 'NUM_HEADS="${NUM_HEADS:-8}"' in text
    assert 'CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn_transformer}"' in text
    assert '--encoder "$ENCODER"' in text
    assert '--num-heads "$NUM_HEADS"' in text
    assert 'train_arch_v11_transformer_${LOG_TAG}.log' in text


def test_train_hgt_script_defaults_to_hgt_encoder():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENCODER="${ENCODER:-hgt}"' in text
    assert 'NUM_HEADS="${NUM_HEADS:-4}"' in text
    assert 'CHECKPOINT_PREFIX="${CHECKPOINT_PREFIX:-gnn_hgt}"' in text
    assert 'EPOCHS="${EPOCHS:-6}"' in text
    assert 'LR="${LR:-1.5e-4}"' in text
    assert 'ASPECT_WEIGHT="${ASPECT_WEIGHT:-0.06}"' in text
    assert 'HIGH_RISK_ORDER_MULTIPLIER="${HIGH_RISK_ORDER_MULTIPLIER:-1.8}"' in text
    assert 'HIGH_RISK_PAIRWISE_MULTIPLIER="${HIGH_RISK_PAIRWISE_MULTIPLIER:-1.6}"' in text
    assert 'HIGH_RISK_MIN_BLOCKS="${HIGH_RISK_MIN_BLOCKS:-100}"' in text
    assert 'HIGH_RISK_MIN_CONSTRAINTS="${HIGH_RISK_MIN_CONSTRAINTS:-10}"' in text
    assert 'HGT_RELATION_GATE_MIN="${HGT_RELATION_GATE_MIN:-0.20}"' in text
    assert 'BATCH_SIZE="${BATCH_SIZE:-8}"' in text
    assert '_bs${BATCH_SIZE}' in text
    assert '--encoder "$ENCODER"' in text
    assert '--num-heads "$NUM_HEADS"' in text
    assert '--batch-size "$BATCH_SIZE"' in text
    assert '--high-risk-order-multiplier "$HIGH_RISK_ORDER_MULTIPLIER"' in text
    assert '--high-risk-pairwise-multiplier "$HIGH_RISK_PAIRWISE_MULTIPLIER"' in text
    assert '--hgt-relation-gate-min "$HGT_RELATION_GATE_MIN"' in text
    assert 'train_arch_v11_hgt_${LOG_TAG}.log' in text


def test_train_diffusion_script_defaults_to_hgt_lite():
    script = Path("scripts/train_diffusion.sh")

    text = script.read_text(encoding="utf-8")

    assert 'NUM_SAMPLES="${NUM_SAMPLES:-800000}"' in text
    assert 'VARIANT="${VARIANT:-hgt_lite}"' in text
    assert 'HIDDEN_DIM="${HIDDEN_DIM:-128}"' in text
    assert 'LAYERS="${LAYERS:-2}"' in text
    assert 'DIFFUSION_STEPS="${DIFFUSION_STEPS:-1000}"' in text
    assert 'NOISE_SCHEDULE="${NOISE_SCHEDULE:-cosine}"' in text
    assert 'EMA_DECAY="${EMA_DECAY:-0.9999}"' in text
    assert 'WANDB_PROJECT="${WANDB_PROJECT:-floorset-v11-diffusion}"' in text
    assert 'uv run -m floorset_arch.training.train_diffusion' in text
    assert '--variant "$VARIANT"' in text
    assert '--max-diffusion-steps "$DIFFUSION_STEPS"' in text
    assert '--ema-decay "$EMA_DECAY"' in text
    assert 'train_arch_v11_diffusion_${LOG_TAG}.log' in text


def test_train_hgt_script_exposes_pseudo_target_knobs():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'ENABLE_REPAIRED_PSEUDO_TARGETS="${ENABLE_REPAIRED_PSEUDO_TARGETS:-1}"' in text
    assert 'PSEUDO_TARGET_CLEAN_ENOUGH_SOFT="${PSEUDO_TARGET_CLEAN_ENOUGH_SOFT:-0}"' in text
    assert 'DIRTY_PSEUDO_ORDER_WEIGHT="${DIRTY_PSEUDO_ORDER_WEIGHT:-0.20}"' in text
    assert 'DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT="${DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT:-0.35}"' in text
    assert 'PSEUDO_ARGS=()' in text
    assert 'if [ "$ENABLE_REPAIRED_PSEUDO_TARGETS" = "1" ]; then' in text
    assert 'PSEUDO_ARGS+=(--enable-repaired-pseudo-targets)' in text
    assert '--pseudo-target-clean-enough-soft "$PSEUDO_TARGET_CLEAN_ENOUGH_SOFT"' in text
    assert '--dirty-pseudo-order-weight "$DIRTY_PSEUDO_ORDER_WEIGHT"' in text
    assert '--dirty-pseudo-clean-enough-order-weight "$DIRTY_PSEUDO_CLEAN_ENOUGH_ORDER_WEIGHT"' in text
    assert '"${PSEUDO_ARGS[@]}"' in text


def test_train_hgt_script_exposes_training_self_eval_knobs():
    script = Path("scripts/train_hgt.sh")

    text = script.read_text(encoding="utf-8")

    assert 'TRAIN_EVALUATE_EACH_EPOCH="${TRAIN_EVALUATE_EACH_EPOCH:-1}"' in text
    assert 'TRAIN_EVAL_OUTPUT_DIR="${TRAIN_EVAL_OUTPUT_DIR:-}"' in text
    assert 'TRAIN_EVAL_TAIL_IDS="${TRAIN_EVAL_TAIL_IDS:-95,96,97,98,99}"' in text
    assert 'CHECKPOINT_METRICS_MANIFEST="${CHECKPOINT_METRICS_MANIFEST:-}"' in text
    assert 'EVALUATOR_BEST_CHECKPOINT="${EVALUATOR_BEST_CHECKPOINT:-}"' in text
    assert 'EXTRA_ARGS+=(--train-evaluate-each-epoch)' in text
    assert 'EXTRA_ARGS+=(--train-eval-output-dir "$TRAIN_EVAL_OUTPUT_DIR")' in text
    assert 'EXTRA_ARGS+=(--train-eval-tail-ids "$TRAIN_EVAL_TAIL_IDS")' in text


def test_eval_scripts_write_floorplan_pngs():
    single = Path("scripts/eval_single.sh").read_text(encoding="utf-8")
    total = Path("scripts/eval_total.sh").read_text(encoding="utf-8")

    assert "FLOORSET_EVAL_FLOORPLAN_DIR" in single
    assert "--floorplan-output-dir \"$FLOORPLAN_DIR\"" in single
    assert "FLOORSET_EVAL_FLOORPLAN_DIR" in total
    assert "--floorplan-output-dir \"$floorplan_dir\"" in total


def test_eval_scripts_expose_diffusion_checkpoint_ux():
    single = Path("scripts/eval_single.sh").read_text(encoding="utf-8")
    total = Path("scripts/eval_total.sh").read_text(encoding="utf-8")

    for text in (single, total):
        assert "--diffusion-checkpoint" in text
        assert "FLOORSET_DIFFUSION_CHECKPOINT" in text
        assert "FLOORSET_DIFFUSION_USE_EMA" in text
        assert "Using diffusion checkpoint:" in text
        assert "Using diffusion checkpoint state:" in text


def test_train_diffusion_script_exposes_training_self_eval_knobs():
    text = Path("scripts/train_diffusion.sh").read_text(encoding="utf-8")

    assert 'TRAIN_EVALUATE_EACH_EPOCH="${TRAIN_EVALUATE_EACH_EPOCH:-1}"' in text
    assert 'TRAIN_EVAL_OUTPUT_DIR="${TRAIN_EVAL_OUTPUT_DIR:-}"' in text
    assert 'TRAIN_EVAL_TAIL_IDS="${TRAIN_EVAL_TAIL_IDS:-95,96,97,98,99}"' in text
    assert 'CHECKPOINT_METRICS_MANIFEST="${CHECKPOINT_METRICS_MANIFEST:-}"' in text
    assert 'EVALUATOR_BEST_CHECKPOINT="${EVALUATOR_BEST_CHECKPOINT:-}"' in text
    assert 'EXTRA_ARGS+=(--train-evaluate-each-epoch)' in text
    assert 'EXTRA_ARGS+=(--checkpoint-metrics-manifest "$CHECKPOINT_METRICS_MANIFEST")' in text
    assert 'EXTRA_ARGS+=(--evaluator-best-checkpoint "$EVALUATOR_BEST_CHECKPOINT")' in text


def test_eval_total_uses_unique_default_floorplan_run_directory():
    text = Path("scripts/eval_total.sh").read_text(encoding="utf-8")

    assert "latest_total" not in text
    assert "FLOORSET_EVAL_RUN_ID" in text
    assert "date +%Y%m%d_%H%M%S" in text
    assert "total_${EVAL_RUN_ID}" in text


def test_diffusion_diagnostic_script_exposes_case_checkpoint_and_output_args():
    text = Path("scripts/diffusion_diagnostic.py").read_text(encoding="utf-8")

    assert "--test-id" in text
    assert "--checkpoint" in text
    assert "--output-dir" in text
    assert "plot_diffusion_diagnostic" in text


def test_wandb_defaults_use_v11_diffusion_project_name():
    expected = "floorset-v11-diffusion"
    paths = [
        Path("scripts/train.sh"),
        Path("scripts/train_transformer.sh"),
        Path("scripts/train_hgt.sh"),
        Path("scripts/train_diffusion.sh"),
        Path("src/floorset_arch/training/train.py"),
        Path("src/floorset_arch/training/train_diffusion.py"),
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert expected in text
        assert "floorset-arch-v5" not in text


def test_readme_wandb_examples_reference_v11_diffusion_project():
    text = Path("README.md").read_text(encoding="utf-8")
    assert "floorset-v11-diffusion" in text


def test_install_script_documents_fish_activation_command():
    text = Path("scripts/install.sh").read_text(encoding="utf-8")

    assert "source .venv/bin/activate.fish" in text
    assert "source .venv/bin/activate" in text
    assert "uv run <command>" in text
