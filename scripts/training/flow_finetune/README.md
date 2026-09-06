# Flow-prior fine-tune (rounds 1-4, 2026-08-28 .. 08-30)

The shipped `FLOW_CKPT` is a tail-tilted fine-tune of the flow-matching v1 prior
(`submission/cadc1013/checkpoints/flow_matching_v1_final.pt`, 1M steps). The trainer
itself is the shipped module `src/solver/flow_matching_train.py`; `ft_launch.py` wraps it
with block-count-tilted file sampling (`FT_TAIL_TEMP`, `FT_N_INDEX`) and resumable
checkpointing. Status write-ups: `docs/experiments/flow-finetune/`.

| Round | Launcher | Init | Change | Result |
| --- | --- | --- | --- | --- |
| 1 (0828) | `run_ft_round1_0828_tailT24_300k.sh` | v1 | tilt T=24, lr 2e-5, 300k | FT1, shipped in 0828b package |
| 2 (0829) | `run_ft_round2_0829_tailT12_250k.sh` | round-1 EMA | tilt T=12, lr 1e-5, 250k | **FT2 = B' `FLOW_CKPT`** |
| 3 (0830) | `run_ft_round3_0830_x0_hpwl_150k.sh` | round-2 | x0 2.0 / hpwl 0.60, lr 5e-6 | FT3, rejected alone; soup a50 with FT2 = C' |
| 4 (0830b) | `run_ft_round4_0830b_bd_cg_mib_140k.sh` | round-2 | boundary/cluster 1.5, mib 0.5 | rejected (v3/v6 tail) |

Helpers: `build_file_n_index.py` (block-count index of the training files),
`make_seed_checkpoint_round1_from_v1.py` / `make_seed_checkpoint_continuation.py`
(turn a stripped EMA export into a resumable `latest.pt`), `export_ema_only.py`
(EMA-only shipping export), `export_gate_ckpt.py` (snapshot for gate runs).
Outputs: `artifacts/flow_finetune_round{1..4}_*/`.
