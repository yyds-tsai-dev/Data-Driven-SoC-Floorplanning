# G0-v2 transient-fp formal result

## Decision

`TARGET_GAIN_MET`. The complete receipt-bound eligible heldout population
authorizes the approved same-shape student training stage. This result does not
authorize package work, G2, validation/test `fp_sol`, or any production policy
change beyond the separately frozen G1 comparison.

## Frozen run

- Source commit: `81a7d0f3e7abedc4f2a8352718ab213035d2a69b`
- Data: canonical `FloorSet/floorset_lite`, all 9,000 shards
- Population: `n >= 100` and `split_for_id(instance_id, 10) == "heldout"`
- Partitioning: 32 deterministic parts, run sequentially with one production
  optimizer process pinned to CPUs `0-31`; each part used 24 internal workers
- Production base: exactly 3 Direct DPM++/2 plus 3 Flow Euler/8,
  `PARTNER_NREF=6`, `PARTNER_OVERSAMPLE=1`
- Teacher: one transient receipt-bound training `fp_sol` seed converted
  `(w,h,x,y) -> (x,y,w,h)`, followed by exact TFDL, hard audit, and the pinned
  official no-runtime scorer
- First completed part: 2026-08-14 04:27:21 +0800
- Full merge completed: 2026-08-14 11:42:22 +0800

## Authorizing evidence

- Registered/results: `21081 / 21081`
- Teacher admitted: `21081`
- Positive gain: `20954`
- Weighted denominator: `228562167.24091178`
- `B_H = 1.325900364760382`
- `T_H = 1.140123142134694`
- `Delta_H = 0.18577722262568797`
- Target: `0.0261247299384228`
- Target multiple: approximately `7.11x`
- Population SHA256:
  `882945206ca0a365228cee1deffed81968d8ee81545c8ee354bab9beacfe62d9`
- Terminal state: `TARGET_GAIN_MET`

Final artifact hashes:

- `manifest.json`:
  `30902df28356748a0d83414eda193af9db2430274d6532a72b2e09b64d7b70dc`
- `population.json`:
  `8a4b2b57b63d2b0395de4e5f8951e042f9d590878b02fe271144d19c750c9dad`
- `cases.jsonl`:
  `85912e5b0e06b0934b33a3fa1c603dda4a79f6d372dfbe04b229ba4e8153260c`
- `labels.jsonl`:
  `b4ebef4dcf625ae8a3b8271ff762b5e849e72d3f0f97a49f2641e2d92ec4b7ea`

The immutable generated directory is
`artifacts/icdc_g0_v2_area_corpus/` and remains ignored by Git.

## Independent final checks

- 32/32 part manifests matched the hashes sealed by the merged manifest.
- Part registered counts summed exactly to 21,081.
- `cases.jsonl` and `labels.jsonl` each contained exactly 21,081 rows.
- Ordered case/label instance-ID streams had the same SHA256,
  `02c47b2d8c4559d1772b005131ff00897ca35d112d05546d7998c79c9695ea49`.
- Duplicate case IDs: 0; duplicate label IDs: 0; ordered mismatches: 0.
- Bad base or teacher hard audits: 0.
- Manifest support hashes matched the exact file bytes.
- The persisted tree contained none of the keys `fp_sol`, `tree_sol`,
  `metrics_sol`, `golden`, or `dense`.

## Authorized next action

Generate a deterministic receipt-bound training-split sparse-label corpus using
the same two-slot teacher semantics, keeping dense `fp_sol` transient. Train
and heldout-select a same-shape Direct student, then run only the frozen matched
3-Direct/3-Flow G1 causal experiment. No package review precedes that gate.
