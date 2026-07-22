#!/usr/bin/env bash
# Paired retrieval gate.  This runner never builds or overwrites an index.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

uv run pytest tests/test_partner_retrieval_features.py \
  tests/test_partner_retrieval_index.py \
  tests/test_partner_retrieval_transfer.py \
  tests/test_partner_retrieval_scripts.py \
  tests/test_partner_retrieval_integration.py -q

INDEX="$ROOT/artifacts/retrieval/pilot64"
if [[ ! -d "$INDEX" ]]; then
  echo "retrieval gate requires existing train-only index: $INDEX" >&2
  exit 1
fi

PARTNER_DIRECT_MIN=2.5 PARTNER_RETRIEVAL_SLOTS=0 \
  bash scripts/partner_eval_cont.sh retrieval_direct_control

PARTNER_DIRECT_MIN=2.5 PARTNER_RETRIEVAL_INDEX="$INDEX" \
  PARTNER_RETRIEVAL_SLOTS=2 PARTNER_RETRIEVAL_MAX_COST=2.0 \
  bash scripts/partner_eval_cont.sh retrieval_r4_slots2
