#!/bin/bash
# Apply a shipping variant to the working tree BEFORE running scripts/pack_cadc1013.sh.
# Usage: bash apply_pack_variant.sh [k20] [ft2] [s16]   (no args = restore package 0828b settings; s16 = FLOW_SLOTS 16 / NREF 12)
# Edits partner/shipping/op_wrapper.py (budget table string, FLOW_CKPT filename) and
# scripts/pack_cadc1013.sh (which flow checkpoint is copied). Idempotent; prints the resulting lines.
set -e
ROOT=/ldaphome/yyds-tsai-dev/Data-Driven-SoC-Floorplanning; cd $ROOT
W=partner/shipping/op_wrapper.py; P=scripts/pack_cadc1013.sh
MID=$(cat artifacts/p0_newbox/budget_table_mid.txt); K20=$(cat scratchpad/rtaware/budget_table_k20.txt)
FT1=flow_matching_ft0828_tailT24_300k_ema.pt; FT2=flow_matching_ft0829_tailT12_250k_ema.pt
TAB=$MID; CK=$FT1; SLOTS=10; NREF=9
for a in "$@"; do case $a in k20) TAB=$K20;; ft2) CK=$FT2;; s16) SLOTS=16; NREF=12;; *) echo "unknown arg $a"; exit 1;; esac; done
# budget table literal (line starting with _BUDGET_TABLE_MID = "...")
uv run python - "$W" "$TAB" "$CK" "$SLOTS" "$NREF" <<'PY'
import re,sys
p,tab,ck,slots,nref=sys.argv[1:]; s=open(p).read()
s=re.sub(r'"PARTNER_FLOW_SLOTS": "\d+"', '"PARTNER_FLOW_SLOTS": "%s"'%slots, s, count=1)
s=re.sub(r'"PARTNER_NREF": "\d+"', '"PARTNER_NREF": "%s"'%nref, s, count=1)
s2=re.sub(r'^_BUDGET_TABLE_MID = "[0-9.,]+"', '_BUDGET_TABLE_MID = "%s"'%tab, s, count=1, flags=re.M)
s2=re.sub(r'"FLOW_CKPT": str\(HERE / "checkpoints" / "[^"]+"\)', '"FLOW_CKPT": str(HERE / "checkpoints" / "%s")'%ck, s2, count=1)
assert s2.count(tab)==1 and ck in s2 and ('"PARTNER_FLOW_SLOTS": "%s"'%slots) in s2 and ('"PARTNER_NREF": "%s"'%nref) in s2; open(p,"w").write(s2); print("op_wrapper: table=%s... ckpt=%s slots=%s nref=%s"%(tab[-40:],ck,slots,nref))
PY
if [ "$CK" = "$FT2" ]; then
  [ -f submission/cadc1013/checkpoints/$FT2 ] || cp artifacts/flow_ft_0829/flow_ft0829_tailT12_lr1e-5_250k_ema.pt submission/cadc1013/checkpoints/$FT2
fi
sed -i -E "s#submission/cadc1013/checkpoints/flow_matching_ft08[0-9]+_[A-Za-z0-9_]+\.pt\" \"\\\$PK/checkpoints/flow_matching_ft08[0-9]+_[A-Za-z0-9_]+\.pt\"#submission/cadc1013/checkpoints/$CK\" \"\$PK/checkpoints/$CK\"#" $P
grep -n "FLOW_CKPT\"\|_BUDGET_TABLE_MID = \|PARTNER_FLOW_SLOTS\|\"PARTNER_NREF\"" $W | cut -c1-110; grep -n "checkpoints/flow_matching_ft" $P | cut -c1-160
