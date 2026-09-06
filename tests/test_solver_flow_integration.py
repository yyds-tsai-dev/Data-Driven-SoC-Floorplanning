import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "partner"))
CONTEST = Path(__file__).resolve().parents[1] / "FloorSet" / "iccad2026contest"
if str(CONTEST) not in sys.path:
    sys.path.insert(0, str(CONTEST))

from candidate_supply import allocate_quotas
from flow_matching_train import checkpoint_method


def test_flow_disabled_keeps_direct_capacity():
    assert allocate_quotas(12, {"direct": 12, "flow": 0}, ("direct",)) == {"direct": 12}


def test_flow_replaces_not_adds_capacity():
    got = allocate_quotas(12, {"direct": 6, "flow": 6}, ("direct", "flow"))
    assert got == {"direct": 6, "flow": 6}


def test_flow_loader_rejects_untagged_checkpoint():
    try:
        checkpoint_method({"args": {}})
    except ValueError:
        pass
    else:
        raise AssertionError("untagged checkpoint accepted")


def test_flow_env_defaults_off():
    assert os.environ.get("FLOW_CKPT") is None
    from contest_optimizer import MyOptimizer
    opt = MyOptimizer.__new__(MyOptimizer)      # no heavy init
    assert getattr(opt, "flow_model", None) is None
