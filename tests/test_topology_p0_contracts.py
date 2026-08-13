import hashlib
import pytest
import torch
from icdc.topology_artifact_guard import assert_no_dense_fp_artifact, write_checked_json

def test_checked_artifacts_reject_dense_fp_but_keep_sparse_topology(tmp_path):
    sparse={"schema":"fp_topology_v1","version":1,"axis_edges":[{"src":0,"dst":1,"axis":0}],"contacts":[]}
    assert_no_dense_fp_artifact(sparse)
    digest=write_checked_json(tmp_path/"sparse.json",sparse)
    assert digest==hashlib.sha256((tmp_path/"sparse.json").read_bytes()).hexdigest()
    for bad in ({"fp_sol": [[1.,2.,3.,4.]]},{"rects": [[0.,0.,1.,1.]]},{"axis_edges":[],"origin":2.0}):
        with pytest.raises(ValueError,match="dense fp artifact"): assert_no_dense_fp_artifact(bad)

def test_raw_source_shape_contract():
    from icdc.topology_data import validate_raw_source
    src=[torch.zeros((1,2,6)),torch.zeros((1,1,3)),torch.zeros((1,0,3)),torch.zeros((1,0,2)),torch.zeros((1,1,3)),torch.ones((1,2,4)),torch.ones((1,8))]
    assert validate_raw_source(src)==(1,2)
