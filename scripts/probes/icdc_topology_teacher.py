import argparse, hashlib, io, json, math, re, sys
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Mapping, Sequence
import torch
repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/"partner"))
from icdc.checkpoint_identity import *
__all__=["teacher_main"]
@dataclass(frozen=True)
class TeacherTrustPolicy:
    canonical_root: Path; expected_checkpoint_sha256: str; allowed_model_identity: Mapping[str,str]; expected_scorer_sha256: str; scorer_contract: str; shapely_version: str
def _checkpoint_identity_schema(): return IDENTITY_SCHEMA
def _checkpoint_identity(checkpoint): return canonical_checkpoint_identity(checkpoint)
def _load_verified_checkpoint_bytes(path, policy):
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=policy.expected_checkpoint_sha256: raise ValueError("checkpoint hash")
    try: payload=torch.load(io.BytesIO(b),weights_only=True,map_location="cpu")
    except Exception as e: raise ValueError("checkpoint load") from e
    actual=canonical_checkpoint_identity(payload)
    if actual["model_keyset_sha256"]!=actual["ema_keyset_sha256"] or dict(policy.allowed_model_identity)!=actual: raise ValueError("checkpoint identity")
    return payload,{**actual,"checkpoint_sha256":policy.expected_checkpoint_sha256}
def _sample_seed(seed,case_id,ordinal):
    return int.from_bytes(hashlib.sha256(f"{seed}\0{case_id}\0{ordinal}".encode()).digest()[:8],"big")%(2**63)
def _proposal_intent_holds(intent,rects,case):
    try:
        if intent=="base": return rects.ndim==2 and tuple(rects.shape[1:])==(4,) and bool(torch.isfinite(rects).all()) and bool((rects[:,2:]>0).all())
        p=intent.split(":"); kind=p[0]
        if kind in ("axis", "pin") and len(p)!=5: return False
        if kind == "contact" and len(p)!=6: return False
        if kind not in ("axis", "pin", "contact"): return False
        vals=list(map(int,p[1:])); a,b,axis,order=vals[-4:]
        n=int(case["n"])
        if not (0<=a<n and 0<=b<n and a!=b and axis in (0,1) and order in (0,1)): return False
        r=rects; cons=torch.as_tensor(case.get("cons"));
        if kind=="axis":
            from icdc.topology_prior import _pair_state
            return _pair_state(r,a,b)==(axis,order)
        if kind=="pin":
            tp=torch.as_tensor(case.get("tp")); return bool(cons[a,1]!=0 and torch.equal(r[a,:2],tp[a,:2])) and _proposal_intent_holds(f"axis:{a}:{b}:{axis}:{order}",r,case)
        gid=vals[0]; a,b,axis,order=vals[1:]; return bool(cons[a,3]==gid>0 and cons[b,3]==gid>0) and __import__('icdc.topology_prior',fromlist=['has_exact_positive_contact']).has_exact_positive_contact(r,a,b,axis,bool(order),1e-12)
    except Exception:return False
def _select_official_winner(rows):
    if not isinstance(rows,Sequence) or not rows: raise ValueError("rows")
    names=[r.get("name") for r in rows]
    if names.count("base")!=1 or len(set(names))!=len(names): raise ValueError("names")
    for r in rows:
        if type(r.get("ordinal")) is not int or r.get("feasible") is not True or isinstance(r.get("cost_no_runtime"),bool) or not math.isfinite(float(r["cost_no_runtime"])): raise ValueError("row")
    return min(range(len(rows)),key=lambda i:(float(rows[i]["cost_no_runtime"]),rows[i]["ordinal"],rows[i]["name"]))
def _weighted_population(rows):
    if any("weight" in r for r in rows): raise ValueError("weight")
    rs=sorted(rows,key=lambda r:(r["relative_path"],r["layout_index"])); seen=set(); out=[]
    for r in rs:
        if r["instance_id"] in seen or type(r["n"]) is not int or r["n"]<0 or not all(math.isfinite(float(r[k])) for k in ("base_cost","teacher_cost")): raise ValueError("population")
        seen.add(r["instance_id"]); out.append({**r,"w":math.exp(r["n"]/12)})
    den=sum(r["w"] for r in out); b=sum(r["w"]*r["base_cost"] for r in out)/den; t=sum(r["w"]*r["teacher_cost"] for r in out)/den
    raw=json.dumps(out,sort_keys=True,separators=(",",":"),allow_nan=False).encode(); return {"denominator":den,"B_H":b,"T_H":t,"Delta_H":b-t,"population_sha256":hashlib.sha256(raw).hexdigest()}
def _manifest_self_sha256(m): return hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!="self_sha256"},sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()
def _g0_state(c):
    if not all(c.get(k) is True for k in ("trust_ok","scorer_ok","input_ok")): return "KILLED_INPUT_CHECKPOINT_OR_SCORER"
    if not c.get("legal") or not c.get("coverage"): return "KILLED_LEGALITY_OR_COVERAGE"
    tm=float(c.get("teacher_mean",0));
    if not math.isfinite(tm): raise ValueError("teacher_mean")
    if tm>1.5:return "KILLED_TEACHER_GT_1_5"
    d=float(c.get("delta"));
    if d<.0181504738793652:return "STOP_HARD_GAIN_MISSED"
    return "TARGET_GAIN_MET" if d>=.0261247299384228 else "TARGET_GAIN_MISSED_NO_TRAINING_AUTHORITY"
def teacher_main(argv=None,*,_trust_policy=None):
    p=argparse.ArgumentParser(); p.add_argument('--data-root',required=True); p.add_argument('--out-dir',required=True); p.add_argument('--index-out',required=True); p.add_argument('--checkpoint',required=True); a=p.parse_args(argv)
    root=Path(a.data_root); out=Path(a.out_dir); idx=Path(a.index_out)
    if root.is_symlink() or not root.is_dir() or root.resolve()!=root: raise ValueError("canonical provenance root")
    if idx!=out/"training_index.json" or idx.exists(): raise ValueError("index-out")
    raise RuntimeError("teacher pipeline not implemented")
if __name__=='__main__': teacher_main()
