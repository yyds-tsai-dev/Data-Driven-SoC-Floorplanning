"""Leak-free sparse topology labels and differentiable topology losses."""
from __future__ import annotations

import math
from typing import Any, Dict, Mapping

import torch

from .topology_data import ContactLabel, SparseEdge, SparseTopologyBatch, TopologyLabel


def _validate_rects(rects: torch.Tensor) -> None:
    if not isinstance(rects, torch.Tensor) or not rects.is_floating_point() or rects.ndim != 3 or rects.shape[-1] != 4:
        raise ValueError("rects")
    if not torch.isfinite(rects).all() or (rects[..., 2:] <= 0).any():
        raise ValueError("rects values")


def _validate_batch(batch: SparseTopologyBatch, rects: torch.Tensor) -> None:
    if type(batch) is not SparseTopologyBatch:
        raise ValueError("batch")
    fields = list(batch.__dataclass_fields__)
    vals = [getattr(batch, n) for n in fields]
    if any(not isinstance(v, torch.Tensor) or v.ndim != 1 for v in vals):
        raise ValueError("batch fields")
    if any(v.device != rects.device for v in vals): raise ValueError("device")
    groups = [("edge", fields[:6]), ("contact", fields[6:])]
    for _, names in groups:
        lens = {getattr(batch, n).numel() for n in names}
        if len(lens) != 1:
            raise ValueError("batch lengths")
    long_names = {n for n in fields if n.endswith(("batch", "src", "dst", "axis", "a", "b", "order"))}
    float_names = set(fields) - long_names
    for n in long_names:
        if getattr(batch, n).dtype != torch.long:
            raise ValueError("integer field")
    for n in float_names:
        v = getattr(batch, n)
        if not v.is_floating_point() or v.dtype != rects.dtype:
            raise ValueError("float field")
        if not torch.isfinite(v).all() or (v < 0).any():
            raise ValueError("float values")
    b, n, _ = rects.shape
    for nme in ("edge_batch", "contact_batch"):
        v = getattr(batch, nme)
        if (v < 0).any() or (v >= b).any(): raise ValueError("batch index")
    for nme in ("edge_src", "edge_dst", "contact_a", "contact_b"):
        v = getattr(batch, nme)
        if (v < 0).any() or (v >= n).any(): raise ValueError("node index")
    for nme in ("edge_axis", "contact_axis"):
        if ((getattr(batch, nme) < 0) | (getattr(batch, nme) > 1)).any(): raise ValueError("axis")
    if ((batch.contact_order < 0) | (batch.contact_order > 1)).any(): raise ValueError("order")
    if (batch.edge_src == batch.edge_dst).any() or (batch.contact_a == batch.contact_b).any(): raise ValueError("self edge")


def topology_losses(rects: torch.Tensor, labels: SparseTopologyBatch, scale: torch.Tensor) -> Dict[str, torch.Tensor]:
    _validate_rects(rects)
    if not isinstance(scale, torch.Tensor) or not scale.is_floating_point() or scale.ndim != 1 or scale.shape[0] != rects.shape[0] or scale.device != rects.device or not torch.isfinite(scale).all().item() or (scale < 0).any():
        raise ValueError("scale")
    _validate_batch(labels, rects)
    x, y, w, h = rects.unbind(-1)
    def weighted(values, weights, batch):
        if not values.numel(): return rects.sum() * 0
        ww = weights * scale[batch]
        return (values * ww).sum() / ww.sum().clamp_min(1e-6)
    eb, es, ed, ea = labels.edge_batch, labels.edge_src, labels.edge_dst, labels.edge_axis
    src = torch.stack((x[eb, es], y[eb, es]), -1); dst = torch.stack((x[eb, ed], y[eb, ed]), -1)
    sz = torch.stack((w[eb, es], h[eb, es]), -1)
    gap = dst.gather(1, ea[:, None]).squeeze(1) - (src.gather(1, ea[:, None]).squeeze(1) + sz.gather(1, ea[:, None]).squeeze(1))
    sep = torch.relu((labels.edge_margin - gap) / scale[eb].clamp_min(1e-6))
    cb, ca, cc, cax = labels.contact_batch, labels.contact_a, labels.contact_b, labels.contact_axis
    aa = torch.stack((x[cb, ca], y[cb, ca]), -1); bb = torch.stack((x[cb, cc], y[cb, cc]), -1)
    aw = torch.stack((w[cb, ca], h[cb, ca]), -1); bw = torch.stack((w[cb, cc], h[cb, cc]), -1)
    av=aa.gather(1,cax[:,None]).squeeze(1); bv=bb.gather(1,cax[:,None]).squeeze(1)
    asz=aw.gather(1,cax[:,None]).squeeze(1); bsz=bw.gather(1,cax[:,None]).squeeze(1)
    ordered_gap=torch.where(labels.contact_order.bool(), bv-av-asz, av-bv-bsz)
    perp = 1 - cax
    overlap = torch.minimum((aa.gather(1, perp[:, None]).squeeze(1) + aw.gather(1, perp[:, None]).squeeze(1)), (bb.gather(1, perp[:, None]).squeeze(1) + bw.gather(1, perp[:, None]).squeeze(1))) - torch.maximum(aa.gather(1, perp[:, None]).squeeze(1), bb.gather(1, perp[:, None]).squeeze(1))
    contact = (ordered_gap.abs() + torch.relu(labels.contact_margin - overlap)) / scale[cb].clamp_min(1e-6)
    separation = weighted(sep, labels.edge_weight, eb); contact_mean = weighted(contact, labels.contact_weight, cb)
    return {"separation": separation, "contact": contact_mean, "total": separation + contact_mean}


def extract_sparse_label(legal: torch.Tensor, case: Mapping[str, Any], instance_id: str, sample_seed: int, teacher_cost: float, base_cost: float) -> TopologyLabel:
    if not isinstance(legal, torch.Tensor) or legal.device.type != "cpu" or legal.dtype != torch.float64 or legal.ndim != 2 or legal.shape[1] != 4: raise ValueError("legal")
    n = case.get("n")
    if type(n) is not int or n != legal.shape[0] or not isinstance(instance_id, str) or not instance_id.strip() or type(sample_seed) is not int or isinstance(sample_seed, bool): raise ValueError("metadata")
    if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) and float(v) > 0 for v in (teacher_cost, base_cost)) or teacher_cost > base_cost: raise ValueError("cost")
    if (legal[:, 2:] <= 0).any() or not torch.isfinite(legal).all(): raise ValueError("legal values")
    r = legal.tolist(); edges=[]
    for axis in (0,1):
        candidates=[]
        for i in range(n):
            for j in range(i+1,n):
                gx=max(r[i][0]-r[j][0]-r[j][2],r[j][0]-r[i][0]-r[i][2]); gy=max(r[i][1]-r[j][1]-r[j][3],r[j][1]-r[i][1]-r[i][3])
                if (axis==0 and gx < gy) or (axis==1 and gx >= gy): continue
                c_i=r[i][axis]+r[i][axis+2]/2; c_j=r[j][axis]+r[j][axis+2]/2; src,dst=(i,j) if (c_i,c_j)<(c_j,c_i) else (j,i)
                candidates.append((src,dst,max(0,r[dst][axis]-r[src][axis]-r[src][axis+2])))
        for e in candidates:
            if any(a==e[0] and b==e[1] for a,b,_ in candidates if (a,b)!=(e[0],e[1]) and False): continue
            redundant=False
            for first in [v for u,v,_ in candidates if u==e[0] and v!=e[1]]:
                reach={first}; changed=True
                while changed:
                    changed=False
                    for u,v,_ in candidates:
                        if u in reach and v not in reach: reach.add(v); changed=True
                if e[1] in reach: redundant=True; break
            if not redundant: edges.append(SparseEdge(e[0],e[1],axis,e[2],"sep",1.0))
    cons=case.get("cons", [[0,0]]*n)
    clusters={}
    for i,row in enumerate(cons):
        if len(row)>=4 and int(row[3])>0: clusters.setdefault(int(row[3]),[]).append(i)
    contacts=[]
    for members in clusters.values():
        if len(members)>1:
            chosen=[]; parent={i:i for i in members}
            def find(i):
                while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
                return i
            cand=[]
            for ii,a in enumerate(members):
                for b in members[ii+1:]:
                    for ax in (0,1):
                        perp=1-ax; end=r[a][ax]+r[a][ax+2]; gap=abs(end-r[b][ax])
                        if gap>1e-9 and abs((r[b][ax]+r[b][ax+2])-r[a][ax])>1e-9: continue
                        ov=min(r[a][perp]+r[a][perp+2],r[b][perp]+r[b][perp+2])-max(r[a][perp],r[b][perp])
                        if ov>0:
                            before=(r[a][ax]+r[a][ax+2]/2)<(r[b][ax]+r[b][ax+2]/2) or ((r[a][ax]+r[a][ax+2]/2)==(r[b][ax]+r[b][ax+2]/2) and a<b)
                            x,y=(a,b) if a<b else (b,a); cand.append((-ov,ax,x,y,int(before if x==a else not before),ov))
            for _,ax,a,b,order,ov in sorted(cand):
                if find(a)!=find(b): parent[find(a)]=find(b); contacts.append(ContactLabel(a,b,ax,bool(order),ov,1.0))
            if len({find(i) for i in members})!=1: raise ValueError("disconnected cluster")
    return TopologyLabel(instance_id,n,sample_seed,float(teacher_cost),float(base_cost),float(base_cost/teacher_cost),tuple(sorted(set(edges),key=lambda e:(e.axis,e.src,e.dst,e.kind))),tuple(sorted(contacts,key=lambda c:(c.axis,c.a,c.b))),())
