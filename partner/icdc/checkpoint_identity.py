"""Canonical, fail-closed checkpoint identity codec."""
import hashlib, json, math
from collections.abc import Mapping
import torch

IDENTITY_SCHEMA = "icdc_canonical_state_v1"
_enc = lambda x: json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
def _config(c):
    if not isinstance(c, Mapping) or not c: raise ValueError("configuration")
    try: return _enc(dict(c))
    except Exception as e: raise ValueError("configuration") from e
def canonical_config_sha256(config): return hashlib.sha256(_config(config)).hexdigest()
def _state(s):
    if not isinstance(s, Mapping) or not s: raise ValueError("state")
    out=[]
    for k in sorted(s):
        if not isinstance(k,str) or not k or "\0" in k: raise ValueError("key")
        t=s[k]
        if not isinstance(t,torch.Tensor) or t.is_meta or t.is_quantized or t.layout is not torch.strided: raise ValueError("tensor")
        if not torch.is_floating_point(t) and not torch.is_complex(t):
            pass
        elif not bool(torch.isfinite(t).all()): raise ValueError("nonfinite")
        try: c=t.detach().cpu().contiguous(); dtype=str(c.dtype).removeprefix("torch."); shape=_enc(list(c.shape))
        except Exception as e: raise ValueError("tensor") from e
        h=k.encode()+b"\0"+dtype.encode()+b"\0"+shape+b"\0"
        out.append((h,c))
    return out
def canonical_keyset_sha256(state): return hashlib.sha256(b"".join(h for h,_ in _state(state))).hexdigest()
def canonical_state_sha256(state):
    chunks=[]
    for h,t in _state(state):
        try: chunks += [h,t.view(torch.uint8).numpy().tobytes()]
        except Exception as e: raise ValueError("tensor") from e
    return hashlib.sha256(b"".join(chunks)).hexdigest()
def canonical_checkpoint_identity(checkpoint):
    if not isinstance(checkpoint,Mapping): raise ValueError("checkpoint")
    model,ema=checkpoint.get("model"),checkpoint.get("ema")
    if not isinstance(model,Mapping) or not model or not isinstance(ema,Mapping) or not ema: raise ValueError("model/ema")
    mk=canonical_keyset_sha256(model); ek=canonical_keyset_sha256(ema)
    return {"identity_schema":IDENTITY_SCHEMA,"model_config_sha256":canonical_config_sha256(checkpoint.get("model_config")),"model_keyset_sha256":mk,"ema_keyset_sha256":ek,"ema_state_sha256":canonical_state_sha256(ema)}
