"""Why does resuming the conv-encoder arm die in foreach AdamW?

Reports the dtype/device of every optimizer-state tensor in the checkpoint,
grouped, so a mixed-dtype state shows up immediately.
"""
import sys
from collections import Counter

import torch

path = sys.argv[1]
ck = torch.load(path, map_location="cpu", weights_only=False)
print("top-level keys:", sorted(k for k in ck.keys())[:12])
print("global_step:", ck.get("global_step"))

opts = ck.get("optimizer_states", [])
print("num optimizers:", len(opts))
for oi, o in enumerate(opts):
    st = o.get("state", {})
    print(f"\n=== optimizer {oi}: {len(st)} params with state ===")
    per_key = {}
    for pid, s in st.items():
        for k, v in s.items():
            if torch.is_tensor(v):
                per_key.setdefault(k, Counter())[(str(v.dtype), str(v.device))] += 1
            else:
                per_key.setdefault(k, Counter())[(type(v).__name__, "nontensor")] += 1
    for k, c in per_key.items():
        print(f"  {k}: {dict(c)}")
    pg = o.get("param_groups", [])
    print(f"  param_groups: {len(pg)}")
    for gi, g in enumerate(pg):
        print(f"    group {gi}: {len(g.get('params', []))} params, lr={g.get('lr')}, "
              f"foreach={g.get('foreach')}, fused={g.get('fused')}, capturable={g.get('capturable')}")

sd = ck.get("state_dict", {})
dt = Counter(str(v.dtype) for v in sd.values() if torch.is_tensor(v))
print("\n=== model state_dict dtypes ===")
print(dict(dt))

fp32 = [k for k, v in sd.items() if torch.is_tensor(v) and v.dtype == torch.float32]
print(f"\n=== the {len(fp32)} float32 entries ===")
for k in fp32:
    print("   ", k, tuple(sd[k].shape))
# Which module prefixes do they belong to?
pref = Counter(".".join(k.split(".")[:3]) for k in fp32)
print("grouped by module prefix:", dict(pref))
