"""Did the private audio q/k/v actually train, or are they frozen copies?

modality_specific_qkv initialises audio_{q,k,v}_proj as exact copies of the text
projections, so the model starts identical to the shared-KVQ control. If the audio
copies never receive updates -- e.g. because they were created after the optimizer
captured its parameter list -- then audio would keep projecting with STALE weights
while the text ones keep moving. That is not a no-op: it would make the model
steadily WORSE than shared, which is exactly the symptom being investigated.
"""

import argparse

import torch

p = argparse.ArgumentParser()
p.add_argument("--ckpt", required=True)
a = p.parse_args()

sd = torch.load(a.ckpt, map_location="cpu", weights_only=False)
opt = sd.get("optimizer_states")
sd = sd.get("state_dict", sd)

pairs = sorted((k, k.replace("audio_", "")) for k in sd if "audio_" in k and k.endswith(".weight"))
print(f"audio projection tensors: {len(pairs)}")
if not pairs:
    raise SystemExit("none found -- the feature did not install in this checkpoint")

for ak, tk in pairs:
    if tk not in sd:
        print(f"  {ak}: no text counterpart {tk}")
        continue
    wa, wt = sd[ak].float(), sd[tk].float()
    diff = (wa - wt).abs().max().item()
    rel = (wa - wt).norm().item() / (wt.norm().item() + 1e-9)
    tag = "IDENTICAL (never trained!)" if diff == 0.0 else "diverged"
    print(f"  {ak.split('layers.')[-1]:40s} max|a-t|={diff:.3e} rel={rel:.4f}  {tag}")

if opt:
    n_params = sum(len(g.get("params", [])) for st in opt for g in st.get("param_groups", []))
    print(f"\noptimizer param entries: {n_params}")
print(f"total tensors in state_dict: {len(sd)}")
