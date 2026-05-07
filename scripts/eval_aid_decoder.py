"""Quick evaluation of AidDecoder: L1/L2 metrics and mask ablation."""
import os, sys, numpy as np, torch
sys.path.insert(0, ".")
from scripts.aid_decoder import AidDecoder, prefix_mask, sample_tail_mask

CKPT = "ckpt/aid_decoder/055000.pth"
ZQ   = "data/aid_zq.npy"
ACT  = "data/aid_actions.npy"
N_EVAL = 2000
SEED   = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load checkpoint
ck = torch.load(CKPT, map_location="cpu", weights_only=False)
cfg = ck["cfg"]
decoder = AidDecoder(horizon=cfg["horizon"], action_dim=cfg["action_dim"],
                     K=cfg["K"], d_enc=cfg["d_enc"], d_model=cfg["d_model"],
                     num_layers=cfg["num_layers"], num_heads=cfg["num_heads"])
decoder.load_state_dict(ck["decoder_state_dict"])
decoder.to(device).eval()
K = cfg["K"]

# Load eval subset
rng = np.random.default_rng(SEED)
z_mm = np.load(ZQ, mmap_mode="r")
x_mm = np.load(ACT, mmap_mode="r")
N = len(z_mm)
idx = np.sort(rng.choice(N, N_EVAL, replace=False))

z_eval = torch.from_numpy(z_mm[idx].astype(np.float32)).to(device)  # (N_EVAL,K,d)
x_eval = torch.from_numpy(x_mm[idx].astype(np.float32)).to(device)  # (N_EVAL,H,A)

BS = 256
results = {}

def run_batched(zq, x, mask_fn):
    l1s, l2s = [], []
    for i in range(0, len(zq), BS):
        z = zq[i:i+BS]; gt = x[i:i+BS]
        B = z.shape[0]
        mask = mask_fn(B)
        with torch.no_grad():
            recon = decoder(z, mask)
        l1s.append(torch.mean(torch.abs(recon - gt)).item())
        l2s.append(torch.mean((recon - gt)**2).item())
    return np.mean(l1s), np.mean(l2s)

# Full K tokens
l1, l2 = run_batched(z_eval, x_eval,
    lambda B: torch.ones(B, K, dtype=torch.bool, device=device))
print(f"n=K (all {K} tokens):  L1={l1:.4f}  L2={l2:.4f}")
results["all_tokens"] = {"L1": l1, "L2": l2}

# Prefix n = 1, 4, 8, 16, 32
for n in [1, 4, 8, 16, K]:
    l1, l2 = run_batched(z_eval, x_eval,
        lambda B, n=n: prefix_mask(n, K, B, device))
    print(f"n={n:2d} prefix:            L1={l1:.4f}  L2={l2:.4f}")
    results[f"prefix_{n}"] = {"L1": l1, "L2": l2}

# No tokens (all masked)
l1, l2 = run_batched(z_eval, x_eval,
    lambda B: torch.zeros(B, K, dtype=torch.bool, device=device))
print(f"n=0 (no tokens):       L1={l1:.4f}  L2={l2:.4f}")
results["no_tokens"] = {"L1": l1, "L2": l2}

import json
out = {"ckpt": CKPT, "n_eval": N_EVAL, "seed": SEED, "metrics": results}
os.makedirs("outputs/aid_decoder_eval", exist_ok=True)
with open("outputs/aid_decoder_eval/eval_055000.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved: outputs/aid_decoder_eval/eval_055000.json")
