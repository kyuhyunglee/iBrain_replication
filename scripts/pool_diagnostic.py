"""Does pooling erase what tells windows apart? No training: the same backbone tokens of real windows, compared before
pooling and after each pooling (collapse ablation, 2026-10-08). GPU: run through scripts/slurm/pool_diagnostic.sbatch.
For the model at initialization and for each checkpoint given, per signal type:
  tokens      up to 16 random valid tokens (channel x patch) per window, after the final LayerNorm
              cos_other  mean cosine between tokens of different windows
              between    share of the token variance that lies between window means (1 = tokens only differ by window)
              erank      effective rank of the centred token cloud
  mean        mean over valid tokens after the final LayerNorm (the decided pooling, U6 + U23)
  max         max over valid tokens after the final LayerNorm
  mean_preLN  mean over valid tokens before the final LayerNorm
              cos  mean cosine between the pooled vectors of different windows (1 = one point); erank as above"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch
import torch.nn.functional as F
import yaml

from check_collapse import ieeg_windows, spike_windows
from ibrain.data_spike import collate
from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import pack
from ibrain.repro import load_checkpoint


def erank(r):
    s = torch.linalg.svdvals(r - r.mean(0))
    p = s / s.sum()
    return torch.exp(-(p * p.clamp_min(1e-12).log()).sum()).item()


def mean_cos(r):
    rn = F.normalize(r, dim=-1)
    off = ~torch.eye(len(r), dtype=torch.bool)
    return (rn @ rn.T)[off].mean().item()


@torch.no_grad()
def parts(model, windows, sig, budget, device, per_window=16, seed=0):
    """Token sample with window ids, and the three pooled vectors per window."""
    x, valid = collate(windows)
    n = valid.sum(1)
    g = torch.Generator().manual_seed(seed)
    toks, ids = [], []
    pooled = {k: torch.zeros(len(windows), model.d) for k in ("mean", "max", "mean_preLN")}
    for idx in pack(n, budget):
        c = int(n[idx].max())
        xc, vc = x[idx, :c].to(device), valid[idx, :c].to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16):
            z = model.enc[sig](xc) + model.type_emb[sig] + model.time_emb
            for b in model.backbone.blocks:
                z = b(z, vc)
            u = model.backbone.norm(z)
        z, u = z.float(), u.float()
        w = vc[..., None, None].float()
        pooled["mean"][idx] = ((u * w).sum((1, 2)) / (w.sum((1, 2)) * u.shape[2])).cpu()
        pooled["mean_preLN"][idx] = ((z * w).sum((1, 2)) / (w.sum((1, 2)) * z.shape[2])).cpu()
        pooled["max"][idx] = u.masked_fill(~vc[:, :, None, None], float("-inf")).amax((1, 2)).cpu()
        for j, i in enumerate(idx.tolist()):
            t = u[j, :int(n[i])].reshape(-1, u.shape[-1]).cpu()
            pick = torch.randperm(len(t), generator=g)[:per_window]
            toks.append(t[pick])
            ids += [i] * len(pick)
    return torch.cat(toks), torch.tensor(ids), pooled


def token_stats(t, ids):
    tn = F.normalize(t, dim=-1)
    k = torch.randint(len(t), (20000, 2), generator=torch.Generator().manual_seed(0))
    k = k[ids[k[:, 0]] != ids[k[:, 1]]]
    cos_other = (tn[k[:, 0]] * tn[k[:, 1]]).sum(-1).mean().item()
    means = torch.zeros(int(ids.max()) + 1, t.shape[1]).index_add_(0, ids, t)
    counts = torch.bincount(ids, minlength=len(means)).clamp(min=1)[:, None]
    between = ((means / counts)[ids] - t.mean(0)).pow(2).sum() / (t - t.mean(0)).pow(2).sum()
    return {"cos_other": cos_other, "between": between.item(), "erank": erank(t)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="*", default=[], help="checkpoints to compare with the model at initialization")
    ap.add_argument("--config", default="configs/smoke.yaml", help="model and data of the initialized model")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--pile-files", type=int, default=4)
    ap.add_argument("--ieeg-hours", type=float, default=0.3)
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = yaml.safe_load(Path(a.config).read_text())
    torch.manual_seed(a.seed)
    models = [("init", IBrain(**cfg["model"]))]
    for path in a.ckpt:
        ck = load_checkpoint(path, "cpu")
        m = IBrain(**ck["meta"]["config"]["model"])
        m.load_state_dict(ck["model"])
        models.append((f"{path} (step {ck['step']})", m))
    data = {SPIKE: spike_windows(cfg["data"]["spike"], a.n, a.pile_files, a.seed),
            IEEG: ieeg_windows(cfg, a.n, a.ieeg_hours, a.seed)}
    for sig, name in ((SPIKE, "spike"), (IEEG, "iEEG")):
        for label, m in models:
            m.to(device).eval()
            t, ids, pooled = parts(m, data[sig], sig, a.budget, device, seed=a.seed)
            ts = token_stats(t, ids)
            print(f"{name:5s} {label}")
            print(f"    tokens      cos_other={ts['cos_other']:6.3f} between={ts['between']:6.3f} erank={ts['erank']:6.1f}")
            for k, r in pooled.items():
                print(f"    {k:11s} cos={mean_cos(r):6.3f} erank={erank(r):6.1f}")


if __name__ == "__main__":
    main()
