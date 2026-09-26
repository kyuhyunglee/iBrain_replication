"""M4: downstream velocity regression (SPEC 1.8, U8, U9, U11~U13, U27).
Three arms: finetune (full-parameter fine-tuning from a pretraining checkpoint) /
scratch (random init, same training) / ridge (linear regression on binned counts).
Labels are the mean behavior (velocity) per 100 ms patch of the window, (S, 2) (U9)."""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from ibrain.data_spike import P, S, collate, to_patches
from ibrain.model import SPIKE, IBrain
from ibrain.repro import load_checkpoint, seed_all


# ---------------- Labels and splits ----------------

def patch_labels(start, ts, vel, n_patches=S, patch_seconds=0.1):
    """U9: mean behavior per 100 ms patch from the window start (start, in s). ts (T,), vel (T, D) -> (S, D).
    Patches with no samples are NaN. Adds 1e-6 patch (0.1 us) so a boundary sample (e.g. start + 0.1)
    does not fall into the previous patch because of floating-point error."""
    idx = np.floor((np.asarray(ts, dtype=np.float64) - start) / patch_seconds + 1e-6).astype(int)
    keep = (idx >= 0) & (idx < n_patches)
    vel = np.asarray(vel, dtype=np.float64)
    sums = np.zeros((n_patches, vel.shape[1]))
    np.add.at(sums, idx[keep], vel[keep])
    n = np.bincount(idx[keep], minlength=n_patches)
    return sums / np.where(n > 0, n, np.nan)[:, None]


def read_nwb_velocity(path, name="hand_vel"):
    """processing['behavior'][name] of an NLB NWB (DANDI 000128) -> (ts (T,), vel (T, 2)).
    Not verified on real data: check in the M4 real-data stage once corpus access is available (SPEC M4)."""
    from pynwb import NWBHDF5IO
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        series = io.read().processing["behavior"][name]
        data = np.asarray(series.data[:], dtype=np.float64)
        if series.timestamps is not None:
            ts = np.asarray(series.timestamps[:], dtype=np.float64)
        else:
            ts = series.starting_time + np.arange(len(data)) / series.rate
    return ts, data


def attach_velocity(windows, ts, vel):
    """Attaches the patch label "vel" to corpus windows (start_seconds). Windows with empty (NaN) labels are dropped."""
    out = []
    for w in windows:
        y = patch_labels(w["start_seconds"], ts, vel)
        if not np.isnan(y).any():
            out.append({**w, "vel": y})
    return out


def split_trials(windows, frac=0.8, seed=0):
    """U11: random 80/20 split by trial (interval_id).
    seed is fixed independently of the run seed so that the three arms share the same split (M4)."""
    ids = sorted({w["interval_id"] for w in windows})
    np.random.default_rng(seed).shuffle(ids)
    train = set(ids[: int(round(frac * len(ids)))])
    return [w for w in windows if w["interval_id"] in train], [w for w in windows if w["interval_id"] not in train]


def synthetic_labeled(n_trials=200, n_units=30, seed=0):
    """M4 synthetic check: one 1 s window per trial. 2-D velocity = smooth random walk,
    unit firing rate = 0.5 softplus(v W + b) / bin, 20 ms Poisson counts. Velocity is an (almost) linear function
    of the firing rate, so ridge should reach a high R² and the model should also reach R² > 0."""
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 1.0, (2, n_units))
    b = rng.normal(0, 0.5, n_units)
    out = []
    for i in range(n_trials):
        v = np.cumsum(rng.normal(0, 0.3, (S * P, 2)), 0)  # (50, 2)
        rate = 0.5 * np.log1p(np.exp(v @ W + b))
        out.append({"counts": rng.poisson(rate).astype(np.uint32), "vel": v.reshape(S, P, 2).mean(1),
                    "interval_id": str(i), "start_seconds": float(i), "session_id": "synthetic"})
    return out


class LabeledWindows(Dataset):
    """Window dict ("counts" [50, C], "vel" [S, 2]) -> ((C, S, P), (S, 2))."""

    def __init__(self, windows):
        self.w = list(windows)
        assert all(not np.isnan(w["vel"]).any() for w in self.w), "NaN label: must be filtered out by attach_velocity"

    def __len__(self):
        return len(self.w)

    def __getitem__(self, i):
        return to_patches(self.w[i]["counts"]), torch.as_tensor(self.w[i]["vel"], dtype=torch.float32)


def collate_labeled(batch):
    x, valid = collate([b[0] for b in batch])
    return x, valid, torch.stack([b[1] for b in batch])


# ---------------- Task heads (U8) ----------------

class AttnPoolHead(nn.Module):
    """U8 default: per time patch, attention pooling over the channel axis (one learned query) -> Linear(d -> out).
    Padded channels are excluded from the softmax."""

    def __init__(self, d, C=None, out=2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(d) * 0.02)
        self.k = nn.Linear(d, d)
        self.out = nn.Linear(d, out)

    def forward(self, u, valid):  # (B, C, S, d), (B, C) -> (B, S, out)
        logit = (self.k(u) * self.q).sum(-1) / u.shape[-1] ** 0.5  # (B, C, S)
        a = logit.masked_fill(~valid[:, :, None], float("-inf")).softmax(1)
        return self.out((a[..., None] * u).sum(1))


class MeanPoolHead(nn.Module):
    """U8 alternative 1: per time patch, mean over valid channels -> Linear."""

    def __init__(self, d, C=None, out=2):
        super().__init__()
        self.out = nn.Linear(d, out)

    def forward(self, u, valid):
        w = valid[:, :, None, None].to(u.dtype)
        return self.out((u * w).sum(1) / w.sum(1))


class FlattenHead(nn.Module):
    """U8 alternative 2: flatten channels -> Linear(C*d -> out). Session-specific (fixed C). Padded channels are 0."""

    def __init__(self, d, C, out=2):
        super().__init__()
        self.out = nn.Linear(C * d, out)

    def forward(self, u, valid):
        B, C, S_, d = u.shape
        u = u * valid[:, :, None, None].to(u.dtype)
        return self.out(u.permute(0, 2, 1, 3).reshape(B, S_, C * d))


HEADS = {"attn": AttnPoolHead, "mean": MeanPoolHead, "flatten": FlattenHead}


class Regressor(nn.Module):
    """Task head on top of the pretrained model's encode (Eq. 5).
    If frozen, the backbone runs with no-grad in eval mode (no dropout)."""

    def __init__(self, model, head, frozen=False):
        super().__init__()
        self.model, self.head, self.frozen = model, head, frozen
        if frozen:
            model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, x, valid):
        if self.frozen:
            with torch.no_grad():
                u = self.model.encode(x, valid, SPIKE)
        else:
            u = self.model.encode(x, valid, SPIKE)
        return self.head(u, valid)


# ---------------- Training and evaluation ----------------

def r2(pred, y):
    """Mean of the per-output-dimension R² (U27, sklearn r2_score default). pred, y (N, D)."""
    ss_res = ((y - pred) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return float((1 - ss_res / ss_tot).mean())


def train_regressor(reg, dl, epochs, lr=1e-4, wd=5e-2, device="cpu"):
    """AdamW, MSE, clip 1.0, constant lr (U27). Returns the mean loss per epoch."""
    reg.to(device).train()
    opt = torch.optim.AdamW([p for p in reg.parameters() if p.requires_grad], lr=lr, weight_decay=wd)
    hist = []
    for _ in range(epochs):
        tot, n = 0.0, 0
        for x, valid, y in dl:
            x, valid, y = x.to(device), valid.to(device), y.to(device)
            loss = F.mse_loss(reg(x, valid), y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(reg.parameters(), 1.0)
            opt.step()
            tot, n = tot + loss.item() * len(x), n + len(x)
        hist.append(tot / n)
    return hist


@torch.no_grad()
def evaluate(reg, dl, device="cpu"):
    reg.eval()
    preds, ys = [], []
    for x, valid, y in dl:
        preds.append(reg(x.to(device), valid.to(device)).cpu())
        ys.append(y)
    D = ys[0].shape[-1]
    return r2(torch.cat(preds).reshape(-1, D), torch.cat(ys).reshape(-1, D))


def fit_arm(train, test, model_cfg, seed, ckpt=None, head="attn", frozen=False, epochs=20, lr=1e-4, wd=5e-2,
            batch_size=32, device="cpu"):
    """One arm, one seed: finetune if ckpt is given, else scratch. Returns (test R², per-epoch loss history)."""
    seed_all(seed)
    model = IBrain(**model_cfg)
    if ckpt:
        model.load_state_dict(load_checkpoint(ckpt, device)["model"])
    C = max(w["counts"].shape[1] for w in train + test)
    reg = Regressor(model, HEADS[head](model_cfg.get("d", 256), C), frozen)
    g = torch.Generator().manual_seed(seed)
    dl_tr = DataLoader(LabeledWindows(train), batch_size, shuffle=True, collate_fn=collate_labeled, generator=g)
    dl_te = DataLoader(LabeledWindows(test), batch_size, collate_fn=collate_labeled)
    hist = train_regressor(reg, dl_tr, epochs, lr, wd, device)
    return evaluate(reg, dl_te, device), hist


def ridge_r2(train, test, lams=(1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0), seed=0):
    """U13 baseline: binned counts of the patch (P*C) -> velocity, closed-form ridge. Features are standardized.
    λ is chosen by an inner validation that re-splits train 80/20 by trial, then refit on all of train (U27)."""
    def xy(ws):
        X = np.concatenate([np.asarray(w["counts"], np.float64).reshape(S, -1) for w in ws])  # (N*S, P*C)
        Y = np.concatenate([np.asarray(w["vel"], np.float64) for w in ws])  # (N*S, D)
        return X, Y

    def fit(X, Y, lam):
        mu, sd, ym = X.mean(0), X.std(0) + 1e-8, Y.mean(0)
        Xs = (X - mu) / sd
        W = np.linalg.solve(Xs.T @ Xs + lam * np.eye(X.shape[1]), Xs.T @ (Y - ym))
        return lambda Z: ((Z - mu) / sd) @ W + ym

    inner_tr, inner_va = split_trials(train, 0.8, seed)
    Xi, Yi = xy(inner_tr)
    Xv, Yv = xy(inner_va)
    lam = max(lams, key=lambda l: r2(fit(Xi, Yi, l)(Xv), Yv))
    Xtr, Ytr = xy(train)
    Xte, Yte = xy(test)
    return r2(fit(Xtr, Ytr, lam)(Xte), Yte), lam
