"""M3/M6: pretraining. Masking (Eq. 6), channel views (Eq. 12), step loss (Eq. 13), 1:1 alternation loop,
checkpoints and logs (SPEC 1.5, 1.6, U26)."""
import json
import math
from pathlib import Path

import torch

from ibrain.model import SPIKE, masked_mse, masked_poisson_nll, simsiam_loss
from ibrain.repro import load_checkpoint, save_checkpoint, set_rng_state

LR, LR_MIN, WD, CLIP = 5e-4, 1e-5, 5e-2, 1.0  # SPEC 1.6


def sample_mask(valid, S):
    """M of Eq. 6, (B, C, S) bool. Per sample, exactly half of the valid tokens are drawn uniformly at random (U15).
    Padded channel tokens are never drawn."""
    B, C = valid.shape
    tok = valid[:, :, None].expand(B, C, S).reshape(B, C * S)
    rank = torch.rand(B, C * S, device=valid.device).masked_fill(~tok, 2.0).argsort(1).argsort(1)  # valid tokens first
    return (rank < tok.sum(1, keepdim=True) // 2).view(B, C, S)


def sample_views(valid):
    """Two channel views v1, v2 (B, C) bool (SPEC 1.5, U16). The n valid channels get a random rank:
    the first s are shared, the next u are view 1 only, the last u are view 2 only. u = round(0.2n), s = n - 2u."""
    n = valid.sum(1, keepdim=True)
    u = (n + 2) // 5  # integer arithmetic for round(0.2n)
    rank = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, 2.0).argsort(1).argsort(1)
    v1 = rank < n - u  # shared + view 1 only
    v2 = (rank < n - 2 * u) | ((rank >= n - u) & (rank < n))  # shared + view 2 only
    return v1, v2


def lr_at(step, steps, warmup):
    """SPEC 1.6, U20. Linear from 0 at step 0 to LR at step warmup,
    then cosine down to LR_MIN at the last step (steps - 1)."""
    if step < warmup:
        return LR * step / warmup
    t = min(1.0, (step - warmup) / max(1, steps - 1 - warmup))
    return LR_MIN + 0.5 * (LR - LR_MIN) * (1 + math.cos(math.pi * t))


def step_losses(model, x, valid, sig):
    """(L_rec, L_align) for one step (Eq. 13). sig = SPIKE: Eq. 10, 11. sig = IEEG: Eq. 8, 9.
    iEEG x is already channel-normalized by the Dataset (U4), so input and target are the same values."""
    mask = sample_mask(valid, x.shape[2])
    u = model.encode(x.masked_fill(mask[..., None], 0.0), valid, sig)  # Eq. 6: mask in raw signal space, then encode
    xhat = model.reconstruct(u, sig)
    rec = masked_poisson_nll(xhat, x, mask, valid) if sig == SPIKE else masked_mse(xhat, x, mask, valid)
    v1, v2 = sample_views(valid)  # views use the unmasked x, with the view mask in place of V_c (U17)
    q1, p1 = model.align(model.pool(model.encode(x, v1, sig), v1))
    q2, p2 = model.align(model.pool(model.encode(x, v2, sig), v2))
    return rec, simsiam_loss(p1, q2, p2, q1)  # Eq. 12


def pretrain(model, loaders, steps, warmup=2000, device="cpu", out_dir=None, ckpt_every=1000, resume=None, meta=None):
    """loaders = [(sig, DataLoader), ...], batches are collate's (x, valid).
    step t uses one batch from loaders[t % len(loaders)] (the 1:1 alternation of Eq. 13). Returns per-step records.
    If out_dir is set: one line per step in log.jsonl, ckpt.pt (latest) every ckpt_every steps, final.pt at the end.
    resume is a ckpt path: restores model, optimizer, step, records and RNG state. Loader order is reshuffled (U26)."""
    model.to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)  # U19
    start, hist = 0, []
    if resume:
        ck = load_checkpoint(resume, device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start, hist = ck["step"] + 1, ck["hist"]
        set_rng_state(ck["rng"])
    its = [iter(dl) for _, dl in loaders]
    log = None
    if out_dir:
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        log = open(Path(out_dir) / "log.jsonl", "a", encoding="utf-8")
    for t in range(start, steps):
        i = t % len(loaders)
        sig, dl = loaders[i]
        batch = next(its[i], None)
        if batch is None:  # end of a pass, reshuffle and cycle (U20)
            its[i] = iter(dl)
            batch = next(its[i])
        x, valid = batch[0].to(device), batch[1].to(device)
        lr = lr_at(t, steps, warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        rec, align = step_losses(model, x, valid, sig)
        opt.zero_grad()  # set_to_none: the inactive type's encoder/decoder/type_emb keep grad None, AdamW skips them (U24)
        (rec + align).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        hist.append({"step": t, "sig": sig, "lr": lr, "rec": rec.item(), "align": align.item()})
        if log:
            log.write(json.dumps(hist[-1]) + "\n")
            log.flush()
        if t % 100 == 0:
            print(hist[-1])
        if out_dir and ((t + 1) % ckpt_every == 0 or t == steps - 1):
            save_checkpoint(Path(out_dir) / "ckpt.pt", model, opt, t, hist, meta)
    if out_dir:
        save_checkpoint(Path(out_dir) / "final.pt", model, opt, steps - 1, hist, meta)
        log.close()
    return hist
