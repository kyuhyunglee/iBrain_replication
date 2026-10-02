"""M3 verification: mask, channel views, lr schedule, loss decrease on synthetic shards."""
import math

import torch
from torch.utils.data import DataLoader

from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, SPIKE, masked_poisson_nll
from ibrain.pretrain import LR, LR_MIN, lr_at, pretrain, sample_mask, sample_views, total_steps


def valid_of(ns, C):
    return torch.arange(C) < torch.tensor(ns)[:, None]


def test_mask_half_of_valid_tokens():
    torch.manual_seed(0)
    ns = [1, 3, 7, 10]
    valid = valid_of(ns, 10)
    mask = sample_mask(valid, 10)
    assert not mask[~valid].any()  # padding channels are never selected
    assert mask.sum((1, 2)).tolist() == [5 * n for n in ns]  # exactly 50% of valid tokens per sample (U15)
    freq = sample_mask(torch.ones(4000, 3, dtype=torch.bool), 10).float().mean(0)
    assert (freq - 0.5).abs().max() < 0.05  # positions are uniformly random
    # values at padding positions do not enter the loss
    x, xhat = torch.poisson(torch.ones(4, 10, 10, 5)), torch.randn(4, 10, 10, 5)
    x2, xhat2 = x.clone(), xhat.clone()
    x2[~valid], xhat2[~valid] = 99.0, 99.0
    assert torch.allclose(masked_poisson_nll(xhat, x, mask, valid), masked_poisson_nll(xhat2, x2, mask, valid))


def test_views():
    torch.manual_seed(0)
    ns = [1, 2, 3, 5, 37, 182]
    valid = valid_of(ns, 256)
    v1, v2 = sample_views(valid)
    for i, n in enumerate(ns):
        a, b = v1[i], v2[i]
        assert not (a & ~valid[i]).any() and not (b & ~valid[i]).any()  # valid channels only
        assert torch.equal(a | b, valid[i])  # union = all valid channels (U16)
        k, o = a.sum().item(), (a & b).sum().item()
        assert b.sum().item() == k >= min(2, n)
        if n >= 30:
            assert abs(k / n - 0.8) < 0.03 and abs(o / k - 0.75) < 0.03
    assert (v1[2].sum().item(), (v1[2] & v2[2]).sum().item()) == (2, 1)  # n = 3: configuration of the bottom panel of Fig. 2
    assert not torch.equal(sample_views(valid)[0], v1)  # resampled at every step


def test_view_mask_equals_channel_subset():
    """Encoding with a view mask = encoding only the view's channels sliced out (U17). Channels outside the view do not affect the view representation."""
    torch.manual_seed(0)
    m = IBrain().eval()
    x = torch.poisson(torch.full((1, 8, 10, 5), 2.0))
    v1, _ = sample_views(torch.ones(1, 8, dtype=torch.bool))
    xs = x[:, v1[0]]
    ones = torch.ones(1, xs.shape[1], dtype=torch.bool)
    assert torch.allclose(m.pool(m.encode(x, v1), v1), m.pool(m.encode(xs, ones), ones), atol=1e-5)


def test_lr_schedule():
    steps, W = 101, 10
    assert lr_at(0, steps, W) == 0.0
    assert math.isclose(lr_at(W, steps, W), LR)
    assert math.isclose(lr_at(steps - 1, steps, W), LR_MIN)
    lrs = [lr_at(t, steps, W) for t in range(steps)]
    assert all(a < b for a, b in zip(lrs[:W], lrs[1:W + 1]))  # increases during warmup
    assert all(a > b for a, b in zip(lrs[W:-1], lrs[W + 1:]))  # decreases afterwards


def test_pretrain_reduces_loss(tmp_path):
    write_synthetic(tmp_path, (3, 8, 20), n_windows=16)
    torch.manual_seed(0)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path, SYNTHETIC)), batch_size=8, shuffle=True, collate_fn=collate)
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)  # small model for CPU tests
    hist = pretrain(m, [(SPIKE, dl)], steps=40, warmup=5)  # 6 batches × about 7 passes, including loader cycling
    assert [h["lr"] for h in hist] == [lr_at(t, 40, 5) for t in range(40)]

    def avg(key, hs):
        return sum(h[key] for h in hs) / len(hs)

    print(f"\nrec {avg('rec', hist[:5]):.4f} -> {avg('rec', hist[-5:]):.4f}, "
          f"align {avg('align', hist[:5]):.4f} -> {avg('align', hist[-5:]):.4f}")
    assert avg("rec", hist[-5:]) < avg("rec", hist[:5])
    assert avg("rec", hist[-5:]) + avg("align", hist[-5:]) < avg("rec", hist[:5]) + avg("align", hist[:5])


def test_checkpoint_and_resume(tmp_path):
    """U26: log.jsonl, ckpt.pt, final.pt in out_dir. resume carries over the step, history, model, and optimizer."""
    import json
    from ibrain.repro import load_checkpoint
    write_synthetic(tmp_path / "data", (3, 8), n_windows=8)
    dl = DataLoader(SpikeWindows(read_windows(tmp_path / "data", SYNTHETIC)), batch_size=4, shuffle=True, collate_fn=collate)
    torch.manual_seed(0)
    m = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    out = tmp_path / "run"
    h1 = pretrain(m, [(SPIKE, dl)], steps=6, warmup=2, out_dir=out, ckpt_every=4, meta={"seed": 0})
    ck = load_checkpoint(out / "ckpt.pt")
    assert ck["step"] == 5 and len(ck["hist"]) == 6 and ck["meta"] == {"seed": 0}
    assert all(torch.equal(v, ck["model"][k]) for k, v in m.state_dict().items())
    assert len((out / "log.jsonl").read_text().splitlines()) == 6
    # again from the ckpt saved at 4 steps: after running 6 steps ckpt.pt is at step 5, resume from the ckpt before that (step 3)
    torch.manual_seed(0)
    m2 = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    out2 = tmp_path / "run2"
    pretrain(m2, [(SPIKE, dl)], steps=4, warmup=2, out_dir=out2, ckpt_every=100)
    m3 = IBrain(d=32, H=4, ffn=64, L=1, d_proj=16)
    h3 = pretrain(m3, [(SPIKE, dl)], steps=10, warmup=2, out_dir=out2, resume=out2 / "final.pt")
    assert [h["step"] for h in h3] == list(range(10)) and h3[4]["lr"] == lr_at(4, 10, 2)
    assert len((out2 / "log.jsonl").read_text().splitlines()) == 10
    assert json.loads((out2 / "log.jsonl").read_text().splitlines()[-1])["step"] == 9
    assert load_checkpoint(out2 / "final.pt")["step"] == 9


def test_total_steps_counts_spike_epochs():
    """U20: epochs are counted on the spike loader. Under 1:1 alternation spike still gets exactly epochs × batches."""
    assert total_steps(30, 7, 1) == 210  # spike-only: unchanged
    steps = total_steps(30, 7, 2)
    assert steps == 420
    assert sum(t % 2 == 0 for t in range(steps)) == 30 * 7  # loaders[t % 2], spike is index 0


def test_script_joint_step_count(tmp_path):
    """U20 at script level: with --spike-only the spike loader is seen `epochs` times; joint runs double the steps so
    spike is still seen exactly `epochs` times (the halving bug was in scripts/pretrain.py, not in total_steps).
    U33: with gradient accumulation each optimizer step consumes `accum` loader batches, and the exposure is unchanged."""
    import json
    import subprocess
    import sys
    from pathlib import Path
    import yaml
    root = Path(__file__).resolve().parent.parent
    cfg = yaml.safe_load((root / "configs/tiny.yaml").read_text())
    cfg["model"].update(d=16, H=2, ffn=32, L=1, d_proj=8)
    for accum in (1, 2):
        cfg["pretrain"].update(steps=None, epochs=2, warmup=1, ckpt_every=1000, grad_accum=accum)
        (tmp_path / "c.yaml").write_text(yaml.safe_dump(cfg))
        for extra, n_loaders in ((["--spike-only"], 1), ([], 2)):
            out = tmp_path / f"run{n_loaders}_a{accum}"
            r = subprocess.run([sys.executable, str(root / "scripts/pretrain.py"), "--config", str(tmp_path / "c.yaml"),
                                "--out", str(out), "--device", "cpu", *extra], capture_output=True, text=True, cwd=root)
            assert r.returncode == 0, r.stderr[-1500:]
            meta = json.loads((out / "meta.json").read_text())
            log = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
            spike_batches = meta["data"]["spike_windows"] // cfg["pretrain"]["batch_size"]
            assert meta["data"]["steps"] == len(log) == 2 * (spike_batches // accum) * n_loaders
            assert sum(h["sig"] == SPIKE for h in log) * accum == 2 * spike_batches  # spike sees exactly 2 epochs


def test_accumulation_draws_accum_batches_per_step(tmp_path):
    """U33: each optimizer step really draws `accum` batches of its type (counted at the dataset)."""
    write_synthetic(tmp_path, (3, 8), n_windows=8)
    ds = SpikeWindows(read_windows(tmp_path, SYNTHETIC))
    drawn = []

    class Counting(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            drawn.append(i)
            return ds[i]

    dl = DataLoader(Counting(), batch_size=4, shuffle=True, collate_fn=collate)
    torch.manual_seed(0)
    hist = pretrain(IBrain(d=16, H=2, ffn=32, L=1, d_proj=8), [(SPIKE, dl)], steps=5, warmup=1, accum=3)
    assert len(hist) == 5 and len(drawn) == 5 * 3 * 4
