"""M5 verification (synthetic): iEEG patches, encoder shape, channel normalization, MSE loss. M6 verification: gradient isolation between types under 1:1 alternation, both losses decrease."""
import copy

import numpy as np
import torch
from torch.utils.data import DataLoader

from ibrain.data_ieeg import SyntheticIEEG, channel_normalize, channel_stats, to_patches
from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, IEEG, SPIKE, masked_mse
from ibrain.pretrain import pretrain, sample_mask, step_losses


def ieeg_batch(n=3, channels=(4, 16, 64)):
    ds = SyntheticIEEG(channels, n_windows=1)
    return collate([ds[i] for i in range(n)])


def test_patches_and_encoder_shape():
    ds = SyntheticIEEG((4, 16), n_windows=2)
    x = to_patches(ds.windows[0])
    assert x.shape == (4, 10, 50) and x.dtype == torch.float32  # (C, S, P_iEEG), SPEC 1.1
    assert torch.equal(x.reshape(4, 500), torch.as_tensor(ds.windows[0], dtype=torch.float32))  # x[c, s, p] = wave[c, 50s + p]
    assert ds[0].shape == (4, 10, 50) and ds[0].dtype == torch.float32
    x, valid = ieeg_batch()
    assert x.shape == (3, 64, 10, 50) and valid.sum(1).tolist() == [4, 16, 64]
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)
    u = m.encode(x, valid, IEEG)
    assert u.shape == (3, 64, 10, 64)  # Eq. 5
    assert m.reconstruct(u, IEEG).shape == (3, 64, 10, 50)  # Eq. 7


def test_channel_normalize():
    """U4: statistics come from the whole recording, not from the window, so a window's values
    do not depend on which of its patches are masked and the window mean is not forced to 0."""
    ds = SyntheticIEEG((4, 16), n_windows=8)
    for i in range(2):
        rec = np.concatenate([w for w, j in zip(ds.windows, ds.session) if j == i], axis=1)
        z = np.concatenate([ds[k].reshape(len(rec), -1).numpy() for k in range(len(ds)) if ds.session[k] == i], axis=1)
        q25, q50, q75 = np.percentile(z, [25, 50, 75], axis=1)
        assert np.allclose(q50, 0, atol=1e-4) and np.allclose((q75 - q25) / 1.349, 1, atol=1e-3)  # per channel, per recording
        c, sc = channel_stats(rec)
        assert np.allclose(z, channel_normalize(rec, c, sc), atol=1e-4)  # same stats for every window of the recording
    assert not np.allclose(ds[0].numpy().mean((1, 2)), 0, atol=1e-2)  # not a per-window z-score
    x, valid = ieeg_batch()
    assert not x[~valid].any()  # padding channels stay 0


def test_masked_mse_ignores_padding_and_unmasked():
    torch.manual_seed(0)
    x, valid = ieeg_batch()
    mask = sample_mask(valid, 10)
    xhat = torch.randn_like(x)
    x2, xhat2 = x.clone(), xhat.clone()
    x2[~valid], xhat2[~valid] = 99.0, 99.0  # padding
    xhat2[~mask] = 99.0  # unmasked patches
    assert torch.allclose(masked_mse(xhat, x, mask, valid), masked_mse(xhat2, x2, mask, valid))
    # value matches the definition: mean of (1/P)||xhat - x||^2 over masked valid patches (Eq. 8, 9)
    per = ((xhat - x) ** 2).mean(-1)
    w = mask & valid[..., None]
    assert torch.isclose(masked_mse(xhat, x, mask, valid), per[w].mean())


def test_ieeg_step_losses_decrease():
    torch.manual_seed(0)
    x, valid = ieeg_batch(channels=(4, 8, 16))  # kept small because the conv adapter is expensive on CPU (U1)
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    first = None
    for i in range(25):
        rec, align = step_losses(m, x, valid, IEEG)
        opt.zero_grad()
        (rec + align).backward()
        opt.step()
        first = first or rec.item()
    print(f"\niEEG rec {first:.4f} -> {rec.item():.4f}")
    assert rec.item() < first < 1.5  # targets have robust std 1, so predicting 0 gives about 1 or less


def test_type_isolation():
    """M6: a step for one type does not change the other type's encoder, decoder, or type_emb (U24). Including weight decay."""
    torch.manual_seed(0)
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2, weight_decay=0.5)
    before = copy.deepcopy(m.state_dict())
    xs, vs = torch.poisson(torch.full((2, 5, 10, 5), 2.0)), torch.ones(2, 5, dtype=torch.bool)
    rec, align = step_losses(m, xs, vs, SPIKE)
    opt.zero_grad()
    (rec + align).backward()
    opt.step()
    for k, v in m.state_dict().items():
        same = torch.equal(v, before[k])
        if k.startswith(("enc.1.", "dec.1.")) or k == "type_emb.1":
            assert same, k  # iEEG side unchanged
        elif k.startswith(("enc.0.", "dec.0.", "backbone.")) or k == "type_emb.0":
            assert not same, k  # spike side and shared backbone are updated


def test_joint_pretrain_reduces_both(tmp_path):
    """M6: alternate the spike loader and the iEEG loader 1:1. Reconstruction losses of both types decrease."""
    write_synthetic(tmp_path, (3, 8, 20), n_windows=16)
    torch.manual_seed(0)
    dl_s = DataLoader(SpikeWindows(read_windows(tmp_path, SYNTHETIC)), batch_size=8, shuffle=True, collate_fn=collate)
    dl_i = DataLoader(SyntheticIEEG((4, 8, 16), n_windows=8), batch_size=8, shuffle=True, collate_fn=collate)
    m = IBrain(d=64, H=4, ffn=128, L=2, d_proj=32)
    hist = pretrain(m, [(SPIKE, dl_s), (IEEG, dl_i)], steps=40, warmup=5)
    assert [h["sig"] for h in hist[:4]] == [SPIKE, IEEG, SPIKE, IEEG]  # 1:1 alternation (Eq. 13)
    for sig in (SPIKE, IEEG):
        hs = [h["rec"] for h in hist if h["sig"] == sig]
        a, b = sum(hs[:5]) / 5, sum(hs[-5:]) / 5
        print(f"\nsig {sig} rec {a:.4f} -> {b:.4f}")
        assert b < a
