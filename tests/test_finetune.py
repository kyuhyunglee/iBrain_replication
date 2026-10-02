"""M4 verification (synthetic): patch labels, trial split, head shape and padding invariance, ridge baseline, scratch regression R² > 0, frozen."""
import numpy as np
import torch

from ibrain.data_nwb import attach_velocity, patch_labels
from ibrain.finetune import HEADS, Regressor, fit_arm, r2, ridge_r2, split_trials, synthetic_labeled
from ibrain.model import IBrain

TINY = dict(d=32, H=4, ffn=64, L=1, d_proj=16)


def test_patch_labels():
    ts = 10.0 + np.arange(0, 1.2, 0.001)  # 1 kHz, window [10, 11) + tail
    vel = np.stack([ts, -2 * ts], 1)
    y = patch_labels(10.0, ts, vel)
    assert y.shape == (10, 2)
    centers = 10.0 + 0.1 * np.arange(10) + 0.0495  # mean of 100 samples
    assert np.allclose(y[:, 0], centers, atol=1e-9) and np.allclose(y[:, 1], -2 * centers)
    assert np.isnan(patch_labels(10.0, ts[:500], vel[:500])[5:]).all()  # patches with no samples are NaN
    w = attach_velocity([{"start_seconds": 10.0}, {"start_seconds": 10.6}], ts, vel)
    assert len(w) == 1 and w[0]["start_seconds"] == 10.0  # NaN windows are dropped


def test_split_trials_fixed_and_disjoint():
    ws = synthetic_labeled(50, 5)
    a, b = split_trials(ws, 0.8, seed=0)
    assert len(a) == 40 and len(b) == 10
    assert not {w["interval_id"] for w in a} & {w["interval_id"] for w in b}
    a2, _ = split_trials(ws, 0.8, seed=0)
    assert [w["interval_id"] for w in a2] == [w["interval_id"] for w in a]  # same seed gives the same split (M4)
    assert [w["interval_id"] for w in split_trials(ws, 0.8, seed=1)[0]] != [w["interval_id"] for w in a]


def test_heads_shape_and_padding():
    torch.manual_seed(0)
    m = IBrain(**TINY).eval()
    x = torch.poisson(torch.full((2, 6, 10, 5), 2.0))
    valid = torch.ones(2, 6, dtype=torch.bool)
    valid[:, 4:] = False
    for name, H in HEADS.items():
        reg = Regressor(m, H(TINY["d"], C=6)).eval()
        y = reg(x, valid)
        assert y.shape == (2, 10, 2), name
        x2 = x.clone()
        x2[:, 4:] = 7.0
        assert torch.allclose(y, reg(x2, valid), atol=1e-5), name  # padding channel values do not affect the output
    assert r2(torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]]), torch.tensor([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])) == 1.0


def test_ridge_recovers_synthetic():
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    score, lam = ridge_r2(tr, te)
    print(f"\nridge R2 {score:.3f} lam {lam}")
    assert score > 0.5


def test_scratch_and_frozen_arms():
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    score, hist = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    print(f"\nscratch R2 {score:.3f} loss {hist[0]:.3f} -> {hist[-1]:.3f}")
    assert hist[-1] < hist[0] and score > 0.0
    s2, _ = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    assert s2 == score  # same seed gives the same result (U26)
    frozen, _ = fit_arm(tr, te, TINY, seed=0, frozen=True, epochs=3, lr=1e-3)
    assert np.isfinite(frozen)


def test_split_keys_by_session():
    """U11: trial numbers repeat across sessions, so a trial is (session_id, interval_id)."""
    ws = [{**w, "session_id": sess} for sess in ("a", "b") for w in synthetic_labeled(10, 5)]
    tr, te = split_trials(ws, 0.8, seed=0)
    keys = lambda part: {(w["session_id"], w["interval_id"]) for w in part}  # noqa: E731
    assert len(keys(tr)) == 16 and len(keys(te)) == 4  # 20 trials in total, not 10
    assert not keys(tr) & keys(te)


def test_label_scale_does_not_matter():
    """U27: labels are z-scored with train statistics, so raw velocity units (mm/s, cm/s) do not change R²."""
    ws = synthetic_labeled(200, 30, seed=0)
    tr, te = split_trials(ws, 0.8, seed=0)
    scale = lambda part, k: [{**w, "vel": w["vel"] * k} for w in part]  # noqa: E731
    r1, _ = fit_arm(tr, te, TINY, seed=0, epochs=15, lr=1e-3)
    r300, _ = fit_arm(scale(tr, 300), scale(te, 300), TINY, seed=0, epochs=15, lr=1e-3)
    print(f"\nscratch R2 x1 {r1:.3f}, x300 {r300:.3f}")
    assert r300 > 0.1 and abs(r300 - r1) < 0.02


def test_ridge_is_per_session():
    """Units are different neurons in every session. Session b has the same counts as session a but the opposite
    velocity: separate decoders fit both, a decoder shared across sessions cancels out (R² near 0). Different unit
    counts must not crash either (the Perich T-CO run has 6 sessions)."""
    a = [{**w, "session_id": "a"} for w in synthetic_labeled(150, 20, seed=1)]
    flipped = a + [{**w, "session_id": "b", "vel": -w["vel"]} for w in a]
    diff = [{**w, "session_id": "a"} for w in synthetic_labeled(150, 30, seed=1)] + \
           [{**w, "session_id": "b"} for w in synthetic_labeled(150, 20, seed=2)]
    for ws in (flipped, diff):
        tr, te = split_trials(ws, 0.8, seed=0)
        score, _ = ridge_r2(tr, te)
        assert score > 0.8, score
