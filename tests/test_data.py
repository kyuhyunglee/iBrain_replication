"""M2 verification: patch conversion, corpus shard reading, channel padding."""
import shutil

import numpy as np
import torch

from ibrain.data_spike import (BIN_SECONDS, SYNTHETIC, SpikeWindows, collate, read_windows, to_patches,
                               write_synthetic)


def test_patches_shape_and_counts():
    counts = np.random.default_rng(0).poisson(0.5, (50, 7)).astype(np.uint32)
    counts[0, 0] = 300  # large counts are kept as-is
    x = to_patches(counts)
    assert x.shape == (7, 10, 5) and x.dtype == torch.float32  # (C, S, P), SPEC 1.1
    assert torch.equal(x.reshape(7, 50).T, torch.from_numpy(counts.astype(np.float32)))  # x[c, s, p] = counts[5s + p, c]
    assert torch.equal(x.sum((1, 2)), torch.from_numpy(counts.sum(0).astype(np.float32)))  # per-unit patch sum = bin sum


def test_reader_matches_rates(tmp_path):
    """When synthetic shards are read in corpus format, per-unit mean counts match the firing rates (SPEC M2)."""
    n_units, n_windows = (3, 8, 20), 64
    rates = write_synthetic(tmp_path, n_units, n_windows)
    asset = next((tmp_path / "dandi-999999").iterdir())
    shutil.copytree(asset, asset.parent / (asset.name + ".partial-x"))  # staging copies must not be read
    windows = list(read_windows(tmp_path, SYNTHETIC))
    assert len(windows) == len(n_units) * n_windows
    for i, (U, hz) in enumerate(zip(n_units, rates)):
        ws = [w for w in windows if w["session_id"] == f"synthetic-{i}"]
        assert len(ws) == n_windows and ws[0]["counts"].shape == (50, U) and ws[0]["counts"].dtype == np.uint32
        mean = torch.stack([to_patches(w["counts"]) for w in ws]).mean((0, 2, 3)).numpy()  # per-unit mean per bin
        lam = hz * BIN_SECONDS
        assert np.all(np.abs(mean - lam) < 4 * np.sqrt(lam / (n_windows * 50)) + 1e-3)


def test_collate_pads_channels(tmp_path):
    write_synthetic(tmp_path, (3, 8, 20), n_windows=4)
    ds = SpikeWindows(read_windows(tmp_path, SYNTHETIC))
    batch = [ds[0], ds[4], ds[8]]  # one window per asset
    x, valid = collate(batch)
    assert x.shape == (3, 20, 10, 5) and valid.dtype == torch.bool
    assert valid.sum(1).tolist() == [3, 8, 20]
    for i, t in enumerate(batch):
        assert valid[i, :len(t)].all() and not valid[i, len(t):].any()
        assert torch.equal(x[i, :len(t)], t) and not x[i, len(t):].any()  # padding is 0
