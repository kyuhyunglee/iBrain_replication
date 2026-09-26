"""M2: nhp-spike-corpus count shards -> iBrain spike input (SPEC 1.1, M2).
The shard format follows convert.py in Transconnectome/nhp-spike-corpus.
  <root>/<source_id with ':' replaced by '-'>/<asset UUID>/manifest.jsonl, shard-NNNNNN.npz
  NPZ fields: counts uint32[windows, time_bins, units], start_seconds, interval_ids, unit_ids
"""
import json
import re
import uuid
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

BIN_SECONDS, WINDOW_BINS = 0.02, 50  # 20 ms bin, 1 s window (SPEC 1.1, corpus default)
S, P = 10, 5  # 10 patches of 100 ms, 5 bins per patch
SYNTHETIC = "dandi:999999"  # synthetic source ID (same convention as the corpus tests)
_UUID = re.compile(r"[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.I)


def to_patches(counts):
    """One window uint32[50, units] -> (units, S, P) float32 (Eq. 1 input). x[c, s, p] = counts[5s + p, c]."""
    x = torch.from_numpy(np.ascontiguousarray(counts.T, dtype=np.float32))  # (units, 50)
    return x.view(counts.shape[1], S, P)


def read_windows(root, source_id):
    """Yields the windows of one source (e.g. "dandi:000128") in order. Keys as in corpus iter_windows (a subset).
    Reads only published assets (UUID dirs); staging (*.partial-*) is skipped (same rule as corpus.read_manifests).
    Subject-level train/validation splits and audits use the corpus-side tools."""
    for manifest in sorted((Path(root) / source_id.replace(":", "-")).glob("*/manifest.jsonl")):
        if not _UUID.fullmatch(manifest.parent.name):
            continue
        for line in manifest.read_text(encoding="utf-8").splitlines():
            e = json.loads(line)
            if (e["bin_seconds"], e["window_bins"]) != (BIN_SECONDS, WINDOW_BINS):
                raise ValueError(f"only shards with 50 bins of 20 ms (1 s) are accepted: {manifest}")
            with np.load(manifest.parent / e["shard"], allow_pickle=False) as z:
                unit_ids = z["unit_ids"]
                for counts, start, trial in zip(z["counts"], z["start_seconds"], z["interval_ids"], strict=True):
                    yield {"counts": counts, "start_seconds": float(start), "interval_id": str(trial),
                           "unit_ids": unit_ids, "session_id": e["session_id"]}


class SpikeWindows(Dataset):
    """Window dicts -> (C, S, P) patches per window. Accepts the output of read_windows or corpus iter_windows as is."""

    def __init__(self, windows):
        self.counts = [w["counts"] for w in windows]  # [time_bins, units], all in memory (M3 scale)

    def __len__(self):
        return len(self.counts)

    def __getitem__(self, i):
        return to_patches(self.counts[i])


def collate(batch):
    """Zero-pad the channel axis (SPEC 1.1). list of (C_i, S, P) -> x (B, Cmax, S, P), valid (B, Cmax) bool = V_c."""
    x = pad_sequence(batch, batch_first=True)
    n = torch.tensor([len(t) for t in batch])
    return x, torch.arange(x.shape[1]) < n[:, None]


def write_synthetic(root, n_units=(3, 8, 20), n_windows=16, seed=0):
    """Synthetic corpus for tests. Each asset (session) has a different number of units; each unit has Poisson
    counts with its own firing rate (1~50 Hz, log-uniform). Writes one shard per asset to root/dandi-999999/<UUID>/
    with the same layout and fields as convert.py. Returns the per-asset firing rates (Hz)."""
    rng = np.random.default_rng(seed)
    rates = []
    for i, U in enumerate(n_units):
        hz = 10 ** rng.uniform(0, np.log10(50), U)
        counts = rng.poisson(hz * BIN_SECONDS, (n_windows, WINDOW_BINS, U)).astype(np.uint32)
        d = Path(root) / SYNTHETIC.replace(":", "-") / str(uuid.UUID(int=i))
        d.mkdir(parents=True)
        np.savez_compressed(
            d / "shard-000000.npz",
            counts=counts,
            start_seconds=np.arange(n_windows, dtype=np.float64),
            interval_ids=np.arange(n_windows).astype(str),
            unit_ids=np.array([f"unit-{j}" for j in range(U)]),
        )
        # minimal manifest that corpus iter_windows can also read (role=evaluation is read without a subject ID)
        entry = {"shard": "shard-000000.npz", "source_id": SYNTHETIC, "asset_id": d.name,
                 "session_id": f"synthetic-{i}", "role": "evaluation", "n_units": U,
                 "n_windows": n_windows, "bin_seconds": BIN_SECONDS, "window_bins": WINDOW_BINS}
        (d / "manifest.jsonl").write_text(json.dumps(entry) + "\n", encoding="utf-8")
        rates.append(hz)
    return rates
