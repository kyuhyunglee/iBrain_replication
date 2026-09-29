"""M5: iEEG input (SPEC 1.1). 500 Hz, 1 s window = 500 samples, 10 patches of 100 ms × 50 samples.
One channel is one sequence. Real-data (AJILE12, SWEC) loaders come once access is secured
(offline preprocessing plan: docs/260929_ieeg_offline_preprocessing.md).
For now only the (C, 500) waveform -> patch conversion, recording-level channel normalization (U4)
and synthetic data (U28). Channel padding reuses data_spike.collate as is."""
import numpy as np
import torch
from torch.utils.data import Dataset

FS, WINDOW_SAMPLES = 500, 500  # Hz, 1 s window
S, P = 10, 50  # 10 patches of 100 ms, 50 samples per patch
IQR_TO_STD = 1.349  # IQR of a standard normal


def to_patches(wave):
    """One window (C, 500) float -> (C, S, P) float32 (Eq. 1 input). x[c, s, p] = wave[c, 50s + p]."""
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
    return x.view(x.shape[0], S, P)


def channel_stats(wave):
    """Per-channel center and scale of one recording (U4). wave (C, T), T = the whole recording, not one window.
    center = median, scale = IQR / 1.349 (robust std; long recordings contain large artifacts)."""
    q25, q50, q75 = np.percentile(wave, [25, 50, 75], axis=1)
    return q50, (q75 - q25) / IQR_TO_STD


def channel_normalize(x, center, scale, eps=1e-6):
    """iEEG: normalize with the recording's per-channel statistics (U4). x has the channel axis first,
    e.g. (C, 500) or (C, S, P); center, scale (C,) from channel_stats. Applied in the Dataset, so the
    input and the Eq. (8) target are the same values and nothing depends on the mask."""
    shape = (-1,) + (1,) * (x.ndim - 1)
    return (x - center.reshape(shape)) / (scale.reshape(shape) + eps)


class SyntheticIEEG(Dataset):
    """Synthetic iEEG for tests. Each 'session' (= one recording) has a different number of channels;
    each channel is 3 sinusoids + noise + a DC offset. Per-channel scales differ, so each session is
    normalized with its own channel_stats over all of its windows (U4).
    The conv adapter (U1) is expensive per token, so CPU tests keep the channel count small."""

    def __init__(self, n_channels=(4, 8, 16), n_windows=16, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(WINDOW_SAMPLES) / FS
        self.windows, self.session, self.stats = [], [], []
        for i, C in enumerate(n_channels):
            f = rng.uniform(1, 40, (C, 3, 1))
            a = rng.uniform(0.5, 2, (C, 3, 1))
            ph = rng.uniform(0, 2 * np.pi, (C, 3, 1))
            scale = rng.uniform(10, 500, (C, 1))  # uV-ish, per channel
            dc = rng.normal(0, 100, (C, 1))
            ws = []
            for _ in range(n_windows):
                off = rng.uniform(0, 1)
                w = (a * np.sin(2 * np.pi * f * (t + off) + ph)).sum(1)  # (C, 500)
                ws.append(scale * (w + rng.normal(0, 0.3, (C, WINDOW_SAMPLES))) + dc)
            self.stats.append(channel_stats(np.concatenate(ws, axis=1)))  # the whole recording (C, n_windows × 500)
            self.windows += ws
            self.session += [i] * n_windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return to_patches(channel_normalize(self.windows[i], *self.stats[self.session[i]]))
