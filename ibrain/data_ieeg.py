"""M5: iEEG input (SPEC 1.1). 500 Hz, 1 s window = 500 samples, 10 patches of 100 ms × 50 samples.
One channel is one sequence. Real-data (AJILE12, SWEC) loaders come once access is secured.
For now only the (C, 500) waveform -> patch conversion and synthetic data (U28).
Channel padding reuses data_spike.collate as is."""
import numpy as np
import torch
from torch.utils.data import Dataset

FS, WINDOW_SAMPLES = 500, 500  # Hz, 1 s window
S, P = 10, 50  # 10 patches of 100 ms, 50 samples per patch


def to_patches(wave):
    """One window (C, 500) float -> (C, S, P) float32 (Eq. 1 input). x[c, s, p] = wave[c, 50s + p]."""
    x = torch.from_numpy(np.ascontiguousarray(wave, dtype=np.float32))
    return x.view(x.shape[0], S, P)


class SyntheticIEEG(Dataset):
    """Synthetic iEEG for tests. Each 'session' has a different number of channels;
    each channel is 3 sinusoids + noise + a DC offset. Per-channel scales differ,
    so reconstruction is hard without channel_normalize (U4).
    The conv adapter (U1) is expensive per token, so CPU tests keep the channel count small."""

    def __init__(self, n_channels=(4, 8, 16), n_windows=16, seed=0):
        rng = np.random.default_rng(seed)
        t = np.arange(WINDOW_SAMPLES) / FS
        self.windows = []
        for C in n_channels:
            f = rng.uniform(1, 40, (C, 3, 1))
            a = rng.uniform(0.5, 2, (C, 3, 1))
            ph = rng.uniform(0, 2 * np.pi, (C, 3, 1))
            scale = rng.uniform(10, 500, (C, 1))  # uV-ish, per channel
            dc = rng.normal(0, 100, (C, 1))
            for _ in range(n_windows):
                off = rng.uniform(0, 1)
                w = (a * np.sin(2 * np.pi * f * (t + off) + ph)).sum(1)  # (C, 500)
                self.windows.append(scale * (w + rng.normal(0, 0.3, (C, WINDOW_SAMPLES))) + dc)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return to_patches(self.windows[i])
