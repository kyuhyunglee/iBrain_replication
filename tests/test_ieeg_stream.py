"""M5 checks for the streamed iEEG loaders on synthetic files with the real layouts:
AJILE12 NWB (ElectricalSeries (T, C) at 500 Hz, electrodes.good, Blocklist epochs) and
SWEC-ETHZ part files (Blosc-compressed data/ieeg (C, T) at 1024 Hz, sampling_rate attribute, _total.h5 alongside)."""
from datetime import datetime, timezone

import h5py
import hdf5plugin
import numpy as np
import torch
from pynwb import NWBHDF5IO, NWBFile
from pynwb.ecephys import ElectricalSeries
from scipy.signal import resample_poly
from torch.utils.data import DataLoader

from ibrain.data_ieeg import (IEEGStream, ajile12_blocks, channel_normalize, channel_stats, read_block,
                              select_hours, swec_blocks)


def write_ajile12(path, seconds=20, C=5):
    """data[t, c] = 10 t + c, so a window tells where it came from. Channel 2 is marked bad. Blocklist epoch over
    5.5-7.2 s. A NaN in good channel 0 at 12.3 s, and one in bad channel 2 at 15.1 s (must not matter)."""
    data = (10.0 * np.arange(seconds * 500)[:, None] + np.arange(C)[None]).astype(np.float32)
    data[int(12.3 * 500), 0] = np.nan
    data[int(15.1 * 500), 2] = np.nan
    nwb = NWBFile(session_description="synthetic", identifier=path.stem,
                  session_start_time=datetime(2020, 1, 1, tzinfo=timezone.utc))
    grp = nwb.create_electrode_group("g", description="g", location="unknown", device=nwb.create_device("amp"))
    nwb.add_electrode_column("good", "good channel")
    for c in range(C):
        nwb.add_electrode(group=grp, location="unknown", good=(c != 2))
    region = nwb.create_electrode_table_region(list(range(C)), "all")
    nwb.add_acquisition(ElectricalSeries(name="ElectricalSeries", data=data, electrodes=region, rate=500.0,
                                         conversion=1e-6))
    nwb.add_epoch_column("labels", "annotation")
    nwb.add_epoch(start_time=0.0, stop_time=5.0, labels="Sleep/rest")
    nwb.add_epoch(start_time=5.5, stop_time=7.2, labels="Blocklist (Data break)")
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)


def starts(windows):
    """Window start in seconds, decoded from sample 0 of channel 0 (10 t + c)."""
    return sorted(int(w[0, 0, 0].item() // 10) / 500 for w in windows)


def test_ajile12_stream(tmp_path):
    write_ajile12(tmp_path / "sub-01_ses-3.nwb")
    blocks = ajile12_blocks(tmp_path, block_seconds=4)
    assert len(blocks) == 5 and blocks[0].cols == (0, 1, 3, 4)
    ws = list(IEEGStream(blocks, buffer=6, normalize=False))
    # 20 windows; 5, 6, 7 overlap the Blocklist epoch; 12 has a NaN in a good channel; 15 has one only in bad ch 2
    assert starts(ws) == [float(s) for s in range(20) if s not in (5, 6, 7, 12)]
    w = next(w for w in ws if w[0, 0, 0].item() == 10.0 * 500 * 3)  # window starting at 3 s
    expect = 10.0 * (1500 + np.arange(500))[None] + np.array([0, 1, 3, 4])[:, None]
    assert w.shape == (4, 10, 50) and torch.equal(w.reshape(4, 500), torch.from_numpy(expect.astype(np.float32)))


def test_stream_order_and_workers(tmp_path):
    write_ajile12(tmp_path / "a.nwb")
    blocks = ajile12_blocks(tmp_path, block_seconds=4)
    torch.manual_seed(0)
    a = starts(list(IEEGStream(blocks, buffer=6, normalize=False)))
    torch.manual_seed(0)
    first = [w[0, 0, 0].item() for w in IEEGStream(blocks, buffer=6, normalize=False)]
    torch.manual_seed(0)
    assert [w[0, 0, 0].item() for w in IEEGStream(blocks, buffer=6, normalize=False)] == first  # same seed, same order
    assert first != sorted(first)  # but shuffled
    # two workers: every window exactly once
    dl = DataLoader(IEEGStream(blocks, buffer=6, normalize=False), batch_size=None, num_workers=2)
    assert starts(list(dl)) == a


def test_select_hours(tmp_path):
    write_ajile12(tmp_path / "a.nwb")
    blocks = ajile12_blocks(tmp_path, block_seconds=4)
    sub = select_hours(blocks, 8.5 / 3600, seed=0)
    assert len(sub) == 2 and sub == select_hours(blocks, 8.5 / 3600, seed=0) and select_hours(blocks, None) == blocks


def test_swec_resampling_and_skips(tmp_path):
    fs, sec, C = 1024, 20, 3
    t = np.arange(fs * sec) / fs
    # sines repeat every 0.1 s, so a slow ramp makes each 1 s window unique (lets us tell windows apart)
    x = np.stack([np.sin(2 * np.pi * 10 * t + c) + 0.5 * np.sin(2 * np.pi * 400 * t) + t / sec for c in range(C)])
    x = x.astype(np.float32)
    with h5py.File(tmp_path / "ID01_1h.h5", "w") as h:
        h.create_dataset("data/ieeg", data=x, chunks=(C, 4096), **hdf5plugin.Blosc())
        h.attrs["sampling_rate"], h.attrs["channels"] = fs, C
    with h5py.File(tmp_path / "ID01_total.h5", "w") as h:  # must be ignored
        h.create_dataset("data/ieeg", data=np.zeros((C, 10 * fs), np.float32))
        h.attrs["sampling_rate"] = fs
    (tmp_path / "ID02_1h.h5").write_bytes(b"partial download")  # unreadable, must be skipped
    blocks = swec_blocks(tmp_path, block_seconds=4)
    assert len(blocks) == 5 and {b.path.rsplit("/", 1)[-1] for b in blocks} == {"ID01_1h.h5"}
    ws = [w.reshape(C, 500).numpy() for w in IEEGStream(blocks, buffer=4, normalize=False)]
    ref = resample_poly(x, 125, 256, axis=1)  # whole recording at 500 Hz
    assert len(ws) == 20
    hit = []
    for w in ws:
        err = [np.abs(w - ref[:, k * 500:(k + 1) * 500]).max() for k in range(20)]
        hit.append(int(np.argmin(err)))
        assert min(err) < 1e-4  # block edges match whole-recording resampling (context padding works)
    assert sorted(hit) == list(range(20))
    w = ws[0][0] - np.polyval(np.polyfit(np.arange(500), ws[0][0], 1), np.arange(500))  # remove the ramp
    spec = np.abs(np.fft.rfft(w))
    assert np.argmax(spec[5:]) + 5 == 10 and spec[100] < 0.01 * spec[10]  # 10 Hz kept, 400 Hz not aliased to 100 Hz


def test_block_length_is_whole_windows(tmp_path):
    """A 168.75 s block (one storage chunk) would drop 0.75 s per block and restart the 1 s grid: 675 s -> 672 windows."""
    import pytest
    write_ajile12(tmp_path / "a.nwb")
    with pytest.raises(ValueError, match="whole number"):
        ajile12_blocks(tmp_path, block_seconds=168.75)
    assert (ajile12_blocks(tmp_path)[0].stop - ajile12_blocks(tmp_path)[0].start) == 20 * 500  # default 675 s, file is 20 s


def test_stream_normalizes_per_block_without_window_leak(tmp_path):
    """U4 at chunk level: each block is normalized with robust statistics of its usable samples (NaN and Blocklist
    spans left out), so a window's values do not depend on its own patches: window means are not forced to 0."""
    from ibrain.data_ieeg import block_windows
    write_ajile12(tmp_path / "a.nwb", seconds=80)
    blocks = ajile12_blocks(tmp_path, block_seconds=40)
    expected = []
    for b in blocks:  # independent re-implementation of the rule
        x = read_block(b)
        u = x.astype(np.float64).copy()
        for s0, s1 in b.bad:
            u[:, int(np.floor((s0 - b.start / 500) * 500)):int(np.ceil((s1 - b.start / 500) * 500))] = np.nan
        c, sc = channel_stats(u)
        expected += [w.numpy() for w in block_windows(b, channel_normalize(x, c, sc))]
    got = [w.numpy() for w in IEEGStream(blocks, buffer=4)]
    key = lambda w: np.round(w, 4).tobytes()  # noqa: E731
    assert len(got) == 76 and sorted(map(key, got)) == sorted(map(key, expected))  # 80 minus Blocklist 5-7, NaN 12
    means = np.array([w.reshape(4, 500).mean(1) for w in got])
    assert np.abs(means).max() > 0.5  # a per-window z-score would make every window mean exactly 0


def write_block_h5(path, x, fs=500):
    """A SWEC-style file (no channel metadata) holding x (C, T)."""
    with h5py.File(path, "w") as h:
        h.create_dataset("data/ieeg", data=x.astype(np.float32))
        h.attrs["sampling_rate"] = fs


def test_flat_and_artifact_spans_do_not_set_the_scale(tmp_path):
    """Channel 1 is constant for 55% of the block and channel 2 is all zeros: with statistics over the whole block
    their scale is ~0 and eps blows them up to ~1e8. They must be left out instead. Short files (< 30 s of usable
    samples) give no windows, because their statistics would come from the windows themselves."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 50, (3, 200 * 500))
    x[1, : 110 * 500] = 7.0
    x[2] = 0.0
    write_block_h5(tmp_path / "ID01_1h.h5", x)
    ws = list(IEEGStream(swec_blocks(tmp_path, block_seconds=200), buffer=4))
    assert len(ws) == 200 and all(w.shape[0] == 1 for w in ws)  # only channel 0 is kept
    z = np.concatenate([w.reshape(1, 500).numpy() for w in ws], axis=1)
    assert np.abs(z).max() < 10 and abs(np.subtract(*np.percentile(z, [75, 25])) / 1.349 - 1) < 0.05
    write_block_h5(tmp_path / "ID01_1h.h5", rng.normal(0, 50, (3, 10 * 500)))
    assert list(IEEGStream(swec_blocks(tmp_path, block_seconds=200), buffer=4)) == []


def test_short_tail_joins_previous_block(tmp_path):
    write_ajile12(tmp_path / "a.nwb")  # 20 s
    spans = [(b.start // 500, b.stop // 500) for b in ajile12_blocks(tmp_path, block_seconds=9)]
    assert spans == [(0, 9), (9, 20)]  # the 2 s tail is too short for its own statistics
    spans = [(b.start // 500, b.stop // 500) for b in ajile12_blocks(tmp_path, block_seconds=8)]
    assert spans == [(0, 8), (8, 16), (16, 20)]  # a 4 s tail (half a block) keeps its own block


def test_swec_blocks_follow_storage_chunks(tmp_path):
    """Blocks are a whole number of storage chunks (read per file), so they start on chunk boundaries."""
    fs, C = 1024, 2
    x = np.random.default_rng(0).normal(size=(C, fs * 40)).astype(np.float32)
    with h5py.File(tmp_path / "ID01_1h.h5", "w") as h:
        h.create_dataset("data/ieeg", data=x, chunks=(C, 4 * fs), **hdf5plugin.Blosc())  # 4 s chunks
        h.attrs["sampling_rate"] = fs
    blocks = swec_blocks(tmp_path, target_seconds=9)  # closest whole number of chunks: 2 x 4 s = 8 s
    assert [(b.start // fs, b.stop // fs) for b in blocks] == [(0, 8), (8, 16), (16, 24), (24, 32), (32, 40)]
    assert all(b.start % (4 * fs) == 0 for b in blocks)
