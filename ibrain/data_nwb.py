"""M4: downstream NWB files -> window dicts with behavior labels (SPEC U9, U21, U29).
Reads DANDI NWB files directly: NLB MC-Maze (000128), Area2-Bump (000127), Perich-Miller (000688).
Windows have the same keys as data_spike.read_windows, plus "vel" (S, D) when a behavior series is given.
Labels are attached file by file, so a window never gets the behavior of another session."""
from pathlib import Path

import numpy as np

from ibrain.data_spike import BIN_SECONDS, S, WINDOW_BINS

WINDOW_SECONDS = BIN_SECONDS * WINDOW_BINS  # 1 s


class MissingSeries(KeyError):
    """The requested behavior TimeSeries is not in the file (e.g. NLB test files carry no behavior)."""


def find_series(container, name):
    """The TimeSeries called `name` anywhere in the NWB tree: acquisition, processing modules, and nested
    containers such as Position or Velocity (Perich keeps cursor_vel in processing/behavior/Velocity).
    Raises MissingSeries if it is absent and ValueError if the name is not unique."""
    from pynwb import TimeSeries
    hits, stack, seen = [], [container], set()
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, TimeSeries) and obj.name == name:
            hits.append(obj)
        stack.extend(getattr(obj, "children", ()))
    if not hits:
        raise MissingSeries(name)
    if len(hits) > 1:
        raise ValueError(f"{len(hits)} TimeSeries named {name!r}: {[h.parent.name for h in hits]}")
    return hits[0]


def series_arrays(series):
    """TimeSeries -> (timestamps (T,), data (T, D)) as float64. Handles stored timestamps, timestamps linked from
    another series (Perich cursor_vel uses cursor_pos), and starting_time + rate."""
    data = np.asarray(series.data[:], dtype=np.float64)
    ts = np.asarray(series.get_timestamps()[:], dtype=np.float64)
    return ts, data.reshape(len(data), -1)


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


def attach_velocity(windows, ts, vel):
    """Attaches the patch label "vel" to windows of ONE session (start_seconds in that session's clock).
    Windows whose label has an empty or NaN patch are dropped (U29)."""
    out = []
    for w in windows:
        y = patch_labels(w["start_seconds"], ts, vel)
        if not np.isnan(y).any():
            out.append({**w, "vel": y})
    return out


def window_starts(starts, stops, length=WINDOW_SECONDS):
    """U21: non-overlapping 1 s windows from each trial start; the tail shorter than 1 s is dropped.
    Returns (window start times (W,), index of the trial each window comes from (W,))."""
    t0, k = [], []
    for i, (a, b) in enumerate(zip(starts, stops)):
        n = int(np.floor((b - a) / length + 1e-9))
        t0.extend(a + length * np.arange(n))
        k.extend([i] * n)
    return np.asarray(t0, dtype=np.float64), np.asarray(k, dtype=int)


def bin_counts(spike_times, starts):
    """20 ms spike counts in left-closed bins [t, t + 0.02). spike_times: list of C sorted arrays (s),
    starts (W,) -> uint32 (W, 50, C), the layout of the corpus shards."""
    edges = starts[:, None] + BIN_SECONDS * np.arange(WINDOW_BINS + 1)  # (W, 51)
    out = np.zeros((len(starts), WINDOW_BINS, len(spike_times)), dtype=np.uint32)
    for c, st in enumerate(spike_times):
        out[:, :, c] = np.diff(np.searchsorted(st, edges, side="left"), axis=1)
    return out


def trial_mask(trials):
    """U29: trials to keep. NLB marks trials outside train/val with split == "none" (Area2 has 462, some 0.19 s long);
    Perich records the outcome in result (R rewarded, A aborted, F failed, I incomplete) and only R is kept,
    as in the brainsets Perich pipeline. Tables without these columns keep every trial."""
    keep = np.ones(len(trials), dtype=bool)
    if "split" in trials.colnames:
        keep &= np.asarray(trials["split"][:]).astype(str) != "none"
    if "result" in trials.colnames:
        keep &= np.asarray(trials["result"][:]).astype(str) == "R"
    return keep


def read_nwb(path, behavior=None):
    """One NWB file -> list of window dicts {counts uint32 [50, C], start_seconds, interval_id, unit_ids, session_id}.
    Every unit of the units table is used (NLB held-out units included). Windows come from the trials kept by
    trial_mask, cut by the U21 rule.
    session_id is the file stem. With `behavior` (e.g. "hand_vel" for NLB, "cursor_vel" for Perich) each window
    also gets "vel" (S, D). Raises MissingSeries when the file has no such series, before reading any spikes."""
    from pynwb import NWBHDF5IO
    path = Path(path)
    with NWBHDF5IO(str(path), "r", load_namespaces=True) as io:
        nwb = io.read()
        beh = series_arrays(find_series(nwb, behavior)) if behavior else None
        if nwb.units is None or nwb.trials is None:
            raise ValueError(f"{path.name}: needs both a units table and a trials table")
        units = nwb.units
        spikes = [np.sort(np.asarray(units.get_unit_spike_times(i), dtype=np.float64)) for i in range(len(units))]
        unit_ids = np.asarray(units.id[:]).astype(str)
        keep = trial_mask(nwb.trials)
        trial_ids = np.asarray(nwb.trials.id[:])[keep]
        t0, k = window_starts(np.asarray(nwb.trials["start_time"][:])[keep], np.asarray(nwb.trials["stop_time"][:])[keep])
    counts = bin_counts(spikes, t0)
    windows = [{"counts": c, "start_seconds": float(s), "interval_id": str(trial_ids[i]), "unit_ids": unit_ids,
                "session_id": path.stem} for c, s, i in zip(counts, t0, k)]
    return attach_velocity(windows, *beh) if beh else windows


def nwb_files(paths):
    """Files and directories -> sorted list of .nwb files (directories are searched recursively)."""
    out = []
    for p in map(Path, paths):
        out.extend(sorted(p.rglob("*.nwb")) if p.is_dir() else [p])
    return out
