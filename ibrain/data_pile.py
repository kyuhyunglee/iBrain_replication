"""M3: Neural Pile (primate v1.0) parquet -> 1 s spike windows for pretraining (SPEC U10, U14, U30).
Each row is one segment: spike_counts list<list<uint8>> of shape (n_units, t) in 20 ms bins, with subject_id,
session_id, segment_id and source_dataset. There are no spike times and no trials, so each row is cut into
consecutive non-overlapping 50-bin windows from bin 0 and the tail shorter than 50 bins is dropped."""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset

from ibrain.data_spike import BIN_SECONDS, WINDOW_BINS, to_patches

# Pile sources that hold downstream evaluation sessions (lab-server survey, 2026-09-28, U14):
# perich = all 111 sessions of DANDI 000688, area2-bump = both files of 000127. dmfc-rsg (000130) is kept: it is not an
# iBrain downstream task (decided 2026-09-29).
LEAKED = ("perich", "area2-bump")


def pile_files(root, split="train"):
    """Parquet files of one split, HF naming <split>-NNNNN-of-NNNNN.parquet, searched recursively."""
    files = sorted(Path(root).rglob(f"{split}-*.parquet"))
    if not files:
        found = sorted(p.name for p in Path(root).rglob("*.parquet"))[:5]
        raise FileNotFoundError(f"no {split}-*.parquet under {root}; parquet files found: {found}")
    return files


def row_matrices(column):
    """list<list<uint8>> column -> list of uint8 arrays (n_units, t), one per row, without per-element Python work."""
    arr = column.combine_chunks()
    units = arr.value_lengths().to_numpy()  # units per row
    inner = arr.flatten()  # one list per unit, all rows concatenated
    t = inner.value_lengths().to_numpy()
    vals = inner.flatten().to_numpy()
    u_end, v_end = np.cumsum(units), np.cumsum(t)
    out = []
    for r, n in enumerate(units):
        if n == 0:
            out.append(np.zeros((0, 0), dtype=np.uint8))
            continue
        u0 = u_end[r] - n
        tt = t[u0:u_end[r]]
        if (tt != tt[0]).any():
            raise ValueError(f"row {r}: units have different lengths {sorted(set(tt.tolist()))[:5]}")
        v0 = v_end[u0] - t[u0]
        out.append(np.array(vals[v0:v0 + n * tt[0]], dtype=np.uint8).reshape(n, tt[0]))
    return out


class PileWindows(Dataset):
    """1 s windows (C, S, P) from the Neural Pile. Rows are held in memory as uint8, windows are indexed lazily.
    exclude_sources: pile sources left out (U14). The default is none, as in the paper; LEAKED gives the variant without
    evaluation sessions. A name that is not a source of the split raises, so a typo cannot silently keep leaked data. max_rows: a seeded random subset of the remaining rows (small-scale runs)."""

    def __init__(self, root, split="train", exclude_sources=(), max_rows=None, seed=0):
        files = pile_files(root, split)
        meta = []
        for fi, f in enumerate(files):
            src = pq.read_table(f, columns=["source_dataset"]).column(0).to_pylist()
            meta += [(fi, ri, s) for ri, s in enumerate(src)]
        sources = sorted({s for _, _, s in meta})
        unknown = sorted(set(exclude_sources) - set(sources))
        if unknown:
            raise ValueError(f"exclude_sources not in the {split} split: {unknown}. Sources: {sources}")
        kept = [(fi, ri) for fi, ri, s in meta if s not in set(exclude_sources)]
        if max_rows is not None and max_rows < len(kept):
            pick = np.random.default_rng(seed).choice(len(kept), max_rows, replace=False)
            kept = sorted(kept[i] for i in pick)
        self.mats = []
        for fi in sorted({fi for fi, _ in kept}):
            rows = [ri for f, ri in kept if f == fi]
            table = pq.read_table(files[fi], columns=["spike_counts"]).take(rows)
            self.mats += row_matrices(table.column(0))
        n = np.array([m.shape[1] // WINDOW_BINS for m in self.mats], dtype=np.int64)
        self.end = np.cumsum(n)  # window i lives in matrix searchsorted(end, i, "right")
        self.sources = sources
        self.excluded = sorted(exclude_sources)
        self.leaked = sorted(set(LEAKED) - set(exclude_sources))
        self.hours = int(self.end[-1]) * WINDOW_BINS * BIN_SECONDS / 3600 if len(n) else 0.0
        print(f"Neural Pile {split}: {len(self.mats)} rows kept of {len(meta)}, {len(self)} windows, "
              f"{self.hours:.1f} h. Excluded sources: {self.excluded}")
        if self.leaked:
            print(f"WARNING: pretraining includes sources with evaluation sessions: {self.leaked} (U14)")

    def __len__(self):
        return int(self.end[-1]) if len(self.end) else 0

    def __getitem__(self, i):
        m = int(np.searchsorted(self.end, i, side="right"))
        o = (i - (self.end[m - 1] if m else 0)) * WINDOW_BINS
        return to_patches(self.mats[m][:, o:o + WINDOW_BINS].T)
