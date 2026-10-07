"""M3: Neural Pile (primate v1.0) parquet -> 1 s spike windows for pretraining (SPEC U10, U14, U30).
Each row is one segment: spike_counts list<list<uint8>> of shape (n_units, t) in 20 ms bins, with subject_id,
session_id, segment_id and source_dataset. There are no spike times and no trials, so each row is cut into
consecutive non-overlapping 50-bin windows from bin 0 and the tail shorter than 50 bins is dropped."""
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
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


def select_rows(sources_by_row, split, exclude_sources=(), max_rows=None, seed=0):
    """Rows to use, shared by PileWindows and PileMemmap so both pick the same rows in the same order.
    sources_by_row: source per row in pile order (file, then row). Returns (kept row positions ascending, all sources).
    A name in exclude_sources that is not a source of the split raises (U14); max_rows is a seeded random subset."""
    sources = sorted(set(sources_by_row))
    unknown = sorted(set(exclude_sources) - set(sources))
    if unknown:
        raise ValueError(f"exclude_sources not in the {split} split: {unknown}. Sources: {sources}")
    kept = [k for k, s in enumerate(sources_by_row) if s not in set(exclude_sources)]
    if max_rows is not None and max_rows < len(kept):
        pick = np.random.default_rng(seed).choice(len(kept), max_rows, replace=False)
        kept = sorted(kept[i] for i in pick)
    return kept, sources


class _Windows(Dataset):
    """Window indexing and the run report shared by the two Neural Pile readers. Subclasses set the windows per kept
    row (`_finish`) and implement `_window(row, bin offset)` -> uint8 (50, units)."""

    def _finish(self, split, n_rows, n_windows, sources, exclude_sources):
        self.end = np.cumsum(np.asarray(n_windows, dtype=np.int64))  # window i lives in row searchsorted(end, i, "right")
        self.sources = sources
        self.excluded = sorted(exclude_sources)
        self.leaked = sorted(set(LEAKED) - set(exclude_sources))
        self.hours = int(self.end[-1]) * WINDOW_BINS * BIN_SECONDS / 3600 if len(self.end) else 0.0
        print(f"Neural Pile {split}: {len(self.end)} rows kept of {n_rows}, {len(self)} windows, "
              f"{self.hours:.1f} h. Excluded sources: {self.excluded}")
        if self.leaked:
            print(f"WARNING: pretraining includes sources with evaluation sessions: {self.leaked} (U14)")

    def __len__(self):
        return int(self.end[-1]) if len(self.end) else 0

    def __getitem__(self, i):
        m = int(np.searchsorted(self.end, i, side="right"))
        return to_patches(self._window(m, (i - (self.end[m - 1] if m else 0)) * WINDOW_BINS))


class PileWindows(_Windows):
    """1 s windows (C, S, P) from the Neural Pile parquet files. Rows are held in memory as uint8, windows are indexed
    lazily. Every run decompresses the parquet files (one row group each, so even a few rows read all of them, about
    30 min for the whole pile); PileMemmap reads a one-time converted copy instead (U30).
    exclude_sources: pile sources left out (U14). The default is none, as in the paper; LEAKED gives the variant without
    evaluation sessions. A name that is not a source of the split raises, so a typo cannot silently keep leaked data.
    max_rows: a seeded random subset of the remaining rows (small-scale runs)."""

    def __init__(self, root, split="train", exclude_sources=(), max_rows=None, seed=0):
        files = pile_files(root, split)
        pos = []
        for fi, f in enumerate(files):
            pos += [(fi, ri) for ri in range(pq.read_metadata(f).num_rows)]
        src = [s for f in files for s in pq.read_table(f, columns=["source_dataset"]).column(0).to_pylist()]
        kept, sources = select_rows(src, split, exclude_sources, max_rows, seed)
        self.mats = []
        for fi in sorted({pos[k][0] for k in kept}):
            rows = [pos[k][1] for k in kept if pos[k][0] == fi]
            table = pq.read_table(files[fi], columns=["spike_counts"]).take(rows)
            self.mats += row_matrices(table.column(0))
        self._finish(split, len(src), [m.shape[1] // WINDOW_BINS for m in self.mats], sources, exclude_sources)

    def _window(self, m, o):
        return self.mats[m][:, o:o + WINDOW_BINS].T


# One-time conversion for PileMemmap (U30). Layout of <out>/<split>/:
#   shard-NNNNN.u8   the rows of parquet file NNNNN, each stored time-major (n_bins, n_units) uint8 and concatenated,
#                    so one 1 s window is one contiguous block of 50 x n_units bytes
#   rows.npz         per row in pile order: shard, offset (bytes), n_units, n_bins, source_dataset, session_id
#   meta.json        written last: format version, split, and name/size of every source parquet file
MEMMAP_VERSION = 1


def _convert_file(args):
    """One parquet file -> its shard (written to a temporary name, then renamed). Returns the file's row index."""
    path, shard, out = args
    t = pq.read_table(path, columns=["spike_counts", "source_dataset", "session_id"])
    mats = row_matrices(t.column(0))
    tmp = out / f"shard-{shard:05d}.u8.partial"
    rows, offset = [], 0
    with open(tmp, "wb") as f:
        for m in mats:
            f.write(np.ascontiguousarray(m.T).tobytes())  # (n_bins, n_units)
            rows.append((offset, m.shape[0], m.shape[1]))
            offset += m.size
    tmp.rename(out / f"shard-{shard:05d}.u8")
    return rows, t.column(1).to_pylist(), t.column(2).to_pylist()


def convert_memmap(root, out, split="train", workers=1):
    """Neural Pile parquet files of `split` -> the PileMemmap layout under out/<split>. One shard per parquet file,
    converted in `workers` processes. Overwrites an earlier conversion of the split. Returns the output directory."""
    files = pile_files(root, split)
    out = Path(out) / split
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").unlink(missing_ok=True)  # an interrupted conversion is never read
    jobs = [(f, i, out) for i, f in enumerate(files)]
    if workers > 1:
        # spawn: pyarrow keeps threads, and forking a threaded process can deadlock
        with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
            results = list(ex.map(_convert_file, jobs))
    else:
        results = [_convert_file(j) for j in jobs]
    shard, offset, n_units, n_bins, source, session = [], [], [], [], [], []
    for i, (rows, src, ses) in enumerate(results):
        shard += [i] * len(rows)
        offset += [r[0] for r in rows]
        n_units += [r[1] for r in rows]
        n_bins += [r[2] for r in rows]
        source += src
        session += ses
    np.savez(out / "rows.npz", shard=np.array(shard, np.int32), offset=np.array(offset, np.int64),
             n_units=np.array(n_units, np.int32), n_bins=np.array(n_bins, np.int64), source=np.array(source),
             session=np.array(session))
    meta = {"version": MEMMAP_VERSION, "split": split, "root": str(Path(root).resolve()),
            "files": [{"name": f.name, "bytes": f.stat().st_size} for f in files],
            "rows": len(source), "bytes": int(sum(u * b for u, b in zip(n_units, n_bins)))}
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return out


class PileMemmap(_Windows):
    """The same windows as PileWindows (same rows, order, exclusion and max_rows draw), read from the converted copy
    of `convert_memmap` through np.memmap: start-up reads only the row index, and each window reads its own
    50 x n_units bytes from the page cache, which the DataLoader workers (and other jobs on the node) share (U30).
    root: the conversion output (the directory that holds <split>/meta.json)."""

    def __init__(self, root, split="train", exclude_sources=(), max_rows=None, seed=0):
        self.dir = Path(root) / split
        meta_path = self.dir / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"{meta_path} missing: no finished conversion (scripts/pile_to_memmap.py)")
        meta = json.loads(meta_path.read_text())
        if meta["version"] != MEMMAP_VERSION:
            raise ValueError(f"{self.dir}: conversion version {meta['version']}, reader expects {MEMMAP_VERSION}")
        idx = np.load(self.dir / "rows.npz")
        kept, sources = select_rows(idx["source"].tolist(), split, exclude_sources, max_rows, seed)
        self.shard, self.offset = idx["shard"][kept], idx["offset"][kept]
        self.n_units, n_bins = idx["n_units"][kept], idx["n_bins"][kept]
        self._mm = {}
        self._finish(split, meta["rows"], np.where(self.n_units > 0, n_bins // WINDOW_BINS, 0), sources,
                     exclude_sources)

    def __getstate__(self):  # DataLoader workers open their own maps
        return {**self.__dict__, "_mm": {}}

    def _window(self, m, o):
        k = int(self.shard[m])
        if k not in self._mm:
            self._mm[k] = np.memmap(self.dir / f"shard-{k:05d}.u8", dtype=np.uint8, mode="r")
        n = int(self.n_units[m])
        a = int(self.offset[m]) + o * n
        return np.array(self._mm[k][a:a + WINDOW_BINS * n]).reshape(WINDOW_BINS, n)
