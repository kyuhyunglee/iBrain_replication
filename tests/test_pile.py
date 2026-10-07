"""M3 checks on a synthetic Neural Pile with the real schema (spike_counts list<list<uint8>> (n_units, t),
subject_id, session_id, segment_id, source_dataset) and HF file naming."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from torch.utils.data import DataLoader

from ibrain.data_pile import LEAKED, PileMemmap, PileWindows, convert_memmap, row_matrices
from ibrain.data_spike import collate

SCHEMA = pa.schema([("spike_counts", pa.list_(pa.list_(pa.uint8()))), ("subject_id", pa.string()),
                    ("session_id", pa.string()), ("segment_id", pa.string()), ("source_dataset", pa.string())])


def write_pile(root, seed=0):
    """Two train files and one test file. Rows differ in units (3..40) and length (0.7 s .. 7.3 s).
    Returns {(source, session): matrix} for the train split."""
    rng = np.random.default_rng(seed)
    (root / "data").mkdir(parents=True)
    rows = {"train-00000-of-00002": [("perich", 4, 250), ("xiao", 40, 365), ("churchland", 3, 35)],
            "train-00001-of-00002": [("xiao", 12, 101), ("dmfc-rsg", 7, 120), ("area2-bump", 9, 60)],
            "test-00000-of-00001": [("xiao", 5, 100)]}
    truth = {}
    for name, spec in rows.items():
        mats = [rng.poisson(0.3, (u, t)).astype(np.uint8) for _, u, t in spec]
        table = pa.table({"spike_counts": [m.tolist() for m in mats],
                          "subject_id": [f"s{i}" for i in range(len(spec))],
                          "session_id": [f"{name}-{i}" for i in range(len(spec))],
                          "segment_id": ["0"] * len(spec), "source_dataset": [s for s, _, _ in spec]}, schema=SCHEMA)
        pq.write_table(table, root / "data" / f"{name}.parquet")
        if name.startswith("train"):
            truth.update({(s, f"{name}-{i}"): m for i, ((s, _, _), m) in enumerate(zip(spec, mats))})
    return truth


def test_row_matrices_roundtrip():
    mats = [np.arange(12, dtype=np.uint8).reshape(3, 4), np.zeros((0, 0), np.uint8), np.ones((1, 7), np.uint8)]
    col = pa.chunked_array([pa.array([m.tolist() for m in mats[:2]], pa.list_(pa.list_(pa.uint8()))),
                            pa.array([mats[2].tolist()], pa.list_(pa.list_(pa.uint8())))])
    out = row_matrices(col)
    assert [o.shape for o in out] == [(3, 4), (0, 0), (1, 7)] and all(np.array_equal(a, b) for a, b in zip(out, mats))


def test_windows_and_exclusion(tmp_path):
    truth = write_pile(tmp_path)
    ds = PileWindows(tmp_path, "train", exclude_sources=LEAKED)  # variant without evaluation sessions (U14)
    assert ds.excluded == sorted(LEAKED) and not ds.leaked
    # kept: xiao 365 bins -> 7 windows, churchland 35 -> 0, xiao 101 -> 2, dmfc-rsg 120 -> 2
    assert len(ds) == 11 and len(ds.mats) == 4
    x = ds[7]  # first window of the xiao row in file 1
    m = truth[("xiao", "train-00001-of-00002-0")]
    assert x.shape == (12, 10, 5) and x.dtype == torch.float32
    assert torch.equal(x.reshape(12, 50), torch.from_numpy(m[:, :50].astype(np.float32)))
    assert torch.equal(ds[6].reshape(40, 50), torch.from_numpy(truth[("xiao", "train-00000-of-00002-1")][:, 300:350]
                                                               .astype(np.float32)))


def test_exclusion_guards(tmp_path, capsys):
    write_pile(tmp_path)
    with pytest.raises(ValueError, match="perichh"):
        PileWindows(tmp_path, "train", exclude_sources=["perichh"])  # typo must not silently keep leaked data
    ds = PileWindows(tmp_path, "train", exclude_sources=[])
    assert ds.leaked == sorted(LEAKED) and "WARNING" in capsys.readouterr().out
    assert len(ds) == 5 + 9 + 2 + 1  # perich 250 -> 5, dmfc 120 -> 2, area2 60 -> 1


def test_max_rows_is_seeded(tmp_path):
    write_pile(tmp_path)
    a = PileWindows(tmp_path, "train", exclude_sources=[], max_rows=3, seed=1)
    b = PileWindows(tmp_path, "train", exclude_sources=[], max_rows=3, seed=1)
    assert len(a.mats) == 3 and len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a.mats, b.mats))


def test_default_is_the_paper_pile(tmp_path, capsys):
    """U14: by default nothing is excluded, as in the paper, and the leaked sources are reported."""
    write_pile(tmp_path)
    ds = PileWindows(tmp_path, "train")
    assert ds.excluded == [] and ds.leaked == sorted(LEAKED) and len(ds) == 5 + 9 + 2 + 1
    assert "WARNING" in capsys.readouterr().out


def same_windows(a, b):
    return len(a) == len(b) and all(torch.equal(a[i], b[i]) for i in range(len(a)))


@pytest.mark.parametrize("workers", [1, 2])
def test_memmap_equals_parquet(tmp_path, workers):
    """U30: the converted copy gives exactly PileWindows' windows, for every exclusion and max_rows draw."""
    write_pile(tmp_path / "pile")
    out = convert_memmap(tmp_path / "pile", tmp_path / "mm", "train", workers)
    assert sorted(p.name for p in out.iterdir()) == ["meta.json", "rows.npz", "shard-00000.u8", "shard-00001.u8"]
    for kw in ({}, {"exclude_sources": LEAKED}, {"max_rows": 3, "seed": 1}, {"max_rows": 2, "seed": 5}):
        a, b = PileWindows(tmp_path / "pile", "train", **kw), PileMemmap(tmp_path / "mm", "train", **kw)
        assert same_windows(a, b) and a.hours == b.hours and (a.excluded, a.leaked) == (b.excluded, b.leaked)


def test_memmap_guards(tmp_path):
    write_pile(tmp_path / "pile")
    out = convert_memmap(tmp_path / "pile", tmp_path / "mm")
    with pytest.raises(ValueError, match="perichh"):
        PileMemmap(tmp_path / "mm", exclude_sources=["perichh"])  # same typo guard as PileWindows (U14)
    (out / "meta.json").unlink()  # meta.json is written last, so an interrupted conversion is never read
    with pytest.raises(FileNotFoundError, match="meta.json"):
        PileMemmap(tmp_path / "mm")


def test_memmap_empty_row_and_workers(tmp_path):
    """A row without units gives no windows; DataLoader workers open their own maps and read the same windows."""
    (tmp_path / "pile" / "data").mkdir(parents=True)
    rng = np.random.default_rng(0)
    mats = [rng.poisson(0.5, (6, 120)).astype(np.uint8), np.zeros((0, 0), np.uint8),
            rng.poisson(0.5, (3, 260)).astype(np.uint8)]
    pq.write_table(pa.table({"spike_counts": [m.tolist() for m in mats], "subject_id": ["s"] * 3,
                             "session_id": ["a", "b", "c"], "segment_id": ["0"] * 3,
                             "source_dataset": ["xiao", "kim", "xiao"]}, schema=SCHEMA),
                   tmp_path / "pile" / "data" / "train-00000-of-00001.parquet")
    convert_memmap(tmp_path / "pile", tmp_path / "mm")
    a, b = PileWindows(tmp_path / "pile"), PileMemmap(tmp_path / "mm")
    assert len(b) == 2 + 0 + 5 and same_windows(a, b)
    for k, (x, valid) in enumerate(DataLoader(b, batch_size=3, collate_fn=collate, num_workers=2)):
        rx, rvalid = collate([b[i] for i in range(3 * k, min(3 * k + 3, len(b)))])
        assert torch.equal(x, rx) and torch.equal(valid, rvalid)
