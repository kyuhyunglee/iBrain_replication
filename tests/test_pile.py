"""M3 checks on a synthetic Neural Pile with the real schema (spike_counts list<list<uint8>> (n_units, t),
subject_id, session_id, segment_id, source_dataset) and HF file naming."""
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from ibrain.data_pile import LEAKED, PileWindows, row_matrices

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
    ds = PileWindows(tmp_path, "train")  # default excludes perich, area2-bump, dmfc-rsg (U14)
    assert ds.excluded == sorted(LEAKED) and not ds.leaked
    # kept: xiao 365 bins -> 7 windows, churchland 35 -> 0, xiao 101 -> 2
    assert len(ds) == 9 and len(ds.mats) == 3
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
