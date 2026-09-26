"""Run pretraining. Example: python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
Outputs: out/meta.json, log.jsonl, ckpt.pt (latest), final.pt. Resume with --resume out/ckpt.pt."""
import argparse
import sys
import itertools
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import torch
import yaml
from torch.utils.data import DataLoader

from ibrain.data_ieeg import SyntheticIEEG
from ibrain.data_spike import SYNTHETIC, SpikeWindows, collate, read_windows, write_synthetic
from ibrain.model import IBrain, IEEG, SPIKE
from ibrain.pretrain import pretrain
from ibrain.repro import run_meta, seed_all, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    out = Path(a.out)
    seed_all(a.seed)
    meta = run_meta(cfg, a.seed)
    write_json(out / "meta.json", meta)

    pc, dc = cfg["pretrain"], cfg["data"]
    if dc.get("synthetic"):
        root = out / "data"
        if not root.exists():
            write_synthetic(root, (3, 8, 20), n_windows=16, seed=a.seed)
        windows = read_windows(root, SYNTHETIC)
    else:
        windows = itertools.chain.from_iterable(read_windows(dc["corpus_root"], s) for s in dc["spike_sources"])
    g = torch.Generator().manual_seed(a.seed)
    dl_spike = DataLoader(SpikeWindows(windows), pc["batch_size"], shuffle=True, collate_fn=collate, generator=g,
                          num_workers=pc.get("num_workers", 0))
    loaders = [(SPIKE, dl_spike)]
    if dc.get("ieeg") == "synthetic":
        loaders.append((IEEG, DataLoader(SyntheticIEEG(n_windows=16, seed=a.seed), pc["batch_size"], shuffle=True,
                                         collate_fn=collate, generator=g)))
    elif dc.get("ieeg"):
        raise NotImplementedError("iEEG real-data loader is M5 (after AJILE12/SWEC access)")
    steps = pc["steps"] or pc["epochs"] * len(dl_spike)  # U20
    print(f"steps={steps} spike batches/epoch={len(dl_spike)} loaders={len(loaders)} device={a.device}")

    model = IBrain(**cfg["model"])
    pretrain(model, loaders, steps, pc["warmup"], a.device, out_dir=out, ckpt_every=pc["ckpt_every"],
             resume=a.resume, meta=meta)


if __name__ == "__main__":
    main()
