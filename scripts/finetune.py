"""M4 downstream. Three arms (finetune / scratch / ridge) × seeds, same trial split. Results in out/results.json.
Example: python scripts/finetune.py --config configs/tiny.yaml --ckpt runs/tiny_s0/final.pt --out runs/ft_tiny --synthetic
Real data: --corpus <root> --source dandi:000128 --nwb <NLB train NWB>  (real data unverified, SPEC M4)"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import numpy as np
import torch
import yaml

from ibrain.data_spike import read_windows
from ibrain.finetune import attach_velocity, fit_arm, read_nwb_velocity, ridge_r2, split_trials, synthetic_labeled
from ibrain.repro import run_meta, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Pretraining checkpoint. If absent, the finetune arm is skipped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--source", default="dandi:000128")
    ap.add_argument("--nwb", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    fc = cfg["finetune"]
    out = Path(a.out)
    write_json(out / "meta.json", {**run_meta(cfg, None), "ckpt": a.ckpt})

    if a.synthetic:
        windows = synthetic_labeled(200, 30, seed=fc["split_seed"])
    else:
        ts, vel = read_nwb_velocity(a.nwb)
        windows = attach_velocity(list(read_windows(a.corpus, a.source)), ts, vel)
    train, test = split_trials(windows, 0.8, fc["split_seed"])  # all three arms use the same split (M4)
    print(f"windows train={len(train)} test={len(test)}")

    results = {"split_seed": fc["split_seed"], "n_train": len(train), "n_test": len(test), "arms": {}}
    score, lam = ridge_r2(train, test, seed=fc["split_seed"])
    results["arms"]["ridge"] = {"r2": [score], "lambda": lam}
    arms = {"scratch": None} if a.ckpt is None else {"finetune": a.ckpt, "scratch": None}
    for name, ckpt in arms.items():
        for frozen in ([False, True] if ckpt else [False]):
            key = name + ("_frozen" if frozen else "")
            r2s = []
            for seed in fc["seeds"]:
                r2, hist = fit_arm(train, test, cfg["model"], seed, ckpt=ckpt, head=fc["head"], frozen=frozen,
                                   epochs=fc["epochs"], lr=fc["lr"], wd=fc["wd"], batch_size=fc["batch_size"],
                                   device=a.device)
                r2s.append(r2)
                print(f"{key} seed={seed} R2={r2:.4f} loss {hist[0]:.4f} -> {hist[-1]:.4f}")
            results["arms"][key] = {"r2": r2s, "seeds": fc["seeds"]}
    for k, v in results["arms"].items():
        v["mean"], v["std"] = float(np.mean(v["r2"])), float(np.std(v["r2"]))  # U12
        print(f"{k:16s} R2 = {v['mean']:.4f} ± {v['std']:.4f}")
    write_json(out / "results.json", results)


if __name__ == "__main__":
    main()
