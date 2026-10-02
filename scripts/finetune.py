"""M4 downstream. Three arms (finetune / scratch / ridge) × seeds, same trial split. Results in out/results.json.
Example: python scripts/finetune.py --config configs/tiny.yaml --ckpt runs/tiny_s0/final.pt --out runs/ft_tiny --synthetic
Real data: --nwb <files or folders> --behavior hand_vel (NLB MC-Maze, Area2-Bump) or cursor_vel (Perich).
Files without that behavior series (NLB test files) are skipped. Not yet run on real data (SPEC M4)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root, so `ibrain` imports without install

import numpy as np
import torch
import yaml

from ibrain.data_nwb import MissingSeries, nwb_files, read_nwb
from ibrain.finetune import fit_arm, ridge_r2, split_trials, synthetic_labeled
from ibrain.repro import run_meta, write_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None, help="Pretraining checkpoint. If absent, the finetune arm is skipped")
    ap.add_argument("--out", required=True)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--nwb", nargs="+", default=None, help="NWB files or folders (searched recursively)")
    ap.add_argument("--behavior", default="hand_vel", help="TimeSeries to decode: hand_vel (NLB), cursor_vel (Perich)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(Path(a.config).read_text(encoding="utf-8"))
    fc = cfg["finetune"]
    out = Path(a.out)
    if a.synthetic:
        windows, used = synthetic_labeled(200, 30, seed=fc["split_seed"]), []
    else:
        if not a.nwb:
            ap.error("give --nwb or --synthetic")
        windows, used = [], []
        for f in nwb_files(a.nwb):
            try:
                ws = read_nwb(f, a.behavior)
            except MissingSeries:
                print(f"skip {f.name}: no {a.behavior!r} series (no labels)")
                continue
            print(f"{f.name}: {len(ws)} windows, {ws[0]['counts'].shape[1] if ws else 0} units")
            windows += ws
            used.append(str(f))
        if not windows:
            sys.exit(f"no labeled windows found for behavior {a.behavior!r}")
    write_json(out / "meta.json", {**run_meta(cfg, None), "ckpt": a.ckpt, "behavior": a.behavior, "nwb_files": used})
    train, test = split_trials(windows, 0.8, fc["split_seed"])  # all three arms use the same split (M4)
    print(f"sessions={len({w['session_id'] for w in windows})} windows train={len(train)} test={len(test)}")

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
