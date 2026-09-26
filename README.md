# ibrain-repro
Replication codebase for iBrain, written from the paper since no official code was released
[[arxiv]](https://arxiv.org/abs/2609.06960)
[[spec]](SPEC.md)

Everything described in the paper is implemented and tested on synthetic data. No real-data run has been done yet.
The choices the paper leaves open, and why we made them, are listed in `SPEC.md` section 2.

### Installation

First, install `uv` by following the steps [here](https://docs.astral.sh/uv/getting-started/installation/). Then, create your Python environment:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

To check that everything works, run the test suite. It takes about two minutes on CPU.

```bash
pytest
```

### Pretraining on synthetic data
To see the whole pipeline run before you have any data, train on the built-in synthetic spikes and iEEG:

```bash
python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
```

This writes `meta.json`, `log.jsonl`, `ckpt.pt` and `final.pt` under `runs/tiny_s0`.
Checkout `configs/tiny.yaml` for all configurations available.

### Pretraining on spikes
To pretrain on real spikes you first need the 20 ms count shards from the lab corpus [`nhp-spike-corpus`](https://github.com/Transconnectome/nhp-spike-corpus).
Set `data.corpus_root` in `configs/paper.yaml` to its `data/verified` directory and list the sources to use under `data.spike_sources`. Evaluation datasets must not be in that list.

Then you can pretrain by running:

```bash
python scripts/pretrain.py --config configs/paper.yaml --seed 0 --out runs/paper_s0
```

On the lab server, submit the same run through SLURM instead. GPU jobs must not be launched from a login shell.

```bash
sbatch scripts/pretrain.sbatch configs/paper.yaml 0 runs/paper_s0
```

We run seeds 0, 1 and 2 for every configuration. To continue an interrupted run, add `--resume runs/paper_s0/ckpt.pt`. The SLURM script does this by itself when a checkpoint exists.
Checkout `configs/paper.yaml` for all configurations available.

### Evaluating on MC-Maze
To evaluate you first need the MC-Maze counts from the corpus (`dandi:000128`) and the NLB train NWB file for the hand velocity labels.

Then you can run the three arms (fine-tuning from the checkpoint, the same model from scratch, and ridge regression on binned counts) by running:

```bash
python scripts/finetune.py --config configs/paper.yaml --ckpt runs/paper_s0/final.pt --out runs/ft_mcmaze_s0 \
    --corpus /path/to/nhp-spike-corpus/data/verified --source dandi:000128 --nwb /path/to/mc_maze_train.nwb
```

All arms share one trial-level 80/20 split. R² for every arm and seed goes to `runs/ft_mcmaze_s0/results.json`.
To check the evaluation code without data, replace the last three flags with `--synthetic`.
The NWB reader has not been run on real data yet, so expect to touch it on the first try.

### iEEG
The iEEG encoder, decoder and loss are in place and covered by the tests. Loaders for AJILE12 and SWEC will be added to `ibrain/data_ieeg.py` once we have the recordings, so `data.ieeg` stays `null` in `configs/paper.yaml` until then.

## Cite
This repository is not affiliated with the iBrain authors. Please cite [their paper](https://arxiv.org/abs/2609.06960) if you use this code in your own work:

```bibtex
@article{
    chen2026ibrain,
    title={iBrain: A Unified Foundation Model Reading the Brain from Surface to Spikes},
    author={Ying Chen and Tiou Wang and Zhifeng Yue},
    journal={arXiv preprint arXiv:2609.06960},
    year={2026},
}
```
