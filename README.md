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
To pretrain on real spikes you first need the [Neural Pile primate](https://huggingface.co/datasets/eminorhan/neural-pile-primate) parquet files, the corpus the paper used.
Set `data.spike.root` in `configs/paper.yaml` to the folder that holds them. As in the paper the whole pile is used, although it contains the Perich and Area2-Bump evaluation sessions; set `data.spike.exclude_sources: [perich, area2-bump]` for the variant without them. Each step averages 8 batches of 32 (`grad_accum`), the paper's 8 GPUs x 32.

Then you can pretrain on spikes only by running:

```bash
python scripts/pretrain.py --config configs/paper.yaml --seed 0 --out runs/spike_s0 --spike-only
```

On the lab server, submit the same run through SLURM instead. GPU jobs must not be launched from a login shell.

```bash
sbatch scripts/pretrain.sbatch configs/paper.yaml 0 runs/spike_s0 --spike-only
```

The pile is held in memory, so set `data.spike.max_rows` for a small first run. We run seeds 0, 1 and 2 for every configuration. To continue an interrupted run, add `--resume runs/spike_s0/ckpt.pt`. The SLURM script does this by itself when a checkpoint exists.
Checkout `configs/paper.yaml` for all configurations available.

### Evaluating on MC-Maze
To evaluate you first need the NLB MC-Maze files from [DANDI 000128](https://dandiarchive.org/dandiset/000128). The spikes and the hand velocity labels are both read from the NWB files.

Then you can run the three arms (fine-tuning from the checkpoint, the same model from scratch, and ridge regression on binned counts) by running:

```bash
python scripts/finetune.py --config configs/paper.yaml --ckpt runs/paper_s0/final.pt --out runs/ft_mcmaze_s0 \
    --nwb /path/to/MC_Maze --behavior hand_vel
```

A folder is searched for `.nwb` files, and files without the behavior series, such as the NLB test file, are skipped.
All arms share one trial-level 80/20 split. R² for every arm and seed goes to `runs/ft_mcmaze_s0/results.json`.
Area2-Bump ([DANDI 000127](https://dandiarchive.org/dandiset/000127)) works the same way. For the Perich T-CO and T-RT tasks ([DANDI 000688](https://dandiarchive.org/dandiset/000688)), pass the monkey T files of one task, for example `sub-T/*ses-CO*`, and use `--behavior cursor_vel`.
To check the evaluation code without data, replace the last two flags with `--synthetic`.
The NWB reader has been tested on synthetic files in both layouts but not yet on the real files, so expect to touch it on the first try.

### Joint pretraining with iEEG
To add iEEG you first need AJILE12 ([DANDI 000055](https://dandiarchive.org/dandiset/000055)) and the [SWEC-ETHZ](https://huggingface.co/datasets/NeuroTec/SWEC_iEEG_Dataset) HDF5 files. Reading SWEC needs `hdf5plugin`, which is installed with this package.
Set the two `root` paths under `data.ieeg` in `configs/paper.yaml`. Recordings are streamed block by block, AJILE12 keeps only good channels and skips Blocklist epochs, and SWEC is resampled from 1024 to 500 Hz.

Then you can run the paper's joint pretraining, alternating spike and iEEG batches, by leaving out `--spike-only`:

```bash
sbatch scripts/pretrain.sbatch configs/paper.yaml 0 runs/joint_s0
```

### Brain Treebank
The [Brain Treebank](https://braintreebank.dev) reader builds the Pitch, Volume, Onset and Speech examples with the PopT rules, cut to 1 s windows at 500 Hz (SPEC U34). `ibrain.data_treebank.split_subject(root, "sub_1", "speech", cache_dir=...)` returns train, val and test windows, testing on the subject's held-out PopT trial. A command-line entry point and the classification head with AUC are not implemented yet.

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
