# iBrain Replication Specification

arXiv 2609.06960v1 (2026-09-07). No public code. We implement from the paper alone.
Written 2026-09-23, updated 2026-09-26. Code: Wooseok Choi. Kyuhyung Lee: data survey (division of work discussed 9/24).

## 0. Scope

**What we do now (spike-only baseline).** Implement the model as described in the paper faithfully, confirm the pipeline runs end to end on MC-Maze alone, and produce downstream numbers. M1–M4.
**All code gets written.** Everything described in the paper, including the iEEG encoder, decoder, and MSE loss (M5) and the 1:1 alternation (M6), is implemented and **verified on synthetic data**. Only real-data runs are deferred until access and compute are secured (AJILE12, SWEC, the 7,159-hour M7). The reason is data access, unrelated to DIVER.
**Reproducing the numbers is not the goal.** The paper does not state the number of seeds and has no code. Our deliverables are (1) code that works as the paper describes, (2) the list of decisions (Section 2), and (3) run records that keep seeds, checkpoints, and configs (U26). Gaps against the paper's numbers are only reported.

The replication has two deliverables. (1) Working code. (2) **The list of items not stated in the paper that we decided ourselves** (Section 2). The second is more valuable to hand to the team.

## 1. What the paper states

### 1.1 Input representation

| Item | Value |
|---|---|
| Window | 1 s |
| Patch | 100 ms × 10 ($S = 10$) |
| iEEG | 500 Hz, $P_{\text{iEEG}} = 50$ samples per patch. One channel is one sequence |
| Spike | **20 ms binning**, $P_{\text{spike}} = 5$ counts per patch. One unit is one sequence |
| Tensor | $X^m \in \mathbb{R}^{C_m \times S \times P_m}$ |
| Padding | Padding along the channel axis; a binary valid mask $V_c$ excludes padded positions from attention, pooling, and loss |

### 1.2 Per-signal encoder $f_m: \mathbb{R}^{P_m} \to \mathbb{R}^d$

| | Structure |
|---|---|
| $f_{\text{iEEG}}$ | Temporal CNN adapter + **residual linear path** (CNN captures local waveforms; the linear path maps samples directly into token space) |
| $f_{\text{spike}}$ | MLP |

Applied independently to each patch. Output $H^m \in \mathbb{R}^{C_m \times S \times d}$.

### 1.3 Embedding

$$Z^m_{c,s} = H^m_{c,s} + e^m_{\text{type}} + e^{\text{time}}_s$$

$e^m_{\text{type}}$ is one per signal type; $e^{\text{time}}_s$ is one per patch position. **No term depends on the channel index $c$.**

### 1.4 Backbone (criss-cross ST attention)

Each block runs two stages in order.

1. Channel attention: independently for each time index $s$, over the channel axis. $\tilde Z^m_{:,s} = \text{Attn}_{\text{ch}}(Z^m_{:,s})$
2. Temporal attention: independently for each channel $c$, over the time axis. $Z^m_{c,:} = \text{Attn}_{\text{time}}(\tilde Z^m_{c,:})$

A **learned relative time bias** is added to the temporal attention logits. Pre-norm, residual, FFN.

| Hyperparameter | Value |
|---|---|
| Layers | 6 |
| $d$ | 256 |
| Heads | 8 |
| FFN | 1024 |
| dropout | 0.1 |

Output $U^m = F_{\text{ST}}(Z^m) \in \mathbb{R}^{C_m \times S \times d}$. The backbone is **shared** by both signal types.

### 1.5 Pretraining objective

**Masked reconstruction.** Mask $M \in \{0,1\}^{C_m \times S}$, **50%** of valid tokens. Masking is applied in raw signal space ($\tilde x = (1-M)x$) before the encoder. The decoder $g_m$ is a 2-layer MLP, $d \to P_m$. The loss is computed only at masked valid positions.

- iEEG: target is the **channel-normalized waveform patch**. $\ell = \frac{1}{P}\|\hat x - x\|_2^2$
- Spike: $\lambda = \text{softplus}(\hat x) + \epsilon$, Poisson NLL

**channel-view alignment.** From the same segment, two views each with **about 80%** of the valid channels, **75% overlap**, **at least 2 channels**. Each view → encoder → backbone → global pooling over valid tokens → segment representation $r_1, r_2$ → projection $q$ (**128-dim**) → prediction $p$. SimSiam:

$$\mathcal{L}_{\text{align}} = \tfrac12 D(p_1, \text{sg}(q_2)) + \tfrac12 D(p_2, \text{sg}(q_1)), \quad D = -\cos$$

**Schedule.** iEEG batches and spike batches **alternate 1:1**. One type per step. $\mathcal{L}_t = \mathcal{L}^{m_t}_{\text{rec}} + \mathcal{L}^{m_t}_{\text{align}}$.

### 1.6 Training setup

| | Value |
|---|---|
| epoch | 30 |
| Optimizer | AdamW, lr $5\times10^{-4}$, wd $5\times10^{-2}$ |
| clip | 1.0 |
| Schedule | warmup 2,000 steps → cosine → $1\times10^{-5}$ |
| Compute | 8 GPUs, batch 32 per GPU per type |

### 1.7 Data

| | Pretraining | Downstream |
|---|---|---|
| iEEG | AJILE12 + SWEC, 3,950 h | Brain Treebank (Pitch, Volume, Onset, Speech), AUC |
| Spike | Neural Pile, 3,209 h | MC-Maze, Area2-Bump, Perich T-CO, T-RT, $R^2$ |

### 1.8 Downstream protocol

- Spike: **80/20** split within each dataset. **Full-parameter fine-tuning** + lightweight task head
- iEEG: BrainBERT/PopT subject-specific protocol (fine-tune per subject)
- Paper numbers: MC-Maze 0.914 / Area2 0.903 / T-CO 0.785 / T-RT 0.692. Frozen: MC-Maze 0.904, Area2 0.867

## 2. Not stated in the paper and our decisions

This table is the core deliverable of the replication. Any change to a decision is recorded here.

| # | Not stated | Our decision (draft) | Rationale |
|---|---|---|---|
| U1 | Number of layers and kernel size of $f_{\text{iEEG}}$ | 3-layer Conv1d (k=5, padding 2, channels 1→64→128→$d$, GELU) → **mean pooling over the time axis** → $d$. A residual Linear(50→$d$) is added on top | Minimal form of "temporal conv adapter + residual linear". The paper does not say how the conv output is reduced to a single token, so mean pooling. **Cost note:** about 20 MFLOP per token ($d = 256$), comparable to the 6-layer backbone. Fine on GPU, but on CPU it is about 50× the spike encoder, so local tests keep the channel count small |
| U2 | Number of layers of $f_{\text{spike}}$ | Linear(5→256) → GELU → Linear(256→256) | 2-layer MLP |
| U3 | Form of the relative time bias | One scalar per offset $s-s'$, learned per head (T5-style) | Simplest |
| U4 | Scope of "channel normalization" | Per-channel z-score **within the 1 s window** (mean and std over the $S \times P = 500$ samples). **Both input and target** use the same normalized values; masking is applied after normalization | Consistent with segment-level processing. The paper says only the target is "channel-normalized" and does not state anything about the input. Without input normalization, per-channel scales (tens to hundreds of µV) would enter the encoder as-is |
| U5 | Poisson NLL form | `PoissonNLLLoss(log_input=False, full=False)`, $\epsilon = 10^{-6}$ | Exactly Eqs. (10)(11) |
| U6 | Global pooling | Mean over valid tokens | "globally pooled" |
| U7 | SimSiam head sizes | Projection 256 → 256 → 128, prediction 128 → 64 → 128 | SimSiam convention (predictor bottleneck 1/4). No BN |
| U8 | **Task head structure** | Regression: per time patch, **attention pooling** over the channel axis (one learned query) → Linear(256→2) | Mean pooling loses unit identity; flattening is session-specific. **This choice determines how frozen results are interpreted. Run all 3 and compare** |
| U9 | Temporal resolution of regression output | One per patch (100 ms) → behavior labels averaged to 100 ms | Matches the token grid. Finer output is not possible with this architecture |
| U10 | Which part of Neural Pile | **The pile is not used directly.** Take the pile's composition list (primate), remove the evaluation sets, and **rebuild the rest from the original DANDI NWB files** | The pile has only 20 ms counts, with no spike times, LFP, coordinates, or behavior. Also, the pile contains Perich 000688, Area2-bump 000127, and DMFC-rsg 000130 (confirmed in README, 2026-09-23) |
| U11 | MC-Maze variant | 80/20 within the NLB MC-Maze (DANDI 000128) train file. Locked as `evaluation_only` in the nhp-spike-corpus catalog | The paper does not say L/M/S. The NLB test file has hidden labels, so the paper's 80/20 can only be a split within train |
| U12 | Seeds | 3 seeds, mean ± std | The paper does not mention seeds |
| U13 | Baselines | Must include **from scratch (same architecture, no pretraining)** + **binned firing rates + ridge** | Neither is in the paper |
| U14 | Overlap between pretraining corpus and downstream | **Any dandiset used for evaluation is excluded from pretraining entirely.** MC-Maze (000128) is not in the pile, but Churchland 000070 from the same monkey Jenkins is; decide after checking whether sessions coincide | iBrain did not do this check. The Area2-Bump, T-CO, and T-RT results overlap with the pretraining data |
| U15 | Mask sampling | Per window, **exactly 50%** of valid tokens (number of valid units × 5; divides evenly since $S = 10$), sampled uniformly at random on the $(C, S)$ grid. No channel-wise or time-wise block structure. Padded channels are never sampled | "50% of the valid tokens" read literally as a ratio (exact ratio, MAE-style). Unlike Bernoulli, the number of masked tokens per window is fixed |
| U16 | Channel view sampling | Randomly rank the $n$ valid channels and split them into $n - 2u$ shared and $u = \mathrm{round}(0.2n)$ exclusive per view. View = shared + exclusive. Size $n - u \approx 0.8n$, overlap (shared / view size) $\approx 0.75$, union of the two views = all valid channels. If $n = 2$, both views are the full 2 channels; if $n = 1$, both views are that single channel (minimum of 2 impossible; the alignment loss is still computed) | "75% overlap" read as the shared fraction relative to view size. Since $0.8 \times 0.75 = 0.6$, it splits exactly into 60% shared + 20% exclusive × 2. With $n = 3$ the configuration matches the bottom panel of Fig. 2 (2 channels each, 1 shared). For $n \ge 2$ the minimum of 2 channels holds without special handling |
| U17 | Whether alignment view inputs are masked | Views are built from the **unmasked raw input**. 3 forward passes per step (1 full masked input + 2 views), all on the same batch. Views do not slice out channels; instead the view mask is placed in the $V_c$ slot for encoding (channels outside the view drop out of attention and pooling through the same path as padding; output identical to slicing) | The text describes a view only as a channel subset of the same segment and does not mention masking. The input tokens in the bottom panel of Fig. 2 show no mask marks either. Simplest interpretation that does not mix the two augmentations |
| U18 | Loss combination and batch-axis averaging | $\mathcal{L} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{align}}$, weights 1:1 (exactly Eq. (13)). $\mathcal{L}_{\text{rec}}$ is the mean over all masked valid tokens in the batch (as in M1 `masked_poisson_nll`; windows with more units weigh more), $\mathcal{L}_{\text{align}}$ is the mean over windows. Batches are drawn at random across mixed sessions and padded to the maximum unit count in the batch, with no cap on unit count | Eq. (11) only writes the normalization within a single segment |
| U19 | Optimizer details | AdamW $\beta = (0.9, 0.999)$, $\epsilon = 10^{-8}$ (PyTorch default). Weight decay on **all parameters** (bias, LayerNorm, and embeddings are not excluded). Clip is global L2 norm 1.0 | The paper gives only the lr, wd, and clip values. No exclusion rule is the simplest |
| U20 | lr schedule details and training length | Warmup is linear from 0 (0 at step 0, $5\times10^{-4}$ at step $W$), then cosine down to $1\times10^{-5}$ at the last step. Training length is given as a **total step count** instead of epochs. The loader reshuffles and cycles when exhausted. For spike-only (M3), the paper's 30 epochs = 30 × (number of loader batches) steps | Under 1:1 alternation the two corpora differ in size, so an epoch is ambiguous |
| U21 | 1 s window boundaries and input values | Follows the corpus rule. Non-overlapping 1 s windows from the start of each trial (observation interval); the sub-1 s tail at the end of a segment and inter-trial intervals are discarded. Input is the raw counts as float without normalization (same values as the Eq. (11) target) | The paper only says the whole recording was split into non-overlapping 1 s windows. The corpus never creates windows that cross a trial |
| U22 | Embedding initialization | Both `type_emb` and `time_emb` are $\mathcal{N}(0, 0.02^2)$ | With the PyTorch `nn.Embedding` default (std 1), a constant vector of norm about 16 would overwhelm the token content (norm about 2). Embedding initialization convention of BERT and ViT |
| U23 | Final LayerNorm on the backbone output | **Kept.** One LayerNorm after the 6 blocks (`Backbone.norm`) | Not in the paper. Pre-norm transformer convention (GPT-2, ViT). Stabilizes the scale of the input to pooling, decoder, and task head. Note when interpreting frozen results that the representation includes this layer |
| U24 | Inactive-type parameters under 1:1 alternation | `type_emb` is split into two per-type `nn.Parameter`s (`ParameterList`). Encoders and decoders are also per-type modules | On a step for one type, the other type's parameters have grad `None`, so AdamW skips them (no decay, no momentum). Resolves the M6 caveat. Verified by `test_type_isolation` |
| U25 | Encoder/decoder activation and decoder hidden width | All GELU. Decoder $g_m$: Linear($d$→$d$) → GELU → Linear($d$→$P_m$), same structure for both types | The $\phi$ in Eq. (7) and the hidden width are not stated in the paper |
| U26 | **Replication protocol** (seeds, checkpoints, records) | (a) Seeds: `seed_all(seed)` fixes `random`, `numpy`, and `torch` (including CUDA/MPS). DataLoader uses `torch.Generator().manual_seed(seed)`. The pretraining seed and the split seed (U11) are separate. (b) Per run, `out/meta.json`: full config, seed, git hash (`-dirty` if there are uncommitted changes), torch and Python versions, platform, argv, timestamp. (c) `out/log.jsonl`: one line `{step, sig, lr, rec, align}` per step. (d) `out/ckpt.pt` (overwritten every `ckpt_every` steps) and `out/final.pt`: model and optimizer state, step, full log, meta, RNG state (torch CPU/CUDA, numpy, python). (e) `--resume` restores everything in (d). Only the loader order is reshuffled (mid-epoch position is not saved). (f) Results are mean ± std over 3 seeds, with per-seed values also in `results.json` | The paper does not state the number of seeds or the variance. We leave every run so it can be rerun exactly as it was |
| U27 | Fine-tuning details (M4) | Loss MSE (2-dim velocity per patch). AdamW lr $10^{-4}$ fixed (no schedule), wd $5\times10^{-2}$, clip 1.0, batch 32, 50 epochs fixed, evaluate the last-epoch model (no validation set). Metric $R^2$ = mean of per-output-dimension $R^2$ (sklearn default), over all test patches. Ridge baseline: features = the patch's $P \times C$ 20 ms counts (standardized) → velocity; $\lambda \in \{10^{-2}, \dots, 10^3\}$ chosen by an inner trial-level 80/20 validation within train, then refit on all of train. Frozen arm: backbone `requires_grad=False`, eval mode (no dropout), only the head is trained | The paper states only "full-parameter fine-tuning + lightweight head" and $R^2$. Holding out a validation set would break the 80/20, so epochs are fixed |
| U28 | iEEG synthetic verification | Until real data is available, M5 and M6 code is verified with `SyntheticIEEG` (3 sinusoids + noise per channel, per-channel scale 10–500 and DC offset). The real-data loader (the AJILE12/SWEC part of `data_ieeg.py`) comes after access | The encoder, loss, and alternation loop can be verified independently of the data format |

## 3. Milestones and verification criteria

| # | Stage | Verification |
|---|---|---|
| M0 | This specification | Finalize Section 2 after team review |
| M1 | Model skeleton (spike encoder and decoder, backbone, SimSiam head) on synthetic data | forward/backward pass. Output shapes match Eqs. (1)(5)(7). **Parameter count around 6M** (6 blocks × about 1.05M + encoder). Loss decreases on synthetic data. **Status: done** (`test_shapes.py`). Measured 2026-09-24: 6.57M (about 6.8M including the iEEG encoder), 1.054M per block |
| M2 | Spike pipeline. **Attached as an adapter to the lab repository `Transconnectome/nhp-spike-corpus`.** The 20 ms counts for the iBrain replication only require regrouping that repository's `data/verified` shards (`uint32[windows, time_bins, units]`, 1 s window = 50 bins) into 10 patches of 100 ms. The spike-time adapter for POYO is separate (original NWB → spike times). No local download; runs on DGX | shape $(B, C, 10, 5)$. Count distribution matches firing rates. Valid mask correct. Only shards that pass `audit` are used. **Status: verified on synthetic data; real data after corpus access** |
| M3 | Spike-only pretraining (small scale). **Pretraining sources are the corpus's `pretrain_candidate` (currently Xiao 000628) and motor sets approved after review.** MC-Maze is **not used for pretraining**, per U14 and corpus policy (NLB evaluation only). For reference, more than half of the Neural Pile primate data used by iBrain is also Xiao | Loss curves qualitatively match Fig. 3 of the paper (spike recon low and stable, align small). Poisson NLL of reconstruction at masked positions is **lower than predicting each unit's mean firing rate** (masked patches are zero-filled and look like no firing, so compare against a firing-rate baseline rather than random). **Status: verified on synthetic data; real data after corpus access** |
| M4 | Downstream MC-Maze velocity regression. **First fix a trial-level 80/20 split within the NLB train file**, and all three arms use the same split. Corpus conversion uses the default 50 bins (1 s) | Three arms: pretrain+fine-tune / **from scratch** / **ridge**. Produce numbers. Report the gap against the paper's 0.89 (spike-only), but do not expect a match since pretraining is small scale. **Status: code done, verified on synthetic data** (`finetune.py`, `test_finetune.py`: on synthetic velocity→firing-rate data, ridge $R^2 > 0.5$, scratch $R^2 > 0$, same seed gives the same result). Remaining for real data: (1) confirm `read_nwb_velocity` on the NLB 000128 train NWB (unverified), (2) a rule to select only train assets among the corpus 000128 shards, (3) run the three arms |
| M5 | iEEG pipeline + encoder | shape $(B, C, 10, 50)$. **Status: encoder, decoder, Eqs. (8)(9), and channel normalization done, verified on synthetic data** (`test_ieeg.py`). AJILE12/SWEC loaders after access is secured |
| M6 | 1:1 alternating joint pretraining (small scale) | Both losses decrease. A batch of one type produces no gradient for the other type's encoder. **Status: loop done, verified on synthetic data** (`test_joint_pretrain_reduces_both`, `test_type_isolation`). The `type_emb` issue is resolved by U24. Real data after M5 |
| M7 | Scale-up | NERSC/lab server |

## 4. Environment

- Location `~/Desktop/Lab/Connectome/DIVER-POYO/260924/ibrain-repro/`
- Python 3.12, uv venv, torch 2.14
- Local M4 16 GB (MPS) for M1–M2, GPU thereafter
- Paths, GPU count, and batch size via config file. No NERSC-specific code

## 5. File layout

```
ibrain-repro/
  SPEC.md              this document
  pyproject.toml
  ibrain/
    model.py           encoders (spike MLP, iEEG conv+residual), backbone, decoders, SimSiam head, losses (Eqs. 1-12)
    data_spike.py      nhp-spike-corpus shards -> (C, 10, 5) patches, channel padding, synthetic shards
    data_ieeg.py       (C, 500) waveforms -> (C, 10, 50) patches, synthetic iEEG. Real-data loader is M5
    pretrain.py        masking, view generation, step loss (Eq. 13), 1:1 alternation loop, checkpoints and logs
    finetune.py        patch labels (U9), trial split (U11), 3 head types (U8), three arms (U13), R²
    repro.py           seeds, meta, RNG state, checkpoints (U26)
  configs/
    tiny.yaml          local verification (synthetic data, CPU)
    paper.yaml         paper settings (only paths adapted to the environment)
  scripts/
    pretrain.py        pretraining CLI (--config --seed --out --resume)
    finetune.py        downstream CLI (three arms × seeds -> results.json)
    pretrain.sbatch    lab-server Slurm template (auto-resume from ckpt.pt on preemption)
  tests/
    test_shapes.py     M1
    test_data.py       M2
    test_pretrain.py   M3 + checkpoint and resume
    test_ieeg.py       M5 + M6 (synthetic)
    test_finetune.py   M4 (synthetic)
```

## 6. How to run

```bash
# Local verification (synthetic data, tests take about 2 min; the iEEG conv adapter is slow on CPU)
.venv/bin/python -m pytest -q
.venv/bin/python scripts/pretrain.py --config configs/tiny.yaml --seed 0 --out runs/tiny_s0
.venv/bin/python scripts/finetune.py --config configs/tiny.yaml --ckpt runs/tiny_s0/final.pt --out runs/ft_tiny --synthetic

# Lab server (GPU jobs go through Slurm only)
sbatch scripts/pretrain.sbatch configs/paper.yaml 0 runs/paper_s0   # submit seeds 0, 1, 2 separately
# Real-data M4 (after corpus and NWB access. read_nwb_velocity unverified)
python scripts/finetune.py --config configs/paper.yaml --ckpt runs/paper_s0/final.pt --out runs/ft_mcmaze_s0 \
    --corpus /scratch/connectome/nhp-spike-corpus/data/verified --source dandi:000128 --nwb <NLB train NWB>
```

The outputs of a single run are `out/meta.json`, `log.jsonl`, `ckpt.pt`, `final.pt` (pretraining) or `results.json` (downstream). When reporting, include the git hash, seed, and config from `meta.json` (U26).
