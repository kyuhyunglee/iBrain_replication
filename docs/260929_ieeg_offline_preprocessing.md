# iEEG offline preprocessing: feasibility and implementation plan

Status: **proposal, not implemented.** Written 2026-09-29. Nothing below has been run on real data; every dataset fact marked *(confirm)* must be checked against the files before the plan is fixed.

## 1. Why a separate offline stage

The paper states only two things about iEEG preprocessing: signals are resampled to 500 Hz, and recordings are cut into non-overlapping 1 s windows (Datasets, "Pretraining Datasets"). The reconstruction target is "the channel-normalized waveform patch" (Eq. 8), without saying over what the normalization is computed.

The current code does everything per 1 s window at load time. That is fine for normalization (now recording-level, applied in the Dataset, SPEC U4), but it does not work for the other steps:

| Step | Why it cannot run on 1 s windows at load time |
|---|---|
| High-pass / notch filtering | Filtering each 1 s piece separately creates edge transients at every window boundary, and a 0.5 Hz high-pass cannot act within 1 s at all. Filters must run on the continuous signal |
| Resampling to 500 Hz | Same boundary problem for non-integer ratios (e.g. 512 → 500), and repeating it every epoch over 14.2 M windows is wasted compute |
| Bad channel / bad segment detection | Needs statistics over minutes to hours, not one window |
| Normalization statistics (U4) | Need the whole recording (or a long chunk). Computing them at every Dataset construction means reading all data twice |

So the proposal is: **filter, resample, flag and compute statistics once per recording, write the result to shards, and keep only `channel_normalize` + patching in the Dataset.** Keeping normalization at load time (instead of baking it into the shards) lets us change the normalization rule without re-running the expensive steps.

## 2. Feasibility

**Data volume.** Paper counts: AJILE12 4,662,027 windows (1,295 h), SWEC 9,560,582 windows (2,656 h), total 3,950.7 h. Storage of the processed signal depends on the channel count, which is not in the paper. As a rough scale, at an average of 100 channels:

| dtype | Size |
|---|---|
| float32 | 14.2 M × 100 × 500 × 4 B ≈ 2.8 TB |
| float16 | ≈ 1.4 TB |

float16 is enough if shards store the **already scaled** signal (values of order 1, clipped; see 3.6) and not raw µV. If shards store filtered µV, float32 is safer. Check free space on `/scratch` and `/storage` before choosing.

**Compute.** Every step is CPU-only (scipy filtering, `resample_poly`, percentiles). It parallelizes trivially over recordings. On the lab server it must still go through Slurm as a CPU job (`sbatch` with `--gres` omitted), not the login shell. No GPU is needed.

**Code.** It fits next to the existing layout without touching the model: one preprocessing module, one CLI script, one Dataset class that reads the shards. The spike side already follows the same pattern (nhp-spike-corpus shards → `data_spike.SpikeWindows`).

**Sanity check we get for free.** Counting the windows we produce per dataset and comparing with the paper's 4,662,027 / 9,560,582 tells us how much the authors discarded (NaN gaps, bad segments, tails). A large mismatch means our rejection rules differ from theirs.

## 3. Pipeline

Per recording (one continuous file; see 3.1 for what counts as one):

```
read → channel selection / bad channels → (re-reference) → notch → high-pass
     → resample to 500 Hz → channel statistics → bad-segment flags → cut 1 s windows → write shard
```

### 3.1 Reading and the unit of a "recording"

| Dataset | Source | Format / rate | Notes |
|---|---|---|---|
| AJILE12 | DANDI 000055 | NWB, ECoG at 500 Hz in the public release *(confirm)* | 12 subjects, several days each; files are split per subject per day *(confirm)*. If already 500 Hz, no resampling |
| SWEC iEEG (Carzaniga et al. 2025) | release linked from the paper *(confirm location and license)* | sampling rate 512 or 1024 Hz *(confirm)* | Long-term epilepsy monitoring; files likely split by hour *(confirm)* |
| Brain Treebank (downstream) | braintreebank.dev | 2048 Hz sEEG *(confirm)* | Per-movie sessions. Must go through the **same** pipeline |

A "recording" for U4 is one continuous file. If a file is much longer than a few hours (e.g. a full day), split it into fixed chunks (proposal: 1 h) and compute statistics per chunk, so slow changes (sleep/wake, impedance) do not share one set of statistics. Chunk boundaries never fall inside a window.

### 3.2 Channel selection and bad channels

- Keep only neural channels (drop EKG, EMG, trigger, reference, empty channels) using the channel metadata in each file.
- Flag a channel as bad over a recording if any of: flat (robust scale below a threshold), robust scale above ~5× or below ~0.2× the median over channels, a large fraction of NaN, or line-noise power dominating. Thresholds are proposals to tune after looking at the data.
- AJILE12 NWB files may already carry good/bad channel annotations *(confirm)*; use them if present and log disagreements with our rule.
- Bad channels are **dropped** from the shard (not zeroed), so they never appear as valid channels to V_c.

### 3.3 Re-referencing

Not stated in the paper. Options: none (as recorded), common average reference (CAR) over good channels, bipolar (sEEG). **Proposal: none** for pretraining, because the paper mentions no reference step and CAR mixes channels, which changes what channel attention sees. Record it as an open decision; revisit if line noise or common-mode artifacts dominate.

### 3.4 Filtering

- Notch at the mains frequency and harmonics below Nyquist of 500 Hz (50 Hz for SWEC in Switzerland, 60 Hz for AJILE12 in the US *(confirm)*; Brain Treebank is US, 60 Hz).
- High-pass at 0.5 Hz (zero-phase, `sosfiltfilt`), to remove DC and slow drift so that the per-recording median is close to 0 and window means are real signal, not electrode offset.
- No low-pass beyond the anti-aliasing filter of the resampler; 500 Hz keeps up to 250 Hz (high gamma).

### 3.5 Resampling

`scipy.signal.resample_poly` with the exact rational ratio (e.g. 512 → 500 is up 125, down 128; 2048 → 500 is up 125, down 512). Done after filtering, before statistics, on the continuous signal.

### 3.6 Statistics and scaling (U4)

- Per channel, per recording (or chunk): center = median, scale = IQR / 1.349, computed on good segments only (3.7). This is exactly `data_ieeg.channel_stats`.
- Store `center`, `scale` (float32, shape (C,)) in each shard. The Dataset applies `data_ieeg.channel_normalize` at load time.
- Optional clip after normalization (proposal ±20) to bound the MSE contribution of residual artifacts. If clipping is adopted, it goes into `channel_normalize` so it stays a load-time choice.

### 3.7 Bad segments and windowing

- Mark samples that are NaN, saturated (at the ADC rail), or part of a recording gap (discontinuous timestamps).
- Cut non-overlapping 1 s windows (500 samples) from the start of each continuous good stretch; discard any window touching a marked sample and the sub-1 s tail. This mirrors U21 on the spike side.
- **Epileptic activity.** Both pretraining sets come from epilepsy monitoring; SWEC includes seizures. The paper does not say whether ictal or interictal-spike periods were removed. **Proposal: keep them** (the paper's window counts look like whole-recording counts; verify with 2, "sanity check"), but store per-window seizure flags when annotations exist so this can be changed without reprocessing.

### 3.8 Shard format

Same layout idea as nhp-spike-corpus, so the two loaders look alike:

```
<root>/<dataset>/<subject>/<recording>/manifest.jsonl, shard-NNNNNN.npz
NPZ: signal float32|float16 [windows, C, 500]  (filtered, resampled, NOT normalized)
     center float32 [C], scale float32 [C]     (U4 statistics of this recording/chunk)
     start_seconds float64 [windows]
     channel_names str [C]
     seizure bool [windows] (optional)
manifest line: {shard, dataset, subject, recording, n_channels, n_windows, fs: 500, window_samples: 500,
                filters, reference, bad_channels, pipeline_version}
```

If storage forces float16, store the normalized signal instead and set center = 0, scale = 1 in the shard; the Dataset code does not change.

## 4. Where it plugs into the code

| File | Change |
|---|---|
| `ibrain/preprocess_ieeg.py` (new) | Steps 3.2–3.7 as pure functions on `(C, T)` arrays, testable on `SyntheticIEEG`-like signals |
| `scripts/preprocess_ieeg.py` (new) | CLI: one recording in, one shard directory out; resumable; run as a Slurm array job over recordings |
| `ibrain/data_ieeg.py` | Add `read_windows(root, dataset)` and `IEEGWindows(Dataset)`; `__getitem__` = `to_patches(channel_normalize(signal, center, scale))` |
| `scripts/pretrain.py` | Replace the `NotImplementedError` branch with `IEEGWindows`; the step count under 1:1 alternation is already handled by `pretrain.total_steps` (U20) |
| `configs/paper.yaml` | `data.ieeg: {root: ..., datasets: [ajile12, swec]}` |
| `SPEC.md` Section 2 | New decision rows for 3.1–3.8 (chunk length, bad channel rule, reference, filters, clip, bad segments, seizures, dtype) |

Memory note for the real run: the conv adapter (U1) holds about B·C·S × 256 × 50 activations per conv layer per forward, times three forwards per step (masked input + two views). At B = 32 and C = 128 that is about 2 GB per layer per forward. A per-window channel cap (random subset of good channels) may be needed; it is a training-time choice and does not affect the shards.

## 5. Verification before a full run

1. Run the pipeline on one AJILE12 day and one SWEC hour; plot raw vs processed PSD (notch and high-pass visible, nothing else changed) and a few windows.
2. Check the normalized distribution per channel: median ≈ 0, robust scale ≈ 1, fraction clipped small.
3. Count windows per dataset and compare with the paper (Section 2).
4. Unit tests: filter/resample on synthetic sinusoids (frequency preserved, amplitude preserved in the passband), window cutting never crosses a gap, bad channels never reach the shard.
5. A short pretraining run on the processed subset: iEEG reconstruction loss should start below 1 (predicting 0 gives roughly the normalized variance) and decrease.

## 6. Open decisions (to become SPEC rows)

| Decision | Proposal |
|---|---|
| Unit of normalization statistics | Recording, split into 1 h chunks when longer |
| Bad channel rule | Metadata annotations if present, else flat / outlier-scale / NaN rules |
| Re-reference | None |
| Filters | Notch at mains + harmonics, high-pass 0.5 Hz zero-phase |
| Clip | ±20 after normalization, at load time |
| Bad segments | Drop windows touching NaN, saturation or gaps |
| Seizure periods | Keep, store flags |
| Shard dtype | float32 unnormalized if storage allows, else float16 normalized |
