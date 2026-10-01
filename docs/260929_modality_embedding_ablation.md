# How much does the modality embedding matter? Counterfactual and probe experiments

Status: **proposal, not implemented.** Written 2026-09-29. Needs a checkpoint pretrained jointly on real spike and iEEG data (M6 on real data, or M7). Until then only the code path can be tested, on synthetic data.

## 1. Question

iBrain adds a learned type embedding $e_{\text{type}}$ to every token before the shared backbone (Eq. 2). The paper's claim is that one backbone learns a representation shared by both signal types. Two questions follow:

1. **Does the model use the marker?** If spike tokens get the iEEG marker, does performance drop?
2. **Does the backbone actually mix the two types into a common space**, or does it keep them apart (with or without the marker's help)?

Question 2 is the one we care about. Question 1 alone cannot answer it (see 2.2), so the experiment measures both, and all measurements are made **without fine-tuning**. Fine-tuning would let the model adapt to the changed condition and hide the effect.

## 2. What the marker is in our code

### 2.1 Facts

From `ibrain/model.py`:

```python
h = self.enc[sig](x)                                   # Eq. 1, per-type encoder
z = h + self.type_emb[sig] + self.time_emb             # Eq. 2
return self.backbone(z, valid)                         # Eq. 5, shared, 6 pre-norm blocks + final LayerNorm (U23)
```

- `type_emb` is two separate `nn.Parameter`s of size $d = 256$, initialized `randn * 0.02` (norm ≈ 0.32 at init, U22, U24).
- Encoders and decoders are also per-type (`enc[sig]`, `dec[sig]`). The spike input patch has 5 values, the iEEG patch 50.
- Under 1:1 alternation (Eq. 13) **every batch contains one type only**. The backbone never sees a window that mixes spike and iEEG tokens.

### 2.2 Consequences for interpretation

- **The marker is redundant as a parameter.** Both encoders end in a `Linear` with a bias, so $h + e_{\text{type}} = Wx + (b + e_{\text{type}})$. Any modality offset the model needs can live in the encoder bias $b$ instead. How the offset splits between $b$ and $e_{\text{type}}$ depends on initialization and weight decay, not on what the model needs. So "Swap does nothing" does not mean "the model does not need a modality offset". It only means this particular vector is not carrying it.
- **The marker is a constant offset within a batch.** Since a window holds one type, $e_{\text{type}}$ is the same vector on every token. Before the LayerNorm, a vector added to every key shifts every attention logit of a given query by the same amount, which softmax cancels. The marker can only act through the LayerNorm nonlinearity, the value path and the FFN, in effect as a per-type bias. This is a hypothesis about the mechanism, and it is a reason to expect a small effect.
- **Separate encoders make the input to the backbone easy to tell apart anyway.** The per-type encoder output $h$ is expected to separate the two types almost perfectly before any marker is added. The informative measurement is therefore **how separation changes with depth**, not whether it exists at the input.

For these reasons the layer-wise probe (4.2) is the primary measurement for question 2, and the counterfactual conditions (5) are the primary measurement for question 1.

## 3. Prerequisites

| Item | Needed for | State |
|---|---|---|
| Joint-pretrained checkpoint (spike + iEEG, real data) | Everything below | Not available. M6 is verified only on synthetic data |
| Held-out pretraining-type data: recordings/sessions **not seen in pretraining**, both types | Reconstruction loss, modality probe | Split not defined yet (open decision D1) |
| Downstream labeled data for frozen linear probes | 6.2 | Spike: MC-Maze velocity (M4 split). iEEG: Brain Treebank, after access |
| Reference losses for normalization | 6.1 | Spike: per-unit mean firing rate (as in M3). iEEG: predict 0 (the signal is median-centered, U4) |

If affordable, repeat everything on **3 pretraining seeds**. Otherwise one checkpoint, with confidence intervals from a bootstrap over held-out recordings.

## 4. Cheap diagnostics first

Both need one forward pass over a held-out sample and no training of the model. Run them before the counterfactual conditions: they tell us what to expect and may make parts of section 5 unnecessary.

### 4.1 Norm and direction of the marker

Compute, per modality $m$, over held-out unmasked tokens:

| Quantity | Meaning |
|---|---|
| $\lVert e_{\text{spike}} \rVert$, $\lVert e_{\text{iEEG}} \rVert$, $\lVert e_{\text{spike}} - e_{\text{iEEG}} \rVert$ | Marker size. The difference is what Swap changes |
| $\cos(e_{\text{spike}}, e_{\text{iEEG}})$ | Near 1 means the two markers barely differ |
| mean $\lVert h \rVert$ per modality | Encoder output size. Compare the marker norms against this |
| mean $\lVert e_{\text{time}}[s] \rVert$ | A second additive embedding, as a point of comparison |
| $\lVert b_m \rVert$ (last encoder bias) and $\cos(b_m, e_m)$ | Whether the encoder bias has taken over the offset (2.2) |
| mean $\lVert \mathrm{LN}_1(h + e_{\text{swap}}) - \mathrm{LN}_1(h + e_m) \rVert \,/\, \lVert \mathrm{LN}_1(h + e_m) \rVert$ | What the first block actually sees change under Swap, after its pre-norm |

If the marker difference is a few percent of $\lVert h \rVert$ and the post-LayerNorm change is equally small, a large Swap effect is unlikely.

### 4.2 Layer-wise modality probe

Train a linear classifier to predict the modality from token representations at each depth:

| Depth | Representation |
|---|---|
| 0a | $h$, encoder output **before** the marker |
| 0b | $z_0 = h + e_{\text{type}} + e_{\text{time}}$ |
| 1–6 | residual stream after block $\ell$ |
| 7 | $u$ = after the final LayerNorm (the representation used by pooling, decoders and heads, U23) |

Also at each depth, a second probe on window-level mean-pooled features (`IBrain.pool`), since downstream heads read pooled features.

Protocol:

- Inputs unmasked. Tokens sampled in equal numbers per modality. Features standardized. Logistic regression with L2, strength chosen by inner CV.
- **Split by recording**, not by token: train on some recordings, test on held-out ones. Otherwise the probe can learn recording identity.
- Caveat: modality is perfectly confounded with dataset, subject and species (spike = monkey Utah arrays, iEEG = human ECoG/sEEG). The probe measures separability of "these two corpora", which is an upper bound on separability of the signal types.

Accuracy will probably saturate near 100% at every depth ($d = 256$, many samples, separate encoders). To see a trend below saturation, also report per depth:

| Measure | Why |
|---|---|
| Difference-of-means (rank-1) probe accuracy | Separability along one direction only. Saturates much later than a full probe |
| Between-class / within-class variance ratio (Fisher ratio along the class-mean difference) | Continuous, no saturation |
| Principal angles between the top-$k$ PCA subspaces of the two modalities ($k$ at 90% variance) | Directly asks whether the two types occupy the same subspace. Small angles = shared space |

**Probe under Swap.** Take the probe trained on baseline representations at each depth and apply it to representations computed under Swap (5). Report the fraction of tokens classified as the **marker's** modality rather than the encoder's. If this fraction grows with depth, the marker steers the representation. If it stays near 0, the representation follows the encoder output.

## 5. Counterfactual conditions

All conditions change only the vector added in Eq. 2. Encoder, decoder, backbone and all other weights stay as trained. The decoder always matches the input type, so reconstruction stays well defined.

| Condition | Vector added to modality $m$'s tokens | Meaning |
|---|---|---|
| Baseline | $e_m$ | Reference |
| **Swap** | $e_{\bar m}$ (spike tokens get the iEEG marker, and the reverse) | Core counterfactual. No change means the model does not use the marker |
| Neutral | $\tfrac12(e_{\text{spike}} + e_{\text{iEEG}})$ | "No marker" with less distribution shift than removing it |
| Zero | $0$ | Reference only. A large gap between Zero and Neutral means the effect of Zero is mostly distribution shift |
| Random direction (control) | $e_m + r$, $r$ random with $\lVert r \rVert = \lVert e_{\text{spike}} - e_{\text{iEEG}} \rVert$, orthogonal to $e_{\text{spike}} - e_{\text{iEEG}}$; 10 draws | Is the model sensitive to the **modality direction** specifically, or to any perturbation of that size? |

Run every condition on the **same** windows with the **same** masks (fixed mask seed), so comparisons are paired.

## 6. Measurements

### 6.1 Held-out masked reconstruction loss

Per modality, the pretraining loss on held-out data: Poisson NLL for spike (Eq. 11), MSE for iEEG (Eq. 9), 50% masking as in pretraining (U15).

The two losses are on different scales, so do not compare them across modalities. Report per modality:

- $\Delta L = L_{\text{cond}} - L_{\text{baseline}}$, with a paired bootstrap CI over recordings
- **Skill retained** $= (L_{\text{ref}} - L_{\text{cond}}) / (L_{\text{ref}} - L_{\text{baseline}})$, where $L_{\text{ref}}$ is the reference loss of section 3. 1 = no change, 0 = no better than the trivial predictor

### 6.2 Frozen-backbone linear probes

Backbone in eval mode, no gradients. A linear head on mean-pooled features per time patch (`MeanPoolHead`, which is linear), on the downstream tasks: spike MC-Maze velocity ($R^2$, M4 split); iEEG Brain Treebank tasks (AUC) when available.

Two variants, because they answer different questions:

| Variant | Procedure | Answers |
|---|---|---|
| A. Refit | Train a new linear head on features computed under each condition | Is the task information still **present**? |
| B. Fixed | Train the head once on Baseline features, evaluate it on each condition's features | Did the representation **move**? |

A drop in B with no drop in A means the marker shifts the representation without destroying information.

## 7. Interpreting the results

| Result | Reading |
|---|---|
| Swap: large drop, larger than the random-direction control | The model relies on the marker. It is a sign of type-specific processing paths inside the backbone |
| Swap: large drop, similar to the random-direction control | The model is sensitive to perturbations of that size in general, not to the modality direction. Not evidence of type-specific use |
| Swap: no change, and the encoder-output probe (depth 0a) ≈ 100% | The marker is not used, but the types are already separated by the encoders. Removing the marker cannot tell us whether the backbone shares a space. Look at the depth trend |
| Swap: no change, and separation falls with depth (rank-1 accuracy, Fisher ratio, principal angles) | The backbone likely mixes the two types into a common space. The result most favorable to iBrain |
| Separation stays high or rises with depth | The backbone keeps the types apart. "Shared backbone" then means shared weights, not shared representation |
| Zero ≫ Neutral in effect | The effect of removing the marker is mostly distribution shift. Use Neutral, not Zero, as the "no marker" condition |

## 8. Implementation

No change to `model.py`. One analysis module reimplements `IBrain.encode` with an explicit marker argument and returns every layer:

```python
def encode_with(model, x, valid, sig, emb):
    """Same as IBrain.encode, but with emb in place of type_emb[sig]. Returns h and the output of every depth."""
    h = model.enc[sig](x)
    z = h + emb + model.time_emb
    outs = [h, z]
    for b in model.backbone.blocks:
        z = b(z, valid)
        outs.append(z)
    outs.append(model.backbone.norm(z))
    return outs
```

| File | Content |
|---|---|
| `ibrain/modality.py` | `encode_with`, the condition vectors of section 5, norm statistics (4.1), probe and subspace measures (4.2) |
| `scripts/modality_ablation.py` | CLI: checkpoint + held-out data → `results.json` with `meta.json` (git hash, seed, config, U26) |
| `tests/test_modality.py` | (1) `encode_with(..., emb=type_emb[sig])[-1]` equals `model.encode` exactly in eval mode. (2) Swap on a synthetic joint checkpoint runs and gives finite losses. (3) Same mask seed gives the same masks across conditions |

Compute is forward passes only, plus sklearn probes on CPU. The forward passes go through `sbatch` with 1 GPU on the lab server, never from the login shell. Probes can run in a CPU-only job.

Order of work: 4.1 → 4.2 → 5 + 6.1 → 6.2. After 4.1 and 4.2, decide whether 6.2 is worth running.

## 9. Open decisions

| # | Question | Proposal |
|---|---|---|
| D1 | Held-out split for pretraining-type data | Hold out whole recordings (iEEG) and whole sessions (spike) before pretraining, about 5% of each. Must be fixed before the pretraining run, not after |
| D2 | Tokens for the probe: masked or unmasked input | Unmasked for the probe (4.2). Masked only for the reconstruction loss (6.1) |
| D3 | Number of pretraining seeds | 3 if M7 compute allows, otherwise 1 plus bootstrap |
| D4 | Whether to also train a model with no marker at all as a reference | Out of scope here (needs pretraining). Worth doing if Swap shows a large effect |
