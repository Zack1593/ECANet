# Training Dynamics and Cost Profile of ECA-ResNet-50 on ImageNet-100

**A single-arm empirical study**

---

## Abstract

We report a complete 50-epoch training run of ECA-ResNet-50 on ImageNet-100, a
100-class subset of ILSVRC-2012, executed on a single consumer GPU (NVIDIA RTX 5060,
8 GB) in 2.31 hours of wall time. The model reaches **80.08% top-1** and **94.82%
top-5** validation accuracy. We verify by direct enumeration that the Efficient
Channel Attention (ECA) modules contribute **80 parameters** to a 23,713,012-parameter
network — an overhead of 0.00034% — confirming the cost claim of Wang et al. (2020)
exactly on our configuration.

Our principal empirical finding concerns not the attention mechanism but the optimization
schedule surrounding it. The step learning-rate decay at epoch 30 produces a **+11.24
point** top-1 improvement within two epochs, exceeding the total gain of the preceding
twenty epochs (+10.98 points). The final 18 epochs — 36% of the compute budget —
contribute a net **−0.04 points**. We further observe that the train–validation gap
evolves monotonically from −3.68 points at epoch 0 to +8.46 at epoch 49, crossing zero
near epoch 14.

**We state plainly at the outset that this study includes no baseline arm.** Without a
matched ResNet-50 trained without ECA modules, no result here can be attributed to
channel attention. What we contribute is a precisely characterized cost profile, a
reproducible training-dynamics analysis, and a concrete ablation protocol (§7) that
would license the causal claims this run cannot support.

---

## 1. Introduction

Channel attention improves convolutional network accuracy by reweighting feature-map
channels according to global context. SENet (Hu et al., 2018) established the approach
using two fully connected layers with a dimensionality-reduction bottleneck. ECA-Net
(Wang et al., 2020) argued that this reduction is harmful, and replaced the bottleneck
with a single 1-D convolution over the channel-descriptor vector — reducing the module
to a handful of parameters while reporting accuracy gains over SENet.

The published evaluation is on ImageNet-1k at full resolution, a regime requiring
multi-GPU infrastructure. This report asks a narrower, practical question: **what does
the ECA-ResNet-50 recipe actually do when transplanted to a single consumer GPU and a
100-class subset**, and where does the compute go?

Our contributions:

1. **Exact parameter accounting** for ECA in ResNet-50 by module enumeration (§4.1),
   independently confirming the authors' "80 parameters" figure.
2. **A four-phase decomposition** of the training curve isolating the dominant role of
   the LR schedule (§4.3–4.4).
3. **Measured throughput** across precision and resolution settings on 8 GB of VRAM,
   including the OOM boundary (§4.6).
4. **An explicit statement of what this run cannot show**, with a protocol that would
   (§6–7).

---

## 2. The ECA module

For an input feature map $X \in \mathbb{R}^{C \times H \times W}$, the module computes a
channel descriptor by global average pooling, applies a single 1-D convolution of kernel
size $k$ across the channel dimension, and gates the input:

$$
y = \sigma\!\left(\mathrm{Conv1D}_k\big(\mathrm{GAP}(X)\big)\right), \qquad
\tilde{X} = X \odot y
$$

The implementation (`models/eca_module.py:12-28`) uses `nn.Conv1d(1, 1, k, padding=(k-1)//2, bias=False)`.
The single input and output channel, and the absence of a bias term, are what reduce the
module's parameter count to exactly $k$. The 1-D convolution captures local cross-channel
interaction across $k$ adjacent channels without the dimensionality reduction SENet's
bottleneck imposes.

In `eca_resnet50`, one module is inserted per bottleneck block, after the third
convolution and before the residual addition (`models/eca_resnet.py:59`). Kernel sizes
are set per stage via `--ksize 3557`, which the argument parser splits character-wise
into `['3','5','5','7']`.

**Note on adaptivity.** The paper's central methodological proposal is that $k$ be
selected *adaptively* from channel count $C$ via $k = \psi(C) = \left|\frac{\log_2 C + b}{\gamma}\right|_{odd}$.
The `--ksize` pathway used here — and in every command in the repository README — instead
sets $k$ manually per stage. This run therefore evaluates the ECA *module*, not the
adaptive kernel-selection rule. `models/eca_ns.py` contains a separate implementation;
it was not exercised.

---

## 3. Experimental setup

### 3.1 Data

ImageNet-100 (`clane9/imagenet-100`), a 100-class subset of ILSVRC-2012 following the
split of Tian et al. (2020): 126,689 training and 5,000 validation images, distributed
at 160 px on the shorter side. Images were exported from the source parquet shards to
the `ImageFolder` layout the training script expects.

Augmentation is the stock ImageNet recipe: `RandomResizedCrop` + horizontal flip for
training, `Resize`/`CenterCrop` for validation, with ImageNet channel statistics.

### 3.2 Model and optimization

| Setting | Value |
|---|---|
| Architecture | `eca_resnet50` (ECABottleneck ×16, layers [3,4,6,3]) |
| ECA kernel sizes | 3 / 5 / 5 / 7 (stages 1–4) |
| Classes | 100 |
| Crop size | 160 px |
| Batch size | 192 (660 iterations/epoch) |
| Optimizer | SGD, momentum 0.9, weight decay 1e-4 |
| Learning rate | 0.075, ×0.1 every 30 epochs |
| Epochs | 50 |
| Precision | Mixed (fp16 autocast + loss scaling) |
| Loss | Cross-entropy |

The learning rate follows the linear scaling rule from the reference 0.1 @ batch 256.
No warmup, no label smoothing, no EMA.

### 3.3 Hardware

NVIDIA RTX 5060 (8 GB, Blackwell), Intel Core i5-13500 (20 threads), PyTorch 2.13.0+cu130.
Of the 8,151 MiB of device memory, approximately 1,356 MiB is held by the desktop
compositor, leaving ~6.3 GiB available to training — a constraint that materially shapes
the feasible configuration space (§4.6).

### 3.4 Modifications to the reference implementation

The repository targets PyTorch 1.0/1.1. Four changes were required:

1. **`--num-classes`** threaded into model construction; the classifier head was
   otherwise hardcoded to 1000 outputs regardless of dataset.
2. **`--image-size`** parameterizing the train crop and validation resize/crop, which
   were fixed at 224/256.
3. **`--amp`** adding autocast and gradient scaling to the train and validation loops.
4. **A correctness fix in `accuracy()`**: `correct[:k].view(-1)` raises on modern
   PyTorch because `pred.t()` followed by `expand_as` yields a non-contiguous tensor;
   changed to `.reshape(-1)`. This defect is inherited from the reference torchvision
   ImageNet example and is unrelated to ECA.

Items 1 and 4 are correctness fixes. Items 2 and 3 change the training regime and are
therefore part of what is being characterized, not neutral infrastructure.

---

## 4. Results

### 4.1 Parameter cost

Enumerating every `eca_layer` instance in the constructed model:

**Table 1 — ECA parameter accounting, `eca_resnet50`, 100 classes**

| Stage | Blocks | $k$ | ECA params |
|---|---:|---:|---:|
| layer1 | 3 | 3 | 9 |
| layer2 | 4 | 5 | 20 |
| layer3 | 6 | 5 | 30 |
| layer4 | 3 | 7 | 21 |
| **Total** | **16** | — | **80** |

| Quantity | Value |
|---|---:|
| Total model parameters | 23,713,012 |
| ECA parameters | 80 |
| Backbone excluding ECA | 23,712,932 |
| **ECA share of total** | **0.00034%** |

This reproduces the "80 vs. 24.37M" claim of Wang et al. exactly. (Our total is lower
than the published 24.37M solely because the 100-class head is 900 × 2048 smaller than
the 1000-class one.)

The computational cost is likewise negligible: each module performs a global average
pool followed by $C \times k$ multiply-accumulates. Summed over all 16 modules this is
under $10^5$ MACs per forward pass, against the network's ~2 GMACs at 160 px — a
relative cost below $10^{-4}$. **This is the paper's strongest and most robust claim,
and our measurement supports it without qualification.** It is also the one claim that
requires no baseline to verify.

### 4.2 Accuracy

**Table 2 — Final and best validation accuracy**

| Metric | Value | Epoch |
|---|---:|---:|
| Best top-1 | **80.08%** | 38 |
| Best top-5 | **94.82%** | 42 |
| Final top-1 (epoch 49) | 79.00% | 49 |
| Final top-5 (epoch 49) | 94.22% | 49 |
| Final train top-1 | 87.46% | 49 |
| Total wall time | 2.31 h | — |

Top-1 peaks at epoch 38 and drifts down 1.08 points by epoch 49. Checkpoint selection on
validation top-1 therefore matters: reporting the final epoch understates the run's best
achievable result. The run reaches within one point of its eventual best at **epoch 34**.

These numbers are not comparable to the published ImageNet-1k figures (77.42/93.62 top-1/top-5)
— a 100-class problem at 160 px is a fundamentally easier task than 1000 classes at 224 px,
and the higher accuracy here reflects the task, not a better model.

### 4.3 Four-phase training dynamics

**Table 3 — Phase decomposition of the training curve**

| Phase | Epochs | Val top-1 | Δ | Train−val gap | Train loss |
|---|---|---|---:|---|---|
| A — Rapid fitting | 0 → 9 | 11.38 → 57.48 | **+46.10** | −3.68 → −0.46 | 4.131 → 1.635 |
| B — Plateau | 10 → 29 | 56.58 → 67.56 | +10.98 | +2.48 → +5.38 | 1.550 → 0.999 |
| C — LR step | 29 → 31 | 67.56 → 78.80 | **+11.24** | +5.38 → +2.98 | 0.999 → 0.672 |
| D — Post-step | 32 → 49 | 79.04 → 79.00 | **−0.04** | +3.44 → +8.46 | 0.642 → 0.464 |

Phase A extracts nearly three-fifths of the final accuracy in the first ten epochs.
Phase B is a long, noisy climb at high learning rate (validation top-1 standard deviation
over epochs 20–29: **1.23 points**). Phase C is the decisive event. Phase D is
essentially free of accuracy gain while the network continues to fit the training set.

### 4.4 The schedule dominates the observable dynamics

The single most consequential number in this study is that **two epochs of decayed
learning rate (phase C, +11.24 points) outperform twenty epochs at the original rate
(phase B, +10.98 points)**.

Two consequences follow.

**The last 18 epochs were wasted compute.** Phase D delivers a net −0.04 points across
36% of the run — roughly 50 minutes of the 2.31-hour budget. The step schedule
(`main.py:392`, ×0.1 every 30 epochs) is tuned for the paper's 100-epoch ImageNet-1k
protocol, which places decays at 30/60/90. Truncating that schedule to 50 epochs yields
exactly one decay followed by a long flat tail, with the second decay (epoch 60) never
arriving. A cosine schedule over `args.epochs`, or step boundaries scaled to the actual
budget, would likely reach comparable accuracy in appreciably less time.

**Validation noise collapses after the decay.** Standard deviation of validation top-1
falls from 1.23 points (epochs 20–29) to **0.18 points** (epochs 40–49), a 7.0× reduction.
This has a direct methodological implication for the ablation in §7: at a high learning
rate, run-to-run differences of one to two points are indistinguishable from step noise.
Any ECA-vs-baseline comparison must be read *after* the decay, and differences smaller
than roughly 0.4 points (≈2σ in the stable regime) should not be interpreted from single
runs.

### 4.5 The generalization gap inverts, then reopens

Mean train−validation top-1 gap by decade of the run:

| Epochs | 0–9 | 10–19 | 20–29 | 30–39 | 40–49 |
|---|---:|---:|---:|---:|---:|
| Mean gap (pts) | **−2.61** | +1.06 | +3.35 | +4.21 | **+7.13** |

The gap is **negative through most of the first ten epochs** — validation accuracy exceeds
training accuracy — with 8 of the first 10 epochs below zero and the last negative value at
epoch 14. This is not an anomaly, and two mechanisms produce it. First, `RandomResizedCrop`
with the default scale range (0.08–1.0) presents aggressively cropped, sometimes
near-uninformative views during training, while validation uses a gentle center crop; the
training task is genuinely harder than the validation task. Second, the reported training
figure is a *running average across the epoch*, including early low-accuracy iterations,
whereas validation is measured once with the epoch's final weights.

The trajectory is **not monotone**, and two features deserve note:

- Through epochs 10–29 the gap oscillates in a wide band (−0.59 at epoch 11 to +5.77 at
  epoch 25) at high learning rate, mirroring the validation noise quantified in §4.4.
- **The LR decay transiently closes the gap**, from +5.38 at epoch 29 to +1.45 at epoch 30
  — the decayed step improves validation faster than it improves the training average.

From epoch 31 onward the gap widens steadily and nearly monotonically, +2.98 → +8.46.
Combined with the top-1 peak at epoch 38 and subsequent 1.08-point decline, this is mild
overfitting across the final third of the run, consistent with phase D's absence of
validation gain. Top-5 shows the same pattern attenuated (final gap 2.12 points), as
expected of a metric already above 94%.

### 4.6 Throughput and the 8 GB constraint

Measured on synthetic tensors (six steps, first three discarded as warmup), `eca_resnet50`
at 100 classes:

**Table 4 — Peak memory and throughput, RTX 5060 (~6.3 GiB usable)**

| Precision | Crop | Batch | Peak VRAM | Throughput |
|---|---:|---:|---:|---:|
| fp32 | 224 | 32 | 3.59 GiB | 221 img/s |
| fp32 | 224 | 64 | — | **OOM** |
| fp32 | 160 | 96 | 5.32 GiB | 417 img/s |
| amp | 224 | 96 | 5.36 GiB | 398 img/s |
| amp | 160 | 128 | 3.79 GiB | 763 img/s |
| **amp** | **160** | **192** | **5.47 GiB** | **776 img/s** |
| amp | 160 | 256 | — | OOM |

The configuration used for this study (amp, 160 px, batch 192) is **3.5× faster** than
the largest feasible fp32 224 px configuration. Mixed precision and reduced resolution
contribute roughly equally and compose almost multiplicatively.

Two constraints deserve emphasis. First, the reference batch size of 256 is unreachable
on this hardware in any configuration tested, so the LR schedule is necessarily a scaled
approximation of the published recipe. Second, resolution reduction is *free* here in a
way it would not be on ImageNet-1k: the source images are distributed at 160 px on the
short side, so training at 160 px removes an upsampling step rather than discarding
information.

For completeness, CPU-only training was benchmarked at 24.6 img/s (ResNet-50, 160 px) on
the i5-13500 — **31× slower** than the GPU configuration, placing a comparable run at
roughly six days. CPU training is viable only for substantially smaller datasets.

---

## 5. Threats to validity

**No baseline arm — the binding limitation.** This study trained one model. No
ResNet-50 without ECA modules was trained under matched conditions. Consequently:

- The 80.08% top-1 figure **cannot be attributed to channel attention**. It is the
  accuracy of this architecture under this recipe on this data, nothing more.
- No statement in this report should be read as evidence for or against ECA's efficacy.
  The parameter-cost result (§4.1) is the sole exception, being a structural property of
  the model that requires no comparison.

**Single seed.** One run per configuration. Given the measured 1.23-point validation
standard deviation at high LR, single-run differences are weakly informative.

**Modified recipe.** Reduced resolution (160 vs. 224), reduced batch (192 vs. 256),
halved epochs (50 vs. 100), and mixed precision jointly distinguish this run from the
published protocol. Interactions between these changes are not isolated.

**Manual kernel sizes.** As noted in §2, `--ksize 3557` bypasses the adaptive
kernel-selection function that constitutes the paper's methodological contribution.

**Checkpoint provenance.** The metric files under `runs/eca_resnet50_in100/` disagree
with `ECANet_output.txt` (epoch-0 validation top-1 of 14.86 vs. 11.38, and a collapse to
12.06 at epoch 5 absent from the log). They originate from a different run. All analysis
here derives from the log; the saved `model_best.pth.tar` should **not** be assumed to
correspond to the curves reported above.

**Synthetic throughput.** Table 4 excludes JPEG decode. End-to-end epoch time from the
log (166.2 s mean, 660 iterations at batch 192 ⇒ ~763 img/s) closely matches the
synthetic figure, indicating the input pipeline kept pace at 8 workers — but this
agreement is a property of this machine, not a general result.

---

## 6. What would license a causal claim

The analysis above characterizes a recipe. To evaluate ECA itself, the minimum credible
experiment is:

1. **Matched pair.** Train `resnet50` and `eca_resnet50` under byte-identical settings —
   same seed, schedule, augmentation, precision, resolution, batch size. The repository
   already exposes torchvision's `resnet50` through `models.__dict__`, so no new code is
   needed for the baseline arm.
2. **Three seeds per arm.** Report mean ± standard deviation. Given §4.4, treat
   differences below ~0.4 points as unresolved.
3. **Post-decay measurement.** Compare in the stable regime (epochs 40–49), not at peak
   and not at high LR.
4. **Report top-1.** Top-5 is at 94.8% and near saturation; it has little power to
   separate the arms.
5. **Budget.** At 776 img/s, six runs of 50 epochs is approximately 14 GPU-hours — an
   overnight job on the hardware described here.

A secondary experiment of independent interest: compare `--ksize 3557` against the
adaptive $\psi(C)$ rule, which would test the paper's actual kernel-selection proposal
rather than a hand-tuned instantiation of it.

---

## 7. Conclusion

ECA-ResNet-50 trains to 80.08% top-1 on ImageNet-100 in 2.31 hours on a single 8 GB
consumer GPU. The attention modules cost 80 parameters — 0.00034% of the network —
verified by direct enumeration, confirming the efficiency claim that motivates the method.

The dominant factor in the observable training dynamics is not the attention mechanism
but the learning-rate schedule: a single decay at epoch 30 delivers more accuracy in two
epochs than the twenty preceding epochs combined, after which the run gains nothing across
its final 36% of compute. For practitioners adapting this repository to shorter budgets,
matching the schedule to the epoch count is the highest-value change available, and is
independent of any question about attention.

Whether ECA contributes to the 80.08% figure remains, on this evidence, **unknown**. The
protocol in §6 would settle it for roughly 14 GPU-hours.

---

## References

1. Wang, Q., Wu, B., Zhu, P., Li, P., Zuo, W., Hu, Q. **ECA-Net: Efficient Channel
   Attention for Deep Convolutional Neural Networks.** CVPR 2020. arXiv:1910.03151
2. Hu, J., Shen, L., Sun, G. **Squeeze-and-Excitation Networks.** CVPR 2018.
3. He, K., Zhang, X., Ren, S., Sun, J. **Deep Residual Learning for Image Recognition.**
   CVPR 2016.
4. Tian, Y., Krishnan, D., Isola, P. **Contrastive Multiview Coding.** ECCV 2020.
   (Source of the ImageNet-100 class split.)
5. Goyal, P. et al. **Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour.**
   arXiv:1706.02677. (Linear LR scaling rule.)

---

## Appendix A — Reproduction

```bash
# Data
python export_in100.py -o ~/datasets/imagenet-100

# Training run analyzed in this report
python main.py -a eca_resnet50 --ksize 3557 --num-classes 100 \
  --amp --image-size 160 -b 192 --lr 0.075 -j 8 \
  --epochs 50 --action in100 ~/datasets/imagenet-100
```

Per-epoch metrics: `ECANet_output.txt`. Interactive curves: `accuracy_chart.html`.

## Appendix B — Selected per-epoch values

| Epoch | Train top-1 | Val top-1 | Train top-5 | Val top-5 | Train loss |
|---:|---:|---:|---:|---:|---:|
| 0 | 7.70 | 11.38 | 23.27 | 32.68 | 4.1313 |
| 5 | 45.74 | 45.22 | 73.29 | 74.00 | 2.1314 |
| 10 | 59.06 | 56.58 | 83.33 | 83.98 | 1.5503 |
| 20 | 68.95 | 67.88 | 88.99 | 89.74 | 1.1475 |
| 29 | 72.94 | 67.56 | 91.00 | 88.36 | 0.9992 |
| 30 | 79.43 | 77.98 | 93.64 | 94.02 | 0.7568 |
| 31 | 81.78 | 78.80 | 94.46 | 94.38 | 0.6717 |
| 38 | 85.12 | **80.08** | 95.64 | 94.70 | 0.5486 |
| 42 | 86.14 | 79.48 | 96.00 | **94.82** | 0.5093 |
| 49 | 87.46 | 79.00 | 96.34 | 94.22 | 0.4635 |

Full 50-row table in `accuracy_chart.html`.
