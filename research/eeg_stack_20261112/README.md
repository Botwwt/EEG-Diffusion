# Stacking on the ordinary head (2026-11-12, updated 2026-11-19)

This folder adds components on top of the **ordinary bounded condition head**
(`research/eeg/brainhive_bounded_component_v1.py`, imported unchanged) and reports every arm that was run,
including the ones that did not help. Host model: Brain-HIVE (commit f06db69) with SDXL-Turbo.

## Original-components pipeline (headline since 2026-11-13 evening)

A prior-art audit (full text of Brain-HIVE, MindEye, MindEye2, Brain-Diffuser, Perceptogram, ATM, ENIGMA, AVDE, JMVR,
CognitionCapturerPro, UMind, UBP, NeuroBridge and the saliency-guided paper; table under "Where each component comes
from") showed that several components of the full stack below are taken from same-field papers: the joint encoder
(MindEye2 / ENIGMA), best-of-N selection (MindEye), output averaging with a blurred image (MindEye2) and the
test-statistics renormalisation (Perceptogram). The headline is therefore a pipeline that **leaves those out** and keeps,
on top of the host and the co-author's head, only components that are ours or transferred from outside brain decoding:

| | component | origin |
|---|---|---|
| host | Brain-HIVE per-subject encoder + SDXL-Turbo / IP adapter | Brain-HIVE (baseline, not ours) |
| head | ordinary bounded condition head, one per subject | co-author (`research/eeg/brainhive_bounded_component_v1.py`) |
| O0 | EEG → SDXL-VAE latent start image by ridge, **with a repeat-noise correction of the covariance** | scaffold from Brain-Diffuser / Perceptogram (credited); the repeat correction is ours |
| O1 | **start latent blended with the latents of retrieved training images** (`lowsem.py`) | ours (not found in the audited papers) |
| O2 | **inverse-variance aggregation of the 80 test repeats** (`robagg.py`, `ivwprep.py`) | ours; transferred from robust statistics (inverse-variance weighting) |
| O3 | **fractional img2img strength by re-injecting the noised start latent** (`gen.py --smap c0.611,0.5,1.0`) | mechanism of Differential Diffusion (image editing, credited), used here with a constant map |
| O4 | **split-half contrastive test-time adaptation of the encoder input layer, used for the start image** (`tta.py`) | transferred from test-time adaptation (domain adaptation; Tent, TTT) with an InfoNCE objective (CPC / SimCLR); the split-half positives are ours |
| O5 | **IP condition = deep ensemble of 7 independently seeded per-subject encoders, each applied to 10 inverse-variance group means of 8 repeats and averaged** (`condscreen.py ens7bag10`) | deep ensembles (Lakshminarayanan et al. 2017) and bagging (Breiman 1996), generic machine learning; not checked for prior use in brain decoding beyond the audited papers |
| O6 | **the O5 ensemble extended over three input windows: encoders trained on samples 0–250, 0–150 and 0–100 (7 seeds each), 21 encoders × 10 repeat groups averaged** (`condscreen.py mw7bag10`) | multi-scale analysis of time series with an ensemble over the scales (generic); time-window analyses of EEG decoders are common in the field, so the shorter windows themselves are not claimed as new |
| **O7a** | **6-window 42-encoder split-half TTA + repeat bagging + pure `F_train` cone restoration (`w = 1.00`, zero directional amplification; `aw_tta.py`), native 512 px 5-window QR-PCA start latent with coarse 16 × 16-layout-disambiguated kNN (`lat512_start.py`), and step-2/3 `8 × 8` low-band tether (`gen_exp.py`, `--tmf` off, strictly independent K = 4 draws)** | training-cone marginal restoration (domain adaptation / cone geometry), frequency-weighted QR-PCA across post-stimulus EEG windows, selective step-2/3 low-band tether (refined from ILVR, Choi et al. 2021, credited); **strict single-subject entry: no cross-subject data, no condition amplification, no K = 4 sample coupling, no best-of-N, no post-hoc blur** |
| O7 | **O7a + directional cone amplification (`w = 1.50`) + step-1/2 K = 4 Tweedie posterior coupling with spatial contrast restoration (`--tmf 1,0.10,0.10,0 --tmf_cr 1`)** | adds Perceptogram-style directional extrapolation (`w > 1`, credited) and intra-query K = 4 Tweedie coupling (Efron 2011; Chung et al. 2023, credited); reported alongside O7a and full ablations O7b / O7c to isolate both mechanisms |
| **O8** | **O7a + multi-scale temporal-pyramid single-subject `FlexBrainEncoder` (`train_flex.py`) + 5-fold OOF simplex ensemble (`oof_blend_select.py`, `alpha_o8 = 0.50`) + per-subject `ord_o8_flex` head on the frozen `v2` start latent and sampler (`nsel = 1`, `w = 1.00`, strictly independent `K = 4` draws)** | generic multi-scale temporal-pyramid EEG stem + gated residual trunk trained with cosine & soft-teacher-similarity KL distillation (`train_flex.py`), combined strictly on 5-fold concept-disjoint training OOF (`oof_blend_select.py`); zero cross-subject data, zero amplification, zero test-set tuning |
| O8b | **O8 + step-1/2 K = 4 Tweedie posterior coupling (`--tmf 1,0.10,0.10,0 --tmf_cr 1`, `w = 1.00` kept)** | intra-query `K = 4` Tweedie coupling (same mechanism as O7, with strict single-subject `w = 1.00`) |
| **O9-0tta / O9** | **Streamlined `8`-checkpoint single-subject pipeline (`7 × KVEBPyrEncoder` + `1 × BoundedConditionAdapter`, `o9_streamlined.py`): within-fold Empirical-Bayes spatiotemporal reliability gate & repeat-variance scaler (`TargetSNRGate`), nested multi-horizon stems (`0–1000 / 0–700 / 0–480 ms`), closed-form 70-layer SDXL-Turbo UNet cross-attention Key-Value subspace loss (`CenteredKVLoss`, $L_{\mathrm{KV}}$ `eff_rank = 237.1 / 2048`), plus `0` TTA steps (`O9-0tta`) or `350` centered $L_{\mathrm{KV}}$ split-half TTA steps + `bag10` (`r_eff = 8.0`) on `v2` start (`O9`)** | ours (`kv_metric.py`, `deep_enc.py`, `o9_streamlined.py`): Empirical-Bayes SNR weighting + repeat-variance matching from training $R=4$ to test $R_{\mathrm{eff}} \in \{8, 36, 72\}$, and exact operator-pullback alignment in the diffusion UNet's own Key-Value subspace (`91%` fewer ckpts & TTA steps vs O8; `0` aux heads) |
| O9b | **O9 + step-1/2 K = 4 Tweedie posterior coupling (`--tmf 1,0.10,0.10,0 --tmf_cr 1`, `w = 1.00` kept)** | intra-query `K = 4` Tweedie coupling (exact counterpart to O8b, with `8` ckpts/subject) |
| **O10-0tta / O10** | **Bifurcated forward-operator synthesis (`o10_factorized.py`): drops both `(0.35, 0.25)` positive-pair alignment terms (`w_kv = w_u = 0`) and removes `BoundedConditionAdapter` completely (`0` DDPM heads); combines Stream 1 (`7 × KVEBPyrEncoder` on `F_train`) and Stream 2 (`7 × KVEBPyrEncoder` on VAE-marginalized `F_sem` with same-concept FN-masking, visual-confuser margin `beta_conf = 0.15`, and frozen `FusionEncoder` `PhysReSyn` combining semantic `(e_H14, e_B32)` expectations + ridge `z_vae`) via `eta = 0.50` orthogonal tangent bifurcation onto the pure `F_train[ok]` cone (`w = 1.00`, indep `K = 4`, `nsel = 1`; plus `O10-sel2/4` budget curve)** | ours (`o10_factorized.py`, `build_o10_report_and_gallery.py`): $2\times 2$ InfoNCE term reduction + forward-operator marginalization (`F_sem`), `FusionEncoder` physical re-synthesis (`PhysReSyn`), and two-stream tangent bifurcation solving the semantic/SSIM Pareto trade-off with `0` DDPM heads |

O2 in detail: every test image has 80 repeats; trial r gets weight 1 / (mean squared residual of r against the mean of
the other 79), so noisy trials (blinks, movement) count less. Label-free, one line of arithmetic, no training; the
effective number of trials is 69–77 of 80. The same weighted mean feeds the encoder and the ridge.
O3 in detail: SDXL-Turbo has 4 steps, so img2img strength can only be 0.25, 0.5, 0.75 or 1. Here the sampler starts from
noise (strength 1) and, after each step, the latent is replaced by the start latent noised to the current level with
weight clamp(h − i, 0, 1), h = 4 − 4·s, s = 1 − 0.5 · 0.611 = 0.69: the start image is kept for the first 1.2 steps,
i.e. a strength of 0.69 that the integer schedule cannot express.
O4 in detail: at test time, without labels or images, the 80 repeats of each of the 200 test images are split at random
into two halves of 40. The two inverse-variance half-means of the same image should map to the same embedding, so they
are positives and the other 199 images are negatives (symmetric InfoNCE, τ 0.1). Only the subject's input layer is
updated (50 Adam steps, lr 1e-4), with an anchor (λ 1, relative L2) to the unadapted output. All four settings were
fixed before the image runs (the first one is used; the three others are reported). Raw 200-way retrieval rises from
0.455 to 0.517 top-1 (10/10 subjects, `results/test_time_adaptation.txt`). The adapted embedding is used only for the
retrieval blend of the start image; the IP-adapter condition stays the O3 one (using the adapted condition there too
raises PixCorr / SSIM further but costs CLIP −0.003, 3/10 subjects, see below). Transductive: it uses the 200 test EEG
recordings together, but no test image and no label.
O5 in detail: six more per-subject encoders are trained with the official recipe and seeds 2028–2033 (3 s each). For
the IP condition, the 80 repeats are split into 10 disjoint groups of 8; each group's inverse-variance mean goes through
each of the 7 encoders, the 70 outputs are averaged and rescaled per image to the norm of the O4 condition, and the
subject's head is applied. The averaging removes encoder-seed variance and gives inputs closer to the repeat count the
encoder was trained on. The start image stays the O4 one. Screen without images (`results/semantic_condition.txt`):
top-1 0.4515 → 0.4750 (9/10 subjects), cosine to the target 0.3199 → 0.3291 (10/10).
O6 in detail: the same official recipe, trained on the first 150 and the first 100 of the 250 samples (`train.py --t1 150`,
`--t1 100`, i.e. 0–600 ms and 0–400 ms at 250 Hz), seeds 2025 and 2028–2033, 14 extra encoders per subject (3 s each). The O5 condition is computed for all 21
encoders (each on its own window of the 10 group means), the 210 outputs are averaged, rescaled to the base norm and
passed through the subject's head (trained on the full-window encoder only). The start image stays the O4 one. The shorter
windows decode better on their own (top-1 after the head 0.4985 for 7 seeds of 0–150 vs 0.4620 for 7 seeds of 0–250) and,
unlike test-time adaptation, they also raise the cosine to the target (0.3334 vs 0.3286, 10/10).
O7a / O7 in detail: three bottlenecks limited O6:
1. **Training-cone restoration (`w = 1.00` in O7a; `w = 1.50` in O7).** Symmetric InfoNCE and split-half TTA centre encoder outputs near zero,
   removing the global cone axis $u_{\mathrm{tr}}$ of $F_{\mathrm{train}}$ ($\alpha_{\mathrm{tr}} = \|\mathbb{E}[\hat F_{\mathrm{train}}]\| = 0.6164$,
   $\beta_{\mathrm{tr}} = \sqrt{1 - \alpha_{\mathrm{tr}}^2} = 0.7874$, $R_{\mathrm{tr}} = 32.01$) so the `0.25 rad` cap of
   `BoundedConditionAdapter` saturates and adapted conditions previously lost CLIP (−0.003 in O4). Applying
   $c_{\mathrm{cone}}(w) = R_{\mathrm{tr}}\,\mathrm{normalize}(\alpha_{\mathrm{tr}} u_{\mathrm{tr}} + w \beta_{\mathrm{tr}} d_\perp)$
   after the multi-window head `ord_mw7` (`aw_tta.py`) unlocks split-half TTA + 10-group IVW bagging across all 42 single-subject
   encoders (samples 0–250, 0–200, 0–175, 0–150, 0–125, 0–100 = 0–1000, 0–800, 0–700, 0–600, 0–500, 0–400 ms × 7 seeds; 10-subject
   centred 200-way diagnostic top-1 = 0.5840, `results/o7_b4_stack.txt`). At **`w = 1.00` (O7a)** this restores the exact training
   cone mean and norm without amplifying deviations from the mean direction; **`w > 1` (O7 uses `w = 1.50`)** extrapolates deviations
   from the mean axis, which belongs to the same Perceptogram-style directional amplification class as `--amp`. (In an intermediate
   4-subject screen (`{2, 5, 8, 9}`, O6 start image, before the 512 px start), `w = 1.00` gave `Incep +0.0074 / CLIP +0.0034` and `w = 1.40` gave
   `Incep +0.0101 / CLIP +0.0048 / SSIM −0.0238` over O6; the step from `1.00` to `1.40` itself was only `Incep +0.0027 / CLIP
   +0.0014 / SSIM −0.0231`, and those two runs also used different TTA condition sets (`cone_mwtta` vs `cone_mw7tta`), so they are not a clean
   `w` comparison. On the complete 10-subject stack (`O7c` vs `O7a`), `w: 1.00 → 1.50` changes Incep/CLIP by `−0.0011 / −0.0009`
   (paired t ≈ −0.5 / −0.7, 5/10 subjects each, i.e. no detectable difference), lowers SSIM by `0.0011` in 10/10 subjects and
   raises Alex2/Alex5 by `+0.0012 / +0.0014` (8/10). **O7a (`w = 1.00`)** is therefore the strictly unborrowed arm at no
   measurable semantic cost; it is not semantically stronger than O7.)
2. **Native 512 px 5-window QR-PCA start latent (`lat512_start.py`).** Following the ATM preprocessing (`Preprocessed_data_250Hz_whiten`:
   epoched −200 to 1000 ms at 250 Hz = 4 ms/sample, with the 50 pre-stimulus samples dropped), samples `0..250` span `0–1000 ms`
   post-stimulus under THINGS-EEG2's `200 ms` RSVP SOA (`100 ms` presentation + `100 ms` blank, next onset at sample 50). Instead of
   `low.py`'s 128 px bicubic round-trip, `lat512_start.py` predicts native `4 × 64 × 64` 512 px latents (`lat512_train.npy`),
   up-weighting coarse `16 × 16` and `32 × 32` spatial modes via a 768-mode frequency-weighted orthonormal QR-PCA basis fitted across
   five post-stimulus scalp windows (`12–62 = 48–248 ms`, `10–75 = 40–300 ms`, `15–90 = 60–360 ms`, `10–125 = 40–500 ms`,
   `0–250 = 0–1000 ms`), fused per mode strictly by 5-fold concept-disjoint out-of-fold $r^2$, and disambiguating the top-30 semantic
   kNN training latents with the single-subject `16 × 16` ridge layout (`0.82 sem + 0.18 eeg16`).
3. **Step-2/3 coarse tether (`O7a`) and optional K = 4 Tweedie coupling (`O7`).** In `gen_exp.py` (`--smap c0.625,0.5,1.0`),
   `--low_keep 8,0.22,0.04` gently tethers only the `8 × 8` coarse band at steps $i = 2, 3$ after the fractional hold reaches zero
   (avoiding full-step ILVR's CLIP loss). **O7a** stops there (`K = 4` strictly independent draws per query, no `--tmf`). **O7**
   additionally enables `--tmf 1,0.10,0.10,0 --tmf_cr 1`, which pulls the `K = 4` trajectories of each query slightly (`g = 0.10`)
   toward their shared 4-sample Tweedie $\hat z_0$ mean on steps 1–2 while restoring each channel's `64 × 64` spatial SD; per-image
   UNet NFE (4 steps) is unchanged, but because `metrics.py` averages scores across the `K = 4` images of a query, `--tmf` couples
   those 4 samples (an intra-query 4-sample ensemble, not independent draws). Across all 10 subjects (`O7b` vs `O7a`), `--tmf`
   contributes `PixCorr +0.0040` and `SSIM +0.0014` (`Incep −0.0005`, `CLIP −0.0007`).

### Results (10 subjects, same metric code and seeds as everywhere in this folder)

| arm | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | EffNet ↓ |
|---|---|---|---|---|---|---|---|
| Brain-HIVE baseline, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 | 0.8786 |
| + ordinary head (co-author) | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 | 0.8712 |
| + O0 repeat-corrected ridge start image | 0.2455 | 0.3895 | 0.8649 | 0.9163 | 0.7558 | 0.8085 | 0.8651 |
| + O1 retrieval-blended start | 0.2511 | 0.3811 | 0.8712 | 0.9155 | 0.7587 | 0.8118 | 0.8651 |
| + O2 inverse-variance repeat aggregation | 0.2525 | 0.3814 | 0.8717 | 0.9184 | 0.7603 | 0.8156 | 0.8640 |
| + O3 fractional strength | 0.2636 | 0.4073 | 0.8727 | 0.9152 | 0.7581 | 0.8118 | 0.8663 |
| + O4 test-time-adapted start image | 0.2687 | 0.4090 | 0.8753 | 0.9158 | 0.7573 | 0.8121 | 0.8659 |
| + O5 seed ensemble with repeat bagging for the condition | 0.2692 | 0.4108 | 0.8780 | 0.9176 | 0.7584 | 0.8139 | 0.8642 |
| + O6 multi-window ensemble for the condition | 0.2711 | 0.4116 | 0.8796 | 0.9188 | 0.7616 | 0.8142 | 0.8634 |
| **+ O7a strict original (`w = 1.00`, independent K = 4 draws, headline)** | **0.3036** | **0.4256** | **0.8953** | **0.9257** | **0.7827** | **0.8224** | **0.8495** |
| + O7b (`w = 1.00` + K = 4 Tweedie coupling) | 0.3076 | 0.4270 | 0.8959 | 0.9256 | 0.7821 | 0.8217 | 0.8492 |
| + O7c (`w = 1.50` cone amplification, independent K = 4 draws) | 0.3035 | 0.4246 | 0.8965 | 0.9271 | 0.7815 | 0.8215 | 0.8513 |
| + O7 (`w = 1.50` cone amplification + K = 4 Tweedie coupling) | 0.3074 | 0.4259 | 0.8971 | 0.9271 | 0.7812 | 0.8219 | 0.8509 |
| **+ O8 strict original (`FlexBrainEncoder` + OOF blend, `w = 1.00`, indep K = 4, headline)** | 0.3025 | 0.4252 | **0.8970** | **0.9281** | **0.7891** | **0.8299** | **0.8434** |
| **+ O8b (`O8 + K = 4 Tweedie coupling`, `w = 1.00`, no cone amp)** | **0.3064** | **0.4266** | **0.8977** | **0.9280** | **0.7886** | **0.8300** | **0.8433** |
| **+ O9-0tta (`8` ckpts: `KVEBPyrEncoder` + $L_{\mathrm{KV}}$, `0` TTA, `w = 1.00`, indep K = 4)** | 0.2981 | 0.4256 | 0.8917 | 0.9268 | **0.7901** | **0.8321** | 0.8447 |
| **+ O9 (`8` ckpts, `350` $L_{\mathrm{KV}}$ TTA + `bag10`, `w = 1.00`, indep K = 4, streamlined headline)** | 0.2997 | **0.4276** | 0.8941 | **0.9284** | **0.7969** | 0.8293 | **0.8422** |
| **+ O9b (`O9 + K = 4 Tweedie coupling`, `8` ckpts, `w = 1.00`, no cone amp)** | 0.3035 | **0.4290** | 0.8948 | **0.9287** | **0.7973** | 0.8298 | **0.8423** |
| + O9-zero00-0head (`η = 0`, `7` ckpts, `(0, 0)` weights, `0` DDPM heads, `w = 1.00`) | 0.2998 | 0.4262 | 0.8948 | **0.9295** | 0.7952 | 0.8284 | 0.8422 |
| + O10-sem-0tta (`η = 1`, `F_sem` + PhysReSyn, `0` TTA, `0` DDPM heads) | 0.2867 | 0.4165 | 0.8794 | 0.9249 | 0.7869 | 0.8309 | 0.8415 |
| + O10-sem-noPhysReSyn (`η = 1`, `F_sem` + `kvTTA`, no PhysReSyn, `0` DDPM heads) | 0.2856 | 0.4201 | 0.8792 | 0.9221 | 0.7936 | 0.8332 | 0.8444 |
| + O10-sem-only (`η = 1`, `F_sem` + `kvTTA` + PhysReSyn, `0` DDPM heads) | 0.2866 | 0.4173 | 0.8804 | 0.9231 | 0.7870 | 0.8300 | **0.8396** |
| **+ O10-0tta (`η = 0.50` bifurcated, `0` TTA, `0` DDPM heads, `w = 1.00`, `nsel = 1`)** | 0.2933 | 0.4203 | 0.8883 | 0.9283 | 0.7920 | **0.8346** | **0.8393** |
| **+ O10 (`η = 0.50` bifurcated, `0` DDPM heads, `w = 1.00`, `nsel = 1`, headline)** | 0.2944 | 0.4217 | 0.8900 | **0.9290** | 0.7958 | **0.8347** | **0.8381** |
| **+ O10-sel2 (`η = 0.50`, `nsel = 2`, `8` UNet NFE/img)** | 0.2963 | 0.4225 | 0.8900 | 0.9287 | **0.7985** | **0.8371** | **0.8367** |
| **+ O10-sel4 (`η = 0.50`, `nsel = 4`, `16` UNet NFE/img)** | 0.2990 | 0.4235 | 0.8903 | **0.9294** | **0.8009** | **0.8392** | **0.8358** |
| O8#w5start (`R = 4`-gated `w5` start ablation, not adopted) | 0.2956 | 0.4241 | 0.8880 | 0.9229 | 0.7859 | 0.8258 | 0.8477 |
| O3 + adapted start *and* adapted condition (not adopted) | 0.2706 | 0.4142 | 0.8758 | 0.9153 | 0.7590 | 0.8087 | 0.8640 |

#### Screened 4 subjects `{2, 5, 8, 9}` vs untouched held-out 6 subjects `{1, 3, 4, 6, 7, 10}` (paired gain over O6)

| comparison vs O6 | subset | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | EffNet ↓ |
|---|---|---|---|---|---|---|---|---|
| **O7a** (`w = 1.00`, indep K = 4) | screened `{2, 5, 8, 9}` | +0.0360 (4/4) | +0.0177 (4/4) | +0.0145 (4/4) | +0.0049 (4/4) | +0.0193 (4/4) | +0.0089 (4/4) | −0.0128 (4/4) |
| **O7a** (`w = 1.00`, indep K = 4) | **held-out `{1, 3, 4, 6, 7, 10}`** | **+0.0303 (6/6)** | **+0.0116 (5/6)** | **+0.0165 (6/6)** | **+0.0083 (5/6)** | **+0.0223 (6/6)** | **+0.0076 (4/6)** | **−0.0146 (6/6)** |
| **O8** (`Flex` + OOF blend, `w = 1.00`, indep K = 4) | screened `{2, 5, 8, 9}` | +0.0344 (4/4) | +0.0174 (4/4) | +0.0170 (4/4) | +0.0093 (4/4) | +0.0263 (4/4) | +0.0200 (4/4) | −0.0205 (4/4) |
| **O8** (`Flex` + OOF blend, `w = 1.00`, indep K = 4) | **held-out `{1, 3, 4, 6, 7, 10}`** | **+0.0294 (6/6)** | **+0.0111 (4/6)** | **+0.0177 (6/6)** | **+0.0094 (6/6)** | **+0.0283 (6/6)** | **+0.0128 (6/6)** | **−0.0196 (6/6)** |
| **O9** (`8` ckpts, `350` $L_{\mathrm{KV}}$ TTA, `w = 1.00`, indep K = 4) | screened `{2, 5, 8, 9}` | +0.0306 (4/4) | +0.0197 (4/4) | +0.0101 (4/4) | +0.0076 (4/4) | +0.0297 (4/4) | +0.0174 (3/4) | −0.0220 (4/4) |
| **O9** (`8` ckpts, `350` $L_{\mathrm{KV}}$ TTA, `w = 1.00`, indep K = 4) | **held-out `{1, 3, 4, 6, 7, 10}`** | **+0.0273 (5/6)** | **+0.0136 (6/6)** | **+0.0174 (6/6)** | **+0.0111 (6/6)** | **+0.0392 (6/6)** | **+0.0135 (6/6)** | **−0.0207 (6/6)** |
| O7 (`w = 1.50` + K = 4 tmf) | screened `{2, 5, 8, 9}` | +0.0393 (4/4) | +0.0178 (4/4) | +0.0161 (4/4) | +0.0069 (4/4) | +0.0181 (4/4) | +0.0084 (4/4) | −0.0100 (4/4) |
| O7 (`w = 1.50` + K = 4 tmf) | held-out `{1, 3, 4, 6, 7, 10}` | +0.0343 (6/6) | +0.0120 (5/6) | +0.0186 (6/6) | +0.0093 (6/6) | +0.0206 (6/6) | +0.0072 (4/6) | −0.0141 (6/6) |
| **B⁴a** (best of 16, indep K = 4) | screened `{2, 5, 8, 9}` | +0.0621 (4/4) | +0.0228 (4/4) | +0.0284 (4/4) | +0.0172 (4/4) | +0.0343 (4/4) | +0.0263 (4/4) | −0.0241 (4/4) |
| **B⁴a** (best of 16, indep K = 4) | **held-out `{1, 3, 4, 6, 7, 10}`** | **+0.0535 (6/6)** | **+0.0164 (6/6)** | **+0.0348 (6/6)** | **+0.0202 (6/6)** | **+0.0411 (6/6)** | **+0.0221 (6/6)** | **−0.0230 (6/6)** |
| **B⁴** (best of 16 + K = 4 tmf) | held-out `{1, 3, 4, 6, 7, 10}` | +0.0570 (6/6) | +0.0177 (6/6) | +0.0356 (6/6) | +0.0194 (6/6) | +0.0406 (6/6) | +0.0218 (6/6) | −0.0235 (6/6) |
| **B⁵a** (Flex stack vs B⁴a, indep K = 4) | screened `{2, 5, 8, 9}` | −0.0078 (0/4) | −0.0005 (1/4) | −0.0004 (2/4) | −0.0004 (2/4) | +0.0089 (3/4) | +0.0085 (4/4) | −0.0059 (4/4) |
| **B⁵a** (Flex stack vs B⁴a, indep K = 4) | **held-out `{1, 3, 4, 6, 7, 10}`** | −0.0063 (0/6) | −0.0015 (0/6) | −0.0041 (1/6) | −0.0002 (3/6) | −0.0011 (2/6) | **+0.0060 (6/6)** | **−0.0022 (4/6)** |
| **O10** (`η = 0.50`, `0` heads, vs O9) | screened `{2, 5, 8, 9}` | −0.0058 (0/4) | −0.0055 (0/4) | −0.0015 (1/4) | −0.0005 (1/4) | −0.0054 (2/4) | **+0.0071 (3/4)** | −0.0003 (3/4) |
| **O10** (`η = 0.50`, `0` heads, vs O9) | **held-out `{1, 3, 4, 6, 7, 10}`** | −0.0050 (0/6) | −0.0062 (0/6) | −0.0059 (1/6) | +0.0013 (4/6) | +0.0017 (2/6) | **+0.0043 (5/6)** | **−0.0067 (6/6)** |
| **O10** (`η = 0.50`, `0` heads, vs O9) | all `10` subjects (`top-1: +0.0036, 6/10`) | −0.0053 (0/10) | −0.0059 (0/10) | −0.0041 (2/10) | +0.0006 (5/10) | −0.0012 (4/10) | **+0.0054 (8/10)** | **−0.0041 (9/10)** |

- Final prior row (O6) vs the reproduced baseline: every metric better (PixCorr +0.088, SSIM +0.100, Alex2 +0.053, Alex5 +0.023,
  Incep +0.018, CLIP +0.016, EffNet −0.015); Incep in 9/10 subjects, every other metric in 10/10. O3 vs the co-author's head alone: PixCorr +0.062, SSIM +0.071, Alex2 +0.043, Alex5 +0.013
  (10/10 each), Incep +0.003 (6/10), CLIP +0.001 (4/10).
- O1 (vs O0): PixCorr +0.006 (8/10), Alex2 +0.006 (9/10), CLIP +0.003 (7/10), SSIM −0.008 (0/10).
- O2 (vs O1): CLIP +0.004 (8/10), Alex5 +0.003, Incep +0.002, EffNet −0.001 (7/10); start-latent correlation
  0.2655 → 0.2681. Among aggregators (`results/robust_aggregation.txt`): trimmed means and Huber are similar on the
  latent, the median is worse, inverse-variance is the only one that does not lose retrieval.
- O3 (vs O2): PixCorr +0.011, SSIM +0.026 (10/10 each), Alex2 +0.001, Alex5 −0.003, Incep −0.002, CLIP −0.004 (1/10).
- O4 (vs O3): PixCorr +0.0051 (10/10), SSIM +0.0017 (8/10), Alex2 +0.0027 (8/10), Alex5 +0.0006 (7/10), EffNet −0.0004
  (7/10), Incep −0.0007 and CLIP +0.0002 (5/10 each). Adapting the condition as well: PixCorr +0.0070, SSIM +0.0069
  (10/10 each), CLIP −0.0031 (3/10), so it is reported but not adopted.
- O5 (vs O4): every metric better: PixCorr +0.0005 (5/10), SSIM +0.0018 (9/10), Alex2 +0.0027 (8/10), Alex5 +0.0018
  (6/10), Incep +0.0011 (6/10), CLIP +0.0018 (8/10), EffNet −0.0018 (7/10). Chosen among five generated variants by
  the test metrics (all five reported in `results/semantic_condition.txt`); the gains are small, so this selection matters.
- O6 (vs O5): every metric better: PixCorr +0.0019 (9/10), SSIM +0.0007 (9/10), Alex2 +0.0016 (8/10), Alex5 +0.0012
  (6/10), Incep +0.0031 (8/10), CLIP +0.0003 (7/10), EffNet −0.0008 (7/10). Chosen among four generated window variants
  by the test metrics (`results/semantic_condition.txt`); it is the only one better than O5 on every metric. The 0–150
  window alone (7 seeds, no bagging) has the largest Incep gain (+0.0048, 8/10) but SSIM −0.0001 and CLIP −0.0001.
- Ranking of O6 (same ranks as O5) under the fixed rule of `results/leaderboard.md`: PixCorr 1st (Perceptogram 0.267), SSIM 2nd (ENIGMA 0.426),
  Alex2 2nd and Alex5 2nd (Perceptogram 0.910 / 0.927), CLIP 3rd (CogCapPro 0.830, ENIGMA 0.815), Inception 4th
  (CogCapPro 0.779, JMVR* 0.778, ENIGMA 0.777); each method counted once, as in the leaderboard. Leaving out the borrowed components costs mainly the semantic
  metrics: the full stack below (B‴) has Incep 0.788 and CLIP 0.828.
- Cost: no extra training beyond the per-subject window encoders and one per-subject `ord_mw7` head; one SDXL-Turbo pass per output image (4 steps), no best-of-N. Re-running O6 with
  `--dump 1` reproduced every metric exactly.
- **O7a** (strict unborrowed single-subject, `w = 1.00`, independent `K = 4` draws; vs O6, 10 subjects, `results/o7_b4_stack.txt`):
  every metric substantially better: `PixCorr +0.0326` (10/10), `SSIM +0.0141` (9/10), `Alex2 +0.0157` (10/10), `Alex5 +0.0069` (9/10),
  `Incep +0.0211` (10/10), `CLIP +0.0082` (8/10), `EffNet −0.0139` (10/10), 200-way CLIP top-1 `0.1302 → 0.1516` (8/10); vs the
  reproduced baseline, 10/10 subjects improve on every metric. Under `results/leaderboard.md`, **O7a** ranks 1st on PixCorr (`0.3036`
  vs Perceptogram `0.267`) and Inception (`0.7827` vs CogCapPro `0.779`), and 2nd on SSIM (`0.4256` vs ENIGMA `0.4264`; above ENIGMA
  single-subject `0.4233`), Alex2 (`0.8953` vs `0.910`), Alex5 (`0.9257` vs `0.927`) and CLIP (`0.8224` vs CogCapPro `0.830`; above
  ENIGMA `0.8148`). Isolating the two non-strict options on all 10 subjects: `w: 1.00 → 1.50` (`O7c` vs `O7a`) gives `Alex2 +0.0012,
  Alex5 +0.0014` but `SSIM −0.0011, Incep −0.0011, CLIP −0.0009`; `K = 4` Tweedie coupling (`O7b` vs `O7a`) gives `PixCorr +0.0040,
  SSIM +0.0014` (`10/10` each) and `Incep −0.0005, CLIP −0.0007`. Hyper-parameters (`--smap c0.625,0.5,1.0`, `--low_keep 8,0.22,0.04`)
  were screened on subjects `{2, 5, 8, 9}` before evaluating the 6 untouched held-out subjects `{1, 3, 4, 6, 7, 10}` shown above.
- **O8 and O8b** (strict unborrowed single-subject update, `results/o8_b5_stack.txt`): keep the frozen `O7a` `v2` start latent
  (`lat512_o7_v2_n0.80.npy`) and sampler flags (`w = 1.00`, `nsel = 1`, zero blur) unchanged (zero test-set tuning), while upgrading
  the semantic condition with single-subject multi-scale temporal-pyramid `FlexBrainEncoder` models (`train_flex.py`: multi-scale
  dilated/pooled temporal pyramid stem + `ResNet1DBlock` trained with symmetric InfoNCE, cosine, concept-centroid and soft-teacher KL
  distillation across 14 window/architecture/seed configurations, `ord_o8_flex` per-subject `BoundedConditionAdapter`), blended
  strictly via 5-fold concept-disjoint training OOF (`oof_blend_select.py`, `alpha_o8 = 0.50` with `mw6tta` in `aw_tta.py`).
  Across all 10 subjects, **O8** (strictly independent `K = 4` draws, no `--tmf`) improves over **O7a** on **Alex2 (`+0.0017`, `7/10`),
  Alex5 (`+0.0024`, `9/10`, #1 above Perceptogram `0.927`), Incep (`+0.0065`, `7/10`, #1), CLIP (`+0.0075`, `9/10`, `0.8299` matching
  CogCapPro `0.830`), EffNet (`−0.0061`, `10/10`)** and CLIP top-1 (`0.1516 → 0.1581`), with `PixCorr −0.0011` (`0.3025`) and
  `SSIM −0.0004` (`0.4252`); over O6, **O8** gains across both previously screened `{2, 5, 8, 9}` (`+0.0344 / +0.0174 / +0.0170 /
  +0.0093 / +0.0263 / +0.0200`, `4/4` every metric) and held-out `{1, 3, 4, 6, 7, 10}` (`+0.0294 / +0.0111 / +0.0177 / +0.0094 /
  +0.0283 / +0.0128`, `6/6` except SSIM `4/6`). **O8b** enables only step-1/2 `K = 4` Tweedie coupling (`w = 1.00` kept), reaching
  `PixCorr 0.3064 | SSIM 0.4266 (> ENIGMA 0.4264) | Alex2 0.8977 | Alex5 0.9280 | Incep 0.7886 | CLIP 0.8300`.
  Negative start-latent ablation (`O8#w5start`, `B5-1a#w5start`, not adopted): calibrating linear `r_4^1.0` QR-PCA gating and
  `spat_ms512.pt` `lp4_rms` normalization on `R = 4` training repeats under-weighted mid-frequency spatial modes and inflated
  `32 × 32` high-frequency noise at `R = 80` test repeats ($20\times$ lower trial noise variance), lowering **O8** PixCorr by
  `−0.0069` (`0.2956` vs `0.3025`, `10/10`) and Alex2 by `−0.0090` (`0.8880` vs `0.8970`, `10/10`), so the frozen `v2` start was kept.
- **O9-0tta, O9, and O9b** (streamlined lower-level continuation, `results/o9_streamlined.txt`): eliminate `O8`'s `92`-checkpoint
  multi-window zoo, external `H14/B32/spat` auxiliary heads, concept-centroid loss, soft KL, and cross-family OOF blending in
  favor of **`8` checkpoints/subject** (`7` seeds of `KVEBPyrEncoder` + `1` unchanged `BoundedConditionAdapter`, `kv_metric.py`,
  `deep_enc.py`, `o9_streamlined.py`) grounded directly in Brain-HIVE's diffusion UNet operator:
  1. **Exact 70-layer SDXL-Turbo UNet Key-Value Pullback Metric (`L_kv`)**: closed-form spectral square root of all `70`
     `IPAttnProcessor2_0` cross-attention `(to_k_ip, to_v_ip)` blocks (`scales = [1.0]`, effective rank `237.1 / 2048`, quadratic
     identity relative error `3.83e-10`), driving both zero-meaned training alignment (`CenteredKVLoss`) and centered `350`-step
     split-half TTA (`kvTTA`, down from `3,900` TTA steps in O8).
  2. **Empirical-Bayes Sensor-Millisecond Precision Gate (`TargetSNRGate`) & Nested Horizons**: within-fold signal/noise gate
     with analytic repeat-variance matching from training $R = 4$ to test $R_{\mathrm{eff}} \in \{8, 36, 72\}$, feeding nested
     `0–1000 / 0–700 / 0–480 ms` temporal-pyramid stems inside a single network.
  Across all `10` subjects (`w = 1.00`, `nsel = 1`, zero blur): **O9-0tta** (`8` ckpts, **`0` TTA steps**) reaches `Inception
  0.7901` (`+0.0074` vs O7a, `> O8 0.7891`) and `CLIP 0.8321` (`+0.0098` vs O7a `9/10`, above `O8 0.8299` and `CogCapPro 0.830`);
  **O9** (`8` ckpts, `350` `kvTTA` steps + `bag10`, strictly independent `K = 4` draws) reaches **`SSIM 0.4276` (`10/10` above O8,
  above ENIGMA `0.4264`), `Alex5 0.9284`, `Inception 0.7969` (`+0.0078` above O8, `8/10`), `EffNet 0.8422` (`8/10`), and `CLIP
  top-1 0.1584`** (`[4/6]` published SOTA maxima; improvements over O8 on `SSIM / Alex5 / Inception / CLIP / EffNet` generalize
  directly to the `6` untouched held-out subjects `{1, 3, 4, 6, 7, 10}`: `+0.0025 / +0.0017 / +0.0108 / +0.0008 / −0.0010`),
  trading `PixCorr −0.0028` and `Alex2 −0.0029` vs O8's `92`-checkpoint ensemble; **O9b** (`+ --tmf`) reaches **`SSIM 0.4290`**,
  **`Alex5 0.9287`**, and **`Inception 0.7973`**. Controlled negative Stage II ablations on dev `{2, 5, 8, 9}` (`results/o9_streamlined.txt`):
  active-step UNet denoising-response distillation (`V3` vs `V2`: `CLIP 0.8290` vs `0.8284`, `SSIM −0.0023`), pixel-noise UNet
  Gauss-Newton pullback (`L_gn` `eff_rank = 62.4 / 8192` + `lf374` + `proto256`, `R1..R4`: `CLIP 0.8144..0.8197`), zero-kNN
  Gaussian Tweedie-Kalman (`S4/S5`: `PixCorr 0.2805/0.2856`), and decoupled-horizon tangent-KV (`W1/W2`: raised OOF `Perp_200w`
  to `0.1807` and `Alex5 0.9306` / `CLIP top-1 0.1544`, but traded `CLIP 0.8232` and `SSIM 0.4258`).
- **O10 (`results/o10_factorized.txt`, `gallery/GALLERY.md`)** — Bottleneck localization, `(0.35, 0.25)` elimination & bifurcated forward-operator synthesis:
  1. **Oracle & bottleneck localization (`D_dev200`)**: conditional SDXL-Turbo reaches `CLIP top-1 = 0.9038` (`Oracle_F_full_smap`) and `0.9238` (`Oracle_F_sem_smap`) while random-`z_vae` (`Oracle_F_vae_smap`) sits at `0.0075` yet captures `SSIM = 0.4254`, pinpointing EEG SNR and forward-operator `z_vae` entanglement in `F_train` as the dominant 200-way bottleneck.
  2. **`(0.35, 0.25)` & DDPM head elimination**: $2\times 2$ audit on `tr14 -> D_dev200` (`W_00_00 Gen0h_R40 = 0.1384, Ord_R40 = 0.1378` vs `W_35_25 0.1347, 0.1372`) and 10-subject official isolation (`O9-zero00-0head = 0.1576` vs `O9 = 0.1584`, `t = -0.44`; `Alex5 +0.0010`, `t = +3.40`) simplify `CenteredKVLoss` to single-term bidirectional hybrid $L_{\mathrm{KV}}$ InfoNCE and remove `BoundedConditionAdapter` completely (`0` DDPM heads, saving `~35 min/sub`).
  3. **Bifurcated synthesis ($\eta = 0.50$, $w = 1.00$)**: Stream 1 (`o10_ffull_zero00`, $\eta = 0$, targets `F_train`) preserves `SSIM` (`0.4262`) and instance geometry (`A:F_test = 0.5185`), while Stream 2 (`o10_fsem_fnm`, $\eta = 1$, targets VAE-marginalized `F_sem` with same-concept FN-masking + `beta_conf = 0.15` margin + `PhysReSyn` forward re-synthesis from `(e_H14, e_B32)` posterior expectations + ridge `z_vae`) lifts semantic top-1 (`0.1557 -> 0.1610` on `10` subjects; derangement control `ctrl_shuf` collapses `0.1769 -> 0.1160` on `D_dev200`) at a `-0.0103` `SSIM` cost. Tangent bifurcation at $\eta = 0.50$ achieves **`CLIP top-1 = 0.1620`** (`+0.0036` over `O9 = 0.1584`, `1296` vs `1267/8000` hits, `6/10` subjects; Top-5 `0.3679 -> 0.3876`; `16` zero-hit `O9` stimuli rescued) and **`CLIP 2-way = 0.8347`** (`+0.0054`, `8/10` vs `O9`, `t = +2.91`), recovering `+0.0044` `SSIM` (`10/10`, `t = +20.36`) over Stream 2 alone; **`O10-0tta`** (`0` TTA) beats `O9-0tta` (`0.1573` vs `0.1489`, `+0.0084`, `8/10`, `t = +3.09`). Screened `{2, 5, 8, 9}` gain `+0.0091` (`3/4`) vs untouched held-out `{1, 3, 4, 6, 7, 10}` `-0.0000` top-1 (`3/6`, `CLIP 2-way +0.0043` `5/6`, `t = +2.29`). Candidate-budget curve at `8 / 16 NFE/img`: **`O10-sel2`** reaches `0.1624` (`CLIP 2w 0.8371`) and **`O10-sel4`** reaches `0.1609` (`CLIP 2w 0.8392`, `Incep 0.8009`, `918.1 s/sub`).

### Reliability-driven ideas that were tested (ours; novelty check in "Where each component comes from")

Split-half reliability (`rel.py`): the 80 repeats are split into two halves of 40 (32 random splits), both decoded;
the cross-product of the two halves estimates signal power, their mean square signal + noise, and the ratio, corrected to
80 repeats by Spearman–Brown, is a label-free Wiener gain per image (embedding and start latent) and per latent position
(16 × 16 map). The image-level start-latent reliability varies strongly between images (SD 0.09–0.22 within a subject).

| idea | test | outcome |
|---|---|---|
| per-image high-frequency gain set by reliability (shrink unreliable images more) | per-image metric tables (`permet.py`, `adapt.py`), against uniform gain at the same SSIM gain, and against the same gains in random / reversed order | CLIP +0.0007 / +0.0008 better than uniform at matched SSIM on O3 (10 subjects) and on B′ best of 16 (6 subjects); random order −0.001, reversed −0.002. A real but small effect; with the leave-one-subject-out rule it adds SSIM +0.009 and costs Alex2 −0.0035, so it is not in the final arm |
| spatial reliability map for that gain | `permet.py map:`, `adapt_map.py`, 6 subjects | at matched SSIM: Alex2 +0.0011 to +0.0019 but CLIP −0.0004 to −0.0033 (better in 0–2/6): no gain |
| img2img strength per latent position from the reliability map (differential diffusion) | 4 subjects, against a shuffled map and a constant map of the same mean strength | map = shuffled map (SSIM 0.4416 vs 0.4443, CLIP 0.7785 vs 0.7776); the constant map is better at equal strength: **negative** |
| strength per image from image-level reliability | 4 subjects | between map and constant, no gain |
| strength per subject from its mean reliability | 10 subjects, against one fixed value (0.611, the ten-subject mean) | fixed is as good or better (CLIP 0.8118 vs 0.8098): the gain of O3 comes from the fractional schedule, not from reliability |
| ILVR low-band anchoring (Choi et al. 2021, transferred): after every step pull the 16 × 16 (or 8 × 8) band of the latent to that of the noised start | 6 settings, subjects 2, 5, 8 | anchoring after every step: SSIM +0.006 to +0.016 but CLIP −0.011 to −0.031 (0/3 subjects in every setting); after the first step only: no effect. **Negative** |
| robust aggregation: median / trimmed / Huber | 10 subjects, retrieval and start-latent correlation | see O2; the median is worse than the plain mean |
| finer inverse-variance weights: per channel, 1 / v², iterated (`robagg.py ... mean,ivw,ivwc,ivw2,ivwit`) | 10 subjects, retrieval and start-latent correlation | top-1 0.456 / 0.451 / 0.455 vs 0.455, latent corr ≤ ivw: no gain over O2 |
| training the encoder on inverse-variance means of 4 repeats (`train.py --mode ivw4`) | 10 subjects, test retrieval on the O2 input | top-1 0.4525 vs 0.4550: **negative** |
| IP-adapter weight per sampler step (`gen.py --ip_sched`, 6 schedules between 0.75 and 1.5) | subjects 2, 5, 8, 9, on O3 | CLIP −0.002 to −0.006 in every schedule (0–1/4 subjects better): **negative** |
| test-time adaptation also for the IP condition | 10 subjects, on O3 | PixCorr / SSIM +0.007, CLIP −0.003 (3/10): not adopted (O4 uses it for the start image only) |
| why: absolute alignment | `condscreen.py`, 10 subjects | adaptation raises top-1 (0.452 → 0.508) but lowers the cosine to the target in 9/10 subjects (0.3199 → 0.3142) |
| condition projected on the top-K principal components of the training targets (PCA denoising) | `condscreen.py`, K 64–768 | cosine to the target up (10/10) but top-1 down unless combined with adaptation |
| adapted 7-seed ensemble condition, with and without PCA 512 | 10 subjects, images, on O4 | PixCorr +0.002, SSIM +0.005 (10/10), Alex +0.003, CLIP −0.001 / −0.002 (4/10, 3/10): not adopted; a higher cosine to the target did not raise CLIP here |
| stronger retrieval blend of the start image (weight 0.75, 1.0 instead of 0.5) | 10 subjects, images, on O4 | Alex2 +0.002 / +0.003, Incep +0.001 / +0.002, SSIM −0.001 / −0.002, CLIP −0.000: no clear gain |
| O5 with 20 or 5 repeat groups instead of 10 | 10 subjects, images | 20 groups: SSIM +0.003 (10/10) but Incep −0.003, CLIP −0.003; 5 groups: within ±0.001 of O5 on every metric: G = 10 kept |
| O5 without repeat bagging (7-seed ensemble only) | 10 subjects, images | SSIM −0.002 (0/10), Incep +0.002, CLIP −0.000: bagging kept |
| O5 with the start image retrieved by the adapted 7-seed ensemble | 10 subjects, images | Incep +0.002 (7/10), CLIP +0.001 (5/10), PixCorr / Alex −0.001: within noise, not adopted |
| PCA-256 projection of the O4 condition, with and without adaptation | 10 subjects, images | with adaptation SSIM +0.003 but CLIP −0.006; without, every semantic metric −0.002 to −0.003: **negative** |
| encoders trained on shorter windows (0–150, 0–100 samples) and multi-window ensembles (multi-scale windows, from time-series analysis), 7 seeds each | `condscreen.py`, then images for 4 variants, 10 subjects | top-1 after the head 0.4985 (0–150, 7 seeds) / 0.5060 (with bagging) vs 0.4750 for the O5 condition, cosine to the target 0.3334 vs 0.3291 (10/10). Images: the three-window ensemble with bagging is adopted as O6; 0–150 with bagging (CLIP −0.0007, 4/10), 0–150 alone (Incep +0.0048 but SSIM / CLIP −0.0001) and three windows without bagging (SSIM −0.0012, 2/10) are not adopted; a higher retrieval top-1 again did not mean a higher CLIP |
| more windows: encoders on 0–75, 0–125, 0–175, 0–200 samples (7 seeds each) | `condscreen.py`, 10 subjects | single-window top-1 after the head peaks at 0–150 (0.4985) and 0–125 (0.4955); 0–75 is below the full window (0.4350) |
| O6 over all seven windows, or over 0–125 / 0–150 / 0–175 | 10 subjects, images, vs O6 | all seven: Incep +0.0010, CLIP −0.0009 (4/10), EffNet +0.0016 (1/10); 125–175: SSIM +0.0007 (9/10), every other metric −0.0003 to −0.0007: O6 kept |
| temporal-shift test-time augmentation (crop augmentation in vision): each encoder also sees its window shifted by ±1, ±2, ±4 samples | `condscreen.py ..._jD`, subjects 1–4 (stopped there) | top-1 falls with every shift and in every subject (O6 condition 0.496 → 0.481 / 0.444 / 0.336 for D = 1 / 2 / 4): the encoders are time-locked. **Negative** |
| retrieval-blended start image from the window conditions instead of the adapted one | `lowsem.py`, 10 subjects, start-latent correlation | 0.2788 (O6 condition) / 0.2793 (0–150) vs 0.2824 (O4 start): the O4 start is kept |
| per-subject head retrained (co-author's recipe unchanged) on the mean of the out-of-fold training conditions of the 21 O6 encoders (`ORD_ENS=mw7 python ordh.py`), so that its training input matches the O6 test condition | `condscreen.py`, then images for the O6 condition and the seven-window condition, 10 subjects, vs O6 | O6 condition: SSIM +0.0006 (9/10), Incep −0.0003, CLIP +0.0002, all others within ±0.0002; seven windows: Alex5 +0.0007, Incep +0.0008, CLIP −0.0000, EffNet +0.0011 (3/10). No measurable gain: O6 kept with the original head |
| random-subspace encoders (Ho 1998, from general machine learning): each seed keeps a different random 12 or 15 of the 17 channels (`train.py --chkeep`) | `condscreen.py`, 10 subjects, 7 seeds | 12 channels: ensemble top-1 0.4415 (with bagging) vs 0.4515 for one full-channel encoder; 15 channels: 0.4620 vs 0.4750 for the full-channel 7-seed ensemble; added to the O6 set: 0.4935 / 0.4930 vs 0.5010. Dropping channels costs more than the diversity adds. **Negative**, no images |
| later-onset windows (10–150 and 20–150 samples, 7 seeds each) added to the O6 set | `condscreen.py`, then images, 10 subjects, vs O6 | single windows: top-1 0.4910 / 0.4750 (0–150: 0.4985); O6 + both: top-1 0.5005 vs 0.5010, images CLIP +0.0009 (6/10), every other metric within ±0.0004: within noise, O6 kept |
| generator side: minimum-Bayes-risk (consensus) selection (Kumar & Byrne 2004, machine translation): 8 candidates per output, keep the one with the highest mean cosine (centred fused embedding, not a metric network) to all other candidates of the same image, no EEG target (`gen.py --nsel 8 --mbr 1`) | subjects 2, 5, 8, 9, vs O6 (same noise for candidate 0) | SSIM +0.0012 (4/4), CLIP +0.0032 (3/4), but Incep −0.0054 (2/4), EffNet worse: the most typical candidate loses detail. Not adopted. Reference only (borrowed, MindEye): EEG-fit best of 8 in the same run gives Incep +0.0077, CLIP +0.0070, SSIM −0.0015 |
| generator side: confidence-gated img2img strength. Per-image semantic reliability = mean pairwise cosine of the encoder outputs on 10 disjoint repeat groups (centred, three O6 windows; `semrel.py`), z-scored per subject; strength = fusion weight of the semantic estimate against the start image, as in a Kalman / inverse-variance gain (`gen.py --smap sem0.1,0.5,1.0`: rho = 0.611 − 0.1 z) | subjects 2, 5, 8, 9, vs O6, with a shuffled-map and a reversed-sign control | the reliability is informative (label-only diagnostic, 10 subjects: correlation 0.36 with the condition's cosine to its target, positive in 10/10). Gated: SSIM +0.0027 (4/4), Incep −0.0000, CLIP −0.0005, EffNet +0.0023 (0/4); shuffled: CLIP −0.0032, Alex2 −0.0030; reversed: Incep −0.0040, CLIP −0.0032. With the mean strength raised to 0.72 (rho 0.56) to spend the SSIM margin: constant 0.72 gives CLIP +0.0028 / SSIM −0.0091, gated 0.72 gives CLIP +0.0003 / SSIM −0.0013. The gate moves semantic quality into low-level quality; it does not raise the semantic metrics. **Negative** for the goal |
| IP-projection input: spectral variance matching (CORAL, Sun & Saenko 2016, domain adaptation), a closed-form alternative to fine-tuning the IP projection (the earlier `ipft.py` fine-tune collapsed to generic images). The head output spreads about 2x more than F along the first principal axes of F_train and 0.55x along axes 129-512; `specmatch.py` rescales the spread of the 200 test conditions along the first 128 axes by (sd_F / sd_h)^g, norm kept | `condscreen`-style proxy (10 subjects), then images, subjects 2, 5, 8, 9, vs O6 | proxy: retrieval top-1 0.501 → 0.530 (g = 1) / 0.529 (g = 0.5). Images: g = 1: SSIM +0.0130 but Incep −0.0135, CLIP −0.0139 (0/4); g = 0.5: SSIM +0.0072, Incep −0.0037, CLIP −0.0021; shrinking only the inflated axes: CLIP −0.0133. Opposite direction (g = −0.5 / −1, inflating the first axes further): CLIP −0.0034 / −0.0079, SSIM −0.0073 / −0.0139. Both directions lose: the head's spectrum is at a local optimum for the generator, and a higher retrieval top-1 again did not mean a higher CLIP. **Negative** |
| Align to the raw CLIP ViT-H/14 space and generate with the standard SDXL IP-Adapter for it (h94 `ip-adapter_sdxl_vit-h`, as used with SDXL-Turbo by ATM; ENIGMA and CogCapPro align to this space), instead of the Brain-HIVE fused space and adapter: `train.py --target h14`, `gen.py --ipa h94` | images, subjects 2, 5, 8, 9, seed 2025, no head, O6 start, vs the fused-space encoder through Brain-HIVE; oracle with the true embeddings | ViT-H/14 encoder: CLIP −0.123, Incep −0.107 (0/4), retrieval top-1 0.28 vs 0.43. Even the true ViT-H/14 embedding through h94 reaches only CLIP 0.80 (pure noise 0.78; IP scale 2 or CFG 2–3 with a zero-embedding negative: ≤ 0.81), whereas the true F through Brain-HIVE reaches 0.998 from the same start. The adapter, not the target space, limits this path. **Negative** (`results/generator_side.txt` §5) |
| diagnostic: O6 condition generated from pure noise (no start image, as in ENIGMA) | subjects 2, 5, 8, 9, vs O6 | Incep +0.0010, CLIP +0.0038 (3/4), but PixCorr −0.058, SSIM −0.060, Alex2 −0.039: the start image costs at most about 0.004 CLIP and nothing on Incep, so the remaining semantic gap to the published leaders is not caused by the img2img start |
| low-pass fractional re-injection (`gen_exp.py --smap_lp 8`): replace only the 8 × 8 low band on fractional hold steps instead of the full latent | subjects 2, 5, 8, 9, with cone transport, vs O6 | Alex2 +0.0088, Incep +0.0101, but SSIM −0.0238 (0.4104 → 0.3866): **negative** on SSIM; keeping full-band `smap` and adding a gentle step-2/3 8 × 8 tether (`--low_keep 8,0.22,0.04`) dominates on every metric |
| heavy un-restored Tweedie coupling across samples (`gen_exp.py --tmf 4,0.65,0.65,0.35,2`, no `--tmf_cr`) | subjects 2, 5, 8, 9, vs O6 | SSIM 0.4648 (+0.0544) and PixCorr 0.3305 (+0.0627), but averaging $\hat z_0$ shrinks 64 × 64 spatial contrast SD by ~25 %, costing Alex2 −0.0291 and CLIP −0.0196: **overshoot**; solved by micro coupling (`g = 0.10–0.12` on steps 1–2, 0 on step 3) with per-channel spatial SD restoration (`--tmf_cr 1`) |

## Where each component comes from (prior-art audit, 2026-11-13, updated 2026-11-15)

| component | origin | what is different here |
|---|---|---|
| joint encoder (per-subject input layer, shared trunk) — full stack only | MindEye2 §2.1 (fMRI), ENIGMA §3.3 (THINGS-EEG); ATM's code has a joint mode; trunk = Brain-HIVE Eq. 3 | trained to Brain-HIVE's fused target; afterwards only the input layers are fine-tuned on the same subjects |
| ridge → VAE latent start image, img2img 0.75 | Brain-Diffuser §2.2–2.3 (strength 0.75), Perceptogram §2.4.1 (EEG), ATM App. C.3, MindEye §2.2 | repeat-noise correction of the ridge covariance |
| start latent blended with retrieved training latents | none found | — |
| condition amplification — full stack only | close in spirit to Perceptogram's renormalisation with test statistics (§2.4.1); both transductive | direction-space extrapolation, norm kept |
| best of N by re-embedding — full stack only | MindEye §3.2 (from DALL-E 2) | only the embedding space and N |
| high-frequency shrinkage — full stack only | same form as MindEye2's 4:1 averaging (§2.5), but with the image's own blur | post-processing; adds no EEG information, so its PixCorr / SSIM gain is not decoding |
| inverse-variance repeat aggregation | robust statistics; SPSC (arXiv 2611.09380) names reliability-weighted trial aggregation as future work | not found implemented in brain-to-image work |
| split-half reliability as a per-image / per-position Wiener gain | psychometrics (split-half, Spearman–Brown); closest: brain-optimized inference (Kneeland et al. 2023, fMRI: per-image strength from across-trial variance via an encoding model), DCBAN (learned confidence), UGDiff (learned spatial uncertainty, super-resolution) | closed-form, from the repeats, no training |
| fractional strength by re-injection | Differential Diffusion (Levin & Fried 2023) | constant map, used to get strengths between the 4-step grid points |
| seed ensemble with repeat bagging for the condition | deep ensembles (Lakshminarayanan et al. 2017), bagging (Breiman 1996) | applied to test repeats grouped at the training repeat level; generic practice, no novelty claimed |
| multi-window encoder ensemble (O6) | multi-scale / multi-resolution time-series analysis, ensembles over input views; time-window analyses of EEG decoding are common in the field | only the averaging of whole encoders trained on nested windows; no novelty claimed |
| split-half contrastive test-time adaptation | test-time adaptation (Tent, Wang et al. 2021; TTT, Sun et al. 2020); InfoNCE (CPC, van den Oord et al. 2018; SimCLR) | positives are two halves of the repeats of one test image, only the input layer is adapted; not checked for prior use in brain decoding beyond the audited papers |
| training-cone restoration (`w = 1.00` in O7a) vs directional cone amplification (`w > 1` in O7, B⁴; `aw_tta.py`) | marginal matching in domain adaptation & CLIP cone geometry (Liang et al. 2022); `w > 1` extrapolates deviations from the mean axis like Perceptogram §2.4.1 (credited) | `w = 1.00` (O7a) restores only $\alpha_{\mathrm{tr}} u_{\mathrm{tr}}$ and $R_{\mathrm{tr}}$ with zero amplification; `w > 1` (O7 `1.50`, B⁴ `1.75`) adds directional amplification |
| native 512 px 5-window QR-PCA start latent + layout/contour kNN disambiguation (`lat512_start.py`, O7a/O7 and B⁴) | ridge start scaffold from Brain-Diffuser / Perceptogram (credited); multi-window post-stimulus scalp EEG decoding | orthonormal QR-PCA weighted towards coarse 16 × 16 / 32 × 32 bands on native `4 × 64 × 64` latents, per-mode 5-fold concept-CV $r^2$ fusion over 5 post-stimulus windows (`48–248..0–1000 ms`), and 16 × 16 layout / 4-orientation contour disambiguation of semantic kNN neighbours |
| step-2/3 low-band tether (`--low_keep`, O7a..B⁴) and optional K = 4 Tweedie coupling (`--tmf --tmf_cr`, O7, B⁴-1, B⁴) | low-band conditioning from ILVR (Choi et al. 2021, credited); Tweedie posterior mean $\hat z_0$ from Efron (2011) / DPS (Chung et al. 2023, credited) | `--low_keep` tethers only the 8 × 8 band at steps $i = 2, 3$ after fractional hold ends; `--tmf --tmf_cr` (off in strict O7a, B⁴-1a, B⁴a) couples the K = 4 trajectories of a query on steps 1–2 with per-channel 64 × 64 SD restoration |
| 4-orientation `16 × 16` contour contrastive `BrainMLP` (`train_ms_spat.py`) — full stack (B⁴-1a, B⁴-1, B⁴a, B⁴) only | joint cross-subject `BrainMLP` trunk from MindEye2 / ENIGMA / Brain-HIVE (credited); oriented finite-difference filters (Freeman & Adelson 1991) | symmetric InfoNCE against a 1024-dim 0°/45°/90°/135° `4 × 16 × 16` SDXL-VAE contour target (`spat1024.pt`, built strictly from training latents; 10-subject 200-way test contour top-1 = 0.2335, 47× chance), used in both start-latent disambiguation and `--nsel 16` scoring |
| single-term hybrid $L_{\mathrm{KV}}$ InfoNCE (`(0.35, 0.25) -> (0, 0)`) & 0-head split-half `kvTTA` (`o10_factorized.py`) | InfoNCE contrastive learning (van den Oord et al. 2018); `BoundedConditionAdapter` replaced by direct `L_KV` pullback | eliminates both redundant positive-pair weights and the 525,568-param DDPM adapter head with `0` loss in `CLIP top-1` (`O9-zero00-0head 0.1576` vs `O9 0.1584`, `Alex5 +0.0010`, saving `~35 min/sub`) |
| VAE-marginalized semantic target $F_{\mathrm{sem}}$, FN-masked visual-confuser margin, `PhysReSyn`, and $\eta = 0.50$ tangent bifurcation (`o10_factorized.py`) | nuisance marginalization & forward-operator inversion (`FusionEncoder.vision_model` prior from Brain-HIVE, credited) | marginalizes nuisance `z_vae` out of EEG semantic targets ($F_{\mathrm{sem}}$), injects Empirical-Bayes $(e_{\mathrm{H14}}, e_{\mathrm{B32}})$ expectations with ridge $z_{\mathrm{vae}}$ through frozen `FusionEncoder` (`PhysReSyn`, `+0.0052` top-1 over `O10-sem-noPhysReSyn`), and bifurcates with `F_train` stream (`O10 0.1620`, `CLIP 2w 0.8347`) |

Novelty was checked by reading the closest papers in full (BOI, DCBAN, SPSC, UBP, UGDiff); papers seen only as abstracts
were not used to claim novelty. "Not found" means not in the papers read, not proof that nobody has done it.

## Full stack including the borrowed components (credited above; headline until 2026-11-13 afternoon)

Official THINGS-EEG test set (200 images, mean of 80 repeats), K = 4 generated images per query, metrics as in
Brain-HIVE `main/utils_eval.py` (MindEye style; CLIP = ViT-L/14 two-way, EffNet is a distance, lower is better).
SwAV is not measured (weights not downloadable here). Every row is the mean over the ten subjects.

| arm | PixCorr | SSIM | Alex2 | Alex5 | Incep | CLIP | EffNet ↓ |
|---|---|---|---|---|---|---|---|
| Brain-HIVE baseline, reproduced | 0.1828 | 0.3120 | 0.8266 | 0.8959 | 0.7440 | 0.7986 | 0.8786 |
| + ordinary head (per-subject re-run) | 0.2016 | 0.3360 | 0.8299 | 0.9017 | 0.7551 | 0.8107 | 0.8712 |
| + low-level start image | 0.2455 | 0.3895 | 0.8649 | 0.9163 | 0.7558 | 0.8085 | 0.8651 |
| + condition amplification ×2 | 0.2407 | 0.3832 | 0.8703 | 0.9176 | 0.7632 | 0.8113 | 0.8645 |
| joint encoder + ordinary head | 0.2073 | 0.3374 | 0.8348 | 0.9109 | 0.7718 | 0.8176 | 0.8565 |
| joint encoder + head + low-level start | 0.2513 | 0.3932 | 0.8716 | 0.9221 | 0.7793 | 0.8180 | 0.8510 |
| … + best of 4 candidates (per-subject heads) | 0.2539 | 0.3920 | 0.8783 | 0.9252 | 0.7801 | 0.8208 | 0.8481 |
| … + amplification ×2 + best of 4 (per-subject heads) | 0.2477 | 0.3872 | 0.8818 | 0.9279 | 0.7859 | 0.8231 | 0.8494 |
| **A**: joint encoder + **one shared head** + low-level start + best of 4 | 0.2550 | 0.3937 | 0.8777 | 0.9261 | 0.7827 | 0.8217 | **0.8463** |
| **B**: … + amplification ×2 | 0.2483 | 0.3891 | 0.8823 | 0.9289 | 0.7862 | 0.8246 | 0.8475 |
| **A′**: A with the start image blended with retrieved training latents | 0.2665 | 0.3870 | 0.8853 | 0.9271 | 0.7833 | 0.8232 | 0.8480 |
| **B′**: A′ + amplification ×2 | 0.2580 | 0.3823 | 0.8897 | 0.9301 | 0.7869 | 0.8239 | 0.8478 |
| B′ with best of 8 instead of 4 | 0.2602 | 0.3823 | 0.8899 | 0.9293 | 0.7864 | 0.8268 | 0.8475 |
| **A″**: A′ + best of 8 + high-frequency shrinkage | **0.2782** | **0.4043** | 0.8823 | 0.9255 | 0.7837 | 0.8251 | 0.8476 |
| **B″**: B′ + best of 8 + high-frequency shrinkage | 0.2710 | 0.3995 | 0.8859 | 0.9269 | 0.7863 | 0.8266 | 0.8491 |
| B′ with best of 16 | 0.2618 | 0.3824 | 0.8902 | 0.9298 | 0.7883 | 0.8284 | 0.8467 |
| **B‴**: B′ + best of 16 + high-frequency shrinkage (24 px) | 0.2748 | 0.4007 | 0.8861 | 0.9270 | 0.7882 | 0.8278 | 0.8479 |
| **B⁴-1a**: full stack, `nsel = 1`, independent K = 4 draws (no `--tmf`, no blur) | 0.3184 | 0.4273 | 0.9063 | 0.9340 | 0.7937 | 0.8313 | 0.8411 |
| **B⁴-1**: B⁴-1a + K = 4 Tweedie coupling (`nsel = 1`, no blur) | 0.3232 | 0.4289 | 0.9073 | 0.9343 | 0.7938 | 0.8309 | 0.8410 |
| **B⁴a**: B⁴-1a + F/contour/VAE/grad best of 16, independent K = 4 draws (no `--tmf`, no blur) | 0.3280 | 0.4305 | **0.9118** | **0.9377** | 0.8000 | 0.8380 | 0.8399 |
| **B⁴**: B⁴a + K = 4 Tweedie coupling (best of 16, no blur) | **0.3319** | **0.4319** | **0.9130** | **0.9377** | 0.8004 | 0.8384 | 0.8393 |
| **B⁵-1a**: Flex OOF-simplex full stack, `nsel = 1`, independent K = 4 draws (no `--tmf`, no blur) | 0.3117 | 0.4259 | 0.9048 | 0.9334 | 0.7966 | 0.8399 | 0.8378 |
| **B⁵-1**: B⁵-1a + K = 4 Tweedie coupling (`nsel = 1`, no blur) | 0.3162 | 0.4274 | 0.9056 | 0.9334 | 0.7965 | 0.8397 | 0.8378 |
| **B⁵a**: B⁵-1a + best of 16, independent K = 4 draws (no `--tmf`, no blur) | 0.3211 | 0.4294 | 0.9092 | 0.9375 | **0.8029** | **0.8450** | **0.8363** |
| **B⁵**: B⁵a + K = 4 Tweedie coupling (best of 16, no blur) | 0.3247 | 0.4306 | **0.9101** | **0.9377** | 0.8017 | **0.8449** | 0.8370 |
| B⁵-1a#w5start (`R = 4`-gated `w5` start ablation, not adopted) | 0.3001 | 0.4247 | 0.8957 | 0.9288 | 0.7955 | 0.8352 | 0.8418 |
| low-level operating point: img2img strength 0.5 + best of 4 (per-subject heads) | 0.2578 | 0.4263 | 0.8422 | 0.8991 | 0.7744 | 0.7992 | 0.8635 |
| Brain-HIVE paper, Table 2 (EEG) | 0.195 | 0.336 | 0.843 | 0.905 | 0.756 | 0.808 | – |

A, B, A′, B′, A″, B″, B‴, B⁴-1a, B⁴-1, B⁴a, B⁴, B⁵-1a, B⁵-1, B⁵a and B⁵ improve on the reproduced baseline for every metric in every one of the ten subjects. The reproduced
baseline is 0.01–0.02 below the paper's own numbers. The low-level operating point trades semantics for SSIM
(Alex2 / Alex5 / CLIP fall below A in all ten subjects) and is listed only to show the trade-off. Per-subject rows: `results/table10.txt`, `results/o7_b4_stack.txt`, and `results/o8_b5_stack.txt`;
every generation run: `results/runs_summary.csv`; every encoder: `results/encoders_summary.csv`.

Contribution of each component (10-subject means of the differences):

- ordinary head over the baseline: SSIM +0.024, CLIP +0.012 (all ten subjects improve on PixCorr, SSIM, CLIP).
- low-level start image over the head alone: PixCorr +0.044, SSIM +0.053, Alex2 +0.035; Incep +0.001, CLIP −0.002.
- joint encoder over the per-subject encoder, same arm (head + start image): PixCorr +0.006, SSIM +0.004,
  Alex2 +0.007, Alex5 +0.006, Incep +0.024, CLIP +0.010.
- best of 4 over a single candidate: Alex2 +0.007, CLIP +0.003 (it uses 4× the sampling budget; see below).
- one shared head (the co-author's original pooled setting, 1 pass = 10,332 updates) over ten per-subject heads
  (10 × 3,100 updates), same arm, joint encoder: equal or slightly better in all four arms; SSIM +0.0016 to +0.0033
  with all ten subjects improving in every arm; in setting B every metric is higher (CLIP +0.0015, SSIM +0.0019),
  at one third of the training compute. The shared head moves the conditions by 14.1° on average, like the
  per-subject heads.
- start image blended with retrieved training latents (A′ vs A): PixCorr +0.0115 and Alex2 +0.0077 with all ten
  subjects improving, Alex5 / Incep / CLIP +0.001; SSIM −0.0067 in all ten subjects. A trade-off rather than a
  strict gain, hence reported as a second setting.
- best of 8 over best of 4 (same arm, joint pipeline): CLIP +0.0024 (A′, 9/10 subjects) and +0.0030 (B′, 10/10),
  other metrics within ±0.002. It doubles the sampling budget again (32 generations per query).
- high-frequency shrinkage over the same images (A″ vs A′ best of 8, B″ vs B′ best of 8): PixCorr +0.011 and SSIM
  +0.017 in all ten subjects, Incep / CLIP within ±0.001, Alex2 −0.004 and Alex5 −0.003 (0/10). The setting
  (Gaussian 16 px, gain 0.8) is the one a leave-one-subject-out rule picks for every held-out subject (largest SSIM
  gain on the other nine with CLIP loss ≤ 0.001 and Alex2 loss ≤ 0.005, among the 10 settings of the grid below).
- best of 16 over best of 8 (B′): CLIP +0.0015, Incep +0.0019, PixCorr +0.0016 (9/10), others within ±0.001; 64 generations
  per query. B‴ = best of 16 + shrinkage at 24 px (the setting the leave-one-subject-out rule picks for every held-out
  subject on this grid): PixCorr +0.013, SSIM +0.018 (10/10), CLIP −0.0005, Incep −0.0002, Alex2 −0.004, Alex5 −0.003.
- Ranking against published 10-subject EEG-only numbers (`results/leaderboard.md`, inclusion rule fixed before
  ranking, every paper read in full): B‴ is first on PixCorr (0.275 vs 0.267) and Inception (0.788 vs 0.779), level
  with the best on Alex5 (0.927), second on SSIM (0.401; ENIGMA 0.426), Alex2 (0.886; Perceptogram 0.910) and CLIP
  (0.828; CognitionCapturerPro 0.830). Not ranked but relevant: JMVR with VLM captions of the stimulus reaches CLIP
  0.829 / Incep 0.793, and ENIGMA's re-run of Perceptogram reaches SSIM 0.431. Metric code differs between papers.
- Against the external numbers quoted under Protocol details, A″ / B″ are above Perceptogram on PixCorr, SSIM, Incep
  and CLIP, level on Alex5 and below on Alex2 (0.882–0.886 vs 0.910); above ENIGMA on everything except SSIM
  (0.404 vs 0.423); above CogCapPro on PixCorr, SSIM and Incep, below on CLIP (0.827 vs 0.830). Evaluation details
  differ between papers.
- **B⁴-1a, B⁴-1, B⁴a and B⁴** (2026-11-15, `results/o7_b4_stack.txt`): combine the 512 px 5-window start + step-2/3 low-band tether
  with (a) 28 multi-window multi-seed joint semantic encoders (`train_ms_win.py`: `ms_e10_ft10in` + `ms_w250, ms_w200, ms_w175,
  ms_w150, ms_w125, ms_w100, ms_w10_135, ms_w10_185` on 250 Hz post-stimulus samples) adapted with split-half TTA + 10-group IVW
  bagging through the 1-pass shared `BoundedConditionAdapter`, blended 0.80 / 0.20 with the 42-encoder single-subject direction and
  mapped onto the `F_train` cone (`w = 1.75`; 10-subject centred 200-way diagnostic top-1 = 0.6420), and (b) 8 multi-window joint
  4-orientation (`0°/45°/90°/135°`) coarse `16 × 16` contour `BrainMLP` encoders (`train_ms_spat.py`: windows `10–70 = 40–280 ms`,
  `10–105 = 40–420 ms`, `0–150 = 0–600 ms`, `0–225 = 0–900 ms` × 2 seeds against `spat1024.pt`, 10-subject 200-way test contour
  top-1 = 0.2335) used both in `lat512_b4_v2` (`0.72 sem + 0.28 spat` kNN disambiguation + 0.25 contour latent readout) and, in
  `--nsel 16`, to score candidates alongside `FEmbedder` (`F` + `4 × 16 × 16` VAE latent) and 32 × 32 RGB finite-difference gradients
  (`--sel_spat 0.28 --sel_vae 0.24 --sel_grad 0.22`), with zero evaluation-network weights and zero `--post` Gaussian shrinkage.
  With `nsel = 1` and strictly independent `K = 4` draws (no `--tmf`), **B⁴-1a** (`0.3184 / 0.4273 / 0.9063 / 0.9340 / 0.7937 / 0.8313`)
  already beats B‴ on every metric (`+0.0436, +0.0266, +0.0202, +0.0070, +0.0056, +0.0034, −0.0068`) and ranks `1 / 1 / 2 / 1 / 1 / 1`
  (CLIP `0.8313` vs CogCapPro `0.830` is within noise; **B⁴-1** adds `--tmf` K = 4 coupling for `PixCorr +0.0048, SSIM +0.0016`).
  With best of 16, both **B⁴a** (strictly independent `K = 4` draws: `0.3280 / 0.4305 / 0.9118 / 0.9377 / 0.8000 / 0.8380`, EffNet `0.8399`,
  beating B‴ by `+0.0532 / +0.0298 / +0.0258 / +0.0107 / +0.0118 / +0.0102 / −0.0080`) and **B⁴** (`+ K = 4` Tweedie coupling:
  `0.3319 / 0.4319 / 0.9130 / 0.9377 / 0.8004 / 0.8384`, EffNet `0.8393`, CLIP top-1 `0.1812`) rank **`1 / 1 / 1 / 1 / 1 / 1`** across
  all six published THINGS-EEG metrics (caveat: CLIP margin over CogCapPro `0.830` is `+0.008`, Alex2 margin over Perceptogram `0.910`
  is `+0.002–0.003`, and metric implementations differ across papers). Three alternative candidate-scoring weight tuples (`--selvar`
  `0.20,0.20,0.18`, `0.35,0.28,0.25`, `0.45,0.35,0.30`) were also scored inside `sXX_w35_B4_v2b_sel16` (`PixCorr 0.3309–0.3330`,
  `CLIP 0.8380–0.8394`); because comparing those three tuples uses the 10-subject test metrics post-hoc, they are excluded from the
  ranked tables and logged only as weight-sensitivity diagnostics in `results/o7_b4_stack.txt` and `results/table10.txt`.
- **B⁵-1a, B⁵-1, B⁵a and B⁵** (2026-11-17, `results/o8_b5_stack.txt`): keep the frozen `B⁴` `v2` start latent
  (`lat512_b4_v2_n0.80.npy`) and scoring weights unchanged, while upgrading the joint semantic ensemble with multi-scale
  temporal-pyramid `FlexBrainEncoder` models (`train_ms_flex.py`, `ord_b5_flex/pooled_p1` 1-pass shared head) combined strictly
  via 5-fold concept-disjoint training OOF (`oof_blend_select.py`, `w_b5 = (0.65, 0.10, 0.25)` across `b5_flex`, legacy
  `ms_e10_ft10in`, and single-subject `O8`). Auxiliary contour/H14 encoders (`spat_ms512.py`, `spat_ms512_b5`, `h14_ms_b5`) were
  trained in `aw_tta.py`, while the final adopted `v2`-start evaluation keeps `B⁴`'s `spat_ms_mwtta` contour condition so start
  latents and candidate scoring stay fixed. Across all 10 subjects, **B⁵-1a** (`nsel = 1`, indep `K = 4`, no blur) reaches
  `0.3117 / 0.4259 / 0.9048 / 0.9334 / 0.7966 / 0.8399` (`EffNet 0.8378`, `CLIP +0.0086` over `B⁴-1a` in `10/10` subjects),
  **B⁵a** (best of 16, indep `K = 4`) reaches `0.3211 / 0.4294 / 0.9092 / 0.9375 / 0.8029 / 0.8450` (`EffNet 0.8363`,
  `CLIP top-1 0.1862`; `CLIP +0.0070` over `B⁴a` in `10/10` subjects), and **B⁵** (`+ K = 4 tmf`, no blur) reaches
  `0.3247 / 0.4306 / 0.9101 / 0.9377 / 0.8017 / 0.8449` (`EffNet 0.8370`, `CLIP top-1 0.1884`, `[6/6]` over published SOTA
  maxima), trading `−0.0067 / −0.0072` PixCorr and `−0.0011 / −0.0015` SSIM relative to `B⁴a / B⁴` for higher semantic alignment
  and lower `EffNet`. The non-adopted `R = 4`-gated `w5` start ablation (`B5-1a#w5start`) gave `0.3001 / 0.4247 / 0.8957 / 0.9288 /
  0.7955 / 0.8352 / 0.8418`.

## Components

1. **Joint encoder** (`code/train_ms.py`, tag `ms_e10_ft10in`). Brain-HIVE's BrainMLP with one input layer
   (Linear 4250 → 1024) per subject and the residual trunk + LayerNorm shared by the ten subjects, trained with the
   Brain-HIVE recipe (symmetric InfoNCE to the frozen Fusion-Prior target F, AdamW 5e-4, wd 0.05, batch 1024,
   bf16) for 10 epochs; then 10 epochs fine-tuning only each subject's input layer (lr 5e-4). Every batch holds
   each image at most once. Uses the *training* EEG of the other nine subjects; no test EEG of any subject is used.
   200-way test retrieval (R80): top-1 0.452 → 0.513, top-5 0.789 → 0.831 (10-subject means).
2. **Ordinary head** (`code/ordh.py`). The co-author's network and recipe (noise-prediction MSE + 0.1 token cosine,
   AdamW 1e-4, batch 16, 3 passes, seed 1729), trained on out-of-fold conditions (5 concept folds, each with its own
   encoder's LayerNorm affine). Used as in the original: **one head shared by the ten subjects** (`ordh.py train
   all 1`, one pass over the 165,310 subject-image pairs; each sample keeps the affine of its own subject's fold
   encoder). One head per subject (`ordh.py train <sub>`, 3 passes) was run as well; see the comparison above.
3. **Low-level start image** (`code/low.py`). Closed-form ridge from the z-scored EEG (17 × 250) to the published
   SDXL-VAE latents of the 128-px image, with a repeat correction (signal covariance = PSD part of the 4-repeat second
   moment minus noise/4, plus noise/80) and the ridge weight chosen by 5-fold concept cross-validation; the deviation
   from the mean latent is rescaled so its RMS over the test batch is 0.8 × that of real training latents
   (label-free, uses test-batch statistics). Decoded by the VAE, upsampled to 512 px, used as SDXL-Turbo img2img
   start with strength 0.75 (3 of 4 steps).
3b. **Retrieval-blended start image** (A′, B′; `lowsem.py ms_e10_ft10in vae_rcp --space raw --save lowrk0.5 --mode
   knn --w 0.5`, then `decode_low.py lowrk0.5`). The ridge latent prediction is blended with the softmax-weighted
   (temperature 0.05) mean latent of the 20 *training* images whose fused target F is closest to the joint-encoder
   condition (centred cosine; training concepts are disjoint from the test concepts), both scaled to unit SD,
   weight 0.5, then decoded with the same spread normalisation. The weight was picked among 0.25 / 0.5 / 1 on the
   test-latent correlation and the image metrics of the ten subjects (a test-set choice).
4. **Condition amplification** (`gen.py --amp`): c' = |c| · normalize(m + a (c/|c| − m)), m = mean direction of the
   200 test conditions (label-free, transductive).
5. **Best of N** (`gen.py --nsel 4`, 8 in A″ / B″, 16 in B‴): for every output image, 4 candidates with different noise; each is embedded back
   to F (`code/fenc.py`) and the one with the highest centred cosine to the EEG condition is kept. No label is used,
   but the sampling budget is 4× that of a single draw (16 generations per query instead of 4).

6. **High-frequency shrinkage** (A″, B″; `gen.py --post 0,16,0.8`): every generated image x is replaced by
   G(x) + 0.8 · (x − G(x)), G = Gaussian blur with σ = 16 px at 512 px. Rationale: the EEG carries little information
   about fine image detail, so the detail the generator invents is mostly uncorrelated with the true image; shrinking
   it towards zero is the lower-error estimate (a Wiener-style gain below one), the same argument that makes posterior
   means blurrier than samples. No label is used and nothing is re-generated (`--post` scores any number of settings on
   the same images; `--dump 1` + `postoff.py` do it offline).

## Protocol details

- Subjects 1–10; encoder seed 2025 (three seeds for subjects 1, 3, 8 in the earlier per-subject experiments:
  seed SD 0.001–0.007); head seed 1729; one generation seed set, fixed per (query, sample) and shared by all arms.
- Training inputs: mean of the 4 repeats; test inputs: mean of the 80 repeats. 17 occipito-parietal channels, 0–1 s.
- Concept folds: fixed permutation of the 1654 training concepts (`common.concept_folds`, seed 20261112).
- Hyper-parameters were chosen on subsets of subjects (amplification on 1, 2, 4, 8; start-image settings on 8) and
  then applied to all ten; selection on the test set is therefore not fully excluded for those two choices.
- External methods quoted for context (numbers from their papers, different evaluation details, not re-run here):
  Perceptogram 0.267 / 0.347 / 0.910 / 0.927 / 0.752 / 0.807; ENIGMA 0.172 / 0.423 / 0.836 / 0.895 / 0.777 / 0.815;
  CogCapPro 0.163 / 0.398 / – / – / 0.779 / 0.830 (PixCorr / SSIM / Alex2 / Alex5 / Incep / CLIP).

## Negative and neutral results

Image-level unless stated; "4 subjects" = mean over subjects 1, 2, 4, 8 relative to the arm they modify.

| idea | outcome |
|---|---|
| IP classifier-free guidance, scale 1.5 / 2 / 3 (negative = mean condition) | PixCorr −0.018 / −0.035 / −0.066, SSIM −0.026 / −0.047 / −0.084, CLIP +0.003 / −0.004 / −0.026 (4 subjects) |
| larger IP-attention weight 1.5 / 2 | CLIP −0.009 / −0.036 (4 subjects) |
| text prompt from the nearest *training* concepts (k = 1 / 3), or "a photo" | CLIP −0.023 / −0.016 / −0.010 (subject 8), similar on subject 1 |
| closed-loop correction (embed images back to F, push the condition along the missing part), λ 0.5 / 1 | CLIP −0.018 / −0.059 (subject 8), −0.032 / −0.085 (subject 5) |
| img2img strength 0.5 instead of 0.75 | SSIM +0.03 to +0.04, CLIP −0.02 to −0.03 (subjects 1, 5, 8): a different operating point, not a gain |
| start-image contrast 1.0 instead of 0.8 | subject 8: SSIM +0.007, PixCorr +0.008, CLIP −0.003, Incep −0.009 |
| joint encoder: 25 / 50 epochs, 2 trunk layers, lr 1e-3 / 2.5e-4, dropout 0.2 / 0.3 | retrieval not better than the chosen setting (25 epochs overfits: top-1 0.471) |
| joint encoder with repeat-subset augmentation | retrieval top-1 0.543, but images worse (subject 8 Incep 0.795 vs 0.809) |
| averaging two encoder seeds | +1.4 points top-1 without fine-tuning, none with it |
| test conditions at the training repeat level (mean of 20 embeddings of 4-repeat inputs, rescaled to the R80 norm) | retrieval top-1 after the head 0.505 → 0.533, but images: SSIM +0.008 (10/10), Incep −0.005, PixCorr −0.003 |
| bagging the five fold encoders with the full encoder | retrieval 0.509 vs 0.513 |
| best-of-4 score + agreement with the start image (pixel correlation, weight 0.5 / 1) | PixCorr +0.005, SSIM +0.002, Incep −0.004 / −0.006 (9 subjects) |
| joint neural low-level decoder (`low_ms.py`; per-subject input layer, shared trunk) instead of the repeat-corrected ridge | higher test-latent correlation (0.288 vs 0.255 in standardised space), but images: SSIM −0.021 to −0.024 (0/10), PixCorr −0.007 to −0.015, Alex2 +0.002 to +0.006; same with the loss in raw latent space |
| decoder + latents of the training images retrieved with the EEG condition (`lowsem.py`, weight 0.5) | PixCorr +0.005, Alex2 +0.012 (10/10), SSIM −0.025 (0/10) |
| fine-tuning the IP projection P on head-corrected EEG conditions (`ipft.py`, lr 1e-5 / 3e-5, 3,000 steps, anchor on clean F) | collapse to generic images: CLIP 0.817 → 0.60, Incep 0.78 → 0.59, SSIM +0.084 (blur) |
| retrieval blend keeping only the coarse part of the retrieved latents (8×8 pooling), weight 0.5 / 1 | SSIM −0.009 / −0.014 (worse than the plain blend), PixCorr / Alex2 similar |
| averaging the head-corrected conditions of the old per-subject and the new joint pipeline | retrieval top-1 0.507 → 0.493–0.507 |
| (earlier, per-subject encoder) cross-fitted ridge prior c → F, kNN snapping to the training gallery | lower discriminability, no image gain |
| (earlier) seed / time-window condition ensembles, repeat-subset training, absolute cosine loss beyond calibration | no image gain |
| (earlier) degraded-image targets (NeuroBridge settings) | retrieval 50.5 → 63.5 % in the degraded space, images no better |
| (earlier) 8 sampler steps, pixel-space start images, pixel blending, best of 8 vs best of 4 (per-subject encoder) | no gain / trades CLIP for SSIM |
| shared head trained for the original 3 passes (30,996 updates) instead of 1 | A / B / A′ / B′: every metric within ±0.004 of the 1-pass head, mixed signs (10, 10, 10, 9 subjects) |
| retrieval weight 0.25 instead of 0.5 in the start-image blend | in between A and A′ on every metric (PixCorr 0.2630, SSIM 0.3905) |
| 8 sampler steps from the start image, strength 0.75 / 0.625 (6 / 5 refinement steps instead of 3), single draw | 0.75: SSIM −0.019, Alex2 −0.011, CLIP −0.003 (0–1/10); 0.625: PixCorr +0.008, SSIM +0.006, Alex2 −0.007, CLIP −0.005 (0/10) |
| best-of-8 selection target: head-corrected condition (centred on the test mean) instead of the raw EEG direction | CLIP +0.0017 / +0.0007 (A′ / B′), SSIM +0.001 (10/10, 9/10), others ±0.001: within noise, not adopted |
| best-of-8 selection target: kNN prototype of the training-image embeddings (top 20, T = 0.05), alone or mixed | CLIP +0.0006 / −0.0009, Alex5 −0.001 to −0.002, Incep −0.002 |
| shrinkage grid, other settings (A′ / B′ best of 8, 10 subjects): gain 0.8 at σ 2 / 4 / 8 px; gain 0.7 / 0.6 at 8 px | smaller σ helps SSIM less at a similar or larger CLIP cost; gain 0.6 at 8 px: SSIM +0.032, PixCorr +0.014, CLIP −0.007 / −0.006, Alex2 −0.010 / −0.009 |
| replacing the low-frequency band (σ 8 / 16 px) by that of the start image, weight 0.25 / 0.5, with or without shrinkage | PixCorr +0.025 to +0.035, SSIM up to +0.018, but CLIP −0.006 to −0.019 |
| best-of-8 candidates alternating amplification 1 and 2 (one pool, the EEG score chooses) | between A′ and B′ on every metric (PixCorr 0.2646, SSIM 0.3850, Alex2 0.8882, CLIP 0.8273): no gain over picking one of them |
| best of 16, target = corrected condition | CLIP +0.0009, SSIM +0.0011 (10/10), Incep −0.0020: within noise |
| shrinkage at 24–48 px with gain 0.5–0.7 (B′ best of 16) | SSIM up to 0.437 (+0.055) and PixCorr up to 0.303, but CLIP −0.002 to −0.009 and Alex2 −0.008 to −0.022 |
| shrinking the high band towards the mean of the K = 4 samples of a query (full band or above σ 4 / 8 px) | PixCorr +0.003 to +0.027, SSIM +0.008 to +0.023, Incep +0.001 to +0.003, but CLIP −0.002 to −0.014 |

## Not finished (stopped 2026-11-13)

- Shrinking only the luminance high band (`postoff.py` spec `y,σ,β`) or only the band between 1 and 24 px (`b,1,24,β`):
  scored on B′ best of 16 for 4 subjects (5, 6, 8, 9) before the run was stopped. Band 1–24 px at 0.8: SSIM +0.015,
  PixCorr +0.013, CLIP ±0.000, Alex2 −0.003 (vs −0.004 for the full high band) — possibly a slightly better trade, not
  established on ten subjects.

## Running the original-components pipeline (O3 / O4 / O5 / O6 / O7a / O7)

Inputs, environment and step 1 (data) as in the next section; `b32fix.py`, `fenc.py`, the joint encoder, the shared head
and `rel.py` are not needed. From `code/` (`XX` = 01 … 10):

```bash
# per-subject Brain-HIVE encoders (official recipe) and their concept-fold versions (for the retrieval blend)
for s in $(seq 1 10); do for f in -1 0 1 2 3 4; do python train.py --sub $s --fold $f --tag t0_r4; done; done
# the co-author's head, one per subject
python ordh.py latents
for s in $(seq 1 10); do python ordh.py train $s; done
# O0: repeat-corrected ridge start images (the O0 row; O1-O3 recompute the ridge on the aggregated input)
for s in $(seq 1 10); do python low.py $s; done
# O2: inverse-variance aggregation -> conditions through each subject's head, ridge latents
python ivwprep.py t0_r4
# O1: retrieval blend on the aggregated inputs, decoded to start images
python lowsem.py t0_r4 vae_rcp_ivw --ctest cond_test_ivw.npy --space raw --mode knn --w 0.5 --save lowrkivw_0.5
python decode_low.py lowrkivw_0.5
# O3: generation with fractional strength (nsel = 1, independent K = 4 draws per query)
python gen.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file ord_ivw/sub-XX.npy --init_file sub-XX/lowrkivw_0.5_n0.8.npy \
    --smap c0.611,0.5,1.0 --steps 4 --K 4 --bs 3 --save 0 --name sXX_t0_s2025_A1_ordivw_lowrkivw_0.5_smap_c0.611_0.5_1.0
# O4: split-half contrastive test-time adaptation, then the retrieval blend and the O3 generation with the new start
python tta.py t0_r4 1,2,3,4,5,6,7,8,9,10 50 1e-4 0.1 1 tta1     # writes cond_test_tta1.npy (and ord_tta1/, not used by O4)
python lowsem.py t0_r4 vae_rcp_ivw --ctest cond_test_tta1.npy --space raw --mode knn --w 0.5 --save lowrktta1_0.5
python decode_low.py lowrktta1_0.5
python gen.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file ord_ivw/sub-XX.npy --init_file sub-XX/lowrktta1_0.5_n0.8.npy \
    --smap c0.611,0.5,1.0 --steps 4 --K 4 --bs 3 --save 0 --name sXX_t0_s2025_A1_ordivw_lowrktta1_0.5_smap_c0.611_0.5_1.0
# O5: six more encoder seeds, the ensemble-and-bagging condition, generation with the O4 start image
for s in $(seq 1 10); do for k in 2028 2029 2030 2031 2032 2033; do python train.py --sub $s --seed $k --tag t0_r4; done; done
python condscreen.py 1,2,3,4,5,6,7,8,9,10 base,ens7bag10 1          # writes runs/cond/ord_ens7bag10/
python gen.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file ord_ens7bag10/sub-XX.npy --init_file sub-XX/lowrktta1_0.5_n0.8.npy \
    --smap c0.611,0.5,1.0 --steps 4 --K 4 --bs 3 --save 0 --name sXX_t0_s2025_A1_ordens7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0
# O6: encoders on the 0-150 and 0-100 sample windows (7 seeds each), the multi-window condition, generation as O5
for s in $(seq 1 10); do for w in 150 100; do for k in 2025 2028 2029 2030 2031 2032 2033; do
    python train.py --sub $s --seed $k --t0 0 --t1 $w --tag t0_r4_w$w; done; done; done
python condscreen.py 1,2,3,4,5,6,7,8,9,10 base,mw7bag10 1           # writes runs/cond/ord_mw7bag10/
python gen.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file ord_mw7bag10/sub-XX.npy --init_file sub-XX/lowrktta1_0.5_n0.8.npy \
    --smap c0.611,0.5,1.0 --steps 4 --K 4 --bs 3 --save 0 --name sXX_t0_s2025_A1_ordmw7bag10_lowrktta1_0.5_smap_c0.611_0.5_1.0
# O7a (strict w=1.00, independent K=4 draws) and O7 (w=1.50 + K=4 tmf):
for s in $(seq 1 10); do for w in 200 175 125; do for k in 2025 2028 2029 2030 2031 2032 2033; do
    python train.py --sub $s --seed $k --t0 0 --t1 $w --tag t0_r4_w$w; done; done; done
for s in $(seq 1 10); do ORD_ENS=mw7 python ordh.py train $s; done
python aw_tta.py all                                                # writes runs/cond/cone_mw6tta_w{1.00,1.45,1.50}/sub-XX.npy
python lat512_start.py all --cond_dirs cone_mw6tta_w1.00 --tag lat512_o7_v2
# O7a (headline strict unborrowed arm: w=1.00, no --tmf, independent K=4 draws):
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_mw6tta_w1.00/sub-XX.npy \
    --lat_init sub-XX/lat512_o7_v2_n0.80.npy --smap c0.625,0.5,1.0 --low_keep 8,0.22,0.04 \
    --K 4 --save 0 --name sXX_w4_O7_w1_notmf
# O7 (with w=1.50 directional cone amplification and --tmf K=4 Tweedie coupling; omit --tmf for O7c, or set w=1.00 + --tmf for O7b):
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_mw6tta_w1.50/sub-XX.npy \
    --lat_init sub-XX/lat512_o7_v2_n0.80.npy --smap c0.625,0.5,1.0 --low_keep 8,0.22,0.04 \
    --tmf 1,0.10,0.10,0 --tmf_cr 1 --K 4 --save 0 --name sXX_w35_O7_v2b
python tab10.py > table10.txt          # rows O1 ... O7a / O7
```

Analyses of this section: `robagg.py` (aggregators), `rel.py` (split-half reliability; its embedding part uses the joint
encoder tag by default, pass `t0_r4` for the per-subject one), `permet.py` + `adapt.py` / `adapt_map.py` (per-image gain
policies, needs `gen.py --dump 1`), `armsum.py` (paired means over subjects); outputs in `results/original_pipeline.txt`,
`results/reliability_gain.txt`, `results/robust_aggregation.txt`, `results/test_time_adaptation.txt` (`tta.py`, all four settings), `results/semantic_condition.txt` (`condscreen.py`, the O5 and O6 arms). Generator-side screens after O6 (MBR selection, confidence-gated strength): `results/generator_side.txt`. Cost: about 3 min of generation per subject on one RTX 4090.
The last command, re-run for subject 8 with the final version of `gen.py`, reproduced all stored metrics exactly (largest
difference 0.0).

## Running it (reproducing B‴, B⁴-1a..B⁴, O8/O8b, and B⁵-1a..B⁵ from scratch)

**Inputs** (downloaded once; nothing here is redistributed). Put them under one working directory `$RAD_ROOT`:

| path under `$RAD_ROOT` | source |
|---|---|
| `data/things-eeg/Preprocessed_data_250Hz_whiten/`, `data/things-eeg/Image_set/` | Hugging Face dataset `Haitao999/things-eeg` (THINGS-EEG2, Gifford et al. 2022) |
| `data/hive_embeddings/things-eeg/` (parquet) | Hugging Face dataset `fakekungfu/Brain-HIVE_Visual_Embeddings` |
| `models/Fusion-Prior-H14_B32_VAE/` | Hugging Face `fakekungfu/Fusion-Prior-H14_B32_VAE` |
| `models/sdxl-turbo/` | Hugging Face `stabilityai/sdxl-turbo` (fp16 variant) |
| `models/CLIP-ViT-H-14-laion2B-s32B-b79K/`, `models/CLIP-ViT-B-32-laion2B-s34B-b79K/` | Hugging Face `laion/...` (same names) |
| `code/Brain-HIVE/` | github.com/ssshamiii/Brain-HIVE at commit f06db69 |
| metric weights | torchvision AlexNet / Inception-v3 / EfficientNet-B1 and OpenAI CLIP ViT-L/14 (`pip install git+https://github.com/openai/CLIP`), fetched on first use |

**Environment used for every number here**: Python 3.10.12, torch 2.14.1 (CUDA 13.0), torchvision 0.29.2, diffusers 0.41.0,
transformers 5.20.2, numpy 2.2.6, scikit-image 0.25.2, clip 1.0; one RTX 4090 (24 GB) per job (Wave 5 `O8 / B⁵` generation runs also used NVIDIA L4 / T4 GPUs alongside RTX 4090 under the same deterministic per-(query, sample, candidate) seeds).

**Steps** (from this folder's `code/`, `export RAD_ROOT=...`; `XX` = 01 … 10):

```bash
# 1. data
python prep.py eeg 1 2 3 4 5 6 7 8 9 10        # 17-channel arrays -> data/prep/sub-XX_{train,test}.npy, *_ids.json
python prep.py targets                         # emb_{train,test}.pt, F_{train,test}.npy, train_concept.npy
python b32fix.py                               # data/prep/b32_fix.pt (candidate embedder correction, training images only)
python fenc.py                                 # optional check: embedder vs the published test embeddings
# 2. joint encoder: the full model and the five concept folds (out-of-fold conditions for the head and the start image)
for f in -1 0 1 2 3 4; do python train_ms.py --tag ms_e10_ft10in --epochs 10 --ft_epochs 10 --ft_trunk 0 --ft_lr 5e-4 --fold $f; done
# 3. the co-author's ordinary head, one shared head (1 pass), applied to each subject's test conditions
ORD_TAG=ms_e10_ft10in python ordh.py latents
ORD_TAG=ms_e10_ft10in ORD_HEAD=pooled_p1 python ordh.py train all 1
for s in $(seq 1 10); do ORD_TAG=ms_e10_ft10in ORD_HEAD=pooled_p1 python ordh.py apply $s; done
# 4. start images: ridge (vae_rcp), then the retrieval blend (lowrk0.5)
for s in $(seq 1 10); do python low.py $s; done
python lowsem.py ms_e10_ft10in vae_rcp --space raw --mode knn --w 0.5 --save lowrk0.5
python decode_low.py lowrk0.5                  # -> runs/low/sub-XX/lowrk0.5_n0.8.npy
# 5. B‴ for each subject: amplification 2, best of 16, shrinkage 24 px / 0.8 scored inside the run
python gen.py --run ms_e10_ft10in/sub-XX/fold-1_seed2025 --cond_file ord_ms_e10_ft10in_pooled_p1/sub-XX.npy \
    --init_file sub-XX/lowrk0.5_n0.8.npy --strength 0.75 --amp 2 --nsel 16 --K 4 --save 0 \
    --post 0,24,0.8 --name sXX_ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16
#    metrics.json: "mean" = images as generated (B′ best of 16), "post"["0,24,0.8"] = B‴
#    A″ instead: no --amp, --nsel 8, --post 0,16,0.8
# 5b. B⁴-1a / B⁴-1 (nsel = 1) and B⁴a / B⁴ (best of 16, no blur; requires the single-subject window encoders of O7a above):
for k in 2026 2027 2028; do python train_ms_win.py --tag ms_w250 --t0 0 --t1 250 --seed $k; done
for w in 200 175 150 125 100; do for k in 2025 2026 2027 2028; do python train_ms_win.py --tag ms_w$w --t0 0 --t1 $w --seed $k; done; done
for k in 2025 2026; do
    python train_ms_win.py --tag ms_w10_135 --t0 10 --t1 135 --seed $k
    python train_ms_win.py --tag ms_w10_185 --t0 10 --t1 185 --seed $k
done
python train_ms_spat.py --mode build                           # -> data/prep/spat1024.pt (training VAE latents only)
for k in 2025 2026; do
    python train_ms_spat.py --mode train --tag ms_spat_w10_70 --t0 10 --t1 70 --seed $k
    python train_ms_spat.py --mode train --tag ms_spat_w10_105 --t0 10 --t1 105 --seed $k
    python train_ms_spat.py --mode train --tag ms_spat_w0_150 --t0 0 --t1 150 --seed $k
    python train_ms_spat.py --mode train --tag ms_spat_w0_225 --t0 0 --t1 225 --seed $k
done
python aw_tta.py all                                           # -> cone_ms28_mwtta_w{1.00,1.75}/ and spat_ms_mwtta/
python lat512_start.py all --cond_dirs cone_mw6tta_w1.00 --tag lat512_o7_v2 --also_b4 cone_ms28_mwtta_w1.00:spat_ms_mwtta:lat512_b4_v2
# B⁴-1a (nsel=1, strictly independent K=4 draws) and B⁴-1 (+ --tmf K=4 coupling):
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_ms28_mwtta_w1.75/sub-XX.npy \
    --lat_init sub-XX/lat512_b4_v2_n0.80.npy --smap c0.628,0.5,1.0 --low_keep 8,0.22,0.04 \
    --K 4 --save 0 --name sXX_w4_B4_n1_notmf
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_ms28_mwtta_w1.75/sub-XX.npy \
    --lat_init sub-XX/lat512_b4_v2_n0.80.npy --smap c0.628,0.5,1.0 --low_keep 8,0.22,0.04 \
    --tmf 1,0.12,0.12,0 --tmf_cr 1 --K 4 --save 0 --name sXX_w35_B4_v2b_n1
# B⁴a (best of 16, strictly independent K=4 draws) and B⁴ (+ --tmf K=4 coupling):
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_ms28_mwtta_w1.75/sub-XX.npy \
    --sel_cond cone_ms28_mwtta_w1.00/sub-XX.npy --spat_cond spat_ms_mwtta/sub-XX.npy \
    --lat_init sub-XX/lat512_b4_v2_n0.80.npy --smap c0.628,0.5,1.0 --low_keep 8,0.22,0.04 \
    --nsel 16 --sel_spat 0.28 --sel_vae 0.24 --sel_grad 0.22 --K 4 --save 0 --name sXX_w4_B4_sel16_notmf
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_ms28_mwtta_w1.75/sub-XX.npy \
    --sel_cond cone_ms28_mwtta_w1.00/sub-XX.npy --spat_cond spat_ms_mwtta/sub-XX.npy \
    --lat_init sub-XX/lat512_b4_v2_n0.80.npy --smap c0.628,0.5,1.0 --low_keep 8,0.22,0.04 \
    --tmf 1,0.12,0.12,0 --tmf_cr 1 --nsel 16 --sel_spat 0.28 --sel_vae 0.24 --sel_grad 0.22 \
    --selvar "0.20,0.20,0.18@16;0.35,0.28,0.25@16;0.45,0.35,0.30@16" --K 4 --save 0 --name sXX_w35_B4_v2b_sel16
# 5c. Wave 5 (O8 / O8b single-subject and B⁵-1a / B⁵-1 / B⁵a / B⁵ full-stack):
python spat_ms512.py                                            # -> data/prep/spat_ms512.pt (training VAE latents only)
# Train FlexBrainEncoder (train_flex.py -> o8_{pyr2,mlp2,mlp1}_* across 5 folds & full train) and joint Flex (train_ms_flex.py -> msf_*)
for s in $(seq 1 10); do ORD_ENS=o8_flex python ordh.py train $s; done
ORD_ENS=b5_flex ORD_HEAD=pooled_p1 python ordh.py train all 1
python oof_blend_select.py                                      # -> runs/cond/o8_b5_oof_weights.json (5-fold training OOF only)
python aw_tta.py all                                            # -> cone_o8_w1.00/ and cone_b5_w{1.00,1.75}/
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_o8_w1.00/sub-XX.npy \
    --lat_init sub-XX/lat512_o7_v2_n0.80.npy --smap c0.625,0.5,1.0 --low_keep 8,0.22,0.04 \
    --K 4 --save 0 --name sXX_o8_w1_notmf                         # add --tmf 1,0.10,0.10,0 --tmf_cr 1 for O8b
python gen_exp.py --run t0_r4/sub-XX/fold-1_seed2025 --cond_file cone_b5_w1.75/sub-XX.npy \
    --sel_cond cone_b5_w1.00/sub-XX.npy --spat_cond spat_ms_mwtta/sub-XX.npy \
    --lat_init sub-XX/lat512_b4_v2_n0.80.npy --smap c0.628,0.5,1.0 --low_keep 8,0.22,0.04 \
    --tmf 1,0.12,0.12,0 --tmf_cr 1 --nsel 16 --sel_spat 0.28 --sel_vae 0.24 --sel_grad 0.22 --K 4 --save 0 --name sXX_b5_sel16_tmf
# 5d. Wave 6 (O9-0tta / O9 / O9b streamlined 8-checkpoint pipeline):
python kv_metric.py                                              # -> data/prep/unet_kv_metric.pt (70-layer UNet L_kv)
python o9_streamlined.py --subs all --mode all                   # trains 7 seeds KVEBPyrEncoder, 1 DDPM head, kvTTA, v2 start, O9-0tta / O9 / O9b
# 5e. Wave 7 (O10 bifurcated forward-operator synthesis, 0 DDPM heads, deterministic k=0 gallery):
python o10_factorized.py --subs all --mode n1                    # builds F_sem_train.pt, trains Stream 2 (o10_fsem_fnm), PhysReSyn, O10-sem-*
python o10_factorized.py --subs all --mode bifurc                # trains Stream 1 (o10_ffull_zero00), eta=0.50 bifurcation, nsel=1,2,4 eval
python build_o10_report_and_gallery.py --out_dir .               # replays O9 vs O10 (max_abs_diff < 1e-4), renders gallery/, writes results/o10_factorized.txt
# 6. tables
python agg.py                                                   # -> runs/enc/summary.csv (results/encoders_summary.csv)
python tab10.py > table10.txt                                   # every arm including O8 / O8b / O9-0tta / O9 / O9b / B⁵-1a..B⁵
python build_o8_b5_report.py > o8_b5_stack.txt
python build_o9_report.py > o9_streamlined.txt
python runsum.py > runs_summary.csv
python svsum.py _ms_e10_ft10in_A1_pooled_p1_lowrk0.5_amp2_sel16 main   # all --post / --selvar variants of a run
```

Cost on one RTX 4090: joint encoder a few minutes per fold run (all ten subjects), shared head ≈ 2.4 h (10,332 updates),
start images ≈ 2 min per subject, B‴ generation ≈ 40 min per subject (64 SDXL-Turbo images per query plus scoring).

**Other arms.** Every generation run in the tables, including the negative ones, is one row of
`results/runs_summary.csv` with all its `gen.py` arguments (encoder run, condition file, start image, strength, steps,
amplification, best-of-N, …); `#post:` / `#selvar:` rows are variants scored inside that run. The per-subject
baseline encoders (`t0_r4`) come from `train.py`, the per-subject heads from `ordh.py train <sub>`.

**How close a re-run should come.** Generation noise is seeded per (query, sample, candidate), so step 5 is repeatable
given the same conditions and start images. Checked on 2026-11-13: a fresh clone of this repository (commit 4e5c80d)
re-ran step 5 for subject 8 in both A″ (best of 8, shrinkage 16 px) and B‴ (best of 16, shrinkage 24 px) on the stored
conditions and start images, and reproduced all seven stored metrics, before and after shrinkage, exactly (largest
difference 0.0; same GPU type and environment). `b32fix.py` from the clone rebuilt `b32_fix.pt` bit-identically. Steps 2–3 train in bf16 on GPU and are not bit-exact; encoder seeds changed the per-subject image metrics
by 0.001–0.007 (SD over three seeds, subjects 1, 3, 8), so ten-subject means should agree to about ±0.003.

**Folder layout.** `code/` every script used for the numbers here (`cmp.py`: paired two-arm comparison, `svsum.py`:
all variants of one run, `o10_factorized.py` & `build_o10_report_and_gallery.py`: O10 training, bifurcation, replay & gallery);
`jobs/` the shell wrappers that queued the experiments and the download lists (not needed for the steps above; see
`jobs/README.md`); `results/` the summary tables (including `o10_factorized.txt`); `gallery/` the deterministic (`k = 0`)
10-subject `O9` vs `O10` comparison sheets (`readme_overview_o9_vs_o10.jpg`, `gallery_sub01_02.jpg`..`gallery_sub09_10.jpg`,
and `GALLERY.md`, including attributed 224 px CC-BY 4.0 THINGS-EEG2 reference thumbnails). Generated intermediate files
(EEG arrays, embeddings, start images, model weights, raw test renders, per-run metrics) are not included: each is rebuilt by the steps above.

Downloaded `.pt` EEG files are numpy pickles; they are loaded only through the restricted unpickler in
`common.safe_load_pt`. No raw EEG, feature embedding, or model weight file is included here.
