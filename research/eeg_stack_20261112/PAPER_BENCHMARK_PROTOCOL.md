# THINGS-EEG2 paired reconstruction benchmark

The primary outcome is the object depicted by the generated image. Each method in the main comparison receives one person's EEG, a blank text prompt and a seed determined by the registered row number. The evaluator has the photographs and the THINGS dictionary. The same evaluator also scores the original photographs, providing a reference for its naming accuracy. [Benchmark report](../../docs/PAPER_BENCHMARK_RESULTS.md) · [Machine-readable results](results/paper_benchmark.json)

## Recompute the tables

Run the following commands from the repository root with Python 3.11+, NumPy and Pillow. The score archive contains per-stimulus observations and source checksums. Recomputing the tables runs on CPU.

```sh
python research/eeg_stack_20261112/code/package_paper_scores.py --unpack research/eeg_stack_20261112/results/paper_benchmark_scores.zip outputs/paper-benchmark/verified
python research/eeg_stack_20261112/code/build_paper_benchmark_report.py --root outputs/paper-benchmark/verified
```

For a specific comparison, the command below reports participant estimates and paired uncertainty across participants and stimuli. On the multi-photo panel it resamples whole concepts with their ten photographs. Intervals describe one prespecified sample per EEG query and a fixed retrieval gallery.

```sh
python research/eeg_stack_20261112/code/paper_aggregate.py --root outputs/paper-benchmark/verified --training personal --budget official-r4 --arms native-paper d2 wrong --pair d2 native-paper --pair d2 wrong
```

## Data and test inputs

The public-data recipes reproduce the 17-channel EEG arrays, row identities, frozen visual gallery and photo manifests from their pinned releases. The handoff verifies the individual files before installing the benchmark inputs. [Public input and verification guide](public_inputs/README.md)

| Panel | Stimuli per participant | EEG at inference | Primary image endpoint |
| --- | ---: | --- | --- |
| Official R4 | 200 distinct test objects, one photo each | Presentations 0, 1, 2, 3 | Object naming against 1,854 THINGS concepts |
| Official R4, session-balanced | The same 200 photos | Presentations 0, 20, 40, 60 | Sensitivity to the four acquisition sessions |
| Official R80 | The same 200 photos | All 80 presentations | Object naming and published reconstruction metrics |
| Fresh R4 | 50 held-out concepts × 10 photos | All four presentations of each photo | Identify the exact photograph within its object |
| Development R4 | 20 further held-out concepts × 10 photos | All four presentations, sub-01 and sub-03 | Model selection on separate concepts |

The restricted training split contains 1,323 concepts and 13,230 photographs. All the photographs of each concept are assigned together across participants. Fresh and development use different objects and both exclude every possible member of the historical 50-object image-generation gallery. An earlier EEG-feature experiment compared 12 readouts, including M1a, on 327 of the same 331 held-out concepts. The archived metrics imply that 46–50 fresh concepts and 16–20 development concepts participated in that earlier feature-level study. These panels provide new image-generation outcomes on training-disjoint concepts. The official 200 objects are disjoint from all 1,654 THINGS training concepts. The full Brain-HiVE reference uses all 1,654 concepts and 16,540 photographs and is evaluated on the official test. Shared training uses ten people's training examples. Every test query comes from a single person. [Historical feature comparison](TRIAL_PREC_DINOV2_REPORT.md) · [Split construction](code/prepare_paper_benchmark.py) · [Dataset paper](https://doi.org/10.1016/j.neuroimage.2022.119754)

The released EEG has 250 stored samples per trial. Both methods use the same 17 distinct posterior electrodes, including POz and Oz. The source has 63 channels. The Brain-HiVE reconstruction launch example lists Oz twice, while its intra-subject training launcher lists both electrodes. We use the 17 distinct channels throughout. The waveform, participant, stimulus, physical repeats and session order are preserved. [EEG release](https://huggingface.co/datasets/Haitao999/things-eeg/tree/944f0d50e37e125eb6b7b98d14a41571c9ed0ef7/Preprocessed_data_250Hz_whiten) · [Author training script](https://github.com/ssshamiii/Brain-HIVE/blob/4201b820a37bfbac2b298694f62e928dbac2b7e9/scripts/train_clip_intra.sh) · [Official photographs](https://osf.io/y63gw/)

## Models and scoring

- **Brain-HiVE.** Retrain the reconstruction encoder on each person's EEG for 25 epochs using the released frozen image teacher. The `native-paper` arm sends the encoder output through the fusion adapter and then the projection. The `native-code` arm uses the projection directly. Both use the same fitted EEG weights and the same released attention adapter. The fused reconstruction reference uses ViT-H/14, ViT-B/32 and VAE targets. The paper's dedicated retrieval training uses a different encoder combination including RN50. [Author implementation](https://github.com/ssshamiii/Brain-HIVE/tree/4201b820a37bfbac2b298694f62e928dbac2b7e9) · [Fusion Prior](https://huggingface.co/fakekungfu/Fusion-Prior-H14_B32_VAE/tree/135a284109e71f5cde3d3e6d5dee6268a3c7987a) · [Visual embeddings](https://huggingface.co/datasets/fakekungfu/Brain-HIVE_Visual_Embeddings/tree/4efe61ddfde96eea962c391ec01736d42d9b60ef/things-eeg)
- **M1a + D2.** Combine the encoder output of a channel-weighted four-repeat ERP with the mean of four individual-trial encoder outputs. Predict the initial layout by a participant-specific multi-window ridge model fitted on training photographs. The semantic condition uses training-only cone calibration and a learned residual in the generator's image tokens. D2 learns to match images generated under four same-object EEG conditions to their corresponding training photos. R80 averages twenty nonoverlapping R4 predictions before a single draw. [Training stages](code/paper_training.py) · [Repeat readout and rendering](code/paper_render.py)
- **Controls.** A matched representation-only model, a correctly paired denoising-only control, a denoising exchange model, a positive-only image loss and D2 use the same registered shared training split. The wrong-EEG control exchanges both the semantic condition and the layout and retains the recipient's seed. On the fresh panel its donor is another photograph of the same object. The development panel also crosses the semantic source with the presence of a predicted layout. [Training implementation](code/paper_training.py) · [Sampling implementation](code/paper_render.py)
- **Generation.** Internally sampled arms share SDXL-Turbo, the four timesteps, one draw per query, 512-pixel outputs, an empty text prompt and a repeatable seed derived from panel and stimulus position. Each pair uses exactly the same seed. The external ATM release uses its own prior, native channel configuration and sampling seeds. Its first published candidate, selected by filename for every query, is scored against the same official RGB. [ATM released draws](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/blob/dbe34bb2407164f70883c661c76664f0d596e522/generated_imgs.tar.gz)
- **Outcomes.** Naming uses frozen CLIP ViT-L/14 and all 1,854 concept IDs. Duplicate display words share top-rank credit. Image retrieval searches all 200 or 500 original photographs. The separately reported within-object measure searches ten originals following centering across the gallery. PixCorr, SSIM, AlexNet layers 2/5, Inception, CLIP two-way, EfficientNet and SwAV cover pixel structure and feature matching. Continuous EEG feature retrieval queries a gallery of fused image embeddings before diffusion. [RGB scorer](code/paper_score.py) · [EEG feature scorer](code/paper_retrieval.py) · [Published baseline numbers](results/paper_benchmark_external.md)

## Object-first development after the primary comparison

The compact candidate receives one person's posterior EEG. It averages the physical presentations, applies one BrainMLP, the frozen Brain-HiVE fusion adapter and projection, and generates one four-step SDXL-Turbo image with empty text and pure initial noise. No test vocabulary, reference photo, support gallery or cross-query statistics enters this computation. All three training controls start from the same personal BrainMLP, receive the same training photographs and concept groups, and have identical parameters and update counts. The original native fit uses photo-level supervision.

For training photograph $i$, let $T_i$ be its four frozen SDXL image-conditioning tokens. For each token $k$, the revised target is

$$Y_{ik}=\bar T_{c(i),-i,k}+g_k(T_{ik}-\bar T_{c(i),-i,k}).$$

The mean uses the nine other training photos of the same object. The photo control fixes $g=1$, the object-only control fixes $g=0$, and the reliability arm sets each $g_k$ to the square root of nonnegative out-of-fold explained variance. Three whole-concept folds estimate how well a regularized regression from within-concept EEG deviations predicts the corresponding visual token residual, using 128 prespecified token coordinates. The deployed model contains the single fitted EEG encoder. The object group and reliability calculation are training supervision. Object-positive training and prototype-based EEG alignment both have published precedents. The experiment tests the particular leave-one-photo-out target in the frozen generator's four-token space and its EEG-estimated residual. [Object-token training](code/paper_object_tokens.py) · [Single-query inference](code/paper_object_infer.py) · [SPSC](https://arxiv.org/html/2611.09380v1) · [ProtoEEG](https://arxiv.org/html/2610.33065v1)

The development screen uses sub-01 and sub-03, 20 objects and 200 photos per person. Each condition has one stimulus-paired draw. Before viewing the candidate images, the selection rule was fixed as more correct 1,854-way names than personal native-paper in each development participant and at least as many within-object photo hits pooled. Among qualifying configurations, prefer more correct names, then more within-object hits, then the simpler fixed target. A selected model proceeds to the eight participants outside this development cohort. Naming uncertainty comes from joint resampling of people and stimulus groups. A wrong-EEG control exchanges the object while retaining the query seed. The formal 368-condition comparison retains its original denominator.

For M1a + D2 the additional shared-training comparison holds the renderer and fitted layout fixed across three arms. The first retains the original D2. The second removes its symmetric four-way loss and preserves its 0.5 positive-similarity weight. The third substitutes a plain four-trial average in the semantic path without refitting D2. The historical D1 used 12 times the positive-similarity loss to approximate the early positive-gradient magnitude. [Training objective](code/exp6_draft_intra4.py) · [Paired component results](../../docs/PAPER_BENCHMARK_RESULTS.md)

### Mechanisms suggested by the historical logs

The follow-ups use the same sub-01 and sub-03 development photos, registered physical repeats, four denoising steps, one draw, raw RGB scoring and empty inference prompt. Each keeps one BrainMLP per participant. The development rule and eight other confirmation participants above also apply to each pair. The control has priority on a tie. This repeatedly used development panel measures screening performance. Any claim about a particular training change additionally requires comparison with its own matched control.

1. The earlier visual-layer probe favored CLIP ViT-H/14 block 12 over the final embedding for EEG retrieval. Here each matched arm has an extra training loss and the same auxiliary-head capacity. One uses the final layer, the other the mean of 256 patch tokens after block 12, compressed to the control dimension on training photographs. The head is discarded at inference. The historical temporary extractor did not preserve its CLS-token policy. The explicit patch-only choice defines this experiment. [Original per-seed retrieval](../../results/retrieval.json)
2. In the O9 and D2 results, better within-class matching or image layout did not reliably translate into correct depicted objects. The second pair trains on decoded four-step images and uses the same cross-object distractors. The positive target is either the query training photograph or another training photograph of the same object. DINOv2 supplies the training signal. The primary 1,854-name score uses a separate CLIP evaluator. Training receives the registered training concepts only.
3. The earlier open sub-03 and sub-04 results point to the distribution of the visual condition. A matched pair uses a fixed training-set shift or the same shift plus an out-of-fold scalar spread correction. Both start from the same native participant encoder. The fixed affine transformation is folded into its final LayerNorm weights and bias. Training concepts supply all calibration statistics. Each query still uses one encoder, the original frozen projector and one draw from noise. [Historical calibration results](../../docs/OFFICIAL_REPEAT_IMAGES.md)
4. O13 motivates supervision from other people's training EEG. For each participant a visual ridge teacher is fit and applied out of fold, holding out complete training concepts. The student's training target averages nine other people's predictions. Both arms use the same ten people's training data, parameter count, schedule, photo-token objective and peer loss. They differ only in whether the peer EEG came from the same photo or a different photo of the same training object. One participant supplies all test EEG. Cross-person supervision and object-positive pairing have prior art, so the hypothesis concerns transfer to the depicted object under this controlled difference. [Earlier O13 comparison](results/o12_o13_o14_info_ceiling_xbrain.txt)

## Earlier runs

The historical R80 O10 and O14 studies cover a different operating point. O10 combines two seven-seed encoder ensembles, uses split-half adaptation on test EEG and scored four independent draws per stimulus. Its generated-image Top-1 was 1,296/8,000, versus 941/8,000 for the archived personal Brain-HiVE reproduction. O14's text-free, shared-training version reached 1,477/8,000. The original historical R4 D2 result was 982/5,000 same-object ten-photo matches, using a concept-shared sampling seed and VAE-decoded references. These experiments retain their original sampling and evaluation protocols. [O10 original run](results/o10_factorized.txt) · [O14 original run](results/o12_o13_o14_info_ceiling_xbrain.txt) · [Earlier R4 records](results/exp8c_m1a_heldout6_and_all10_results.json)

## Replaying generation on prepared data

`RAD_ROOT` contains `code/Brain-HIVE`, `models`, `data/prep`, `fits` and the output `images` tree. `ARCHIVE` identifies the staged official RGB, training targets, model assets and scoring weights used by the preparation and evaluation scripts. The two Slurm examples source a site-local `environment` in the submission directory defining these paths and the Python executable `py`. Schedule training, rendering and scoring on compute nodes. [Data and generator preparation](code/prepare_paper_benchmark.py) · [Native R80 job](jobs/run_paper_native_full.sbatch) · [Continuous-feature job](jobs/run_paper_retrieval.sbatch)

For a participant with a prepared native and D2 checkpoint, run the two arms and wrong-EEG control as follows. The native checkpoint resides under `fits/personal-01/enc/native`. The D2 checkpoint resides under the same participant's `swap_diff/draft_intra4` directory.

```sh
export EEG_RUNS_ROOT="$RAD_ROOT/fits/personal-01"
python "$RAD_ROOT/code/paper_render.py" --subject 1 --arms native-paper d2 wrong --budgets official-r4
for arm in native-paper d2 wrong; do
    python "$RAD_ROOT/code/paper_score.py" "$RAD_ROOT/images/personal/$arm/official-r4/sub-01" official
done
```

### Compact-model training and one-query inference

The following commands use the prepared benchmark layout and the subject's original restricted native fit. Each development pair is trained independently. The RGB pair starts from the photo-token control. The spread study starts from the original native model. Prepare the shared H/14 visual targets once on a compute node. A development participant uses 200 test photos from 20 concepts and the same four presentations as the other arms.

```sh
code="$RAD_ROOT/code"
export EEG_RUNS_ROOT="$RAD_ROOT/fits/personal-01"
python "$code/paper_object_tokens.py" --subject 1
python "$code/paper_middle_teacher.py" --prepare
python "$code/paper_middle_teacher.py" --subject 1
python "$code/paper_rgb_object.py" --subject 1 --steps 32
python "$code/paper_native_spread.py" --subject 1
python "$code/paper_render.py" --subject 1 --budgets dev-r4 --arms token-final-aux token-middle-aux
for arm in token-final-aux token-middle-aux; do
    python "$code/paper_score.py" "$RAD_ROOT/images/personal/$arm/dev-r4/sub-01" dev
done
```

For an individual preprocessed query, supply a floating-point array with axes `presentations × 17 electrodes × 250 samples` in the pinned posterior-channel order. Choose the seed before generation. The native and compact experimental checkpoints all load into the same program.

```sh
python "$code/paper_object_infer.py" --eeg one_query.npy \
    --checkpoint "$EEG_RUNS_ROOT/enc/native/sub-01/fold0_seed2025/brain_model.pt" \
    --seed 20261125 --output reconstruction.png
```
