# Does expanding the trial encoder's training set improve object recognition?

Adding 512 training objects improved the **neural encoder alone** on held twelve-object menus. At 1,500 updates it rose from **458 to 509 correct out of 960** (+5.31 percentage points, 95% whole-object interval +2.40 to +8.23). The registered equal-weight mixture of neural and Fisher scores moved from **514 to 512** (−0.21 points, −2.29 to +1.98). The mixed classifier missed its gate for target inference and image generation. [Frozen design](../research/eeg_stack_20261112/results/object_fidelity/scale_trial_design.json) · [Results, paired intervals and hashes](../research/eeg_stack_20261112/results/object_fidelity/scale_trial.json)

| External objects | Updates | Twelve-way mixed | Twelve-way neural | Four-way mixed | Four-way neural |
|---:|---:|---:|---:|---:|---:|
| 416 | 750 | 514 | 474 | 689 | 681 |
| 416 | 1,500 | 514 | 458 | 689 | 679 |
| 928 | 750 | 514 | 492 | 710 | 711 |
| 928 | 1,500 | 512 | 509 | 697 | 717 |

Each entry counts 960 queries from sub-04. The twelve- and four-candidate menus use the same photographs. Every candidate in a menu has the same recording-session pair and contributes eight support photographs distinct from its queries. Each photograph has four physical presentations. The 96 held objects had featured in earlier development experiments.

At the matched 1,500-update budget, the expanded neural head corrected 131 twelve-way decisions and spoiled 80. After combining it with Fisher, the differences were 56 gains and 58 losses. Compared with the original 416-object, 750-step mixed model, the expanded 1,500-step model gained 55 and lost 57 (−0.21 points, −2.19 to +1.77). In the four-way mixed comparison with matched updates, it gained 45 and lost 37 (+0.83 points, −1.04 to +2.71). The individual neural branches and the fixed mixture were specified before these results. We did not select a new mixture weight on these 96 objects.

The row-level decisions show where the twelve-way gain went. At 1,500 updates the number of photographs on which **only the neural head** chose the right object increased from 151 to 173. The mixed classifier retained 106 of those 151 correct choices with the smaller fit and 98 of 173 with the larger fit. Both heads chose correctly on 307 and 336 photographs respectively. These counts describe the tested mixture and do not establish a better weighting rule.

## What changed in training

The 512 additional objects came from the earlier doubled-calibration registration. The existing sub-04 range receipts identify the raw bytes of all 512. Twelve of these objects had already been acquired for the `stress12` study, so this test uses that menu's objects for training and excludes its scores. The remaining 500 objects require 2,520,000,000 bytes of raw EEG. The acquisition code verifies every complete object range and retains four separate 63-channel, 125-sample float32 presentations per photograph. The three historical personal menus contain 29 distinct objects, all disjoint from both training lists. [Object selection](../research/eeg_stack_20261112/results/object_fidelity/double_calibration_design.json) · [Acquisition](../research/eeg_stack_20261112/code/prepare_scale_trial.py)

Every cell uses the same 96-dimensional encoder, initial random seed, reliability pooling, within-photograph auxiliary loss and AdamW settings. Both step budgets use a cosine learning-rate schedule over their full training horizon. The classifier averages each query’s standardized encoder scores and the original Fisher scores. All four cells use the same 416-class Fisher fit. The original 750-step replay reproduced every weight and all 960 four-presentation embeddings exactly, along with all 1,920 mixed decisions across both menus. [Earlier trial-encoder study](TRIAL_OBJECT_ABLATION.md)

The primary comparison pairs the 928-object, 1,500-step model with the 416-object, 1,500-step model. The registration also requires an improvement over the historical 416-object, 750-step model and no decrease in the four-way result against either comparator. The matched-budget twelve-way difference must have a strictly positive 95% lower bound when resampling all ten photographs of each of the 96 objects together.

## Replay

Run from the repository root with the sub-04 source receipts, historical generic decision packets, and original cropped trial inputs from the [earlier encoder experiment](TRIAL_OBJECT_ABLATION.md).

```sh
C=research/eeg_stack_20261112/code
python "$C/prepare_scale_trial.py" --workers 6
python "$C/fit_scale_trial.py" old
python "$C/fit_scale_trial.py" expanded
python "$C/fit_scale_trial.py" score
```

The acquisition resumes per object. Once the additional crops have been verified, `--offline` rechecks the cache under `outputs/scale-trial/extra`. Each replay fit writes its checkpoint, optimization-stream hashes and row decisions under `outputs/scale-trial/replay`. The personal12 fitting and packaging commands require a passing generic gate. [Factorial fit and score](../research/eeg_stack_20261112/code/fit_scale_trial.py) · [Conditional visual packaging](../research/eeg_stack_20261112/code/package_scale_trial.py)

Commit `9a87881482547aa8f3d27f305c11a4121ad95575` preserves the original five training-scale and top-two source files and their original results. The [later prospective study](FRESH_MIX.md) binds the original scale result with SHA-256 `517bfc9001e86fc84847eddfc97846617929d9dc1126df2f0ba44bd08399c368`. The maintained scripts refit all four cells from the verified EEG. All four checkpoint files, four embedding packets and eight per-query output packets were byte-identical. That includes 7,680 mixed decisions and 7,680 neural-only decisions. The current and historical source receipts are linked in the [replay audit](../research/eeg_stack_20261112/results/object_fidelity/scale_trial_replay.json). After both the scale and [top-two](TRIAL_TOP_TWO.md) replay stages, run `python "$C/audit_scale_trial_replay.py"` to verify these identities.
