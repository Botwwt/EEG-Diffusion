# Can individual EEG presentations resolve the two leading objects?

For sub-04, the original trial encoder and Fisher classifier recognized **514 of 960** held photographs in a twelve-object menu. A rule using the four presentations separately changed 106 choices. It corrected 23 and spoiled 33, leaving **504/960**. The difference was **−1.04 percentage points** (95% whole-object interval **−2.71 to +0.52**). The registered threshold for testing target images was unmet. [Design](../research/eeg_stack_20261112/results/object_fidelity/trial_top_two_design.json) · [Aggregate and input hashes](../research/eeg_stack_20261112/results/object_fidelity/trial_top_two.json)

| Final decision | Correct / 960 | Corrected | Newly wrong | Wrong to wrong |
|---|---:|---:|---:|---:|
| Original four-presentation mixture | 514 | — | — | — |
| Learned rule, own presentation evidence | 504 | 23 | 33 | 50 |
| Learned rule, exchanged presentation evidence | 356 | 55 | 213 | 121 |
| Fixed equal-weight rule, own evidence | 506 | 23 | 31 | 42 |
| Fixed equal-weight rule, exchanged evidence | 458 | 38 | 94 | 87 |

All arms used the same four physical presentations per photograph and the same two finalists from the original mixture. Each candidate had eight calibration photographs distinct from the two held photographs in its fold. The exchange preserved the query’s original classifier scores, finalists, supports and recording-session pair. It supplied only the individual and paired presentation evidence from another menu candidate’s held photograph. Own evidence beat this exchange by **15.42 points** [11.35, 19.48] for the learned rule. The fixed rule gained **5.00 points** [2.50, 7.50] against its exchange and differed from the original mixture by **−0.83 points** [−2.19, 0.52]. Correct pairing mattered, yet the extra evidence did not improve on processing the four presentations together.

## Calibration and evaluation

The 416 generic fit objects were divided by recording-session pair into two disjoint sets of 206 and 210. Each set supplied training EEG for a complete 750-update neural fit and a Fisher projection. Both transforms were then applied exclusively to the other set. Out-of-class queries thus provided 4,160 episodes with one twelve-object decision per query. The reversal model used the score gaps from Fisher and the full encoder, four single-presentation scores, all six two-presentation subsets, evidence against complementary and same-session support views, and comparisons of the eight independent support photographs. The fixed rule gave equal weight to Fisher, the full encoder, and the average of the six complementary-pair scores.

In 2,504 calibration decisions the true object appeared among the two finalists. These rows trained the binary logistic loss. A second four-fold split by whole object supplied cutoff predictions. All 4,160 decisions counted when choosing the threshold. The selected probability cutoff of 0.50 produced 135 corrections and 111 new errors during calibration. The complete rule was frozen before extracting the held 96 objects’ individual-presentation embeddings. These existing development objects had previously been evaluated with other models. No personal target panel contributed EEG or images to this experiment.

The held run exactly reproduced every baseline prediction and four-presentation embedding. The largest absolute candidate-score replay difference was below 5 × 10⁻¹⁵. The correct class was among the original two finalists on 662/960 photos; 148 of these were genuine opportunities to repair a baseline error. Paired intervals sampled 96 whole objects with their ten photographs, for 10,000 draws. The decision to advance required at least ten net additional hits and positive lower interval bounds against both the original mixture and exchanged evidence.

## Replay

Use the sub-04 generic trial blocks, posterior features, recording-session schedule, and archived generic encoder/decisions from the [trial encoder study](TRIAL_OBJECT_ABLATION.md). Run the four commands in order from the repository root. The first command trains two neural encoders on separate object folds. The third loads the archived 416-class encoder.

```sh
C=research/eeg_stack_20261112/code
python "$C/fit_trial_top_two.py" folds
python "$C/score_trial_top_two.py" calibrate
python "$C/fit_trial_top_two.py" held
python "$C/score_trial_top_two.py" score
```

The commands write the coefficients, fold receipts, 960 original and exchanged row decisions, and the summary under `outputs/trial-top-two/replay`. The original run remains under `outputs/trial-top-two`. Commit `9a87881482547aa8f3d27f305c11a4121ad95575` preserves its exact source files and public result. The maintained scripts were run from scratch after formatting. All three view packets (two object folds and the held data), the fitted numerical coefficients, and the full per-query packet matched the original. Source hashes in each receipt identify its own version. After also running the [scale replay](SCALE_TRIAL_OBJECT.md), verify both studies with `python "$C/audit_scale_trial_replay.py"`. [Replay hashes and coefficients](../research/eeg_stack_20261112/results/object_fidelity/trial_top_two_replay.json) · [View extraction](../research/eeg_stack_20261112/code/fit_trial_top_two.py) · [Calibration and scoring](../research/eeg_stack_20261112/code/score_trial_top_two.py)
