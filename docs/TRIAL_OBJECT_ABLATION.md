# What helped the single-person trial encoder recognize objects?

The original sub-04 encoder used two trainable ingredients. It downweighted EEG presentations that differed from the within-photo trial mean, and learned to match two complementary pairs of presentations of the same photograph. Retraining all four combinations gave the clearest evidence for the **same-photo training loss**. Averaged across the two pooling rules, it raised twelve-way recognition by **2.60 percentage points** with a paired 95% interval of **+0.99 to +4.22** for the registered mixture of encoder and Fisher scores. [Frozen component design](../research/eeg_stack_20261112/results/object_fidelity/trial_object_ablation_design.json) · [Full results and hashes](../research/eeg_stack_20261112/results/object_fidelity/trial_object_ablation.json)

## Four fits on the same EEG

Each model trained for 750 updates on 416 external object classes. The 96 held classes supplied ten query photos each. Each query was classified once in a four-candidate menu and once in a twelve-candidate menu. All candidates in a menu shared the same recording-session pair. Each candidate contributed eight calibration EEGs. Accuracy below is the number of correct decisions out of 960 in each column.

| Pooling across trials | Same-photo loss coefficient | 12-way encoder | 12-way encoder + Fisher | 4-way encoder | 4-way encoder + Fisher |
|---|---:|---:|---:|---:|---:|
| Learned reliability | 0.2 | 474 | **514** | 681 | **689** |
| Learned reliability | 0 | 433 | 488 | 641 | 684 |
| Arithmetic mean | 0.2 | 470 | 508 | 672 | 688 |
| Arithmetic mean | 0 | 441 | 484 | 646 | 682 |

Each mixed decision gave equal weight to the two standardized candidate-score vectors. Holding reliability pooling fixed, enabling the photo loss raised twelve-way mixed recognition from 488 to 514 (+2.71 pp, 95% interval +0.73 to +4.69). Holding the photo loss fixed at 0.2, replacing the mean with learned reliability raised the count from 508 to 514 (+0.62 pp, −0.63 to +1.98). Averaging over both loss settings gave a reliability effect of +0.52 pp [−0.52, +1.56]. The interaction between the two switches was +0.21 pp [−1.35, +1.67]. For the encoder alone, the average effect of the auxiliary loss was +3.65 pp [+1.77, +5.57].

The auxiliary task identifies the same photograph from its EEG in two disjoint two-presentation views. With exactly two presentations, both observations are equally far from their common mean, so the reliability rule gives them equal weights in exact arithmetic. The rule can assign unequal weights when the network processes all four presentations. In the four-way mixed endpoint the average effect of the photo loss was +0.57 pp [−0.63, +1.77].

This component comparison was registered after the original full-model score on these 96 objects was known. It uses one participant and one seed. The intervals resample the 96 object identities 10,000 times with all arms paired. The result supports a contribution to twelve-way object recognition in this development task. The study measured no generated-image or held-out participant endpoint. [Original recognition experiment](SINGLE_PERSON_OBJECT_ENCODERS.md) · [Independent fixed-baseline image evaluation](PERSONAL12_CONFIRMATION.md)

## Which photos served as negatives?

Zhou and colleagues previously used two repetitions of the same THINGS-EEG2 image as a positive pair during TDE-Net graph-encoder pretraining. Their negatives came from other image conditions in the batch. Here, the original auxiliary task used nine other photographs of the same object. We tested whether this restriction added value in the sub-04 encoder. [TDE-Net protocol, sections 1.2 and 2.4](https://www.jove.com/t/73434/tri-stream-disentangled-network-with-adaptive-dual-space-fusion-for)

One additional fit preserved the positive, its position among the ten logits, the loss coefficient and the complete training random stream. For each anchor, it drew the nine negatives from the same shuffled photo position in nine distinct objects of that same-session training episode. The full model and this control each saw one positive and nine negatives per anchor. [Frozen control design](../research/eeg_stack_20261112/results/object_fidelity/trial_object_negative_set_design.json) · [Results](../research/eeg_stack_20261112/results/object_fidelity/trial_object_negative_set.json)

| Readout | Within-object negatives /960 | Cross-object negatives /960 | Difference, paired 95% interval |
|---|---:|---:|---:|
| Twelve-way encoder + Fisher | 514 | 511 | +0.31 pp [−0.63, +1.25] |
| Twelve-way encoder | 474 | 478 | −0.42 pp [−1.77, +0.94] |
| Four-way encoder + Fisher | 689 | 684 | +0.52 pp [−0.31, +1.46] |
| Four-way encoder | 681 | 676 | +0.52 pp [−0.63, +1.77] |

Eleven twelve-way mixed decisions favored the within-object negatives and eight favored the cross-object negatives. The paired interval includes zero. The gain over disabling the auxiliary loss is established on this development split; the choice of negative set remains unresolved.

## Replay

With the original sub-04 generic trial inputs and score packets prepared, run the two stages from the repository root in the original PyTorch environment.

```sh
C=research/eeg_stack_20261112/code
python "$C/ablate_trial_object_encoder.py" train
python "$C/ablate_trial_object_encoder.py" summarize
python "$C/compare_trial_negative_sets.py"
```

The four fits had identical hashes for their initial state, 750 episode/trial-permutation sequences and per-update generator states. All computed both complementary views and the auxiliary loss, including the fits with zero coefficient. The repeated full fit reproduced every saved weight, normalization parameter, held-out embedding and all 1,920 original mixed decisions exactly. Intermediate weights and per-row predictions are written under `outputs/trial-object-ablation`. [Executable](../research/eeg_stack_20261112/code/ablate_trial_object_encoder.py) · [Archived full-model scores](../research/eeg_stack_20261112/results/object_fidelity/trial_object_encoder.json)
