# Learning to discriminate objects viewed four times by one person

The object task supplies eight labeled support photographs per candidate object and the corresponding EEG from the same person. We tested three changes to the EEG classifier. The candidate menus in each comparison hold recording-session membership fixed. Each reported decision uses one person's four measured presentations of the query photograph. The image generator receives the predicted word as one of its inputs. [Default reconstruction model](PHOTO_STRUCTURE.md) · [Historical episodic classifier](EPISODIC_OBJECT.md)

## Training on the hardest object pairs

The candidate estimates which two competing class centers lie nearest to a focal class using its eight support EEGs. A classifier learns how to distinguish these pairs on other objects. On the historical 96 held-out generic classes, the previous episodic classifier scored 2,937/5,760 and the hard-pair variant scored 2,934/5,760 across the two menu sizes and three people. The difference was −0.05 percentage points (paired 95% object-bootstrap interval −0.64 to +0.54). The four-way task gained 13 hits and the twelve-way task lost 16. On the two previously studied target tasks, Fisher, episodic and hard-pair classifiers scored 377, 394 and 395 out of 720. The hardest-pair variant did not qualify for a prospective reconstruction comparison. [Design](../research/eeg_stack_20261112/results/object_fidelity/hard_object_pairs_design.json) · [Results](../research/eeg_stack_20261112/results/object_fidelity/hard_object_pairs.json)

## Products of different presentations

We separately evaluated whether two electrodes carry a reproducible spatial pattern across the four trials. Within each 80 ms window we centered the waveform in time, then multiplied the electrode values of two different trials at matching time samples. We averaged over the twelve ordered trial pairs and retained the upper triangle of each of six 17-channel matrices. Each fold held out all objects with one recording-session pair in one participant. Eight EEG supports per class determined the prototypes. All twelve objects in a decision shared the query's session pair.

On 128 development objects in three people, the existing mean-waveform Fisher head correctly named 1,041/3,840. Giving 25% of the standardized score to a Fisher head fitted on the cross-trial matrices gave 1,029/3,840 (−0.31 percentage points, 95% interval −1.38 to +0.78). Computing products from the averaged waveform instead gave 1,030/3,840. The 3,840 decisions cover 1,280 different photographs. [Design](../research/eeg_stack_20261112/results/object_fidelity/repeat_cross_moment_design.json) · [Counts and uncertainty](../research/eeg_stack_20261112/results/object_fidelity/repeat_cross_moments.json)

## Learning how to pool four trials

The learned encoder takes four separate 63-channel trials from the first half-second after stimulus onset. It learns spatial and temporal filters, downweights trials that differ from their peers in each latent channel, and produces a 96-dimensional representation. Training compares objects from the same recording-session pair. A second loss matches two disjoint pairs of EEG trials from the same photograph among ten photographs of that object. The encoder receives EEG and training identities. The comparison with the historical baseline averages its standardized prototype scores with the Fisher scores at fixed equal weight. [Training and evaluation design](../research/eeg_stack_20261112/results/object_fidelity/trial_object_encoder_design.json)

| Sub-04 object task | Fisher | Previous episodic classifier | Trial encoder | Fisher + trial encoder |
| --- | ---: | ---: | ---: | ---: |
| 96 development objects, twelve choices | 437/960 | 454/960 | 474/960 | **514/960** |
| Same development photos, four choices | 647/960 | 645/960 | 681/960 | **689/960** |
| Historical test panel, twelve choices | 55/120 | 56/120 | 49/120 | 57/120 |
| Historical test panel, four choices | 79/120 | 87/120 | 86/120 | 85/120 |
| Twelve new objects from one personal recording-session pair | 53/120 | — | 58/120 | **65/120** |

On the new panel, changing the model changed 31 decisions. Sixteen earlier mistakes became correct and four earlier correct decisions became mistakes. The paired gain was **10.0 percentage points** (95% interval 2.5 to 18.3, resampling twelve objects). The trained weights predated the panel comparison. We knew the original classifier's results when we registered the head-to-head, and then froze the comparison before running the new encoder on these objects. Both models had eight labeled calibration photos per candidate. There was one generation per query and arm. [Development and historical tests](../research/eeg_stack_20261112/results/object_fidelity/trial_object_encoder.json) · [Head-to-head registration](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12_design.json) · [Word counts and inputs](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12.json)

### What changed in the actual images?

On that participant's 120 new photographs, switching the word decoder raised correct object naming of the saved own-EEG images from **52 to 65** among the twelve candidate names and from **45 to 55** among 1,854 names. The full-vocabulary difference was +8.33 percentage points (paired 95% interval +2.50 to +15.00). All 89 own-EEG images whose word stayed fixed matched byte for byte. DINOv2 identified the particular photograph in **7 versus 8** cases out of 120 (difference +0.83 points, interval −3.33 to +5.00). When both classifiers got both words right, both assigned 10 of the same 13 photo pairs correctly. The improvement on this participant was in drawing the right object. [Paired JPEG results](../research/eeg_stack_20261112/results/object_fidelity/personal12_trial_images.json)

### Does the new encoder preserve information about individual photographs?

We fitted the same ridge readout on 416 training objects to predict how a photograph differs from its object's H/14 mean. On the 96 held-out development objects, a half-weight blend of deep and posterior predictions correctly assigned 3,333 of 4,320 pairs. The posterior readout alone assigned 3,179 (gain 3.56 percentage points, 95% interval 2.34 to 4.79). In the separate single-query ten-photo test, the counts were 19 and 22 of 960. Eight alternatives in that ten-photo gallery also constructed the visual prior, whereas the two-photo assignment excluded both alternatives from calibration. The candidate missed its registered single-query requirement. The image pipeline keeps the existing photo readout. [Residual design](../research/eeg_stack_20261112/results/object_fidelity/trial_photo_residual_design.json) · [Counts and paired uncertainty](../research/eeg_stack_20261112/results/object_fidelity/trial_photo_residual.json)

### Relation to existing EEG reconstruction methods

TDE-Net also uses repeated EEG responses to the same training photograph as contrastive positive pairs on THINGS-EEG2. Its graph stream uses other conditions in the batch as negatives. Our auxiliary task offers ten photographs of **one object** as alternatives within each training episode, and pairs two complementary halves of the four measured trials. The object task arranges competing categories within a single recording-session pair. [TDE-Net training protocol, October 2026](https://doi.org/10.3791/73434) · [Controlled loss and pooling ablation](TRIAL_OBJECT_ABLATION.md)

ATM aligns EEG to a CLIP embedding space and conditions a diffusion generator. Brain-HiVE learns a hierarchy of visual features using several pretrained encoders. More recent preprints select EEG-predictable visual layers or supervise the internal state of a generator with an image-conditioned teacher. Here the object classifier learns directly from repeated electrical measurements and receives eight EEG supports per candidate at inference. The distinguishing experimental question is whether the signal that survives across presentations helps a generator draw the correct object under matched recording sessions, and whether it also identifies a specific photograph of that object. Comparisons to a zero-shot 200-way retrieval leaderboard would require a separate matched evaluation. [ATM, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/ba5f1233efa77787ff9ec015877dbd1f-Paper-Conference.pdf) · [Brain-HiVE, ICLR 2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/c5506f154e5f2203a7dca2bee2faeced-Abstract-Conference.html) · [Layer-wise EEG visibility, 2026](https://arxiv.org/abs/2609.02582) · [Generation-derived supervision, 2026](https://arxiv.org/abs/2611.19232)

## Contribution of the training components

The four-run component test isolated the auxiliary same-photo loss and the reliability weights. On sub-04's 96 held development objects, adding the auxiliary loss improved twelve-way mixed recognition by 2.60 percentage points averaged over pooling rules (paired 95% interval +0.99 to +4.22). The estimated contribution of reliability weighting was +0.52 points [−0.52, +1.56]. In a fifth fit, changing the nine negatives from other photographs of the same object to photographs of nine other same-session objects gave 511/960 against 514/960. The interval for the difference included zero. [Full component comparison, related work and replay](TRIAL_OBJECT_ABLATION.md)

## Replay

The hard-pair study uses the existing object and episodic caches. The cross-trial study uses the original 128-class source ranges for the three participants. The neural encoder additionally uses sub-04's 512-class trial tensors and historical split. Start from the data and model caches in the earlier replay guides. [Personal EEG and source preparation](FRESH_PHOTO_REPLAY.md) · [Episodic baseline](EPISODIC_OBJECT.md#replay)

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/fit_hard_object_pairs.py" fit
python "$C/fit_hard_object_pairs.py" summarize
python "$C/fit_repeat_cross_moments.py"
python "$C/prepare_trial_object_inputs.py" --group all
python "$C/fit_trial_object_encoder.py" generic
python "$C/score_trial_object_encoder.py" generic
python "$C/fit_trial_object_encoder.py" target
python "$C/score_trial_object_encoder.py" target
python "$C/summarize_trial_object_encoder.py"
python "$C/fit_trial_photo_residual.py"
```

The trial-input preparer verifies the archived EEG bytes, the 63 electrode identities, time axis, photo order and the float32 derivatives. The GPU fit saves its weights, training-only normalization, class IDs and EEG directions locally. The scoring step independently replays the previous classifiers' decisions in the same menus.

## Learning from other participants viewing the same photograph

Three participants shared a temporal EEG encoder with a separate electrode projection for each person. During training, the auxiliary loss asked the model to match each response to another participant's response to the same photo among ten photos of one object. The controls either shifted the photo identities within the object or trained the shared network without the auxiliary loss. All three shared models received 750 updates on the same 416 objects. On the 96 held-out development objects, each decision used only the participant's own four query trials and their EEG from eight support photos per candidate. Each participant's historical same-session menu was fixed across methods. [Registration](../research/eeg_stack_20261112/results/object_fidelity/shared_trial_encoder_design.json)

Across 2,880 twelve-way decisions on 960 different photos, the matched-photo model scored 1,279, the independent models 1,257, the shifted-photo control 1,258 and shared training alone 1,266. The gain over independent training was 0.76 percentage points (paired 95% object-bootstrap interval −0.83 to +2.29). The gain over shifted photos was 0.73 points (−0.49 to +1.94). Four-way recognition gained 19 hits over independent training. Both primary intervals included zero, so the registration stopped this candidate at the development stage. [Results](../research/eeg_stack_20261112/results/object_fidelity/shared_trial_encoder.json)

Run the commands below from the repository root after preparing the three participants' 512-class trial caches, the historical episodic readouts and sub-04's original generic trial-model results. The scoring step checks exact replay of sub-04's published neural vectors and all three participants' historical Fisher decisions. [Participant trial preparation](TRIAL_OBJECT_PERSONAL12.md#replay) · [Training and scoring code](../research/eeg_stack_20261112/code/fit_shared_trial_encoder.py)

```sh
C=research/eeg_stack_20261112/code
W=outputs/shared-trial-encoder
mkdir -p "$W"
cp research/eeg_stack_20261112/results/object_fidelity/shared_trial_encoder_design.json "$W/design.json"
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/fit_shared_trial_encoder.py" train --root "$W"
python "$C/fit_shared_trial_encoder.py" evaluate --root "$W"
cmp "$W/result.json" research/eeg_stack_20261112/results/object_fidelity/shared_trial_encoder.json
```
