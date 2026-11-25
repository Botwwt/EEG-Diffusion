# Four-trial object recognition on the participant-specific 12-object menus

Each participant saw each query photograph four times. The 12 candidate objects all share that participant's recording-session pair. Each word has eight labeled support photographs and their EEG in the current fold. The decoder sees the query EEG and chooses a word among those twelve. The generator also receives the eight permitted support-image embeddings for the chosen word. Each of the ten photos per object serves once as a held-out query. [Panel registration](../research/eeg_stack_20261112/results/object_fidelity/single_person_confirmation_panel.json)

## First participant

Before applying the trial encoder to sub-04's 120 confirmation photographs, we fixed an equal average of its 12 standardized prototype scores and the original Fisher scores. The 512-object checkpoint and the earlier Fisher result on the confirmation panel were already known. On this panel Fisher identified 53/120 objects, the trial encoder alone 58/120, and their average 65/120. The mixed readout corrected 16 Fisher errors and lost four Fisher hits. Its paired gain was 10.0 percentage points, with a 95% interval of 2.5 to 18.3 points from 10,000 whole-object bootstrap draws. These observations cover twelve objects in one participant. [Preregistered comparison](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12_design.json) · [Counts and packet hashes](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12.json)

The original Fisher predictions and H/14 conditions reproduced on all 120 rows. The mixed decoder kept 89 words unchanged. On those rows the selected supports, own directions and uniform directions were byte-identical to Fisher. The wrong-EEG control could exchange with another query of the same predicted word and fold on 107 rows. Word accuracy measures the decoder's choice before rendering. The renderer reuses the original per-photo noise seed and the native H/14 angle, radius and preprocessing. A clean second run reproduced all six output files byte for byte.

## Independent participant replication

The replication was registered after the sub-04 comparison. The original Fisher scores for sub-05 (56/120) and sub-01 (52/120) were already known. For each person, the same training function and seed initialize a separate network and train it for 750 updates using only that person's four presentations of each of the 5,120 generic photos. The 512 training objects exclude all 29 distinct confirmation objects. Both participants use the same training order, architecture, loss, 50:50 readout and render protocol as sub-04. Their scores are reported separately because each person supplied their own recordings and some of the target photographs are shared. [Replication registration](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12_replication_design.json)

The sub-05 menu uses sessions 2 and 4 and the sub-01 menu uses sessions 1 and 4. They share three objects. The two participants together contribute 240 decisions on 210 different photos from 21 objects. The pooled interval resamples those 21 objects and retains both participants' results whenever an object is shared.

| Participant | Fisher | Trial encoder | Fixed average | Changed from wrong to correct / correct to wrong | Hit-rate change and 95% object interval |
| --- | ---: | ---: | ---: | ---: | --- |
| sub-05 | 56/120 | 55/120 | 54/120 | 7 / 9 | −1.7 points [−9.2, +5.8] |
| sub-01 | 52/120 | 56/120 | 68/120 | 19 / 3 | +13.3 points [+7.5, +19.2] |

The two independently fitted mixed classifiers scored 122/240 versus 108/240 for Fisher, a pooled gain of 5.8 percentage points (95% object-bootstrap interval +1.6 to +10.4). The gain differed by participant. The interval describes variation across these objects for these two people. Fisher predictions and all three original H/14 arrays replayed exactly for both participants. The mixed readout kept the same words on 90 sub-05 rows and 85 sub-01 rows. A different query supplied the wrong-EEG control on 108 and 102 rows respectively. [Replication results and input hashes](../research/eeg_stack_20261112/results/object_fidelity/trial_object_personal12_replication.json)

## Generated images on both replications

Independent CLIP-L/14 evaluated every saved matched-EEG JPEG, including the images for which the word was wrong. The two replications together gained 4.58 percentage points in full-vocabulary naming (95% object-bootstrap interval +0.37 to +9.13). Across all three people the gain was +5.83 points (+2.06 to +9.71). The 17-electrode Fisher and 63-electrode trial encoder use the same four measured presentations. The comparison measures the complete change in the object readout, including electrode coverage and the handling of individual trials. [Paired results, including wrong-EEG and visual-mean controls](../research/eeg_stack_20261112/results/object_fidelity/personal12_trial_replication_images.json) · [Equal-budget component ablations](TRIAL_OBJECT_ABLATION.md)

| Participant (120 queries each) | EEG word, Fisher → mixed | Generated image, 12 names | Generated image, 1,854 names | Viewed photo among ten, DINOv2 |
| --- | ---: | ---: | ---: | ---: |
| sub-04, initial | 53 → 65 | 52 → 65 | 45 → 55 | 7 → 8 |
| sub-05, replication | 56 → 54 | 56 → 54 | 55 → 54 | 14 → 13 |
| sub-01, replication | 52 → 68 | 52 → 68 | 45 → 57 | 22 → 21 |

The three panels contain 290 distinct photos. On the 37 pairs where both classifiers correctly named both objects, DINOv2 correctly assigned 27 pairs for each model. In the full ten-photo gallery, eight photos also serve as calibration exemplars when the correct word is chosen. The two strict alternatives are both held out. [Fixed output sheets](../research/eeg_stack_20261112/gallery/GALLERY.md#personal-twelve-object-menus-with-matching-recording-sessions) · [Single-query inference](PERSONAL_OBJECT_INFERENCE.md)

## Color and layout after the object is decoded

We applied the previously fitted eight-by-eight color and luminance estimator to each mixed model's own-EEG image. All comparisons kept the base image and predicted word fixed. On 82 queries from 18 objects where the word was correct and a different EEG donor shared the true object, predicted word and fold, pooled opponent-color RMSE fell from 0.09867 after neutral JPEG re-encoding to **0.08412** with an eight-photo visual prior and **0.08376** with the query EEG. Exchanged EEG gave 0.08401. The matched-minus-prior difference was −0.00035 (95% interval −0.00107 to +0.00041). The image edits improved broad color agreement, with little additional color benefit attributable to the particular EEG. The default image uses the pre-edit generator. Reproduce the secondary experiment with `python "$C/score_personal12_color.py" render` followed by `python "$C/score_personal12_color.py" score`. [Spatial model](SPATIAL_PHOTO.md) · [Registered endpoints](../research/eeg_stack_20261112/results/object_fidelity/personal12_color_design.json) · [All queries, strict donors and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/personal12_color.json)

## Replay

Prepare the original 512-class features, single-person waveforms, registrations and Fisher packets as in the single-person panel replay. For the neural model use the repository environment with PyTorch, CUDA and safetensors. The physical sources are THINGS-EEG2 range blocks at the pinned release. The preparation step verifies every source block against the participant's original SHA-256 manifest. The program retains the 63 electrodes and first 125 post-onset samples at 250 Hz as float32, with four trials in their recorded order. Model fitting uses only the 512 generic objects. [Original 512 training procedure](SINGLE_PERSON_OBJECT_ENCODERS.md) · [Personal panels and generation](../research/eeg_stack_20261112/code/package_single_person_confirmation.py)

```sh
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/package_trial_object_confirmation.py"
for s in sub-05 sub-01; do
    python "$C/replicate_trial_object_confirmation.py" prepare --subject "$s"
    python "$C/replicate_trial_object_confirmation.py" train --subject "$s"
    python "$C/replicate_trial_object_confirmation.py" package --subject "$s"
    python "$C/render_within_photo_ip.py" --subject "$s" --panels personal12 --check-only \
      --root "outputs/trial-object-replication/$s/render_inputs"
done
python "$C/replicate_trial_object_confirmation.py" summarize
```

Each participant directory contains the private fit, row-level EEG decisions and mixed render packet. Under `render_inputs`, `design.json` and `radius.json` accompany `conditions/<subject>/personal12/conditions_512.npz`. The own, uniform and wrong arms share the recipient's predicted word and seed. The wrong arm receives the complete own direction of another query with the same predicted word and fold, with self-assignment for singletons. Scoring uses the separate participant registration and reference photographs.
