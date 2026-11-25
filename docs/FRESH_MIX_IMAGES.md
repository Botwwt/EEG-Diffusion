# What the new EEG object readout changes in generated images

For one participant, independent CLIP-L/14 gave images from the expanded object readout **157.5 of 350** naming credits across 1,854 words for the true object, up from **141** for the legacy readout. The paired gain was **4.71 percentage points** (95% interval **1.29 to 8.14**). DINOv2 identified the exact viewed photograph among ten of the same object 46 times with the new readout and 44 times with the old one. Its interval includes both a decrease and an increase. [Image registration](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_images_design.json) · [All results](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_images.json)

## Results

| Frozen image endpoint | Strict 35 objects, legacy → learned / 350 | Gain in percentage points (95% interval) | All 48 objects, legacy → learned / 480 | Gain in percentage points (95% interval) |
|---|---:|---:|---:|---:|
| True-object naming among 1,854 words, primary | 141 → 157.5 | +4.71 (1.29 to 8.14) | 203.5 → 221 | +3.65 (0.52 to 6.67) |
| True-object naming among the twelve menu words | 159 → 176 | +4.86 (1.14 to 8.57) | 229 → 246 | +3.54 (0.42 to 6.67) |
| DINOv2-S/14 retrieval of the viewed photo among ten of the true object | 44 → 46 | +0.57 (−1.71 to 2.86) | 63 → 66 | +0.63 (−1.04 to 2.50) |

The nomination of the 35 primary objects preceded the object-readout experiment. The other thirteen had been used for historical RGB-gain calibration on a different participant. All 480 queries had the full twelve candidates. The primary comparison retains every query for the 35 objects, including wrong EEG words. The image endpoints were registered after the word result was known and before image generation. Fractional naming credit splits exact top-score ties. [Participant, exclusions and calibration](FRESH_MIX.md) · [Word and image registrations](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_design.json)

| Menu | Primary rows | Full-vocabulary naming, legacy → learned | DINO photo matches, legacy → learned |
|---|---:|---:|---:|
| mix13 | 110 | 55 → 65 | 12 → 10 |
| mix14 | 80 | 33 → 34 | 9 → 11 |
| mix23 | 50 | 19 → 22 | 9 → 8 |
| mix24 | 110 | 34 → 36.5 | 14 → 17 |

On the primary rows, 28 naming credits were gained and 11.5 were lost. Leaving out each menu in turn yielded gains from 2.71 to 5.83 percentage points. The reported intervals use 10,000 paired bootstrap draws of whole objects within the four menus, with seed 260628. They measure uncertainty across objects for one participant, fixed fitted models and one noise seed per photograph. Each DINO gallery includes the eight other-fold original photographs available as visual supports when the decoded word is correct. [Paired counts, controls and intervals](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_images.json)

## What was held fixed

Each method selected a word from the same session-matched menu using sub-04's four presentations and eight calibration photographs per candidate. Both generation paths used the previously frozen 512-object EEG-to-H/14 photo decoder. A changed word selected both a different prompt and that object's registered supports for the visual condition. The generator, training-derived H/14 radius, row order, batch size of eight, four diffusion steps and seed of 923761 plus the original row ID were held fixed. The evaluation zip containing the query originals was opened separately for scoring. The experiment measures the combined change from word and support selection. [Word-to-condition builder](../research/eeg_stack_20261112/code/package_fresh_mix.py) · [Renderer and byte verification](../research/eeg_stack_20261112/code/render_fresh_mix_images.py)

Both methods independently generated all 480 JPEGs. Wherever they chose the same word, the eight support IDs and the visual condition matched exactly. **All 365** such independent draws matched byte for byte, including 95, 94, 91 and 85 per menu. All 115 changed-word pairs produced different JPEGs. The smallest original same-word row in each menu was 1070, 620, 820 and 3351 respectively. The receipt includes the generated hash from each of these four independent rerenders. [Per-menu render audit](../research/eeg_stack_20261112/results/object_fidelity/fresh_mix_images.json)

## Fixed image comparisons

Each gallery shows the lowest original row for every one of a menu's twelve objects. For chili in mix13, the older draw is a waterwheel and the newer draw shows a bowl of chili. The ski-boots row in that menu remains a carriage under both methods. For the tulip row in mix23, the plants change from pineapples to tulips. For the magnifying-glass row in mix24, the new image shows insoles and loses a correct object. The private plates add the original photograph beside both generated images. The linked plates contain only generated JPEGs. [mix13](../research/eeg_stack_20261112/gallery/fresh_mix_mix13.jpg) · [mix14](../research/eeg_stack_20261112/gallery/fresh_mix_mix14.jpg) · [mix23](../research/eeg_stack_20261112/gallery/fresh_mix_mix23.jpg) · [mix24](../research/eeg_stack_20261112/gallery/fresh_mix_mix24.jpg)

## Replay

Prepare the two frozen condition sets with the commands in [the word study](FRESH_MIX.md). From the repository root, build and preflight the worker against the same scientific code and frozen inputs.

```bash
C=research/eeg_stack_20261112/code
P=outputs/fresh-mix
python "$C/package_fresh_mix.py"
python "$C/package_fresh_mix_image_code.py"
for panel in mix13 mix14 mix23 mix24; do
  python "$C/render_fresh_mix_images.py" --root "$P/render_inputs" --panel "$panel" --check-only
done
```

Copy `fresh_mix_blind_inputs.tar`, `fresh_mix_image_code.tar` and the [Slurm array script](../research/eeg_stack_20261112/jobs/run_fresh_mix_images.sbatch) to the submit directory. The archived input SHA-256 is `94cd4280a6885a9a1a359ee674f2cb7a73edd5db95f6930dfc33708a6c38ad76`, and the submitted code SHA-256 is `b74d9c5ac77b8741fe544bf393aa2035b3032ef5a8db3cb4033d1b547aacb40e`. Each worker validates copied archives on node-local storage, draws 240 JPEGs and uses the independent frozen CLIP-L/14 to save 1,854-word logits. Set `ARCHIVE` to the offline generator and scorer asset archive. The official dictionary and base H/14 IP-Adapter reside under `official/` in the submit directory. [Source-only packager](../research/eeg_stack_20261112/code/package_fresh_mix_image_code.py) · [Naming worker](../research/eeg_stack_20261112/code/name_fresh_mix_images.py)

```bash
sbatch --export=ALL,ARCHIVE,CODE_SHA=b74d9c5ac77b8741fe544bf393aa2035b3032ef5a8db3cb4033d1b547aacb40e run_fresh_mix_images.sbatch
```

On the evaluation host, put the four returned tar files in `outputs/fresh-mix/cluster`. The scorer verifies each image, the word packets, all original row identities, the original photo archive, and both evaluation models. It writes the public summary and generated-only plates. Private row scores and plates with originals are stored under `outputs/fresh-mix/evaluation/<menu>/`. [Independent image scorer](../research/eeg_stack_20261112/code/score_fresh_mix_images.py)

```bash
for panel in mix13 mix14 mix23 mix24; do
  tar xf "$P/cluster/$panel.tar" -C "$P/render_inputs"
done
python "$C/score_fresh_mix_images.py"
```
