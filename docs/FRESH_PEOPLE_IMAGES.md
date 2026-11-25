# Generated images from two independently fitted four-presentation EEG readouts

The legacy and expanded readouts were rendered for sub-05 and sub-01 after their registered word-decoding gate passed. Each person contributed four EEG observations of each photograph and had eight support photographs per candidate in each fold. Both saw the same 350 primary photographs. Each person's recording sessions determined how their 35 primary objects were grouped into four twelve-word menus. The word result, 253 versus 281 correct decisions in 700 participant-photo observations, was known when the image evaluation was registered. [Word study](FRESH_PEOPLE.md) · [Image evaluation registration](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_images_design.json) · [Earlier sub-04 images](FRESH_MIX_IMAGES.md)

## Image results

Across the two participants, independent CLIP-L/14 identified the true object among 1,854 names on **243 of 700** primary participant-photo observations with the expanded readout, up from **218**. The difference was **+3.57 percentage points** (paired 95% interval **+0.86 to +6.14**). Both participants had a positive point difference, meeting the registered replication criterion. The individual interval for sub-01 also admits a decrease. The earlier sub-04 result was +4.71 points (+1.29 to +8.14). The new participants saw the same primary objects, regrouped into their own session-matched menus. [Two-person results](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_images.json) · [Discovery image results](FRESH_MIX_IMAGES.md)

| True-object or photograph endpoint | Primary, legacy → expanded / 700 | Change in percentage points (95% interval) | All, legacy → expanded / 960 | Change in percentage points (95% interval) |
|---|---:|---:|---:|---:|
| CLIP naming among 1,854 objects | 218 → 243 | +3.57 (0.86 to 6.14) | 309 → 348 | +4.06 (1.75 to 6.47) |
| CLIP naming among the participant's twelve menu objects | 253 → 281 | +4.00 (1.29 to 6.71) | 353 → 399 | +4.79 (2.33 to 7.33) |
| DINO identification among ten photos of the true object | 67 → 72 | +0.71 (−1.00 to 2.57) | 96 → 104 | +0.83 (−0.74 to 2.50) |

| Primary rows, each participant / 350 | CLIP full vocabulary, legacy → expanded | Change in percentage points (95% interval) | DINO within ten, legacy → expanded |
|---|---:|---:|---:|
| sub-05 | 107 → 123.5 | +4.71 (0.57 to 9.29) | 41 → 41 |
| sub-01 | 111 → 119.5 | +2.43 (−2.00 to 6.57) | 26 → 31 |

The full-vocabulary gain combines 63.5 naming credits gained and 38.5 lost. The exact-photo DINO measure gains 24 matches and loses 19, with an interval spanning zero. For all 960 participant-photo observations, both methods independently generated a JPEG, giving 1,920 actual draws. All 631 row pairs with identical decoded words reproduced the same JPEG bytes (313 in sub-05 and 318 in sub-01). All 329 pairs with different words produced different JPEGs. [Paired counts, re-render hashes and eight menu results](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_images.json)

### Fixed visual examples

The eight sheets show the lowest original row for each registered object in each participant's menu, covering 96 participant-photo comparisons. For sub-05's bolt, the old image is a pearl and the new image shows a bolt. The jeans row loses the correct object when the new image shows hair. For sub-01's bookshelf, a block of luncheon meat changes into a bookcase. The vacuum row changes from a visible vacuum cleaner to a valve. Both methods draw a carriage for sub-01's waterwheel row. The private versions add the actual original beside the generated pair, under `outputs/fresh-people/<subject>/evaluation/images/<menu>/private.jpg`.

| Participant | Generated-only galleries |
|---|---|
| sub-05 | [mix13](../research/eeg_stack_20261112/gallery/fresh_people_sub-05_mix13.jpg) · [mix14](../research/eeg_stack_20261112/gallery/fresh_people_sub-05_mix14.jpg) · [mix23](../research/eeg_stack_20261112/gallery/fresh_people_sub-05_mix23.jpg) · [mix24](../research/eeg_stack_20261112/gallery/fresh_people_sub-05_mix24.jpg) |
| sub-01 | [mix13](../research/eeg_stack_20261112/gallery/fresh_people_sub-01_mix13.jpg) · [mix14](../research/eeg_stack_20261112/gallery/fresh_people_sub-01_mix14.jpg) · [mix23](../research/eeg_stack_20261112/gallery/fresh_people_sub-01_mix23.jpg) · [mix24](../research/eeg_stack_20261112/gallery/fresh_people_sub-01_mix24.jpg) |

## Protocol

The two readouts supply the independently fitted participant's old or new word decision. A changed word changes both the text prompt and the eight permitted support photographs that condition the fixed H/14 photo decoder. Both draws use 512 pixels, the same SDXL-Turbo model, the base ViT-H/14 IP-Adapter, the same visual scale, four diffusion steps, the same batch order and the recipient's seed (923761 plus original row ID). The blind worker checks all ordered support IDs and float32 H/14 inputs for every same-word pair, then independently generates both sets. Query originals are opened only during local evaluation. [Worker and integrity checks](../research/eeg_stack_20261112/code/render_fresh_people_images.py)

The primary endpoint is independent CLIP-L/14 naming of the true object across all 1,854 words. All 700 primary participant-photo observations are included, regardless of either EEG model's chosen word. Secondary endpoints are true-object naming when the same logits are restricted to the person's twelve menu entries, and DINOv2-S/14 top-one identification of the original photograph among the ten photographs of its true object. Eight of those ten gallery entries also serve as potential visual supports when the decoded word is correct. Exact ties receive fractional credit. The primary criterion for replication is a strictly positive lower paired 95% bound on full-vocabulary naming and a nonnegative gain in each of these two participants. [Independent scorer](../research/eeg_stack_20261112/code/score_fresh_people_images.py)

Each bootstrap draw resamples original object IDs and retains both methods, every available participant and every photograph for each selected ID. The 35 primary objects each carry twenty paired observations. The sensitivity analysis covers 960 participant-photo observations on 600 unique photographs from 60 objects. Thirty-six objects occur in both participants, including one supplemental object. The other 24 occur in one participant each. Dividing by the observation count of each draw preserves the registered observation-weighted estimand. The paired intervals use 10,000 draws with seed 260632. The results concern two fixed participants, their session-specific menus, their fitted models and one diffusion seed per photograph. [Frozen analysis and sampling plan](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_images_design.json)

## Replay

Prepare the word models, sealed decisions and H/14 conditions as described in [the word study](FRESH_PEOPLE.md). Run from the repository root.

```bash
C=research/eeg_stack_20261112/code
P=outputs/fresh-people
python "$C/render_fresh_people_images.py" package
for s in sub-05 sub-01; do
  for menu in mix13 mix14 mix23 mix24; do
    python "$C/render_fresh_people_images.py" check --root "$P/$s/render_inputs" --subject "$s" --panel "$menu"
  done
done
```

Copy `fresh_people_blind_inputs.tar`, `fresh_people_image_code.tar` and the [Slurm wrapper](../research/eeg_stack_20261112/jobs/run_fresh_people_images.sbatch) into the cluster submit directory containing the official dictionary and adapter. Set `ARCHIVE` to the offline model archive. The submitted blind input SHA-256 is `d20939ac1fcf44320bdc5ea967e193726fbca81d7ade4513720cb6ac79d0b347`. Each worker checks the input and source archives on node-local storage, verifies the word gate against its hash, independently renders 240 JPEGs, and records the frozen independent CLIP logits. [Word gate result](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_words.json)

```bash
sbatch --export=ALL,ARCHIVE,CODE_SHA=eea3822e46abff440c0962d620a893420bc5a065553e9a12b902cd79ba7523dc run_fresh_people_images.sbatch
```

On the evaluation machine, save the returned archives as `outputs/fresh-people/cluster/<subject>-<menu>.tar`. Extract each archive under its participant's original input root. The scorer uses the verified original photograph archive and the pinned DINOv2-S/14 checkpoint.

```bash
for s in sub-05 sub-01; do
  for menu in mix13 mix14 mix23 mix24; do
    tar xf "$P/cluster/$s-$menu.tar" -C "$P/$s/render_inputs"
  done
done
python "$C/score_fresh_people_images.py"
```
