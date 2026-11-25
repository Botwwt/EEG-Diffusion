# Using canonical EEG neighbors to transfer photograph appearance

A sub-04 EEG query retrieved photographs of other objects from the training set. We tested whether their appearance could improve a visual estimate in the native OpenCLIP ViT-H/14 coordinates accepted by the generator. The initial experiment stopped at the registered development gate. A separately registered experiment tested the same local appearance information after a person's own readout had already selected one of twelve calibrated object names. [Generic registration](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor_design.json) · [Personal12 registration](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor_personal12_design.json)

## Generic photograph retrieval

The training bank comprised 8,840 photographs from 884 objects. The 128 previously designated development objects supplied a separate 1,280-photo gallery, with four actual EEG presentations per query. Rank-32 canonical correlation analysis (CCA) scored the training photographs against each EEG query. A softmax with temperature 0.08 weighted the closest neighbors' unit H/14 vectors. Four concept-disjoint folds within the 884 training objects selected 128 neighbors and a 50% directional blend with the native ridge readout. The candidate and its angle-only control both used the same training-calibrated angle. A global prior supplies the same center to every row. This local prior varies with EEG, and supplies a continuous H/14 direction to the IP-Adapter. Earlier retrieved-latent experiments used training examples to supply a diffusion starting latent. [Estimator and selection](../research/eeg_stack_20261112/code/fit_canonical_neighbor.py) · [Earlier retrieval and starting-latent experiments](../research/eeg_stack_20261112/EXPERIMENT_HISTORY.md)

| Sub-04, 1,280 held photographs | Full-gallery top one | Mean reciprocal rank | Within-object top one | Pairwise wins within object | Target cosine |
|---|---:|---:|---:|---:|---:|
| Original native ridge | 7 | .02220 | 159 | 54.53% | .63342 |
| Native ridge with calibrated angle | 30 | .06405 | 264 | 63.11% | .50997 |
| Add 128 weighted neighbors | 29 | .06557 | 281 | 64.33% | .52757 |
| Exchange the local direction between photos of the same object | 29 | .06713 | 173 | 57.01% | .52826 |
| Add the single closest neighbor | 27 | .05445 | 254 | 61.95% | .50280 |
| Shuffle training photo identities in the 128-neighbor bank | 15 | .03762 | 241 | 59.44% | .48057 |

The candidate lost one identification against the angle-only control. The reciprocal-rank gain was .00152 (paired 95% interval −.00527 to +.00851). The within-object pair win rate rose by 1.22 percentage points (+0.55 to +1.90), and within-ten identification gained 17 hits (+1.33 points, −0.31 to +2.97). Intervals resampled the 128 objects together across methods. Pure CCA identified 47 in its own coordinate system. The registered full-gallery and reciprocal-rank requirements failed, so the official 200-photo R80 experiment stopped before prediction or generation. [Generic results and controls](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor.json)

## Support-centered update on personal12

This follow-up uses the previously studied sub-04 personal12 panel. Its frozen four-presentation EEG object readout had already classified 65 of 120 photographs correctly. We kept each predicted word and its eight designated supports, including for wrong words. The 884 training classes exclude all twelve target objects. For each query, the local feature is the estimated neighbor appearance minus the mean estimate from the eight support EEGs. We project that difference off the unit support-image prior, normalize it, and blend its direction equally with the original native tangent. Rotation from the prior uses the recipient's original angle. This changes how the same frozen word is drawn. [Frozen source protocol, formulas and scoring rule](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor_personal12_design.json)

The three actual draws per row use the original native condition, the matched update and a same-word, same-fold donor update. The donor swaps only the added tangent. The recipient retains their own native tangent, angle, prompt, radius, seed and eight supports. Of 120 rows, 107 have a nonself donor. The stricter same-true-object subset is defined only in evaluation. The zero-update control copies the native JPEG byte for byte. The renderer verifies the native directions and keeps the historical batch size of eight. It reports how many replayed native JPEGs match their historical bytes. [Condition builder](../research/eeg_stack_20261112/code/package_canonical_neighbor_personal12.py) · [Offline renderer](../research/eeg_stack_20261112/code/render_canonical_neighbor_personal12.py)

Before building this packet, we checked the support-centered formula on the same historical 128 generic objects, using known object membership to form each eight-support prior. Matched, native and exchanged conditions identified 28, 29 and 26 of 1,280 within-object photographs. Eight of each ten candidate photographs had contributed to that row's visual prior, placing them at a retrieval advantage. The matched change against native in pairwise win rate was +0.14 points (−0.09 to +0.36). Its mean target cosine gained .00057 (+.00044 to +.00071). This prespecified descriptive check supplied no tuning or promotion decision. [Generic check and sealed packet receipt](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor_personal12_packet.json)

The personal12 primary outcome is independent DINOv2-S/14 identification among the ten photographs of the true object. Advancement requires a positive lower paired 95% bound for matched versus native on all 120 rows, a positive matched-minus-exchanged point estimate on genuine same-object, same-word, same-fold nonself donors, and no decrease in independent CLIP-L/14 true-object naming across 1,854 words. The bootstrap resamples whole objects 10,000 times with seed 260574. Strict two-photo joint assignment among pairs where both words were correct is secondary. The twelve objects and this participant had been studied previously. [Independent image scorer](../research/eeg_stack_20261112/code/score_canonical_neighbor_images.py)

### Rendered photographs

For all 120 queries the native reconstruction reproduced the archived JPEG byte for byte. Replacing the native photo direction with the support-centered update raised DINOv2 identification among ten photographs of the object from 8 to 15/120. The paired difference was +5.83 percentage points (95% whole-object interval +0.83 to +10.83). An exchanged EEG update also reached 12/120. On the 39 rows with a different donor who saw another photograph of the same true object and shared the recipient's fold and predicted word, matched and exchanged EEG each retrieved 5/39. Full-vocabulary naming gave 55/120 native, 54/120 matched and 56/120 exchanged. On the 22 correctly decoded photo pairs, DINO assigned 16, 15 and 12 pairs correctly, respectively. The registered rule retained the native visual decoder. [All image scores and intervals](../research/eeg_stack_20261112/results/object_fidelity/canonical_neighbor_personal12_images.json) · [First archive row for each object](../research/eeg_stack_20261112/gallery/canonical_neighbor_personal12_sub04.jpg)

## Replay

Run from the repository root with the historical sub-04 R4 features for both training pools, the official training H/14 archive and row manifest, and the original mixed personal12 render inputs and native JPEG manifest. The registrations bind their hashes. The CPU stages use NumPy and SciPy. Thread counts below match the packet replay.

```bash
C=research/eeg_stack_20261112/code
P=outputs/canonical-neighbor/personal12
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/fit_canonical_neighbor.py" fit
python "$C/fit_canonical_neighbor.py" predict
python "$C/fit_canonical_neighbor.py" evaluate
python "$C/package_canonical_neighbor_personal12.py"
python "$C/render_canonical_neighbor_personal12.py" --root "$P/render_inputs" --check-only
```

The resulting `canonical-neighbor-personal12.tar` is the private 3,061,760-byte render input. Its SHA-256 is `cc05672fb9cd554f6bdab3d6f18260931cd62f596bae337d0f94aff3fac0042d`. The four script files and sealed condition, scale and protocol files contain no query photographs. The naming stage uses a separate code archive constructed with `scientific_code("score_canonical_neighbor_images")` from [the dependency packager](../research/eeg_stack_20261112/code/package_personal_confirmation.py). Copy both archives to the Slurm submit directory alongside the existing official dictionary and adapter. Set `ARCHIVE` to the offline generator archive.

The submitted naming archive was `d4f9af4ddf11c336d59a11473b36ea0bc93a13d1141191151479ecad6161b16e`. Its tar headers recorded local file times. This portable build uses the same 44 source files, in sorted order with zero timestamps. Their contents were checked against every submitted member. With the historical sources, the portable archive hashes to `d6571c2019376d4f4afe32e774c74c5144aa98ed65d30612fa2766fa0e71a683`.

```bash
PYTHONPATH="$C" python - <<'PY'
import io
import tarfile
from package_personal_confirmation import scientific_code

files = scientific_code("score_canonical_neighbor_images")
with tarfile.open("canonical-neighbor-naming.tar", "w") as archive:
    for name, path in sorted(files.items()):
        data = path.read_bytes()
        entry = tarfile.TarInfo(name)
        entry.size = len(data)
        archive.addfile(entry, io.BytesIO(data))
PY
```


```bash
export BUNDLE_SHA=$(sha256sum canonical-neighbor-personal12.tar | cut -d' ' -f1)
export SCORE_BUNDLE_SHA=$(sha256sum canonical-neighbor-naming.tar | cut -d' ' -f1)
sbatch --export=ALL,ARCHIVE,BUNDLE_SHA,SCORE_BUNDLE_SHA run_canonical_neighbor_personal12.sbatch
```

The [Slurm wrapper](../research/eeg_stack_20261112/jobs/run_canonical_neighbor_personal12.sbatch) checks both copied archives on node-local storage. The output `canonical_neighbor_personal12_images/sub-04.tar` contains the actual JPEGs, their hashes and the optional CLIP logits. On the evaluation host, extract it under the sealed input root and score against the independently indexed originals.

```bash
tar xf canonical_neighbor_personal12_images/sub-04.tar -C "$P/render_inputs"
python "$C/score_canonical_neighbor_images.py" score --root "$P/render_inputs"
```
