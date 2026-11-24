# Which photographs win a within-object comparison?

We revisited the third R4 panel's frozen 1,080 JPEGs. For each of 360 participant–photograph queries, we ranked all ten original photographs of the true object and attributed the winner to the viewed photograph, its one other held-out photograph in that fold, or the eight photographs from the other folds. We re-encoded the JPEGs using the pinned DINOv2-S/14 model and verified all 1,080 original per-query top-1 credits. We performed the same calculation before generation using each frozen H14 conditioning vector and the same 120 original photograph identities. [Complete counts, cosine margins, object intervals and feature digests](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_error_routes.json)

## Where top-1 errors go

Each cell gives the number of winners in the order **viewed photograph / other held-out photograph / eight other-fold photographs**. When the predicted object word is correct, those eight are the actual supports used by the conditioning model. On a wrong-word query all eight actual supports belong to the predicted object and fall outside the true-object gallery.

| Measurement | Condition | Correct word, 209 queries | Wrong word, 151 queries |
|---|---|---:|---:|
| Generated JPEG, DINO | Own EEG | 19 / 17 / 173 | 25 / 19 / 107 |
| Generated JPEG, DINO | Uniform prior | 28 / 22 / 159 | 23 / 18 / 110 |
| Generated JPEG, DINO | Wrong EEG | 22 / 17 / 170 | 21 / 16 / 114 |
| Input condition, H14 | Own EEG | 9 / 3 / 197 | 20 / 20 / 111 |
| Input condition, H14 | Uniform prior | 11 / 8 / 190 | 14 / 19 / 118 |
| Input condition, H14 | Wrong EEG | 8 / 5 / 196 | 16 / 14 / 121 |

The eight supports have eight chances to win. For correct-word, own-EEG queries, one support won 10.35% of all DINO comparisons on average. The other held-out photograph won 8.13%. The paired difference was +2.21 percentage points (95% interval −1.34 to +5.80). The corresponding mean DINO cosines were 0.6504 and 0.6339, a difference of +0.0165 (+0.0034 to +0.0340). In the conditioning space, each support won 11.78% of comparisons and the held-out alternative won 1.44%, a difference of +10.35 points (+7.9 to +12.2). The respective mean H14 cosines were 0.8538 and 0.8266. This H14 comparison directly includes the eight embeddings used to construct the prior. Its clear support preference persisted weakly in DINO cosine, with an inconclusive difference in DINO top-1 winner rates. The audit supplies no basis for a claim that the generator copies a particular support. [All conditions and denominators](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_error_routes.json)

## Viewed photograph versus the other held-out photograph

Each query receives a decision using its own predicted word, irrespective of the other photograph's word. On the 360 queries the counts for own/uniform/wrong were **189/176/182** in DINO and **194/175/173** in H14. On the 209 correct-word queries they were **107/95/102** and **112/104/96**, respectively. On the 151 wrong-word queries they were **82/81/80** and **82/71/77**.

For DINO, own EEG improved the mean viewed-minus-alternative cosine margin by 0.00894 over uniform (0.00133 to 0.01743) and 0.00959 over wrong EEG (0.00190 to 0.01715). The corresponding two-choice accuracy intervals included zero for both comparisons. For H14, the two-choice gains were 5.28 points over uniform (1.67 to 9.17) and 5.83 over wrong EEG (0.56 to 11.39). H14 is the model's own visual representation. Comparisons with DINO use different encoders and cosine scales. These post-hoc intervals resample the 12 shared concepts and keep the three participants fixed. The true-object gallery is known during evaluation. The audit evaluates photograph identity conditional on that gallery. [Per-object two-choice counts and within-encoder paired contrasts](../research/eeg_stack_20261112/results/object_fidelity/fresh_photo_error_routes.json)

## Replay

Complete the [fresh panel replay](FRESH_PHOTO_REPLAY.md), retaining its original-image DINO cache, JPEGs, per-query report, H14 condition packets and author image embeddings. The extract stage uses the same pinned DINO weights, preprocessing, device and batch size. It verifies each JPEG hash and stores normalized generated-image CLS features locally. The CPU stage checks their hashes against the render metadata, validates the original-image CLS cache and reproduces every previously published per-query success or failure before attributing the competitors. Only aggregate counts and receipts enter the public file.

```bash
C=research/eeg_stack_20261112/code
"$DINO_PY" "$C/audit_fresh_photo_errors.py" extract
python "$C/audit_fresh_photo_errors.py" audit
```
