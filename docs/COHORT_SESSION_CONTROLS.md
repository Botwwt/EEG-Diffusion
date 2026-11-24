# Three people viewing the same photograph

Pooling EEG from three people who each saw the same photograph four times classified **69/120** photographs in a 12-object panel where every object had exactly the same three-person recording-session signature. The prespecified individual comparator, sub-04, classified 55/120. The paired gain was 11.7 percentage points (95% object-bootstrap interval 4.2 to 19.2). On a separate set of new objects with four supplied alternatives per query, the cohort classified **93/120**, compared with 79/120 for sub-04. These are higher-acquisition-budget, known-object EEG decisions. Each cohort query consumes twelve physical presentations and produces one decision. [Same-session registration](../research/eeg_stack_20261112/results/object_fidelity/cohort_session_matched_design.json) · [12-way results](../research/eeg_stack_20261112/results/object_fidelity/cohort_session_stress12.json) · [Four-way results](../research/eeg_stack_20261112/results/object_fidelity/cohort_session_new4.json)

## Why session matching matters

The original stimulus schedule assigned all ten photographs of an object to the same pair of recording sessions for each person. The three pairs together give six session IDs. In the original fourth panel each of the twelve objects had a unique joint signature. A classifier given only those session IDs and the registered calibration labels could therefore identify **120/120** objects. The EEG combination scored **116/120** there. This panel cannot establish an object contribution beyond sessions. The same limitation applies to the third panel. The schedule oracle here is given the actual session IDs. The EEG model receives measured waveforms and can use any session information present in them. [Public presentation-order audit](../research/eeg_stack_20261112/results/object_fidelity/cohort_r4_sessions.json)

| Previously used 12-way panel | Three-person EEG | Fixed sub-04 | Mean single participant | Joint session only, ties shared | Number of jointly indistinguishable object pairs |
|---|---:|---:|---:|---:|---:|
| Registered | 112 | 86 | 76.0 | 100 | 2 |
| Replication | 116 | 94 | 80.7 | 110 | 1 |
| Third | 117 | 85 | 69.7 | 120 | 0 |
| Fourth | 116 | 81 | 77.3 | 120 | 0 |
| All 480 photographs | **461** | **346** | **303.7** | **450** | **3** |

For the principal single-person R4 task, the fourth panel supplied four occupied session pairs for each person. Session-only classification gave 40/120 expected hits per person. The three individuals actually obtained 81, 69 and 82, totaling 232/360 against a session-only expectation of 120/360. Holding EEG predictions fixed and permuting whole category names within each person's session tuple gave 113.67 expected hits across the three. These are 120 shared photographs evaluated three times. [Individual and joint session contrasts](../research/eeg_stack_20261112/results/object_fidelity/cohort_r4_sessions.json)

The four-panel joint conditional permutation kept every EEG prediction and recording tuple fixed. It yielded 435.5 expected hits against 461 observed. Only eight distinct permutations existed because just three class pairs shared a tuple. The pooled exact one-sided probability was 0.125. On the non-singleton subsets the original 12-way classifier obtained 37/40 and 18/20 on the first and second panels. A different scoring rule restricted choices to the two indistinguishable categories and gave 38/40 and 18/20. The prespecified fourth-panel image gate had passed on raw EEG word counts before this audit. Those counts are reported to preserve that chronology. [Original acquisition gate](../research/eeg_stack_20261112/results/object_fidelity/cohort_r4_design.json) · [Raw word results and condition hashes](../research/eeg_stack_20261112/results/object_fidelity/cohort_r4.json)

### Fourth-panel images, diagnostic only

The archived fourth-panel cohort output consists of 360 real JPEGs across own EEG, uniform support mean and exchanged EEG. Independent CLIP-L/14 recognized the correct name among the twelve panel labels on 116/120 images in every arm. Across all 1,854 names the respective counts were 95, 93 and 96. DINOv2 identified the viewed photograph among its ten same-object alternatives on 13, 8 and 11 of 120 images, respectively. On the 56 pairs where both EEG words were correct, strict two-photo assignment gave 33, 29 and 27 correct. The paired DINO ten-photo advantage over exchanged EEG was +1.67 points (−3.33 to +6.67). The shared prompt carries a predicted word from a panel where the joint recording schedule could identify all object labels. The high naming count here is a diagnostic observation. [Generated JPEG naming](../research/eeg_stack_20261112/results/object_fidelity/cohort_fourth_clip.json) · [Independent DINO and pixel measurements](../research/eeg_stack_20261112/results/object_fidelity/cohort_fourth_images.json)

## Two session-matched tasks

| Decision and data source | EEG presentations per participant | Participants' individual hits (04, 05, 01) | Cohort hits | Two other people deliberately misaligned | Session-only expected hits |
|---|---:|---|---:|---:|---:|
| Twelve objects in one joint-session cell | 4 | 55 / 45 / 44 | **69/120** | 36/120 | 10/120 |
| Three new blocks of four objects, block supplied | 4 | 79 / 67 / 87 | **93/120** | 53/120 | 30/120 |
| Same new photographs, single unrestricted twelve-object task | 4 | 67 / 64 / 70 | **92/120** | 41/120 | 30/120 |

The original 512-class participant model supplied every score. The one-cell 12-way panel drew twelve objects from the largest eligible cell in the 512 additions previously used to train a separate 1,024-class model. For each participant, all twelve target objects shared that participant's same two recording sessions. Ordinary chance and the session-only rule both yield 10/120. The cohort gained 17.5 points (11.9 to 23.3) over the mean individual. This is a stress test of the original model on objects that had already appeared elsewhere in the project.

The conservative pool for the newer calibration experiment contained 88 objects outside its 1,024 training classes. After protecting its separate twelve-object fifth panel, the other 76 had a largest joint-session cell of five. We registered three cells and sampled four new objects from each before acquiring their EEG. Telling the decoder the block leaves four equiprobable choices. The cohort gained 11.7 points (2.5 to 21.7) over sub-04 and 12.8 points (7.8 to 18.3) over the individual mean. The additional unrestricted 12-way measure on these same photos has 8.3% uniform chance but a 25% session-only baseline because the tuple identifies the block. The study did not use the protected fifth-panel EEG. Intervals resample twelve complete object categories and condition on these participants and chosen session cells. [Frozen selection rules, identities and folds](../research/eeg_stack_20261112/results/object_fidelity/cohort_session_matched_design.json)

## Measurements and generated-image inputs

Each participant had a separately fitted Fisher readout on 512 generic objects using 17 posterior electrodes. For each candidate object the classifier compared the query EEG to that person's eight support EEGs. It centered and standardized the candidate scores across the available alternatives before summing them equally across people. The negative control kept sub-04 aligned and cyclically shifted the two other people's score vectors among held-out queries in the same block and fold. The same stimulus and experimental design can produce common effects across participants. The analysis makes no assumption of independent neural noise.

For a cohort-generated image, each person's frozen EEG-to-H14 head recomputes the residual relative to the fused word's same eight supports. The common-direction estimate is the normalized sum of three unit tangents. The output angle is the arithmetic mean of the three separately calibrated native angles for that word. The wrong control exchanges the complete pooled condition with another query in the same word and fold and uses that donor's angle. A singleton keeps its original condition. The individual sub-04 comparison uses that person's own word and native direction. Both acquisitions have the same candidate menu, calibrated supports for each word and row-wise image seed. When the words agree, the actual support photographs and uniform condition are identical: 68/120 rows in the 12-choice panel and 86/120 in the four-choice panel. This is an end-to-end acquisition comparison.

The two matched panels each have a single-person and a three-person condition with three image arms, totaling 1,440 registered JPEG outputs. Downstream image scores evaluate a separate question from word recognition. Earlier O14 work already combined the conditions and starting latents of ten people observing identical official test photographs with 80 repeats per person. The present task uses three four-repeat measurements, an eight-support closed vocabulary and one output per row and arm. Its recognition measure and acquisition cost differ from the O14 official 200-image CLIP retrieval measure. [Image input receipts](../research/eeg_stack_20261112/results/object_fidelity/cohort_session_render_inputs.json) · [Prior O14 cohort protocol](../research/eeg_stack_20261112/O12_O13_O14_REPORT.md)

## Replay

Use the [posterior feature and stimulus archive](PHOTO_STRUCTURE.md#replay), the frozen first four panel registrations, the original public session-order schedule, and the [1,024-class acquisition registration](../research/eeg_stack_20261112/results/object_fidelity/double_calibration_design.json). The 12-choice stress test reads the three participants' existing verified compact feature stores from the separate calibration project. The four-choice panel fetches 36 pinned byte ranges of 5,040,000 bytes each and records SHA-256 hashes. The compact features average all four original slots in float32 for each photograph.

```bash
C=research/eeg_stack_20261112/code
export OPENBLAS_NUM_THREADS=4
python "$C/fuse_same_stimulus_r4.py" register
python "$C/fuse_same_stimulus_r4.py" run
python "$C/audit_cohort_r4_sessions.py"
python "$C/test_cohort_session_matched.py" register
python "$C/test_cohort_session_matched.py" fetch
python "$C/test_cohort_session_matched.py" score
python "$C/package_session_matched_cohort.py"
```

The image-input root is `outputs/session-matched-cohort/render_inputs`. Score each of `sub-04` and `cohort-3` separately with its matching `references/{panel}_{readout}_lock.json` file. The private per-query choices precede the aggregate calculations.
