# Two-person replication of the expanded EEG object classifier

The expanded classifier recognized **281/700** strict twelve-way query decisions across sub-05 and sub-01. Their earlier personal classifiers recognized **253/700**. The change was **+4.00 percentage points** (paired 95% interval **+1.29 to +6.57**), comprising 72 corrected choices and 44 new errors. The expanded classifier used the sub-04 neural score weight of 0.59078 without refitting the weight in either participant. [Prior sub-04 experiment](FRESH_MIX.md) · [Replication registration](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_design.json) · [Word results and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_words.json)

| Strict query decisions | Original classifier | Expanded, transferred weight | Expanded, equal weight | Expanded, neural alone | Expanded, exchanged EEG |
|---|---:|---:|---:|---:|---:|
| sub-05, 350 | 118 | 135 | 137 | 128 | 16 |
| sub-01, 350 | 135 | 146 | 151 | 145 | 19 |
| **Combined, 700** | **253** | **281** | **288** | **273** | **35** |

Each participant saw the same 35 strict objects and the same ten original photographs per object as sub-04. The registration regrouped these objects by each new participant’s actual recording-session pair. Supplemental objects made each of the four menus twelve-way. Every query used its four original EEG presentations and eight different support photos per candidate. Both the trial encoder and Fisher projection were fitted on the participant’s own recordings from 1,024 external objects. The original comparator used that participant’s 512-object heads and their original equal-weight mixture. Equal weighting also performed well with the expanded models. The transferred weight remained fixed for the registered comparison.

The 95% interval samples the 35 object identities, retaining all ten photographs and both participants together. The result tests transfer to two more people on the fixed strict object set. The same-menu control exchanged query EEG with the next object’s held photograph and recognized 35/700. Including all supplemental queries gave **353 to 399 correct out of 960**, a difference of **+4.79 points** [2.42, 7.29], over 60 distinct object identities across the two people.

## Verified inputs and replay

The additional 512 physical object blocks per participant were checked against previously recorded full-block SHA-256 digests. Each new 48-object target packet was checked against two reads of every pinned public range. The full target waveforms and compact posterior and trial-encoder inputs stay under `outputs/fresh-people/<subject>/eeg`.

```sh
C=research/eeg_stack_20261112/code
python "$C/prepare_fresh_people.py" register
for S in sub-05 sub-01; do
  python "$C/prepare_fresh_people.py" extra --subject "$S" --workers 5
  python "$C/prepare_fresh_people.py" target --subject "$S" --workers 5
  python "$C/fit_fresh_people.py" train --subject "$S"
  python "$C/fit_fresh_people.py" score --subject "$S"
done
python "$C/fit_fresh_people.py" summarize
python "$C/package_fresh_people.py"
```

Both participants met the nonnegative individual word-gain requirement, and both registered paired lower bounds were positive. The gated export produced sixteen condition packets covering two people, two classifiers and four menus per person. The H/14 visual decoder remained calibrated on each person’s original 512 objects. On the 313/480 same-word decisions for sub-05 and 318/480 for sub-01, the selected support IDs and own and uniform visual vectors matched exactly between classifiers. The rendering archive contains 40 files and has SHA-256 `d20939ac1fcf44320bdc5ea967e193726fbca81d7ade4513720cb6ac79d0b347`. The word evidence and the image-condition parity concern different outcomes from recognition of the generated JPEGs. [Acquisition](../research/eeg_stack_20261112/code/prepare_fresh_people.py) · [Training and scoring](../research/eeg_stack_20261112/code/fit_fresh_people.py) · [Paired image conditions](../research/eeg_stack_20261112/code/package_fresh_people.py)
