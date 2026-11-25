# Does comparing every pair of candidate objects help four-trial EEG recognition?

For sub-04, a classifier that learned which EEG time windows distinguish each pair of candidate objects named **492/960** held-out photos correctly among twelve words. The frozen Fisher and trial-encoder average named **514/960**. The change was −2.29 percentage points (paired 95% object-bootstrap interval −4.06 to −0.52). The classifier gained 39 correct answers and lost 61. The preregistered screen stopped the study at the generic development objects. [Registration](../research/eeg_stack_20261112/results/object_fidelity/support_conditioned_object_design.json) · [Results](../research/eeg_stack_20261112/results/object_fidelity/support_conditioned_object.json)

| Supplied menu | Frozen Fisher + trial encoder | Same temporal weights across candidate pairs | Candidate-pair temporal weights | Exchanged query EEG |
| --- | ---: | ---: | ---: | ---: |
| 12 words | 514/960 | 491/960 | 492/960 | 27/960 |
| 4 words | 689/960 | 681/960 | 681/960 | 83/960 |

For the twelve-word menu, the candidate-specific model gained one hit over the control with shared temporal weights (95% interval 0 to +0.31 percentage points). The two new classifiers chose different words on one photo in each menu. The EEG exchange used a query from a different candidate object in the same historical menu and retained the original answer for scoring. Its count measures sensitivity to replacing the query in a particular direction.

## What the model compared

Each photograph supplied four real, 63-channel EEG trials from the first 500 milliseconds. We formed two non-overlapping trial pairs and retained five 100 ms windows with five signed time bins per electrode in each window. A learned projection mapped each window to 48 dimensions. Each class had eight EEG support photos. For every pair of candidate classes, the model compared the query to both support centers across complementary trial views. The temporal gate received the agreement of the two estimated class-separation directions, the within-class spread of the support EEG, and the disagreement between the query's two trial views. Five temporal weights then determined the signed vote for that class pair. Each target candidate received votes against every competitor.

The new term was added to the frozen Fisher and trial-encoder score with a scalar learned on training objects. The control had the same 15,215 parameters, examples, optimizer and number of updates. It averaged the inputs to its temporal gate across candidate pairs within the current query and menu. Both models saw all three ways to split four trials into two pairs and averaged those scores at evaluation. Earlier work in this repository compared a global prototype matcher and learned classifiers for the two closest competing supports. [Episodic classifier](EPISODIC_OBJECT.md) · [Earlier object classifiers](SINGLE_PERSON_OBJECT_ENCODERS.md)

## Split and replay

The readout used one participant, 416 training objects and 96 previously studied development objects from the same 512-class pool. Four inner folds each fitted the original Fisher and trial encoders on 312 training objects. The other 104 supplied class-disjoint baseline scores for training the added term. Each of the 750 updates drew twelve classes from a single inner fold and a single recording-session pair. Each photo served as a held-out query while the eight other photos outside its fold supplied the candidate centers. The registration preceded scoring the new classifiers. Their weights were sealed before preparing the 96-class features. All historical candidate sets, supports, exchanged rows and baseline decisions replayed. The paired intervals resample the 96 objects, keeping all ten photos per object. They pertain to this participant and these menus.

Prepare sub-04's 512-class physical training blocks, posterior EEG, session schedule, the existing episodic split and the original generic four-trial checkpoint and decisions. The scripts use NumPy, SciPy, PyTorch and safetensors. The archived run used PyTorch 2.9.1 with CUDA and four BLAS threads. The training inputs and the independently recomputed baseline features remain in the private workspace. [Trial and baseline preparation](SINGLE_PERSON_OBJECT_ENCODERS.md#replay) · [Feature preparation](../research/eeg_stack_20261112/code/prepare_support_conditioned_object.py) · [Classifier and evaluation](../research/eeg_stack_20261112/code/fit_support_conditioned_object.py)

```sh
C=research/eeg_stack_20261112/code
W=outputs/support-conditioned
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4
python "$C/prepare_support_conditioned_object.py" train --root "$W"
python "$C/fit_support_conditioned_object.py" train --root "$W"
python "$C/prepare_support_conditioned_object.py" held --root "$W"
python "$C/fit_support_conditioned_object.py" evaluate --root "$W" --release "$W/report.json"
cmp "$W/report.json" research/eeg_stack_20261112/results/object_fidelity/support_conditioned_object.json
```
