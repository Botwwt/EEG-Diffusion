# Matching one person's EEG to eight known-object examples

On a previously scored four-choice task, a classifier trained on same-session EEG episodes recovered **247/360** object labels, compared with **233/360** for the existing Fisher classifier. On a separate twelve-choice task it recovered **147/360**, compared with **144/360**. Across both tasks the gain was **17/720** participant–photograph decisions, or **2.36 percentage points** (paired 95% interval **0.00 to 4.72**). The registered advancement rule required the lower bound to exceed zero. These runs evaluated object labels for three fixed participants and 240 distinct photographs. [Design frozen before fitting](../research/eeg_stack_20261112/results/object_fidelity/episodic_object_design.json) · [Counts and paired intervals](../research/eeg_stack_20261112/results/object_fidelity/episodic_object.json)

| Single-person EEG classifier | One 12-object menu, hits / 360 | Three supplied four-object menus, hits / 360 |
|---|---:|---:|
| Existing 96-direction Fisher | 144 | 233 |
| Fisher with the mean of the menu's supports subtracted | 144 | 235 |
| Learned eight-support matcher | **147** | **247** |
| Matcher given another query's EEG in the same menu and fold | 33 | 59 |
| Classifier trained with deliberately incorrect labels | 4 | 13 |

For sub-04, sub-05 and sub-01 respectively, the matcher scored **56, 46, 45** on the twelve-choice panel and **87, 72, 88** on the four-choice panel, each out of 120. The matcher recovered 18 Fisher errors and lost 15 Fisher hits on the former, and recovered 22 errors and lost eight hits on the latter. Gains in percentage points were **0.83** (−2.78 to +3.89) and **3.89** (+0.56 to +7.50). Intervals resample the twelve true objects within each panel, keeping the ten photos and three people together. Several objects share a candidate menu and its support examples. The intervals condition on these participants and menus. [Session-matched task and candidate selection](COHORT_SESSION_CONTROLS.md)

Each person contributed four physical EEG observations per photograph. Each query had eight labeled support photographs for each supplied candidate, with both query photographs of the active fold held out. Within each supplied menu, all candidates had the same recording-session pair. The classifier used EEG from the person being tested. Its inputs were the query, the candidate menu and its support EEGs.

Training used each person's original 512 generic objects. The generic training episodes drew four or twelve alternatives from that person's same recording-session pair. For each episode, two Fisher projections trained on other object classes supplied 16 match scores. One projection used the historical Fisher contrast. In the second, the between-object covariance used class means after removing the mean of each generic session pair. Match scores included prototype cosine, proximity to two individual supports, squared distance, a variance-adjusted score and cosines in three Fisher bands. A regularized conditional softmax combined them. The frozen procedure fitted the weights on four object-disjoint folds. It reproduced all 720 historical individual Fisher decisions on the target panels.

In a separate internal check, 416 generic objects trained the procedure and 96 held-out objects supplied ten queries per person. In four-way episodes the matcher gained **45/2,880** decisions (+1.56 points, +0.42 to +2.74). In twelve-way episodes it gained **42/2,880** (+1.46 points, +0.38 to +2.57). Both menu sizes used the same 960 generic photographs per participant. For the two target panels the fixed procedure refitted on all 512. The old target-panel results were known before designing the matcher, so the target comparisons remain developmental.

The deliberately incorrect training labels excluded the true word by construction and can teach the classifier to avoid it. The wrong-query test used a fixed nonzero cyclic shift within each menu and fold. The relevant nominal chance rates for a uniformly chosen menu entry were 1/12 and 1/4. These manipulated controls check alignment. The comparison to the historical classifier measures the change in object accuracy.

## Replay

From the repository root, using the session-matched data and original 512-object compact EEG files described in the [input replay](COHORT_SESSION_CONTROLS.md), run the following commands. The optional `--subject sub-04` argument runs one participant. Summarizing requires all three.

```shell
python research/eeg_stack_20261112/code/fit_episodic_object.py fit
python research/eeg_stack_20261112/code/fit_episodic_object.py summarize
```

Per-participant models, row predictions and donor indices are under `outputs/episodic-object`. The compact report records the training and target receipts, prediction hashes, source hash and registered design hash. Summarization computes paired intervals from integer per-object hit differences so zero net hits give an exact zero bootstrap difference. [Implementation](../research/eeg_stack_20261112/code/fit_episodic_object.py)
