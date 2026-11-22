# October release evidence

The files here back the CPU verification and the October report. For the November benchmark, see the [current results](../docs/CURRENT_RESEARCH.md) and [stack experiment logs](../research/eeg_stack_20261112/results/). Run `python -m eeg_diffusion verify-stack` from the project root to check November Top-1 arithmetic.

- `retrieval.json`: selected original aggregate fields and training summaries, with source hash. Checkpoint file descriptors were omitted.
- `image_pairing.json`: original aggregate image metrics, paired intervals, query IDs and limitations, with source hash.
- `summary.json`: selected display data; v-parameterization and R4 values are explicitly labeled as rounded transcriptions from archived reports.
- `figure_data.csv`: plotted retrieval seeds, aggregate image values and v-parameterization contrasts; interval columns distinguish archived uncertainty from individual seed values.

No interval is recomputed from aggregates. No synthetic values substitute for missing EEG or images. The release verification checks integrity/arithmetic, not raw-event alignment, image feature extraction or model retraining.

