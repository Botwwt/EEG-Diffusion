# Contributing

Use an issue or pull request with a minimal reproducible example, Python/package versions and the exact commit.

- Put portable tools in `src/eeg_diffusion/` and research-specific runtime code alongside its experiment. Keep new ports separate from the frozen sources in `experiments/archive/` and `research/eeg/`.
- Run `python -m pytest -q`, `python -m eeg_diffusion verify` and `python -m eeg_diffusion verify-stack`. Add a regression test when changing metric definitions, statistical units or numerical reference identities.
- Keep retrieval, semantic-vector and actual-image metrics separate. Record subjects, training seeds, physical repeats, candidate counts, target exposure and selection rules.
- A new method result needs matched information/sampling budgets and both positive and negative outcomes. Report external method attribution.
- Do not commit raw participant data, restricted stimuli, model weights, tokens or server receipts.

