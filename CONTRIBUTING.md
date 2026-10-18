# Contributing

Use an issue or pull request with a minimal reproducible example, Python/package versions and the exact commit.

- Keep historical files in `experiments/archive/` unchanged. Put ports/refactors in separate modules and document their differences.
- Add tests when changing metric definitions, statistical units or numerical reference identities. Run `python -m pytest -q` and `eeg-diffusion verify`.
- Keep retrieval, semantic-vector and actual-image metrics separate. Record subjects, training seeds, physical repeats, candidate counts, target exposure and selection rules.
- A new method result needs matched information/sampling budgets and both positive and negative outcomes. Report external method attribution.
- Do not commit raw participant data, restricted stimuli, model weights, tokens or server receipts.

