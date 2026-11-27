import torch

from paper_stage_reference import BASE, REFERENCE_STEPS, TIMESTEPS, condition_plan, step_tokens


def test_photo_reference_changes_only_its_registered_step():
    sources = {"eeg4": torch.arange(3), "eeg76": torch.arange(3) + 10, "photo": torch.arange(3) + 20}
    for step, arm in enumerate(REFERENCE_STEPS):
        plan = condition_plan(arm)
        assert plan[step] == "photo"
        assert sum(source == "photo" for source in plan) == 1
        applied = step_tokens(plan, sources, [2, 0, 1])
        assert all(value is sources["eeg4"] for i, value in enumerate(applied) if i != step)
        torch.testing.assert_close(applied[step], sources["photo"])
    assert condition_plan(BASE) == ("eeg4",) * len(TIMESTEPS)


def test_high_repeat_and_other_photo_controls_have_separate_provenance():
    sources = {"eeg4": torch.tensor([0, 1]), "eeg76": torch.tensor([10, 11]), "photo": torch.tensor([20, 21])}
    assert all(value is sources["eeg76"] for value in step_tokens(condition_plan("eeg76-all"), sources, [1, 0]))
    other = step_tokens(condition_plan("other-photo-all"), sources, [1, 0])
    for value in other:
        torch.testing.assert_close(value, torch.tensor([21, 20]))
