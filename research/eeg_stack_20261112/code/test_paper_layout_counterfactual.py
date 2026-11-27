import numpy as np
import torch

from paper_layout_counterfactual import concept_residual, fit_layout_response, layout_donors, mix_real_halves, pool_eeg, replace_layout


def test_fold_prediction_does_not_use_eeg_from_the_predicted_concept():
    groups = torch.arange(6).repeat_interleave(3)
    design = torch.tensor([[-1., 1.], [0., -2.], [1., 1.]]).repeat(6, 1)
    signal = torch.tensor([[4., 1.], [1., -2.]])
    eeg = (design @ signal + groups[:, None]).unsqueeze(0)
    expected = concept_residual(eeg, groups)
    predicted, _ = fit_layout_response(design, eeg, groups, folds=3, ridge=1e-5, calibrate=True)
    torch.testing.assert_close(predicted, expected, rtol=1e-4, atol=1e-4)

    edited = eeg.clone()
    edited[:, :3] += torch.tensor([[60., -12.], [0., 0.], [-60., 12.]])
    rerun, _ = fit_layout_response(design, edited, groups, folds=3, ridge=1e-5, calibrate=True)
    torch.testing.assert_close(rerun[:, :3], predicted[:, :3], rtol=0, atol=0)


def test_layout_donors_stay_in_each_fold_and_switch_concept():
    groups = np.arange(12).repeat(3)
    donors = layout_donors(groups, folds=3, rng=np.random.default_rng(48))
    assert np.array_equal(np.sort(donors), np.arange(len(groups)))
    assert np.all(groups[donors] != groups)
    assert np.array_equal(groups[donors] % 3, groups % 3)


def test_opposite_perturbation_matches_the_true_direction_in_size():
    layout = torch.as_tensor([[4., -2.], [-6., 2.]])
    eeg = layout + torch.as_tensor([[10., 9.], [5., 3.]])
    proper = replace_layout(eeg, layout, layout.flip(0), gain=1.)
    reversed_change = replace_layout(eeg, layout, layout.flip(0), gain=-1.)
    torch.testing.assert_close(proper - layout.flip(0), eeg - layout)
    torch.testing.assert_close(proper + reversed_change, 2. * eeg)


def test_same_object_mixture_uses_two_presentations_from_each_photo():
    trial_values = np.array([[2., 4., 8., 10.], [12., 14., 18., 20.]], dtype=np.float16)
    trials = np.broadcast_to(trial_values[:, :, None, None], (2, 4, 17, 250))
    a = pool_eeg(trials, np.array([0, 1]), group_size=2).unsqueeze(0)
    query = torch.tensor([0])
    donor = torch.tensor([1])
    again = mix_real_halves(a, query, query, query)
    across = mix_real_halves(a, query, query, donor)
    torch.testing.assert_close(again, torch.full_like(again, 6.))
    torch.testing.assert_close(across, torch.full_like(across, 11.))
