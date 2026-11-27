import numpy as np
import torch

from paper_background_pairs import border_descriptor, photo_pairs
from paper_object_foreground import assignment_loss, closest_other_context
from paper_object_semantic_timing import image_tokens_by_step


def test_border_proxy_does_not_measure_center_pixels():
    dark = np.zeros((16, 16, 3), dtype=np.uint8)
    center = dark.copy()
    center[4:12, 4:12] = 255
    edges = dark.copy()
    edges[:4] = 255
    np.testing.assert_array_equal(border_descriptor(center), border_descriptor(dark))
    assert np.linalg.norm(border_descriptor(edges) - border_descriptor(dark)) > 0


def test_context_matching_keeps_all_photos_and_increases_background_gap():
    members = np.array([[0, 1, 2, 3], [4, 5, 6, 7]])
    context = np.array([0., 50., 51., 1., 0., 50., 51., 1.])[:, None]
    reference = photo_pairs(members, context, 33, distant=False)
    paired = photo_pairs(members, context, 33, distant=True)
    np.testing.assert_array_equal(np.sort(reference.reshape(2, -1), axis=1), members)
    np.testing.assert_array_equal(np.sort(paired.reshape(2, -1), axis=1), members)
    paired_gap = np.square(context[paired[:, :, 0]] - context[paired[:, :, 1]]).sum()
    reference_gap = np.square(context[reference[:, :, 0]] - context[reference[:, :, 1]]).sum()
    assert paired_gap > reference_gap


def test_nearest_context_never_uses_an_equal_object_target():
    feature = torch.tensor([[0.], [.1], [.2], [4.]])
    labels = torch.tensor([0, 0, 1, 2])
    match = closest_other_context(feature, labels, use_context=True)
    assert match.tolist() == [2, 2, 1, 2]
    assert torch.all(labels[closest_other_context(feature, labels, use_context=False)] != labels)


def test_assignment_prefers_the_correct_pairs_over_the_swapped_pairs():
    targets = torch.eye(3)
    donor = torch.tensor([1, 2, 0])
    own = assignment_loss(targets, targets, donor)
    wrong = assignment_loss(targets, targets[donor], donor)
    assert own < .001 < wrong


def test_early_image_scaling_preserves_the_last_two_calls():
    native = torch.arange(12.).reshape(3, 4).half()
    early, second, late, final = image_tokens_by_step(native, .25)
    assert torch.equal(early, native * .25) and torch.equal(early, second)
    assert torch.equal(late, native) and torch.equal(final, native)
