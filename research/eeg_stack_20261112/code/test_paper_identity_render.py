import torch
import numpy as np
from torch.testing import assert_close

from paper_identity_bank import class_sums, object_logits
from paper_identity_render import ConditionOffset, schedule, SOURCES, STEPS, BATCH


def test_identity_positive_uses_other_photographs_and_shares_all_negative_logits():
    photos = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])
    labels = torch.tensor([0, 0, 1, 1])
    sums = class_sums(photos, labels)
    generated = torch.tensor([[0., 1.]])
    a = object_logits(generated, photos[:1], labels[:1], sums, "identity")
    b = object_logits(generated, photos[:1], labels[:1], sums, "photo")
    assert_close(a, torch.tensor([[1., 0.]]))
    assert_close(b, torch.tensor([[0., 0.]]))


def test_object_class_gradient_reaches_the_generated_feature():
    photos = torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])
    labels = torch.tensor([0, 0, 1, 1])
    view = torch.tensor([[.7, .3]], requires_grad=True)
    loss = torch.nn.functional.cross_entropy(object_logits(view, photos[:1], labels[:1], class_sums(photos, labels),
                                                            "identity"), labels[:1])
    loss.backward()
    assert view.grad is not None and view.grad[0, 1].item() < 0.


def test_condition_updates_begin_at_native_and_are_bounded():
    model = ConditionOffset(torch.zeros(3), torch.ones(3))
    eeg = torch.tensor([[1., 2., 3.]])
    assert_close(model(eeg), eeg, atol=0, rtol=0)
    with torch.no_grad():
        model.offset.weight.fill_(100.)
    assert_close((model(eeg) - eeg).norm(), .5 * eeg.norm())


def test_schedule_pairs_every_source_with_distinct_photos_and_the_same_seeds_on_retry():
    rows, people, seeds = schedule(1200)
    assert rows.shape == people.shape == seeds.shape == (STEPS, BATCH)
    for i in range(len(SOURCES)):
        p = rows[people == i]
        assert len(p) == len(np.unique(p)) == STEPS // 2
    assert all(np.array_equal(old, new) for old, new in zip((rows, people, seeds), schedule(1200), strict=True))
