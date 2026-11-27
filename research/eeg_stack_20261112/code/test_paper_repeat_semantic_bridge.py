import torch

from paper_repeat_semantic_bridge import condition_coordinates, fit_head, predict, repeat_views, stable_coordinates, text_condition


def test_fit_uses_between_concept_signal_and_penalizes_photo_nuisance():
    homes = torch.tensor([[-2., 1.], [-1., -1.], [1., 1.], [2., -1.]])
    labels = torch.arange(4).repeat_interleave(3)
    design = homes[labels].clone()
    design[:, 1] += torch.tensor([-9., 0., 9.]).repeat(4)
    targets = homes[:, :1]
    fit = fit_head(design, targets, labels)
    same = predict(fit, torch.tensor([[2., -10.], [2., 10.]]))
    wrong = predict(fit, torch.tensor([[-2., 0.]]))
    assert torch.all(same > 1) and wrong.item() < -1
    torch.testing.assert_close(same[0], same[1], atol=0.15, rtol=0)


def test_repeat_views_partition_each_input_budget():
    for count in (4, 80):
        views = repeat_views(list(range(count)))
        assert sorted(sum(views, [])) == list(range(count))
        assert len({len(x) for x in views}) == 1


def test_repeat_filter_keeps_consistent_signal_and_shrinks_conflicts():
    query = torch.tensor([[2., 2.]])
    observations = torch.tensor([[[2., 2.]], [[2., -2.]], [[2., 2.]], [[2., -2.]]])
    result = stable_coordinates(query, observations, torch.ones(2))
    assert torch.equal(result, torch.tensor([[2., 0.]]))


def test_text_condition_uses_only_given_coordinates_and_frozen_states():
    states = dict(mean_hidden=torch.ones(2, 3), mean_pooled=torch.ones(4),
                  hidden_basis=torch.arange(12.).reshape(2, 2, 3),
                  pooled_basis=torch.arange(8.).reshape(2, 4))
    query = torch.tensor([[0., 0.], [0., 1.]])
    sequence, pool = text_condition(states, query)
    torch.testing.assert_close(sequence[0], states["mean_hidden"])
    torch.testing.assert_close(sequence[1], states["mean_hidden"] + states["hidden_basis"][1])
    torch.testing.assert_close(pool[1], states["mean_pooled"] + states["pooled_basis"][1])


def test_photo_donor_control_permutes_only_the_photo_learned_coordinates():
    photo = torch.arange(8.).reshape(4, 2)
    codes = {"photo-map": photo, "crossfit-map": photo + 10}
    donors = [2, 3, 0, 1]
    assert torch.equal(condition_coordinates("photo-map", codes, donors), photo)
    assert torch.equal(condition_coordinates("photo-wrong", codes, donors), photo[donors])
    assert torch.equal(condition_coordinates("soft-center", codes, donors), torch.zeros_like(photo))
