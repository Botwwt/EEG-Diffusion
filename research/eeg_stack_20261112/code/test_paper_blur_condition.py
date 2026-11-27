import torch

from paper_blur_condition import BlurCondition, combine_views, object_logits, tangent


def test_sharp_control_has_the_same_input_in_every_view():
    views = torch.tensor([[[1., 3.], [5., 7.], [9., 11.]]])
    weights = torch.zeros(3, 2)
    torch.testing.assert_close(combine_views(views, weights, "sharp"), views[:, 0])
    torch.testing.assert_close(combine_views(views, weights, "multiscale"), views.mean(1))


def test_native_condition_lives_in_the_training_cone_tangent():
    axis = torch.tensor([1., 0., 0.])
    feature = torch.tensor([[4., 3., -4.], [1., -5., 12.]])
    result = tangent(feature, axis)
    torch.testing.assert_close(result @ axis, torch.zeros(2))
    torch.testing.assert_close(result.norm(dim=-1), torch.ones(2))


def test_object_target_removes_only_the_current_photograph():
    pred = torch.tensor([[1., 0.], [0., -1.]])
    photos = torch.tensor([[1., 0.], [0., -1.]])
    sums = torch.tensor([[1., 2.], [3., -1.]])
    expected = torch.tensor([[0., 3. / (10. ** .5)], [-2. / (5. ** .5), 0.]])
    torch.testing.assert_close(object_logits(pred, photos, sums, torch.tensor([0, 1])), expected)


def test_single_query_direction_is_independent_of_other_queries():
    torch.manual_seed(2)
    means = torch.zeros(2, 3, 12)
    scales = torch.ones(2, 3, 1)
    axis = torch.nn.functional.normalize(torch.ones(8), dim=0)
    model = BlurCondition(means, scales, axis, views=3).eval()
    waveforms, people = torch.randn(4, 3, 12), torch.tensor([0, 1, 0, 1])
    together = model(waveforms, people)
    single = model(waveforms[2:3], people[2:3])
    torch.testing.assert_close(together[2:3], single, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(together @ axis, torch.zeros(4), atol=1e-6, rtol=1e-5)
