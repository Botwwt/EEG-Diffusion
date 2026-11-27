import torch
from paper_token_evidence import token_parts, mix_tokens, adjust_part, ridge_weights

def test_exchanging_token_mean_preserves_key_routing_at_every_query():
    rng = torch.Generator().manual_seed(22)
    original, donor = torch.randn(2, 3, 4, 7, generator=rng)
    mean, spatial = token_parts(original)
    donor_mean, donor_spatial = token_parts(donor)
    switched = mix_tokens(donor_mean, spatial, original.dtype)
    key, value = torch.randn(2, 7, 5, generator=rng)
    query = torch.randn(3, 6, 5, generator=rng)
    original_logits = query @ (original @ key).transpose(1,2)
    switched_logits = query @ (switched @ key).transpose(1,2)
    torch.testing.assert_close(original_logits.softmax(-1), switched_logits.softmax(-1))
    baseline = original_logits.softmax(-1) @ (original @ value)
    actual = switched_logits.softmax(-1) @ (switched @ value)
    torch.testing.assert_close(actual - baseline, ((donor_mean-mean) @ value).expand_as(actual))
    torch.testing.assert_close(mix_tokens(mean, donor_spatial, original.dtype).mean(1, keepdim=True), mean)

def test_adjustment_preserves_half_the_native_evidence_and_is_rowwise():
    old = torch.tensor([[[-2.,0.]],[[4.,0.]]])
    own = torch.zeros(1,1,2)
    real = torch.tensor([[[1.,2.]]])
    predictions = torch.tensor([[[0.,3.]],[[0.,2.]]])
    actual = adjust_part(old, own, real, predictions)
    torch.testing.assert_close(actual, torch.tensor([[[0.,3.]],[[3.,4.]]]))
    torch.testing.assert_close(adjust_part(old, own, real, 17*predictions), actual)

def test_photo_and_object_regressions_share_the_same_direction_scale():
    x = torch.tensor([[-1.,-1.],[1.,-1.],[-1.,1.],[1.,1.]])
    y = torch.column_stack((2*x[:,0],-5*x[:,1]))
    weights = ridge_weights(x, y, .1)
    torch.testing.assert_close(x @ weights, y/1.1)
