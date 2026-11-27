import torch

from paper_semantic_basis import predictable_basis


def test_basis_prioritizes_text_direction_supported_by_training_images():
    visual = torch.tensor([[-1.], [-1.], [1.], [1.]], dtype=torch.float64)
    state = torch.tensor([[-1., 10.], [-1., -10.], [1., 10.], [1., -10.]], dtype=torch.float64)
    gram = state @ state.T
    dual = predictable_basis(gram, visual, rank=1)
    decoded = dual.T @ state
    torch.testing.assert_close(decoded.abs(), torch.tensor([[1., 0.]], dtype=state.dtype))
    torch.testing.assert_close(dual.T @ gram @ dual, torch.ones(1, 1, dtype=state.dtype))
