import torch

from paper_late_condition import LateCondition, make_step_tokens, response_gaps


def test_zero_initialization_and_frozen_first_two_conditions():
    model = LateCondition(width=6, rank=3, tokens=2, channels=4)
    tokens = torch.randn(4, 2, 4)
    corrections = model(torch.randn(4, 6), tokens)
    assert corrections.shape == (2, 4, 2, 4)
    assert all(torch.equal(value, tokens) for value in make_step_tokens(tokens, corrections))
    assert make_step_tokens(tokens, corrections)[0] is tokens


def test_condition_donor_only_controls_the_added_late_residual():
    original = torch.arange(12, dtype=torch.float32).view(4, 1, 3)
    delta = torch.ones_like(original)
    late = torch.stack((delta, 3 * delta))
    wrong_delta = late + 20
    own = make_step_tokens(original, late, selected=(False, True))
    wrong = make_step_tokens(original, wrong_delta, selected=(False, True))
    assert all(a is b is original for a, b in zip(own[:3], wrong[:3], strict=True))
    torch.testing.assert_close(own[-1] - original, late[1])
    torch.testing.assert_close(wrong[-1] - own[-1], torch.full_like(original, 20))


def test_signed_distillation_cancels_state_specific_shifts_and_rejects_reversal():
    true = torch.tensor([[3., 1.], [1., -1.]])
    donor = torch.tensor([[1., 3.], [-1., 1.]])
    offset = torch.tensor([[2., 4.], [8., -3.]])
    absolute, directional = response_gaps(true + offset, donor + offset, true, donor)
    assert absolute > 0
    assert directional == 0
    _, reversed_direction = response_gaps(donor, true, true, donor)
    assert reversed_direction > 0


def test_teacher_agreement_has_finite_loss_and_gradients():
    teacher = torch.zeros(2, 4)
    student = torch.ones(2, 4, requires_grad=True)
    absolute, directional = response_gaps(student, student, teacher, teacher)
    (absolute + directional).backward()
    assert torch.isfinite(absolute) and torch.isfinite(directional)
    assert torch.isfinite(student.grad).all()
