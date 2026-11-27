import torch

from paper_token_mean_timing import step_conditions
from paper_token_evidence import token_parts


def test_native_step_preserves_bit_exact_tokens():
    native = torch.randn(6, 4, 23).half()
    fixed = torch.randn(1, 1, 23)
    for t in step_conditions(native, fixed, "native-paper", range(len(native))):
        assert torch.equal(t, native)


def test_timing_removes_only_query_mean_in_selected_steps():
    native = torch.randn(5, 4, 16).half()
    mean, relative = token_parts(native)
    fixed = torch.randn(1, 1, 16)
    early = step_conditions(native, fixed, "fixed-mean-early", range(len(native)))
    late = step_conditions(native, fixed, "fixed-mean-late", range(len(native)))
    always = step_conditions(native, fixed, "fixed-mean-all", range(len(native)))
    for t, goal in zip(early, [fixed, fixed, mean, mean], strict=True):
        center, change = token_parts(t)
        torch.testing.assert_close(center, goal.expand_as(center), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(relative, change, atol=1e-3, rtol=1e-3)
    assert all(torch.equal(a,b) for a,b in zip(early,late[::-1],strict=True))
    assert all(torch.equal(t,early[0]) for t in always)


def test_swapping_relative_tokens_preserves_recipient_mean_on_late_calls():
    native = torch.randn(8, 4, 11).half()
    fixed = torch.randn(1, 1, 11)
    lost = list(range(4,8)) + list(range(4))
    actual, _ = token_parts(native)
    for t, goal in zip(step_conditions(native, fixed, "fixed-mean-early-wrong", lost),
                       (fixed, fixed, actual, actual), strict=True):
        center, relative = token_parts(t)
        torch.testing.assert_close(center, goal.expand_as(center), atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(relative,token_parts(native)[1][lost],atol=1e-3,rtol=1e-3)
