"""Synthetic CPU contracts against the pinned native processor, no weights/data."""
import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_functional_alignment_v2 import (
    NativeIPFunctionalAlignmentCapture, assert_engineering_signal,
    functional_alignment_loss,
)
from brainhive_response_rank_v1 import response_rank_loss
from check_BrainHIVE_response_rank_CPU_v1 import TinyUNet, native_processor_class

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_loss_contracts():
    # A true-vs-negative pair and an orthogonal error expose the rank-only
    # objective's exact blind direction; v2 must penalize that direction.
    positive = torch.tensor([1., 0.]).view(1, 1, 1, 2)
    negative = -positive
    orthogonal = positive + torch.tensor([0., 10.]).view(1, 1, 1, 2)
    correct, diagnostics = functional_alignment_loss(positive, positive, negative)
    wrong, _ = functional_alignment_loss(negative, positive, negative)
    far, _ = functional_alignment_loss(orthogonal, positive, negative)
    old_correct, _ = response_rank_loss(positive, positive, negative)
    old_far, _ = response_rank_loss(orthogonal, positive, negative)
    torch.testing.assert_close(old_correct, old_far, atol=1e-6, rtol=1e-6)
    assert correct < wrong and far > correct + 20
    assert diagnostics['positive_alignment'] == 0
    swapped_true, _ = functional_alignment_loss(negative, negative, positive)
    swapped_wrong, _ = functional_alignment_loss(positive, negative, positive)
    assert swapped_true < swapped_wrong
    torch.testing.assert_close(correct, swapped_true)
    for magnitude in (0., 1., 10., 100.):
        value, _ = functional_alignment_loss(magnitude * orthogonal, positive, negative)
        assert value >= 0 and torch.isfinite(value)
    # Identical teachers provide alignment but no discrimination; report it.
    degenerate, degenerate_info = functional_alignment_loss(positive, positive, positive)
    assert degenerate == 0 and degenerate_info['negative_discrimination'] == 0
    assert degenerate_info['near_zero_teacher_fraction'] == 1
    assert degenerate_info['nondegenerate_teacher_fraction'] == 0
    rejected = False
    try:
        assert_engineering_signal(degenerate_info, student_gradient_norm=1.,
                                  minimum_nondegenerate_teacher_fraction=.5)
    except AssertionError:
        rejected = True
    assert rejected, 'An all-degenerate teacher profile must not be admitted'
    # Even a far-away student must not be pulled by division through a near-zero
    # teacher separation. Empty masks must retain a zero-gradient graph.
    far_degenerate_student = orthogonal.clone().requires_grad_()
    far_degenerate, _ = functional_alignment_loss(far_degenerate_student, positive, positive)
    assert far_degenerate.requires_grad and far_degenerate == 0
    far_degenerate.backward()
    assert torch.equal(far_degenerate_student.grad, torch.zeros_like(far_degenerate_student))
    mixed_student = torch.cat([orthogonal, orthogonal]).requires_grad_()
    mixed_positive = torch.cat([positive, positive])
    mixed_negative = torch.cat([negative, positive + 1e-4])
    mixed_loss, mixed_info = functional_alignment_loss(mixed_student, mixed_positive, mixed_negative)
    mixed_loss.backward()
    assert mixed_info['positive_valid_count'] == 1 and mixed_info['near_zero_teacher_fraction'] == .5
    assert mixed_student.grad[0].norm() > 0 and torch.equal(mixed_student.grad[1], torch.zeros_like(mixed_student.grad[1]))
    student = torch.tensor([.3, .7]).view(1, 1, 1, 2).requires_grad_()
    teacher_positive = positive.clone().requires_grad_()
    teacher_negative = negative.clone().requires_grad_()
    value, info = functional_alignment_loss(student, teacher_positive, teacher_negative)
    value.backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert teacher_positive.grad is None and teacher_negative.grad is None
    assert_engineering_signal(info, student_gradient_norm=student.grad.norm(),
                              minimum_nondegenerate_teacher_fraction=1.)
    # P, PN and S must retain the SAME correct positive objective and mask. Only
    # the ranking teacher pair may change in S; its mask is computed separately.
    independent_positive = torch.tensor([0., 2.]).view(1, 1, 1, 2).requires_grad_()
    independent_negative = torch.tensor([0., -2.]).view(1, 1, 1, 2).requires_grad_()
    shared_student = torch.tensor([.3, .7]).view(1, 1, 1, 2).requires_grad_()
    p_loss, p_info = functional_alignment_loss(shared_student, positive, negative, mode='positive_only')
    pn_loss, pn_info = functional_alignment_loss(shared_student, positive, negative)
    s_loss, s_info = functional_alignment_loss(shared_student, positive, negative,
        rank_positive=independent_positive, rank_negative=independent_negative)
    assert p_info['negative_discrimination'] == 0
    assert torch.equal(p_info['positive_alignment'], pn_info['positive_alignment'])
    assert torch.equal(p_info['positive_alignment'], s_info['positive_alignment'])
    assert torch.equal(p_info['near_zero_teacher_fraction'], s_info['near_zero_teacher_fraction'])
    assert not torch.equal(pn_info['negative_discrimination'], s_info['negative_discrimination'])
    assert pn_info['independent_rank_pair'] == 0 and s_info['independent_rank_pair'] == 1
    _, rank_degenerate_info = functional_alignment_loss(shared_student, positive, negative,
        rank_positive=independent_positive, rank_negative=independent_positive)
    assert rank_degenerate_info['near_zero_teacher_fraction'] == 0
    assert rank_degenerate_info['rank_near_zero_teacher_fraction'] == 1
    assert rank_degenerate_info['negative_discrimination'] == 0
    assert torch.equal(rank_degenerate_info['positive_alignment'], p_info['positive_alignment'])
    s_loss.backward()
    assert shared_student.grad.norm() > 0
    assert independent_positive.grad is None and independent_negative.grad is None
    return dict(correct_pair_loss=float(correct), wrong_pair_loss=float(wrong),
                orthogonal_error_loss=float(far), old_rank_orthogonal_error_invisible=True,
                teacher_swap_reverses_preference=True, loss_nonnegative=True,
                degenerate_both_terms_masked_and_rejected=True,
                mixed_near_zero_teacher_gradient_exactly_zero=True,
                all_degenerate_graph_connected_zero=True,
                correct_positive_alignment_identical_P_PN_S=True,
                independent_rank_pair_and_mask_do_not_change_correct_positive=True,
                direct_and_independent_rank_teacher_gradients_absent=True)


def native_contracts():
    cls, native_sha = native_processor_class()
    unet = TinyUNet(cls(hidden_size=32, cross_attention_dim=8)).eval().requires_grad_(False)
    hidden = torch.randn(3, 37, 32, requires_grad=True)
    text = torch.randn(3, 5, 8)
    component = BoundedConditionAdapter(predict_risk=False)
    assert sum(p.numel() for p in component.parameters()) == 525568 and component.radius == .25
    raw = F.layer_norm(torch.randn(3, 1024), (1024,))
    changed, _ = component(raw, torch.ones_like(raw), torch.zeros_like(raw))
    assert torch.equal(raw, changed)
    projection = nn.Linear(1024, 32, bias=False).eval().requires_grad_(False)
    student = projection(changed).view(3, 4, 8)
    positive = torch.randn(3, 4, 8, requires_grad=True)
    negative = torch.randn(3, 4, 8, requires_grad=True)
    with torch.no_grad():
        reference = unet.a(hidden, text, student)
    collector = NativeIPFunctionalAlignmentCapture(unet, expected_layers=1)
    # An open capture may not build an auxiliary or survive its exception.
    rejected = False
    try:
        with collector.capture():
            unet.a(hidden, text, student)
            collector.auxiliary(student, positive, negative)
    except AssertionError:
        rejected = True
    assert rejected and not collector.active and not collector.queries
    kv_outputs = {'key': [], 'value': []}
    def save_gradient(name):
        def hook(module, inputs, output):
            if output.requires_grad:
                output.retain_grad()
                kv_outputs[name].append(output)
        return hook
    key_handle = unet.a.processor.to_k_ip.register_forward_hook(save_gradient('key'))
    value_handle = unet.a.processor.to_v_ip.register_forward_hook(save_gradient('value'))
    with collector.capture():
        observed = unet.a(hidden, text, student)
    assert torch.equal(reference, observed)
    assert all(not query.requires_grad for query in collector.queries.values())
    auxiliary, info = collector.auxiliary(student, positive, negative)
    assert not collector.active and not collector.queries
    hidden_gradient = torch.autograd.grad(auxiliary, hidden, retain_graph=True, allow_unused=True)[0]
    assert hidden_gradient is None
    student_gradient = torch.autograd.grad(auxiliary, student, retain_graph=True)[0]
    assert_engineering_signal(info, student_gradient_norm=student_gradient.norm(),
                              minimum_nondegenerate_teacher_fraction=1.)
    auxiliary.backward()
    assert component.direction.weight.grad is not None and component.direction.weight.grad.norm() > 0
    assert kv_outputs['key'][-1].grad is not None and kv_outputs['key'][-1].grad.norm() > 0
    assert kv_outputs['value'][-1].grad is not None and kv_outputs['value'][-1].grad.norm() > 0
    assert hidden.grad is None and positive.grad is None and negative.grad is None
    assert all(p.grad is None for p in unet.parameters())
    assert all(p.grad is None for p in projection.parameters())
    key_handle.remove()
    value_handle.remove()
    # Verify the optional rank-token API through the real native processor, not
    # only through the response-level algebra helper.
    positive_mode_student = student.detach().clone().requires_grad_()
    with collector.capture():
        unet.a(hidden, text, positive_mode_student)
    p_auxiliary, p_info = collector.auxiliary(positive_mode_student, positive, negative, mode='positive_only')
    assert p_info['negative_discrimination'] == 0
    torch.testing.assert_close(p_info['positive_alignment'], info['positive_alignment'])
    independent_positive = positive.detach().roll(1, 0).clone().requires_grad_()
    independent_negative = negative.detach().roll(2, 0).clone().requires_grad_()
    s_student = student.detach().clone().requires_grad_()
    with collector.capture():
        unet.a(hidden, text, s_student)
    s_auxiliary, s_info = collector.auxiliary(s_student, positive, negative,
        rank_positive_tokens=independent_positive, rank_negative_tokens=independent_negative)
    torch.testing.assert_close(s_info['positive_alignment'], info['positive_alignment'])
    assert s_info['independent_rank_pair'] == 1 and not collector.queries
    s_auxiliary.backward()
    assert s_student.grad.norm() > 0 and independent_positive.grad is None and independent_negative.grad is None
    with collector.capture(enabled=False):
        disabled = unet.a(hidden, text, student.detach())
    assert torch.equal(reference, disabled) and not collector.queries
    # Actual non-reentrant checkpoint backward must recompute the native
    # processor once with capture closed, while the auxiliary remains external.
    calls = [0]
    hidden2 = hidden.detach().clone().requires_grad_()
    student2 = student.detach().clone().requires_grad_()
    def closure(h, c):
        calls[0] += 1
        return unet.a(h, text, c)
    with collector.capture():
        output = checkpoint(closure, hidden2, student2, use_reentrant=False)
    auxiliary2, _ = collector.auxiliary(student2, positive, negative)
    (output.square().mean() + .1 * auxiliary2).backward()
    assert calls[0] == 2 and not collector.active and not collector.queries
    assert student2.grad is not None and student2.grad.norm() > 0
    assert positive.grad is None and negative.grad is None
    collector.close()
    return dict(native_processor_source_sha256=native_sha,
                native_forward_bit_exact=True, disabled_forward_bit_exact=True,
                open_capture_rejected_and_released=True, auxiliary_Q_gradient_absent=True,
                auxiliary_teacher_gradients_absent=True, student_key_and_value_path_gradients_nonzero=True,
                zero_initialized_direction_head_gradient_nonzero=True,
                native_P_PN_S_correct_positive_unchanged=True,
                native_independent_rank_token_gradients_absent=True,
                frozen_host_and_projection_gradients_absent=True,
                correction_parameters=525568, correction_radius=.25,
                checkpoint_native_calls=calls[0], checkpoint_capture_double_count=False,
                diagnostics={key: float(value) for key, value in info.items()})


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    return dict(status='FUNCTIONAL_ALIGNMENT_V2_CPU_CONTRACTS_PASSED_NO_FIT',
                torch_version=torch.__version__,
                loss_contracts=tensor_loss_contracts(), native_contracts=native_contracts(),
                source_sha256={name: digest(ROOT / name) for name in (
                    'brainhive_functional_alignment_v2.py', 'check_BrainHIVE_functional_alignment_CPU_v2.py',
                    'brainhive_response_rank_v1.py', 'brainhive_bounded_component_v1.py',
                    'check_BrainHIVE_response_rank_CPU_v1.py')},
                trained_parameters=0, GPU_used=False, actual_model_weights_read=False,
                EEG_or_image_data_read=False, real_engineering_profile_complete=False,
                image_gain_established=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_checks()
    if args.output:
        assert not args.output.exists(), 'Preserve historical receipts'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps(report, indent=2, allow_nan=False))
