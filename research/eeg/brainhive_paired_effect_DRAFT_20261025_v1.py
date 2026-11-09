"""DRAFT mathematical primitive, without a trainer, admission, or image claim.

The caller must establish one identical, detached native Q for all responses,
and same-subject / same-OOF-role / different-concept pairing. This tensor
primitive cannot establish either fact. It is not integrated into any run.
"""
import math
import torch


def paired_effect_loss(student_correct, student_permuted, teacher_correct,
                       teacher_negative, *, epsilon):
    tensors=(student_correct,student_permuted,teacher_correct,teacher_negative)
    assert student_correct.ndim >= 2 and student_correct.shape[0] > 0
    assert all(t.shape==student_correct.shape for t in tensors)
    assert all(torch.isfinite(t).all() for t in tensors)
    assert math.isfinite(epsilon) and epsilon > 0
    axes=tuple(range(1,student_correct.ndim))
    target=(teacher_correct-teacher_negative).detach().float()
    student=student_correct.float()-student_permuted.float()
    separation=target.square().mean(axes)
    valid=separation>epsilon
    raw=(student-target).square().mean(axes)/(separation+epsilon)
    # Keep all registered queries in the denominator; expose invalid cases.
    per_query=torch.where(valid,raw,torch.zeros_like(raw))
    return per_query.mean(),dict(per_query=per_query,teacher_separation=separation,
        valid_teacher=valid,valid_fraction=valid.float().mean().detach(),
        integration_status='DRAFT_NO_SAME_Q_OR_IDENTITY_PROOF_NO_TRAINING_ENTRY')
