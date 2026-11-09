"""Bounded-below positive alignment plus discrimination at native IP responses.

This changes only the auxiliary objective. It reuses the actual v1 observation
API, native shared/detached Q, and IP SDP branch. No processor, host weight,
condition head, timestep distribution, or inference path is changed here.
"""
import math

import torch
import torch.nn.functional as F

from brainhive_response_rank_v1 import NativeIPResponseCapture, ip_response


def functional_alignment_loss(student, positive, negative, epsilon=1e-6, *,
                              mode='full', rank_positive=None, rank_negative=None):
    """Correct positive alignment plus optional, independently paired ranking.

    Correct teachers define S0=stopgrad(mean((positive-negative)^2)), v0=S0>eps.
    Lpos=mean_v0(mean((student-positive)^2)/(S0+eps)). This term, its scale and
    its mask are identical for positive_only (P), full (PN), and shuffled (S).

    PN ranks the correct pair; S may supply an independent rank-positive AND
    rank-negative pair without changing the correct positive alignment. For
    that pair Sr=stopgrad(mean((rank_positive-rank_negative)^2)), vr=Sr>eps;
    Lrank=mean_vr(softplus((Drpos-Drneg)/(Sr+eps))). Full loss=Lpos+Lrank.

    Empty valid sets produce graph-connected zero. BOTH terms are masked with
    their own detached validity; invalid denominators are replaced before any
    division, so near-zero teachers cannot dominate positive gradients. Every
    teacher, mask and scale is detached; only the student receives gradients.
    """
    assert student.ndim == 4 and student.shape == positive.shape == negative.shape
    assert epsilon == 1e-6 and student.numel() > 0
    assert mode in ('positive_only', 'full')
    assert (rank_positive is None) == (rank_negative is None)
    if mode == 'positive_only':
        assert rank_positive is None, 'P must not silently consume a shuffled rank teacher'
    assert torch.isfinite(student).all()
    positive, negative = positive.detach().float(), negative.detach().float()
    assert torch.isfinite(positive).all() and torch.isfinite(negative).all()
    dimensions = tuple(range(1, student.ndim))
    separation = (positive - negative).square().mean(dimensions).detach()
    valid = (separation > epsilon).detach()
    weight = valid.float()
    valid_count = weight.sum()
    safe_scale = torch.where(valid, separation + epsilon, torch.ones_like(separation))
    d_positive = (student.float() - positive).square().mean(dimensions)
    d_negative = (student.float() - negative).square().mean(dimensions)
    normalized_positive = (d_positive / safe_scale) * weight
    alignment = normalized_positive.sum() / valid_count.clamp_min(1)
    explicit_rank_pair = rank_positive is not None
    if rank_positive is None:
        rank_positive, rank_negative = positive, negative
    else:
        assert rank_positive.shape == rank_negative.shape == student.shape
        rank_positive, rank_negative = rank_positive.detach().float(), rank_negative.detach().float()
        assert torch.isfinite(rank_positive).all() and torch.isfinite(rank_negative).all()
    rank_separation = (rank_positive - rank_negative).square().mean(dimensions).detach()
    rank_valid = (rank_separation > epsilon).detach()
    rank_weight = rank_valid.float()
    rank_valid_count = rank_weight.sum()
    rank_scale = torch.where(rank_valid, rank_separation + epsilon, torch.ones_like(rank_separation))
    rank_d_positive = (student.float() - rank_positive).square().mean(dimensions)
    rank_d_negative = (student.float() - rank_negative).square().mean(dimensions)
    margin = torch.where(rank_valid, (rank_d_positive - rank_d_negative) / rank_scale,
                         torch.zeros_like(rank_separation))
    if mode == 'full':
        discrimination = (F.softplus(margin) * rank_weight).sum() / rank_valid_count.clamp_min(1)
    else:
        discrimination = student.float().sum() * 0.  # Graph-connected P-only zero.
    total = alignment + discrimination
    assert torch.isfinite(total) and total >= 0
    diagnostics = dict(
        functional_loss=total.detach(), positive_alignment=alignment.detach(),
        negative_discrimination=discrimination.detach(),
        ranking_enabled=student.new_tensor(float(mode == 'full')).detach(),
        independent_rank_pair=student.new_tensor(float(explicit_rank_pair)).detach(),
        query_count=student.new_tensor(float(len(student))).detach(),
        positive_valid_count=valid_count.detach(), rank_valid_count=rank_valid_count.detach(),
        d_positive=d_positive.detach().mean(), d_negative=d_negative.detach().mean(),
        teacher_separation=separation.mean(), minimum_teacher_separation=separation.min(),
        maximum_teacher_separation=separation.max(),
        near_zero_teacher_fraction=(~valid).float().mean(),
        nondegenerate_teacher_fraction=valid.float().mean(),
        rank_teacher_separation=rank_separation.mean(),
        minimum_rank_teacher_separation=rank_separation.min(),
        maximum_rank_teacher_separation=rank_separation.max(),
        rank_near_zero_teacher_fraction=(~rank_valid).float().mean(),
        rank_nondegenerate_teacher_fraction=rank_valid.float().mean(),
        normalized_positive_error=alignment.detach(),
        valid_normalized_margin=(margin.detach() * rank_weight).sum() / rank_valid_count.clamp_min(1),
        teacher_positive_response_energy=positive.square().mean(),
        teacher_negative_response_energy=negative.square().mean(),
        rank_positive_response_energy=rank_positive.square().mean(),
        rank_negative_response_energy=rank_negative.square().mean(),
    )
    return total, diagnostics


class NativeIPFunctionalAlignmentCapture(NativeIPResponseCapture):
    """Drop-in capture()/auxiliary() interface with a NEW v2 loss only.

    The inherited capture closes before this auxiliary is built; checkpoint
    recomputation therefore cannot capture twice or append an auxiliary graph.
    Queries are detached again defensively and released even on an exception.
    """
    def auxiliary(self, student_tokens, positive_tokens, negative_tokens, *,
                  mode='full', rank_positive_tokens=None, rank_negative_tokens=None):
        assert not self.active and set(self.queries) == set(self.layers)
        assert student_tokens.requires_grad
        assert student_tokens.shape == positive_tokens.shape == negative_tokens.shape
        assert self.layers, 'No native IP layers were observed'
        assert mode in ('positive_only', 'full')
        assert (rank_positive_tokens is None) == (rank_negative_tokens is None)
        if mode == 'positive_only':
            assert rank_positive_tokens is None
        if rank_positive_tokens is not None:
            assert rank_positive_tokens.shape == rank_negative_tokens.shape == student_tokens.shape
        losses, values_by_key = [], {}
        try:
            for name, (attention, processor) in self.layers.items():
                query = self.queries[name].detach()
                assert not query.requires_grad and query.shape[0] == student_tokens.shape[0]
                assert all(not p.requires_grad for p in processor.parameters())
                student = ip_response(query, student_tokens, processor, attention.heads)
                with torch.no_grad():
                    positive = ip_response(query, positive_tokens.detach(), processor, attention.heads)
                    negative = ip_response(query, negative_tokens.detach(), processor, attention.heads)
                    if rank_positive_tokens is not None:
                        rank_positive = ip_response(query, rank_positive_tokens.detach(), processor, attention.heads)
                        rank_negative = ip_response(query, rank_negative_tokens.detach(), processor, attention.heads)
                    else:
                        rank_positive = rank_negative = None
                loss, diagnostics = functional_alignment_loss(student, positive, negative,
                    mode=mode, rank_positive=rank_positive, rank_negative=rank_negative)
                losses.append(loss)
                for key, value in diagnostics.items():
                    values_by_key.setdefault(key, []).append(value)
        finally:
            self.queries.clear()
        result = torch.stack(losses).mean()
        diagnostics = {}
        for key, values in values_by_key.items():
            stacked = torch.stack(values)
            if key in ('minimum_teacher_separation', 'minimum_rank_teacher_separation'):
                diagnostics[key] = stacked.min()
            elif key in ('maximum_teacher_separation', 'maximum_rank_teacher_separation'):
                diagnostics[key] = stacked.max()
            elif key in ('positive_valid_count', 'rank_valid_count'):
                diagnostics[key] = stacked.sum()  # Valid query-layer pairs.
            else:
                diagnostics[key] = stacked.mean()
        diagnostics['layer_count'] = result.new_tensor(float(len(self.layers))).detach()
        diagnostics['query_layer_pairs'] = result.new_tensor(float(len(self.layers) * len(student_tokens))).detach()
        assert torch.isfinite(result)
        return result, diagnostics


def assert_engineering_signal(diagnostics, *, student_gradient_norm,
                              minimum_nondegenerate_teacher_fraction):
    """Explicit profile admission, not a claim about a trained model's quality.

    The parent recipe must declare the valid-pair fraction before a real profile.
    The gradient norm must be from THIS auxiliary alone, not total training loss.
    This routine does no backward, optimizer, model or data access itself.
    """
    threshold = float(minimum_nondegenerate_teacher_fraction)
    assert 0 < threshold <= 1
    values = {key: float(value) for key, value in diagnostics.items()}
    assert values and all(math.isfinite(value) for value in values.values())
    assert values['nondegenerate_teacher_fraction'] >= threshold, (
        'Too few teacher pairs have distinct native responses', values)
    assert values['teacher_separation'] > 1e-6, 'Teacher responses are degenerate'
    if values['ranking_enabled']:
        assert values['rank_nondegenerate_teacher_fraction'] >= threshold, (
            'Too few rank teacher pairs have distinct native responses', values)
        assert values['rank_teacher_separation'] > 1e-6
    assert values['positive_alignment'] >= 0 and values['negative_discrimination'] >= 0
    gradient = float(student_gradient_norm)
    assert math.isfinite(gradient) and gradient > 0, 'Auxiliary has no finite nonzero student gradient'
    return dict(status='NONDEGENERATE_FUNCTIONAL_AUXILIARY_ENGINEERING_SIGNAL',
                minimum_nondegenerate_teacher_fraction=threshold,
                measured_nondegenerate_teacher_fraction=values['nondegenerate_teacher_fraction'],
                auxiliary_student_gradient_norm=gradient,
                model_fit=False, image_gain_established=False)
