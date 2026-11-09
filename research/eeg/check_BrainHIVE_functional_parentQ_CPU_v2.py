"""Independent two-forward parent-Q contracts with actual native processors.

Two native TinyUNet attention blocks are composed so the downstream Q really
depends on earlier conditioning. Every observed Q comes from the uncorrected
parent pass. Student passes and their real checkpoint recomputation execute
outside ALL capture contexts. No weights/data, optimizer, SSH or GPU is used.
"""
import argparse
import copy
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
    NativeIPFunctionalAlignmentCapture, functional_alignment_loss,
)
from brainhive_response_rank_v1 import ip_response
from check_BrainHIVE_response_rank_CPU_v1 import TinyUNet, native_processor_class

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


class TwoBlockNativeTinyUNet(nn.Module):
    def __init__(self, processor_class):
        super().__init__()
        self.first = TinyUNet(processor_class(hidden_size=32, cross_attention_dim=8))
        self.second = TinyUNet(processor_class(hidden_size=32, cross_attention_dim=8))
        self.generator_forwards = 0
        self.block_calls = {'first': 0, 'second': 0}
        self.first.a.register_forward_pre_hook(self._count('first'))
        self.second.a.register_forward_pre_hook(self._count('second'))

    def _count(self, name):
        def count(module, args):
            self.block_calls[name] += 1
        return count

    @property
    def attn_processors(self):
        return {'first.a.processor': self.first.a.processor,
                'second.a.processor': self.second.a.processor}

    def forward(self, hidden, text, tokens, *, checkpointed=True):
        self.generator_forwards += 1
        if checkpointed and torch.is_grad_enabled():
            hidden = checkpoint(self.first.a, hidden, text, tokens, use_reentrant=False)
            return checkpoint(self.second.a, hidden, text, tokens, use_reentrant=False)
        hidden = self.first.a(hidden, text, tokens)
        return self.second.a(hidden, text, tokens)


def snapshot_queries(collector):
    assert not collector.active and set(collector.queries) == set(collector.layers)
    return {name: dict(reference=value, value=value.clone(), pointer=value.data_ptr(),
                       version=value._version, sha256=tensor_sha(value))
            for name, value in collector.queries.items()}


def assert_queries_unchanged(collector, snapshot, *, consumed=False):
    assert not collector.active
    assert (not collector.queries) if consumed else set(collector.queries) == set(snapshot)
    for name, record in snapshot.items():
        reference = record['reference']
        assert not reference.requires_grad and reference.grad_fn is None
        assert reference.data_ptr() == record['pointer'] and reference._version == record['version']
        assert torch.equal(reference, record['value']) and tensor_sha(reference) == record['sha256']
        if not consumed:
            assert collector.queries[name] is reference


def copy_fixture(fixture):
    # Reconstruct native modules from an exact state dict, rather than deepcopy
    # closures whose Python hook counters might retain the original instance.
    cls, _ = native_processor_class()
    unet = TwoBlockNativeTinyUNet(cls).eval().requires_grad_(False)
    unet.load_state_dict(fixture['unet_state'], strict=True)
    head = BoundedConditionAdapter(predict_risk=False)
    head.load_state_dict(fixture['head_state'], strict=True)
    projection = nn.Linear(1024, 32, bias=False).eval().requires_grad_(False)
    projection.load_state_dict(fixture['projection_state'], strict=True)
    return unet, head, projection


def parent_capture(unet, collector, hidden, text, base_tokens):
    before_rng = torch.get_rng_state().clone()
    with torch.no_grad(), collector.capture():
        parent_result = unet(hidden, text, base_tokens)
    assert not parent_result.requires_grad
    assert torch.equal(before_rng, torch.get_rng_state()), 'Parent forward unexpectedly consumed RNG'
    return parent_result, snapshot_queries(collector)


def manual_positive(collector, student, positive, negative):
    losses = []
    for name, (attention, processor) in collector.layers.items():
        query = collector.queries[name].detach()
        response = ip_response(query, student, processor, attention.heads)
        with torch.no_grad():
            correct = ip_response(query, positive.detach(), processor, attention.heads)
            wrong = ip_response(query, negative.detach(), processor, attention.heads)
        loss, _ = functional_alignment_loss(response, correct, wrong, mode='positive_only')
        losses.append(loss)
    return torch.stack(losses).mean()


def reference_Q_contracts(fixture):
    unet, head, projection = copy_fixture(fixture)
    collector = NativeIPFunctionalAlignmentCapture(unet, expected_layers=2)
    raw, weight, bias = fixture['raw'], fixture['weight'], fixture['bias']
    hidden = fixture['hidden'].clone().requires_grad_()
    base_tokens = fixture['base_tokens'].clone().requires_grad_()
    _, parent = parent_capture(unet, collector, hidden, fixture['text'], base_tokens)
    changed_a, _ = head(raw, weight, bias)
    head_b = copy.deepcopy(head)
    with torch.no_grad():
        head_b.direction.weight.mul_(-1.)
        head_b.direction.bias.add_(.1)
    changed_b, _ = head_b(raw, weight, bias)
    assert not torch.equal(changed_a, changed_b)
    tokens_a, tokens_b = projection(changed_a).view(4, 4, 8), projection(changed_b).view(4, 4, 8)
    downstream_student_Q = []
    def observe_downstream(module, args, output):
        downstream_student_Q.append(output.detach().clone())
    observer = unet.second.a.to_q.register_forward_hook(observe_downstream)
    # Intentionally NO capture(enabled=False): that entry correctly rejects a
    # pending parent capture. Inactive hooks already leave the parent Q intact.
    student_a = unet(hidden, fixture['text'], tokens_a)
    assert_queries_unchanged(collector, parent)
    student_b = unet(hidden, fixture['text'], tokens_b)
    assert_queries_unchanged(collector, parent)
    assert len(downstream_student_Q) == 2
    downstream_difference = float((downstream_student_Q[0] - downstream_student_Q[1]).abs().max())
    assert downstream_difference > 1e-5, 'The test failed to exercise condition-dependent downstream Q'
    assert not torch.equal(student_a, student_b)
    # Backward while the ORIGINAL parent Q is still pending proves actual
    # checkpoint recomputation cannot overwrite or append a new reference.
    student_a.square().mean().backward()
    assert_queries_unchanged(collector, parent)
    assert len(downstream_student_Q) == 3
    assert head.direction.weight.grad is not None and head.direction.weight.grad.norm() > 0
    assert base_tokens.grad is None
    observer.remove()
    collector.queries.clear()
    # Fresh parent captures remain exactly equal despite changing the student.
    _, parent_again = parent_capture(unet, collector, hidden, fixture['text'], base_tokens)
    assert all(parent_again[name]['sha256'] == record['sha256'] for name, record in parent.items())
    collector.close()
    return dict(parent_Q_sha256={k: v['sha256'] for k, v in parent.items()},
                downstream_student_Q_max_difference=downstream_difference,
                parent_Q_storage_and_values_unchanged_across_students=True,
                pending_parent_Q_unchanged_after_real_checkpoint_backward=True,
                repeated_parent_Q_bit_exact=True, no_parent_RNG_consumption=True)


def run_arm(fixture, arm, *, single_forward_control=False):
    assert arm in ('O', 'P', 'PN', 'S')
    assert not single_forward_control or arm == 'O'
    unet, head, projection = copy_fixture(fixture)
    collector = NativeIPFunctionalAlignmentCapture(unet, expected_layers=2)
    hidden = fixture['hidden'].clone().requires_grad_()
    base_tokens = fixture['base_tokens'].clone().requires_grad_()
    positive = fixture['positive'].clone().requires_grad_()
    negative = fixture['negative'].clone().requires_grad_()
    rank_positive = fixture['rank_positive'].clone().requires_grad_()
    rank_negative = fixture['rank_negative'].clone().requires_grad_()
    if not single_forward_control:
        _, parent = parent_capture(unet, collector, hidden, fixture['text'], base_tokens)
    changed, _ = head(fixture['raw'], fixture['weight'], fixture['bias'])
    student = projection(changed).view(4, 4, 8)
    predicted = unet(hidden, fixture['text'], student)  # Outside every capture.
    expected_forwards = 1 if single_forward_control else 2
    assert unet.generator_forwards == expected_forwards
    if not single_forward_control:
        assert_queries_unchanged(collector, parent)
    image_loss = F.mse_loss(predicted, fixture['epsilon'])
    token_loss = (1 - F.cosine_similarity(student, positive.detach(), dim=-1)).mean()
    base_loss = image_loss + .1 * token_loss
    auxiliary_calls = 0
    positive_gradient = None
    diagnostics = {}
    if arm == 'O':
        collector.queries.clear()
        auxiliary = student.sum() * 0.
        loss = base_loss  # No auxiliary term enters the O graph.
    else:
        expected_positive = manual_positive(collector, student, positive, negative)
        positive_gradient = torch.autograd.grad(expected_positive, head.direction.weight,
                                                retain_graph=True)[0].detach().clone()
        kwargs = dict(mode='positive_only' if arm == 'P' else 'full')
        if arm == 'S':
            kwargs.update(rank_positive_tokens=rank_positive, rank_negative_tokens=rank_negative)
        auxiliary_calls += 1
        auxiliary, diagnostics = collector.auxiliary(student, positive, negative, **kwargs)
        torch.testing.assert_close(diagnostics['positive_alignment'], expected_positive.detach())
        # Auxiliary-only graph isolation; the normal image loss can still and
        # should depend on hidden states through the student UNet.
        isolated = torch.autograd.grad(auxiliary,
            (hidden, base_tokens, positive, negative, rank_positive, rank_negative),
            allow_unused=True, retain_graph=True)
        assert all(value is None for value in isolated)
        assert torch.autograd.grad(auxiliary, head.direction.weight, retain_graph=True)[0].norm() > 0
        loss = base_loss + .1 * auxiliary
    assert not collector.active and not collector.queries
    loss.backward()
    if not single_forward_control:
        assert_queries_unchanged(collector, parent, consumed=True)
    assert unet.generator_forwards == expected_forwards
    assert all(calls == expected_forwards + 1 for calls in unet.block_calls.values())
    assert all(p.grad is None for p in unet.parameters())
    assert all(p.grad is None for p in projection.parameters())
    assert base_tokens.grad is None and positive.grad is None and negative.grad is None
    assert rank_positive.grad is None and rank_negative.grad is None
    gradients = {key: value.grad.detach().clone() for key, value in head.named_parameters()}
    assert gradients['direction.weight'].norm() > 0
    report = dict(arm=arm, explicit_generator_forwards=unet.generator_forwards,
        block_calls_including_checkpoint_recompute=unet.block_calls,
        auxiliary_calls=auxiliary_calls, auxiliary=float(auxiliary.detach()),
        image_loss=float(image_loss.detach()), token_loss=float(token_loss.detach()), total_loss=float(loss.detach()),
        parent_Q_sha256={k: v['sha256'] for k, v in parent.items()} if not single_forward_control else None,
        diagnostics={key: float(value) for key, value in diagnostics.items()},
        direction_gradient_norm=float(gradients['direction.weight'].norm()))
    collector.close()
    return report, predicted.detach(), gradients, positive_gradient


def run_checks():
    torch.set_num_threads(1)
    torch.manual_seed(1729)
    cls, native_sha = native_processor_class()
    unet = TwoBlockNativeTinyUNet(cls).eval().requires_grad_(False)
    head = BoundedConditionAdapter(predict_risk=False)
    assert sum(p.numel() for p in head.parameters()) == 525568 and head.radius == .25
    with torch.no_grad():
        head.direction.weight.normal_(0, .02)
        head.direction.bias.normal_(0, .01)
    projection = nn.Linear(1024, 32, bias=False).eval().requires_grad_(False)
    raw = F.layer_norm(torch.randn(4, 1024), (1024,))
    fixture = dict(unet_state=copy.deepcopy(unet.state_dict()), head_state=copy.deepcopy(head.state_dict()),
        projection_state=copy.deepcopy(projection.state_dict()), raw=raw,
        weight=torch.ones_like(raw), bias=torch.zeros_like(raw),
        base_tokens=projection(raw).detach().view(4, 4, 8), hidden=torch.randn(4, 37, 32),
        text=torch.randn(4, 5, 8), epsilon=torch.randn(4, 37, 32),
        positive=torch.randn(4, 4, 8), negative=torch.randn(4, 4, 8),
        rank_positive=torch.randn(4, 4, 8), rank_negative=torch.randn(4, 4, 8))
    reference = reference_Q_contracts(fixture)
    results = {arm: run_arm(fixture, arm) for arm in ('O', 'P', 'PN', 'S')}
    single = run_arm(fixture, 'O', single_forward_control=True)
    ordinary, ordinary_output, ordinary_gradients, _ = results['O']
    assert ordinary['auxiliary_calls'] == 0 and ordinary['auxiliary'] == 0
    assert ordinary['total_loss'] == single[0]['total_loss']
    assert torch.equal(ordinary_output, single[1])
    assert all(torch.equal(value, single[2][key]) for key, value in ordinary_gradients.items())
    parent_hashes = results['O'][0]['parent_Q_sha256']
    for arm in ('P', 'PN', 'S'):
        result, output, _, positive_gradient = results[arm]
        assert result['parent_Q_sha256'] == parent_hashes
        assert torch.equal(output, ordinary_output)
        assert result['image_loss'] == ordinary['image_loss'] and result['token_loss'] == ordinary['token_loss']
        assert torch.equal(positive_gradient, results['P'][3])
        assert result['diagnostics']['positive_alignment'] == results['P'][0]['diagnostics']['positive_alignment']
        assert result['explicit_generator_forwards'] == 2 and result['auxiliary_calls'] == 1
    assert results['P'][0]['diagnostics']['negative_discrimination'] == 0
    assert results['PN'][0]['diagnostics']['negative_discrimination'] != results['S'][0]['diagnostics']['negative_discrimination']
    return dict(status='FIXED_PARENT_Q_TWO_FORWARD_CPU_CONTRACTS_PASSED_NO_FIT',
        torch_version=torch.__version__, native_processor_source_sha256=native_sha,
        reference_Q_contracts=reference, arms={arm: value[0] for arm, value in results.items()},
        O_two_forward_output_loss_and_head_gradients_bit_exact_to_single_forward=True,
        P_PN_S_correct_positive_value_and_head_gradient_identical=True,
        parent_Q_identical_across_all_arms=True, auxiliary_Q_and_all_teachers_no_gradient=True,
        student_backward_and_checkpoint_cannot_overwrite_parent_Q=True,
        source_sha256={name: sha(ROOT / name) for name in (
            'check_BrainHIVE_functional_parentQ_CPU_v2.py', 'brainhive_functional_alignment_v2.py',
            'brainhive_response_rank_v1.py', 'brainhive_bounded_component_v1.py',
            'check_BrainHIVE_response_rank_CPU_v1.py')},
        model_weights_read=False, EEG_or_image_data_read=False, GPU_used=False,
        fitting=False, optimizer_created=False, real_GPU_profile_complete=False,
        limitations=['Synthetic frozen two-block architecture using the pinned native IP processor class, not full SDXL weights.',
                    'Two explicit generator forwards plus checkpoint recomputation are counted separately.',
                    'No performance, latency, EEG quality, or image-gain conclusion follows.'])


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
