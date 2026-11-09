"""CPU checks of actual pinned processor math and checkpoint-safe observation."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from brainhive_bounded_component_v1 import BoundedConditionAdapter
from brainhive_response_rank_v1 import (NativeIPResponseCapture, concept_derangement,
                                        ip_response, response_rank_loss, spread_positions)

ROOT = Path(__file__).resolve().parent


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def native_processor_class():
    source = ROOT / 'reproduction_sources/BrainHIVE_f06db69f'
    receipt = json.loads((source / '_fetch_receipt.json').read_text())
    expected = next(x['sha256'] for x in receipt['artifacts'] if x['path'] == 'main/models_adapter.py')
    path = source / 'main/models_adapter.py'
    assert digest(path) == expected
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == 'IPAttnProcessor2_0')
    scope = dict(torch=torch, nn=nn, F=F, Dict=Dict, List=List,
                 Optional=Optional, Tuple=Tuple, Union=Union, Attention=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), scope)
    return scope['IPAttnProcessor2_0'], expected


class TinyAttention(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.processor = processor
        self.heads = 4
        self.to_q = nn.Linear(32, 32, bias=False)
        self.to_k = nn.Linear(8, 32, bias=False)
        self.to_v = nn.Linear(8, 32, bias=False)
        self.to_out = nn.ModuleList([nn.Linear(32, 32, bias=False), nn.Identity()])
        self.spatial_norm = None
        self.group_norm = None
        self.norm_q = None
        self.norm_k = None
        self.norm_cross = False
        self.residual_connection = True
        self.rescale_output_factor = 1.0

    def forward(self, hidden, text, tokens):
        return self.processor(self, hidden, encoder_hidden_states=text, ip_hidden_states=tokens)


class TinyUNet(nn.Module):
    def __init__(self, processor):
        super().__init__()
        self.a = TinyAttention(processor)

    @property
    def attn_processors(self):
        return {'a.processor': self.a.processor}


def run_checks():
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    cls, source_sha = native_processor_class()
    unet = TinyUNet(cls(hidden_size=32, cross_attention_dim=8)).eval().requires_grad_(False)
    hidden = torch.randn(3, 37, 32)
    text = torch.randn(3, 5, 8)
    student = torch.randn(3, 4, 8, requires_grad=True)
    positive = torch.randn(3, 4, 8, requires_grad=True)
    negative = torch.randn(3, 4, 8, requires_grad=True)
    with torch.no_grad():
        reference = unet.a(hidden, text, student)
    collector = NativeIPResponseCapture(unet, expected_layers=1)
    with collector.capture():
        observed = unet.a(hidden, text, student)
    assert torch.equal(reference, observed)
    sampled = collector.queries['a.processor']
    q = unet.a.to_q(hidden).view(3, 37, 4, 8).transpose(1, 2)
    positions = spread_positions(37)
    assert torch.equal(sampled, q.index_select(2, positions))
    assert sampled.shape == (3, 4, 16, 8) and not sampled.requires_grad
    actual = ip_response(sampled, student, unet.a.processor, 4)
    keys = unet.a.processor.to_k_ip(student).view(3, 4, 4, 8).transpose(1, 2)
    values = unet.a.processor.to_v_ip(student).view(3, 4, 4, 8).transpose(1, 2)
    manual = (sampled @ keys.transpose(-1, -2) / (8 ** .5)).softmax(-1) @ values
    torch.testing.assert_close(actual, manual, atol=2e-7, rtol=2e-6)
    shift = torch.randn(3, 4, 1, 8)
    shifted = F.scaled_dot_product_attention(sampled, keys + shift, values, dropout_p=0.0)
    torch.testing.assert_close(actual, shifted, atol=2e-7, rtol=2e-6)
    aux, info = collector.auxiliary(student, positive, negative)
    aux.backward()
    assert student.grad is not None and student.grad.norm() > 0
    assert positive.grad is None and negative.grad is None
    assert all(p.grad is None for p in unet.parameters())
    assert not collector.queries
    with collector.capture(enabled=False):
        disabled = unet.a(hidden, text, student.detach())
    assert torch.equal(reference, disabled) and not collector.queries
    # Rank direction is tested separately so random native projections cannot
    # accidentally hide an incorrect positive/negative sign.
    true = torch.randn(3, 4, 16, 8)
    wrong = torch.randn_like(true)
    correct_loss, _ = response_rank_loss(true, true, wrong)
    wrong_loss, _ = response_rank_loss(wrong, true, wrong)
    assert correct_loss < wrong_loss
    degenerate, stats = response_rank_loss(true, true, true)
    torch.testing.assert_close(degenerate, torch.tensor(2.).log())
    assert stats['near_zero_teacher_fraction'] == 1
    # Full backward through a non-reentrant checkpoint: hook capture is closed
    # before recomputation and auxiliary is held outside the checkpoint closure.
    calls = [0]
    hidden2 = hidden.clone().requires_grad_()
    student2 = student.detach().clone().requires_grad_()
    def closure(h, c):
        calls[0] += 1
        return unet.a(h, text, c)
    with collector.capture():
        result = checkpoint(closure, hidden2, student2, use_reentrant=False)
    aux2, _ = collector.auxiliary(student2, positive, negative)
    only_aux_hidden = torch.autograd.grad(aux2, hidden2, retain_graph=True, allow_unused=True)[0]
    assert only_aux_hidden is None
    (result.square().mean() + .1 * aux2).backward()
    assert calls[0] == 2 and not collector.active and not collector.queries
    assert student2.grad is not None and student2.grad.norm() > 0
    # The original host-normalized correction and its number of parameters stay.
    adapter = BoundedConditionAdapter(predict_risk=False)
    assert sum(p.numel() for p in adapter.parameters()) == 525568
    c = F.layer_norm(torch.randn(3, 1024), (1024,))
    assert torch.equal(adapter(c, torch.ones_like(c), torch.zeros_like(c))[0], c)
    rows = [100, 77, 9, 300, 80, 14]
    concepts = {100: 'a', 77: 'a', 9: 'b', 300: 'b', 80: 'c', 14: 'd'}
    negative_index = concept_derangement(rows, concepts)
    assert torch.equal(negative_index, concept_derangement(rows, concepts))
    assert sorted(negative_index.tolist()) == list(range(len(rows)))
    assert all(concepts[rows[i]] != concepts[rows[j]] for i, j in enumerate(negative_index.tolist()))
    rejected = False
    try:
        concept_derangement([0, 1, 2], {0: 'a', 1: 'a', 2: 'b'})
    except AssertionError:
        rejected = True
    assert rejected
    collector.close()
    return dict(status='CPU_NATIVE_MATH_GRADIENT_AND_CHECKPOINT_CHECKS_PASSED',
                native_processor_source_sha256=source_sha, torch_version=torch.__version__,
                exact_native_forward_unchanged=True, disabled_exact=True,
                native_SDP_matches_manual=True, common_key_shift_invariant=True,
                student_nonzero_gradient=True, teacher_and_Q_no_auxiliary_gradient=True,
                all_frozen_weights_no_gradients=True, checkpoint_recomputation_no_capture=True,
                checkpoint_calls=calls[0], rank_sign_passed=True,
                degenerate_teacher_finite_log2=True, negative_input_position_mapping=True,
                correction_parameters=525568, zero_init_exact=True,
                actual_weights_loaded=False, actual_GPU_profile_complete=False,
                actual_image_gain_established=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output')
    args = parser.parse_args()
    report = run_checks()
    if args.output:
        path = Path(args.output)
        assert not path.exists(), 'Keep historical receipts'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
