"""Actual pinned class bodies on synthetic CPU inputs; no dataset or GPU I/O.

This audit establishes local engineering behavior and an exact CPU resume
fixture. It does not validate pretrained payloads, native CUDA/BF16 behavior,
method lock, official full fitting, exported conditions or image performance.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Tuple, Union

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / 'reproduction_sources/BrainHIVE_f06db69f'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_classes():
    scope = dict(torch=torch, nn=nn, F=F, Dict=Dict, List=List, Tuple=Tuple, Union=Union)
    receipt = json.loads((UPSTREAM / '_fetch_receipt.json').read_text())
    expected = {row['path']: row['sha256'] for row in receipt['artifacts']}
    pins = {}
    for filename, names in [('main/layers.py', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock', 'FusionEncoder']),
            ('main/models_brain.py', ['BrainMLP'])]:
        path = UPSTREAM / filename; pins[str(path)] = sha(path)
        assert pins[str(path)] == expected[filename]
        nodes = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.ClassDef) and node.name in names]
        assert [node.name for node in nodes] == names
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)
    return scope, pins


def stochastic_teacher_probe(scope):
    cfg_path = UPSTREAM / '_metadata/H14/fusion_encoder/config.json'
    config = json.loads(cfg_path.read_text())
    assert config['proj_meta'] == {'CLIP-ViT-B-32-laion2B-s34B-b79K': 512,
        'CLIP-ViT-H-14-laion2B-s32B-b79K': 1024, 'vae': 1024}
    torch.manual_seed(1729)
    brain = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).eval()
    assert sum(p.numel() for p in brain.parameters()) == 6452224
    assert brain.state_dict()['block.layers.0.norm.weight'].shape == (1024,)
    assert brain.state_dict()['block.layers.0.norm.bias'].shape == (1024,)
    with torch.inference_mode():
        condition = brain(torch.ones(3, 17, 250))
    assert condition.shape == (3, 1024) and torch.isfinite(condition).all()
    assert torch.all(condition.norm(dim=1) > 30), 'Raw output is not a unit-normalized condition'
    teacher = scope['FusionEncoder'](**{k: v for k, v in config.items() if not k.startswith('_')})
    teacher.requires_grad_(False)
    for module in teacher.modules():
        if isinstance(module, nn.Dropout): module.p = .1
    inputs = {key: torch.arange(3 * width, dtype=torch.float32).reshape(3, width) / (3 * width)
        for key, width in config['proj_meta'].items()}
    saved = {key: value.clone() for key, value in teacher.state_dict().items()}
    teacher.train()
    with torch.no_grad():
        a, b = teacher(**inputs), teacher(**inputs)
    assert a.shape == b.shape == (3, 1024) and not torch.equal(a, b)
    assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())
    teacher.eval()
    with torch.no_grad():
        c, d = teacher(**inputs), teacher(**inputs)
    assert torch.equal(c, d)
    assert all(torch.equal(saved[key], value) for key, value in teacher.state_dict().items())
    adapter_source = (UPSTREAM / 'main/models_adapter.py').read_text()
    wrapper = next(node for node in ast.parse(adapter_source).body if isinstance(node, ast.ClassDef) and node.name == 'FusionEncoderModel')
    forward = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    assert any(isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'adapter'
        for node in ast.walk(forward))
    return dict(full_native_BrainMLP_parameters=6452224, own_output_affine_shape=[1024],
        full_native_F_parameters=sum(p.numel() for p in teacher.parameters()),
        config=dict(path=str(cfg_path), sha256=sha(cfg_path)), actual_config_input_widths=config['proj_meta'],
        raw_condition_unit_norm=False, frozen_no_grad_train_mode_still_stochastic=True,
        eval_mode_repeated_forward_bit_identical=True, synthetic_teacher_weights_unchanged=True,
        top_level_FusionEncoderModel_forward_applies_additional_A=True,
        pretrained_weights_loaded=False, synthetic_class_forwards_only=True)


def setup_toy(scope):
    # Actual upstream class bodies, deliberately SMALL synthetic dimensions.
    # This verifies RNG/optimizer semantics, not actual EEG training efficacy.
    torch.manual_seed(77)
    teacher = scope['FusionEncoder'](hidden_size=8, intermediate_size=16,
        proj_meta={'visual': 8}, dropout=.1).train().requires_grad_(False)
    torch.manual_seed(1729)
    model = scope['BrainMLP'](8, 8, num_layers=1, dropout=.1)
    logit = nn.Parameter(torch.tensor(2.6592))
    norms = {id(p) for module in model.modules() if isinstance(module, nn.LayerNorm) for p in module.parameters()}
    decay = [p for name, p in model.named_parameters() if not name.endswith('bias') and id(p) not in norms] + [logit]
    no_decay = [p for name, p in model.named_parameters() if name.endswith('bias') or id(p) in norms]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05), dict(params=no_decay, weight_decay=0.)],
        lr=5e-4, betas=(.9, .995))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: .5 * (1 + __import__('math').cos(__import__('math').pi * step / 8)))
    shuffle = torch.Generator().manual_seed(1730)
    return model, teacher, logit, optimizer, scheduler, shuffle


def steps(objects, epochs):
    model, teacher, logit, optimizer, scheduler, shuffle = objects
    x = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 64
    visual = torch.cos(torch.arange(64, dtype=torch.float32).reshape(8, 8))
    order_history, losses = [], []
    model.train()
    for _ in range(epochs):
        order = torch.randperm(8, generator=shuffle); order_history.append(order.tolist())
        for ids in order.split(4):
            optimizer.zero_grad(set_to_none=True)
            student = F.normalize(model(x[ids]), dim=-1)
            with torch.no_grad(): target = F.normalize(teacher(visual=visual[ids]), dim=-1)
            logits = student @ target.T * logit.exp(); labels = torch.arange(len(ids))
            loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
            loss.backward(); nn.utils.clip_grad_norm_(list(model.parameters()) + [logit], 1., error_if_nonfinite=True)
            optimizer.step(); scheduler.step(); losses.append(float(loss.detach()))
    return order_history, losses


def state(objects):
    model, teacher, logit, optimizer, scheduler, shuffle = objects
    return copy.deepcopy(dict(model=model.state_dict(), teacher=teacher.state_dict(), logit=logit.detach(),
        optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(), torch_rng=torch.get_rng_state(),
        shuffle_rng=shuffle.get_state()))


def resume(scope, snapshot, *, restore_rng=True, restore_shuffle=True):
    objects = setup_toy(scope)
    model, teacher, logit, optimizer, scheduler, shuffle = objects
    model.load_state_dict(snapshot['model'], strict=True); teacher.load_state_dict(snapshot['teacher'], strict=True)
    logit.data.copy_(snapshot['logit']); optimizer.load_state_dict(copy.deepcopy(snapshot['optimizer']))
    scheduler.load_state_dict(copy.deepcopy(snapshot['scheduler']))
    if restore_shuffle: shuffle.set_state(snapshot['shuffle_rng'])
    if restore_rng: torch.set_rng_state(snapshot['torch_rng'])
    return objects


def resume_probe(scope):
    baseline = setup_toy(scope); base_order, base_losses = steps(baseline, 4); base = state(baseline)
    interrupted = setup_toy(scope); first_order, first_losses = steps(interrupted, 2); snapshot = state(interrupted)
    restored = resume(scope, snapshot); rest_order, rest_losses = steps(restored, 2); final = state(restored)
    assert first_order + rest_order == base_order and first_losses + rest_losses == base_losses
    assert all(torch.equal(value, final['model'][key]) for key, value in base['model'].items())
    assert torch.equal(base['logit'], final['logit'])
    broken_rng = resume(scope, snapshot, restore_rng=False); _, broken_losses = steps(broken_rng, 2)
    assert broken_losses != rest_losses
    broken_shuffle = resume(scope, snapshot, restore_shuffle=False); wrong_order, _ = steps(broken_shuffle, 2)
    assert wrong_order != rest_order
    return dict(actual_upstream_class_bodies=True, synthetic_toy_dimensions=8,
        CPU_float32_only=True, total_uninterrupted_updates=8, split_updates=[4, 4],
        exact_resume_losses_orders_parameters_logit_bit_identical=True,
        missing_torch_RNG_rejected_by_divergent_losses=True, missing_shuffle_RNG_rejected_by_order_difference=True,
        CUDA_RNG_or_BF16_resume_verified=False, actual_EEG_parameter_updates=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt', type=Path, required=True); args = parser.parse_args()
    assert not args.receipt.exists(), 'Preserve previous engineering evidence'
    started = time.monotonic(); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    scope, pins = source_classes()
    result = dict(status='PINNED_NATIVE_SOURCE_SYNTHETIC_CPU_FIDELITY_AUDIT_COMPLETE',
        source_sha256=sha(__file__), torch_version=torch.__version__, upstream_source_sha256=pins,
        teacher_and_native_interface=stochastic_teacher_probe(scope), exact_resume_fixture=resume_probe(scope),
        pretrained_or_dataset_payloads_read=False, actual_EEG_or_image_queries_read=0, official_scores_read=0,
        dataset_training_updates=0, GPU_initialized=torch.cuda.is_initialized(), method_locked=False,
        official_fit_or_export_admitted=False, native_CUDA_BF16_verified=False,
        seconds=time.monotonic() - started)
    assert result['seconds'] < 120 and result['GPU_initialized'] is False
    raw = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode(); assert len(raw) < 32 << 10
    with args.receipt.open('xb') as stream: stream.write(raw)
    print(raw.decode(), end='')


if __name__ == '__main__':
    main()
