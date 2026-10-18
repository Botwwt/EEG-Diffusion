"""SERVER ONLY: stronger single-teacher joint-alignment references.

Final CLIP-H fixed, final CLIP-H joint, block12 joint. Known methods, not NVOL's
ATM/CSLS reproduction. No generated-image or novelty claim from retrieval.
"""
import ast
import ctypes
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Dict, List, Tuple, Union

assert sys.platform == 'linux'
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '2'
R = Path('/root/autodl-tmp/mced_research_bundle')
OUT = Path('/root/mced_eeg_native_single_visual_reference_v1')
PRE = Path('/root/mced_eeg_brainhive_native_reference_v1')
FULL = Path('/root/mced_eeg_native_training_fidelity_v1')
DATA = Path('/root/mced_eeg_full_montage_cache_v3')
AUDIT = Path('/root/mced_eeg_brainhive_reference_audit_v1')
FEATURE = R / 'studies/eeg_clip_visual_feature_cache_v2'
SPLIT = Path('/root/mced_eeg_native_target_level_reference_v1')
assert not OUT.exists()
OUT.mkdir()
START = time.monotonic()
FDS = []
SEEDS = [2026109501, 2026109502]
METHODS = ['final_fixed', 'final_joint', 'middle12_joint']
ROLES = ['heldout_normalized', 'full_raw']
COUNTS = dict(train_EEG_calls=0, train_visual_calls=0, train_states=0,
              backward_calls=0, optimizer_updates=0, eval_EEG_calls=0,
              eval_EEG_states=0, eval_visual_calls=0, eval_visual_states=0)


def identity(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return dict(bytes=p.stat().st_size, sha256=h.hexdigest())


def write(n, v, replace=False):
    p = OUT / n
    if replace:
        t = p.with_name(p.name + '.pending')
        t.write_text(json.dumps(v, indent=2, allow_nan=False) + '\n')
        t.replace(p)
    else:
        with p.open('x') as f:
            json.dump(v, f, indent=2, allow_nan=False)
            f.write('\n')


def verify(d, names=None):
    c = json.loads((d / 'final_commit.json').read_text())
    assert c['complete']
    for n in c['files'] if names is None else names:
        assert identity(d / n) == c['files'][n], (d, n)


def export(name, value):
    import torch
    lib = ctypes.CDLL(None, use_errno=True)
    fd = lib.syscall(ctypes.c_long(319), ctypes.c_char_p(name.encode()), ctypes.c_uint(3))
    assert fd >= 0, ctypes.get_errno()
    FDS.append(fd)
    with os.fdopen(os.dup(fd), 'wb') as f:
        torch.save(value, f)
    fcntl.fcntl(fd, 1033, 15)
    assert fcntl.fcntl(fd, 1034) == 15
    p = Path(f'/proc/{os.getpid()}/fd/{fd}')
    (OUT / name).symlink_to(p)
    return dict(name=name, memfd_path=str(p), **identity(p))


def run():
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 4 << 30
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), torch.get_num_threads(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        for d in [AUDIT, PRE, FULL]:
            verify(d)
        verify(DATA, ['sub-01_train_mean4.pt', 'sub-01_train_layout.json'])
        verify(SPLIT, ['split.json'])
        verify(FEATURE, ['train_image_embeds.pt', 'train_block12_token_mean.pt', 'image_receipts.json'])
        visual_features = {}
        for method in METHODS:
            name = 'train_block12_token_mean.pt' if method == 'middle12_joint' else 'train_image_embeds.pt'
            visual_features[method] = F.normalize(torch.load(FEATURE / name, weights_only=True).float(), dim=-1).cuda()
            assert visual_features[method].shape == (16540, 1280 if method == 'middle12_joint' else 1024)
            assert torch.isfinite(visual_features[method]).all()
        receipts = json.loads((FEATURE / 'image_receipts.json').read_text())
        assert [r['index'] for r in receipts if r['split'] == 'train'] == list(range(16540))
        concepts = set(json.loads((SPLIT / 'split.json').read_text())['heldout_concepts'])
        fit_subset = torch.tensor([i for i in range(16540) if i // 10 not in concepts])
        val = torch.tensor([i for i in range(16540) if i // 10 in concepts])
        assert len(fit_subset) == 15260 and len(val) == 1280
        bindings = json.loads((PRE / 'input_bindings.json').read_text())
        channels = bindings['selected_channel_names']
        channel_ids = bindings['selected_channel_indices']
        raw = torch.load(DATA / 'sub-01_train_mean4.pt', weights_only=True, mmap=True)[:, channel_ids, :].clone()
        reference = torch.load(PRE / f'{SEEDS[0]}_source_final.pt', weights_only=True)
        held_mean = reference['normalizer_mean']
        held_std = reference['normalizer_std']
        del reference
        scope = dict(torch=torch, nn=nn, F=F, Dict=Dict, List=List, Tuple=Tuple, Union=Union)
        for filename, sha, names in [
            ('main/layers.py', '560b038708652fac9baa98a1f8f642812eacb7bbb172a56e2c8199ebbdc5b559', ['BaseMLP', 'ResNet1DLayer', 'ResNet1DBlock']),
            ('main/models_brain.py', 'fc06ed93c1f1f8659605ec561d87e85f91ca72856fe034f4cf8416c2b60b39b1', ['BrainMLP'])]:
            p = AUDIT / 'source' / filename
            assert identity(p)['sha256'] == sha
            nodes = [n for n in ast.parse(p.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name in names]
            assert [n.name for n in nodes] == names
            exec(compile(ast.Module(body=nodes, type_ignores=[]), str(p), 'exec'), scope)

        class Visual(nn.Module):
            def __init__(self, method):
                super().__init__()
                self.method = method
                if method != 'final_fixed':
                    # Construct shared-width weights before dimension-dependent
                    # projection so their initial tensors match across methods.
                    self.block = scope['ResNet1DBlock'](hidden_size=1024,
                        intermediate_size=1024, num_layers=1, dropout=.1)
                    self.proj = nn.Linear(1280 if method == 'middle12_joint' else 1024, 1024)

            def forward(self, x):
                return x if self.method == 'final_fixed' else self.block(self.proj(x))

        old_rows = {r['seed']: r for r in json.loads((PRE / 'results.json').read_text())['training']}
        full_rows = {r['seed']: r for r in json.loads((FULL / 'results.json').read_text())['training'] if r['arm'] == 'full_raw'}
        common_blocks = {}
        dropout_streams = {}
        rows = []
        exports = []
        for role in ROLES:
            fit = fit_subset if role == 'heldout_normalized' else torch.arange(16540)
            mean = held_mean if role == 'heldout_normalized' else torch.zeros_like(held_mean)
            std = held_std if role == 'heldout_normalized' else torch.ones_like(held_std)
            x = ((raw - mean) / std).cuda()
            total = len(fit.split(1024)) * 25
            for method in METHODS:
                features = visual_features[method]
                for seed in SEEDS:
                    torch.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                    brain = scope['BrainMLP'](in_channels=4250, hidden_size=1024, num_layers=1, dropout=.1).cuda()
                    log_scale = nn.Parameter(torch.tensor(math.log(1 / .07), device='cuda'))
                    initial = hashlib.sha256()
                    for n, p in brain.state_dict().items():
                        initial.update(n.encode())
                        initial.update(p.cpu().numpy().tobytes())
                    initial.update(log_scale.detach().cpu().numpy().tobytes())
                    brain_sha = initial.hexdigest()
                    assert brain_sha == full_rows[seed]['initial_state_sha256']
                    cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
                    visual = Visual(method).cuda()
                    torch.set_rng_state(cpu_rng)
                    torch.cuda.set_rng_state_all(gpu_rng)
                    block_sha = None
                    if method != 'final_fixed':
                        digest = hashlib.sha256()
                        for n, p in visual.block.state_dict().items():
                            digest.update(n.encode())
                            digest.update(p.cpu().numpy().tobytes())
                        block_sha = digest.hexdigest()
                        if seed in common_blocks:
                            assert common_blocks[seed] == block_sha
                        else:
                            common_blocks[seed] = block_sha
                    named = [('brain.' + n, p) for n, p in brain.named_parameters()] + [('visual.' + n, p) for n, p in visual.named_parameters()]
                    norm_ids = {id(p) for model in [brain, visual] for m in model.modules()
                        if isinstance(m, nn.LayerNorm) for p in m.parameters()}
                    decay = [p for n, p in named if not n.endswith('bias') and id(p) not in norm_ids] + [log_scale]
                    no_decay = [p for n, p in named if n.endswith('bias') or id(p) in norm_ids]
                    params = [p for n, p in named] + [log_scale]
                    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05),
                        dict(params=no_decay, weight_decay=0.)], lr=5e-4, betas=(.9, .995))
                    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                        lambda step: .5 * (1 + math.cos(math.pi * step / total)))
                    generator = torch.Generator().manual_seed(seed + 100)
                    sequence = hashlib.sha256()
                    dropout = hashlib.sha256()
                    loss_total = 0.
                    updates = 0
                    brain.train()
                    visual.train()
                    for epoch in range(25):
                        for ids in fit[torch.randperm(len(fit), generator=generator)].split(1024):
                            sequence.update(ids.numpy().tobytes())
                            q = brain(x[ids])
                            z = visual(features[ids])
                            dropout.update(torch.cuda.get_rng_state().numpy().tobytes())
                            logits = F.normalize(q, dim=-1) @ F.normalize(z, dim=-1).T * log_scale.exp()
                            labels = torch.arange(len(ids), device='cuda')
                            loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                            assert torch.isfinite(loss)
                            optimizer.zero_grad(set_to_none=True)
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                            optimizer.step()
                            scheduler.step()
                            updates += 1
                            loss_total += float(loss.detach())
                            COUNTS['train_EEG_calls'] += 1
                            COUNTS['train_visual_calls'] += 1
                            COUNTS['train_states'] += len(ids)
                            COUNTS['backward_calls'] += 1
                            COUNTS['optimizer_updates'] += 1
                        if epoch == 0 or (epoch + 1) % 5 == 0:
                            print(json.dumps(dict(stage='SINGLE_VISUAL_TRAIN', role=role, method=method,
                                seed=seed, epoch=epoch + 1, updates=updates, mean_loss=loss_total / updates,
                                counts=COUNTS)), flush=True)
                    assert updates == total
                    assert sequence.hexdigest() == (old_rows if role == 'heldout_normalized' else full_rows)[seed]['input_sequence_sha256']
                    if method != 'final_fixed':
                        k = (role, seed)
                        if k in dropout_streams:
                            assert dropout_streams[k] == dropout.hexdigest()
                        else:
                            dropout_streams[k] = dropout.hexdigest()
                    cp = export(f'{role}_{method}_{seed}.pt', dict(state={k: p.cpu() for k, p in brain.state_dict().items()},
                        visual_state={k: p.cpu() for k, p in visual.state_dict().items()}, method=method, role=role,
                        seed=seed, logit_scale=log_scale.detach().cpu(), normalizer_mean=mean,
                        normalizer_std=std, selected_channels=channels, updates=updates, fit_images=len(fit)))
                    exports.append(cp)
                    row = dict(role=role, method=method, seed=seed, checkpoint=cp, updates=updates,
                        fit_images=len(fit), mean_loss=loss_total / updates,
                        initial_brain_sha256=brain_sha, initial_common_visual_block_sha256=block_sha,
                        input_sequence_sha256=sequence.hexdigest(), dropout_stream_sha256=dropout.hexdigest(),
                        trainable_parameters=sum(p.numel() for p in params),
                        visual_trainable_parameters=sum(p.numel() for p in visual.parameters()))
                    rows.append(row)
                    write(f'{role}_{method}_{seed}_ready.json', dict(complete=True, **row,
                        validation_and_official_test_not_read=True))
                    del brain, visual, log_scale, named, params, optimizer, scheduler, decay, no_decay, q, z, loss, logits
                    torch.cuda.empty_cache()
            del x
        assert COUNTS['optimizer_updates'] == 4800 and COUNTS['train_states'] == 4770000
        write('all_fits_complete_before_evaluation.json', dict(complete=True, counts=dict(COUNTS),
            common_brain_and_visual_blocks_verified=True, joint_method_dropout_streams_matched=True))
        verify(DATA, ['sub-01_test_mean80.pt', 'sub-01_test_layout.json'])
        verify(FEATURE, ['test_image_embeds.pt', 'test_block12_token_mean.pt'])
        test_features = {method: F.normalize(torch.load(FEATURE / ('test_block12_token_mean.pt'
            if method == 'middle12_joint' else 'test_image_embeds.pt'), weights_only=True).float(), dim=-1).cuda()
            for method in METHODS}
        raw_test = torch.load(DATA / 'sub-01_test_mean80.pt', weights_only=True)[:, channel_ids, :].float()
        arrays = {}
        evaluations = []
        predictions = {}
        with torch.no_grad():
            for split in ['source_heldout1280', 'official_test200']:
                arrays[split] = {}
                predictions[split] = {}
                for role in (['heldout_normalized'] if split == 'source_heldout1280' else ROLES):
                    arrays[split][role] = {}
                    predictions[split][role] = {}
                    for method in METHODS:
                        arrays[split][role][method] = {}
                        predictions[split][role][method] = {}
                        for seed in SEEDS:
                            row = next(r for r in rows if r['role'] == role and r['method'] == method and r['seed'] == seed)
                            state = torch.load(row['checkpoint']['memfd_path'], weights_only=True)
                            brain = scope['BrainMLP'](in_channels=4250, hidden_size=1024, num_layers=1, dropout=.1).cuda().eval().requires_grad_(False)
                            brain.load_state_dict(state['state'], strict=True)
                            visual = Visual(method).cuda().eval().requires_grad_(False)
                            visual.load_state_dict(state['visual_state'], strict=True)
                            inputs = raw[val] if split == 'source_heldout1280' else raw_test
                            inputs = ((inputs - state['normalizer_mean']) / state['normalizer_std']).cuda()
                            features = visual_features[method][val] if split == 'source_heldout1280' else test_features[method]
                            fixed = list(brain.parameters()) + list(brain.buffers()) + list(visual.parameters()) + list(visual.buffers())
                            versions = [p._version for p in fixed]
                            qs, gs = [], []
                            for ids in torch.arange(len(inputs)).split(256):
                                qs.append(brain(inputs[ids]))
                                gs.append(visual(features[ids]))
                                COUNTS['eval_EEG_calls'] += 1
                                COUNTS['eval_visual_calls'] += 1
                                COUNTS['eval_EEG_states'] += len(ids)
                                COUNTS['eval_visual_states'] += len(ids)
                            q, g = torch.cat(qs), torch.cat(gs)
                            score = F.normalize(q.double(), dim=-1) @ F.normalize(g.double(), dim=-1).T
                            rank = (score >= score.diag()[:, None]).sum(1) - 1
                            correct = (rank < 1).double().cpu().numpy()
                            arrays[split][role][method][seed] = correct
                            predictions[split][role][method][str(seed)] = dict(queries=q.cpu(), gallery=g.cpu())
                            e = dict(split=split, role=role, method=method, seed=seed,
                                top1=float(correct.mean()), top5=float((rank < 5).double().mean()),
                                mean_cosine=float(score.diag().mean()))
                            evaluations.append(e)
                            print(json.dumps(e), flush=True)
                            assert versions == [p._version for p in fixed] and all(p.grad is None for p in fixed)
                            del brain, visual, state, fixed, inputs, qs, gs, q, g
                            torch.cuda.empty_cache()
        assert COUNTS['eval_EEG_calls'] == COUNTS['eval_visual_calls'] == 42, COUNTS
        assert COUNTS['eval_EEG_states'] == COUNTS['eval_visual_states'] == 10080, COUNTS
        rng = np.random.default_rng(2026111301)
        members = np.stack([np.flatnonzero(np.asarray((val // 10).tolist()) == c) for c in sorted(concepts)])
        assert members.shape == (128, 10)
        comparisons = []
        for split, by_role in arrays.items():
            n = 128 if split == 'source_heldout1280' else 200
            draws = rng.integers(0, n, (20000, n))
            for role, values in by_role.items():
                for arm, control in [('middle12_joint', 'final_joint'), ('final_joint', 'final_fixed'), ('middle12_joint', 'final_fixed')]:
                    d = np.mean([values[arm][s] - values[control][s] for s in SEEDS], axis=0)
                    unit = d[members].mean(1) if split == 'source_heldout1280' else d
                    comparisons.append(dict(split=split, role=role, arm=arm, control=control,
                        mean_top1_difference=float(d.mean()),
                        paired_concept_bootstrap95=np.quantile(unit[draws].mean(1), [.025, .975]).tolist(),
                        uncertainty_conditional_on_two_fixed_training_seeds=True))
        exports.append(export('private_single_visual_predictions.pt', dict(predictions=predictions,
            source_validation_indices=val, test_indices=torch.arange(200), participant='sub-01')))
        write('input_bindings.json', dict(feature_commit=identity(FEATURE / 'final_commit.json'),
            image_receipts=identity(FEATURE / 'image_receipts.json'), brain_reference=identity(PRE / 'final_commit.json'),
            full_reference=identity(FULL / 'final_commit.json'), source_split=identity(SPLIT / 'split.json'),
            selected_channels=channels, training_features_from_TRAIN_partition_only=True))
        return dict(complete=True, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seconds=time.monotonic() - START, counts=COUNTS, training=rows, evaluations=evaluations,
            comparisons=comparisons, exports=exports, full_paper_reproduction=False,
            no_test_hyperparameter_or_checkpoint_selection=True,
            input_projection_parameter_counts_differ_with_visual_width=True,
            old_SAMGA_TSConv512_reference_not_repeated=True,
            author_EEG_image_order_assumed=True, author_MVNN_independence_unresolved=True,
            image_quality_established=False, novelty_established=False, SOTA=False)
    finally:
        cpu, cuda, threads, tf32, cudnn = saved
        torch.set_rng_state(cpu)
        torch.cuda.set_rng_state_all(cuda)
        torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn


try:
    write('plan.json', dict(pid=os.getpid(), script=identity(Path(__file__)),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), methods=METHODS, roles=ROLES, seeds=SEEDS,
        purpose='Establish stronger single-visual-teacher references before adding a generative mechanism',
        EEG='Exact public BrainMLP4250->1024, one1024residual block,17O+Pchannels250ticks',
        visual='One fixed CLIP-H final embedding or one learned1024residual head on final1024/block12mean1280; no feature fusion',
        training='One symmetric cosine InfoNCE; same25epochs,B1024,AdamW5e-4,wd.05 excludingbias/LN,betas(.9,.995),clip1,cosine,warmup0',
        primary='middle12_joint versus final_joint; fixed-final distinguishes adaptation effect; all methods and both seeds reported',
        all12fits_before_validation_or_official_test_tensors_read=True,
        full_source_models_have_no_source_holdout=True,
        no_peak_layer_search=True, no_test_gallery_transductive_CSLS=True,
        joint_methods_share_brain_initialization_common_visual_block_and_dropout_stream=True,
        fixed_head_lacks_visual_dropout_and_parameter_budgets_differ=True,
        current_known_method_reference_not_original_SOTA=True, SOTA=False))
    result = run()
    write('results.json', result)
    write('final_commit.json', dict(complete=True,
        files={p.name: identity(p) for p in OUT.glob('*.json')}, volatile_exports=result['exports']))
    print('READY_SINGLE_VISUAL_REFERENCE', flush=True)
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    until = time.monotonic() + 86400
    while time.monotonic() < until:
        write('keeper_status.json', dict(pid=os.getpid(), ready=True,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            lease_seconds_remaining=until - time.monotonic()), True)
        time.sleep(30)
except BaseException:
    write('failure.json', dict(traceback=traceback.format_exc(), counts=COUNTS,
        seconds=time.monotonic() - START))
    raise
finally:
    for fd in FDS:
        os.close(fd)
