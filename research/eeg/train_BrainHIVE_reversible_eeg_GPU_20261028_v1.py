"""D30 finite pure-PyTorch profile/fit, matched native/serial/parallel arms.

All three arms continue the same pre-existing role checkpoint for25additional
epochs. Fixed cached F targets replace stochastic teacher forwards explicitly.
No diffusers, ordinary, calibration, Dfit, Dcheck, official test or image input.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

PROCESS_START = time.monotonic()
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import brainhive_reversible_eeg_20261028_v1 as V


N = 13230
NATIVE_CONTRACT_SHA = 'bb6e38350282af7b37ece04689adcec7a5f87dccf68de5968058433faece0597'
ROLES = ('full', 'oof0', 'oof1')
TRIALS_SHA = 'b804a105822bfab5eeb853b6942d25109eaf88f3f8a1eb5b503b444a4dc0d393'
TARGETS_SHA = '363e8e74514ca8d1bcbfb29afc2023ed4a5c3a6c4f49bf8b1134e7ae7a090336'
INDEX_SHA = 'a12c2758f8555223e08022e5b4a195da37ad3dac37cb6e2806aaf17bac014e67'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(reference):
    path = Path(reference['path'])
    assert path.is_file() and sha(path) == reference['sha256'], str(path)
    if 'bytes' in reference:
        assert path.stat().st_size == reference['bytes']
    return path


def read(path):
    return json.loads(Path(path).read_bytes())


def atomic_json(path, value):
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists()
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(item) for item in value)
    return value


def atomic_torch(path, value):
    temporary = path.with_name(path.name + '.pending')
    assert not temporary.exists()
    with temporary.open('xb') as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def rss_bytes():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux peak-RSS measurement unavailable')


def wrapper_and_host(authorization):
    pid = os.getppid()
    chain = []
    for _ in range(6):
        root = Path('/proc') / str(pid)
        if not root.exists():
            break
        cmd = (root / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
        chain.append(dict(pid=pid, command=cmd))
        fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
        pid = int(fields[1])
        if pid <= 1:
            break
    assert any('run_BrainHIVE_two_host_GPU_job_20261028_v1.py' in row['command'] for row in chain), 'Required finite host wrapper missing'
    uuid = subprocess.check_output(['nvidia-smi', '--id=0', '--query-gpu=uuid', '--format=csv,noheader'], text=True, timeout=10).strip()
    hosts = [host for host in authorization['authorized_hosts'] if host.get('GPU_UUID') == uuid]
    assert len(hosts) == 1 and hosts[0]['host'] == 'CLOUD_LOGIN_HOST'
    assert authorization['existing_ordinary_baseline_bytes_immutable'] is True
    assert authorization['purchase_or_provider_account_changes_authorized'] is False
    assert authorization['formal_old_D_training_resumed'] is False
    return dict(ancestor_chain=chain, GPU_UUID=uuid, authorized_host=hosts[0])


class Budget:
    def __init__(self, args, spec):
        self.args, self.spec = args, spec
        self.base_rss = rss_bytes()
        self.peak_rss = self.base_rss
        self.stop = False
        self.stage = 'INPUTS'
        self.peak_gpu = 0
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def check(self, stage=None):
        if stage is not None:
            self.stage = stage
        self.peak_rss = max(self.peak_rss, rss_bytes())
        if torch.cuda.is_initialized():
            self.peak_gpu = max(self.peak_gpu, torch.cuda.max_memory_allocated())
        reserve = 15 if self.args.mode == 'profile' else 60
        assert not self.stop and time.monotonic() - PROCESS_START < self.args.max_seconds - reserve, 'Finite job termination/time reserve'
        assert self.peak_rss - self.base_rss < self.spec['maximum_extra_CPU_bytes'], 'extra CPU budget exceeded'
        assert self.peak_gpu < self.spec['maximum_GPU_bytes'], 'GPU allocated budget exceeded'
        assert sum(path.stat().st_size for path in self.args.output.rglob('*') if path.is_file()) < self.spec['maximum_output_bytes'], 'output budget exceeded'
        assert not any(name in sys.modules for name in ('diffusers', 'transformers', 'accelerate'))


def log(path, event):
    event = dict(event, seconds=time.monotonic() - PROCESS_START)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(event, separators=(',', ':')), flush=True)


def tensor_digest(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def native_state_digest(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode())
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_inputs(args, spec):
    assert spec['status'] == 'D30_REVERSIBLE_EEG_SPEC_LOCKED'
    assert spec['seed'] == 1729 and spec['epochs'] == 25 and spec['batch_size'] == 1024
    assert spec['maximum_GPU_bytes'] == 20 << 30 and spec['maximum_extra_CPU_bytes'] == 10 << 30
    assert spec['maximum_output_bytes'] == 2 << 30
    for item in spec['source_pins']:
        checked(item)
    assert args.input_contract.resolve() == checked(spec['physical_R4_input_contract']).resolve()
    assert args.native_contract.resolve() == checked(spec['native_input_contract']).resolve()
    assert args.extra_authorization.resolve() == checked(spec['extra_authorization']).resolve()
    assert sha(args.native_contract) == NATIVE_CONTRACT_SHA
    physical, native = read(args.input_contract), read(args.native_contract)
    authorization = read(args.extra_authorization)
    assert authorization['maximum_GPU_memory_bytes_per_job'] == spec['maximum_GPU_bytes']
    assert authorization['maximum_extra_CPU_memory_bytes_per_job'] == spec['maximum_extra_CPU_bytes']
    assert physical['status'] == 'PHYSICAL_R4_T_INPUTS_READY' and physical['export_shape'] == [N, 4, 17, 250]
    assert physical['mean_replay_exact'] is True and physical['input_contract_sha256'] == NATIVE_CONTRACT_SHA
    trials_reference = spec.get('T_trials', physical['files']['trials.npy'])
    index_reference = spec.get('T_index', physical['files']['index.jsonl'])
    targets_reference = spec['T_targets']
    assert trials_reference['sha256'] == TRIALS_SHA == physical['files']['trials.npy']['sha256']
    assert index_reference['sha256'] == INDEX_SHA == physical['files']['index.jsonl']['sha256']
    assert targets_reference['sha256'] == TARGETS_SHA == physical['files']['targets.npy']['sha256']
    paths = [checked(item) for item in (trials_reference, index_reference, targets_reference)]
    index = [json.loads(line) for line in paths[1].read_text(encoding='utf-8').splitlines()]
    assert len(index) == N and all(row['row'] == i and row['partition'] == 'T' and row['subject_id'] == 'sub-01' for i, row in enumerate(index))
    folds = np.array([row['fold'] for row in index], dtype=np.int8)
    assert set(folds.tolist()) == {0, 1}
    groups = {}
    for row in index:
        groups.setdefault(row['concept_id'], []).append(row['row'])
    assert len(groups) == 1323 and set(map(len, groups.values())) == {10}
    assert all(len(set(folds[rows].tolist())) == 1 for rows in groups.values())
    assert len({row['array_row'] for row in index}) == len({row['stimulus_id'] for row in index}) == N
    subject = native['subjects'][0]
    assert subject['subject'] == 'sub-01'
    sources = {source['role']: source for source in subject['sources']}
    role_data = {}
    for role in ROLES:
        item, source = spec['roles'][role], sources[role]
        assert item['checkpoint']['sha256'] == source['checkpoint_sha256']
        assert item['condition_cache']['sha256'] == source['condition_cache_sha256']
        checkpoint_path, cache_path = checked(item['checkpoint']), checked(item['condition_cache'])
        if role == 'full':
            fit, held = np.arange(N), np.array([], dtype=np.int64)
        else:
            held_fold = int(role[-1])
            assert item.get('held_fold', held_fold) == held_fold
            fit, held = np.flatnonzero(folds != held_fold), np.flatnonzero(folds == held_fold)
            assert {index[i]['array_row'] for i in held} == {row['array_row'] for row in source['groups']['component_train']}
            assert len(fit) in (6610, 6620) and not set(fit).intersection(held)
        role_data[role] = dict(fit=fit, held=held, checkpoint_path=checkpoint_path, condition_cache=cache_path)
    return dict(index=index, folds=folds, role_data=role_data, paths=paths,
                actual_inputs=[trials_reference, index_reference, targets_reference], authorization=authorization,
                core_class=V.author_brain_class(checked(spec['author_models_source']), checked(spec['author_layers_source'])))


def model_and_optimization(inputs, role, arm):
    old = torch.load(inputs['role_data'][role]['checkpoint_path'], map_location='cpu', weights_only=True)
    assert old['epochs'] == 25
    model = V.from_core_state(inputs['core_class'], old['model'], arm, 1729).cuda()
    scale = nn.Parameter(old['logit_scale'].detach().float().cuda().clone())
    optimizer = torch.optim.AdamW(V.optimizer_groups(model, scale), lr=5e-4, betas=(.9, .995))
    return model, scale, optimizer, old


@torch.no_grad()
def baseline_replay(inputs, role, model, eeg):
    cache = torch.load(inputs['role_data'][role]['condition_cache'], weights_only=True, map_location='cpu', mmap=True)
    assert cache['model_sha256'] == sha(inputs['role_data'][role]['checkpoint_path'])
    allowed = inputs['role_data'][role]['held'] if role != 'full' else np.arange(N)
    by_array_row = {int(row): i for i, row in enumerate(cache['array_rows'])}
    positions = [int(i) for i in allowed if inputs['index'][i]['array_row'] in by_array_row][:80]
    assert len(positions) == 80
    expected = cache['condition'][[by_array_row[inputs['index'][i]['array_row']] for i in positions]].clone().cuda()
    model.eval()
    observed = model(eeg[positions])
    model.enabled = False
    disabled = model(eeg[positions])
    model.enabled = True
    assert torch.equal(observed, disabled), 'Zero-head native recovery failed'
    error = float(((observed - expected).norm(dim=1) / expected.norm(dim=1).clamp_min(1e-12)).max())
    assert error < 3e-5, 'Original80row native cache replay mismatch'
    return dict(rows=80, compact_rows=positions, array_rows=[inputs['index'][i]['array_row'] for i in positions],
                cache_sha256=sha(inputs['role_data'][role]['condition_cache']), maximum_relative_error=error,
                zero_head_disabled_exact=True, non_T_tensor_values_read=False)


def train_step(model, scale, optimizer, eeg, target, ids, seed):
    model.train()
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    rng_before = tensor_digest(torch.cuda.get_rng_state())
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = V.symmetric_infonce(model(eeg[ids]), target[ids], scale)
    assert torch.isfinite(loss)
    rng_after = tensor_digest(torch.cuda.get_rng_state())
    loss.backward()
    parameters = list(model.parameters()) + [scale]
    norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
    optimizer.step()
    return float(loss.detach()), float(norm), rng_before, rng_after


def state_snapshot(model, scale, optimizer, scheduler, generator, **metadata):
    return dict(model=cpu_tree(model.state_dict()), logit_scale=scale.detach().cpu().clone(),
                optimizer=cpu_tree(optimizer.state_dict()), scheduler=scheduler.state_dict(),
                CPU_rng=torch.get_rng_state().clone(), CUDA_rng=cpu_tree(torch.cuda.get_rng_state_all()),
                shuffle_rng=generator.get_state().clone(), **metadata)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'input-contract', 'native-contract', 'extra-authorization', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--profile', type=Path)
    parser.add_argument('--mode', choices=('profile', 'fit'), required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    parser.add_argument('--gpu-job-name', default='D30_reversible_EEG')
    args = parser.parse_args()
    assert args.max_seconds == (120 if args.mode == 'profile' else 900)
    assert not args.output.exists(), 'Use a new output directory; failures are retained'
    args.output.mkdir(parents=True)
    spec = read(args.spec)
    budget = Budget(args, spec)
    active = None
    try:
        spec_ref, source_ref, module_ref = ref(args.spec), ref(__file__), ref(V.__file__)
        inputs = load_inputs(args, spec)
        atomic_json(args.output / 'role_rows.json', {role: dict(
            fit_compact_rows=inputs['role_data'][role]['fit'].tolist(),
            held_compact_rows=inputs['role_data'][role]['held'].tolist(),
            fit_array_rows=[inputs['index'][i]['array_row'] for i in inputs['role_data'][role]['fit']],
            held_array_rows=[inputs['index'][i]['array_row'] for i in inputs['role_data'][role]['held']]) for role in ROLES})
        host = wrapper_and_host(inputs['authorization'])
        if args.mode == 'fit':
            assert args.profile is not None
            profile = read(args.profile)
            assert profile['status'] == 'D30_PURE_TORCH_PROFILE_PASSED'
            assert profile['spec']['sha256'] == spec_ref['sha256']
            assert profile['source']['sha256'] == source_ref['sha256'] and profile['model_source']['sha256'] == module_ref['sha256']
            assert profile['host']['GPU_UUID'] == host['GPU_UUID']
        torch.set_num_threads(2)
        torch.cuda.set_device(0)
        assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        budget.check('LOAD_COMPLETE_PHYSICAL_MEAN4_AND_FIXED_TARGETS')
        trials = np.load(inputs['paths'][0], mmap_mode='r', allow_pickle=False)
        target_array = np.load(inputs['paths'][2], mmap_mode='r', allow_pickle=False)
        assert trials.shape == (N, 4, 17, 250) and trials.dtype == np.float32
        assert target_array.shape == (N, 1024) and target_array.dtype == np.float32
        eeg = torch.empty((N, 17, 250), dtype=torch.float32, device='cuda')
        for lo in range(0, N, 256):
            block = np.asarray(trials[lo:lo + 256])
            assert np.isfinite(block).all()
            eeg[lo:lo + len(block)] = torch.from_numpy(block.mean(axis=1, dtype=np.float32)).cuda()
            budget.check()
        target = torch.from_numpy(np.array(target_array, copy=True)).cuda()
        assert torch.isfinite(eeg).all() and torch.isfinite(target).all()
        del trials, target_array, block
        gc.collect()
        epochs_log = args.output / 'training_log.jsonl'
        baseline_proofs, profiles, models, affines, receipts = [], [], {}, {}, []
        oof = np.full((N, 3, 1024), np.nan, dtype=np.float32) if args.mode == 'fit' else None
        coverage = np.zeros((N, 3), dtype=np.int8) if args.mode == 'fit' else None
        fit_body_seconds = 0.
        for role_index, role in enumerate(ROLES):
            models[role], affines[role] = {}, {}
            paired_epoch_proofs = {}
            front_digest = None
            role_item = inputs['role_data'][role]
            total_updates = 25 * math.ceil(len(role_item['fit']) / 1024)
            for arm_index, arm in enumerate(V.ARMS):
                budget.check('PROFILE_OR_FIT_' + role + '_' + arm)
                model, scale, optimizer, old = model_and_optimization(inputs, role, arm)
                assert native_state_digest(model.core.state_dict()) == native_state_digest(old['model'])
                if arm_index:
                    current_front = native_state_digest(model.front.state_dict())
                    assert front_digest is None or current_front == front_digest
                    front_digest = current_front
                proof = baseline_replay(inputs, role, model, eeg)
                baseline_proofs.append(dict(role=role, arm=arm, **proof))
                scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 * (1 + math.cos(math.pi * step / total_updates)))
                generator = torch.Generator().manual_seed(1730 + role_index)
                active = dict(model=model, scale=scale, optimizer=optimizer, scheduler=scheduler,
                              generator=generator, role=role, arm=arm, epoch=0, batch=0, updates=0)
                if args.mode == 'profile':
                    ids = torch.as_tensor(role_item['fit'][:1024], device='cuda')
                    torch.cuda.synchronize()
                    tic = time.monotonic()
                    loss, norm, before, after = train_step(model, scale, optimizer, eeg, target, ids, 1729 + role_index * 100000)
                    torch.cuda.synchronize()
                    step_seconds = time.monotonic() - tic
                    profiles.append(dict(role=role, arm=arm, batch_size=len(ids), full_update_seconds=step_seconds,
                                         loss=loss, grad_norm=norm, CUDA_rng_before=before, CUDA_rng_after=after,
                                         estimated_training_seconds=step_seconds * total_updates,
                                         engineering_updates_discarded=1, scientific_checkpoint_exported=False))
                    if arm_index:
                        baseline_stream = profiles[-arm_index - 1]
                        assert (before, after) == (baseline_stream['CUDA_rng_before'], baseline_stream['CUDA_rng_after'])
                    budget.check()
                else:
                    tic = time.monotonic()
                    for epoch in range(25):
                        permutation = torch.randperm(len(role_item['fit']), generator=generator)
                        order = role_item['fit'][permutation.numpy()]
                        order_sha = hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()
                        cover = np.zeros(len(role_item['fit']), dtype=np.int8)
                        active.update(current_order_compact_rows=torch.from_numpy(order.copy()),
                                      current_epoch_coverage=torch.from_numpy(cover), spec=spec_ref,
                                      initial_checkpoint=spec['roles'][role]['checkpoint'])
                        stream_digest = hashlib.sha256()
                        loss_sum, seen, last_norm = 0., 0, 0.
                        for batch_index, lo in enumerate(range(0, len(order), 1024)):
                            budget.check()
                            chosen = permutation[lo:lo + 1024]
                            cover[chosen.numpy()] += 1
                            ids = torch.as_tensor(order[lo:lo + 1024], device='cuda')
                            seed = 1729 + role_index * 100000 + epoch * 1000 + batch_index
                            loss, last_norm, before, after = train_step(model, scale, optimizer, eeg, target, ids, seed)
                            stream_digest.update((before + after).encode())
                            scheduler.step()
                            active.update(epoch=epoch, batch=batch_index + 1, updates=active['updates'] + 1)
                            seen += len(ids)
                            loss_sum += loss * len(ids)
                        assert seen == len(role_item['fit']) and np.all(cover == 1)
                        epoch_proof = (order_sha, stream_digest.hexdigest())
                        if arm_index == 0:
                            paired_epoch_proofs[epoch] = epoch_proof
                        else:
                            assert epoch_proof == paired_epoch_proofs[epoch], 'Data order/core dropout random stream differs between arms'
                        event = dict(role=role, arm=arm, epoch=epoch + 1, updates=active['updates'], expected_updates=total_updates,
                                     seen=seen, coverage_min=int(cover.min()), coverage_max=int(cover.max()),
                                     order_sha256=order_sha, CUDA_dropout_stream_sha256=stream_digest.hexdigest(),
                                     mean_InfoNCE=loss_sum / seen, grad_norm_last=last_norm,
                                     fixed_cached_F=True, new_repeat_losses=0, Dfit_read=False)
                        log(epochs_log, event)
                        if (epoch + 1) % 5 == 0:
                            atomic_torch(args.output / 'latest_training_state.pt', state_snapshot(
                                model, scale, optimizer, scheduler, generator, role=role, arm=arm, epoch=epoch + 1,
                                updates=active['updates'], spec=spec_ref, coverage_complete=True))
                    torch.cuda.synchronize()
                    fit_body_seconds += time.monotonic() - tic
                    assert active['updates'] == total_updates
                    model.eval()
                    state = cpu_tree(model.state_dict())
                    models[role][arm] = state
                    gamma, beta = model.affine()
                    affines[role][arm] = dict(gamma=gamma.detach().cpu().clone(), beta=beta.detach().cpu().clone())
                    final_path = args.output / (role + '_' + arm + '_final.pt')
                    atomic_torch(final_path, state_snapshot(model, scale, optimizer, scheduler, generator,
                                 role=role, arm=arm, epoch=25, updates=total_updates, spec=spec_ref, initial_checkpoint=spec['roles'][role]['checkpoint']))
                    receipts.append(dict(role=role, arm=arm, checkpoint=ref(final_path),
                                         updates=total_updates, exposures=25 * len(role_item['fit']),
                                         unique_fit_T_rows=len(role_item['fit']), parameters=sum(p.numel() for p in model.parameters()),
                                         trainable_logit_parameters=1, initial_epochs=old['epochs'], additional_epochs=25,
                                         fixed_final_checkpoint=True, paired_order_and_dropout_stream=True))
                    if role != 'full':
                        held = role_item['held']
                        with torch.no_grad():
                            for lo in range(0, len(held), 256):
                                rows = held[lo:lo + 256]
                                prediction = model(eeg[rows]).float().cpu().numpy()
                                assert np.isfinite(prediction).all()
                                oof[rows, arm_index] = prediction
                                coverage[rows, arm_index] += 1
                                budget.check()
                active = None
                del model, scale, optimizer, scheduler, generator, old
                gc.collect()
                torch.cuda.empty_cache()
                budget.check()
        final_refs = inputs['actual_inputs'] + spec['source_pins'] + [spec_ref, source_ref, module_ref,
                     spec['physical_R4_input_contract'], spec['native_input_contract'], spec['extra_authorization']]
        for role in ROLES:
            final_refs += [spec['roles'][role]['checkpoint'], spec['roles'][role]['condition_cache']]
        for item in final_refs:
            checked(item)
        budget.check('FINAL_RESOURCE_AND_SOURCE_CHECKS')
        common = dict(spec=spec_ref, source=source_ref, model_source=module_ref, host=host,
                      baseline80_proofs=baseline_proofs, arms=V.ARMS, roles=ROLES,
                      role_rows=ref(args.output / 'role_rows.json'), actual_T_inputs=inputs['actual_inputs'],
                      fixed_cached_F=True, original_stochastic_teacher_replayed=False,
                      original_core_continuation_for_all_arms=True, original_ordinary_changed=False,
                      formal_D_resumed=False, new_images=0, Dfit_Dcheck_official_images_read=False,
                      novelty_established=False, image_gain_established=False,
                      maximum_seconds=args.max_seconds, maximum_GPU_bytes=spec['maximum_GPU_bytes'],
                      maximum_extra_CPU_bytes=spec['maximum_extra_CPU_bytes'],
                      peak_GPU_bytes=budget.peak_gpu, peak_extra_CPU_bytes=budget.peak_rss - budget.base_rss,
                      seconds=time.monotonic() - PROCESS_START, gpu_job_name=args.gpu_job_name)
        if args.mode == 'profile':
            # Conservative extrapolation is an admission estimate only, not an
            # effect result or a license to reduce any model's epoch budget.
            estimate = 1.75 * sum(row['estimated_training_seconds'] for row in profiles)
            assert estimate < 780., 'Fixed900second fit requires a different finite execution plan, not fewer epochs'
            done = dict(common, status='D30_PURE_TORCH_PROFILE_PASSED', profiles=profiles,
                        engineering_updates_discarded=len(profiles), scientific_model_updates=0,
                        conservative_estimated_all9_fit_seconds=estimate)
        else:
            assert len(receipts) == 9 and np.all(coverage == 1) and np.isfinite(oof).all()
            oof_path = args.output / 'OOF_T_native.npy'
            with oof_path.open('xb') as stream:
                np.save(stream, oof, allow_pickle=False)
                stream.flush()
                os.fsync(stream.fileno())
            bundle_path = args.output / 'models_and_affines.pt'
            atomic_torch(bundle_path, dict(models=models, affines=affines, arms=V.ARMS, roles=ROLES, spec=spec_ref,
                         architecture='four_additive4250RevNet_before_native4250to1024', author_models_source=spec['author_models_source'],
                         author_layers_source=spec['author_layers_source'], fixed_cached_F=True))
            done = dict(common, status='D30_ALL9_MATCHED_ADDITIONAL25_EPOCH_T_FITS_AND_TRUE_OOF_COMPLETE',
                        checkpoints=receipts, model_bundle=ref(bundle_path), OOF_T_native=ref(oof_path),
                        OOF_shape=[N, 3, 1024], OOF_coverage_min=int(coverage.min()), OOF_coverage_max=int(coverage.max()),
                        new_training_updates=sum(item['updates'] for item in receipts),
                        new_query_exposures=sum(item['exposures'] for item in receipts),
                        fit_body_seconds=fit_body_seconds, automatic_images_started=False,
                        no_Dfit_or_held_metric_checkpoint_selection=True)
        budget.check('ALL_OUTPUTS_READY')
        done.update(peak_GPU_bytes=budget.peak_gpu, peak_extra_CPU_bytes=budget.peak_rss - budget.base_rss,
                    seconds=time.monotonic() - PROCESS_START)
        atomic_json(args.output / 'completion.json', done)
        print(json.dumps(dict(status=done['status'], completion=ref(args.output / 'completion.json'))), flush=True)
    except BaseException as exc:
        partial = None
        if active is not None:
            try:
                partial_path = args.output / 'failure_partial_model_optimizer_RNG.pt'
                metadata = {key: value for key, value in active.items() if key not in ('model', 'scale', 'optimizer', 'scheduler', 'generator')}
                atomic_torch(partial_path, state_snapshot(active['model'], active['scale'], active['optimizer'],
                             active['scheduler'], active['generator'], **metadata))
                partial = ref(partial_path)
            except BaseException as save_error:
                partial = dict(error=repr(save_error))
        atomic_json(args.output / 'failure.json', dict(status='D30_FINITE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',
                    stage=budget.stage, error=repr(exc), traceback=traceback.format_exc(), partial_state=partial,
                    seconds=time.monotonic() - PROCESS_START, new_images=0, original_ordinary_changed=False,
                    formal_D_resumed=False, no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
