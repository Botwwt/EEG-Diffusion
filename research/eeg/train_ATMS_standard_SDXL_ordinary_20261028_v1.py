"""Equivalent ordinary in native ATMS / pooled CLIP-H / standard SDXL space.

Fresh525568-parameter ordinary weights; original ATMS, native P/IP/SDXL frozen.
No BrainHIVE F/A or learned component weights are loaded. Full10subject genuine
concept-OOF T, three complete passes and a fixed last checkpoint are required.
Finite segments preserve committed model, optimizer, random streams and order.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import csv
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path
import random
import signal
import sys
import time
import traceback
PROCESS_START = time.monotonic()
# Match the unchanged standard loader before importing either NumPy or Torch.
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import numpy as np

ROOT = Path(__file__).resolve().parent
N_IMAGES, N_QUERIES, PASSES, BATCH = 13230, 132300, 3, 16
EXPECTED_UPDATES, EXPECTED_EXPOSURES = 24807, 396900
ARMS = 'ATMS_standard_SDXL_ordinary'
STRENGTHS = (0., .25, .5, .75, 1.)
torch = F = None


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def reference(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    if 'bytes' in item:
        assert path.stat().st_size == item['bytes'], str(path)
    return path


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value, *, overwrite=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    if temporary.exists() and overwrite:
        os.replace(temporary, temporary.with_name(temporary.name + '.incomplete_' + str(time.time_ns())))
    assert not temporary.exists() and (overwrite or not path.exists()), str(path)
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


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
    path = Path(path); temporary = path.with_name(path.name + '.pending')
    if temporary.exists() and path.name == 'resume.pt':
        os.replace(temporary, temporary.with_name(temporary.name + '.incomplete_' + str(time.time_ns())))
    assert not temporary.exists(), str(temporary)
    with temporary.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append_event(path, event):
    with Path(path).open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(event, separators=(',', ':'), allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


def prefix_ref(path):
    path = Path(path)
    if not path.exists():
        return dict(bytes=0, sha256=hashlib.sha256(b'').hexdigest())
    return dict(bytes=path.stat().st_size, sha256=sha(path))


def verify_prefix(path, record):
    path = Path(path); expected = int(record['bytes']); h = hashlib.sha256()
    if not path.exists():
        assert expected == 0 and record['sha256'] == h.hexdigest()
        return dict(bytes=0, sha256=h.hexdigest())
    assert path.stat().st_size >= expected, 'Committed training log was truncated'
    with path.open('rb') as stream:
        left = expected
        while left:
            data = stream.read(min(left, 8 << 20)); assert data
            h.update(data); left -= len(data)
        tail = stream.read()
    assert h.hexdigest() == record['sha256'], 'Committed training log prefix changed'
    return dict(bytes=len(tail), sha256=hashlib.sha256(tail).hexdigest())


def seed(namespace):
    return int(hashlib.sha256(('1729|ATMS-standard-SDXL-ordinary|' + namespace).encode()).hexdigest()[:16], 16) % (2**63 - 1)


def rss_highwater():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Linux VmHWM measurement required')


class Resources:
    def __init__(self, maximum_seconds, spec):
        self.started = PROCESS_START; self.maximum_seconds = maximum_seconds
        self.base_cpu = rss_highwater(); self.peak_cpu = self.base_cpu; self.peak_gpu = 0
        self.maximum_cpu = int(spec['maximum_extra_CPU_bytes'])
        self.maximum_gpu = int(spec['maximum_GPU_bytes'])
        self.stop = False; self.last_stage = 'PREPARATION'
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, 'stop', True))
        signal.signal(signal.SIGINT, lambda *_: setattr(self, 'stop', True))

    def check(self, stage=None):
        if stage is not None:
            self.last_stage = stage
        self.peak_cpu = max(self.peak_cpu, rss_highwater())
        self.peak_gpu = max(self.peak_gpu, torch.cuda.max_memory_allocated())
        assert self.peak_cpu - self.base_cpu <= self.maximum_cpu, 'Extra CPU peak limit'
        assert self.peak_gpu <= self.maximum_gpu, 'GPU allocated peak limit'

    def stopping(self):
        return self.stop or time.monotonic() - self.started >= self.maximum_seconds - 90

    def result(self):
        return dict(seconds=time.monotonic() - self.started,
                    peak_extra_CPU_bytes=self.peak_cpu - self.base_cpu,
                    peak_GPU_bytes=self.peak_gpu)


def wrapper_proof():
    parent = Path('/proc') / str(os.getppid()) / 'cmdline'
    command = parent.read_bytes().split(b'\0')
    assert any(Path(part.decode()).name == 'run_gpu_job_v2.py' for part in command if part), 'Use unchanged old-host finite GPU wrapper'
    return dict(wrapper_pid=os.getppid(), parent_command=[part.decode() for part in command if part])


def load_spec(path):
    spec = read(path)
    assert spec['status'] == 'ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_LOCKED_BEFORE_EFFECTS'
    assert spec['seed'] == 1729 and spec['passes'] == PASSES and spec['batch_size'] == BATCH
    assert spec['segment_max_updates'] == 512
    assert spec['maximum_GPU_bytes'] == 20 << 30 and spec['maximum_extra_CPU_bytes'] == 10 << 30
    assert spec['maximum_output_bytes'] == 4 << 30
    assert spec.get('radius', .25) == .25 and spec.get('token_loss_weight', .1) == .1
    assert spec.get('learning_rate', 1e-4) == 1e-4 and spec.get('weight_decay', .01) == .01
    assert spec.get('gradient_clip', 1.) == 1.
    assert spec['CLIP_H_targets']['sha256'] == '9fad7215e04dad0e03d5d95304109974d00b8ed43599a9bbd4eac5f52e0e15eb', 'Use original raw pooled CLIP-H teacher, not BrainHIVE F'
    pinmap = {}
    for item in spec['source_pins']:
        checked(item); pinmap[str(Path(item['path']).resolve())] = item['sha256']
    for key in ('split', 'image_manifest', 'ATMS_export_completion', 'ATMS_input_completion',
                'CLIP_H_targets', 'standard_latents_completion'):
        item = spec[key]; checked(item)
        assert pinmap.get(str(Path(item['path']).resolve())) == item['sha256'], key
    assert pinmap.get(str(Path(__file__).resolve())) == sha(__file__)
    for name in ('profile_standard_generator.py', 'brainhive_bounded_component_v1.py'):
        assert pinmap.get(str((ROOT / name).resolve())) == sha(ROOT / name), name
    return spec


def validate_inputs(spec):
    split = read(checked(spec['split']))
    assert len(split['subjects']) == 10 and split['subjects'] == [f'sub-{i:02d}' for i in range(1, 11)]
    T_rows = split['image_rows']['T']; D_rows = split['image_rows']['D']
    assert len(T_rows) == N_IMAGES and len(D_rows) == 1650 and len(set(T_rows)) == N_IMAGES
    metadata = {int(row['array_row']): row for row in csv.DictReader(checked(spec['image_manifest']).open(encoding='utf-8')) if row['dataset_split'] == 'train'}
    assert len(metadata) == 16540
    assert {metadata[row]['concept_id'] for row in T_rows} == set(split['train_concepts']['T'])
    export = read(checked(spec['ATMS_export_completion']))
    assert export['status'] == 'D15_ALL10_COMPLETE_ATMS_OOF_T_AND_FULL_D_REEXPORTED'
    assert export['T_each_query_from_model_excluding_its_concept'] and export['D_models_fit_only_T']
    assert export['T_queries'] == N_QUERIES and export['D_queries'] == 16500 and export['input_shape'] == [63, 250]
    assert export['inputs']['sha256'] == spec['ATMS_input_completion']['sha256']
    upstream = read(checked(spec['ATMS_input_completion']))
    assert upstream['status'] == 'D15_ALL30_STRICT_MODELS_ORIGINAL_R4_INPUTS_AND_OOF_ROLES_CPU_VERIFIED'
    assert upstream['T_array_rows'] == T_rows and upstream['D_array_rows'] == D_rows
    folds = np.array([split['oof_fold_by_concept'][metadata[row]['concept_id']] for row in T_rows], np.int64)
    assert set(folds.tolist()) == {0, 1}
    for subject in split['subjects']:
        models = upstream['subjects'][subject]['models']
        assert set(models) == {'oof0', 'oof1', 'full'}
        for role in ('oof0', 'oof1'):
            expected = {row for row, fold in zip(T_rows, folds) if fold == int(role[-1])}
            assert set(models[role]['export_array_rows']) == expected
        assert models['full']['export_array_rows'] == D_rows
    for key in ('T_ATMS_OOF', 'D_ATMS_full', 'affines'):
        checked(export[key])
    C_T = np.load(export['T_ATMS_OOF']['path'], mmap_mode='r', allow_pickle=False)
    assert C_T.shape == (N_QUERIES, 1024) and C_T.dtype == np.float32
    affines = torch.load(export['affines']['path'], map_location='cpu', weights_only=True)
    assert set(affines) == {subject + '/' + role for subject in split['subjects'] for role in ('oof0', 'oof1', 'full')}
    for subject in split['subjects']:
        for role in ('oof0', 'oof1', 'full'):
            checkpoint = upstream['subjects'][subject]['models'][role]['checkpoint']
            state = torch.load(checked(checkpoint), map_location='cpu', weights_only=True, mmap=True)
            actual = affines[subject + '/' + role]
            assert torch.equal(state['proj_eeg.2.weight'], actual['weight'])
            assert torch.equal(state['proj_eeg.2.bias'], actual['bias'])
            assert actual['weight'].shape == actual['bias'].shape == (1024,)
            assert torch.isfinite(actual['weight']).all() and torch.isfinite(actual['bias']).all()
            assert actual['weight'].abs().min() > 1e-6 and float(actual['eps']) > 0.
            del state
    weights, biases, affine_records = [], [], []
    for subject in split['subjects']:
        for role in ('oof0', 'oof1'):
            item = affines[subject + '/' + role]
            w, b = item['weight'], item['bias']
            assert w.shape == b.shape == (1024,) and torch.isfinite(w).all() and torch.isfinite(b).all()
            assert w.abs().min() > 1e-6 and float(item['eps']) > 0
            weights.append(w); biases.append(b)
            affine_records.append(dict(subject=subject, role=role, actual_LN_eps=float(item['eps'])))
    targets = torch.load(checked(spec['CLIP_H_targets']), map_location='cpu', weights_only=True, mmap=True)
    assert torch.is_tensor(targets) and targets.shape == (16540, 1024) and targets.dtype == torch.float32
    return dict(split=split, metadata=metadata, T_rows=T_rows, D_rows=D_rows, export=export,
                upstream=upstream, conditions=C_T, folds=folds, target_cache=targets,
                weight=torch.stack(weights), bias=torch.stack(biases), affines=affines,
                affine_records=affine_records)


def load_latents(spec, data, budget):
    done = read(checked(spec['standard_latents_completion']))
    assert done['status'] == 'COMPLETED_ALL13230_T_IMAGE_LATENTS'
    assert done['array_rows'] == data['T_rows'] and done['shape'] == [N_IMAGES, 4, 64, 64]
    assert done['dtype'] == 'float32' and done['V_or_D_or_official_test_images_read'] is False
    assert done['generator_revision'] == '462165984030d82259a11f4367a4eed129e94a7b'
    output = torch.empty((N_IMAGES, 4, 64, 64), dtype=torch.float32, device='cuda')
    covered = np.zeros(N_IMAGES, np.uint8)
    for part in done['parts']:
        path = checked(part); saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        start, stop = int(part['start']), int(part['stop'])
        assert 0 <= start < stop <= N_IMAGES and not covered[start:stop].any()
        assert saved['array_rows'] == data['T_rows'][start:stop]
        assert saved['recipe_sha256'] == done['recipe_sha256']
        values = saved['latents']; assert values.shape == (stop - start, 4, 64, 64)
        assert values.dtype == torch.float32 and torch.isfinite(values).all()
        output[start:stop] = values.cuda(); covered[start:stop] = 1
        del saved, values; budget.check('LOAD_NATIVE_STANDARD_SDXL_T_LATENTS')
    assert covered.all()
    return output


def state_digest(module):
    result = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        result.update(name.encode()); result.update(str(value.dtype).encode()); result.update(str(tuple(value.shape)).encode())
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
        result.update(raw.tobytes())
    return result.hexdigest()


def host_digests(pipe):
    return {name: state_digest(getattr(pipe, name)) for name in ('unet', 'vae', 'text_encoder', 'text_encoder_2')}


def setup_host(spec, budget):
    global torch, F
    from profile_standard_generator import load_pipeline, ASSETS
    import diffusers
    import transformers
    from diffusers import DDPMScheduler
    assert sha(ASSETS / 'adapter.safetensors') == 'ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831'
    pipe, loading, snapshot, prompts = load_pipeline()
    assert snapshot['revision'] == '462165984030d82259a11f4367a4eed129e94a7b'
    assert not any(p.requires_grad for name in ('unet', 'vae', 'text_encoder', 'text_encoder_2') for p in getattr(pipe, name).parameters())
    projection = pipe.unet.encoder_hid_proj.image_projection_layers[0]
    assert projection.image_embeds.in_features == 1024 and sum(p.numel() for p in projection.parameters()) == 8400896
    assert all(not isinstance(module, torch.nn.Dropout) or module.p == 0. for module in pipe.unet.modules())
    prompts = tuple(value.detach().clone() for value in prompts)
    prompt, negative, pool, negative_pool = prompts
    time_ids = pipe._get_add_time_ids((512, 512), (0, 0), (512, 512), dtype=prompt.dtype,
                                     text_encoder_projection_dim=pool.shape[-1]).cuda()
    scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
    assert scheduler.config.prediction_type == 'epsilon' and scheduler.config.num_train_timesteps == 1000
    runtime_sources = sorted({inspect.getfile(type(pipe.unet)), inspect.getfile(type(projection)),
        inspect.getfile(type(scheduler)), inspect.getfile(type(pipe.text_encoder)),
        inspect.getfile(type(pipe.text_encoder_2)),
        *(inspect.getfile(type(processor)) for processor in pipe.unet.attn_processors.values())})
    runtime = dict(python=sys.executable, python_version=sys.version, torch=torch.__version__,
        torch_CUDA_build=torch.version.cuda, cuDNN=torch.backends.cudnn.version(),
        diffusers=diffusers.__version__, transformers=transformers.__version__, numpy=np.__version__,
        GPU_name=torch.cuda.get_device_name(0), imported_sources=[reference(path) for path in runtime_sources],
        UNet_dtype=str(next(pipe.unet.parameters()).dtype), VAE_dtype=str(next(pipe.vae.parameters()).dtype))
    original = host_digests(pipe); budget.check('STRICT_NATIVE_HOST_LOADED')
    return dict(pipe=pipe, P=projection, prompts=prompts, time_ids=time_ids, scheduler=scheduler,
                original=original, loading=loading, snapshot=snapshot, runtime=runtime,
                counts=dict(UNet_forward_calls=0, UNet_sample_evaluations=0))


def predict(host, noisy, timesteps, condition):
    prompt, _, pool, _ = host['prompts']; n = len(condition)
    assert condition.shape == (n, 1024)
    host['counts']['UNet_forward_calls'] += 1; host['counts']['UNet_sample_evaluations'] += n
    return host['pipe'].unet(noisy.to(torch.bfloat16), timesteps,
        encoder_hidden_states=prompt.repeat(n, 1, 1),
        added_cond_kwargs=dict(text_embeds=pool.repeat(n, 1), time_ids=host['time_ids'].repeat(n, 1),
                               image_embeds=[condition[:, None, :]]), return_dict=False)[0]


def project_tokens(host, condition):
    tokens = host['P'](condition)
    assert tokens.shape == (len(condition), 4, 2048)
    return tokens


def configure_gradients(host):
    from torch.utils.checkpoint import checkpoint
    unet = host['pipe'].unet; unet.train()
    kwargs = {}
    if 'gradient_checkpointing_func' in inspect.signature(unet.enable_gradient_checkpointing).parameters:
        kwargs['gradient_checkpointing_func'] = functools.partial(checkpoint, use_reentrant=False)
    unet.enable_gradient_checkpointing(**kwargs)
    if not kwargs:
        # Older Diffusers modules expose this callable after enabling checkpointing.
        for module in unet.modules():
            if hasattr(module, '_gradient_checkpointing_func'):
                module._gradient_checkpointing_func = functools.partial(checkpoint, use_reentrant=False)
    return dict(enabled=True, custom_nonreentrant_argument=bool(kwargs))


def new_adapter():
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
    model = BoundedConditionAdapter(predict_risk=False).cuda()
    assert sum(p.numel() for p in model.parameters()) == 525568
    assert model.risk is None and torch.count_nonzero(model.direction.weight) == 0
    return model


def training_batch(data, indices, target_T, latents):
    ids = indices.cpu().numpy(); positions = ids % N_IMAGES; subjects = ids // N_IMAGES
    condition = torch.from_numpy(np.array(data['conditions'][ids], copy=True)).cuda()
    assert torch.isfinite(condition).all()
    affine_ids = torch.from_numpy(2 * subjects + data['folds'][positions]).cuda()
    return (condition, data['weight_cuda'][affine_ids], data['bias_cuda'][affine_ids],
            target_T[torch.from_numpy(positions).cuda()], latents[torch.from_numpy(positions).cuda()])


def update(model, optimizer, host, batch, noise_rng, *, verify_epsilon_gradient=False):
    c, w, b, target, latent = batch; n = len(c)
    noise = torch.randn(latent.shape, device='cuda', dtype=torch.float32, generator=noise_rng)
    timesteps = torch.randint(0, 1000, (n,), device='cuda', generator=noise_rng)
    noisy = host['scheduler'].add_noise(latent, noise, timesteps)
    optimizer.zero_grad(set_to_none=True)
    changed, info = model(c, w, b, gate_mode='fixed')
    with torch.autocast('cuda', dtype=torch.bfloat16):
        with torch.no_grad():
            teacher = project_tokens(host, target)
        tokens = project_tokens(host, changed)
        prediction = predict(host, noisy, timesteps, changed)
        epsilon_loss = F.mse_loss(prediction.float(), noise)
        token_loss = (1. - F.cosine_similarity(tokens.float(), teacher.float(), dim=-1)).mean()
        loss = epsilon_loss + .1 * token_loss
    assert torch.isfinite(loss)
    epsilon_gradient_norm = None
    if verify_epsilon_gradient:
        epsilon_gradient = torch.autograd.grad(epsilon_loss, model.direction.weight,
            retain_graph=True, allow_unused=True)[0]
        assert epsilon_gradient is not None and torch.isfinite(epsilon_gradient).all(), 'Missing epsilon-only input gradient'
        epsilon_gradient_norm = float(epsilon_gradient.float().norm())
        assert epsilon_gradient_norm > 0., 'Token loss alone cannot validate frozen-generator gradient'
    loss.backward()
    assert model.direction.weight.grad is not None and torch.isfinite(model.direction.weight.grad).all()
    assert float(model.direction.weight.grad.float().norm()) > 0., 'Missing real generator gradient to ordinary'
    assert all(p.grad is None for p in host['pipe'].unet.parameters()), 'Frozen generator has gradients'
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
    optimizer.step()
    return dict(loss=float(loss.detach()), epsilon_loss=float(epsilon_loss.detach()),
                token_loss=float(token_loss.detach()), grad_norm_preclip=float(norm),
                batch=n, timesteps=timesteps.detach().cpu().tolist(),
                correction_RMS=float(info['correction'].detach().square().mean().sqrt()),
                epsilon_only_gradient_probe_norm=epsilon_gradient_norm)


def rng_state(noise_rng):
    return dict(python=random.getstate(), numpy=np.random.get_state(), CPU=torch.get_rng_state(),
                CUDA=torch.cuda.get_rng_state_all(), noise=noise_rng.get_state())


def restore_rng(value, noise_rng):
    random.setstate(value['python']); np.random.set_state(value['numpy']); torch.set_rng_state(value['CPU'])
    torch.cuda.set_rng_state_all(value['CUDA']); noise_rng.set_state(value['noise'])


def order_sha(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def make_orders():
    return [torch.randperm(N_QUERIES, generator=torch.Generator().manual_seed(seed('pass' + str(i)))) for i in range(PASSES)]


def verify_cursor(cursor, updates, seen, coverage, orders):
    assert type(cursor) is int and 0 <= cursor <= EXPECTED_EXPOSURES and seen == cursor
    assert coverage.shape == (PASSES, N_QUERIES) and coverage.dtype == torch.bool
    complete, offset = divmod(cursor, N_QUERIES)
    assert updates == complete * math.ceil(N_QUERIES / BATCH) + math.ceil(offset / BATCH)
    assert offset == 0 or offset % BATCH == 0
    for epoch in range(PASSES):
        expected = torch.zeros(N_QUERIES, dtype=torch.bool)
        if epoch < complete:
            expected[:] = True
        elif epoch == complete:
            expected[orders[epoch][:offset]] = True
        assert torch.equal(coverage[epoch], expected), 'Coverage/order prefix mismatch'


def source_reverify(spec, output):
    for item in spec['source_pins']:
        checked(item)
    total = sum(path.stat().st_size for path in Path(output).rglob('*') if path.is_file())
    assert total < int(spec['maximum_output_bytes']), 'Output allocation cap exceeded'


def provenance(spec_path, spec, data, host, profile=None):
    return dict(spec=reference(spec_path), source=reference(__file__),
        export=spec['ATMS_export_completion'], input_contract=spec['ATMS_input_completion'],
        raw_CLIP_H_target=spec['CLIP_H_targets'], standard_latents=spec['standard_latents_completion'],
        affine_source=data['export']['affines'], affine_roles=data['affine_records'],
        original_host_state_digests=host['original'], generator_revision=host['snapshot']['revision'],
        runtime=host['runtime'], original_ATMS_V_checkpoint_selection_preserved=True,
        source_pins_sha256=canonical_sha(spec['source_pins']), profile=profile,
        scope='all10 genuine63-channel R4-derived concept-OOF T, fresh ordinary only',
        parameters=525568, passes=PASSES, queries_per_pass=N_QUERIES,
        raw_target_coordinates='pooled CLIP-H, no BrainHIVE F/A', seed=1729)


def setup_training_data(spec, data, host, budget):
    data['weight_cuda'] = data['weight'].cuda(); data['bias_cuda'] = data['bias'].cuda()
    target_T = data['target_cache'][torch.tensor(data['T_rows'])].clone().cuda()
    assert target_T.shape == (N_IMAGES, 1024) and torch.isfinite(target_T).all()
    latents = load_latents(spec, data, budget)
    return target_T, latents


def run_profile(args, spec, data, host, budget):
    output = args.output / 'profile'; assert not output.exists(); output.mkdir(parents=True)
    target_T, latents = setup_training_data(spec, data, host, budget)
    model = new_adapter(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    orders = make_orders(); rng = torch.Generator(device='cuda').manual_seed(seed('profile_noise'))
    batch = training_batch(data, orders[0][:BATCH], target_T, latents)
    c, w, b, _, latent = batch
    torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
    noise = torch.randn(latent.shape, device='cuda', generator=rng)
    timesteps = torch.full((BATCH,), 501, dtype=torch.long, device='cuda')
    noisy = host['scheduler'].add_noise(latent, noise, timesteps)
    model.eval(); host['pipe'].unet.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        zero, _ = model(c, w, b, gate_mode='fixed'); off, _ = model(c, w, b, enabled=False)
        assert torch.equal(zero, c) and torch.equal(off, c)
        native_prediction = predict(host, noisy, timesteps, c)
        zero_prediction = predict(host, noisy, timesteps, zero)
        off_prediction = predict(host, noisy, timesteps, off)
        assert torch.equal(native_prediction, zero_prediction) and torch.equal(native_prediction, off_prediction)
    gradients = configure_gradients(host); model.train(); steps = []
    for i in range(2):
        budget.check('PROFILE_REAL_UPDATE_' + str(i + 1)); assert not budget.stopping()
        batch = training_batch(data, orders[0][i * BATCH:(i + 1) * BATCH], target_T, latents)
        torch.cuda.synchronize(); tick = time.monotonic()
        event = update(model, optimizer, host, batch, rng, verify_epsilon_gradient=(i == 0)); torch.cuda.synchronize()
        event['seconds'] = time.monotonic() - tick; steps.append(event)
        budget.check()
    actual_host = host_digests(host['pipe']); assert actual_host == host['original']
    source_reverify(spec, args.output); budget.check('PROFILE_COMPLETE')
    done = dict(status='ATMS_STANDARD_ORDINARY_ENGINEERING_PROFILE_PASSED',
        provenance=provenance(args.spec, spec, data, host), steps=steps,
        zero_head_condition_exact=True, disabled_condition_exact=True,
        zero_and_disabled_native_noise_predictions_exact=True,
        nonzero_finite_adapter_gradient_through_frozen_P_IP_UNet=True,
        all_frozen_host_tensor_digests_unchanged=True,
        gradient_checkpointing=gradients, actual_calls=host['counts'],
        engineering_updates_discarded=2, scientific_updates=0, new_images=0,
        epsilon_only_gradient_probes=1,
        Dfit_Dcheck_values_read=False, **budget.result())
    atomic_json(output / 'completion.json', done)
    print(json.dumps(dict(status=done['status'], completion=reference(output / 'completion.json'))), flush=True)


def run_train(args, spec, data, host, budget):
    output = args.output / 'train'; profile_path = args.output / 'profile/completion.json'
    profile = read(profile_path)
    assert profile['status'] == 'ATMS_STANDARD_ORDINARY_ENGINEERING_PROFILE_PASSED'
    assert profile['provenance']['spec']['sha256'] == sha(args.spec)
    assert profile['provenance']['source']['sha256'] == sha(__file__)
    assert profile['provenance']['original_host_state_digests'] == host['original']
    assert profile['provenance']['runtime'] == host['runtime'], 'Profile/training runtime changed'
    assert not (output / 'completion.json').exists()
    if args.resume:
        assert (output / 'resume.pt').exists() and (output / 'input_provenance.json').exists()
    else:
        assert not output.exists(); output.mkdir(parents=True)
    lock_value = provenance(args.spec, spec, data, host, reference(profile_path))
    lock = output / 'input_provenance.json'
    if lock.exists():
        assert read(lock) == lock_value
    else:
        atomic_json(lock, lock_value)
    target_T, latents = setup_training_data(spec, data, host, budget)
    model = new_adapter(); optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
    gradients = configure_gradients(host); model.train()
    noise_rng = torch.Generator(device='cuda').manual_seed(seed('shared_training_noise'))
    orders = make_orders(); orders_sha = [order_sha(order) for order in orders]
    cursor = updates = seen = 0; loss_sum = cumulative_seconds = 0.
    coverage = torch.zeros((PASSES, N_QUERIES), dtype=torch.bool)
    training_log = output / 'training.jsonl'; resumed_from = None
    if args.resume:
        saved = torch.load(output / 'resume.pt', map_location='cpu', weights_only=False)
        assert saved['status'] == 'COMMITTED_COMPLETE_UPDATE_BOUNDARY'
        assert saved['input_provenance_sha256'] == sha(lock) and saved['orders_sha256'] == orders_sha
        assert all(torch.equal(left, right) for left, right in zip(saved['orders'], orders))
        assert saved['original_host_state_digests'] == host['original']
        verify_cursor(saved['cursor'], saved['updates'], saved['seen'], saved['coverage'], orders)
        tail = verify_prefix(training_log, saved['log_prefix'])
        resumed_from = dict(checkpoint=reference(output / 'resume.pt'), updates=saved['updates'],
                            cursor=saved['cursor'], committed_log_prefix=saved['log_prefix'], uncommitted_tail=tail)
        # Preserve an uncommitted journal tail rather than deleting or relabelling it.
        append_event(training_log, dict(event='EXACT_CHECKPOINT_RESUME', **resumed_from,
                     uncommitted_attempts_are_not_on_the_final_optimizer_path=True))
        model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
        cursor, updates, seen = saved['cursor'], saved['updates'], saved['seen']
        loss_sum, cumulative_seconds = saved['loss_sum'], saved['training_seconds']
        coverage = saved['coverage']; restore_rng(saved['RNG'], noise_rng)
    else:
        random.seed(1729); np.random.seed(1729); torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
    segment_start_updates = updates; segment_start_time = time.monotonic()
    def snapshot(reason):
        verify_cursor(cursor, updates, seen, coverage, orders)
        state = dict(status='COMMITTED_COMPLETE_UPDATE_BOUNDARY', model=cpu_tree(model.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()), RNG=rng_state(noise_rng), cursor=cursor,
            updates=updates, seen=seen, loss_sum=loss_sum, coverage=coverage.clone(),
            orders=[order.clone() for order in orders], orders_sha256=orders_sha,
            input_provenance_sha256=sha(lock), original_host_state_digests=host['original'],
            log_prefix=prefix_ref(training_log), reason=reason,
            training_seconds=cumulative_seconds + time.monotonic() - segment_start_time)
        atomic_torch(output / 'resume.pt', state)
        append_event(output / 'checkpoint_commits.jsonl', dict(checkpoint=reference(output / 'resume.pt'),
            cursor=cursor, updates=updates, reason=reason, log_prefix=state['log_prefix'],
            RNG_tensor_sha256=dict(CPU=order_sha(state['RNG']['CPU']), noise=order_sha(state['RNG']['noise']),
                                  CUDA=[order_sha(value) for value in state['RNG']['CUDA']])))
    snapshot('SEGMENT_START_COMMITTED_STATE')
    stop_reason = None
    while cursor < EXPECTED_EXPOSURES:
        if budget.stopping() or updates - segment_start_updates >= 512:
            stop_reason = 'SIGNAL_OR_FINITE_TIME' if budget.stopping() else 'FIXED_512_UPDATE_SEGMENT'
            break
        budget.check('REAL_FULL_OOF_TRAINING')
        epoch, offset = divmod(cursor, N_QUERIES); ids = orders[epoch][offset:offset + BATCH]
        assert len(ids) == (12 if offset + BATCH > N_QUERIES else BATCH)
        assert not coverage[epoch, ids].any()
        event = update(model, optimizer, host, training_batch(data, ids, target_T, latents), noise_rng)
        coverage[epoch, ids] = True; cursor += len(ids); seen += len(ids); updates += 1
        loss_sum += event['loss'] * len(ids)
        append_event(training_log, dict(event='REAL_COMPLETE_OPTIMIZER_UPDATE', update=updates,
            pass_number=epoch + 1, offset_in_pass=offset, cursor=cursor, seen=seen,
            query_indices_sha256=order_sha(ids), **event))
        if updates % 32 == 0 or cursor % N_QUERIES == 0:
            snapshot('REGISTERED_32_UPDATE_OR_PASS_BOUNDARY')
            progress = dict(status='RUNNING_FULL132300x3_ORDINARY_TRAINING', updates=updates,
                expected_updates=EXPECTED_UPDATES, seen=seen, expected_seen=EXPECTED_EXPOSURES,
                complete_passes=cursor // N_QUERIES, cumulative_mean_loss=loss_sum / seen,
                segment_updates=updates - segment_start_updates, last_losses=event, **budget.result())
            atomic_json(output / 'progress.json', progress, overwrite=True); print(json.dumps(progress), flush=True)
    snapshot('FULL_FINAL_UPDATE' if cursor == EXPECTED_EXPOSURES else stop_reason)
    assert host_digests(host['pipe']) == host['original'], 'Frozen native host changed'
    source_reverify(spec, args.output); budget.check('SEGMENT_SOURCE_AND_FROZEN_STATE_CLOSED')
    segment = dict(status='TRAINING_SEGMENT_COMPLETED_RESUMABLE', segment_updates=updates - segment_start_updates,
        updates=updates, seen=seen, cursor=cursor, expected_updates=EXPECTED_UPDATES,
        expected_seen=EXPECTED_EXPOSURES, checkpoint=reference(output / 'resume.pt'),
        input_provenance=reference(lock), resumed_from=resumed_from,
        exact_order_CPU_CUDA_noise_rng_coverage_preserved=True,
        all_frozen_host_tensor_digests_unchanged=True, gradient_checkpointing=gradients,
        actual_calls=host['counts'], training_complete_is_not_image_success=True,
        Dfit_Dcheck_values_read=False, new_images=0, **budget.result())
    if cursor == EXPECTED_EXPOSURES:
        assert updates == EXPECTED_UPDATES and seen == EXPECTED_EXPOSURES and coverage.all()
        final = dict(model=cpu_tree(model.state_dict()), arm=ARMS, seed=1729, passes=PASSES,
            updates=updates, seen_queries=seen, parameters=525568,
            input_provenance=reference(lock), spec=reference(args.spec),
            original_ordinary_HIVE_weights_loaded=False, checkpoint_selection='Fixed last registered update')
        atomic_torch(output / 'component_final.pt', final)
        segment.update(status='COMPLETE_ALL10_ATMS_OOF_THREE_PASSES_ORDINARY_TRAINING',
            final_checkpoint=reference(output / 'component_final.pt'),
            training_seconds=cumulative_seconds + time.monotonic() - segment_start_time,
            per_pass_unique_query_coverage=coverage.sum(1).tolist(),
            native_conditions='raw pooledCLIP-H, real role-specific ATMS LN affine',
            inference='Original standardSDXL512/20Euler/CFG5/IP1; oneimage/query; no extra UNet calls')
        atomic_json(output / 'completion.json', segment)
    else:
        if segment['segment_updates'] == 0:
            segment['status'] = 'RESOURCE_STOP_BEFORE_ANY_NEW_UPDATE_NO_AUTOMATIC_SPIN'
        atomic_json(output / ('segment_' + str(time.time_ns()) + '.json'), segment)
    atomic_json(output / 'progress.json', segment, overwrite=True)
    print(json.dumps(segment), flush=True)
    return 0 if segment['segment_updates'] > 0 or cursor == EXPECTED_EXPOSURES else 2


def run_calibrate(args, spec, data, host, budget):
    output = args.output / 'calibrate'; assert not output.exists(); output.mkdir(parents=True)
    complete_path = args.output / 'train/completion.json'; complete = read(complete_path)
    assert complete['status'] == 'COMPLETE_ALL10_ATMS_OOF_THREE_PASSES_ORDINARY_TRAINING'
    assert complete['updates'] == EXPECTED_UPDATES and complete['seen'] == EXPECTED_EXPOSURES
    checked(complete['final_checkpoint']); final = torch.load(complete['final_checkpoint']['path'], map_location='cpu', weights_only=True)
    assert final['spec']['sha256'] == sha(args.spec) and final['updates'] == EXPECTED_UPDATES
    trained_provenance = read(checked(final['input_provenance']))
    assert trained_provenance['original_host_state_digests'] == host['original']
    assert trained_provenance['runtime'] == host['runtime'], 'Training/calibration runtime changed'
    model = new_adapter().eval(); model.load_state_dict(final['model'], strict=True); host['pipe'].unet.eval()
    fit_concepts = set(data['split']['d_fit_concepts']); check_concepts = set(data['split']['d_check_concepts'])
    assert not fit_concepts.intersection(check_concepts)
    positions = [i for i, row in enumerate(data['D_rows']) if data['metadata'][row]['concept_id'] in fit_concepts]
    fit_rows = [data['D_rows'][i] for i in positions]; assert len(fit_rows) == 820
    assert all(data['metadata'][row]['concept_id'] not in check_concepts for row in fit_rows)
    dcache = np.load(checked(data['export']['D_ATMS_full']), mmap_mode='r', allow_pickle=False)
    assert dcache.shape == (16500, 1024) and dcache.dtype == np.float32
    target = data['target_cache'][torch.tensor(fit_rows)].clone().cuda(); sums = np.zeros(5, np.float64); counts = np.zeros(5, np.int64)
    with torch.no_grad():
        for si, subject in enumerate(data['split']['subjects']):
            affine = data['affines'][subject + '/full']; weight = affine['weight'].cuda(); bias = affine['bias'].cuda()
            for lo in range(0, 820, 256):
                budget.check('DFIT_ONLY_FIXED_FINAL_STRENGTH_CALIBRATION'); assert not budget.stopping()
                local = positions[lo:lo + 256]; selected = si * 1650 + np.array(local)
                c = torch.from_numpy(np.array(dcache[selected], copy=True)).cuda(); n = len(c)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    teacher = project_tokens(host, target[lo:lo + n])
                    for ai, alpha in enumerate(STRENGTHS):
                        model.radius = .25 * alpha
                        changed = c if alpha == 0. else model(c, weight.expand(n, -1), bias.expand(n, -1), gate_mode='fixed')[0]
                        tokens = project_tokens(host, changed)
                        error = (1. - F.cosine_similarity(tokens.float(), teacher.float(), dim=-1)).mean(-1)
                        assert torch.isfinite(error).all(); sums[ai] += float(error.double().sum().cpu()); counts[ai] += n
    model.radius = .25; assert (counts == 8200).all(); errors = sums / counts
    index = min(range(5), key=lambda i: (float(errors[i]), STRENGTHS[i])); selected = STRENGTHS[index]
    assert host_digests(host['pipe']) == host['original']; source_reverify(spec, args.output); budget.check('CALIBRATION_COMPLETE')
    done = dict(status='ATMS_STANDARD_ORDINARY_DFIT_ONLY_FIXED_GRID_CALIBRATION_COMPLETE',
        train_completion=reference(complete_path), fixed_final_checkpoint=complete['final_checkpoint'],
        spec=reference(args.spec), source=reference(__file__), strengths=list(STRENGTHS),
        Dfit_token_angular_errors=errors.tolist(), Dfit_queries=8200, selected_strength=selected,
        scalar_is_inside_radius=True, radius_before_strength=.25, ties_prefer_smaller_strength=True,
        all_frozen_host_tensor_digests_unchanged=True, actual_calls=host['counts'],
        Dcheck_tensor_values_selected=0, Dcheck_or_official_test_used_to_select=False,
        Dfit_array_rows=fit_rows, raw_target_coordinates='Native pooledCLIP-H',
        no_new_training_updates=True, image_gain_established=False, new_images=0, **budget.result())
    atomic_json(output / 'calibration.json', done)
    calibrated = dict(final); calibrated['calibration'] = done
    atomic_torch(output / 'component_calibrated.pt', calibrated)
    atomic_json(output / 'completion.json', dict(done, calibrated_checkpoint=reference(output / 'component_calibrated.pt')))
    print(json.dumps(dict(status=done['status'], selected_strength=selected,
        completion=reference(output / 'completion.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('profile', 'train', 'calibrate'), required=True)
    parser.add_argument('--max-seconds', type=int, required=True); parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); assert 120 <= args.max_seconds <= 1800 and (not args.resume or args.mode == 'train')
    wrapper = wrapper_proof(); spec = load_spec(args.spec)
    global torch, F
    # The standard loader installs its original runtime path BEFORE Torch loads.
    from profile_standard_generator import torch as _torch
    import torch.nn.functional as _functional
    torch, F = _torch, _functional
    torch.set_num_threads(2); torch.cuda.set_device(0); assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
    budget = Resources(args.max_seconds, spec); args.output.mkdir(parents=True, exist_ok=True)
    try:
        data = validate_inputs(spec); host = setup_host(spec, budget); budget.check('INPUTS_AND_HOST_READY')
        if args.mode == 'profile':
            run_profile(args, spec, data, host, budget); return 0
        if args.mode == 'train':
            return run_train(args, spec, data, host, budget)
        run_calibrate(args, spec, data, host, budget); return 0
    except BaseException as exc:
        # Do not snapshot partially executed optimizer updates or CUDA failures.
        # Last atomic complete-update resume.pt is authoritative; journal tails stay.
        failure = dict(status='ATMS_STANDARD_ORDINARY_FINITE_ATTEMPT_FAILED_PRESERVED',
            mode=args.mode, stage=budget.last_stage, error=repr(exc), traceback=traceback.format_exc(),
            wrapper=wrapper, last_committed_checkpoint=(reference(args.output / 'train/resume.pt')
                if (args.output / 'train/resume.pt').exists() else None),
            in_memory_partial_optimizer_state_not_claimed_committed=True,
            no_automatic_retry=True, new_images=0, **budget.result())
        atomic_json(args.output / ('failure_' + str(time.time_ns()) + '.json'), failure)
        raise


if __name__ == '__main__':
    raise SystemExit(main())
