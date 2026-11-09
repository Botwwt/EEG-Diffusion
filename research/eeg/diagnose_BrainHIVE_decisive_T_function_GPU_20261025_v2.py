"""Registered, frozen T-only diagnostics; never fits or starts a candidate.

The ordinary import and --check-contract path use only the standard library.
Real execution requires all four fixed finals and an exclusive run_gpu_job_v2
parent. Ground-truth images enter DDPM probes/teachers only; they never enter
the native parent sampling trajectory. A trajectory has no known noise target.
"""
import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parent
ARMS = ('O', 'P', 'PN', 'S')
DOMAINS = ('DDPM_real_T', 'native_parent_trajectory')
SCHEMA = 'BrainHIVE_decisive_T_function_GPU_20261025_v2'
COMPLETE = 'COMPLETE_FROZEN_T_ONLY_FUNCTIONAL_DIAGNOSTIC'
SOURCE_NAMES = (
    Path(__file__).name, 'brainhive_bounded_component_v1.py',
    'brainhive_component_generator_v1.py', 'brainhive_native_models_v1.py',
    'brainhive_response_rank_v1.py', 'brainhive_functional_alignment_v2.py',
    'brainhive_functional_training_contract_v2.py',
)
EPSILON = 1e-6


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with path.open('xb') as stream:
        stream.write(payload)


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, (1729,) + parts)).encode()).hexdigest()[:16], 16) % (2**63 - 1)


def mean(values):
    values = list(values)
    return statistics.fmean(values) if values else None


def decision(result, evidence, interpretation):
    assert result in ('SUPPORTED', 'NOT_SUPPORTED', 'UNKNOWN')
    return dict(result=result, evidence=evidence, interpretation=interpretation)


def classify_directions(values):
    """A directional probe, not a significance test or image-quality gate."""
    if not values or any(value is None or not math.isfinite(value) for value in values):
        return 'UNKNOWN'
    if all(value > 0 for value in values):
        return 'SUPPORTED'
    if all(value <= 0 for value in values):
        return 'NOT_SUPPORTED'
    return 'UNKNOWN'


def validate_registration(document):
    assert document['schema'] == SCHEMA
    assert document['status'] == 'REGISTERED_T_ONLY_FROZEN_DIAGNOSTIC'
    assert document['seed'] == 1729 and document['batch_size'] == 2
    assert set(document['formal_recipes']) == set(ARMS)
    assert 0 < document['maximum_seconds'] <= 7200
    assert 0 < document['maximum_output_bytes'] <= 512 << 20
    assert 0 < document['maximum_GPU_bytes'] <= 20 << 30
    assert document['minimum_CPU_headroom_bytes'] >= 10 << 30
    assert document.get('maximum_extra_CPU_bytes', 10 << 30) <= 10 << 30
    assert Path(document['output_directory']).is_absolute()
    assert document['gpu_job_name'] == Path(document['gpu_job_name']).name
    assert set(SOURCE_NAMES) <= set(document['source_sha256'])
    for name in SOURCE_NAMES:
        assert sha(ROOT / name) == document['source_sha256'][name], ('source mismatch', name)
    for reference in document['formal_recipes'].values():
        assert Path(reference['path']).is_absolute() and sha(reference['path']) == reference['sha256']
    auth = document['resource_authorization']
    assert Path(auth['path']).is_absolute() and sha(auth['path']) == auth['sha256']
    return document


def inspect_finals(document, *, load_tensors=False):
    from brainhive_functional_training_contract_v2 import core_sha
    result = {}
    common = None
    for arm in ARMS:
        recipe = read(document['formal_recipes'][arm]['path'])
        assert recipe['arm'] == arm and recipe['candidate'] == 'functional_alignment_v2_' + arm
        training = Path(recipe['output_directory'])
        done_path = training / 'completion.json'
        done = read(done_path)
        assert done['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
        assert done['updates'] == 24807 and done['seen_queries'] == 396900
        assert done['unique_training_queries'] == 132300 and done['subjects'] == 10 and done['passes'] == 3
        assert done['parameters'] == 525568 and done['recipe_core_sha256'] == core_sha(recipe)
        checkpoint = Path(done['checkpoint_path'])
        assert checkpoint.resolve() == (training / 'component_final.pt').resolve()
        assert sha(checkpoint) == done['checkpoint_sha256']
        provenance_path = training / 'input_provenance.json'
        assert sha(provenance_path) == done['input_provenance_sha256']
        # Existing scalar deployment choice is metadata, not a new calibration.
        calibration_path = training / 'calibration.json'
        assert sha(calibration_path) == done['calibration_sha256']
        calibration = read(calibration_path)
        assert calibration['kind'] == recipe['strength_policy']
        assert calibration['selected_strength'] in (0, .25, .5, .75, 1)
        identity = {key: recipe[key] for key in ('input_contract_sha256', 'latent_receipt', 'generator', 'reference_Q_policy')}
        if common is None:
            common = identity
        else:
            assert identity == common, ('four-arm common host/input mismatch', arm)
        entry = dict(recipe=recipe, completion=done, checkpoint_path=str(checkpoint),
                     checkpoint_sha256=done['checkpoint_sha256'], completion_path=str(done_path),
                     completion_sha256=sha(done_path), provenance_sha256=sha(provenance_path),
                     calibration_sha256=sha(calibration_path), selected_strength=calibration['selected_strength'])
        if load_tensors:
            import torch
            state = torch.load(checkpoint, weights_only=True, map_location='cpu')
            assert state['arm'] == arm and state['updates'] == 24807 and state['seen_queries'] == 396900
            assert state['recipe_core_sha256'] == core_sha(recipe)
            assert state['input_provenance_sha256'] == done['input_provenance_sha256']
            assert state['calibration']['selected_strength'] == entry['selected_strength']
            assert state['risk_gate'] is False and state['trained_from_scratch'] is True
            entry['state'] = state['model']
        result[arm] = entry
    return result


def choose_rows(protocol, metadata, contracts):
    """Same 8 T stimuli for 10 subjects; 4 per OOF source role, no score access."""
    allowed = set(protocol['image_rows']['T'])
    assert len(allowed) == 13230 and len(contracts['subjects']) == 10
    assert allowed.isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    candidates, routes = {}, {}
    subjects = sorted(contracts['subjects'], key=lambda row: row['subject'])
    assert [row['subject'] for row in subjects] == ['sub-%02d' % i for i in range(1, 11)]
    for subject in subjects:
        by_row = {}
        for source in subject['sources']:
            if source['role'] not in ('oof0', 'oof1'):
                continue
            for route in source['groups']['component_train']:
                index = route['array_row']
                assert index in allowed and index not in by_row
                by_row[index] = dict(source_role=source['role'], cache_row=route['cache_row'])
        assert set(by_row) == allowed
        routes[subject['subject']] = by_row
    for index in allowed:
        meta = metadata[index]
        assert meta['dataset_split'] == 'train'
        roles = {routes[subject['subject']][index]['source_role'] for subject in subjects}
        assert len(roles) == 1, 'OOF role assignment must agree for each held concept across subjects'
        role = roles.pop()
        candidates.setdefault(role, {}).setdefault(meta['concept_id'], []).append(index)
    chosen = []
    for role in ('oof0', 'oof1'):
        concepts = sorted(candidates[role], key=lambda key: stable_seed('concept', role, key))[:4]
        assert len(concepts) == 4
        for concept in concepts:
            index = min(candidates[role][concept], key=lambda row: stable_seed('stimulus', row))
            chosen.append(dict(array_row=index, concept_id=concept, stimulus_id=metadata[index]['stimulus_id'], source_role=role))
    assert len({item['concept_id'] for item in chosen}) == 8
    result = []
    for subject in subjects:
        for position, item in enumerate(chosen):
            negative_position = position // 4 * 4 + (position + 1) % 4
            negative = chosen[negative_position]
            route = routes[subject['subject']][item['array_row']]
            neg_route = routes[subject['subject']][negative['array_row']]
            assert route['source_role'] == neg_route['source_role'] == item['source_role']
            assert item['concept_id'] != negative['concept_id']
            result.append(dict(**item, subject=subject['subject'], cache_row=route['cache_row'],
                negative_array_row=negative['array_row'], negative_cache_row=neg_route['cache_row'],
                negative_concept_id=negative['concept_id'], negative_query_index=len(result) - position + negative_position,
                trajectory_seed=stable_seed('native_parent_trajectory', subject['subject'], item['stimulus_id'])))
    assert len(result) == 80 and all(result[row['negative_query_index']]['subject'] == row['subject'] for row in result)
    return result


def verify_wrapper(document):
    parent = Path('/proc') / str(os.getppid())
    command = parent.joinpath('cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in command and b'--name' in command
    assert command[command.index(b'--name') + 1].decode() == document['gpu_job_name']
    lock_path = ROOT / 'audit/GPU_JOB.lock'
    identity = lock_path.stat()
    assert any((fd.stat().st_dev, fd.stat().st_ino) == (identity.st_dev, identity.st_ino)
               for fd in parent.joinpath('fd').iterdir() if fd.exists())
    import fcntl
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            fcntl.flock(lock, fcntl.LOCK_UN)
            raise AssertionError('Trusted wrapper does not hold GPU lock')
    job = read(ROOT / 'audit' / (document['gpu_job_name'] + '_job.json'))
    assert job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
    assert 0 < job['max_seconds'] <= document['maximum_seconds']
    assert job['authorization_sha256'] == document['resource_authorization']['sha256']
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                            check=True, capture_output=True, text=True, timeout=10)
    assert {int(line) for line in active.stdout.splitlines() if line.strip()} <= {os.getpid()}
    return dict(wrapper_pid=os.getppid(), job_name=document['gpu_job_name'])


def process_rss():
    # VmHWM records transient native-loader peaks between budget checkpoints.
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise AssertionError('Missing Linux peak RSS measurement')


class Budget:
    def __init__(self, document):
        self.document, self.started = document, time.monotonic()
        self.baseline_rss = process_rss()
        self.uncompressed_bytes = 0
        self.peak_extra_CPU_bytes = 0
        self.peak_GPU_bytes = 0

    def check(self):
        from brainhive_functional_training_contract_v2 import resource_headroom
        assert time.monotonic() - self.started < self.document['maximum_seconds'], 'Diagnostic time budget exhausted'
        self.peak_extra_CPU_bytes = max(self.peak_extra_CPU_bytes, process_rss() - self.baseline_rss)
        assert self.peak_extra_CPU_bytes <= self.document.get('maximum_extra_CPU_bytes', 10 << 30)
        assert resource_headroom()['effective_headroom_bytes'] >= self.document['minimum_CPU_headroom_bytes']
        assert self.uncompressed_bytes <= self.document['maximum_output_bytes'] - (16 << 20)
        if 'torch' in globals():
            self.peak_GPU_bytes = max(self.peak_GPU_bytes, torch.cuda.max_memory_allocated())
            assert self.peak_GPU_bytes <= self.document['maximum_GPU_bytes'], 'GPU allocation budget exceeded'

    def append(self, stream, value):
        payload = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()
        self.uncompressed_bytes += len(payload)
        assert self.uncompressed_bytes < self.document['maximum_output_bytes'] - (16 << 20)
        stream.write(payload)


def selected_inputs(recipe, budget):
    import numpy as np
    work = Path('/root/brainhive_reproduction_20261020_v1')
    protocol_path = ROOT / 'configs/split_manifest.json'
    assert sha(protocol_path) == recipe['input_sha256'][str(protocol_path)]
    protocol = read(protocol_path)
    manifest_path = ROOT / 'audit/image_manifest.csv'
    assert sha(manifest_path) == protocol['image_manifest_sha256']
    contract_path = work / 'data/component_input_contract_v1.json'
    assert sha(contract_path) == recipe['input_contract_sha256']
    contract = read(contract_path)
    assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    metadata = {}
    allowed = set(protocol['image_rows']['T'])
    with manifest_path.open(encoding='utf-8', newline='') as stream:
        for item in csv.DictReader(stream):
            if item['dataset_split'] == 'train' and int(item['array_row']) in allowed:
                index = int(item['array_row'])
                assert index not in metadata
                metadata[index] = item
    assert set(metadata) == allowed
    rows = choose_rows(protocol, metadata, contract)
    c, weights, biases, provenance = [], [], [], []
    for subject in sorted(contract['subjects'], key=lambda item: item['subject']):
        subject_rows = [item for item in rows if item['subject'] == subject['subject']]
        routed = {}
        for source in subject['sources']:
            if source['role'] not in ('oof0', 'oof1'):
                continue
            budget.check()
            assert sha(source['condition_cache_path']) == source['condition_cache_sha256']
            assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
            cache = torch.load(source['condition_cache_path'], weights_only=True, map_location='cpu', mmap=True)
            brain = torch.load(source['checkpoint_path'], weights_only=True, map_location='cpu')['model']
            assert cache['model_sha256'] == source['checkpoint_sha256']
            w, b = brain['block.layers.0.norm.weight'], brain['block.layers.0.norm.bias']
            assert w.shape == b.shape == (1024,) and (w.abs() > 1e-6).all()
            for item in subject_rows:
                if item['source_role'] != source['role']:
                    continue
                index = item['cache_row']
                assert int(cache['array_rows'][index]) == item['array_row']
                routed[item['array_row']] = (cache['condition'][index].clone(), w.clone(), b.clone())
            provenance.append({key: source[key] for key in ('role', 'condition_cache_path', 'condition_cache_sha256', 'checkpoint_path', 'checkpoint_sha256')})
            del cache, brain
        for item in subject_rows:
            ci, wi, bi = routed[item['array_row']]
            c.append(ci); weights.append(wi); biases.append(bi)
    assert sha(contract['target_cache_path']) == contract['target_cache_sha256']
    targets = torch.load(contract['target_cache_path'], weights_only=True, map_location='cpu', mmap=True)
    target = targets[[item['array_row'] for item in rows]].clone()
    del targets
    assert sha(recipe['latent_receipt']) == recipe['input_sha256'][recipe['latent_receipt']]
    latent_receipt = read(recipe['latent_receipt'])
    assert latent_receipt['status'] == 'COMPLETE_ALL13230_ALLOWED_T_NATIVE_VAE_MODE_LATENTS'
    assert latent_receipt['array_rows'] == protocol['image_rows']['T']
    assert latent_receipt['format'] == 'numpy_npy_float32_C_contiguous_v2'
    assert sha(latent_receipt['path']) == latent_receipt['sha256']
    assert latent_receipt['sha256'] == recipe['input_sha256'][latent_receipt['path']]
    latent_array = np.load(latent_receipt['path'], mmap_mode='r')
    assert latent_array.shape == (13230, 4, 64, 64) and latent_array.dtype == np.float32
    position = {row: index for index, row in enumerate(protocol['image_rows']['T'])}
    latent = torch.from_numpy(np.array(latent_array[[position[item['array_row']] for item in rows]], copy=True))
    del latent_array
    tensors = dict(condition=torch.stack(c), weight=torch.stack(weights), bias=torch.stack(biases), target=target, latent=latent)
    assert all(torch.isfinite(value).all() for value in tensors.values())
    assert all(value.dtype == torch.float32 for value in tensors.values()), 'Match actual FP32 condition/head and latent training inputs'
    assert tensors['condition'].shape == tensors['weight'].shape == tensors['bias'].shape == target.shape == (80, 1024)
    return rows, {key: value.cuda() for key, value in tensors.items()}, dict(
        split_manifest_sha256=sha(protocol_path), image_manifest_sha256=sha(manifest_path),
        input_contract_sha256=sha(contract_path), latent_receipt_sha256=sha(recipe['latent_receipt']),
        latent_sha256=latent_receipt['sha256'], target_cache_sha256=contract['target_cache_sha256'], OOF_sources=provenance)


def tensor_digest(value):
    value = value.detach().contiguous().cpu()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode()); digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def encode_tokens(adapter, projector, condition):
    result = projector(adapter(condition.to(torch.float16)))
    assert result.shape == (len(condition), 4, 2048) and torch.isfinite(result).all()
    return result


def condition_bank(rows, inputs, heads, finals, adapter, projector):
    permutation = torch.tensor([row['negative_query_index'] for row in rows], device='cuda')
    raw = inputs['condition']
    bank = dict(parent=raw, parent_permuted=raw[permutation], missing=torch.zeros_like(raw),
                correct_F=inputs['target'], wrong_F=inputs['target'][permutation])
    saturation = []
    with torch.no_grad():
        for arm, head in heads.items():
            for label, indices in (('', torch.arange(80, device='cuda')), ('_permuted', permutation)):
                head.radius = .25 * finals[arm]['selected_strength']
                changed, info = head(raw[indices], inputs['weight'][indices], inputs['bias'][indices], gate_mode='fixed')
                bank[arm + label] = changed
                if not label:
                    for radius_label, radius in (('training_radius', .25), ('deployment_radius', head.radius)):
                        head.radius = radius
                        _, ri = head(raw, inputs['weight'], inputs['bias'], gate_mode='fixed')
                        denominator = radius * ri['standardized_input_norm']
                        ratio = ri['standardized_delta_norm'] / denominator.clamp_min(1e-12)
                        for index, value in enumerate(ratio.tolist()):
                            saturation.append(dict(arm=arm, query_index=index, radius_kind=radius_label, radius=radius,
                                utilization=value if radius > 0 else None, disabled=radius == 0))
                    head.radius = .25 * finals[arm]['selected_strength']
    with torch.no_grad():
        tokens = {key: encode_tokens(adapter, projector, value) for key, value in bank.items()}
    return tokens, saturation


def native_parent_states(pipe, prompts, tokens, seeds, collector):
    """Hook each actual parent UNet call separately; preserve its return and RNG."""
    states = []
    pending = {}

    def before(module, args, kwargs):
        assert not collector.active and not collector.queries and not pending
        sample = args[0] if args else kwargs['sample']
        timestep = args[1] if len(args) > 1 else kwargs['timestep']
        assert sample.dtype == torch.float16 and tokens.dtype == torch.float16
        assert torch.equal(kwargs['encoder_hidden_states'], prompts[0].expand(len(tokens), -1, -1))
        actual_added = kwargs['added_cond_kwargs']
        assert torch.equal(actual_added['text_embeds'], prompts[1].expand(len(tokens), -1))
        assert torch.equal(actual_added['time_ids'], prompts[2].expand(len(tokens), -1))
        assert torch.equal(kwargs['cross_attention_kwargs']['ip_hidden_states'], tokens)
        pending.update(sample=sample.detach().clone(), timestep=timestep.detach().clone())
        collector.active = True

    def after(module, args, kwargs, output):
        assert collector.active and len(collector.queries) == 70
        prediction = output[0] if isinstance(output, tuple) else output.sample
        states.append(dict(**pending, queries=dict(collector.queries), prediction=prediction.detach().clone()))
        collector.active = False
        collector.queries.clear(); pending.clear()

    handles = [pipe.unet.register_forward_pre_hook(before, with_kwargs=True),
               pipe.unet.register_forward_hook(after, with_kwargs=True)]
    try:
        generators = [torch.Generator(device='cuda').manual_seed(seed) for seed in seeds]
        with torch.no_grad():
            pipe(height=512, width=512, generator=generators,
                 prompt_embeds=prompts[0].expand(len(tokens), -1, -1),
                 pooled_prompt_embeds=prompts[1].expand(len(tokens), -1),
                 cross_attention_kwargs={'ip_hidden_states': tokens},
                 num_inference_steps=4, guidance_scale=0., output_type='latent')
        assert len(states) == 4 and not pending
        assert [float(item['timestep']) for item in states] == [float(value) for value in pipe.scheduler.timesteps]
        assert all(item['sample'].shape == (len(tokens), 4, 64, 64) for item in states)
        return states
    finally:
        for handle in handles:
            handle.remove()
        collector.active = False; collector.queries.clear()


def measure_state(domain, timestep_index, sample, timestep, queries, prediction, indices,
                  token_bank, noise, collector, pipe, prompts, budget, stream, cells, network):
    from brainhive_response_rank_v1 import ip_response
    from brainhive_component_generator_v1 import denoise
    assert len(queries) == len(collector.layers) == 70 and not collector.active
    versions = {name: value._version for name, value in queries.items()}
    with torch.no_grad():
        for name, (attention, processor) in collector.layers.items():
            q = queries[name]
            positive = ip_response(q, token_bank['correct_F'][indices], processor, attention.heads).float()
            negative = ip_response(q, token_bank['wrong_F'][indices], processor, attention.heads).float()
            dimensions = (1, 2, 3)
            separation = (positive - negative).square().mean(dimensions)
            valid = separation > EPSILON
            for condition, tokens in token_bank.items():
                response = ip_response(q, tokens[indices], processor, attention.heads).float()
                dp = (response - positive).square().mean(dimensions)
                dn = (response - negative).square().mean(dimensions)
                error = dp / (separation + EPSILON)
                margin = (dn - dp) / (separation + EPSILON)
                for offset, query_index in enumerate(indices.tolist()):
                    row = dict(domain=domain, timestep_index=timestep_index, timestep=float(timestep),
                        query_index=query_index, layer=name, condition=condition,
                        teacher_separation=float(separation[offset]), valid=bool(valid[offset]),
                        positive_error=float(error[offset]) if valid[offset] else None,
                        normalized_margin=float(margin[offset]) if valid[offset] else None,
                        raw_positive_MSE=float(dp[offset]), raw_negative_MSE=float(dn[offset]))
                    budget.append(stream, row)
                    cell = cells.setdefault((domain, query_index, timestep_index, condition), dict(layers=0, valid=0, error=0., margin=0.))
                    cell['layers'] += 1
                    if row['valid']:
                        cell['valid'] += 1
                        cell['error'] += row['positive_error']; cell['margin'] += row['normalized_margin']
            assert queries[name]._version == versions[name]
        predictions = {'parent': prediction}
        for condition in ('PN', 'correct_F', 'wrong_F', 'missing', 'parent_permuted'):
            predictions[condition] = denoise(pipe, prompts, sample, timestep, token_bank[condition][indices])
        for offset, query_index in enumerate(indices.tolist()):
            for condition, value in predictions.items():
                record = dict(domain=domain, query_index=query_index, timestep_index=timestep_index,
                    condition=condition, known_training_noise=noise is not None,
                    epsilon_MSE=float((value[offset].float() - noise[offset].float()).square().mean()) if noise is not None else None,
                    prediction_to_correct_F_MSE=float((value[offset].float() - predictions['correct_F'][offset].float()).square().mean()),
                    prediction_to_wrong_F_MSE=float((value[offset].float() - predictions['wrong_F'][offset].float()).square().mean()))
                network.append(record)
    assert all(queries[name]._version == version for name, version in versions.items())
    budget.check()


def flatten_gradients(loss, parameters, retain_graph):
    gradients = torch.autograd.grad(loss, parameters, retain_graph=retain_graph, allow_unused=True)
    return torch.cat([(torch.zeros_like(parameter) if gradient is None else gradient).reshape(-1).detach().float()
                      for parameter, gradient in zip(parameters, gradients)])


def separate_gradient_vectors(losses, parameters):
    # All losses share the head/token graph. Release it on the final epsilon
    # derivative, then let the caller release the auxiliary branch references
    # before constructing the next arm's UNet graph.
    order = ('token_weighted', 'positive_weighted', 'ranking_weighted', 'epsilon')
    assert set(losses) == set(order)
    return {key: flatten_gradients(losses[key], parameters, retain_graph=index < len(order) - 1)
            for index, key in enumerate(order)}


def training_head_forward(head, condition, weight, bias):
    """The real trainer runs its FP32 head BEFORE generator BF16 autocast."""
    assert all(value.dtype == torch.float32 for value in (condition, weight, bias))
    assert all(parameter.dtype == torch.float32 for parameter in head.parameters())
    with torch.autocast(condition.device.type, enabled=False):
        changed, information = head(condition, weight, bias, gate_mode='fixed')
    assert changed.dtype == torch.float32
    return changed, information


def compound_gradient_diagnostics(gradients, arm):
    """Compare the ACTUAL weighted base and active auxiliary, no new forward."""
    assert arm in ARMS
    base = gradients['epsilon'] + gradients['token_weighted']
    auxiliary = torch.zeros_like(base)
    if arm in ('P', 'PN', 'S'):
        auxiliary = auxiliary + gradients['positive_weighted']
    if arm in ('PN', 'S'):
        auxiliary = auxiliary + gradients['ranking_weighted']
    assert torch.isfinite(base).all() and torch.isfinite(auxiliary).all()
    base_norm, auxiliary_norm = float(base.norm()), float(auxiliary.norm())
    assert math.isfinite(base_norm) and math.isfinite(auxiliary_norm)
    dot = float(torch.dot(base, auxiliary))
    total_dot = float(torch.dot(base, base + auxiliary))
    assert math.isfinite(dot) and math.isfinite(total_dot)
    denominator = base_norm * auxiliary_norm
    return dict(base_norm=base_norm, active_auxiliary_norm=auxiliary_norm,
        base_dot_active_auxiliary=dot,
        base_vs_active_auxiliary_cosine=dot / denominator if denominator > 1e-20 else None,
        active_auxiliary_to_base_norm_ratio=auxiliary_norm / base_norm if base_norm > 1e-20 else None,
        base_dot_base_plus_auxiliary=total_dot,
        summed_gradient_reverses_base_descent=total_dot < 0,
        base_definition='epsilon + token_weighted',
        auxiliary_definition='zero' if arm == 'O' else ('positive_weighted' if arm == 'P' else 'positive_weighted + ranking_weighted'))


def gradient_probe(pipe, prompts, adapter, projector, heads, inputs, rows, collector, ddpm, timesteps, budget):
    """Fixed first selected stimulus from EACH OOF role per subject, four times.

    Student-independent parent Q is captured once per batch/time. Each loss is
    differentiated separately at the original training radius. No optimizer,
    backwards into host parameters, parameter update or inference-time teacher.
    S's registered shuffled training pair is reconstructed on all allowed T
    metadata; this does not select negative examples using an observed score.
    """
    from brainhive_component_generator_v1 import denoise
    from brainhive_response_rank_v1 import ip_response
    from brainhive_functional_alignment_v2 import functional_alignment_loss
    from brainhive_functional_training_contract_v2 import teacher_mappings
    # Load only training targets for the registered shuffled-pair mechanism.
    recipe_contract = read(Path('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json'))
    protocol = read(ROOT / 'configs/split_manifest.json')
    concept_by_row = {}
    with (ROOT / 'audit/image_manifest.csv').open(encoding='utf-8', newline='') as source:
        for item in csv.DictReader(source):
            if item['dataset_split'] == 'train':
                concept_by_row[int(item['array_row'])] = item['concept_id']
    mapping = teacher_mappings(protocol['image_rows']['T'], concept_by_row)
    positions = {row: i for i, row in enumerate(protocol['image_rows']['T'])}
    targets = torch.load(recipe_contract['target_cache_path'], weights_only=True, map_location='cpu', mmap=True)
    shuffled_positive, shuffled_negative, training_negative = [], [], []
    for row in rows:
        pos = positions[row['array_row']]
        shuffled_positive.append(targets[protocol['image_rows']['T'][mapping['rank_positive_shuffled'][pos]]].clone())
        shuffled_negative.append(targets[protocol['image_rows']['T'][mapping['rank_negative_shuffled'][pos]]].clone())
        training_negative.append(targets[protocol['image_rows']['T'][mapping['negative'][pos]]].clone())
    del targets
    shuffled_positive = torch.stack(shuffled_positive).cuda()
    shuffled_negative = torch.stack(shuffled_negative).cuda()
    training_negative = torch.stack(training_negative).cuda()
    records = []
    # The inference probes use native FP16. Reproduce load_native(training=True)
    # prompt conditioning for the separate training-objective gradient probe.
    training_prompts = tuple(value.detach().clone().to(torch.bfloat16) for value in prompts)
    pipe.unet.enable_gradient_checkpointing()
    assert pipe.unet.config.dropout == 0
    pipe.unet.train()
    try:
        for subject_index in range(10):
            indices = torch.tensor([subject_index * 8, subject_index * 8 + 4], device='cuda')
            for timestep_index, timestep in enumerate(timesteps):
                budget.check()
                generator = torch.Generator(device='cuda').manual_seed(stable_seed('gradient_noise', subject_index, timestep_index))
                noise = torch.randn(inputs['latent'][indices].shape, generator=generator, device='cuda')
                time_batch = timestep.long().expand(2)
                noisy = ddpm.add_noise(inputs['latent'][indices], noise, time_batch)
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    base_tokens = projector(adapter(inputs['condition'][indices]))
                    positive_tokens = projector(adapter(inputs['target'][indices]))
                    negative_tokens = projector(adapter(training_negative[indices]))
                    rank_positive_tokens = projector(adapter(shuffled_positive[indices]))
                    rank_negative_tokens = projector(adapter(shuffled_negative[indices]))
                    with collector.capture():
                        denoise(pipe, training_prompts, noisy, time_batch, base_tokens)
                queries = dict(collector.queries); collector.queries.clear()
                for arm in ARMS:
                    head = heads[arm]
                    head.radius = .25
                    head.requires_grad_(True)
                    parameters = list(head.parameters())
                    changed, information = training_head_forward(head, inputs['condition'][indices], inputs['weight'][indices], inputs['bias'][indices])
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        student_tokens = projector(adapter(changed))
                        prediction = denoise(pipe, training_prompts, noisy, time_batch, student_tokens)
                        epsilon_loss = (prediction.float() - noise).square().mean()
                        token_loss = (1 - torch.nn.functional.cosine_similarity(student_tokens.float(), positive_tokens.float(), dim=-1)).mean() * .1
                        positive_terms, rank_terms = [], []
                        for name, (attention, processor) in collector.layers.items():
                            q = queries[name]
                            response = ip_response(q, student_tokens, processor, attention.heads)
                            with torch.no_grad():
                                positive = ip_response(q, positive_tokens, processor, attention.heads)
                                negative = ip_response(q, negative_tokens, processor, attention.heads)
                                rank_positive = ip_response(q, rank_positive_tokens, processor, attention.heads) if arm == 'S' else None
                                rank_negative = ip_response(q, rank_negative_tokens, processor, attention.heads) if arm == 'S' else None
                            ploss, _ = functional_alignment_loss(response, positive, negative, mode='positive_only')
                            full, _ = functional_alignment_loss(response, positive, negative,
                                rank_positive=rank_positive, rank_negative=rank_negative)
                            positive_terms.append(ploss); rank_terms.append(full - ploss)
                        losses = dict(epsilon=epsilon_loss, token_weighted=token_loss,
                            positive_weighted=.1 * torch.stack(positive_terms).mean(),
                            ranking_weighted=.1 * torch.stack(rank_terms).mean())
                        gradients = separate_gradient_vectors(losses, parameters)
                    assert all(torch.isfinite(value).all() for value in gradients.values())
                    norms = {key: float(value.norm()) for key, value in gradients.items()}
                    assert all(math.isfinite(value) for value in norms.values())
                    compound = compound_gradient_diagnostics(gradients, arm)
                    cosine = {}
                    for left, right in (('epsilon', 'token_weighted'), ('epsilon', 'positive_weighted'), ('epsilon', 'ranking_weighted'),
                                        ('positive_weighted', 'ranking_weighted')):
                        denominator = norms[left] * norms[right]
                        cosine[left + '__' + right] = float(torch.dot(gradients[left], gradients[right]) / denominator) if denominator > 1e-20 else None
                    records.append(dict(arm=arm, subject=rows[int(indices[0])]['subject'], query_indices=indices.tolist(),
                        timestep_index=timestep_index, timestep=float(timestep), weighted_gradient_norms=norms, cosines=cosine,
                        compound_gradient=compound,
                        precision=dict(head='FP32 outside autocast', A_P_IP_UNet='BF16 autocast over unchanged FP16 host weights', prompts='BF16 detached copy'),
                        positive_active_in_training=arm in ('P', 'PN', 'S'), ranking_active_in_training=arm in ('PN', 'S'),
                        radius=.25, negative_policy='original_all_T_registered_teacher_mappings_negative',
                        ranking_pair_policy='registered_shuffled_pair' if arm == 'S' else 'correct_pair_probe'))
                    assert all(parameter.grad is None for parameter in parameters)
                    head.requires_grad_(False)
                    del losses, gradients, prediction, student_tokens, positive_terms, rank_terms
                    del epsilon_loss, token_loss, ploss, full, changed, information, response, parameters
                    budget.check()
                del queries
    finally:
        pipe.unet.eval()
        for head in heads.values():
            head.requires_grad_(False)
        collector.queries.clear()
    return records


def summarize(rows, cells, network, gradients, saturation):
    conditions = ('parent', 'O', 'P', 'PN', 'S')
    expected_conditions = set(conditions) | {key + '_permuted' for key in conditions} | {'correct_F', 'wrong_F', 'missing'}
    expected_cells = {(domain, query, ti, condition) for domain in DOMAINS for query in range(80)
                      for ti in range(4) for condition in expected_conditions}
    assert set(cells) == expected_cells and all(item['layers'] == 70 for item in cells.values())
    assert all(type(item['valid']) is int and 0 <= item['valid'] <= 70 and
               math.isfinite(item['error']) and item['error'] >= 0 and math.isfinite(item['margin']) for item in cells.values())
    expected_network = {(domain, query, ti, condition) for domain in DOMAINS for query in range(80) for ti in range(4)
                        for condition in ('parent', 'PN', 'correct_F', 'wrong_F', 'missing', 'parent_permuted')}
    assert len(network) == len(expected_network) and {
        (item['domain'], item['query_index'], item['timestep_index'], item['condition']) for item in network} == expected_network
    for item in network:
        assert item['known_training_noise'] == (item['domain'] == DOMAINS[0])
        if item['known_training_noise']:
            assert item['epsilon_MSE'] is not None and math.isfinite(item['epsilon_MSE']) and item['epsilon_MSE'] >= 0
        else:
            assert item['epsilon_MSE'] is None, 'No ground-truth epsilon exists on a native sampling trajectory'
        assert all(math.isfinite(item[key]) and item[key] >= 0 for key in ('prediction_to_correct_F_MSE', 'prediction_to_wrong_F_MSE'))
    expected_gradients = {(arm, 'sub-%02d' % subject, ti) for arm in ARMS for subject in range(1, 11) for ti in range(4)}
    assert len(gradients) == len(expected_gradients) and {(item['arm'], item['subject'], item['timestep_index']) for item in gradients} == expected_gradients
    expected_saturation = {(arm, query, kind) for arm in ARMS for query in range(80) for kind in ('training_radius', 'deployment_radius')}
    assert len(saturation) == len(expected_saturation) and {(item['arm'], item['query_index'], item['radius_kind']) for item in saturation} == expected_saturation
    for item in saturation:
        assert item['disabled'] == (item['radius'] == 0)
        assert item['utilization'] is None if item['disabled'] else (math.isfinite(item['utilization']) and item['utilization'] >= 0)
    per_query = {}
    for domain in DOMAINS:
        for qi in range(80):
            for condition in expected_conditions:
                items = [cells[(domain, qi, ti, condition)] for ti in range(4)]
                valid = sum(item['valid'] for item in items)
                # Equal times first; invalid-only time remains None, never removed.
                errors = [item['error'] / item['valid'] if item['valid'] else None for item in items]
                margins = [item['margin'] / item['valid'] if item['valid'] else None for item in items]
                per_query[(domain, qi, condition)] = dict(valid_fraction=valid / 280,
                    error=mean(errors) if None not in errors else None,
                    margin=mean(margins) if None not in margins else None)
    aggregate = {}
    minimum_valid = min(value['valid_fraction'] for value in per_query.values())
    for domain in DOMAINS:
        aggregate[domain] = {}
        for condition in expected_conditions:
            values = [per_query[(domain, qi, condition)] for qi in range(80)]
            aggregate[domain][condition] = {key: mean(item[key] for item in values) if all(item[key] is not None for item in values) else None
                                            for key in ('error', 'margin', 'valid_fraction')}

    def difference(domain, first, second, metric):
        first, second = aggregate[domain][first][metric], aggregate[domain][second][metric]
        return None if first is None or second is None else first - second

    def evidence_compare(first, second):
        return {domain: dict(margin_gain=difference(domain, first, second, 'margin'),
                            positive_error_reduction=difference(domain, second, first, 'error')) for domain in DOMAINS}

    def paired_signal_compare(first, second):
        evidence = {}
        for domain in DOMAINS:
            first_margin = difference(domain, first, first + '_permuted', 'margin')
            second_margin = difference(domain, second, second + '_permuted', 'margin')
            first_error = difference(domain, first + '_permuted', first, 'error')
            second_error = difference(domain, second + '_permuted', second, 'error')
            evidence[domain] = dict(paired_margin_gap_gain=None if first_margin is None or second_margin is None else first_margin - second_margin,
                paired_positive_error_gap_gain=None if first_error is None or second_error is None else first_error - second_error)
        return evidence

    def compare_decision(evidence, explanation):
        values = [value for item in evidence.values() for value in item.values()]
        return decision(classify_directions(values) if minimum_valid >= .9 else 'UNKNOWN', evidence, explanation)

    decisions = {}
    ddpm = [record for record in network if record['domain'] == DOMAINS[0]]
    noise_means = {condition: mean(row['epsilon_MSE'] for row in ddpm if row['condition'] == condition)
                   for condition in ('parent', 'PN', 'correct_F', 'wrong_F', 'missing', 'parent_permuted')}
    teacher_gains = dict(correct_vs_wrong=noise_means['wrong_F'] - noise_means['correct_F'],
                         correct_vs_missing=noise_means['missing'] - noise_means['correct_F'])
    decisions['teacher_discrimination'] = decision(classify_directions(list(teacher_gains.values())), teacher_gains,
        'Known-noise DDPM teacher usefulness only; self-distance and teacher J identities are excluded.')
    paired_signal = {arm: paired_signal_compare(arm, 'O') for arm in ('P', 'PN')}
    decisions['paired_EEG_signal'] = compare_decision(paired_signal['PN'],
        'PN must increase correct-vs-permuted stimulus-specific response gaps relative to O in both domains; absolute teacher proximity alone is insufficient.')
    for key, left, right in (('PN_over_O', 'PN', 'O'), ('PN_over_P', 'PN', 'P'), ('shuffled_response_control', 'PN', 'S')):
        decisions[key] = compare_decision(evidence_compare(left, right), 'Frozen postfit T response comparison; no held-out or image-gain claim.')
    transfer = {comparison: evidence_compare('PN', comparison) for comparison in ('O', 'P', 'S')}
    transfer_values = {domain: [value for comparison in transfer.values() for value in comparison[domain].values()] +
                       list(paired_signal['PN'][domain].values()) for domain in DOMAINS}
    ddpm_sign = classify_directions(transfer_values[DOMAINS[0]])
    trajectory_sign = classify_directions(transfer_values[DOMAINS[1]])
    transfer_result = ('SUPPORTED' if ddpm_sign == trajectory_sign == 'SUPPORTED' else
                       'NOT_SUPPORTED' if ddpm_sign == 'SUPPORTED' and
                           all(value is not None and value < 0 for value in transfer_values[DOMAINS[1]]) else 'UNKNOWN')
    if minimum_valid < .9:
        transfer_result = 'UNKNOWN'
    decisions['trajectory_transfer'] = decision(transfer_result,
        dict(absolute_controls=transfer, paired_signal_over_O=paired_signal['PN']),
        'All PN-vs-O/P/S absolute DDPM margin/error directions AND PN-vs-O correct-minus-permuted signal increments must support gain first. SUPPORTED needs both corresponding rollout criteria; NOT_SUPPORTED requires every rollout direction strictly reversed. All mixed/common-shift/unsupported-DDPM evidence is UNKNOWN.')
    relevant = [row for row in gradients if row['arm'] in ('P', 'PN', 'S')]
    cosines = [row['compound_gradient']['base_vs_active_auxiliary_cosine'] for row in relevant]
    assert len(cosines) == 120
    finite = [value for value in cosines if value is not None and math.isfinite(value)]
    negative_fraction = mean(value < 0 for value in finite)
    conflict_result = 'UNKNOWN' if len(finite) != len(cosines) else ('SUPPORTED' if negative_fraction > .5 else 'NOT_SUPPORTED')
    decisions['gradient_conflict'] = decision(conflict_result,
        dict(actual_active_auxiliary_vs_weighted_base_negative_fraction=negative_fraction, finite_pairs=len(finite), expected_pairs=120,
             fixed_aggregation='Equal P/PN/S x 10 subjects x four times, 120 composite cosines; base=epsilon+.1token, auxiliary=.1positive[+.1ranking]; fraction(cosine<0)>.5.',
             summed_gradient_reverses_base_descent_fraction=mean(row['compound_gradient']['summed_gradient_reverses_base_descent'] for row in relevant),
             mean_active_auxiliary_to_base_norm_ratio=mean(row['compound_gradient']['active_auxiliary_to_base_norm_ratio'] for row in relevant)
                 if all(row['compound_gradient']['active_auxiliary_to_base_norm_ratio'] is not None for row in relevant) else None),
        'A majority negative actual composite cosine indicates local objective competition; individual branches, norm dominance, and summed-gradient reversal remain descriptive. It does not establish image benefit or useful gradient surgery.')
    values = [row['utilization'] for row in saturation if row['arm'] == 'PN' and row['radius_kind'] == 'training_radius']
    decisions['radius_saturation'] = decision('SUPPORTED' if mean(value >= .99 for value in values) >= .9 else 'NOT_SUPPORTED',
        dict(PN_training_radius_fraction_at_least_99_percent=mean(value >= .99 for value in values), mean_utilization=mean(values)),
        'Saturation is not evidence that enlarging the radius helps; deployment radius zero is separately marked disabled.')
    per_query_rows = [dict(domain=key[0], query_index=key[1], subject=rows[key[1]]['subject'], condition=key[2], **value)
                      for key, value in per_query.items()]
    return dict(status=COMPLETE, diagnostic_decisions=decisions, aggregate=aggregate,
        DDPM_epsilon_MSE=noise_means, paired_signal_gain_over_O=paired_signal,
        minimum_per_query_valid_layer_time_fraction=minimum_valid,
        decision_validity_threshold=.9, decision_thresholds_are_directional_not_significance=True,
        coverage=dict(expected_queries=80, expected_timesteps_per_domain=4, expected_domains=list(DOMAINS),
            observed_query_timestep_domain_cells=640, expected_native_IP_layers=70, missing_cells=0,
            unique_T_stimuli=8, unique_subjects=10, observations_are_not_80_independent_stimuli=True),
        per_query=per_query_rows, no_optimizer_created=True, parameter_updates=0,
        protected_scores_read=False, generated_images=0, candidate_auto_started=False,
        scope='Fixed postfit training-set probes, not independent confirmation, SOTA, image quality, or causal proof.',
        trajectory_noise_error_available=False, trajectory_x0_error_omitted=True,
        teacher_permissions='Allowed T-only F targets in probes; no target enters parent trajectory state updates.')


def execute(document, registration_path):
    global torch
    import torch
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native, denoise
    from brainhive_response_rank_v1 import NativeIPResponseCapture
    from diffusers import DDPMScheduler
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    budget = Budget(document)
    budget.check()
    finals = inspect_finals(document, load_tensors=True)
    wrapper = verify_wrapper(document)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    output = Path(document['output_directory'])
    output.mkdir(parents=True, exist_ok=False)
    rows = cells = collector = None
    completed_stages = []
    before = {arm: {key: item[key] for key in ('checkpoint_path', 'checkpoint_sha256', 'completion_sha256', 'calibration_sha256', 'provenance_sha256')}
              for arm, item in finals.items()}
    try:
        rows, inputs, input_provenance = selected_inputs(finals['O']['recipe'], budget)
        write(output / 'queries.json', dict(selection='seed1729_hash_ordered_4_concepts_per_OOF_role_same8_T_stimuli_all10subjects', rows=rows))
        write(output / 'provenance.json', dict(registration_path=str(registration_path), registration_sha256=sha(registration_path),
            source_sha256=document['source_sha256'], inputs=input_provenance, wrapper=wrapper, finals_before=before,
            conditions_share_parent_Q=True, Q_positions_per_layer=16, native_IP_layers=70, batch_size=2,
            permutation_carries_own_affine=True, normal_inference_reads_teacher=False,
            layer_time_aggregation='Per query: equal 70 layers within each time (mask reported), then equal four times, then equal subjects/stimuli; never choose a layer/time.',
            DDPM_definition='Native scheduler four times; DDPM forward-noised real T latent with known epsilon; input matches unscaled DDPM training state.',
            trajectory_definition='Actual unchanged parent pipeline four-step rollout, FP16, captured UNet inputs AFTER scale_model_input; no real latent/image/teacher used in rollout updates.',
            functional_precision='Native FP16 for both state domains; gradient probe uses training BF16 autocast.',
            gradients='Frozen final heads at original training radius .25, FP32 head outside autocast; unchanged FP16 A/P/IP/UNet weights computed under BF16 autocast with BF16 prompt copies; original full-T teacher mappings. Actual base/active auxiliary composite vectors. No optimizer/updates.'))
        heads = {}
        for arm in ARMS:
            head = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
            head.load_state_dict(finals[arm].pop('state'), strict=True)
            assert sum(parameter.numel() for parameter in head.parameters()) == 525568
            heads[arm] = head.cuda()
        head_before = {arm: module_digest(head) for arm, head in heads.items()}
        pipe, adapter, projector, prompts, host = load_native(training=False)
        assert type(pipe.scheduler).__name__ == 'EulerAncestralDiscreteScheduler'
        assert pipe.scheduler.config.prediction_type == 'epsilon' and pipe.scheduler.config.timestep_spacing == 'trailing'
        assert all(not parameter.requires_grad for module in (pipe.unet, adapter, projector) for parameter in module.parameters())
        ddpm = DDPMScheduler.from_config(pipe.scheduler.config)
        assert ddpm.config.prediction_type == 'epsilon' and ddpm.config.num_train_timesteps == 1000
        pipe.scheduler.set_timesteps(4, device='cuda')
        timesteps = pipe.scheduler.timesteps.clone()
        assert len(timesteps) == 4 and torch.equal(timesteps, timesteps.long().to(timesteps.dtype))
        collector = NativeIPResponseCapture(pipe.unet)
        token_bank, saturation = condition_bank(rows, inputs, heads, finals, adapter, projector)
        write(output / 'radius_saturation.json', saturation)
        cells, network = {}, []
        with gzip.open(output / 'functional_query_layer_measurements.jsonl.gz', 'xb', compresslevel=3) as stream:
            for start in range(0, 80, 2):
                budget.check()
                indices = torch.arange(start, start + 2, device='cuda')
                with torch.no_grad():
                    trajectory = native_parent_states(pipe, prompts, token_bank['parent'][indices],
                        [rows[index]['trajectory_seed'] for index in range(start, start + 2)], collector)
                    for ti, timestep in enumerate(timesteps):
                        generator = torch.Generator(device='cuda').manual_seed(stable_seed('DDPM_noise', start, ti))
                        noise = torch.randn(inputs['latent'][indices].shape, generator=generator, device='cuda')
                        noisy = ddpm.add_noise(inputs['latent'][indices], noise, timestep.long().expand(2))
                        with collector.capture():
                            prediction = denoise(pipe, prompts, noisy, timestep, token_bank['parent'][indices])
                        queries = dict(collector.queries); collector.queries.clear()
                        measure_state(DOMAINS[0], ti, noisy, timestep, queries, prediction, indices,
                            token_bank, noise, collector, pipe, prompts, budget, stream, cells, network)
                        state = trajectory[ti]
                        measure_state(DOMAINS[1], ti, state['sample'], state['timestep'], state['queries'], state['prediction'], indices,
                            token_bank, None, collector, pipe, prompts, budget, stream, cells, network)
                    del trajectory, queries
                print(json.dumps(dict(stage='FROZEN_T_FUNCTIONAL_STATES', queries_completed=start + 2, expected_queries=80)), flush=True)
        completed_stages.append('both_state_domains_80x4x70_common_parent_Q')
        write(output / 'network_controls.json', network)
        gradients = gradient_probe(pipe, prompts, adapter, projector, heads, inputs, rows, collector, ddpm, timesteps, budget)
        assert len(gradients) == 160
        write(output / 'gradient_conflicts.json', gradients)
        completed_stages.append('fixed20_query_four_time_four_arm_gradient_probe')
        assert {arm: module_digest(head) for arm, head in heads.items()} == head_before
        assert all(parameter.grad is None for module in (pipe.unet, adapter, projector, *heads.values()) for parameter in module.parameters())
        after = inspect_finals(document, load_tensors=False)
        after_hashes = {arm: {key: item[key] for key in before[arm]} for arm, item in after.items()}
        assert before == after_hashes
        summary = summarize(rows, cells, network, gradients, saturation)
        budget.check()
        summary.update(native_scheduler_timesteps=timesteps.tolist(), completed_stages=completed_stages,
            resource_observations=dict(elapsed_seconds=time.monotonic() - budget.started,
                peak_GPU_allocated_bytes=budget.peak_GPU_bytes, peak_extra_CPU_RSS_bytes=budget.peak_extra_CPU_bytes),
            host=host, final_checkpoint_SHA_before=before, final_checkpoint_SHA_after=after_hashes,
            heads_in_memory_unchanged=True)
        write(output / 'function_summary.json', summary)
        write(output / 'completion.json', dict(status=COMPLETE, summary_sha256=sha(output / 'function_summary.json'),
            provenance_sha256=sha(output / 'provenance.json'), final_checkpoint_SHA_before=before,
            final_checkpoint_SHA_after=after_hashes, parameter_updates=0, optimizer_created=False,
            protected_scores_read=False, generated_images=0, missing_stages=[]))
        print(COMPLETE, flush=True)
    except BaseException as error:
        failure = dict(status='FAILED_OR_INCOMPLETE_T_ONLY_DIAGNOSTIC', error_type=type(error).__name__,
            error=str(error), completed_stages=completed_stages, expected_queries=80,
            observed_condition_cells=len(cells) if isinstance(cells, dict) else 0,
            expected_condition_cells=80 * 4 * 2 * 13, parameter_updates=0,
            optimizer_created=False, protected_scores_read=False, no_success_denominator_reduction=True)
        write(output / 'failure.json', failure)
        raise
    finally:
        if collector is not None:
            collector.close()


def contract_check():
    assert classify_directions([1., 2.]) == 'SUPPORTED'
    assert classify_directions([0., -1.]) == 'NOT_SUPPORTED'
    assert classify_directions([1., -1.]) == 'UNKNOWN'
    assert classify_directions([None]) == 'UNKNOWN'
    assert stable_seed('a') == stable_seed('a') != stable_seed('b')
    return dict(status='PURE_CPU_SYNTHETIC_CONTRACT_CHECK_PASSED', checks=5,
                GPU_allocated=False, real_T_inputs_read=False, scores_read=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--check-contract', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.check_contract:
        print(json.dumps(contract_check())); return
    assert args.registration is not None
    document = validate_registration(read(args.registration))
    if args.preflight_only:
        finals = inspect_finals(document)
        print(json.dumps(dict(status='CPU_PREFLIGHT_ALL_FOUR_FINALS_VERIFIED_NO_GPU',
            finals={arm: value['checkpoint_sha256'] for arm, value in finals.items()}))); return
    execute(document, args.registration)


if __name__ == '__main__':
    main()
