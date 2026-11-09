"""DRAFT: bounded-risk direction/radius diagnostic on the existing oracle T32.

No fit, UNet, generator, raw D/V/test condition, or image score is accessed.
The risk head and its already-fitted slope/intercept are deployment metadata;
their use here does not establish physiological reliability or image benefit.
Run only after review, with an immutable, hash-complete --config registration.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import signal
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
NAME = 'T32_radius_CPU_v1'
MINIMUM_HEADROOM = 3 << 30
SECONDS = 180
REQUIRED_SOURCES = (
    'diagnose_BrainHIVE_T32_radius_CPU_v1.py',
    'brainhive_native_models_v1.py', 'brainhive_bounded_component_v1.py',
    'reproduction_sources/BrainHIVE_f06db69f/main/models_adapter.py',
    'reproduction_sources/BrainHIVE_f06db69f/main/layers.py',
    'reproduction_sources/BrainHIVE_f06db69f/_fetch_receipt.json',
)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def memory_guard(minimum=MINIMUM_HEADROOM):
    """Fresh effective headroom = min(host available, finite cgroup ancestors)."""
    assert os.name == 'posix' and Path('/proc/meminfo').is_file()
    values = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        fields = line.split()
        if fields[0] == 'MemAvailable:':
            values['host_MemAvailable_bytes'] = int(fields[1]) * 1024
    assert values.get('host_MemAvailable_bytes', 0) > 0
    root = Path('/sys/fs/cgroup').resolve()
    candidates = [root]
    for line in Path('/proc/self/cgroup').read_text().splitlines():
        fields = line.split(':', 2)
        if len(fields) == 3 and fields[0] == '0':
            relative = Path(fields[2].lstrip('/'))
            candidate = (root / relative).resolve()
            if candidate.is_relative_to(root) and candidate.exists():
                candidates.append(candidate)
                candidates.extend(p for p in candidate.parents if p.is_relative_to(root))
    bounds = []
    seen = set()
    observed_controller = False
    for folder in candidates:
        if folder in seen:
            continue
        seen.add(folder)
        usage_path = folder / 'memory.current'
        limit_path = folder / 'memory.max'
        if usage_path.is_file() and limit_path.is_file():
            observed_controller = True
            usage = int(usage_path.read_text().strip())
            for name in ('memory.max', 'memory.high'):
                path = folder / name
                if not path.exists():
                    continue
                limit = path.read_text().strip()
                if limit != 'max':
                    bounds.append(dict(path=str(path), limit=int(limit), usage=usage,
                                       headroom=max(0, int(limit) - usage)))
    if not observed_controller:
        limit_path = root / 'memory/memory.limit_in_bytes'
        usage_path = root / 'memory/memory.usage_in_bytes'
        assert limit_path.exists() and usage_path.exists(), 'No readable cgroup memory controller'
        limit = int(limit_path.read_text())
        usage = int(usage_path.read_text())
        bounds.append(dict(path=str(limit_path), limit=limit, usage=usage,
                           headroom=max(0, limit - usage)))
    headroom = min([values['host_MemAvailable_bytes']] + [v['headroom'] for v in bounds])
    assert headroom >= minimum, f'Fresh CPU headroom guard failed: {headroom} < {minimum}'
    return dict(**values, finite_cgroup_bounds=bounds,
                effective_headroom_bytes=headroom, minimum_headroom_bytes=minimum,
                monotonic_seconds=time.monotonic())


def fixed_negative_positions(selected):
    """A fixed bijection within the SAME 32 allowed images, different concepts."""
    groups = {}
    for position, query in enumerate(selected):
        groups.setdefault(query['concept_id'], []).append(position)
    assert len(selected) == 32 and len(groups) > 1
    rng = random.Random(1729)
    names = sorted(groups)
    rng.shuffle(names)
    ordered = []
    for name in names:
        positions = groups[name][:]
        rng.shuffle(positions)
        ordered.extend(positions)
    shift = max(map(len, groups.values()))
    assert shift <= 16, 'Different-concept T32 bijection is not feasible'
    negative = [-1] * 32
    for i, position in enumerate(ordered):
        negative[position] = ordered[(i + shift) % 32]
    assert sorted(negative) == list(range(32))
    assert all(selected[i]['concept_id'] != selected[j]['concept_id'] for i, j in enumerate(negative))
    return negative


def fixed_selection(oracle, protocol, manifest):
    assert isinstance(oracle, list) and len(oracle) == 32
    allowed = set(protocol['image_rows']['T'])
    assert allowed.isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    assert len({q['array_row'] for q in oracle}) == 32
    result = []
    for index, query in enumerate(oracle):
        assert set(query) == {'stimulus_id', 'array_row'}
        row = query['array_row']
        assert isinstance(row, int) and row in allowed
        target = manifest[row]
        assert target['dataset_split'] == 'train' and target['stimulus_id'] == query['stimulus_id']
        result.append(dict(query, query_index=index, concept_id=target['concept_id']))
    return result


def validate_before_torch(reg):
    assert reg['diagnostic'] == NAME and reg['subject'] == 'sub-01'
    assert reg['queries'] == 32 and reg['maximum_seconds'] == SECONDS
    assert reg['minimum_cgroup_headroom_bytes'] == MINIMUM_HEADROOM
    assert reg['output_path'] == str(WORK / 'audit/T32_radius_CPU_v1.json')
    assert not Path(reg['output_path']).exists(), 'Preserve the original diagnostic receipt'
    for name in REQUIRED_SOURCES:
        assert reg['source_sha256'][name] == sha(ROOT / name), name
    paths = [reg[k] for k in ('oracle_inference_inputs', 'input_contract', 'risk_completion',
                              'risk_checkpoint', 'main_recipe', 'asset_verification')]
    paths += [str(ROOT / 'configs/split_manifest.json'), str(ROOT / 'audit/image_manifest.csv')]
    assert set(paths) <= set(reg['input_sha256'])
    for path, expected in reg['input_sha256'].items():
        assert Path(path).is_absolute() and sha(path) == expected, path
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    assert sha(ROOT / 'audit/image_manifest.csv') == protocol['image_manifest_sha256']
    with (ROOT / 'audit/image_manifest.csv').open() as stream:
        manifest = {int(r['array_row']): r for r in csv.DictReader(stream) if r['dataset_split'] == 'train'}
    selected = fixed_selection(json.loads(Path(reg['oracle_inference_inputs']).read_text()), protocol, manifest)
    contract = json.loads(Path(reg['input_contract']).read_text())
    assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    assert contract['split_manifest_sha256'] == sha(ROOT / 'configs/split_manifest.json')
    subject = next(s for s in contract['subjects'] if s['subject'] == 'sub-01')
    sources = [s for s in subject['sources'] if s['role'] in ('oof0', 'oof1')]
    assert len(sources) == 2
    bindings = {}
    verified = {}
    allowed = set(protocol['image_rows']['T'])
    for source in sources:
        assert set(source['groups']) == {'component_train'}
        for field, hash_field in [('condition_cache_path', 'condition_cache_sha256'),
                                 ('checkpoint_path', 'checkpoint_sha256'),
                                 ('parent_registration_path', 'parent_registration_sha256')]:
            path = source[field]
            assert sha(path) == source[hash_field], path
            verified[path] = source[hash_field]
        registration = json.loads(Path(source['parent_registration_path']).read_text())
        fit_rows = registration['training_array_rows']
        assert set(fit_rows) <= allowed
        fit_concepts = {manifest[i]['concept_id'] for i in fit_rows}
        for group in source['groups']['component_train']:
            row = group['array_row']
            assert row in allowed and manifest[row]['concept_id'] not in fit_concepts
            assert row not in bindings
            bindings[row] = (source, group['cache_row'])
    assert all(q['array_row'] in bindings for q in selected)
    assert sha(contract['target_cache_path']) == contract['target_cache_sha256']
    verified[contract['target_cache_path']] = contract['target_cache_sha256']
    # Verify only F/A/P model payloads; the attention payload is never needed.
    assets = json.loads(Path(reg['asset_verification']).read_text())
    for name in ('fusion_encoder', 'proj'):
        directory = WORK / 'assets/prior_H14' / name
        payload = directory / 'diffusion_pytorch_model.safetensors'
        assert sha(payload) == assets['files'][str(payload)]['sha256']
        assert reg['input_sha256'][str(directory / 'config.json')] == sha(directory / 'config.json')
        verified[str(payload)] = assets['files'][str(payload)]['sha256']
    done = json.loads(Path(reg['risk_completion']).read_text())
    assert done['status'] == 'COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_COMPONENT'
    assert done['arm'] == 'risk' and done['updates'] == 24807
    assert done['checkpoint_path'] == reg['risk_checkpoint']
    assert done['checkpoint_sha256'] == reg['input_sha256'][reg['risk_checkpoint']]
    return selected, fixed_negative_positions(selected), contract, bindings, verified


def main(registration_path):
    started = time.monotonic()
    # The OS default terminates this diagnostic even inside a long native call;
    # a Python signal exception could wait for that native call to return.
    signal.signal(signal.SIGALRM, signal.SIG_DFL)
    signal.alarm(SECONDS)
    before = memory_guard()  # MUST precede importing torch or the native runtime.
    reg = json.loads(Path(registration_path).read_text())
    selected, negative, contract, bindings, verified = validate_before_torch(reg)
    after_hashing = memory_guard()
    import brainhive_native_models_v1 as native
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    torch, F = native.torch, native.F
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    # A fresh post-import floor also protects an already-running GPU job's CPU RAM.
    after_import = memory_guard(1 << 30)
    state = torch.load(reg['risk_checkpoint'], weights_only=True, map_location='cpu', mmap=True)
    assert state['arm'] == 'risk' and state['updates'] == 24807
    assert state['recipe_sha256'] == sha(reg['main_recipe'])
    calibration = state['calibration']
    assert calibration['kind'] == 'Dfit_nonnegative_affine_prediction_risk_calibration'
    cal = (float(calibration['slope']), float(calibration['intercept']))
    assert cal[0] >= 0 and all(math.isfinite(v) for v in cal)
    component = BoundedConditionAdapter(predict_risk=True).eval().requires_grad_(False)
    component.load_state_dict(state['model'], strict=True)
    condition = torch.empty(32, 1024)
    weight = torch.empty_like(condition)
    bias = torch.empty_like(condition)
    # Never invoke load_training(), and never load a full/full-T condition cache.
    for source in {v[0]['role']: v[0] for v in bindings.values()}.values():
        relevant = [(i, bindings[q['array_row']][1]) for i, q in enumerate(selected)
                    if bindings[q['array_row']][0]['role'] == source['role']]
        if not relevant:
            continue
        data = torch.load(source['condition_cache_path'], weights_only=True, map_location='cpu', mmap=True)
        brain = torch.load(source['checkpoint_path'], weights_only=True, map_location='cpu', mmap=True)['model']
        assert data['model_sha256'] == source['checkpoint_sha256']
        assert set(data['array_rows'].tolist()) <= set(contract_row for contract_row in bindings)
        for dst, cache_row in relevant:
            assert int(data['array_rows'][cache_row]) == selected[dst]['array_row']
            condition[dst] = data['condition'][cache_row]
            weight[dst] = brain['block.layers.0.norm.weight']
            bias[dst] = brain['block.layers.0.norm.bias']
            selected[dst].update(parent_role=source['role'], parent_cache_row=cache_row,
                                 parent_checkpoint_sha256=source['checkpoint_sha256'])
        del data, brain
    target_cache = torch.load(contract['target_cache_path'], weights_only=True, map_location='cpu', mmap=True)
    target = target_cache[[q['array_row'] for q in selected]].clone()
    del target_cache
    assert condition.shape == target.shape == (32, 1024)
    assert all(torch.isfinite(x).all() for x in (condition, target, weight, bias))
    models = native.prior_models(device='cpu')  # include_attention defaults False.
    adapter, projector = models['fusion_encoder'].adapter, models['proj']
    def angle(a, b):
        a, b = a.double(), b.double()
        cosine = F.cosine_similarity(a, b, dim=-1).clamp(-1, 1)
        return cosine.acos() * (180 / math.pi)
    with torch.no_grad():
        changed, info = component(condition, weight, bias, calibration=cal)
        assert torch.equal(component(condition, weight, bias, enabled=False)[0], condition)
        base_tokens = projector(adapter(condition))
        risk_tokens = projector(adapter(changed))
        true_tokens = projector(adapter(target))
        wrong_tokens = true_tokens[negative]
        assert all(torch.isfinite(x).all() for x in (changed, base_tokens, risk_tokens, true_tokens))
        u = (condition - bias) / weight
        v = (changed - bias) / weight
        actual_ratio = info['standardized_delta_norm'] / info['standardized_input_norm']
        cap_ratio = .25 * info['gate']
        angular_cap = cap_ratio.double().atan() * (180 / math.pi)
        actual_angle = angle(u, v)
        assert torch.all(actual_ratio <= cap_ratio + 2e-6)
        assert torch.all(actual_angle <= angular_cap + 2e-3)
        torch.testing.assert_close(actual_angle, actual_ratio.double().atan() * (180 / math.pi),
                                   atol=2e-3, rtol=2e-3)
        # Target projected onto the host's standardized mean-zero sphere is the
        # exact nearest feasible direction for this specific angle-bound check.
        target_u = (target - bias) / weight
        projected = target_u - target_u.mean(-1, keepdim=True)
        assert torch.all(projected.norm(dim=-1) > 1e-8)
        target_angle = angle(u, projected)
        target_after_angle = angle(v, projected)
        reachable_lower_bound = (target_angle - angular_cap).clamp_min(0)
        parent_positive = F.cosine_similarity(base_tokens, true_tokens, dim=-1).mean(-1)
        parent_negative = F.cosine_similarity(base_tokens, wrong_tokens, dim=-1).mean(-1)
        risk_positive = F.cosine_similarity(risk_tokens, true_tokens, dim=-1).mean(-1)
        risk_negative = F.cosine_similarity(risk_tokens, wrong_tokens, dim=-1).mean(-1)
        def token_angle(tokens):
            return F.cosine_similarity(tokens.double(), true_tokens.double(), dim=-1).clamp(-1, 1).acos().mean(-1) * (180 / math.pi)
        base_token_angle, risk_token_angle = token_angle(base_tokens), token_angle(risk_tokens)
        rows = []
        for i, query in enumerate(selected):
            rows.append(dict(**query, negative_query_index=negative[i],
                negative_stimulus_id=selected[negative[i]]['stimulus_id'],
                parent_token_angular_error=float(1 - parent_positive[i]),
                risk_token_angular_error=float(1 - risk_positive[i]),
                parent_token_mean_angle_degrees=float(base_token_angle[i]),
                risk_token_mean_angle_degrees=float(risk_token_angle[i]),
                parent_matched_minus_wrong_cosine=float(parent_positive[i] - parent_negative[i]),
                risk_matched_minus_wrong_cosine=float(risk_positive[i] - risk_negative[i]),
                parent_matched_token_cosine=float(parent_positive[i]), parent_wrong_token_cosine=float(parent_negative[i]),
                risk_matched_token_cosine=float(risk_positive[i]), risk_wrong_token_cosine=float(risk_negative[i]),
                predicted_conditional_risk=float(info['risk'][i]), gate=float(info['gate'][i]),
                standardized_delta_ratio=float(actual_ratio[i]), permitted_ratio=float(cap_ratio[i]),
                radius_utilization=float(actual_ratio[i] / cap_ratio[i]) if cap_ratio[i] > 1e-12 else None,
                actual_standardized_angle_degrees=float(actual_angle[i]),
                permitted_standardized_angle_degrees=float(angular_cap[i]),
                nearest_target_standardized_angle_before_degrees=float(target_angle[i]),
                nearest_target_standardized_angle_after_degrees=float(target_after_angle[i]),
                best_possible_target_angle_under_this_gate_degrees=float(reachable_lower_bound[i]),
                nearest_target_inside_current_cone=bool(target_angle[i] <= angular_cap[i])))
    final_memory = memory_guard(1 << 30)
    def distribution(key):
        x = torch.tensor([r[key] for r in rows if r[key] is not None], dtype=torch.float64)
        if not len(x):
            return dict(count=0)
        return dict(count=len(x), mean=float(x.mean()), minimum=float(x.min()), maximum=float(x.max()),
                    quantiles_05_50_95=x.quantile(torch.tensor([.05, .5, .95], dtype=torch.float64)).tolist())
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, float)})
    utilization = [r['radius_utilization'] for r in rows if r['radius_utilization'] is not None]
    report = dict(status='COMPLETE_FIXED_T32_OOF_RADIUS_AND_TOKEN_DIAGNOSTIC_CPU_NO_FIT',
        subject='sub-01', queries=32, seed=1729, threads=1,
        input_registration_sha256=sha(registration_path),
        source_sha256={name: sha(ROOT / name) for name in REQUIRED_SOURCES},
        registered_input_sha256=reg['input_sha256'], verified_derived_dependencies=verified,
        risk_checkpoint_sha256=sha(reg['risk_checkpoint']), deployment_calibration=dict(slope=cal[0], intercept=cal[1]),
        rows=rows, distributions={key: distribution(key) for key in keys},
        utilization_fraction_at_least_090=sum(x >= .9 for x in utilization) / len(utilization) if utilization else None,
        utilization_fraction_at_least_099=sum(x >= .99 for x in utilization) / len(utilization) if utilization else None,
        gate_zero_queries=sum(r['gate'] <= 1e-12 for r in rows),
        maximum_ungated_angle_degrees=math.degrees(math.atan(.25)),
        exact_fallback_checked=True, radius_and_angle_bounds_checked=True,
        memory_before_torch=before, memory_after_hashing=after_hashing,
        memory_after_import=after_import, memory_final=final_memory,
        maximum_seconds=SECONDS, seconds=time.monotonic() - started,
        no_D_V_test_condition_or_image_score_access=True,
        no_UNet_or_attention_weights_loaded=True, full_training_matrix_loaded=False,
        diagnostic_precision='CPU FP32 component/A/P; cached F was computed BF16 then stored FP32',
        exact_FP16_generation_replay=False,
        target_cache_value_rows_read=[q['array_row'] for q in selected],
        integrity_hashing_reads_complete_bytes_of_target_cache=True,
        trained_parameters=0, generated_images=0, GPU_used=False,
        limits=['The component trained on these T images; these are in-fit component diagnostics, not generalization.',
                'OOF parents saw half T and differ from the deployment full-T parents.',
                'Angle bounds are in the host affine-undone coordinates, not raw token coordinates.',
                'A capacity limit is proven; saturation or token improvement does not prove an image bottleneck or gain.',
                'Risk is predicted token error; neither physiological reliability nor causal utility is established.',
                'Actual fixed32 oracle images had object successes and failures; previous inspection was not blinded.'])
    path = Path(reg['output_path'])
    assert not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
    signal.alarm(0)
    print(json.dumps(dict(status=report['status'], queries=32, seconds=report['seconds'],
                         output_path=str(path), output_sha256=sha(path))))


def self_test():
    oracle = [dict(stimulus_id=f'image_{i}.jpg', array_row=100 + 3 * i) for i in range(32)]
    protocol = dict(image_rows=dict(T=[q['array_row'] for q in oracle], D=[9], V=[10]))
    manifest = {q['array_row']: dict(stimulus_id=q['stimulus_id'], concept_id=str(i // 2), dataset_split='train')
                for i, q in enumerate(oracle)}
    selected = fixed_selection(oracle, protocol, manifest)
    negative = fixed_negative_positions(selected)
    assert negative == fixed_negative_positions(selected)
    broken = [dict(q) for q in oracle]
    broken[0]['array_row'] = 9
    try:
        fixed_selection(broken, protocol, manifest)
    except AssertionError:
        pass
    else:
        raise AssertionError('D row was accepted')
    assert math.isclose(math.degrees(math.atan(.25)), 14.036243467926479)
    assert math.atan(.25 * .5) < math.atan(.25)
    print(json.dumps(dict(status='STDLIB_MAPPING_AND_RADIUS_ALGEBRA_CHECK_PASSED', torch_imported=False,
                         query_count=32, generated_images=0, fit=False, GPU_used=False)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        assert args.config, 'A hash-complete, reviewed --config is required'
        main(args.config)
