"""Fixed 32 T-only engineering images after one arm's complete formal fit.

--recipe accepts a future independently reviewed postfit wrapper, NOT the query
DRAFT or a training recipe. It binds query_manifest, four formal_recipes, own
final_completion/calibration/provenance/checkpoint, source_sha256, input_sha256,
postfit_admission, job_name and output_directory. No receipt is issued here.
The frozen v2 training recipes/checkpoints are never changed to add this job.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import time

from brainhive_functional_training_contract_v2 import (
    ROOT, REPO, WORK, MAIN, DIAG, ARMS, DIAG_TERMINAL, core_sha, sha, read,
    bound_read, canonical, validate_recipe, verify_admission, process_live,
    resource_headroom, directory_bytes, query_identity,
)

SCHEMA = 'BrainHIVE_functional_postfit_T32_v1_recipe'
READY = 'REVIEWED_POSTFIT_T32_ENGINEERING_READY'
QUERY_SHA = '5019b618f570d9454be0744ffeb5985a939227a77835f7f990c4f755b45a29e5'
BASE = WORK / 'audit/functional_postfit_T32_v1'
ARM_BYTES, TOTAL_BYTES, SECONDS = 64 << 20, 256 << 20, 1200
BUFFER, METADATA_RESERVE = 64 << 20, 1 << 20
GENERATOR = dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1)
SOURCES = {Path(__file__).name, 'check_BrainHIVE_functional_postfit_T32_CPU_v1.py',
    'brainhive_functional_training_contract_v2.py', 'brainhive_component_generator_v1.py',
    'brainhive_native_models_v1.py', 'brainhive_bounded_component_v1.py',
    'generate_BrainHIVE_functional_alignment_V_v2.py', 'run_gpu_job_v2.py',
    'audit_BrainHIVE_spatial_image_integrity_CPU_v1.py'}


def strict(value, expected=True):
    assert type(value) is bool and value is expected, 'Unknown/string truthiness is not engineering evidence'


def plan_core(plan):
    return hashlib.sha256(canonical({k: v for k, v in plan.items() if k != 'postfit_admission'})).hexdigest()


def shape(plan):
    # MUST precede all tensor imports, GPU access, input hashing or output creation.
    assert plan.get('schema') == SCHEMA and plan.get('status') == READY, 'DRAFT/unreviewed postfit execution is refused'
    arm = plan['arm']; assert arm in ARMS
    assert plan['candidate'] == 'functional_alignment_v2_' + arm
    assert plan['generator'] == GENERATOR and plan['training_seed'] == 1729
    assert plan['subject'] == 'sub-01' and plan['queries'] == 32
    assert plan['maximum_seconds'] == SECONDS and plan['maximum_arm_output_bytes'] == ARM_BYTES
    assert plan['maximum_four_arm_output_bytes'] == TOTAL_BYTES
    assert plan['output_directory'] == str(BASE / arm)
    assert plan['query_manifest']['path'] == str(REPO / 'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json')
    assert plan['query_manifest']['sha256'] == QUERY_SHA
    assert {item['arm'] for item in plan['formal_recipes']} == set(ARMS) and len(plan['formal_recipes']) == 4
    assert SOURCES <= set(plan['source_sha256']) and all(Path(name).name == name for name in plan['source_sha256'])
    assert Path(plan['job_name']).name == plan['job_name']
    strict(plan['engineering_only']); strict(plan['semantic_evaluation_enabled'], False)
    strict(plan['automatic_retry'], False)


def terminal_check():
    states = {}
    for path, accepted in ((MAIN, {'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'}), (DIAG, DIAG_TERMINAL)):
        value = read(str(path))
        assert value['status'] in accepted and isinstance(value.get('pid'), int)
        assert not process_live(value['pid']), 'Original main/T32 controller is active'
        states[str(path)] = sha(str(path))
    return states


def verify_queries(document, oracle, protocol, image_rows, contract):
    assert document['schema'] == 'BrainHIVE_functional_postfit_T32_queries_DRAFT_v1'
    assert document['subject'] == 'sub-01' and document['queries'] == len(document['rows']) == len(oracle) == 32
    assert document['protocol_version'] == protocol['version']
    positions = {row: i for i, row in enumerate(protocol['image_rows']['T'])}
    person = next(item for item in contract['subjects'] if item['subject'] == 'sub-01')
    sources = {item['role']: item for item in person['sources'] if item['role'] in ('oof0', 'oof1')}
    assert set(sources) == {'oof0', 'oof1'}
    for index, (row, original) in enumerate(zip(document['rows'], oracle)):
        assert row['query_index'] == row['oracle_query_index'] == index and row['subject'] == 'sub-01'
        assert set(original) == {'stimulus_id', 'array_row'}
        assert all(row[key] == original[key] for key in original)
        array_row = row['array_row']; actual = image_rows[array_row]
        assert array_row in positions and row['T_position'] == positions[array_row]
        assert array_row not in protocol['image_rows']['D'] + protocol['image_rows']['V']
        assert actual['dataset_split'] == 'train' and actual['stimulus_id'] == row['stimulus_id']
        assert actual['concept_id'] == row['concept_id']
        assert row['source_role'] == 'oof' + str(protocol['oof_fold_by_concept'][row['concept_id']])
        source = sources[row['source_role']]
        matches = [item for item in source['groups']['component_train'] if item['array_row'] == array_row]
        assert len(matches) == 1 and row['cache_row'] == matches[0]['cache_row']
        for new, old in (('condition_cache', 'condition_cache'), ('parent_checkpoint', 'checkpoint'), ('parent_registration', 'parent_registration')):
            for suffix in ('_path', '_sha256'):
                assert row[new + suffix] == source[old + suffix]
        identity, seed = query_identity('sub-01', original, protocol['version'])
        assert row['query_id'] == identity and row['image_seed'] == seed
    for key in ('query_id', 'stimulus_id', 'array_row', 'image_seed'):
        assert len({row[key] for row in document['rows']}) == 32
    return sources


def final_metadata(recipe, done, calibration, provenance):
    assert done['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
    for key, expected in dict(arm=recipe['arm'], candidate=recipe['candidate'], updates=24807, seen_queries=396900,
            passes=3, subjects=10, parameters=525568, explicit_UNet_forwards_per_update=2).items():
        assert done[key] == expected
    assert done['recipe_core_sha256'] == core_sha(recipe)
    assert done['checkpoint_path'] == str(PurePosixPath(recipe['output_directory']) / 'component_final.pt')
    assert provenance['recipe_core_sha256'] == core_sha(recipe) and provenance['arm'] == recipe['arm']
    assert provenance['source_sha256'] == recipe['source_sha256']
    assert provenance['input_contract_sha256'] == recipe['input_contract_sha256']
    strict(provenance['risk_head_or_gate'], False); strict(provenance['all_frozen_host_weights'])
    assert calibration['kind'] == recipe['strength_policy'] == 'Dfit_only_global_strength_calibration'
    assert calibration['strengths'] == [0., .25, .5, .75, 1.]
    values = calibration['Dfit_token_errors']
    assert len(values) == 5 and all(math.isfinite(value) for value in values)
    index = min(range(5), key=lambda i: (values[i], i))
    assert calibration['selected_strength'] == calibration['strengths'][index]
    strict(calibration['selected_without_V_or_test_images'])


def current_final(plan, own, admission):
    """Only this arm's real final artifacts are prerequisites; future heads may be absent."""
    folder = Path(own['output_directory'])
    done, calibration, provenance = [read(folder / name) for name in ('completion.json', 'calibration.json', 'input_provenance.json')]
    final_metadata(own, done, calibration, provenance)
    for path, digest in ((folder / 'completion.json', admission['final_completion_sha256']),
            (folder / 'calibration.json', done['calibration_sha256']), (folder / 'input_provenance.json', done['input_provenance_sha256']),
            (Path(done['checkpoint_path']), done['checkpoint_sha256'])):
        assert plan['input_sha256'][str(path)] == digest == sha(path)
    return done, calibration, provenance


def prepare(plan):
    shape(plan)
    terminals = terminal_check()  # Reject live main before multi-GB CPU hashing.
    out = Path(plan['output_directory'])
    assert not out.exists(), 'Preserve earlier attempt; no overwrite/resampling/retry'
    for name, digest in plan['source_sha256'].items(): assert sha(ROOT / name) == digest
    admission = bound_read(plan['postfit_admission'])
    assert admission['schema'] == 'BrainHIVE_functional_postfit_T32_v1_reviewed_admission'
    assert admission['status'] == 'CURRENT_ARM_FINAL_POSTFIT_T32_ENGINEERING_ADMITTED'
    assert admission['arm'] == plan['arm'] and admission['plan_core_sha256'] == plan_core(plan)
    strict(admission['reviewed_internal_engineering_facts']); strict(admission['image_gain_claimed'], False)
    assert admission['issued_epoch'] <= time.time() < admission['expires_epoch']
    assert 0 < admission['expires_epoch'] - admission['issued_epoch'] <= 7 * 86400
    assert admission['terminal_state_sha256'] == terminals
    assert admission['maximum_seconds'] == SECONDS and admission['maximum_arm_output_bytes'] == ARM_BYTES
    assert admission['maximum_four_arm_output_bytes'] == TOTAL_BYTES
    recipes = {item['arm']: bound_read(item) for item in plan['formal_recipes']}
    for arm, recipe in recipes.items():
        assert recipe['arm'] == arm; validate_recipe(recipe)
    own = recipes[plan['arm']]
    # Reuse actual profile/ledger/full-V lifecycle contracts, not fabricated heads.
    assert sha(own['admission_receipt']) == own['admission_sha256']
    formal_receipt = read(own['admission_receipt'])
    review = bound_read(formal_receipt['functional_prerequisite_review'])
    for key in ('actual_GPU_diagnostic', 'native_fixed_Q_reused', 'teacher_response_nondegenerate', 'EEG_pairing_signal_present'):
        strict(review[key])
    for item in formal_receipt['actual_GPU_profiles']:
        profile = bound_read(item)
        strict(profile['actual_GPU']); strict(profile['engineering_checks_passed']); strict(profile['actual_parent_Q_engineering_verified'])
    formal = verify_admission(own, 'formal')
    lifecycle = bound_read(formal_receipt['four_arm_lifecycle_plan'])
    assert {item['arm']: item['recipe_core_sha256'] for item in lifecycle['recipes']} == {arm: core_sha(value) for arm, value in recipes.items()}
    for path, digest in plan['input_sha256'].items(): assert sha(path) == digest
    document = bound_read(plan['query_manifest'])
    protocol = read(ROOT / 'configs/split_manifest.json')
    oracle_path = document['source_selection']['oracle_inference_inputs_path']
    assert plan['input_sha256'][oracle_path] == document['source_selection']['oracle_inference_inputs_sha256'] == sha(oracle_path)
    contract_path = str(WORK / 'data/component_input_contract_v1.json')
    assert plan['input_sha256'][contract_path] == own['input_contract_sha256'] == sha(contract_path)
    contract = read(contract_path)
    assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
    manifest_path = ROOT / 'audit/image_manifest.csv'
    assert sha(manifest_path) == protocol['image_manifest_sha256'] == own['input_sha256'][str(manifest_path)]
    with manifest_path.open() as stream:
        image_rows = {int(row['array_row']): row for row in csv.DictReader(stream) if row['dataset_split'] == 'train'}
    sources = verify_queries(document, read(oracle_path), protocol, image_rows, contract)
    for source in sources.values():
        for path_key, digest_key in (('condition_cache_path', 'condition_cache_sha256'), ('checkpoint_path', 'checkpoint_sha256'), ('parent_registration_path', 'parent_registration_sha256')):
            assert plan['input_sha256'][source[path_key]] == source[digest_key] == sha(source[path_key])
        registration = read(source['parent_registration_path'])
        assert set(registration['training_array_rows']) <= set(protocol['image_rows']['T'])
        trained_concepts = {image_rows[row]['concept_id'] for row in registration['training_array_rows']}
        assert all(row['concept_id'] not in trained_concepts for row in document['rows'] if row['source_role'] == source['role'])
    done, calibration, provenance = current_final(plan, own, admission)
    limits = admission['resource_limits']
    assert limits['minimum_CPU_headroom_bytes'] >= 8 << 30 and limits['minimum_free_GPU_bytes'] >= 16 << 30
    assert resource_headroom()['effective_headroom_bytes'] >= limits['minimum_CPU_headroom_bytes']
    existing = directory_bytes(str(BASE)); assert existing <= TOTAL_BYTES
    device = Path(str(WORK)).stat().st_dev
    reserved_remaining = sum(volume['required'] for volume in formal['remaining_storage'] if Path(volume['root']).stat().st_dev == device)
    assert reserved_remaining > 0, 'Do not guess pending full-V lifecycle storage as zero'
    assert shutil.disk_usage(str(WORK)).free >= reserved_remaining + max(0, TOTAL_BYTES - existing) + BUFFER
    assert plan['job_name'] not in {name for recipe in recipes.values() for name in recipe['formal_job_names'] + [recipe['profile_job_name']] + recipe.get('generation_job_names', [])}
    return dict(document=document, sources=sources, recipe=own, completion=done, calibration=calibration,
        provenance=provenance, formal_check=formal, admission=admission, terminal_state_sha256=terminals,
        four_recipe_cores={arm: core_sha(recipe) for arm, recipe in recipes.items()})


def exclusive_wrapper(plan, evidence):
    assert str(ROOT) == str(REPO)
    parent = Path('/proc') / str(os.getppid()); command = (parent / 'cmdline').read_bytes().split(b'\0')
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in command and b'--name' in command
    assert command[command.index(b'--name') + 1].decode() == plan['job_name']
    lock = (ROOT / 'audit/GPU_JOB.lock').stat()
    descriptors = []
    for fd in (parent / 'fd').iterdir():
        try:
            value = fd.stat(); descriptors.append((value.st_dev, value.st_ino))
        except FileNotFoundError: pass
    assert (lock.st_dev, lock.st_ino) in descriptors
    # Verify the parent's existing lock. Never acquire a second exclusive lock.
    import fcntl
    with (ROOT / 'audit/GPU_JOB.lock').open('a') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN); raise AssertionError('Trusted parent must already hold GPU lock')
    job = read(ROOT / 'audit' / (plan['job_name'] + '_job.json'))
    assert job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid()
    assert 0 < job['max_seconds'] <= SECONDS
    auth = str(REPO / 'configs/resource_authorization_20261020_v3.json')
    assert job['authorization_sha256'] == evidence['recipe']['input_sha256'][auth]
    args = dict(check=True, capture_output=True, text=True, timeout=10)
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], **args)
    assert {int(line) for line in apps.stdout.splitlines() if line.strip()} <= {os.getpid()}
    capacity = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], **args)
    free = [int(line) * (1 << 20) for line in capacity.stdout.splitlines() if line.strip()]
    assert len(free) == 1 and free[0] >= evidence['admission']['resource_limits']['minimum_free_GPU_bytes']
    return dict(job_name=plan['job_name'], wrapper_pid=os.getppid(), job_receipt_sha256=sha(ROOT / 'audit' / (plan['job_name'] + '_job.json')))


def check_state(state, recipe, done, calibration):
    for key, value in dict(arm=recipe['arm'], candidate=recipe['candidate'], recipe_core_sha256=core_sha(recipe),
            seed=1729, updates=24807, seen_queries=396900, passes=3).items(): assert state[key] == value
    strict(state['risk_gate'], False); strict(state['trained_from_scratch'])
    assert state['calibration'] == calibration and state['input_provenance_sha256'] == done['input_provenance_sha256']


def load_oof(sources, torch_load):
    loaded = {}
    for role, source in sources.items():
        data = torch_load(source['condition_cache_path'], weights_only=True, map_location='cpu', mmap=True)
        brain = torch_load(source['checkpoint_path'], weights_only=True, map_location='cpu')['model']
        assert data['model_sha256'] == source['checkpoint_sha256']
        assert data['condition'].shape == (len(data['array_rows']), 1024)
        ids = data['array_rows'].tolist()
        assert all(type(value) is int for value in ids) and len(set(ids)) == len(ids)
        weight, bias = brain['block.layers.0.norm.weight'][None], brain['block.layers.0.norm.bias'][None]
        assert weight.shape == bias.shape == (1, 1024)
        loaded[role] = dict(data=data, weight=weight, bias=bias)
    return loaded


def route(row, loaded):
    source = loaded[row['source_role']]; position = row['cache_row']
    assert 0 <= position < len(source['data']['array_rows'])
    assert int(source['data']['array_rows'][position]) == row['array_row'], 'array_row is not cache/T position'
    condition = source['data']['condition'][position:position + 1].clone()
    assert condition.shape == source['weight'].shape == source['bias'].shape == (1, 1024)
    return condition, source['weight'], source['bias']


def encode(torch, component, adapter, projector, condition, weight, bias, device='cuda'):
    assert all(torch.isfinite(value).all() for value in (condition, weight, bias))
    raw, information = component(condition.to(device), weight.to(device), bias.to(device), gate_mode='fixed')
    assert torch.all(information['gate'] == 1) and torch.isfinite(raw).all()
    native = raw.to(torch.float16); tokens = projector(adapter(native))
    assert tokens.shape == (1, 4, 2048) and torch.isfinite(tokens).all()
    return raw, native, tokens, information


def sample_native(pipe, prompts, tokens, image_seed, torch):
    return pipe(height=512, width=512, generator=torch.Generator(device='cuda').manual_seed(image_seed),
        prompt_embeds=prompts[0], pooled_prompt_embeds=prompts[1], cross_attention_kwargs={'ip_hidden_states': tokens},
        num_inference_steps=4, guidance_scale=0., output_type='pil').images[0]


def save_bytes(out, path, content, reserve=True, append=False):
    margin = METADATA_RESERVE if reserve else 0
    assert directory_bytes(out) + len(content) <= ARM_BYTES - margin
    assert directory_bytes(str(BASE)) + len(content) <= TOTAL_BYTES - margin
    with Path(path).open('ab' if append else 'xb') as stream:
        stream.write(content); stream.flush(); os.fsync(stream.fileno())


def save_json(out, path, value, reserve=True):
    save_bytes(out, path, (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode(), reserve)


def execute(plan, evidence, wrapper):
    started = time.monotonic()
    import torch
    import numpy as np
    from PIL import Image
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    from brainhive_component_generator_v1 import load_native
    from audit_BrainHIVE_spatial_image_integrity_CPU_v1 import pixel_statistics
    torch.set_num_threads(2); torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    recipe, done, calibration = evidence['recipe'], evidence['completion'], evidence['calibration']
    state = torch.load(done['checkpoint_path'], weights_only=True, map_location='cpu')
    check_state(state, recipe, done, calibration)
    component = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    component.load_state_dict(state['model'], strict=True)
    assert sum(p.numel() for p in component.parameters()) == 525568
    assert all(torch.isfinite(p).all() for p in component.parameters())
    component.radius = .25 * calibration['selected_strength']
    loaded = load_oof(evidence['sources'], torch.load)
    for query in evidence['document']['rows']:
        selected = route(query, loaded)
        assert all(value.dtype == torch.float32 and torch.isfinite(value).all() for value in selected)
    # Everything above this line is CPU. No future arm's head is opened.
    assert torch.cuda.is_available(); component.cuda()
    pipe, adapter, projector, prompts, host = load_native(training=False)
    processors = [value for value in pipe.unet.attn_processors.values() if type(value).__name__ == 'IPAttnProcessor2_0']
    assert len(processors) == 70 and all(float(value.scale) == 1. for value in processors)
    out = Path(plan['output_directory']); out.mkdir(parents=True, exist_ok=False)
    images = out / 'images'; images.mkdir()
    model_receipt = dict(plan_core_sha256=plan_core(plan), query_manifest_sha256=QUERY_SHA, arm=plan['arm'],
        candidate=plan['candidate'], source_sha256=plan['source_sha256'], formal_recipe_core_sha256=core_sha(recipe),
        four_recipe_cores=evidence['four_recipe_cores'], component_checkpoint_sha256=done['checkpoint_sha256'],
        calibration=calibration, generator=host, real_OOF_sources=evidence['sources'], wrapper=wrapper,
        normal_inference_teacher_or_target_access=False, engineering_T_only=True, performance_pilot=False)
    save_json(out, out / 'model_receipt.json', model_receipt)
    progress, handles, rows = {}, [], []
    def unet_hook(module, arguments, output):
        assert arguments[0].shape[0] == 1 and torch.isfinite(output[0]).all()
        if progress['UNet_NFE'] == 0:
            progress['initial_UNet_input_sha256'] = hashlib.sha256(arguments[0].detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        progress['UNet_NFE'] += 1
    def key_hook(module, arguments):
        assert arguments[0].shape == (1, 4, 2048) and arguments[0].data_ptr() == progress['token_pointer']
        progress['IP_key_calls'] += 1
    def value_hook(module, arguments):
        assert arguments[0].shape == (1, 4, 2048) and arguments[0].data_ptr() == progress['token_pointer']
        progress['IP_value_calls'] += 1
    def vae_hook(module, arguments, output):
        assert torch.isfinite(output).all(); progress['VAE_decoder_calls'] += 1
        progress['VAE_float_min'], progress['VAE_float_max'] = float(output.min()), float(output.max())
    handles += [pipe.unet.register_forward_hook(unet_hook), pipe.vae.decoder.register_forward_hook(vae_hook)]
    for processor in processors:
        handles += [processor.to_k_ip.register_forward_pre_hook(key_hook), processor.to_v_ip.register_forward_pre_hook(value_hook)]
    try:
        for query in evidence['document']['rows']:
            assert time.monotonic() - started < SECONDS - 30, 'Finite current-arm engineering limit; preserve prefix'
            condition, weight, bias = route(query, loaded)
            progress.clear(); progress.update(UNet_NFE=0, IP_key_calls=0, IP_value_calls=0, VAE_decoder_calls=0)
            with torch.inference_mode():
                raw, native, tokens, information = encode(torch, component, adapter, projector, condition, weight, bias)
                progress['token_pointer'] = tokens.data_ptr()
                image = sample_native(pipe, prompts, tokens, query['image_seed'], torch).convert('RGB')
            assert progress['UNet_NFE'] == 4 and progress['IP_key_calls'] == progress['IP_value_calls'] == 280
            assert progress['VAE_decoder_calls'] == 1; progress.pop('token_pointer')
            pixels = np.asarray(image); assert pixels.shape == (512, 512, 3) and pixels.dtype == np.uint8
            buffer = io.BytesIO(); image.save(buffer, format='WEBP', lossless=True, quality=90, method=4)
            path = images / f"{query['query_index']:02d}_{Path(query['stimulus_id']).stem}.webp"
            save_bytes(out, path, buffer.getvalue())
            with Image.open(path) as decoded:
                assert decoded.size == (512, 512) and np.array_equal(pixels, np.asarray(decoded.convert('RGB')))
            row = dict(query, arm=plan['arm'], candidate=plan['candidate'], repeat_protocol='R4', intervention='correct',
                EEG_source_stimulus_id=query['stimulus_id'], condition_sha256=hashlib.sha256(condition.contiguous().numpy().tobytes()).hexdigest(),
                own_OOF_affine_weight_sha256=hashlib.sha256(weight.contiguous().numpy().tobytes()).hexdigest(),
                own_OOF_affine_bias_sha256=hashlib.sha256(bias.contiguous().numpy().tobytes()).hexdigest(),
                condition_norm=float(condition.norm()), component_correction_RMS=float((raw.cpu() - condition).square().mean().sqrt()),
                component_effective_radius=component.radius, component_gate=float(information['gate'][0]),
                actual_A_input_fp16_sha256=hashlib.sha256(native.cpu().contiguous().numpy().tobytes()).hexdigest(),
                IP_token_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                generated_path=str(path), generated_sha256=sha(path), RGB_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                spatial_integrity_diagnostics=pixel_statistics(pixels, np),
                component_checkpoint_sha256=done['checkpoint_sha256'], formal_recipe_core_sha256=core_sha(recipe),
                model_receipt_sha256=sha(out / 'model_receipt.json'), generation_source_sha256=sha(__file__),
                teacher_or_visual_features_used=False, selected_best_seed=False, lossless_WebP_roundtrip=True, **progress)
            assert all(math.isfinite(row[key]) for key in ('condition_norm', 'component_correction_RMS', 'VAE_float_min', 'VAE_float_max'))
            save_bytes(out, out / 'inference_manifest.jsonl', (json.dumps(row, allow_nan=False) + '\n').encode(), append=True)
            rows.append(row); print(json.dumps(dict(stage='POSTFIT_T32_ENGINEERING_ONLY', arm=plan['arm'], completed=len(rows), expected=32)), flush=True)
        assert len(rows) == 32
        noise_compared = []
        for arm in ARMS:
            previous = Path(str(BASE / arm))
            if arm != plan['arm'] and (previous / 'completion.json').exists():
                previous_done = read(previous / 'completion.json')
                assert previous_done['status'] == 'CURRENT_ARM_FIXED_POSTFIT_T32_ENGINEERING_IMAGES_SAVED'
                assert sha(previous / 'inference_manifest.jsonl') == previous_done['inference_manifest_sha256']
                prior = [json.loads(line) for line in (previous / 'inference_manifest.jsonl').read_text().splitlines()]
                assert len(prior) == 32
                for before, current in zip(prior, rows):
                    for key in ('query_id', 'image_seed', 'condition_sha256', 'own_OOF_affine_weight_sha256', 'own_OOF_affine_bias_sha256', 'initial_UNet_input_sha256'):
                        assert before[key] == current[key], 'Cross-arm engineering identity/noise mismatch; no resampling'
                noise_compared.append(arm)
        for name, digest in plan['source_sha256'].items(): assert sha(ROOT / name) == digest
        for path, digest in plan['input_sha256'].items(): assert sha(path) == digest
        assert terminal_check() == evidence['terminal_state_sha256']
        save_json(out, out / 'completion.json', dict(status='CURRENT_ARM_FIXED_POSTFIT_T32_ENGINEERING_IMAGES_SAVED',
            arm=plan['arm'], queries=32, image_seed_policy='unchanged frozen query_identity',
            inference_manifest_sha256=sha(out / 'inference_manifest.jsonl'), model_receipt_sha256=sha(out / 'model_receipt.json'),
            actual_GPU_used=True, total_UNet_NFE=128, native_IP_layers=70, images_selected_or_regenerated=False,
            teacher_or_target_at_inference=False, V_scores_read=False, semantic_scores_computed=False, picture_gain_established=False,
            exact_spatial_constants=sum(row['spatial_integrity_diagnostics']['exact_spatial_constant'] for row in rows),
            near_spatial_constants=sum(row['spatial_integrity_diagnostics']['near_spatial_constant'] for row in rows),
            decoded_pixel_duplicates_beyond_first=32-len({row['RGB_pixel_sha256'] for row in rows}),
            constant_or_duplicate_flags_do_not_select_models_or_seeds=True, cross_arm_noise_compared_to_completed_arms=noise_compared,
            T_OOF_engineering_is_not_full_parent_V_validation=True,
            fixed_T_engineering_is_not_held_out_performance=True, seconds=time.monotonic() - started,
            resource_budget_seconds=SECONDS, resource_budget_not_yet_profile_validated_in_this_source=True), reserve=False)
    except BaseException as error:
        save_json(out, out / 'failure.json', dict(status='POSTFIT_T32_PREFIX_PRESERVED_NO_RETRY', error=repr(error),
            completed_queries=len(rows), expected_queries=32, prefix_manifest_sha256=sha(out / 'inference_manifest.jsonl') if rows else None,
            no_score_or_seed_selection=True), reserve=False)
        raise
    finally:
        for handle in handles: handle.remove()


def run(path):
    plan = read(path)
    evidence = prepare(plan)
    wrapper = exclusive_wrapper(plan, evidence)
    execute(plan, evidence, wrapper)


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--recipe', required=True)
    run(parser.parse_args().recipe)


if __name__ == '__main__': main()
