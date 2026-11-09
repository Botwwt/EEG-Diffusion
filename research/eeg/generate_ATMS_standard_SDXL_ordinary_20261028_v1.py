"""Paired ATMS native/ordinary images in the unchanged public IP/SDXL host.

prepare exports all400 real full-ATMS conditions and CUDA query-noise tensors
before any image. generate commits finite, resumable prefixes without rerolls.
Only the fixed80 and fixed400 closures permit downstream scoring. No target
image or visual feature is opened by either preparation or normal generation.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
START = time.monotonic()
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import numpy as np
from PIL import Image
import train_ATMS_standard_SDXL_ordinary_20261028_v1 as T
import brainhive_gain_calibration_20261027_v1 as G

ROOT = Path(__file__).resolve().parent
SCHEMA = 'ATMS_standard_SDXL_ordinary_paired_images_20261028_v1'
ARMS = ('native_ATMS', 'ordinary_ATMS')
IDENTITY = ('query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'partition', 'image_seed')
GENERATION = dict(resolution=[512, 512], steps=20, scheduler='EulerDiscreteScheduler',
    CFG=5., IP_scale=1., UNet_forward_calls_per_image=20, UNet_sample_evaluations_per_image=40,
    generation_batch=1, one_image_per_query=True, no_best_of_N=True, negative_IP_condition='all-zero1024')
PANEL_SHA = 'e36ce08aa3d0f8ff2415c09253820e7a5ca7604d59a252813400eedc4d151be3'
TARGET_SHA = '1467a4e202d1cc031246d1c58cac407c404ecc148b0c51fa1422046fb77b8649'
sha, ref, read, checked, atomic = T.sha, T.reference, T.read, T.checked, T.atomic_json
torch = None


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def save_array(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())


def make_policy(args):
    assert args.training_spec and args.training_output_root and args.D09_completion
    assert not args.spec.exists(), 'Lock a new, reviewable image policy before any image'
    training_ref = ref(args.training_spec); training = read(checked(training_ref))
    assert training['status'] == 'ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_LOCKED_BEFORE_EFFECTS'
    provenance_ref = ref(args.D09_completion); previous = read(checked(provenance_ref))
    assert previous['status'] == 'COMPLETE_D09_STAGE400'
    inputs = read(checked(previous['inputs'])); panel_ref = inputs['panel']; target_ref = previous['check_target_features']
    assert panel_ref['sha256'] == PANEL_SHA and target_ref['sha256'] == TARGET_SHA
    names = (Path(__file__), Path(T.__file__), Path(G.__file__), ROOT / 'brainhive_bounded_component_v1.py',
        ROOT / 'profile_standard_generator.py', ROOT / 'score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2.py',
        ROOT / 'score_ATMS_standard_SDXL_ordinary_20261028_v1.py', ROOT / 'image_evaluation.py', ROOT / 'semantic_evaluator.py')
    pins = [ref(name) for name in names] + [training_ref, panel_ref, provenance_ref, target_ref]
    spec = dict(schema=SCHEMA, status='ATMS_EQUIVALENT_ORDINARY_IMAGE_POLICY_FIXED_BEFORE_IMAGES',
        source_pins=pins, training_spec=training_ref,
        training_output_root=str(args.training_output_root.resolve()), image_output_root=str(args.output.resolve()),
        Dcheck_panel=panel_ref, D09_target_provenance=provenance_ref, Dcheck_target_features=target_ref,
        generation=GENERATION, arms=list(ARMS), stages=[80, 400], gallery_targets=400,
        maximum_GPU_bytes=20 << 30, maximum_extra_CPU_bytes=10 << 30, maximum_output_bytes=1 << 30,
        maximum_seconds_per_GPU_segment=1800, segment_max_new_research_images=40,
        stage80=dict(stop_only_when_DINO_difference_at_most=-.02, and_descriptive95_upper_strictly_below=0.,
            otherwise='EXPAND400'),
        stage400=dict(practical_DINO_gain=.01, descriptive95_lower_strictly_positive=True,
            target_cosine_difference_strictly_positive=True, within_concept10_top1_difference_nonnegative=True,
            otherwise='STOP_INSTANCE'),
        stage80_research_images=160, stage400_additional_research_images=640,
        engineering_disabled_native_parity_images=1, projected_image_count_if_expanded=801,
        noise='Existing query seed, CUDA BF16 randn, exported full400 before first image; same tensor both arms',
        target_reuse='Only original real400 DINO features; native ATMS images/features must be newly generated',
        source_of_baseline='D15 genuine63x250 R4 ATMS full-T output; no BrainHIVE F/A or generated baseline',
        checkpoint='New cross-framework ordinary fixed last24807 update, complete three132300-query passes',
        strength='Fixed Dfit8200 token-angle grid0,.25,.5,.75,1; ties smaller; no Dcheck recalibration',
        no_intermediate_subset_scoring=True, no_best_of_N_or_seed_selection=True,
        independent_confirmation=False, official_author_ATM_full_pipeline_reproduced=False, SOTA_established=False)
    atomic(args.spec, spec); load_policy(args.spec)
    print(json.dumps(dict(status=spec['status'], image_policy=ref(args.spec), GPU_started=False,
        new_images=0, policy_is_not_actual_image_or_gain_evidence=True)), flush=True)


def load_policy(path, *, allow_target_features=False):
    spec = read(path)
    assert spec['schema'] == SCHEMA and spec['status'] == 'ATMS_EQUIVALENT_ORDINARY_IMAGE_POLICY_FIXED_BEFORE_IMAGES'
    assert spec['generation'] == GENERATION and spec['arms'] == list(ARMS)
    assert spec['stages'] == [80, 400] and spec['gallery_targets'] == 400
    assert spec['maximum_GPU_bytes'] == 20 << 30 and spec['maximum_extra_CPU_bytes'] == 10 << 30
    assert spec['maximum_output_bytes'] == 1 << 30 and spec['segment_max_new_research_images'] == 40
    assert spec['Dcheck_panel']['sha256'] == PANEL_SHA
    assert spec['Dcheck_target_features']['sha256'] == TARGET_SHA
    assert spec['maximum_seconds_per_GPU_segment'] == 1800
    assert spec['stage80'] == dict(stop_only_when_DINO_difference_at_most=-.02,
        and_descriptive95_upper_strictly_below=0., otherwise='EXPAND400')
    assert spec['stage400'] == dict(practical_DINO_gain=.01, descriptive95_lower_strictly_positive=True,
        target_cosine_difference_strictly_positive=True, within_concept10_top1_difference_nonnegative=True,
        otherwise='STOP_INSTANCE')
    pinmap = {}
    for item in spec['source_pins']:
        # Generation does not even hash the target-feature payload; the CPU
        # scorer verifies it after both image arms and their audit are closed.
        target_payload = Path(item['path']).resolve() == Path(spec['Dcheck_target_features']['path']).resolve()
        if allow_target_features or not target_payload:
            checked(item)
        pinmap[str(Path(item['path']).resolve())] = item['sha256']
    required = (Path(__file__), Path(T.__file__), Path(G.__file__), ROOT / 'brainhive_bounded_component_v1.py',
        ROOT / 'profile_standard_generator.py', ROOT / 'score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2.py',
        ROOT / 'score_ATMS_standard_SDXL_ordinary_20261028_v1.py', ROOT / 'image_evaluation.py', ROOT / 'semantic_evaluator.py')
    for item in required:
        assert pinmap.get(str(item.resolve())) == sha(item), str(item)
    for key in ('training_spec', 'Dcheck_panel', 'D09_target_provenance', 'Dcheck_target_features'):
        item = spec[key]
        if allow_target_features or key != 'Dcheck_target_features':
            checked(item)
        assert pinmap.get(str(Path(item['path']).resolve())) == item['sha256'], key
    return spec


def panel_data(spec):
    recipe = read(checked(spec['training_spec']))
    assert recipe['status'] == 'ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_LOCKED_BEFORE_EFFECTS'
    split = read(checked(recipe['split']))
    assert split['subjects'] == [f'sub-{i:02d}' for i in range(1, 11)]
    assert set(split['train_concepts']) == {'T', 'D', 'V'}
    assert len(split['train_concepts']['T']) == 1323 and len(split['train_concepts']['D']) == 165
    assert set(split['d_fit_concepts']).isdisjoint(split['d_check_concepts'])
    assert set(split['d_fit_concepts']) | set(split['d_check_concepts']) == set(split['train_concepts']['D'])
    with checked(recipe['image_manifest']).open(encoding='utf-8') as stream:
        metadata = {int(row['array_row']): row for row in csv.DictReader(stream) if row['dataset_split'] == 'train'}
    assert len(metadata) == 16540
    assert {metadata[row]['concept_id'] for row in split['image_rows']['T']} == set(split['train_concepts']['T'])
    assert {metadata[row]['concept_id'] for row in split['image_rows']['D']} == set(split['train_concepts']['D'])
    panel = read(checked(spec['Dcheck_panel']))
    assert panel['status'] == 'FIXED_D09_DFIT200_DCHECK400_BLIND_PANEL'
    queries = panel['Dcheck']; assert queries == G.choose_queries(split, metadata, 'Dcheck', 40)
    assert len(queries) == 400 and len({q['query_id'] for q in queries}) == 400
    assert all(q['partition'] == 'Dcheck' and q['subject_id'] == 'sub-01' for q in queries)
    assert not {q['concept_id'] for q in queries}.intersection(set(split['train_concepts']['T']) | set(split['d_fit_concepts']))
    assert len({q['concept_id'] for q in queries[:80]}) == 8
    return recipe, split, metadata, queries


def trained_inputs(spec):
    root = Path(spec['training_output_root'])
    training = read(root / 'train/completion.json')
    assert training['status'] == 'COMPLETE_ALL10_ATMS_OOF_THREE_PASSES_ORDINARY_TRAINING'
    assert training['updates'] == 24807 and training['seen'] == 396900
    assert training['per_pass_unique_query_coverage'] == [132300] * 3
    calibration = read(root / 'calibrate/completion.json')
    assert calibration['status'] == 'ATMS_STANDARD_ORDINARY_DFIT_ONLY_FIXED_GRID_CALIBRATION_COMPLETE'
    assert calibration['train_completion'] == ref(root / 'train/completion.json')
    assert calibration['fixed_final_checkpoint'] == training['final_checkpoint']
    assert calibration['spec']['sha256'] == spec['training_spec']['sha256']
    assert calibration['strengths'] == [0., .25, .5, .75, 1.] and calibration['Dfit_queries'] == 8200
    errors = calibration['Dfit_token_angular_errors']
    assert len(errors) == 5 and np.isfinite(errors).all()
    expected = min(range(5), key=lambda i: (errors[i], calibration['strengths'][i]))
    assert calibration['selected_strength'] == calibration['strengths'][expected]
    assert calibration['Dcheck_tensor_values_selected'] == 0 and not calibration['Dcheck_or_official_test_used_to_select']
    final = torch.load(checked(training['final_checkpoint']), map_location='cpu', weights_only=True)
    assert final['arm'] == 'ATMS_standard_SDXL_ordinary' and final['parameters'] == 525568
    assert final['updates'] == 24807 and final['seen_queries'] == 396900 and final['passes'] == 3
    assert final['spec']['sha256'] == spec['training_spec']['sha256']
    assert final['original_ordinary_HIVE_weights_loaded'] is False
    provenance = read(checked(final['input_provenance']))
    assert provenance['source']['sha256'] == sha(T.__file__)
    return training, calibration, final, provenance


def bootstrap(spec, maximum_seconds, *, CUDA=True):
    global torch
    from profile_standard_generator import torch as actual_torch
    import torch.nn.functional as F
    torch = actual_torch; T.torch = torch; T.F = F
    torch.set_num_threads(2)
    if not CUDA:
        return None
    wrapper = T.wrapper_proof(); torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True); torch.cuda.reset_peak_memory_stats()
    budget = T.Resources(maximum_seconds, spec); budget.started = START
    return budget, wrapper


def check_budget(spec, output, budget, *, work=True):
    budget.check()
    if work:
        assert not budget.stopping(), 'Finite image work limit; preserve existing prefix'
    size = sum(p.stat().st_size for p in Path(output).rglob('*') if p.is_file())
    assert size <= spec['maximum_output_bytes'], 'Output allocation cap'


def prepare(args, spec):
    directory = args.output / 'prepared'; assert not directory.exists(); directory.mkdir(parents=True)
    budget, wrapper = bootstrap(spec, args.max_seconds)
    recipe, split, metadata, queries = panel_data(spec)
    training, calibration, final, trained = trained_inputs(spec)
    exported = read(checked(recipe['ATMS_export_completion'])); upstream = read(checked(recipe['ATMS_input_completion']))
    assert exported['status'] == 'D15_ALL10_COMPLETE_ATMS_OOF_T_AND_FULL_D_REEXPORTED'
    assert exported['inputs']['sha256'] == recipe['ATMS_input_completion']['sha256']
    assert exported['D_models_fit_only_T'] and exported['D_queries'] == 16500 and exported['input_shape'] == [63, 250]
    assert upstream['D_array_rows'] == split['image_rows']['D']
    assert set(upstream['subjects']) == set(split['subjects'])
    assert upstream['subjects']['sub-01']['models']['full']['export_array_rows'] == split['image_rows']['D']
    checkpoint = upstream['subjects']['sub-01']['models']['full']['checkpoint']
    parent = torch.load(checked(checkpoint), map_location='cpu', weights_only=True, mmap=True)
    affines = torch.load(checked(exported['affines']), map_location='cpu', weights_only=True)
    affine = affines['sub-01/full']; weight, bias = affine['weight'], affine['bias']
    assert torch.equal(weight, parent['proj_eeg.2.weight']) and torch.equal(bias, parent['proj_eeg.2.bias'])
    assert weight.shape == bias.shape == (1024,) and weight.abs().min() > 1e-6
    cache = np.load(checked(exported['D_ATMS_full']), mmap_mode='r', allow_pickle=False)
    assert cache.shape == (16500, 1024) and cache.dtype == np.float32
    lookup = {row: i for i, row in enumerate(upstream['D_array_rows'])}
    positions = [lookup[q['array_row']] for q in queries]  # sub-01 is the first genuine1650-query subject block.
    native = np.array(cache[positions], copy=True); assert np.isfinite(native).all()
    model = T.new_adapter().eval(); model.load_state_dict(final['model'], strict=True)
    model.radius = .25 * calibration['selected_strength']
    bank = np.empty((400, 2, 1024), np.float32); noise = torch.empty((400, 4, 64, 64), dtype=torch.bfloat16)
    identities = []; maximum_replay_error = 0.
    # CPU device replay uses the same class implementation, not independent formulas.
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    cpu_model = BoundedConditionAdapter(predict_risk=False).eval(); cpu_model.load_state_dict(final['model'], strict=True)
    cpu_model.radius = model.radius
    with torch.no_grad():
        for i, query in enumerate(queries):
            check_budget(spec, args.output, budget)
            c = torch.from_numpy(native[i:i + 1]).cuda(); w = weight[None].cuda(); b = bias[None].cuda()
            disabled = model(c, w, b, enabled=False)[0]; assert torch.equal(c, disabled)
            changed = c if model.radius == 0 else model(c, w, b, gate_mode='fixed')[0]
            replay = torch.from_numpy(native[i:i + 1])
            replay = replay if model.radius == 0 else cpu_model(replay, weight[None], bias[None], gate_mode='fixed')[0]
            error = float((changed.cpu() - replay).abs().max()); maximum_replay_error = max(maximum_replay_error, error)
            assert torch.allclose(changed.cpu(), replay, atol=3e-5, rtol=1e-5), 'Same-implementation CPU device replay'
            bank[i, 0], bank[i, 1] = native[i], changed[0].cpu().numpy()
            value = torch.randn((1, 4, 64, 64), device='cuda', dtype=torch.bfloat16,
                generator=torch.Generator(device='cuda').manual_seed(query['image_seed']))
            noise[i] = value[0].cpu()
            item = {key: query[key] for key in IDENTITY}
            item.update(query_index=i, condition_source_block='sub-01/full', condition_source_D_row=positions[i],
                native_condition_sha256=array_sha(bank[i, 0]), ordinary_condition_sha256=array_sha(bank[i, 1]),
                native_condition_BF16_sha256=tensor_sha(c.to(torch.bfloat16)),
                ordinary_condition_BF16_sha256=tensor_sha(changed.to(torch.bfloat16)),
                initial_noise_latents_sha256=tensor_sha(value), full_ATMS_checkpoint=checkpoint,
                full_LN_weight_sha256=tensor_sha(weight), full_LN_bias_sha256=tensor_sha(bias),
                ordinary_final_checkpoint=training['final_checkpoint'], selected_strength=calibration['selected_strength'],
                target_image_sha256=metadata[query['array_row']]['image_sha256'])
            identities.append(item)
    save_array(directory / 'conditions.npy', bank); T.atomic_torch(directory / 'initial_noise.pt', noise)
    T.atomic_torch(directory / 'full_ATMS_affine.pt', dict(weight=weight, bias=bias, eps=affine['eps']))
    atomic(directory / 'identities.json', identities)
    lock = dict(schema=SCHEMA, status='ALL400_ATMS_NATIVE_ORDINARY_CONDITIONS_NOISE_FIXED_BEFORE_IMAGES',
        spec=ref(args.spec), source=ref(__file__), panel=spec['Dcheck_panel'], training_spec=spec['training_spec'],
        training_completion=ref(Path(spec['training_output_root']) / 'train/completion.json'),
        calibration_completion=ref(Path(spec['training_output_root']) / 'calibrate/completion.json'),
        raw_final_checkpoint=training['final_checkpoint'], trained_input_provenance=final['input_provenance'],
        selected_strength=calibration['selected_strength'], full_ATMS_checkpoint=checkpoint,
        D15_conditions=exported['D_ATMS_full'], original_affines=exported['affines'],
        affine=ref(directory / 'full_ATMS_affine.pt'), conditions=ref(directory / 'conditions.npy'),
        noise=ref(directory / 'initial_noise.pt'), identities=ref(directory / 'identities.json'),
        generation=GENERATION, arms=list(ARMS), queries=400, shape=[400, 2, 1024],
        trained_host_state_digests=trained['original_host_state_digests'], trained_runtime=trained['runtime'],
        all400_disabled_conditions_exact=True, all400_same_implementation_CPU_replay_max_abs_error=maximum_replay_error,
        Dcheck_target_images_or_features_read=False, Dcheck_conditions_used_for_any_fit_or_selection=False,
        HIVE_images_used_as_ATMS_baseline=False, normal_inference_other_subject_EEG=False,
        scientific_updates=0, new_images=0, wrapper=wrapper, **budget.result())
    atomic(directory / 'completion.json', lock)
    print(json.dumps(dict(status=lock['status'], completion=ref(directory / 'completion.json'))), flush=True)


def prepared(output, spec, spec_path):
    lock = read(Path(output) / 'prepared/completion.json')
    assert lock['schema'] == SCHEMA and lock['status'] == 'ALL400_ATMS_NATIVE_ORDINARY_CONDITIONS_NOISE_FIXED_BEFORE_IMAGES'
    assert lock['spec'] == ref(spec_path) and lock['panel'] == spec['Dcheck_panel']
    assert lock['generation'] == GENERATION and lock['arms'] == list(ARMS) and lock['queries'] == 400
    for key in ('raw_final_checkpoint', 'trained_input_provenance', 'training_completion', 'calibration_completion',
                'full_ATMS_checkpoint', 'original_affines', 'conditions', 'noise', 'affine', 'identities'):
        checked(lock[key])
    bank = np.load(lock['conditions']['path'], allow_pickle=False)
    assert bank.shape == (400, 2, 1024) and bank.dtype == np.float32 and np.isfinite(bank).all()
    identities = read(lock['identities']['path']); assert len(identities) == 400
    _, _, _, queries = panel_data(spec)
    for i, (query, item) in enumerate(zip(queries, identities)):
        assert all(query[k] == item[k] for k in IDENTITY) and item['query_index'] == i
        assert array_sha(bank[i, 0]) == item['native_condition_sha256']
        assert array_sha(bank[i, 1]) == item['ordinary_condition_sha256']
    return lock, bank, identities, queries


def image_paths(output, ordinal, arm):
    folder = Path(output) / 'images' / arm; folder.mkdir(parents=True, exist_ok=True)
    return folder / f'{ordinal:03d}.webp', folder / f'{ordinal:03d}.json'


def pixels(path):
    with Image.open(path) as image:
        assert image.size == (512, 512)
        return np.array(image.convert('RGB'), copy=True)


def validate_image(row, path, identity, arm, lock_ref):
    assert row['arm'] == arm and row['prepared'] == lock_ref and row['generation'] == GENERATION
    assert all(row[key] == value for key, value in identity.items()), 'Locked identity/target/affine/weight binding changed'
    assert row['query_index'] == identity['query_index'] and Path(row['generated_path']).resolve() == Path(path).resolve()
    assert sha(path) == row['generated_sha256'] and array_sha(pixels(path)) == row['generated_pixel_sha256']
    key = 'native' if arm == ARMS[0] else 'ordinary'
    assert row['condition_sha256'] == identity[key + '_condition_sha256']
    assert row['condition_BF16_sha256'] == identity[key + '_condition_BF16_sha256']
    assert row['initial_noise_latents_sha256'] == identity['initial_noise_latents_sha256']
    assert row['normal_inference_target_access'] is False and row['host_family'] == 'ATMS_standard_SDXL'


def stage_rows(output, count, identities, lock_ref):
    rows = []
    for i in range(count):
        for arm in ARMS:
            path, receipt = image_paths(output, i, arm)
            assert path.is_file() and receipt.is_file(), f'Incomplete fixed stage: {i} {arm}'
            row = read(receipt); validate_image(row, path, identities[i], arm, lock_ref); rows.append(row)
    return rows


def stage400_gate(output, spec_path, lock_ref):
    score_path = Path(output) / 'score80/completion.json'; previous = read(score_path)
    assert previous['schema'] == SCHEMA and previous['status'] == 'COMPLETE_ATMS_ORDINARY_PAIRED_DINO_SCREEN'
    assert previous['stage'] == 80 and previous['decision']['action'] == 'EXPAND400'
    assert previous['spec'] == ref(spec_path) and previous['prepared'] == lock_ref
    checked(previous['image_closure']); checked(previous['feature_completion']); checked(previous['per_query'])
    return ref(score_path)


def generate(args, spec):
    lock, bank, identities, queries = prepared(args.output, spec, args.spec)
    lock_ref = ref(args.output / 'prepared/completion.json')
    previous = stage400_gate(args.output, args.spec, lock_ref) if args.stage == 400 else None
    completion_path = args.output / f'image_closure{args.stage}.json'; assert not completion_path.exists()
    existing = list((args.output / 'images').glob('*/*.json')) if (args.output / 'images').exists() else []
    assert args.resume or not existing, 'Existing images require explicit --resume'
    budget, wrapper = bootstrap(spec, args.max_seconds)
    host = T.setup_host(read(checked(spec['training_spec'])), budget)
    assert host['original'] == lock['trained_host_state_digests'] and host['runtime'] == lock['trained_runtime']
    pipe = host['pipe']; pipe.set_progress_bar_config(disable=True)
    assert type(pipe.scheduler).__name__ == 'EulerDiscreteScheduler'
    from diffusers.utils.torch_utils import randn_tensor
    noise = torch.load(checked(lock['noise']), map_location='cpu', weights_only=True, mmap=True)
    assert noise.shape == (400, 4, 64, 64) and noise.dtype == torch.bfloat16
    prompt, negative, pool, negative_pool = host['prompts']
    actual_calls = dict(UNet_forward_calls=0, UNet_sample_evaluations=0); first_input = []
    def observe(module, arguments, kwargs):
        value = arguments[0]; actual_calls['UNet_forward_calls'] += 1; actual_calls['UNet_sample_evaluations'] += len(value)
        packed = kwargs['added_cond_kwargs']['image_embeds'][0]
        assert packed.shape == (2, 1, 1024) and torch.count_nonzero(packed[0]) == 0
        if not first_input:
            first_input.append(tensor_sha(value[1:2]))
    observer = pipe.unet.register_forward_pre_hook(observe, with_kwargs=True)
    attempt = args.output / 'generation_attempts' / str(time.time_ns()); attempt.mkdir(parents=True)
    rendered = 0
    def render(i, arm, path, condition=None):
        check_budget(spec, args.output, budget)
        c = torch.from_numpy(bank[i, ARMS.index(arm):ARMS.index(arm) + 1].copy()).cuda() if condition is None else condition
        deployed = c.to(torch.bfloat16); initial = noise[i:i + 1].clone().cuda()
        assert tensor_sha(initial) == identities[i]['initial_noise_latents_sha256']
        # Reconstruct the private CUDA stream independently of the exported tensor.
        regenerated = randn_tensor((1, 4, 64, 64), generator=torch.Generator(device='cuda').manual_seed(queries[i]['image_seed']),
            device=torch.device('cuda'), dtype=torch.bfloat16)
        assert torch.equal(initial, regenerated)
        first_input.clear(); before = dict(actual_calls); tick = time.monotonic()
        with torch.inference_mode():
            with torch.autocast('cuda', dtype=torch.bfloat16):
                tokens = host['P'](c)
            latent = pipe(prompt_embeds=prompt, negative_prompt_embeds=negative,
                pooled_prompt_embeds=pool, negative_pooled_prompt_embeds=negative_pool,
                ip_adapter_image_embeds=[torch.cat([torch.zeros_like(deployed), deployed])[:, None, :]],
                height=512, width=512, num_inference_steps=20, guidance_scale=5., num_images_per_prompt=1,
                latents=initial, output_type='latent').images
            assert latent.shape == (1, 4, 64, 64) and torch.isfinite(latent).all()
            decoded = pipe.vae.decode(latent.float() / pipe.vae.config.scaling_factor, return_dict=False)[0]
            assert torch.isfinite(decoded).all()
            image = pipe.image_processor.postprocess(decoded, output_type='pil')[0]
        assert actual_calls['UNet_forward_calls'] - before['UNet_forward_calls'] == 20
        assert actual_calls['UNet_sample_evaluations'] - before['UNet_sample_evaluations'] == 40
        path.parent.mkdir(parents=True, exist_ok=True); pending = path.with_name(path.name + '.pending')
        assert not path.exists() and not pending.exists()
        image.save(pending, format='WEBP', lossless=True, exact=True); os.replace(pending, path)
        row = dict(identities[i], arm=arm, prepared=lock_ref, generation=GENERATION,
            host_family='ATMS_standard_SDXL', generated_path=str(path.resolve()), generated_sha256=sha(path),
            generated_pixel_sha256=array_sha(pixels(path)), condition_sha256=tensor_sha(c),
            condition_BF16_sha256=tensor_sha(deployed), tokens_sha256=tensor_sha(tokens),
            initial_UNet_input_sha256=first_input[0], ordinary_checkpoint=lock['raw_final_checkpoint'],
            normal_inference_target_access=False, target_metadata_attached_without_pixels=True,
            actual_UNet_forward_calls=20, actual_UNet_sample_evaluations=40,
            compression=dict(format='WEBP', lossless=True), seconds=time.monotonic() - tick)
        return row
    try:
        start = 80 if args.stage == 400 else 0
        if args.stage == 400:
            old = read(args.output / 'image_closure80.json')
            prefix = stage_rows(args.output, 80, identities, lock_ref)
            assert old['manifest']['sha256'] == sha(checked(old['manifest']))
            assert [json.loads(line) for line in checked(old['manifest']).read_text().splitlines()] == prefix
        for i in range(start, args.stage):
            for arm in ARMS:
                path, receipt = image_paths(args.output, i, arm)
                if path.exists() or receipt.exists():
                    assert path.exists() and receipt.exists(), 'Preserve orphan output; no overwrite/reroll'
                    row = read(receipt); validate_image(row, path, identities[i], arm, lock_ref)
                else:
                    if rendered >= 40 or budget.stopping():
                        break
                    row = render(i, arm, path); atomic(receipt, row); rendered += 1
                    T.append_event(attempt / 'committed_images.jsonl', row)
                if i == 0 and arm == ARMS[0] and not (args.output / 'disabled_pixel_parity.json').exists():
                    final = torch.load(checked(lock['raw_final_checkpoint']), map_location='cpu', weights_only=True)
                    affine = torch.load(checked(lock['affine']), map_location='cpu', weights_only=True)
                    model = T.new_adapter().eval(); model.load_state_dict(final['model'], strict=True)
                    c = torch.from_numpy(bank[0, 0:1].copy()).cuda()
                    with torch.no_grad():
                        disabled = model(c, affine['weight'][None].cuda(), affine['bias'][None].cuda(), enabled=False)[0]
                    assert torch.equal(c, disabled)
                    replay = render(0, ARMS[0], args.output / 'engineering/disabled_ordinary.webp', condition=disabled)
                    assert all(row[k] == replay[k] for k in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256',
                        'condition_BF16_sha256', 'tokens_sha256', 'generated_sha256', 'generated_pixel_sha256'))
                    atomic(args.output / 'disabled_pixel_parity.json', dict(status='ATMS_NATIVE_DISABLED_ORDINARY_EXACT_PIXELS_PASSED',
                        native=row, disabled=replay, new_engineering_images=1, new_research_images=0, no_trained_head_changes=True))
                    del model
            if rendered >= 40 or budget.stopping():
                break
        assert T.host_digests(pipe) == host['original']; check_budget(spec, args.output, budget, work=False)
        parity_path = args.output / 'disabled_pixel_parity.json'
        assert parity_path.exists() and read(parity_path)['status'] == 'ATMS_NATIVE_DISABLED_ORDINARY_EXACT_PIXELS_PASSED'
        try:
            records = stage_rows(args.output, args.stage, identities, lock_ref)
        except AssertionError:
            records = None
        status = 'ATMS_PAIRED_IMAGE_SEGMENT_COMPLETED_RESUMABLE'
        if records is not None:
            manifest = args.output / f'manifest{args.stage}.jsonl'
            with manifest.open('x', encoding='utf-8') as stream:
                for row in records:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush(); os.fsync(stream.fileno())
            closure = dict(schema=SCHEMA, status='COMPLETE_FIXED_ATMS_PAIRED_IMAGE_STAGE', stage=args.stage,
                queries=args.stage, images=2 * args.stage, arms=list(ARMS), generation=GENERATION,
                spec=ref(args.spec), prepared=lock_ref, manifest=ref(manifest), source=ref(__file__),
                disabled_pixel_parity=ref(parity_path), previous_score=previous,
                prefix80_manifest=ref(args.output / 'manifest80.jsonl') if args.stage == 400 else None,
                frozen_host_state_digests=host['original'], all_host_tensor_digests_unchanged=True,
                HIVE_images_used_as_ATMS_baseline=False, no_intermediate_subset_scored=True,
                normal_inference_target_access=False, independent_confirmation=False, SOTA_established=False)
            atomic(completion_path, closure); status = closure['status']
        done = dict(status=status, stage=args.stage, new_research_images=rendered,
            actual_calls=actual_calls, source=ref(__file__), wrapper=wrapper,
            image_closure=ref(completion_path) if completion_path.exists() else None, **budget.result())
        atomic(attempt / 'completion.json', done); print(json.dumps(done), flush=True)
        return 0 if rendered or records is not None else 2
    except BaseException as exc:
        atomic(attempt / 'failure.json', dict(status='ATMS_PAIRED_IMAGE_ATTEMPT_FAILED_PREFIX_PRESERVED',
            error=repr(exc), traceback=traceback.format_exc(), new_research_images=rendered,
            no_automatic_retry=True, **budget.result()))
        raise
    finally:
        observer.remove()


def audit(args, spec):
    lock, bank, identities, queries = prepared(args.output, spec, args.spec)
    lock_ref = ref(args.output / 'prepared/completion.json')
    closure = read(args.output / f'image_closure{args.stage}.json')
    assert closure['status'] == 'COMPLETE_FIXED_ATMS_PAIRED_IMAGE_STAGE' and closure['prepared'] == lock_ref
    assert closure['spec'] == ref(args.spec) and closure['generation'] == GENERATION
    records = stage_rows(args.output, args.stage, identities, lock_ref)
    assert records == [json.loads(line) for line in checked(closure['manifest']).read_text().splitlines()]
    for i in range(args.stage):
        a, b = records[2 * i:2 * i + 2]
        assert a['initial_noise_latents_sha256'] == b['initial_noise_latents_sha256']
        assert a['initial_UNet_input_sha256'] == b['initial_UNet_input_sha256']
        assert a['generated_path'] != b['generated_path'] and not os.path.samefile(a['generated_path'], b['generated_path'])
    parity = read(checked(closure['disabled_pixel_parity']))
    assert parity['status'] == 'ATMS_NATIVE_DISABLED_ORDINARY_EXACT_PIXELS_PASSED'
    assert sha(parity['disabled']['generated_path']) == parity['disabled']['generated_sha256']
    assert array_sha(pixels(parity['disabled']['generated_path'])) == parity['native']['generated_pixel_sha256']
    if args.stage == 400:
        stage400_gate(args.output, args.spec, lock_ref)
        old = [json.loads(line) for line in checked(closure['prefix80_manifest']).read_text().splitlines()]
        assert records[:160] == old
    path = args.output / f'image_audit{args.stage}.json'
    atomic(path, dict(schema=SCHEMA, status='ALL_FIXED_STAGE_ATMS_IMAGES_IDENTITY_CONDITION_NOISE_PIXELS_CPU_AUDITED',
        stage=args.stage, image_closure=ref(args.output / f'image_closure{args.stage}.json'),
        spec=ref(args.spec), prepared=lock_ref, images=len(records), source=ref(__file__),
        hidden_exclusions=0, CUDA_used=False, HIVE_images_used_as_ATMS_baseline=False,
        actual_image_quality_or_gain_established=False))
    print(json.dumps(dict(status='CPU_ATMS_PAIRED_IMAGE_AUDIT_COMPLETE', audit=ref(path))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('policy', 'prepare', 'generate', 'audit'), required=True)
    parser.add_argument('--training-spec', type=Path)
    parser.add_argument('--training-output-root', type=Path)
    parser.add_argument('--D09-completion', dest='D09_completion', type=Path)
    parser.add_argument('--stage', type=int, choices=(80, 400), default=80)
    parser.add_argument('--max-seconds', type=int, default=1800)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); assert 120 <= args.max_seconds <= 1800
    if args.mode == 'policy':
        assert not args.resume; return make_policy(args)
    spec = load_policy(args.spec)
    assert Path(spec['image_output_root']).resolve() == args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'prepare':
        assert not args.resume; return prepare(args, spec)
    if args.mode == 'generate':
        return generate(args, spec)
    assert not args.resume; return audit(args, spec)


if __name__ == '__main__':
    raise SystemExit(main())
