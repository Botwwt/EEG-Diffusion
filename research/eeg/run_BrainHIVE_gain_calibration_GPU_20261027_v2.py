"""D09: frozen three-arm images -> Dfit gain calibration -> locked EEG policy.

v2 only runs the fixed400 expansion, overlapping byte-identical CPU saves.
One host load per phase. All three images are research controls; normal policy
inference selects its arm BEFORE generation and uses only four UNet forwards.
The old formal D arm and all pretrained component weights remain untouched.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import traceback
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import brainhive_gain_calibration_20261027_v1 as U
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N
from brainhive_bounded_component_v1 import BoundedConditionAdapter

ROOT = Path(__file__).resolve().parent


@torch.inference_mode()
def three_conditions(native, weight, bias, models, calibrations):
    outputs = []; gates = []
    for start in range(0, len(native), 512):
        x = native[start:start + 512]
        w = weight[None].expand_as(x); b = bias[None].expand_as(x)
        ordinary, _ = models['ordinary'](x, w, b)
        risk, info = models['risk'](x, w, b, calibration=calibrations['risk'])
        outputs.append(torch.stack([x, ordinary, risk], 1)); gates.append(info['gate'])
    return torch.cat(outputs), torch.cat(gates)


def feature_rows(conditions, gates, center, scale, basis):
    c = conditions[:, 0]; d1 = conditions[:, 1] - c; d2 = conditions[:, 2] - c
    rows = [((c - center) / scale) @ basis, (d1 / scale) @ basis, (d2 / scale) @ basis,
            d1.norm(dim=1, keepdim=True) / c.norm(dim=1, keepdim=True),
            d2.norm(dim=1, keepdim=True) / c.norm(dim=1, keepdim=True),
            F.cosine_similarity(d1, d2, dim=1, eps=1e-8)[:, None], gates[:, None]]
    result = torch.cat(rows, 1); assert result.shape == (len(c), 28) and torch.isfinite(result).all()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('input-completion', 'spec', 'native-registration', 'output', 'previous'):
        parser.add_argument('--' + key, type=Path)
    parser.add_argument('--stage', choices=('80', '400'), required=True)
    parser.add_argument('--gpu-job-name', required=True)
    parser.add_argument('--max-seconds', type=float, default=840)
    args = parser.parse_args(); stage = int(args.stage)
    assert stage == 400, 'v2 is engineering-only fixed-policy expansion; no calibration fitting path'
    assert all((args.input_completion, args.spec, args.native_registration, args.output))
    assert (stage == 400) == bool(args.previous) and 0 < args.max_seconds <= 840
    assert not args.output.exists(); args.output.mkdir(parents=True)
    invoked = time.monotonic(); stopped = [False]; generated = []; active = 'INITIALIZING'
    active_image_saves = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    def log(value):
        value = value | dict(seconds=time.monotonic() - invoked)
        with (args.output / 'progress.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(value, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        print(json.dumps(value), flush=True)
    try:
        spec = U.read(args.spec); assert spec['schema'] == U.SCHEMA and spec['feature_dimensions'] == 28
        assert spec['ridge'] == 1 and spec['coefficients'] == 58 and spec['PCA_components'] == 8
        inputs = U.read(args.input_completion); assert inputs['status'] == 'D09_REAL_CPU_INPUTS_AND_FROZEN_MODELS_VERIFIED'
        assert inputs['spec']['sha256'] == U.sha(args.spec)
        for source in spec['source_pins']: U.checked(source)
        panel = U.read(U.checked(inputs['panel'])); metadata_path = U.checked(inputs['target_metadata'])
        assert len(panel['Dfit']) == 200 and len(panel['Dcheck']) == 400 and len(panel['T_rows']) == 13230
        for key in ('native_cache', 'native_checkpoint', 'baseline_completion', 'parity_condition', 'split', 'image_manifest'): U.checked(inputs[key])
        for component in inputs['components'].values(): U.checked(component['checkpoint']); U.checked(component['completion'])
        import brainhive_paired_study_contract_20261025_v1 as C
        config = C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds, maximum_output_bytes=1536 << 20,
            maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
            resource_authorization=config['resource_authorization'])
        wrapper = D.verify_wrapper(document); budget = D.Budget(document)
        def check():
            assert not stopped[0] and time.monotonic() - invoked < args.max_seconds - 30, 'Finite D09 phase work budget; preserve incomplete artifacts'
            budget.check()
            # Do not stat atomic temporary names while the CPU saver renames.
            # At most964 image files, each independently capped at1MiB, plus small metadata/features.
            if all(future.done() for future in active_image_saves):
                assert safe_bytes_used(args.output) < (1536 << 20) - (32 << 20)
        torch.set_num_threads(2); torch.manual_seed(1729); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        sources = {Path(item['path']).name: item for item in spec['source_pins']}
        U.atomic(args.output / 'provenance.json', dict(schema=U.SCHEMA, stage=stage, inputs=U.ref(args.input_completion), spec=U.ref(args.spec),
            source=U.ref(__file__), source_pins=spec['source_pins'], wrapper=wrapper,
            V_fit_permission_changed=False, original_D_resumed=False, generator_weights_updated=False,
            check_choices_must_precede_check_images=True, sampling_targets_not_inputs=True))
        if stage == 80:
            active = 'FROZEN_CONDITION_AND_T_ONLY_FEATURES'
            cache = torch.load(inputs['native_cache']['path'], weights_only=True, map_location='cpu', mmap=True)
            state = torch.load(inputs['native_checkpoint']['path'], weights_only=True, map_location='cpu')['model']
            weight = state['block.layers.0.norm.weight'].cuda(); bias = state['block.layers.0.norm.bias'].cuda()
            models = {}; calibrations = {}
            for arm in ('ordinary', 'risk'):
                saved = torch.load(inputs['components'][arm]['checkpoint']['path'], weights_only=True, map_location='cpu')
                model = BoundedConditionAdapter(predict_risk=arm == 'risk').cuda().eval().requires_grad_(False)
                model.load_state_dict(saved['model'], strict=True)
                cal = saved['calibration']
                if arm == 'ordinary': model.radius = .25 * float(cal['selected_strength'])
                else: calibrations[arm] = (float(cal['slope']), float(cal['intercept']))
                models[arm] = model
            lookup = {int(row): i for i, row in enumerate(cache['array_rows'])}
            query_rows = panel['Dfit'] + panel['Dcheck']
            raw_T = cache['condition'][[lookup[row] for row in panel['T_rows']]].cuda()
            raw_queries = cache['condition'][[lookup[q['array_row']] for q in query_rows]].cuda()
            with torch.inference_mode():
                center = raw_T.mean(0); scale = raw_T.std(0, unbiased=False).clamp_min(1e-4)
                _, singular, basis = torch.pca_lowrank((raw_T - center) / scale, q=16, center=False, niter=2); basis = basis[:, :8]
                T_conditions, T_gates = three_conditions(raw_T, weight, bias, models, calibrations)
                T_features = feature_rows(T_conditions, T_gates, center, scale, basis)
                feature_center = T_features.mean(0); feature_scale = T_features.std(0, unbiased=False).clamp_min(1e-4)
                conditions, gates = three_conditions(raw_queries, weight, bias, models, calibrations)
                features = ((feature_rows(conditions, gates, center, scale, basis) - feature_center) / feature_scale).cpu().double().numpy()
            conditions = conditions.cpu().numpy(); assert conditions.shape == (600, 3, 1024) and features.shape == (600, 28)
            U.save_array(args.output / 'conditions_Dfit.npy', conditions[:200])
            U.save_array(args.output / 'conditions_Dcheck.npy', conditions[200:])
            U.save_array(args.output / 'features_Dfit.npy', features[:200]); U.save_array(args.output / 'features_Dcheck.npy', features[200:])
            torch.save(dict(center=center.cpu(), scale=scale.cpu(), PCA_basis=basis.cpu(), feature_center=feature_center.cpu(),
                feature_scale=feature_scale.cpu(), T_rows=panel['T_rows'], feature_names='cPC8,dOrdinaryPC8,dRiskPC8,relativeNormOrdinary,relativeNormRisk,deltaCosine,riskGate',
                components=inputs['components'], native_checkpoint=inputs['native_checkpoint']), args.output / 'feature_extractor.pt')
            log(dict(stage='ALL_T_ONLY_FEATURE_STATISTICS_AND600_CONDITIONS_COMPLETE', T_queries=13230, calibration_queries=200,
                check_queries=400, feature_dimensions=28, PCA_components=8, no_Dfit_or_Dcheck_targets_read=True))
            del raw_T, raw_queries, T_conditions, T_gates, T_features, center, scale, basis, feature_center, feature_scale, models, cache, state, weight, bias
            torch.cuda.empty_cache()
            check_conditions = conditions[200:]; fit_conditions = conditions[:200]
            previous = None; prefix = []
        else:
            previous = U.read(args.previous); assert previous['status'] == 'COMPLETE_D09_STAGE80'
            assert previous['readout']['decision']['action'] == 'EXPAND400'
            assert previous['spec']['sha256'] == U.sha(args.spec) and previous['inputs']['sha256'] == U.sha(args.input_completion)
            policy_lock_path = U.checked(previous['policy_lock']); policy_lock = U.read(policy_lock_path)
            policy_lock_sha = previous['policy_lock']['sha256']
            U.checked(policy_lock['model']); U.checked(policy_lock['features'])
            U.checked(policy_lock['shuffled_model'])
            check_conditions = np.load(U.checked(previous['check_conditions']), allow_pickle=False)
            assert previous['check_conditions'] == policy_lock['check_conditions']
            assert check_conditions.shape == (400, 3, 1024)
            prefix = [json.loads(line) for line in U.checked(previous['check_manifest']).read_text().splitlines()]
            assert len(prefix) == 240 and len({(row['query_id'], row['arm']) for row in prefix}) == 240
            for i, row in enumerate(prefix):
                assert U.sha(row['generated_path']) == row['generated_sha256']
                query = panel['Dcheck'][i // 3]
                assert row['query_id'] == query['query_id'] and row['array_row'] == query['array_row']
                assert row['image_seed'] == query['image_seed'] and row['arm'] == U.ARMS[i % 3]
                raw = np.array(check_conditions[i // 3, i % 3:i % 3 + 1], copy=True)
                assert hashlib.sha256(raw.tobytes()).hexdigest() == row['condition_sha256']
            assert policy_lock['locked_before_Dcheck_generation'] is True and len(policy_lock['choices']) == 400
            generated.extend(prefix)
        check(); C.resources(config, training_start=True); active = 'NATIVE_HOST_LOADING'
        loader = args.output / 'loader'; loader.mkdir(); audit = ConvertedWeightAudit(torch, loader); audit.install()
        try: pipe, adapter, projector, prompts, host = load_native(training=False)
        finally: audit.restore()
        loader_proof = audit.complete(); assert host == inputs['expected_host']
        def render(raw, seed, path, save_executor=None):
            check(); assert raw.shape == (1, 1024) and raw.dtype == np.float32
            render_started = time.monotonic()
            with torch.inference_mode():
                condition = torch.from_numpy(np.array(raw, copy=True)).cuda()
                tokens = projector(adapter(condition.to(torch.float16)))
                image, observations = N.sample_observed(pipe, prompts, tokens, seed, torch)
            render_seconds = time.monotonic() - render_started
            proof = dict(condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest())
            def encode_same_bytes():
                encode_started = time.monotonic()
                saved = N.save_image(path, image)
                return saved | dict(CPU_encode_wall_seconds=time.monotonic()-encode_started)
            observations = observations | dict(GPU_render_wall_seconds=render_seconds)
            if save_executor is None: return encode_same_bytes(), observations, proof
            future = save_executor.submit(encode_same_bytes); active_image_saves.append(future)
            return future, observations, proof
        active = 'PARITY'
        parity = inputs['parity_reference']; raw = np.load(U.checked(inputs['parity_condition']), allow_pickle=False)
        saved, observations, proof = render(raw, parity['image_seed'], args.output / 'parity' / 'actual_native.webp')
        for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256'): assert observations[key] == parity[key]
        for key in ('condition_sha256', 'condition_FP16_sha256', 'tokens_sha256'): assert proof[key] == parity[key]
        assert U.sha(parity['generated_path']) == parity['generated_sha256']
        assert saved['generated_pixel_sha256'] == parity['generated_pixel_sha256']
        U.atomic(args.output / 'pixel_parity.json', dict(status='ACTUAL_NATIVE_PIXEL_PARITY_PASSED', image=saved, observations=observations, proof=proof))
        threaded_parity = []
        with ThreadPoolExecutor(max_workers=1) as parity_saver:
            for arm_index, arm in enumerate(U.ARMS):
                old = prefix[arm_index]
                future, observation, proof = render(np.array(check_conditions[0, arm_index:arm_index+1], copy=True),
                    panel['Dcheck'][0]['image_seed'], args.output/'parity'/('threaded_'+arm+'.webp'), parity_saver)
                image = future.result()
                assert image['generated_sha256'] == old['generated_sha256']
                assert image['generated_pixel_sha256'] == old['generated_pixel_sha256']
                for key in ('condition_sha256','condition_FP16_sha256','tokens_sha256'): assert proof[key] == old[key]
                for key in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations'):
                    assert observation[key] == old[key]
                threaded_parity.append(dict(arm=arm,image=image,observations=observation,proof=proof))
        U.atomic(args.output/'threaded_prefix_parity.json',dict(status='ALL3_EXISTING_CHECK_ARM_PIXELS_AND_WEBP_BYTES_MATCH_THREADED_SAVER',
            query_id=panel['Dcheck'][0]['query_id'],rows=threaded_parity,original_prefix_images_regenerated_only_as_engineering_parity=True))
        def generate(queries, bank, partition, start=0, prefix_rows=None):
            records = list(prefix_rows or []); path = args.output / f'{partition}_inference_manifest.jsonl'
            with path.open('x',encoding='utf-8') as stream, ThreadPoolExecutor(max_workers=1) as saver:
                for row in records: stream.write(json.dumps(row)+'\n')
                pending = deque()
                def commit_one():
                    future, base = pending.popleft(); image = future.result(); row = base | image
                    records.append(row); generated.append(row)
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
                try:
                    for i in range(start,len(queries)):
                        query=queries[i];same_noise=None
                        for arm_index,arm in enumerate(U.ARMS):
                            future,observation,proof=render(np.array(bank[i,arm_index:arm_index+1],copy=True),query['image_seed'],
                                args.output/partition/arm/f'{i:03d}_{query["array_row"]:05d}.webp',saver)
                            noise={k:observation[k] for k in ('initial_noise_latents_sha256','initial_UNet_input_sha256','ancestral_noise_observations')}
                            if same_noise is None:same_noise=noise
                            else:assert same_noise==noise
                            base=query|observation|proof|dict(arm=arm,normal_inference_target_access=False,
                                per_query_policy_NFE=4,research_control_arm=True,query_index=i)
                            pending.append((future,base))
                            if len(pending)>=2:commit_one()
                        if (i+1)%20==0:log(dict(stage=partition+'_IMAGES',rendered_queries=i+1,expected=len(queries),
                            committed_control_images=len(records),pending_saves=len(pending),maximum_pending_saves=2))
                finally:
                    # Keep every fully computed image even when a finite deadline is reached.
                    while pending:commit_one()
            assert len(records)==len(queries)*3
            return records,path
        def attach_targets(partition):
            items = U.read(metadata_path)[partition]; paths = []
            for i, item in enumerate(items):
                check()
                with zipfile.ZipFile(item['image_zip']) as archive: payload = archive.read(item['image_member'])
                assert hashlib.sha256(payload).hexdigest() == item['image_sha256'] and len(payload) == int(item['image_bytes'])
                path = args.output / 'targets' / partition / f'{i:03d}_{Path(item["stimulus_id"]).name}'
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open('xb') as stream: stream.write(payload); stream.flush(); os.fsync(stream.fileno())
                paths.append(path)
            return paths
        import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT) == E.DINO_SHA256 == U.S.DINO_SHA256
        dino = None
        def extract(paths, real):
            nonlocal dino
            if dino is None: dino = E.dino_model()
            parts = []
            for start in range(0, len(paths), 32):
                check(); parts.append(E.dino_features(paths[start:start + 32], dino, batch_size=32, real_images=real).float().numpy())
            value = np.concatenate(parts); assert value.shape == (len(paths), 768) and np.isfinite(value).all()
            assert np.allclose(np.linalg.norm(value, axis=1), 1, atol=2e-5); return value
        if stage == 80:
            active = 'DFIT_IMAGES'
            fit_records, fit_manifest = generate(panel['Dfit'], fit_conditions, 'Dfit')
            active = 'DFIT_FEATURES_AND_GAIN_CALIBRATION'
            fit_paths = attach_targets('Dfit')
            target_fit = extract(fit_paths, True)
            fit_generated = extract([Path(row['generated_path']) for row in fit_records], False).reshape(200, 3, 768)
            U.save_array(args.output / 'Dfit_target_features.npy', target_fit); U.save_array(args.output / 'Dfit_generated_features.npy', fit_generated)
            # Normalize exactly as in the fixed scorer before forming each scalar target-cosine label.
            labels = (fit_generated / np.linalg.norm(fit_generated, axis=2, keepdims=True) *
                      (target_fit / np.linalg.norm(target_fit, axis=1, keepdims=True))[:, None]).sum(2).astype(np.float64)
            model = U.fit_router(features[:200], labels); U.atomic(args.output / 'gain_calibrator.json', model)
            choices, predicted = U.predict_router(features[200:], model)
            permutation = U.derangement(200)
            shuffled_model = U.fit_router(features[:200], labels[permutation])
            U.atomic(args.output / 'shuffled_gain_calibrator.json', shuffled_model)
            shuffled_choices, shuffled_predicted = U.predict_router(features[200:], shuffled_model)
            policy_lock = dict(status='D09_ALL400_CHECK_CHOICES_FROZEN_BEFORE_CHECK_GENERATION', choices=choices.tolist(), predicted_utilities=predicted.tolist(),
                fixed_global_arm=model['fixed_global_arm'], model=U.ref(args.output / 'gain_calibrator.json'), features=U.ref(args.output / 'features_Dcheck.npy'),
                shuffled_model=U.ref(args.output / 'shuffled_gain_calibrator.json'), shuffled_choices=shuffled_choices.tolist(),
                shuffled_predicted_utilities=shuffled_predicted.tolist(), Dfit_label_derangement=permutation.tolist(),
                shuffled_control_added_GPU_images=0,
                query_ids=[q['query_id'] for q in panel['Dcheck']], check_conditions=U.ref(args.output / 'conditions_Dcheck.npy'),
                input_completion=U.ref(args.input_completion), spec=U.ref(args.spec), locked_before_Dcheck_generation=True,
                check_images_generated=0, check_target_pixels_or_features_read=False, check_labels_used_for_fitting=False,
                development_only=True, independent_confirmation=False)
            policy_lock_path = args.output / 'policy_lock.json'; U.atomic(policy_lock_path, policy_lock)
            policy_lock_sha = U.sha(policy_lock_path)
            log(dict(stage='ALL400_CHECK_CHOICES_LOCKED', choice_counts={arm: int((choices == i).sum()) for i, arm in enumerate(U.ARMS)},
                fixed_global_arm=U.ARMS[model['fixed_global_arm']], coefficients=58, check_images_generated=0))
            if len(set(choices.tolist())) == 1:
                check()
                U.atomic(args.output / 'completion.json', dict(status='D09_ANALYTICAL_CONSTANT_ACTION_STOP', stage=80, schema=U.SCHEMA,
                    reason='All400 choices equal one frozen arm, so it cannot beat that same arm under identical sampling; no check images needed',
                    policy_lock=U.ref(policy_lock_path), calibration_images=600, check_images=0, parity_images=1,
                    check_target_pixels_or_features_read=False, original_D_resumed=False, source=U.ref(__file__), spec=U.ref(args.spec),
                    inputs=U.ref(args.input_completion), seconds=time.monotonic() - invoked, no_automatic_retry=True))
                return 0
        active = 'DCHECK_IMAGES'
        queries = panel['Dcheck'][:stage]
        check_records, check_manifest = generate(queries, check_conditions, 'Dcheck', start=0 if stage == 80 else 80, prefix_rows=prefix)
        assert U.sha(policy_lock_path) == policy_lock_sha and U.read(policy_lock_path) == policy_lock
        U.checked(policy_lock['model']); U.checked(policy_lock['features'])
        U.checked(policy_lock['shuffled_model'])
        choices = np.asarray(policy_lock['choices'], np.int64); fixed_global = int(policy_lock['fixed_global_arm'])
        assert policy_lock['query_ids'] == [q['query_id'] for q in panel['Dcheck']]
        active = 'DCHECK_FEATURES_AND_FROZEN_READOUT'
        if stage == 80:
            target_paths = attach_targets('Dcheck'); target_check = extract(target_paths, True)
            U.save_array(args.output / 'Dcheck_target_features.npy', target_check)
            target_ref = U.ref(args.output / 'Dcheck_target_features.npy')
            check_generated = extract([Path(row['generated_path']) for row in check_records], False).reshape(80, 3, 768)
        else:
            target_ref = previous['check_target_features']; target_check = np.load(U.checked(target_ref), allow_pickle=False)
            old_features = np.load(U.checked(previous['check_generated_features']), allow_pickle=False); assert old_features.shape == (80, 3, 768)
            new_features = extract([Path(row['generated_path']) for row in check_records[240:]], False).reshape(320, 3, 768)
            check_generated = np.concatenate([old_features, new_features]); assert np.array_equal(check_generated[:80], old_features)
        U.save_array(args.output / 'Dcheck_generated_features.npy', check_generated)
        shuffled_choices = np.asarray(policy_lock['shuffled_choices'], np.int64)
        readout = U.score_features(check_generated, target_check, queries, panel['Dcheck'], choices[:stage], fixed_global, stage, shuffled_choices[:stage])
        U.atomic(args.output / 'readout.json', readout)
        U.write_query_metrics(args.output / 'per_query.csv', check_generated, target_check, queries, panel['Dcheck'], choices[:stage], shuffled_choices[:stage])
        for record in check_records: assert U.sha(record['generated_path']) == record['generated_sha256']
        U.checked(inputs['panel']); U.checked(inputs['native_cache'])
        for source in spec['source_pins']: U.checked(source)
        check()
        result = dict(status=f'COMPLETE_D09_STAGE{stage}', schema=U.SCHEMA, stage=stage, inputs=U.ref(args.input_completion), spec=U.ref(args.spec),
            source=U.ref(__file__), policy_lock=U.ref(policy_lock_path), check_manifest=U.ref(check_manifest), readout=readout,
            check_generated_features=U.ref(args.output / 'Dcheck_generated_features.npy'), check_target_features=target_ref,
            check_conditions=U.ref(args.output / 'conditions_Dcheck.npy') if stage == 80 else previous['check_conditions'],
            previous=U.ref(args.previous) if args.previous else None, host=host, loader_proof=loader_proof,
            zero_module_parity=U.ref(args.output / 'pixel_parity.json'), frozen_Dfit_calibration_queries=200,
            new_research_images=960, parity_images=4, native_policy_UNet_NFE=4,
            CPU_save_overlap_only=True, maximum_pending_image_saves=2, threaded_prefix_parity=U.ref(args.output/'threaded_prefix_parity.json'),
            new_UNet_evaluations=4 * (960 + 4),
            research_control_images_are_not_best_of_N=True, choices_locked_before_check_images=True,
            normal_inference_target_access=False, V_fit_permission_changed=False, original_D_resumed=False,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes,
            seconds=time.monotonic() - invoked, development_only=True, independent_confirmation=False, SOTA_established=False)
        U.atomic(args.output / 'completion.json', result)
        log(dict(stage='PHASE_COMPLETE', status=result['status'], decision=readout['decision'])); return 0
    except BaseException as error:
        U.atomic(args.output / 'failure.json', dict(status='INCOMPLETE_D09_PHASE_NO_AUTOMATIC_RETRY', stage=stage,
            active=active, error=repr(error), traceback=traceback.format_exc(), saved_research_images=len(generated), seconds=time.monotonic() - invoked))
        raise


if __name__ == '__main__': raise SystemExit(main())
