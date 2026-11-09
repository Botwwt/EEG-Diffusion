"""D16 separate frozen80 response-head image diagnosis; original proxy gate stays failed."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_response_image_diagnostic_20261027_v1 as V
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

ROOT = Path(__file__).resolve().parent
NOISE_KEYS = ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-completion', 'native-registration', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--diagnostic-spec',type=Path,required=True)
    parser.add_argument('--stage', type=int, choices=(80,), required=True)
    parser.add_argument('--gpu-job-name', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args(); invoked = time.monotonic()
    assert args.input_completion and args.native_registration and args.output
    assert args.stage==80 and args.previous is None and args.max_seconds==480
    assert not args.output.exists(); args.output.mkdir(parents=True)
    stopped = [False]; active = 'INPUTS'; new_saved = []; pending_futures = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    def log(value):
        value = value | dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(value), flush=True)
    try:
        prepared=U.read(args.input_completion)
        assert prepared['status']=='D16_FOUR_MATCHED_HEADS_FULL_T10_AND_ALL8200_DFIT_COMPLETE' and prepared['prediction_gate_passed'] is False
        fit_spec=prepared['spec'];spec=U.read(args.diagnostic_spec)
        assert spec['schema']==V.SCHEMA and spec['original_prediction_gate_passed'] is False and spec['automatic_expansion_allowed'] is False
        for reference in spec['source_pins']:U.checked(reference)
        d09=U.read(U.checked(spec['D09_completion']));inputs=U.read(U.checked(d09['inputs']));panel=U.read(U.checked(inputs['panel']))
        prepared=prepared|dict(D09_completion=spec['D09_completion'],D09_inputs=d09['inputs'],panel=inputs['panel'],
            baseline_generated_features=d09['check_generated_features'],target_features=d09['check_target_features'],baseline_manifest=d09['check_manifest'])
        baseline = np.load(U.checked(prepared['baseline_generated_features']), allow_pickle=False)
        targets = np.load(U.checked(prepared['target_features']), allow_pickle=False)
        baseline_rows = [json.loads(line) for line in U.checked(prepared['baseline_manifest']).read_text().splitlines()]
        assert len(baseline_rows) == 1200
        noise_by_id = {}
        for i, query in enumerate(panel['Dcheck']):
            row = baseline_rows[3*i]
            for key, value in query.items(): assert row[key] == value
            assert row['arm'] == 'parent' and row['UNet_NFE'] == 4
            noise_by_id[query['query_id']] = {k: row[k] for k in NOISE_KEYS}
        prefix=[];old_features=None
        import brainhive_paired_study_contract_20261025_v1 as C
        config = C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
            maximum_output_bytes=1024 << 20, maximum_extra_CPU_bytes=10 << 30,
            maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
            resource_authorization=config['resource_authorization'])
        wrapper = D.verify_wrapper(document); budget = D.Budget(document)
        def check():
            assert not stopped[0] and time.monotonic()-invoked < args.max_seconds-30, 'D16 finite work budget; retain prefix'
            budget.check()
            if all(future.done() for future in pending_futures):
                assert safe_bytes_used(args.output) < (1024 << 20)-(16 << 20)
        torch.set_num_threads(2); torch.manual_seed(1729); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        C.resources(config,training_start=True);active='ALL400_NATIVE_CONDITIONS_LOCK_BEFORE_IMAGES'
        bank,lock=V.build_conditions(prepared,spec,inputs,panel,args.output)
        lock_path=args.output/'all400_condition_lock.json';lock_sha=U.sha(lock_path)
        prepared=prepared|dict(lock=U.ref(lock_path),spec=U.ref(args.diagnostic_spec))
        torch.cuda.empty_cache()
        U.atomic(args.output/'provenance.json', dict(schema=V.SCHEMA, inputs=U.ref(args.input_completion),
            spec=prepared['spec'], condition_lock=prepared['lock'], source=U.ref(__file__), wrapper=wrapper,
            native_inference_NFE=4, new_training=0, normal_inference_target_access=False, original_D_resumed=False))
        check(); C.resources(config, training_start=True); active = 'NATIVE_HOST_LOADING'
        loader = args.output/'loader'; loader.mkdir(); audit = ConvertedWeightAudit(torch, loader); audit.install()
        try: pipe, adapter, projector, prompts, host = load_native(training=False)
        finally: audit.restore()
        loader_proof = audit.complete(); assert host == inputs['expected_host']
        def render(raw, seed, path, executor=None):
            check(); assert raw.shape == (1, 1024) and raw.dtype == np.float32
            started = time.monotonic()
            with torch.inference_mode():
                condition = torch.from_numpy(np.array(raw, copy=True)).cuda()
                token = projector(adapter(condition.to(torch.float16)))
                image, observations = N.sample_observed(pipe, prompts, token, seed, torch)
            observations = observations | dict(GPU_render_wall_seconds=time.monotonic()-started)
            proof = dict(condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                tokens_sha256=hashlib.sha256(token.cpu().contiguous().numpy().tobytes()).hexdigest())
            def encode():
                clock = time.monotonic(); saved = N.save_image(path, image)
                return saved | dict(CPU_encode_wall_seconds=time.monotonic()-clock)
            if executor is None: return encode(), observations, proof
            future = executor.submit(encode); pending_futures.append(future)
            return future, observations, proof
        active = 'T_AND_FULL_DC_NATIVE_PIXEL_PARITIES'; parities = []
        raw_parent = np.load(U.checked(d09['check_conditions']), allow_pickle=False)
        for name, raw, expected in [('T_native', np.load(U.checked(inputs['parity_condition']), allow_pickle=False), inputs['parity_reference']),
                                     ('Dcheck_full_parent', raw_parent[0, 0:1], baseline_rows[0])]:
            saved, observation, proof = render(np.array(raw, copy=True), expected['image_seed'], args.output/'parity'/(name+'.webp'))
            for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256'): assert observation[key] == expected[key]
            for key in proof: assert proof[key] == expected[key]
            for key in ('generated_sha256', 'generated_pixel_sha256'): assert saved[key] == expected[key]
            parities.append(dict(name=name, image=saved, observations=observation, proof=proof))
        U.atomic(args.output/'pixel_parity.json', dict(status='BOTH_T_NATIVE_AND_FULL_DC_PARENT_EXACT_BYTES_PIXELS_PASSED', rows=parities))
        active = 'FOUR_FROZEN_RESPONSE_HEAD_IMAGE_DIAGNOSTIC'; records = list(prefix); manifest = args.output/'inference_manifest.jsonl'
        with manifest.open('x') as stream, ThreadPoolExecutor(max_workers=1) as executor:
            for row in records: stream.write(json.dumps(row)+'\n')
            pending = deque()
            def commit():
                future, base = pending.popleft(); row = base | future.result()
                records.append(row); new_saved.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            try:
                for i in range(80):
                    query = panel['Dcheck'][i]
                    for j, arm in enumerate(V.ARMS):
                        future, observed, proof = render(np.array(bank[i, j:j+1], copy=True), query['image_seed'],
                            args.output/arm/f'{i:03d}_{query["array_row"]:05d}.webp', executor)
                        assert {k: observed[k] for k in NOISE_KEYS} == noise_by_id[query['query_id']]
                        row = query | observed | proof | dict(arm=arm, query_index=i,
                            response_source_query_index=(i if arm=='paired_responses' else lock['shuffled_response_source_query_permutation'][i] if arm=='shuffled_responses' else None),
                            cached_encoder_features_used_for_research=True,normal_inference_additional_native_head_evaluations=(4 if arm in ('paired_responses','shuffled_responses') else 0),
                            condition_head_parameters=527360,condition_inference_is_deterministic=True,diagnostic_only=True,original_prediction_gate_passed=False,
                            normal_inference_target_access=False, research_intervention_arm=True, per_query_policy_NFE=4)
                        pending.append((future, row))
                        if len(pending) >= 2: commit()
                    if (i+1)%20 == 0: log(dict(stage=active, rendered_queries=i+1, expected=args.stage, committed_rows=len(records), pending_saves=len(pending)))
            finally:
                while pending: commit()
        assert len(records) == 4*args.stage and records[:len(prefix)] == prefix
        assert U.sha(lock_path) == lock_sha
        active = 'DINO_NEW_IMAGES_AND_FROZEN_READOUT'
        import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT) == E.DINO_SHA256 == U.S.DINO_SHA256
        dino = E.dino_model(); parts = []
        new_paths = [Path(row['generated_path']) for row in new_saved]
        for lo in range(0, len(new_paths), 32):
            check(); parts.append(E.dino_features(new_paths[lo:lo+32], dino, batch_size=32, real_images=False).float().numpy())
        generated = np.concatenate(parts).reshape(len(new_saved)//4, 4, 768)
        if old_features is not None: generated = np.concatenate([old_features, generated])
        assert generated.shape == (args.stage, 4, 768) and np.isfinite(generated).all()
        if old_features is not None: assert np.array_equal(generated[:80], old_features)
        U.save_array(args.output/'generated_features.npy', generated)
        readout = V.score(generated, targets, baseline, panel['Dcheck'][:args.stage], panel['Dcheck'], args.stage)
        U.atomic(args.output/'readout.json', readout)
        per_query = {}
        for j, arm in enumerate(U.ARMS): per_query[arm] = U.S.metric_arrays(baseline[:args.stage, j], targets, panel['Dcheck'][:args.stage], panel['Dcheck'])
        for j, arm in enumerate(V.ARMS): per_query[arm] = U.S.metric_arrays(generated[:, j], targets, panel['Dcheck'][:args.stage], panel['Dcheck'])
        fields = ['query_id', 'concept_id', 'arm'] + list(next(iter(per_query.values())))
        with (args.output/'per_query.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for i, query in enumerate(panel['Dcheck'][:args.stage]):
                for arm, metrics in per_query.items(): writer.writerow(dict(query_id=query['query_id'], concept_id=query['concept_id'], arm=arm,
                    **{key: float(values[i]) for key, values in metrics.items()}))
        for row in records: assert U.sha(row['generated_path']) == row['generated_sha256']
        for ref in spec['source_pins']: U.checked(ref)
        U.checked(prepared['lock']); U.checked(prepared['baseline_generated_features']); U.checked(prepared['target_features']); check()
        complete = dict(status='COMPLETE_D16_FIXED80_IMAGE_DIAGNOSTIC', schema=V.SCHEMA, stage=args.stage,
            inputs=U.ref(args.input_completion), spec=prepared['spec'], condition_lock=prepared['lock'],
            source=U.ref(__file__), manifest=U.ref(manifest), generated_features=U.ref(args.output/'generated_features.npy'),
            previous=U.ref(args.previous) if args.previous else None, readout=readout, host=host, loader_proof=loader_proof,
            pixel_parity=U.ref(args.output/'pixel_parity.json'), new_research_images=len(new_saved), reused_research_images=len(prefix),
            engineering_parity_images=2, new_UNet_evaluations=4*(len(new_saved)+2), normal_inference_UNet_NFE=4,
            original_D_resumed=False, EEG_or_generator_weights_updated=False, normal_inference_target_access=False,
            fitted_condition_head_parameters=prepared['parameters_per_head'],old_ATMS_parameters_per_model=3200413,
            diagnostic_only=True,original_prediction_gate_passed=False,original_candidate_admitted=False,automatic_expansion_allowed=False,
            fit_spec=fit_spec,all4_frozen_heads_reported=True,V_fit_permission_changed=False,independent_confirmation=False,
            SOTA_established=False, no_automatic_retry_or_sweep=True, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
            peak_GPU_bytes=budget.peak_GPU_bytes, seconds=time.monotonic()-invoked)
        U.atomic(args.output/'completion.json', complete)
        log(dict(stage='COMPLETE', decision=readout['decision'])); return 0
    except BaseException as error:
        U.atomic(args.output/'failure.json', dict(status='D16_INCOMPLETE_NO_AUTOMATIC_RETRY', active=active,
            stage=args.stage, newly_saved_images=len(new_saved), seconds=time.monotonic()-invoked,
            error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': raise SystemExit(main())
