"""D11: three privileged visual-teacher interventions, never an EEG inference method."""
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
import brainhive_teacher_reference_20261027_v1 as V
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

ROOT = Path(__file__).resolve().parent
NOISE_KEYS = ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input-completion', 'native-registration', 'previous', 'output'):
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--stage', type=int, choices=(80, 400), required=True)
    parser.add_argument('--gpu-job-name', required=True)
    parser.add_argument('--max-seconds', type=float, required=True)
    args = parser.parse_args(); invoked = time.monotonic()
    assert args.input_completion and args.native_registration and args.output
    assert args.stage == 80 and not args.previous and 0 < args.max_seconds <= 420
    assert not args.output.exists(); args.output.mkdir(parents=True)
    stopped = [False]; active = 'INPUTS'; new_saved = []; pending_futures = []
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    def log(value):
        value = value | dict(seconds=time.monotonic()-invoked)
        with (args.output/'progress.jsonl').open('a') as f:
            f.write(json.dumps(value)+'\n'); f.flush(); os.fsync(f.fileno())
        print(json.dumps(value), flush=True)
    try:
        prepared = U.read(args.input_completion); assert prepared['status'] == 'D11_TEACHER_SOURCE_AND_ALL400_CONDITIONS_CPU_COMPLETE'
        spec = U.read(U.checked(prepared['spec'])); assert spec['schema'] == V.SCHEMA and spec['diagnostic_only'] is True
        for reference in spec['source_pins']: U.checked(reference)
        d09 = U.read(U.checked(prepared['D09_completion'])); assert d09['status'] == 'COMPLETE_D09_STAGE400'
        inputs = U.read(U.checked(prepared['D09_inputs'])); panel = U.read(U.checked(prepared['panel']))
        lock_path = U.checked(prepared['lock']); lock = U.read(lock_path); lock_sha = U.sha(lock_path)
        U.checked(lock['source_F_cache'])
        assert prepared['F_cache_numeric_replay_passed'] is True
        assert U.read(U.checked(prepared['F_source_replay']))['status'] == 'ALL400_CPU_F_REPLAY_NUMERICALLY_MATCHES_SAVED_GPU_BF16'
        bank = np.load(U.checked(lock['conditions']), allow_pickle=False)
        assert bank.shape == (400, 3, 1024) and bank.dtype == np.float32 and np.isfinite(bank).all()
        assert lock['query_ids'] == [q['query_id'] for q in panel['Dcheck']]
        assert lock['diagnostic_only'] is True and lock['oracle_uses_query_visual_supervision'] is True
        assert lock['normal_inference_supported'] is False and lock['automatic_candidate_promotion'] is False
        assert lock['query_DINO_metrics_used_for_condition_selection'] is False
        baseline = np.load(U.checked(prepared['baseline_generated_features']), allow_pickle=False)
        targets = np.load(U.checked(prepared['target_features']), allow_pickle=False)
        d10_features = np.load(U.checked(prepared['d10_generated_features']), allow_pickle=False)
        baseline_rows = [json.loads(line) for line in U.checked(prepared['baseline_manifest']).read_text().splitlines()]
        assert len(baseline_rows) == 1200
        noise_by_id = {}
        for i, query in enumerate(panel['Dcheck']):
            row = baseline_rows[3*i]
            for key, value in query.items(): assert row[key] == value
            assert row['arm'] == 'parent' and row['UNet_NFE'] == 4
            noise_by_id[query['query_id']] = {k: row[k] for k in NOISE_KEYS}
        prefix = []; old_features = None
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
            assert not stopped[0] and time.monotonic()-invoked < args.max_seconds-30, 'D11 finite diagnostic work budget; retain prefix'
            budget.check()
            if all(future.done() for future in pending_futures):
                assert safe_bytes_used(args.output) < (1024 << 20)-(16 << 20)
        torch.set_num_threads(2); torch.manual_seed(1729); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        U.atomic(args.output/'provenance.json', dict(schema=V.SCHEMA, inputs=U.ref(args.input_completion),
            spec=prepared['spec'], condition_lock=prepared['lock'], source=U.ref(__file__), wrapper=wrapper,
            per_diagnostic_image_NFE=4, new_training=0, uses_Dcheck_visual_supervision=True, diagnostic_only=True, normal_inference_supported=False, original_D_resumed=False))
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
        active = 'TEACHER_DIAGNOSTIC_IMAGES'; records = list(prefix); manifest = args.output/'inference_manifest.jsonl'
        with manifest.open('x') as stream, ThreadPoolExecutor(max_workers=1) as executor:
            for row in records: stream.write(json.dumps(row)+'\n')
            pending = deque()
            def commit():
                future, base = pending.popleft(); row = base | future.result()
                records.append(row); new_saved.append(row)
                stream.write(json.dumps(row, allow_nan=False)+'\n'); stream.flush(); os.fsync(stream.fileno())
            try:
                for i in range(0 if args.stage == 80 else 80, args.stage):
                    query = panel['Dcheck'][i]
                    for j, arm in enumerate(V.ARMS):
                        future, observed, proof = render(np.array(bank[i, j:j+1], copy=True), query['image_seed'],
                            args.output/arm/f'{i:03d}_{query["array_row"]:05d}.webp', executor)
                        assert {k: observed[k] for k in NOISE_KEYS} == noise_by_id[query['query_id']]
                        source_i = lock['source_query_indices'][i][j]
                        row = query | observed | proof | dict(arm=arm, query_index=i,
                            source_F_array_row=lock['source_array_rows'][i][j],
                            source_F_stimulus_id=panel['Dcheck'][source_i]['stimulus_id'],
                            uses_Dcheck_visual_supervision=True, diagnostic_only=True, normal_inference_supported=False,
                            research_intervention_arm=True, per_diagnostic_image_NFE=4)
                        pending.append((future, row))
                        if len(pending) >= 2: commit()
                    if (i+1)%20 == 0: log(dict(stage=active, rendered_queries=i+1, expected=args.stage, committed_rows=len(records), pending_saves=len(pending)))
            finally:
                while pending: commit()
        assert len(records) == 3*args.stage and records[:len(prefix)] == prefix
        assert U.sha(lock_path) == lock_sha
        active = 'DINO_NEW_IMAGES_AND_FROZEN_READOUT'
        import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT) == E.DINO_SHA256 == U.S.DINO_SHA256
        dino = E.dino_model(); parts = []
        new_paths = [Path(row['generated_path']) for row in new_saved]
        for lo in range(0, len(new_paths), 32):
            check(); parts.append(E.dino_features(new_paths[lo:lo+32], dino, batch_size=32, real_images=False).float().numpy())
        generated = np.concatenate(parts).reshape(len(new_saved)//3, 3, 768)
        if old_features is not None: generated = np.concatenate([old_features, generated])
        assert generated.shape == (args.stage, 3, 768) and np.isfinite(generated).all()
        if old_features is not None: assert np.array_equal(generated[:80], old_features)
        U.save_array(args.output/'generated_features.npy', generated)
        readout = V.score(generated, targets, baseline, panel['Dcheck'][:args.stage], panel['Dcheck'], args.stage, d10_features)
        U.atomic(args.output/'readout.json', readout)
        if args.stage == 400:
            for arm in U.ARMS: assert readout['means'][arm] == d09['readout']['means'][arm]
        per_query = {}
        for j, arm in enumerate(U.ARMS): per_query[arm] = U.S.metric_arrays(baseline[:args.stage, j], targets, panel['Dcheck'][:args.stage], panel['Dcheck'])
        for j, arm in enumerate(V.ARMS): per_query[arm] = U.S.metric_arrays(generated[:, j], targets, panel['Dcheck'][:args.stage], panel['Dcheck'])
        for j, arm in enumerate(('D10_support','D10_shuffled_EEG_support')):
            per_query[arm] = U.S.metric_arrays(d10_features[:,j],targets,panel['Dcheck'][:80],panel['Dcheck'])
        fields = ['query_id', 'concept_id', 'arm'] + list(next(iter(per_query.values())))
        with (args.output/'per_query.csv').open('x', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            for i, query in enumerate(panel['Dcheck'][:args.stage]):
                for arm, metrics in per_query.items(): writer.writerow(dict(query_id=query['query_id'], concept_id=query['concept_id'], arm=arm,
                    **{key: float(values[i]) for key, values in metrics.items()}))
        for row in records: assert U.sha(row['generated_path']) == row['generated_sha256']
        for ref in spec['source_pins']: U.checked(ref)
        U.checked(prepared['lock']); U.checked(prepared['baseline_generated_features']); U.checked(prepared['target_features']); check()
        complete = dict(status='COMPLETE_D11_TEACHER_DIAGNOSTIC80', schema=V.SCHEMA, stage=args.stage,
            inputs=U.ref(args.input_completion), spec=prepared['spec'], condition_lock=prepared['lock'],
            source=U.ref(__file__), manifest=U.ref(manifest), generated_features=U.ref(args.output/'generated_features.npy'),
            previous=U.ref(args.previous) if args.previous else None, readout=readout, host=host, loader_proof=loader_proof,
            pixel_parity=U.ref(args.output/'pixel_parity.json'), new_research_images=len(new_saved), reused_research_images=len(prefix),
            engineering_parity_images=2, new_UNet_evaluations=4*(len(new_saved)+2), per_diagnostic_image_NFE=4,
            original_D_resumed=False, EEG_or_generator_weights_updated=False, uses_Dcheck_visual_supervision=True, diagnostic_only=True, normal_inference_supported=False,
            teacher_bank_partition='Dcheck_diagnostic_only', automatic_candidate_promotion=False, EEG_efficacy_or_method_gain=False, V_fit_permission_changed=False, independent_confirmation=False,
            SOTA_established=False, no_automatic_retry_or_sweep=True, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
            peak_GPU_bytes=budget.peak_GPU_bytes, seconds=time.monotonic()-invoked)
        U.atomic(args.output/'completion.json', complete)
        log(dict(stage='COMPLETE', decision=readout['decision'])); return 0
    except BaseException as error:
        U.atomic(args.output/'failure.json', dict(status='D11_DIAGNOSTIC_INCOMPLETE_NO_AUTOMATIC_RETRY', active=active,
            stage=args.stage, newly_saved_images=len(new_saved), seconds=time.monotonic()-invoked,
            error=repr(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__': raise SystemExit(main())
