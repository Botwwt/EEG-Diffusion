"""Engineering-only continuation of a terminal finite-time D09 check80 phase.

Reuses all600 calibration images, model, features, all400 locked decisions and
every verified check image. It does not contain a calibration fit/PCA path.
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
import numpy as np
from PIL import Image
import torch
import brainhive_gain_calibration_20261027_v1 as U
import generate_BrainHIVE_neural_clean_fastscreen_20261027_v1 as N

ROOT = Path(__file__).resolve().parent


def validate(binding_path):
    binding = U.read(binding_path)
    assert binding['status'] == 'D09_TERMINAL_TIME_LIMIT_ENGINEERING_CONTINUATION_BOUND'
    for reference in binding['preserved_files'].values(): U.checked(reference)
    assert not any((Path('/proc') / str(pid)).exists() for pid in binding['old_PIDs'])
    failure = U.read(U.checked(binding['failure']))
    assert failure['status'] == 'INCOMPLETE_D09_PHASE_NO_AUTOMATIC_RETRY' and failure['active'] == 'DCHECK_IMAGES'
    assert 'Finite D09 phase work budget' in failure['error']
    receipt = U.read(U.checked(binding['original_GPU_receipt']))
    assert receipt['status'] == 'FAILED_OR_JOB_TIMEOUT_RESUMABLE' and receipt['exit_code'] != 0
    inputs = U.read(U.checked(binding['inputs'])); spec = U.read(U.checked(binding['spec']))
    for source in spec['source_pins']: U.checked(source)
    panel = U.read(U.checked(inputs['panel'])); lock_path = U.checked(binding['policy_lock']); lock = U.read(lock_path)
    U.checked(lock['model']); U.checked(lock['shuffled_model']); U.checked(lock['features'])
    assert lock['locked_before_Dcheck_generation'] is True and lock['check_images_generated'] == 0
    assert lock['check_target_pixels_or_features_read'] is False
    assert lock['query_ids'] == [q['query_id'] for q in panel['Dcheck']] and len(lock['choices']) == 400
    bank = np.load(U.checked(lock['check_conditions']), allow_pickle=False)
    assert bank.dtype == np.float32 and bank.shape == (400, 3, 1024) and np.isfinite(bank).all()
    prefix = [json.loads(line) for line in U.checked(binding['prefix_manifest']).read_text().splitlines() if line]
    assert len(prefix) == binding['reused_check_images'] and 0 < len(prefix) < 240
    assert failure['saved_research_images'] == 600 + len(prefix)
    expected_noise = {}
    for i, row in enumerate(prefix):
        query = panel['Dcheck'][i // 3]
        for field in ('query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'partition', 'image_seed'):
            assert row[field] == query[field]
        assert row['arm'] == U.ARMS[i % 3] and row['query_index'] == i // 3 and row['UNet_NFE'] == 4
        assert row['normal_inference_target_access'] is False
        raw = np.array(bank[i // 3, i % 3:i % 3 + 1], copy=True)
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['condition_sha256']
        assert U.sha(row['generated_path']) == row['generated_sha256']
        with Image.open(row['generated_path']) as image:
            pixel = np.asarray(image.convert('RGB')); assert pixel.shape == (512, 512, 3)
            assert hashlib.sha256(pixel.tobytes()).hexdigest() == row['generated_pixel_sha256']
        noise = {key: row[key] for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations')}
        if query['query_id'] in expected_noise: assert expected_noise[query['query_id']] == noise
        expected_noise[query['query_id']] = noise
        assert lock_path.stat().st_mtime_ns <= Path(row['generated_path']).stat().st_mtime_ns
    return binding, inputs, spec, panel, lock, bank, prefix, expected_noise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binding', type=Path, required=True); parser.add_argument('--native-registration', type=Path)
    parser.add_argument('--output', type=Path); parser.add_argument('--gpu-job-name')
    parser.add_argument('--max-seconds', type=float, default=300); parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    binding, inputs, spec, panel, lock, bank, prefix, expected_noise = validate(args.binding)
    if args.preflight_only:
        assert not torch.cuda.is_initialized()
        print(json.dumps(dict(status='D09_REUSED_PREFIX_AND_LOCKED_MODELS_CPU_VERIFIED', reused_images=len(prefix),
            new_check_images=240-len(prefix), additional_parity_images=1, new_model_fits=0, CUDA_initialized=False))); return 0
    assert args.output and args.gpu_job_name and args.native_registration and not args.output.exists() and 0 < args.max_seconds <= 300
    args.output.mkdir(parents=True); started = time.monotonic(); stopped = [False]; records = list(prefix)
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    try:
        import brainhive_paired_study_contract_20261025_v1 as C
        config = C.read(args.native_registration); C.validate(config); C.setup_runtime(config)
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_component_generator_v1 import load_native
        from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
        from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds, maximum_output_bytes=768 << 20,
            maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
            resource_authorization=config['resource_authorization'])
        wrapper = D.verify_wrapper(document); budget = D.Budget(document)
        def check():
            assert not stopped[0] and time.monotonic() - started < args.max_seconds - 30
            budget.check()
            assert safe_bytes_used(args.output) < (768 << 20) - (16 << 20)
        torch.set_num_threads(2); torch.manual_seed(1729); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True)
        C.resources(config, training_start=True)
        U.atomic(args.output / 'provenance.json', dict(status='D09_SAME_MODEL_SAME_DECISIONS_PREFIX_REUSE', binding=U.ref(args.binding),
            source=U.ref(__file__), wrapper=wrapper, no_Dfit_images_or_model_fits_repeated=True, reused_check_images=len(prefix)))
        loader = args.output / 'loader'; loader.mkdir(); audit = ConvertedWeightAudit(torch, loader); audit.install()
        try: pipe, adapter, projector, prompts, host = load_native(training=False)
        finally: audit.restore()
        loader_proof = audit.complete(); assert host == inputs['expected_host']
        def token_proof(raw):
            condition = torch.from_numpy(np.array(raw, copy=True)).cuda()
            with torch.inference_mode(): tokens = projector(adapter(condition.to(torch.float16)))
            proof = dict(condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest())
            return tokens, proof
        for i, row in enumerate(prefix):
            check(); _, proof = token_proof(np.array(bank[i // 3, i % 3:i % 3 + 1], copy=True))
            for key in proof: assert row[key] == proof[key]
        parity = inputs['parity_reference']; raw = np.load(U.checked(inputs['parity_condition']), allow_pickle=False)
        tokens, proof = token_proof(raw)
        with torch.inference_mode(): image, observation = N.sample_observed(pipe, prompts, tokens, parity['image_seed'], torch)
        saved = N.save_image(args.output / 'parity' / 'actual_native.webp', image)
        for key in proof: assert proof[key] == parity[key]
        for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256'): assert observation[key] == parity[key]
        assert saved['generated_pixel_sha256'] == parity['generated_pixel_sha256']
        U.atomic(args.output / 'pixel_parity.json', dict(status='ACTUAL_NATIVE_PIXEL_PARITY_PASSED', image=saved, observations=observation, proof=proof))
        manifest = args.output / 'Dcheck_inference_manifest.jsonl'
        with manifest.open('x', encoding='utf-8') as stream:
            for row in prefix: stream.write(json.dumps(row) + '\n')
            for i in range(len(prefix), 240):
                check(); query = panel['Dcheck'][i // 3]; arm = U.ARMS[i % 3]
                raw = np.array(bank[i // 3, i % 3:i % 3 + 1], copy=True); tokens, proof = token_proof(raw)
                with torch.inference_mode(): image, observation = N.sample_observed(pipe, prompts, tokens, query['image_seed'], torch)
                noise = {key: observation[key] for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations')}
                if query['query_id'] in expected_noise: assert noise == expected_noise[query['query_id']]
                expected_noise[query['query_id']] = noise
                saved = N.save_image(args.output / 'Dcheck' / arm / f'{i // 3:03d}_{query["array_row"]:05d}.webp', image)
                row = query | saved | observation | proof | dict(arm=arm, normal_inference_target_access=False,
                    per_query_policy_NFE=4, research_control_arm=True, query_index=i // 3)
                records.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
        assert len(records) == 240 and records[:len(prefix)] == prefix
        import image_evaluation as E
        assert U.sha(E.DINO_WEIGHT) == U.S.DINO_SHA256
        dino = E.dino_model()
        def extract(paths, real):
            parts = []
            for start in range(0, len(paths), 32):
                check(); parts.append(E.dino_features(paths[start:start + 32], dino, batch_size=32, real_images=real).float().numpy())
            value = np.concatenate(parts); assert value.shape == (len(paths), 768) and np.isfinite(value).all(); return value
        target_paths = []
        for i, item in enumerate(U.read(U.checked(inputs['target_metadata']))['Dcheck']):
            check()
            with zipfile.ZipFile(item['image_zip']) as archive: payload = archive.read(item['image_member'])
            assert hashlib.sha256(payload).hexdigest() == item['image_sha256']
            path = args.output / 'targets' / 'Dcheck' / f'{i:03d}_{Path(item["stimulus_id"]).name}'
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload); target_paths.append(path)
        target = extract(target_paths, True); generated = extract([Path(row['generated_path']) for row in records], False).reshape(80, 3, 768)
        U.save_array(args.output / 'Dcheck_target_features.npy', target); U.save_array(args.output / 'Dcheck_generated_features.npy', generated)
        choices = np.asarray(lock['choices'], np.int64); shuffled = np.asarray(lock['shuffled_choices'], np.int64)
        readout = U.score_features(generated, target, panel['Dcheck'][:80], panel['Dcheck'], choices[:80], int(lock['fixed_global_arm']), 80, shuffled[:80])
        U.atomic(args.output / 'readout.json', readout)
        U.write_query_metrics(args.output / 'per_query.csv', generated, target, panel['Dcheck'][:80], panel['Dcheck'], choices[:80], shuffled[:80])
        for item in binding['preserved_files'].values(): U.checked(item)
        for item in spec['source_pins']: U.checked(item)
        check()
        result = dict(status='COMPLETE_D09_STAGE80', schema=U.SCHEMA, stage=80, inputs=binding['inputs'], spec=binding['spec'],
            source=U.ref(__file__), recovery_binding=U.ref(args.binding), policy_lock=binding['policy_lock'], check_manifest=U.ref(manifest),
            check_generated_features=U.ref(args.output / 'Dcheck_generated_features.npy'), check_target_features=U.ref(args.output / 'Dcheck_target_features.npy'),
            check_conditions=lock['check_conditions'], readout=readout, host=host, loader_proof=loader_proof,
            reused_calibration_images=600, reused_check_images=len(prefix), new_check_images=240-len(prefix), parity_images=1,
            new_UNet_evaluations=4*(241-len(prefix)), no_Dfit_fit_or_images_repeated=True, choices_locked_before_check_images=True,
            normal_inference_target_access=False, V_fit_permission_changed=False, original_D_resumed=False,
            seconds=time.monotonic()-started, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes,
            independent_confirmation=False, SOTA_established=False)
        U.atomic(args.output / 'completion.json', result); print(json.dumps(dict(status=result['status'], decision=readout['decision'], seconds=result['seconds']))); return 0
    except BaseException as error:
        U.atomic(args.output / 'failure.json', dict(status='INCOMPLETE_D09_CONTINUATION_NO_AUTOMATIC_RETRY', error=repr(error),
            traceback=traceback.format_exc(), retained_check_images=len(records), seconds=time.monotonic()-started))
        raise


if __name__ == '__main__': raise SystemExit(main())
