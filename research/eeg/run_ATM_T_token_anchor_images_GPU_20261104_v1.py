"""Native P hook applies fixed T anchors to current EEG, with no extra P or UNet."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse, fcntl, hashlib, inspect, io, json, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
import atm_T_token_anchor_image_contract_20261104_v1 as C
ap = argparse.ArgumentParser(); ap.add_argument('--policy', required=True)
ap.add_argument('--mode', choices=('generate', 'encode'), required=True)
ap.add_argument('--start', type=int, default=0); ap.add_argument('--stage', type=int, default=400); a = ap.parse_args()
c = C.setup(a.policy); K, q, out, parent = c['K'], c['q'], c['out'], c['parent']
started = time.monotonic(); rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
before = C.resources(c, starting=True); assert K.ref(__file__) == q['sources']['GPU']
suffix = a.mode + ('_' + str(a.start) if a.mode == 'generate' else '_' + str(a.stage))
job = K.read(c['repo'] / 'audit' / (q['job_prefix'] + '_' + suffix + '_job.json'))
assert job['pid'] == os.getpid() and job['status'] == 'RUNNING'
lock = (c['repo'] / 'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == parent['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
import torch
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(parent['maximum_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory)
from atm_native_training_runtime_20261103_v1 import tensor_sha
from brainhive_native_direction_contract_20261103_v1 import tensor_digest

def used(): return c['up']['cv']['ctx']['base']['ctx']['C'].allocated(q['project_root'])

def guard():
    assert time.monotonic() - started < 850 and used() < q['maximum_output_bytes']
    fs = os.statvfs(q['project_root'])
    assert fs.f_bavail * fs.f_frsize >= before['required_remaining_bytes'] + q['maximum_output_bytes'] - used()
    assert torch.cuda.max_memory_allocated() < parent['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < parent['maximum_extra_CPU_bytes']

try:
    if a.mode == 'generate':
        assert a.start in (0, 80, 160, 240, 320)
        from atm_native_training_runtime_20261103_v1 import load_host
        prepared = K.read(out / 'prepared.json')
        values = np.load(K.checked(prepared['conditions']), allow_pickle=False)
        native = np.load(K.checked(prepared['native']), allow_pickle=False)[:, 1]
        bank = np.load(K.checked(prepared['anchors']), allow_pickle=False)
        means = np.load(K.checked(prepared['statistics']), allow_pickle=False)['token_means'].astype(np.float32)
        assert values.shape == (400, 1024) and native.shape == (400, 4, 2048)
        assert np.array_equal(C.fixed_tokens(native, means), bank)
        loader_dir = out / ('loader_' + str(a.start)); loader_dir.mkdir()
        pipe, projection, prompt, pooled, timeids, diffusion, proof = load_host(c['cv']['bq'], loader_dir)
        from diffusers.models.attention_processor import IPAdapterAttnProcessor2_0
        assert hashlib.sha256(inspect.getsource(IPAdapterAttnProcessor2_0).encode()).hexdigest() == q['installed_IP_class_source_sha256']
        processors = [p for p in pipe.unet.attn_processors.values() if hasattr(p, 'to_k_ip')]
        assert len(processors) == 70 and all(type(p).__name__ == 'IPAdapterAttnProcessor2_0' and list(p.num_tokens) == [4] and p.scale == [1.0] for p in processors)
        frozen = [tensor_digest(m, half=True) for m in (pipe.unet, pipe.vae, projection)]
        mu = torch.tensor(means.copy(), device='cuda')
        import diffusers.schedulers.scheduling_euler_ancestral_discrete as EA
        old_noise, old_prepare, capture, state = EA.randn_tensor, pipe.prepare_latents, {}, None
        def cn(*args, **kwargs):
            value = old_noise(*args, **kwargs); capture['ancestral'].append(tensor_sha(value)); return value
        def ci(*args, **kwargs):
            value = old_prepare(*args, **kwargs); capture['initial'].append(tensor_sha(value)); return value
        def transform(output, j):
            value = output.reshape(1, 4, 2048).float()
            return (value - value.mean(1, keepdim=True) + mu[j][None, None]).half().reshape(output.shape)
        def hook(module, args, output):
            if state is None: return None
            assert torch.equal(output.reshape(1, 4, 2048), state['base'])
            actual = output if state['identity'] else transform(output, state['index'])
            assert torch.equal(actual.reshape(1, 4, 2048), state['expected'])
            state['calls'] += 1; state['actual_SHA'] = tensor_sha(actual.reshape(1, 4, 2048))
            return actual
        handle = projection.register_forward_hook(hook)
        EA.randn_tensor, pipe.prepare_latents = cn, ci
        def render(i):
            value = torch.tensor(values[i][None].copy(), device='cuda')
            with torch.inference_mode():
                image = pipe(prompt_embeds=prompt, pooled_prompt_embeds=pooled, height=512, width=512,
                    ip_adapter_image_embeds=[value[:, None].half()], num_inference_steps=4, guidance_scale=0.,
                    generator=torch.Generator(device='cuda').manual_seed(c['queries'][i]['image_seed'])).images[0]
            return value, image
        try:
            if a.start == 0:
                with torch.inference_mode():
                    for i in range(400):
                        if i % 32 == 0: guard()
                        actual = projection(torch.tensor(values[i][None].copy(), device='cuda').half())
                        assert np.array_equal(actual.reshape(4, 2048).cpu().numpy(), native[i])
                        for j in range(3):
                            assert np.array_equal(transform(actual, j).reshape(4, 2048).cpu().numpy(), bank[i, j]), (i, j, 'native hook token replay')
                reference = c['baseline']['nce_continued'][0]
                capture = {'initial': [], 'ancestral': []}; value, image = render(0)
                assert hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == reference['pixels_sha256'] and capture == reference['noise']
                image.save(out / 'nce_q000_native_replay.webp', format='WEBP', lossless=True, quality=100)
                state = dict(base=torch.tensor(native[0][None].copy(), device='cuda'), expected=torch.tensor(native[0][None].copy(), device='cuda'), identity=True, calls=0)
                capture = {'initial': [], 'ancestral': []}; value, image = render(0)
                assert state['calls'] == 4 and hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest() == reference['pixels_sha256'] and capture == reference['noise']
                image.save(out / 'nce_q000_identity_hook_replay.webp', format='WEBP', lossless=True, quality=100); state = None
                K.write(out / 'actual_input_profile.json', dict(status='PASS_ALL400_NATIVE_P_ALL1200_ANCHOR_TOKENS_AND_TWO_PIXEL_REPLAYS',
                    prepared=K.ref(out / 'prepared.json'), all400_native_P_BIT_EXACT=True,
                    all1200_inplace_hook_tokens_BIT_EXACT=True, native_and_identity_hook_pixels_noise_exact=True,
                    inplace_transform_uses_actual_current_P_output=True, extra_native_P_in_normal_generation=0))
            else:
                assert K.read(out / 'actual_input_profile.json')['status'] == 'PASS_ALL400_NATIVE_P_ALL1200_ANCHOR_TOKENS_AND_TWO_PIXEL_REPLAYS'
            for arm in C.MODELS:
                directory = out / 'images' / arm; directory.mkdir(parents=True, exist_ok=True)
                journal = directory / 'images.jsonl'
                rows = [json.loads(s) for s in journal.read_text().splitlines()] if journal.exists() else []
                assert [r['query_index'] for r in rows] == list(range(a.start))
            for i in range(a.start, a.start + 80):
                guard()
                if i % 16 == 0: C.resources(c)
                for j, arm in enumerate(C.MODELS):
                    state = dict(base=torch.tensor(native[i][None].copy(), device='cuda'), expected=torch.tensor(bank[i, j][None].copy(), device='cuda'), index=j, identity=False, calls=0)
                    capture = {'initial': [], 'ancestral': []}; value, image = render(i)
                    assert state['calls'] == 4 and capture == c['baseline']['direct'][i]['noise']
                    token_SHA = state['actual_SHA']; state = None
                    pixels = np.asarray(image.convert('RGB')).copy(); assert pixels.shape == (512, 512, 3)
                    data = io.BytesIO(); image.save(data, format='WEBP', lossless=True, quality=100)
                    image_path = out / 'images' / arm / f'q{i:03d}.webp'; assert not image_path.exists()
                    guard(); assert used() + len(data.getvalue()) + 65536 < q['maximum_output_bytes']; image_path.write_bytes(data.getvalue())
                    row = dict(c['queries'][i], query_index=i, arm=arm, image=K.ref(image_path), pixels_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(), noise=capture,
                        condition_FP32_sha256=tensor_sha(value), condition_FP16_sha256=tensor_sha(value.half()), tokens_FP16_sha256=token_SHA,
                        native_tokens_FP16_sha256=hashlib.sha256(native[i].tobytes()).hexdigest(), anchor_name=C.ANCHORS[j],
                        anchor_mean_FP32_sha256=hashlib.sha256(means[j].tobytes()).hexdigest(),
                        normal_inference_only_current_EEG_plus_fixed_T_statistics=True,
                        query_image_read_at_inference=False, query_concept_used_at_inference=False,
                        UNet_NFE=4, prior_NFE=0, EEG_encoder_forward_cost=1, extra_native_P_forward_cost=0,
                        projection_calls_per_UNet_trajectory=4, inplace_transform_calls=4,
                        generation_used_frozen_EEG_condition_cache=True,
                        prepared=K.ref(out / 'prepared.json'), input_profile=K.ref(out / 'actual_input_profile.json'))
                    with (out / 'images' / arm / 'images.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
                K.write(out / 'generation_progress.json', dict(block_start=a.start, completed_queries_per_new_arm=i + 1))
        finally:
            handle.remove(); EA.randn_tensor, pipe.prepare_latents = old_noise, old_prepare
        assert [tensor_digest(m, half=True) for m in (pipe.unet, pipe.vae, projection)] == frozen
        result = dict(status='COMPLETE_240_FIXED_T_TOKEN_ANCHOR_IMAGES_BLOCK', start=a.start,
            completed_queries_per_arm=a.start + 80, new_images=240, all_frozen_host_weights_unchanged=True, loader=proof)
    else:
        assert a.stage == 400
        import image_evaluation as E
        rows = K.read(K.checked(K.read(out / 'image_audit400.json')['ordered_records']))
        base = np.load(K.checked(q['baseline_features']), allow_pickle=False); assert base.shape == (400, 8, 768)
        model = E.dino_model(); assert E.DINO_SHA256 == q['DINO_weight_sha256']
        features = np.empty((400, len(C.ARMS), 768), np.float32); features[:, :len(C.BASES)] = base
        replay = E.dino_features([K.checked(c['baseline']['direct'][0]['image'])], model, batch_size=1, real_images=False).float().numpy()[0]
        error = float(abs(replay - base[0, 0]).max()); assert error < 3e-6
        new = [r for r in rows if r['arm'] in C.MODELS]; assert len(new) == 1200
        for lo in range(0, len(new), 32):
            guard(); C.resources(c); batch = new[lo:lo + 32]
            encoded = E.dino_features([K.checked(r['image']) for r in batch], model, batch_size=32, real_images=False).float().numpy()
            for row, vector in zip(batch, encoded): features[row['query_index'], C.ARMS.index(row['arm'])] = vector
        assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features, axis=2), 1, atol=3e-5)
        assert np.array_equal(features[:, :len(C.BASES)], base)
        K.save_array(out / 'features400.npy', features)
        result = dict(status='COMPLETE_ELEVEN_ARM_FROZEN_DINO_FEATURES', stage=400,
            features=K.ref(out / 'features400.npy'), new_images_encoded=1200, old_baseline_features_reused=3200,
            weight_sha256=E.DINO_SHA256, native_single_replay_max_error=error)
    guard(); result.update(source=K.ref(__file__), seconds=time.monotonic() - started,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(), extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0,
        resources_after=C.resources(c))
    K.write(out / (suffix + '_completion.json'), result); print(json.dumps(result), flush=True)
except BaseException as exc:
    K.write(out / (suffix + '_failure.json'), dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY', error=repr(exc), traceback=traceback.format_exc())); raise
