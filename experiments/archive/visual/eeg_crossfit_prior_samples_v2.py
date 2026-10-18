"""SERVER ONLY: matched source images for the encoder-exposure prior experiment.

Original Turbo and original CLIP-H IP-Adapter; no SynCLR adapter or replacement
projection. Eight arms x32 fixed source images; uint8 archives remain sealed RAM.
"""
import ast
import ctypes
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
import time
import traceback

assert sys.platform == 'linux'
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '2'
B = Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT = Path('/root/mced_eeg_crossfit_prior_samples_v2')
ASSET = OUT / 'assets'
PARENT = Path('/root/mced_eeg_generation_ram_assets_v4')
TRAIN = Path('/root/mced_eeg_crossfit_point_priors_v1')
QUERY = Path('/root/mced_eeg_crossfit_conditions_v1')
FEATURE = Path('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2')
FDS = []
START = time.monotonic()
SEEDS = [2026109501, 2026109502]
ARMS = ['point_' + mode + '_' + str(seed) for mode in ['in_sample', 'out_of_fold', 'clean_visual'] for seed in SEEDS] + ['constant_fit_mean', 'true_image_oracle']
COUNTS = dict(UNet_calls=0, VAE_decoder_calls=0, text_encoder_calls=0,
              image_projection_calls=0, optimizer_updates=0, EEG_calls=0, prior_calls=0)
helper = B / 'prepare_eeg_crossfit_conditions_v1.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'ff9a2e7ba44e56e6af1882beccd4eb8fab724e6292145c1fd6927d20f220ae76'
names = ['identity', 'write', 'verify', 'adopt']
nodes = [n for n in ast.parse(helper.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert [n.name for n in nodes] == names
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'), globals())
assert not OUT.exists()
OUT.mkdir(); ASSET.mkdir()


def run():
    import numpy as np
    import torch
    from diffusers import StableDiffusionXLPipeline
    from diffusers.models.embeddings import ImageProjection
    from diffusers.loaders.single_file_utils import convert_ldm_unet_checkpoint, convert_ldm_vae_checkpoint, convert_ldm_clip_checkpoint, convert_open_clip_checkpoint
    from safetensors.torch import load_file
    verify(TRAIN); verify(QUERY)
    verify(FEATURE, ['train_image_embeds.pt'])
    prior = json.loads((TRAIN / 'results.json').read_text())
    assert prior['counts']['optimizer_updates'] == 13500 and prior['all_matching_checks_passed']
    os.kill(prior['pid'], 0)
    pe = next(r for r in prior['exports'] if r['name'] == 'private_crossfit_prior_predictions.pt')
    packet = torch.load(adopt(pe), weights_only=True)
    assert packet['target_space'] == 'Original_CLIP_H_final_raw1024'
    ids = packet['planned_image_indices']
    assert ids == json.loads((QUERY / 'split_and_image_plan.json').read_text())['planned_image_indices']
    assert len(ids) == len(set(ids)) == len({i // 10 for i in ids}) == 32
    source_ids = packet['source_indices'].tolist()
    positions = {v: i for i, v in enumerate(source_ids)}
    select = [positions[i] for i in ids]
    conditions = {}
    for seed in SEEDS:
        for mode in ['in_sample', 'out_of_fold', 'clean_visual']:
            key = 'point_' + mode
            conditions[key + '_' + str(seed)] = packet['predictions'][str(seed)][key][select].float()
    conditions['constant_fit_mean'] = packet['fit_mean'][None].expand(32, -1).float()
    conditions['true_image_oracle'] = torch.load(FEATURE / 'train_image_embeds.pt', weights_only=True)[ids].float()
    assert set(conditions) == set(ARMS)
    assert all(v.shape == (32, 1024) and torch.isfinite(v).all() for v in conditions.values())
    cg = Path('/sys/fs/cgroup')
    stat = dict(x.split() for x in (cg / 'memory.stat').read_text().splitlines())
    host = int((cg / 'memory.max').read_text()) - int((cg / 'memory.current').read_text()) + int(stat['file']) - int(stat['shmem'])
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 20 << 30 and host > 30 << 30
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), torch.get_num_threads(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    hooks = []
    try:
        # Reviewed strict base-loader slice only, before any IP adapter loading or
        # replacement projection. The namespace does not contain itself.
        source = B / 'eeg_generator_conditioning_training_v1.py'
        assert identity(source) == dict(bytes=21503, sha256='8d05598382a46b0f41c0eb97c54ab0570c20fa64b62fc9b6a1694c0385e4a659')
        body = next(n for n in ast.parse(source.read_bytes()).body if isinstance(n, ast.Try)).body
        lo = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'pipe' for t in n.targets))
        hi = next(i for i, n in enumerate(body) if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'load_ip_adapter')
        namespace = {**globals(), **locals()}
        assert 'namespace' not in namespace
        exec(compile(ast.Module(body=body[lo:hi], type_ignores=[]), str(source), 'exec'), namespace)
        pipe = namespace['pipe']
        write('strict_base_weight_coverage.json', namespace['coverage'])
        del namespace
        pipe.load_ip_adapter(str(ASSET), subfolder='', weight_name='adapter.safetensors',
            image_encoder_folder=None, torch_dtype=torch.float32)
        pipe.set_ip_adapter_scale(1.)
        projection = pipe.unet.encoder_hid_proj.image_projection_layers[0]
        assert type(projection) is ImageProjection
        # Verify every loaded adapter tensor against the whole-hash-pinned file.
        weights = load_file(str(ASSET / 'adapter.safetensors'), device='cpu')
        consumed = set()
        for dst, src in [('image_embeds.weight', 'image_proj.proj.weight'), ('image_embeds.bias', 'image_proj.proj.bias'),
                         ('norm.weight', 'image_proj.norm.weight'), ('norm.bias', 'image_proj.norm.bias')]:
            assert torch.equal(projection.state_dict()[dst].cpu().float(), weights[src].float())
            consumed.add(src)
        for index, (name, processor) in enumerate(pipe.unet.attn_processors.items()):
            if name.endswith('attn1.processor'):
                continue
            for suffix in ['to_k_ip', 'to_v_ip']:
                src = f'ip_adapter.{index}.{suffix}.weight'
                dst = suffix + '.0.weight'
                assert torch.equal(processor.state_dict()[dst].cpu().float(), weights[src].float())
                consumed.add(src)
        assert consumed == set(weights)
        write('strict_IP_adapter_coverage.json', dict(complete=True, tensor_keys=len(consumed),
            original_image_projection_input_dimension=int(projection.image_embeds.in_features),
            all_loaded_projection_and_attention_tensors_exact_after_declared_FP32_cast=True,
            projection_replaced=False))
        del weights
        pipe.to('cuda'); pipe.set_progress_bar_config(disable=True)
        assert pipe.image_encoder is None and projection.image_embeds.in_features == 1024
        modules = [pipe.unet, pipe.vae, pipe.text_encoder, pipe.text_encoder_2]
        for model in modules:
            model.eval().requires_grad_(False)
        fixed = [p for model in modules for p in list(model.parameters()) + list(model.buffers())]
        versions = [p._version for p in fixed]
        assert all(p.dtype == torch.float32 for p in fixed if p.is_floating_point())
        conditions = {k: v.cuda() for k, v in conditions.items()}
        trace = []
        run_name = [None]

        def unet_before(model, args, kwargs):
            x = args[0] if args else kwargs['sample']
            t = args[1] if len(args) > 1 else kwargs['timestep']
            c = kwargs['added_cond_kwargs']['image_embeds'][0]
            assert x.shape == (1, 4, 64, 64) and c.shape == (1, 1, 1024)
            COUNTS['UNet_calls'] += 1
            trace.append(dict(run=run_name[0], timestep=float(t),
                latent_sha256=hashlib.sha256(x.detach().cpu().numpy().tobytes()).hexdigest(),
                condition_sha256=hashlib.sha256(c.detach().cpu().numpy().tobytes()).hexdigest()))

        def vae_before(model, args):
            COUNTS['VAE_decoder_calls'] += 1

        def text_before(model, args):
            COUNTS['text_encoder_calls'] += 1

        def project_before(model, args):
            COUNTS['image_projection_calls'] += 1

        hooks = [pipe.unet.register_forward_pre_hook(unet_before, with_kwargs=True),
            pipe.vae.decoder.register_forward_pre_hook(vae_before),
            pipe.text_encoder.register_forward_pre_hook(text_before),
            pipe.text_encoder_2.register_forward_pre_hook(text_before), projection.register_forward_pre_hook(project_before)]
        cohorts, exports, first_latents = [], [], {}
        with torch.no_grad():
            for arm in ARMS:
                images = []
                for position, index in enumerate(ids):
                    run_name[0] = arm + '/' + str(index)
                    start = len(trace)
                    gen = torch.Generator(device='cuda').manual_seed(2026111600 + index)
                    image = pipe(prompt='', ip_adapter_image_embeds=[conditions[arm][position:position + 1, None, :]],
                        num_inference_steps=4, guidance_scale=0., height=512, width=512,
                        generator=gen, output_type='np').images
                    assert image.shape == (1, 512, 512, 3) and np.isfinite(image).all() and len(trace) - start == 4
                    first = trace[start]['latent_sha256']
                    if index in first_latents:
                        assert first_latents[index] == first
                    else:
                        first_latents[index] = first
                    images.append(np.rint(np.clip(image[0], 0, 1) * 255).astype(np.uint8))
                    if (position + 1) % 4 == 0:
                        print(json.dumps(dict(stage='CROSSFIT_SOURCE_IMAGES', arm=arm,
                            images=position + 1, counts=COUNTS)), flush=True)
                name = arm + '_samples.npz'
                lib = ctypes.CDLL(None, use_errno=True)
                fd = lib.syscall(ctypes.c_long(319), ctypes.c_char_p(name.encode()), ctypes.c_uint(3))
                assert fd >= 0, ctypes.get_errno()
                FDS.append(fd)
                with os.fdopen(os.dup(fd), 'wb') as f:
                    np.savez_compressed(f, arr_0=np.stack(images), image_indices=np.asarray(ids, dtype=np.int64))
                fcntl.fcntl(fd, 1033, 15); assert fcntl.fcntl(fd, 1034) == 15
                p = Path(f'/proc/{os.getpid()}/fd/{fd}')
                (OUT / name).symlink_to(p)
                row = dict(arm=arm, file=name, images=32, memfd_path=str(p), identity=identity(p),
                    oracle_reference=arm == 'true_image_oracle', storage='sealed_RAM_volatile')
                cohorts.append(row); exports.append(dict(name=name, memfd_path=str(p), **row['identity']))
                write(arm + '_ready.json', row)
                del images
        assert COUNTS == dict(UNet_calls=1024, VAE_decoder_calls=256, text_encoder_calls=512,
            image_projection_calls=1024, optimizer_updates=0, EEG_calls=0, prior_calls=0), COUNTS
        assert versions == [p._version for p in fixed] and all(p.grad is None for p in fixed)
        write('trace.json', trace)
        write('input_bindings.json', dict(prior_commit=identity(TRAIN / 'final_commit.json'),
            query_plan=identity(QUERY / 'split_and_image_plan.json'), prediction_export=pe,
            final_image_features=identity(FEATURE / 'train_image_embeds.pt'),
            only_explicit_oracle_arm_uses_target_image_features=True))
        return dict(complete=True, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seconds=time.monotonic() - START, counts=COUNTS, cohorts=cohorts, exports=exports,
            image_indices=ids, paired_initial_latent_hashes={str(k): v for k, v in first_latents.items()},
            new_images=256, original_generator_and_IPAdapter_unchanged=True,
            source_heldout_only=True, official_test_read=False, metrics_not_yet_computed=True,
            novelty_established=False, SOTA=False)
    finally:
        for handle in hooks:
            handle.remove()
        cpu, cuda, threads, tf32, cudnn = saved
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda); torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = cudnn


try:
    write('plan.json', dict(pid=os.getpid(), script=identity(Path(__file__)),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), arms=ARMS,
        images_per_arm=32, new_images=256, source_image_subset_locked_before_prior_training=True,
        primary='point_out_of_fold minus point_in_sample, two fixed seeds averaged per image; raw prior outputs without radius selection',
        controls='Matched clean_visual adapter on the SAME EEG query, constant TRAIN final mean, explicit true-final-image oracle',
        sampler='Original strict SDXL-Turbo and IPAdapter image projection1024;FP32/TF32off,4steps,CFG0,512px,B1,noise2026111600+sourceindex',
        source_only=True, no_checkpoint_or_best_of_N_selection=True,
        sample_archives='Sealed read-only RAM with24hlease; independent metric process must adopt descriptors',
        image_quality_established=False, SOTA=False))
    verify(PARENT)
    parent = json.loads((PARENT / 'results.json').read_text())
    keeper = json.loads((PARENT / 'keeper_status.json').read_text())
    os.kill(keeper['pid'], 0)
    assert str(B / 'prepare_eeg_generation_ram_assets_v4.py').encode() in Path('/proc/' + str(keeper['pid']) + '/cmdline').read_bytes().split(b'\0')
    adopted = []
    for name, size, sha in [
        ('adapter.safetensors', 698391064, 'ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831'),
        ('sdxl_turbo_fp16.safetensors', 6938081905, 'e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26')]:
        row = next(r for r in parent['assets'] if r['path'] == name)
        assert {k: row[k] for k in ['bytes', 'sha256']} == dict(bytes=size, sha256=sha)
        p = adopt(row); (ASSET / name).symlink_to(p)
        adopted.append(dict(name=name, own_readonly_path=str(p), **{k: row[k] for k in ['bytes', 'sha256']}))
    cm = PARENT / 'config_manifest.json'
    assert identity(cm) == dict(bytes=6240, sha256='d67a0aae3e7eb5bed4a1876f600eb2fc7a62a00c3b9d6f76a8c8a950e1f7b857')
    for row in json.loads(cm.read_text())['files']:
        p = PurePosixPath(row['path'])
        assert not p.is_absolute() and '..' not in p.parts and '\\' not in row['path']
        assert identity(PARENT / p) == {k: row[k] for k in ['bytes', 'sha256']}
        dest = ASSET / p; dest.parent.mkdir(parents=True, exist_ok=True); dest.symlink_to(PARENT / p)
    write('asset_adoption.json', dict(complete=True, assets=adopted, configs_are_verified_readonly_links=True))
    while not (TRAIN / 'final_commit.json').exists():
        if (TRAIN / 'failure.json').exists():
            raise RuntimeError('Matched prior training failed')
        if (TRAIN / 'plan.json').exists():
            os.kill(json.loads((TRAIN / 'plan.json').read_text())['pid'], 0)
        assert time.monotonic() - START < 10800
        time.sleep(10)
    result = run()
    write('results.json', result)
    write('final_commit.json', dict(complete=True,
        files={p.name: identity(p) for p in OUT.glob('*.json')}, volatile_exports=result['exports']))
    print('READY_CROSSFIT_SOURCE_IMAGES', flush=True)
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    until = time.monotonic() + 86400
    while time.monotonic() < until:
        write('keeper_status.json', dict(pid=os.getpid(), ready=True,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            lease_seconds_remaining=until - time.monotonic()), True)
        time.sleep(30)
except BaseException:
    write('failure.json', dict(traceback=traceback.format_exc(), counts=COUNTS))
    raise
finally:
    for fd in FDS:
        os.close(fd)
