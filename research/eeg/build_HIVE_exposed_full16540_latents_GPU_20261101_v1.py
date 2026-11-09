"""Native FP32 VAE only: profile then two fixed disjoint chunk-prefix segments."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import traceback
import zipfile
import brainhive_exposed_native_latent_contract_CPU_20261101_v1 as K


def run(protocol, mode, segment=None):
    s = K.setup(protocol)
    import numpy as np
    import torch
    import build_BrainHIVE_official_train_latents_v1 as B
    from PIL import Image
    assert not torch.cuda.is_initialized()
    rows = K.input_check(s)
    K.require_wrapper()
    initial = K.resources(s, starting=True)
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.set_per_process_memory_fraction(.8)
    torch.cuda.reset_peak_memory_stats()
    started, base = time.monotonic(), K.rss()
    root = Path(s['output_root'])
    root.mkdir(exist_ok=True)
    where = root / ('profile' if mode == 'profile' else 'segment'+str(segment))
    assert not where.exists()
    where.mkdir()
    def guard():
        assert time.monotonic()-started < 840
        assert K.rss()-base <= s['maximum_extra_CPU_bytes']
        assert torch.cuda.max_memory_allocated() <= s['maximum_GPU_bytes']
        K.resources(s)
    def reserve(n):
        guard()
        K.resources(s, extra_bytes=n)
    try:
        from diffusers import AutoencoderKL
        from diffusers.loaders.single_file_utils import convert_ldm_vae_checkpoint
        from safetensors import safe_open
        config = K.read(K.check(s['VAE_config']))
        with safe_open(str(K.check(s['generator'])), framework='pt', device='cpu') as source:
            raw = {key:source.get_tensor(key) for key in source.keys() if key.startswith('first_stage_model.')}
        assert raw
        converted = convert_ldm_vae_checkpoint(raw, config)
        vae = AutoencoderKL.from_config(config).eval().requires_grad_(False)
        vae.load_state_dict(converted, strict=True)
        assert sum(p.numel() for p in vae.parameters()) == 83653863
        expected, actual = hashlib.sha256(), hashlib.sha256()
        state = vae.state_dict()
        assert set(state) == set(converted)
        for key, value in sorted(state.items()):
            reference = converted[key].to(dtype=value.dtype)
            assert torch.equal(value, reference)
            expected.update(key.encode());expected.update(reference.contiguous().numpy().tobytes())
            actual.update(key.encode());actual.update(value.contiguous().numpy().tobytes())
        assert expected.hexdigest() == actual.hexdigest()
        load_proof = dict(tensors=len(state), parameters=83653863, all_converted_tensors_exact=True,
                          expected_FP32_state_SHA256=expected.hexdigest(), actual_FP32_state_SHA256=actual.hexdigest())
        del state, converted, raw
        vae.cuda().float().eval().requires_grad_(False)
        vae.enable_slicing()
        assert all(p.dtype == torch.float32 and not p.requires_grad for p in vae.parameters())
        preprocess = B.native_preprocess()
        archives = {}
        def pixels(ids):
            result = []
            for index in ids:
                row = rows[index]
                if row['image_zip'] not in archives:
                    archives[row['image_zip']] = zipfile.ZipFile(K.check(s['archives'][row['image_zip']]))
                payload = archives[row['image_zip']].read(row['image_member'])
                assert len(payload) == row['image_bytes'] and hashlib.sha256(payload).hexdigest() == row['image_sha256']
                with Image.open(io.BytesIO(payload)) as image:
                    result.append(preprocess(image.convert('RGB')))
            return torch.stack(result)
        def encode(ids):
            guard()
            return B.encode_native_modes(vae, pixels(ids).cuda(), torch).cpu().numpy()
        if mode == 'profile':
            ids = s['profile_indices']
            first = np.concatenate([encode(ids[lo:lo+4]) for lo in (0, 4)])
            second = np.concatenate([encode([index]) for index in ids])
            assert np.array_equal(first, second)
            manual_error = 0.0
            for index in ids:
                row = rows[index]
                archive = archives[row['image_zip']]
                with Image.open(io.BytesIO(archive.read(row['image_member']))) as image:
                    image = image.convert('RGB'); width,height = image.size
                    width,height = ((int(512*width/height),512) if height <= width else (512,int(512*height/width)))
                    image = image.resize((width,height), Image.Resampling.BILINEAR)
                    x,y = int(round((width-512)/2)),int(round((height-512)/2))
                    array = np.asarray(image.crop((x,y,x+512,y+512)),dtype=np.float32).transpose(2,0,1)
                    manual = (array/np.float32(255)-np.float32(.5))/np.float32(.5)
                error = float(np.max(np.abs(pixels([index])[0].numpy()-manual)))
                manual_error = max(manual_error,error)
                assert error <= 1e-7
            old_checks = []
            if s['old_T']['available']:
                old = np.load(K.check(s['old_T']['bank']), mmap_mode='r', allow_pickle=False)
                assert old.shape == (13230,4,64,64) and old.dtype == np.float32
                for position in (0,13229):
                    index = s['old_T']['array_rows'][position]
                    value = encode([index])
                    expected_old = np.array(old[position:position+1],copy=True)
                    assert np.allclose(value, expected_old, **s['numerical_guard'])
                    old_checks.append(dict(global_train_row=index, old_T_position=position,
                                           maximum_absolute_error=float(np.max(np.abs(value-expected_old)))))
            np.save(where/'native_profile_latents.npy',first,allow_pickle=False)
            result = dict(status='PASS_ACTUAL_NATIVE_VAE_INPUT_PROFILE_NO_FIT',
                          profile_indices=ids, repeated_singleton_batch4_bit_exact=True,
                          independent_manual_pixel_preprocessing_maxerror=manual_error,
                          old_T_replays=old_checks, latent=K.ref(where/'native_profile_latents.npy'))
        else:
            assert segment in (0,1)
            profile = K.read(root/'profile/completion.json')
            assert profile['status'] == 'PASS_ACTUAL_NATIVE_VAE_INPUT_PROFILE_NO_FIT'
            bank = root/'bank'
            if segment == 0:
                assert not bank.exists()
            else:
                previous = K.read(root/'segment0/completion.json')
                assert previous['status'] == 'COMPLETE_PLANNED_FIRST32_CHUNKS_ONLY'
                assert len(list((bank/'chunks').glob('chunk-*.json'))) == 32
            registration = dict(schema='EXPOSED_FULL16540_NATIVE_TRAIN_VAE_INPUTS_V1',
                actual_official_full_train_scope=True, array_rows=list(range(16540)), shape=list(K.SHAPE),
                dtype='float32', batch_size=4, chunk_size=256, preprocessing=s['preprocessing'],
                VAE_parameters=83653863,VAE_precision='FP32 from exact native checkpoint',
                VAE_strict_load=True,VAE_scaling_factor=float(vae.config.scaling_factor),stochastic_sampling=False,
                image_identity_sha256=hashlib.sha256(B.canonical(B.image_identity(rows))).hexdigest(),
                inputs=dict(image_manifest=s['image_manifest'],generator=s['generator'],VAE_config=s['VAE_config'],archives=s['archives']),
                protocol=K.ref(protocol), source_pins=s['source_pins'], official200_previously_exposed=True,
                independent_H=False, legacy_entrypoint_admission=False, new_separate_input_stage_review=s['source_data_resource_review'])
            class PlannedPrefixBoundary(Exception):
                pass
            def boundary(stage, number):
                if segment == 0 and stage == 'after_chunk_commit' and number == 31:
                    raise PlannedPrefixBoundary()
            try:
                done = B._build_bank(bank,registration,encode,guard,reserve=reserve,fault=boundary)
                assert segment == 1
                result = dict(status='COMPLETE_ALL16540_NATIVE_TRAIN_LATENTS',
                              bank_completion=K.ref(bank/'completion.json'), bank=dict(path=done['bank']['path'],
                              bytes=Path(done['bank']['path']).stat().st_size,sha256=done['bank']['sha256']),
                              completed_chunks=65, VAE_encode_calls_this_segment=done['VAE_encode_calls_this_attempt'])
                assert result['VAE_encode_calls_this_segment'] == 2087 and done['committed_chunks_reused'] == 32
            except PlannedPrefixBoundary:
                assert segment == 0 and len(list((bank/'chunks').glob('chunk-*.json'))) == 32
                assert not (bank/'completion.json').exists()
                result = dict(status='COMPLETE_PLANNED_FIRST32_CHUNKS_ONLY', completed_chunks=32,
                              images_encoded_this_segment=8192, VAE_encode_calls_this_segment=2048,
                              prefix_registration=K.ref(bank/'registration.json'))
        for archive in archives.values():archive.close()
        guard()
        for pin in s['source_pins']:K.check(pin)
        result.update(protocol=K.ref(protocol), load_proof=load_proof, initial_resources=initial,
                      seconds=time.monotonic()-started, peak_GPU_bytes=torch.cuda.max_memory_allocated(),
                      maximum_extra_CPU_bytes=K.rss()-base, parameter_updates=0,
                      new_UNet_calls=0, new_research_images=0, official_test_image_payloads_opened=0,
                      old_T_bank_overwritten=False, independent_H=False, NEW_contacted=False)
        K.write(where/'completion.json',result)
        print(json.dumps({k:result[k] for k in ('status','seconds','peak_GPU_bytes','maximum_extra_CPU_bytes')}))
    except BaseException as error:
        K.write(where/'failure.json',dict(error=repr(error),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol',required=True)
    parser.add_argument('--mode',choices=('profile','build'),required=True)
    parser.add_argument('--segment',type=int,choices=(0,1))
    args = parser.parse_args()
    run(args.protocol,args.mode,args.segment)
