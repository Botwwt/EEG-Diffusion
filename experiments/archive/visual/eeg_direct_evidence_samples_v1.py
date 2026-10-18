"""SERVER ONLY: matched source images for direct EEG image evidence.

Original Turbo and original CLIP-H IP-Adapter; no SynCLR adapter or replacement
projection. Five arms x32 fixed source images; uint8 archives remain sealed RAM.
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
OUT = Path('/root/mced_eeg_direct_evidence_samples_v1')
ASSET = OUT / 'assets'
PARENT = Path('/root/mced_eeg_generation_ram_assets_v4')
TRAIN = Path('/root/mced_eeg_direct_evidence_vae_preflight_v1')
QUERY = Path('/root/mced_eeg_norm_stats_conditions_v1')
FEATURE = Path('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2')
FDS = []
START = time.monotonic()
ARMS = ['point_eeg','constant_prior','direct_eeg','direct_shuffled_eeg','direct_true_visual_oracle']
COUNTS = dict(UNet_calls=0,VAE_decoder_calls=0,text_encoder_calls=0,image_projection_calls=0,optimizer_updates=0,EEG_calls=0,prior_calls=0,prefix12_calls=0,prefix12_states=0,visual_head_calls=0,visual_head_states=0,backward_calls=0)
helper = B / 'prepare_eeg_crossfit_conditions_v1.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'ff9a2e7ba44e56e6af1882beccd4eb8fab724e6292145c1fd6927d20f220ae76'
names = ['identity', 'write', 'verify', 'adopt']
nodes = [n for n in ast.parse(helper.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert [n.name for n in nodes] == names
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'), globals())
assert not OUT.exists()
OUT.mkdir(); ASSET.mkdir()


run_source=B/'eeg_direct_evidence_samples_run_v1.py'
assert identity(run_source)=={'bytes': 19604, 'sha256': '167f47a2d487c2d856785a4ca4565da9f8c7f90c18484a87c96e10a92610ab32'}
exec(compile(run_source.read_bytes(),str(run_source),'exec'),globals())


try:
    write('plan.json',dict(pid=os.getpid(),script=identity(Path(__file__)),run_source=identity(run_source),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),arms=ARMS,images_per_arm=32,new_images=160,
        source_subset='Original32 source-heldout concepts reused; exploratory, not independent confirmation',
        primary='DINO2AFC direct_eeg minus point_eeg; content control direct_eeg minus direct_shuffled_eeg',
        evidence='phi=original trained temperature*cosine(frozen EEG query, jointVisual(normalized middle12(image)))',
        guidance='last2of4 EulerAncestral epsilon steps; x0=x-sigma*epsilon; g=grad_x0(phi); epsilon_guided=epsilon-sigma*g; stopped denoiser Jacobian approximation; gamma1 fixed',
        controls='TRAINfitmean raw originalCLIP-H IPAdapter; cyclic1 EEG shuffle; true middle-visual query oracle unavailable at deployment',
        sampler='Original strict SDXL-Turbo+IPAdapter; FP32/TF32off,4steps,CFG0,512,B1,noise2026111600+sourceindex',
        manual_sampler_requires_bitwise_pipeline_replay=True,extra_guidance_compute_not_matched=True,
        source_only=True,no_best_of_N_selection=True,no_new_training=True,calibrated_likelihood=False,
        general_contrastive_guidance_has_prior_art=True,novelty_established=False,SOTA=False))
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
    print('READY_DIRECT_EVIDENCE_IMAGES', flush=True)
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    ctypes.CDLL(None).malloc_trim(0)
    write('runtime_release.json', dict(inference_copies_disposed_to_meta=True,
        libc_malloc_trim_called=True,
        process_memory_status=[line for line in Path('/proc/self/status').read_text().splitlines() if line.startswith(('VmRSS:', 'VmSize:'))],
        pytorch_cuda_allocated_bytes=torch.cuda.memory_allocated(),
        pytorch_cuda_reserved_bytes=torch.cuda.memory_reserved(),
        sealed_samples_and_asset_descriptors_preserved=True))
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
