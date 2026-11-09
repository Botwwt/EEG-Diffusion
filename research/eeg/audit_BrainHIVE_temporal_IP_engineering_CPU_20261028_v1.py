"""Independently decode D37 engineering pixels and audit saved call/noise proof.

This does not recompute UNet responses or establish efficacy from training rows.
"""
import argparse
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def read(p):return json.loads(Path(p).read_bytes())


def run(completion_path,output):
    start=time.monotonic();done=read(completion_path)
    assert done['status']=='D37_ACTUAL_UNET_BINDING_AND_THREE_PIXEL_FALLBACKS_PASSED'
    assert not output.exists()
    import numpy as np
    sys.path[:0]=['/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages']
    from PIL import Image
    spec=read(done['spec']['path']);assert sha(done['spec']['path'])==done['spec']['sha256']
    for item in spec['source_pins']:assert sha(item['path'])==item['sha256']
    native=read(spec['native_engineering_completion']['path'])
    assert sha(spec['native_engineering_completion']['path'])==spec['native_engineering_completion']['sha256']
    old=next(row for row in native['cases'] if row['case']=='frozen_ordinary')
    assert sha(old['generated_path'])==old['generated_sha256']
    def pixels(path):
        with Image.open(path) as image:
            assert image.size==(512,512);return np.asarray(image.convert('RGB')).copy()
    reference=pixels(old['generated_path']);records=[]
    assert [x['case'] for x in done['cases']]==['disabled','zero_conditional','zero_flat','conditional','flat_residual']
    for row in done['cases']:
        assert sha(row['generated_path'])==row['generated_sha256']
        image=pixels(row['generated_path'])
        assert hashlib.sha256(image.tobytes()).hexdigest()==row['pixel_sha256']
        diff=int(np.abs(image.astype(np.int16)-reference.astype(np.int16)).max())
        assert diff==row['baseline_pixel_max_difference']
        assert row['UNet_NFE']==4 and row['native_IP_K_calls']==row['native_IP_V_calls']==280
        assert row['initial_UNet_input_sha256']==old['initial_UNet_input_sha256']
        assert row['initial_noise_latents_sha256']==old['initial_noise_latents_sha256']
        assert row['ancestral_noise_observations']==old['ancestral_noise_observations'] and len(row['ancestral_noise_observations'])==4
        if row['case'] in ['disabled','zero_conditional','zero_flat']:assert diff==0
        if row['enabled']:
            assert row['additional_temporal_K_calls']==row['additional_temporal_V_calls']==280
            assert row['active_IP_layers']==70 and row['total_active_IP_calls']==280
            assert len(row['actual_layer_diagnostics'])==70
        else:assert row['additional_temporal_K_calls']==row['additional_temporal_V_calls']==0
        if row['case']=='conditional':
            assert all(v['native_marginal_max_error']<2e-6 for v in row['actual_layer_diagnostics'].values())
            assert max(v['delta_RMS'] for v in row['actual_layer_diagnostics'].values())>0
        records.append(dict(case=row['case'],RGB_SHA=row['pixel_sha256'],maximum_RGB_difference_from_frozen_ordinary=diff,
                            per_image_UNet_calls=4,base_KV_calls_each=280,extra_KV_calls_each=row['additional_temporal_K_calls']))
    frozen=read(Path(__file__).resolve().parent/'configs/ordinary_frozen_base_20261028_v1.json')
    assert sha(frozen['checkpoint']['path'])==frozen['checkpoint']['sha256']=='a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
    R=Path(__file__).resolve().parent;job=read(R/'audit'/(spec['GPU_job_name']+'_job.json'))
    assert job['status']=='COMPLETED' and job['exit_code']==0
    for pid in [job['pid'],job['wrapper_pid']]:assert not (Path('/proc')/str(pid)).exists()
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip();assert not gpu
    with (R/'audit/GPU_JOB.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    value=dict(status='D37_FIVE_IMAGE_BYTES_RGB_CALL_NOISE_AND_RELEASE_INDEPENDENT_CPU_AUDIT_PASSED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),completion=dict(path=str(completion_path),sha256=sha(completion_path)),
        cases=records,all20_ancestral_noise_returns_verified=True,all_three_fallbacks_pixel_exact=True,
        nonzero_branch_pixel_changes_are_not_image_quality=True,native_weights_unchanged_during_GPU_run=done['ordinary_and_all_UNet_bytes_unchanged'],
        no_independent_UNet_numerical_replay_claimed=True,no_new_training_or_research_images=True,
        GPU_idle_and_lock_released=True,old_ordinary_SHA_verified=True,seconds=time.monotonic()-start)
    with output.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.completion,a.output)
