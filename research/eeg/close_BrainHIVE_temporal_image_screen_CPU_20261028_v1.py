"""Check both immutable prefixes, all parity images, job totals and GPU release."""
import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
R=Path('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1');sys.path.insert(0,str(R))
import brainhive_temporal_image_screen_20261028_v1 as U


def run():
    start=time.monotonic();sys.path[:0]=['/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages','/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages']
    from PIL import Image
    completion=U.B/'image_queue_v1/completion.json';done=U.read(completion)
    assert done['status']=='D37_FIXED_THREE_ARM_IMAGE_SCREEN_ACTUALLY_CLOSED' and done['final_stage']==400
    assert len(done['completed_steps'])==13 and all(step['exit_code']==0 for step in done['completed_steps'])
    spec=U.read(U.check(done['spec']));execution=U.read(U.check(done['execution']))
    for item in execution['source_pins']:U.check(item)
    first=U.read(U.B/'screen80/features/completion.json');last=U.read(U.B/'screen400/features/completion.json')
    f80=np.load(U.check(first['features']),allow_pickle=False);f400=np.load(U.check(last['features']),allow_pickle=False)
    rows80=U.read(U.check(first['rows']));rows400=U.read(U.check(last['rows']))
    assert f80.shape==(80,3,768) and f400.shape==(400,3,768) and f80.tobytes()==f400[:80].tobytes()
    assert rows400[:240]==rows80
    for row in rows80:assert U.sha(row['generated_path'])==row['generated_sha256']
    reference=spec['baseline_first_ordinary'];assert U.sha(reference['generated_path'])==reference['generated_sha256']
    with Image.open(reference['generated_path']) as image:reference_pixels=np.asarray(image.convert('RGB')).copy()
    attempts=sorted((U.B/'screen_images_v1/attempts').iterdir());assert len(attempts)==5
    research=0;parity=0;gpu_jobs=[];maxCPU=maxGPU=0;proofs=[]
    for path in attempts:
        record=U.read(path/'completion.json');native=U.read(path/'ordinary_parity.json');assert record['new_images']==240 and record['parity_images']==1 and record['reused_images']==0
        assert record['all_UNet_weights_unchanged'] is True and U.sha(native['generated_path'])==native['generated_sha256']
        with Image.open(native['generated_path']) as image:pixels=np.asarray(image.convert('RGB')).copy()
        assert np.array_equal(pixels,reference_pixels) and U.byte_sha(pixels)==native['generated_pixel_sha256']
        assert native['ancestral_noise_observations']==reference['ancestral_noise_observations'] and native['UNet_NFE']==4
        research+=record['new_images'];parity+=1;maxCPU=max(maxCPU,record['peak_extra_CPU_bytes']);maxGPU=max(maxGPU,record['peak_GPU_allocated_bytes'])
        job=U.read(R/'audit'/(path.name+'_job.json'));assert job['status']=='COMPLETED' and job['exit_code']==0;gpu_jobs.append(job)
        proofs.append(dict(job=path.name,parity_RGB_SHA=native['generated_pixel_sha256'],new_images=record['new_images']))
    for count in [80,400]:
        job=U.read(R/'audit'/f'D37_features{count}_20261028_v1_job.json');assert job['status']=='COMPLETED';gpu_jobs.append(job)
        cache=U.read(U.B/f'screen{count}/features/completion.json');maxCPU=max(maxCPU,cache['peak_extra_CPU_bytes']);maxGPU=max(maxGPU,cache['peak_GPU_allocated_bytes'])
        audit=U.read(U.B/f'screen{count}/score/independent_CPU_audit.json');assert audit['maximum_metric_absolute_error']==0
    assert research==1200 and parity==5 and maxCPU<10<<30 and maxGPU<20<<30
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    with (R/'audit/GPU_JOB.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(f,fcntl.LOCK_UN)
    frozen=U.read(U.check(spec['ordinary_frozen_spec']));U.check(frozen['checkpoint']);assert frozen['selected_strength']==1
    logical=0;allocated=0;seen=set()
    for p in U.B.rglob('*'):
        if p.is_file():
            s=p.stat();logical+=s.st_size;key=(s.st_dev,s.st_ino)
            if key not in seen:allocated+=s.st_blocks*512;seen.add(key)
    assert logical<2<<30
    value=dict(status='D37_COMPLETE400_PREFIX_PARITY_JOB_TOTALS_AND_RESOURCE_CLOSURE_CPU_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        queue_completion=U.ref(completion),full_first80_feature_bytes_and_240_image_rows_unchanged=True,all5_ordinary_disabled_pixel_replays_passed=True,
        parity_proofs=proofs,new_research_images=research,new_parity_images=parity,new_image_UNet_evaluations=4*(research+parity),
        full1200_new_candidate_image_audit=U.ref(U.B/'screen400/image_audit.json'),full12000_scalar_metric_independent_audit=U.ref(U.B/'screen400/score/independent_CPU_audit.json'),
        GPU_jobs=7,GPU_wrapped_seconds=sum(v['seconds'] for v in gpu_jobs),queue_wall_seconds=done['seconds'],
        maximum_extra_CPU_bytes=maxCPU,maximum_GPU_allocated_bytes=maxGPU,output_logical_bytes_including_own_hardlinks=logical,output_unique_inode_allocated_bytes=allocated,
        old_ordinary_weight_SHA=frozen['checkpoint']['sha256'],old_ordinary_strength=1,GPU_idle_and_lock_released=True,
        no_independent_UNet_or_DINO_backend_replay_claimed=True,new_model_fits=0,SGD_updates=0,seconds=time.monotonic()-start)
    out=U.B/'image_queue_v1/independent_closure_CPU.json'
    if out.exists():value=U.read(out)
    else:U.write(out,value)
    print(json.dumps(value))


if __name__=='__main__':json.loads(sys.stdin.buffer.readline());run()
