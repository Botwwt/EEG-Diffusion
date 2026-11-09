"""Finite authorized S1 queue: full references, actual images, full per-query evaluation.

No final benchmark or candidate adapter is started by this queue.
"""
from pathlib import Path
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'


def run(name,seconds,script,*args,gpu=True):
    base=name;attempt=1
    while (ROOT/'logs'/f'{name}.log').exists() or (ROOT/'audit'/f'{name}_job.json').exists():
        attempt+=1;name=f'{base}_attempt{attempt}'
    command=[PY,str(ROOT/script),*map(str,args)]
    if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(seconds),'--',*command]
    log=ROOT/'logs'/f'{name}.log';assert not log.exists(),'Do not overwrite a prior queued attempt'
    with log.open('wb') as file:
        p=subprocess.Popen(command,cwd=ROOT,stdout=file,stderr=subprocess.STDOUT)
        state=dict(status='RUNNING',stage=name,pid=p.pid,command=command,log=str(log),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
        (ROOT/'audit/S1_queue_state.json').write_text(json.dumps(state,indent=2)+'\n')
        code=p.wait()
    assert code==0,f'{name} failed; preserve files and stop dependent jobs, log={log}'


def main():
    start=time.monotonic();completion=DATA/'runs/S1_full_atm_native_v1/full/completion.json'
    while not completion.exists():
        job=json.loads((ROOT/'audit/S1_full_reference_job.json').read_text())
        assert job['status']=='RUNNING' and Path('/proc',str(job['pid'])).exists(),'Training stopped before all subjects completed'
        assert time.monotonic()-start<7300
        time.sleep(10)
    assert json.loads(completion.read_text())['status']=='COMPLETED'
    # Allow the budget wrapper to finish its receipt and release its flock.
    for _ in range(10):
        if json.loads((ROOT/'audit/S1_full_reference_job.json').read_text())['status']=='COMPLETED':break
        time.sleep(1)
    velocity=DATA/'runs/S1_prior_velocity_repair_v1/full/completion.json'
    wait_start=time.monotonic()
    while not velocity.exists():
        job=json.loads((ROOT/'audit/S1_prior_velocity_repair_job.json').read_text())
        assert job['status']=='RUNNING' and Path('/proc',str(job['pid'])).exists(),'Velocity repair stopped before all subjects completed'
        assert time.monotonic()-wait_start<1830
        time.sleep(10)
    for _ in range(10):
        if json.loads((ROOT/'audit/S1_prior_velocity_repair_job.json').read_text())['status']=='COMPLETED':break
        time.sleep(1)
    run('S0_velocity_prior_T_images',180,'diagnose_prior_velocity_T_images.py')
    gallery=DATA/'evaluation/gallery/V'
    for method in ['REF_REG','REF_PRIOR_V']:
        arm=f'{method}_correct_R4';images=DATA/'images/S1_full_native_batchsafe_v2/V'/arm
        run(f'S1_V_generate_{method}',18000,'generate_reference_images.py','--method',method)
        if not (gallery/'features/completion.json').exists():
            run('S1_prepare_V_gallery',120,'prepare_real_gallery.py','--generation-manifest',images/'manifest.jsonl','--partition','V',gpu=False)
            run('S1_evaluate_V_gallery',1800,'extract_image_features.py','--manifest',gallery/'manifest.jsonl','--output',gallery/'features','--real-gallery')
        features=DATA/'evaluation/V'/arm/'features'
        run(f'S1_V_evaluate_{method}',10800,'extract_image_features.py','--manifest',images/'manifest.jsonl','--output',features)
        run(f'S1_V_score_{method}',600,'score_full_manifest.py','--manifest',images/'manifest.jsonl','--features',features,
            '--gallery-manifest',gallery/'manifest.jsonl','--gallery-features',gallery/'features','--output',features.parent/'metrics',gpu=False)
    (ROOT/'audit/S1_queue_state.json').write_text(json.dumps(dict(status='COMPLETED_TWO_FULL_REFERENCES',
        final_test_started=False,adapter_candidates_started=False,seconds=time.monotonic()-start),indent=2)+'\n')


if __name__=='__main__':
    try:main()
    except Exception as exc:
        state=ROOT/'audit/S1_queue_state.json'
        old=json.loads(state.read_text()) if state.exists() else {}
        state.write_text(json.dumps(old|dict(status='FAILED_STOPPED_DEPENDENT_JOBS',error=str(exc)),indent=2)+'\n')
        raise
