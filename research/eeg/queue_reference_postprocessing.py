"""Finite CPU-only audits stream after complete images; pairing waits for scores.

Does not touch the GPU lock, generate images, select methods, or unlock a test.
"""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'
STATE=ROOT/'audit/S1_postprocessing_state.json'


def write(value):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,STATE)


def run(name,seconds,script,*args):
    log=ROOT/'logs'/f'{name}.log';assert not log.exists(),'Preserve a previous postprocessing attempt'
    command=[PY,str(ROOT/script),*map(str,args)]
    env=os.environ|dict(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with log.open('xb') as file:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT)
        write(dict(status='RUNNING_CPU_ONLY',stage=name,pid=child.pid,parent_pid=os.getpid(),command=command,log=str(log),GPU_used=False))
        try:code=child.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            code=-9
    receipt=dict(stage=name,command=command,seconds=time.monotonic()-start,exit_code=code,GPU_used=False,
                 log=str(log),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:file.write(json.dumps(receipt)+'\n')
    assert code==0,f'{name} failed: inspect {log}'


def main():
    os.nice(10);start=time.monotonic()
    write(dict(status='WAITING_BOTH_FULL_REFERENCES',pid=os.getpid(),GPU_used=False))
    deadline=json.loads((ROOT/'configs/compute_time_bounds.json').read_text())['conservative_occupied_time_deadline_epoch']
    launch=json.loads((ROOT/'audit/S1_queue_launch_batchsafe_v3.json').read_text())
    def wait_for(predicate,description):
        write(dict(status='WAITING_COMPLETE_PREREQUISITE_CPU_ONLY',stage=description,pid=os.getpid(),GPU_used=False))
        while True:
            state=json.loads((ROOT/'audit/S1_queue_state.json').read_text())
            assert state['status']!='FAILED_STOPPED_DEPENDENT_JOBS',state
            if predicate():return
            assert time.time()<deadline,'Campaign bound reached before CPU postprocessing; save state'
            proc=Path('/proc')/str(launch['pid'])/'cmdline'
            assert proc.exists() and str(ROOT/'queue_reference_stage1.py').encode() in proc.read_bytes(), 'Registered image queue stopped; inspect receipts before restarting'
            time.sleep(30)
    gallery=DATA/'evaluation/gallery/V/manifest.jsonl'
    image_base=DATA/'images/S1_full_native_batchsafe_v2/V';evaluation=DATA/'evaluation/V'
    methods=['REF_REG','REF_PRIOR_V']
    for method in methods:
        arm=f'{method}_correct_R4'
        completion=image_base/arm/'completion.json'
        wait_for(lambda:completion.exists() and gallery.exists(),f'complete_images_and_original_gallery_for_{method}')
        generation=json.loads(completion.read_text())
        assert generation['status']=='COMPLETED' and generation['queries']==16600
        run(f'S1_V_low_level_{method}',3600,'score_low_level.py',
            '--manifest',image_base/arm/'manifest.jsonl','--gallery-manifest',gallery,
            '--output',evaluation/arm/'low_level')
    wait_for(lambda:json.loads((ROOT/'audit/S1_queue_state.json').read_text())['status']=='COMPLETED_TWO_FULL_REFERENCES',
        'both_complete_primary_reference_evaluations_before_paired_analysis')
    run('S1_V_paired_PRIOR_V_vs_REG',600,'paired_analysis.py',
        '--reference-manifest',image_base/'REF_REG_correct_R4/manifest.jsonl',
        '--candidate-manifest',image_base/'REF_PRIOR_V_correct_R4/manifest.jsonl',
        '--reference-metrics',evaluation/'REF_REG_correct_R4/metrics/per_query.csv',
        '--candidate-metrics',evaluation/'REF_PRIOR_V_correct_R4/metrics/per_query.csv',
        '--output',DATA/'evaluation/paired/V_PRIOR_V_vs_REG_v1')
    args=[]
    for method in methods:
        arm=f'{method}_correct_R4'
        args+=['--arm',arm,image_base/arm/'manifest.jsonl',evaluation/arm/'metrics/per_query.csv']
    run('S1_V_full_offline_gallery',3600,'build_offline_gallery.py',*args,
        '--output',DATA/'galleries/V_references_batchsafe_v1')
    write(dict(status='COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED',GPU_used=False,
        seconds_including_wait=time.monotonic()-start,final_test_unlocked=False,
        controls_or_new_training_started=False,
        gallery=str(DATA/'galleries/V_references_batchsafe_v1/index.html')))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        old=json.loads(STATE.read_text()) if STATE.exists() else {}
        write(old|dict(status='FAILED_STOPPED_DEPENDENT_POSTPROCESSING',error=str(exc)));raise
