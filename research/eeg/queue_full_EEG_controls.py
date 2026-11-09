"""Finite full-V EEG intervention queue with a complete-scope budget gate.

Waits for two complete references and CPU audit, uses the locked selection rule,
then permuted/missing EEG. It cannot start training or unlock the benchmark.
"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
DATA=Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
PY='/root/miniconda3/bin/python'
STATE=ROOT/'audit/S1_EEG_control_queue_state.json'


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(value):
    tmp=STATE.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,STATE)


def run(name,seconds,script,*args,gpu=True):
    log=ROOT/'logs'/f'{name}.log';receipt=ROOT/'audit'/f'{name}_job.json'
    assert not log.exists() and not receipt.exists(),'Preserve previous attempts; inspect before explicit versioned restart'
    command=[PY,str(ROOT/script),*map(str,args)]
    if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(seconds),'--',*command]
    env=os.environ.copy()
    if not gpu:env.update(CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    start=time.monotonic()
    with log.open('xb') as file:
        child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT)
        write(dict(status='RUNNING',stage=name,pid=child.pid,parent_pid=os.getpid(),command=command,log=str(log),GPU_used=gpu))
        try:code=child.wait(timeout=seconds+60)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=25)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            code=-9
    if not gpu:
        with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
            file.write(json.dumps(dict(stage=name,command=command,seconds=time.monotonic()-start,
                exit_code=code,GPU_used=False,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    assert code==0,f'{name} failed; inspect {log}'


def budget_gate(method,controls_left):
    subprocess.run([PY,str(ROOT/'refresh_status.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    progress=json.loads((ROOT/'audit/current_progress.json').read_text())
    assert not any(j['alive'] for j in progress['live_GPU_jobs']),'Another registered GPU job is live'
    arm=f'{method}_correct_R4'
    generation=json.loads((DATA/'images/S1_full_native_batchsafe_v2/V'/arm/'completion.json').read_text())
    evaluation=json.loads((DATA/'evaluation/V'/arm/'features/completion.json').read_text())
    # Actual complete-run wall times include loading, IO and all evaluator stages.
    measured_hours=(generation['seconds']+evaluation['seconds'])/3600
    required=controls_left*measured_hours*1.10+5.+1.5+.5
    remaining=min(progress['remaining_compute_hours'],progress['remaining_authorized_hours'])
    gate=dict(method=method,controls_remaining=controls_left,measured_complete_correct_arm_hours=measured_hours,
        full_controls_time_with_10pct_allowance=controls_left*measured_hours*1.10,
        final_12000_query_reserve_hours=5.,OOF_reserve_hours=1.5,CPU_failure_margin_hours=.5,
        required_remaining_hours=required,actual_tighter_remaining_hours=remaining,passed=remaining>=required,
        total_GPU_cap=40,query_count_per_control=16600,final_test_unlocked=False,
        utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    path=ROOT/'audit'/f'S1_EEG_control_budget_gate_{controls_left}_remaining.json'
    assert not path.exists();path.write_text(json.dumps(gate,indent=2)+'\n')
    return gate


def main():
    started=time.monotonic()
    config_path=ROOT/'configs/full_EEG_control_queue.json';config=json.loads(config_path.read_text())
    for name,expected in config['source_sha256'].items():assert sha(ROOT/name)==expected,(name,'Frozen controls source mismatch')
    deadline=json.loads((ROOT/'configs/compute_time_bounds.json').read_text())['conservative_occupied_time_deadline_epoch']
    launch=json.loads((ROOT/'audit/S1_postprocessing_launch.json').read_text())
    write(dict(status='WAITING_TWO_REFERENCES_AND_COMPLETE_CPU_AUDIT',pid=os.getpid(),GPU_used=False))
    while True:
        p=json.loads((ROOT/'audit/S1_postprocessing_state.json').read_text())
        if p['status']=='COMPLETED_BOTH_FULL_REFERENCES_POSTPROCESSED':break
        assert not p['status'].startswith('FAILED'),p
        assert not (ROOT/'STOP_EEG_CONTROLS').exists(),'Explicit stop file'
        assert time.time()<deadline,'Campaign time bound before full reference completion'
        proc=Path('/proc')/str(launch['pid'])/'cmdline'
        assert proc.exists() and str(ROOT/'queue_reference_postprocessing.py').encode() in proc.read_bytes(), 'Registered prerequisite queue stopped'
        time.sleep(30)
    for name,expected in config['source_sha256'].items():assert sha(ROOT/name)==expected,(name,'Frozen source changed while waiting')
    run('S1_select_full_reference_for_controls',120,'select_reference_for_controls.py',gpu=False)
    selected=json.loads((ROOT/'audit/reference_control_selection.json').read_text())['selected_reference']
    gallery=DATA/'evaluation/gallery/V'
    image_base=DATA/'images/S1_full_native_batchsafe_v2/V';evaluation_base=DATA/'evaluation/V'
    correct=f'{selected}_correct_R4'
    for i,condition in enumerate(['permuted','missing']):
        assert not (ROOT/'STOP_EEG_CONTROLS').exists(),'Explicit stop file before next control'
        for name,expected in config['source_sha256'].items():assert sha(ROOT/name)==expected,(name,'Frozen source changed before control')
        gate=budget_gate(selected,2-i)
        if not gate['passed']:
            write(dict(status='NOT_RUN_REMAINING_CONTROLS_BUDGET_AT_FULL_SCOPE',selected_reference=selected,
                budget_gate=gate,query_subsampling=False,hypothesis_rejected=False,final_test_unlocked=False));return
        arm=f'{selected}_{condition}_R4';images=image_base/arm;features=evaluation_base/arm/'features'
        run(f'S1_V_generate_{selected}_{condition}',18000,'generate_reference_images.py','--method',selected,'--condition',condition)
        run(f'S1_V_evaluate_{selected}_{condition}',10800,'extract_image_features.py','--manifest',images/'manifest.jsonl','--output',features)
        run(f'S1_V_score_{selected}_{condition}',600,'score_full_manifest.py','--manifest',images/'manifest.jsonl','--features',features,
            '--gallery-manifest',gallery/'manifest.jsonl','--gallery-features',gallery/'features','--output',features.parent/'metrics',gpu=False)
        run(f'S1_V_low_level_{selected}_{condition}',3600,'score_low_level.py','--manifest',images/'manifest.jsonl',
            '--gallery-manifest',gallery/'manifest.jsonl','--output',features.parent/'low_level',gpu=False)
        run(f'S1_V_paired_{selected}_{condition}',600,'paired_analysis.py',
            '--reference-manifest',image_base/correct/'manifest.jsonl','--candidate-manifest',images/'manifest.jsonl',
            '--reference-metrics',evaluation_base/correct/'metrics/per_query.csv','--candidate-metrics',features.parent/'metrics/per_query.csv',
            '--output',DATA/'evaluation/paired'/f'V_{selected}_{condition}_minus_correct_v1',gpu=False)
    args=[]
    for arm in ['REF_REG_correct_R4','REF_PRIOR_V_correct_R4',f'{selected}_permuted_R4',f'{selected}_missing_R4']:
        args+=['--arm',arm,image_base/arm/'manifest.jsonl',evaluation_base/arm/'metrics/per_query.csv']
    run('S1_V_complete_references_EEG_controls_gallery',3600,'build_offline_gallery.py',*args,
        '--output',DATA/'galleries/V_references_and_EEG_controls_batchsafe_v1',gpu=False)
    write(dict(status='COMPLETED_FULL_V_EEG_CONTROLS',selected_reference=selected,
        full_query_count_per_arm=16600,EEG_control_arms=2,query_subsampling=False,final_test_unlocked=False,
        new_training_started=False,seconds_including_wait=time.monotonic()-started,
        paired_difference_direction='intervention minus correct; negate mean and reverse/negate CI endpoints for correct minus intervention',
        gallery=str(DATA/'galleries/V_references_and_EEG_controls_batchsafe_v1/index.html')))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        old=json.loads(STATE.read_text()) if STATE.exists() else {}
        write(old|dict(status='FAILED_STOPPED_DEPENDENT_EEG_CONTROL_JOBS',error=str(exc)));raise
