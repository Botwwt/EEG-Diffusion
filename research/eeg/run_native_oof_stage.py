"""Finite selected-chain OOF -> exact cache -> CPU Dcheck execution.

Only actual resolved/funded configs are accepted. No automatic retries, adapter
training, benchmark unlock, seed search or partial-cache scientific results.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
PY='/root/miniconda3/bin/python'


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def load(path):return json.loads(Path(path).read_text())


def write(path,data):
    data['last_update_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');os.replace(temporary,path)


def fresh_remaining():
    subprocess.run([PY,str(ROOT/'refresh_status.py')],cwd=ROOT,check=True,stdout=subprocess.DEVNULL)
    p=load(ROOT/'audit/current_progress.json')
    assert not any(j['alive'] for j in p['live_GPU_jobs']),'Do not compete with another registered GPU job'
    assert not p['final_test_unlocked'],'This stage precedes benchmark unlock'
    return p['remaining_authorized_hours']*3600


def stop_group(process):
    """Stop only the new session created for this owned phase, never other jobs."""
    if process.poll() is not None:return
    try:
        assert os.getpgid(process.pid)==process.pid
        os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        process.wait(timeout=10)


def recover_gpu_receipt(name,code,elapsed):
    receipt=ROOT/'audit'/f'{name}_job.json';ledger=ROOT/'audit/compute_ledger.jsonl'
    if not receipt.exists():return  # Wrapper may have refused before launching.
    recorded=[row for row in map(json.loads,ledger.read_text().splitlines()) if row.get('name')==name]
    if recorded:return
    old=load(receipt)
    record=old|dict(name=name,status='SUPERVISOR_STOPPED_RESUMABLE',exit_code=code,
        charged_seconds=elapsed,termination_reason='Owned phase stopped by controller; preserve outputs',
        end_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        accounting='Once; wrapper had no terminal ledger entry')
    write(receipt,record)
    with ledger.open('a') as file:file.write(json.dumps(record)+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resolved',required=True);args=parser.parse_args()
    resolved_path=Path(args.resolved);r=load(resolved_path)
    assert r['status']=='RESOLVED_AND_FUNDED_BUT_NOT_STARTED' and r['selected_reference'] in ['REF_REG','REF_PRIOR_V']
    policy=load(r['policy_path']);assert sha(r['policy_path'])==r['policy_sha256']
    for path_key,digest_key in [('gate_path','gate_sha256'),('training_config_path','training_config_sha256'),
        ('cache_config_path','cache_config_sha256'),('validation_plan_path','validation_plan_sha256')]:
        assert sha(r[path_key])==r[digest_key],path_key
    assert load(r['gate_path'])['full_two_fold_training_and_condition_sampling_funded']
    config=load(r['training_config_path']);cache_config=load(r['cache_config_path'])
    assert config['condition_chain']==cache_config['condition_chain']==r['selected_reference']
    assert config['seed']==cache_config['seed']==1729
    assert not r['S4_may_start'] and not r['final_test_may_unlock']
    state_path=ROOT/'audit'/f'{r["run_id"]}_state.json'
    assert not state_path.exists(),'Inspect prior state/outputs before a separately named explicit retry'
    stages=[('oof0','train_native_oof.py',['--config',r['training_config_path'],'--role','oof0']),
            ('oof1','train_native_oof.py',['--config',r['training_config_path'],'--role','oof1']),
            ('cache','build_native_oof_cache.py',['--config',r['cache_config_path']]),
            ('validation','validate_native_error_channel.py',['--plan',r['validation_plan_path'],
                '--cache-config',r['cache_config_path'],'--output',r['validation_output']])]
    state=dict(status='PREPARED_NOT_STARTED',run_id=r['run_id'],controller_pid=os.getpid(),
        resolved_path=str(resolved_path),resolved_sha256=sha(resolved_path),completed_stages=[],
        output_root=config['output_root'],cache_output_root=cache_config['condition_cache_output_root'],
        validation_output=r['validation_output'],final_test_unlocked=False,adapter_training_started=False,
        source_sha256=sha(__file__))
    write(state_path,state)
    process=None;gpu=False;name=None;tick=None
    try:
        for index,(stage,script,arguments) in enumerate(stages):
            for name,digest in policy['source_sha256'].items():assert sha(ROOT/name)==digest,('Frozen stage source changed',name)
            assert not (ROOT/'STOP_S3_NATIVE_OOF').exists(),'Explicit stage stop flag'
            outstanding=sum(r['job_max_seconds'][s[0]] for s in stages[index:])
            required=outstanding+r['retained_full_controls_final_margin_seconds']
            remaining=fresh_remaining()
            if remaining<required:
                state.update(status='STOPPED_BEFORE_NEXT_PHASE_COMPLETE_SCOPE_BUDGET_NOT_AVAILABLE',
                    next_phase=stage,required_remaining_seconds=required,actual_remaining_seconds=remaining,
                    hypothesis_rejected=False)
                write(state_path,state);print(json.dumps(state),flush=True);return
            command=[PY,str(ROOT/script),*arguments];gpu=r['stage_GPU_flags'][stage]
            name=f'{r["run_id"]}_{stage}'
            if gpu:command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(r['job_max_seconds'][stage]),'--',*command]
            log=ROOT/'logs'/f'{name}.log';assert not log.exists()
            env=os.environ.copy()
            if not gpu:env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
            tick=time.monotonic()
            with log.open('xb') as file:
                process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='RUNNING',stage=stage,phase_supervisor_pid=process.pid,command=command,
                    log=str(log),GPU_used=gpu,phase_started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                    required_remaining_seconds_at_phase_start=required,actual_remaining_seconds_at_phase_start=remaining)
                write(state_path,state)
                interrupted=False
                while process.poll() is None:
                    if (ROOT/'STOP_S3_NATIVE_OOF').exists() or time.monotonic()-tick>r['job_max_seconds'][stage]+90:
                        interrupted=True;stop_group(process);break
                    time.sleep(5)
                code=process.wait()
            elapsed=time.monotonic()-tick
            event=dict(stage=name,command=command,seconds=elapsed,exit_code=code,GPU_used=gpu,
                controller_interrupted=interrupted,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            if not gpu:
                with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:file.write(json.dumps(event)+'\n')
            elif interrupted:
                # The process-group shutdown can stop the wrapper before its
                # final receipt. Retain its actual wall time once, never lose or
                # double-charge the owned GPU allocation.
                recover_gpu_receipt(name,code,elapsed)
            state.setdefault('phase_receipts',[]).append(event)
            if code!=0 or interrupted:
                state.update(status='FAILED_OR_BUDGET_STOPPED_WITH_SAVED_OUTPUTS',failed_phase=stage,
                    exit_code=code,hypothesis_rejected=False,automatic_retry=False)
                write(state_path,state);raise RuntimeError(f'{stage} failed/stopped; inspect {log}; no dependent phase will run')
            state['completed_stages'].append(stage)
            state.update(status='PHASE_COMPLETED',stage=stage);write(state_path,state)
        cache_summary=load(Path(cache_config['condition_cache_output_root'])/'completion.json')
        report_path=Path(r['validation_output'])/'summary.json';report=load(report_path)
        assert cache_summary['status']=='COMPLETED_ALL10_CONCEPT_OOF_T_AND_MATCHED_D_CHAINS'
        assert cache_summary['T_OOF_queries']==132300 and cache_summary['D_queries_per_teacher']==16500
        assert report['subjects']==10 and report['Dcheck_queries']==8300 and report['Dfit_queries']==8200
        assert not report['gate']['S4_may_start'] and report['images_generated']==0 and not report['GPU_initialized']
        state.update(status='COMPLETED_ALL10_OOF_CACHE_AND_CPU_DCHECK',
            CPU_validation_status=report['status'],CPU_validation_summary_path=str(report_path),
            CPU_validation_summary_sha256=sha(report_path),full_UNet_response_validated=False,
            formal_EEG_image_gain_established=False,S4_may_start=False,final_test_unlocked=False)
        write(state_path,state);print(json.dumps(state),flush=True)
    except Exception as exc:
        if process is not None and process.poll() is None:
            stop_group(process)
            if gpu:recover_gpu_receipt(name,process.returncode,time.monotonic()-tick)
        if state['status'] not in ['FAILED_OR_BUDGET_STOPPED_WITH_SAVED_OUTPUTS']:
            state.update(status='FAILED_STOPPED_DEPENDENT_PHASES',error=str(exc),hypothesis_rejected=False)
            write(state_path,state)
        raise


if __name__=='__main__':main()
