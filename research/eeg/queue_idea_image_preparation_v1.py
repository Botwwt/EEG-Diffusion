"""Finite automatic continuation when real OOF GPU teachers have finished.

Runs one T-only cost measurement, complete T latent preparation, and (only if
the predeclared CPU gate passes) the full Dcheck UNet response diagnostic.
It cannot train an interface, sample formal V/test images, or unlock benchmark.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import time
from run_native_oof_stage import stop_group,recover_gpu_receipt

ROOT=Path(__file__).resolve().parent
PY='/root/miniconda3/bin/python'
STATE=ROOT/'audit/idea_image_preparation_queue_state_v1.json'


def load(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:return hashlib.file_digest(file,'sha256').hexdigest()


def write(value):
    value['UTC']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    temp=STATE.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');os.replace(temp,STATE)


def remaining():
    ledger=ROOT/'audit/compute_ledger.jsonl'
    charged=sum(float(x.get('charged_seconds',0)) for x in map(json.loads,ledger.read_text().splitlines()))
    deadline=load(ROOT/'configs/compute_time_bounds.json')['conservative_occupied_time_deadline_epoch']
    return min(40*3600-charged,deadline-time.time())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);args=parser.parse_args()
    path=Path(args.plan);plan=load(path)
    assert plan['version']=='idea_image_preparation_queue_v1' and plan['one_shot']
    assert not STATE.exists(),'Inspect prior actual execution before a separately named recovery'
    state=dict(status='WAITING_REAL_COMPLETE_OOF_GPU_TEACHERS',controller_pid=os.getpid(),plan_sha256=sha(path),
        phase_receipts=[],adapter_training_started=False,formal_images_generated=False,final_test_unlocked=False)
    write(state)
    def guard():
        assert not (ROOT/'STOP_IDEA_IMAGE_PREPARATION_V1').exists(),'Explicit preparation stop flag'
        for name,digest in plan['source_sha256'].items():assert sha(ROOT/name)==digest,name
        assert not (ROOT/'configs/FINAL_EVAL_UNLOCK.json').exists()
    def current():
        x=load(ROOT/'audit/idea_first_prerequisite_state_v1.json')
        if x['status'].startswith('FAILED'):return x
        if x['status'].startswith('COMPLETED_'):return x
        p=Path('/proc')/str(x['controller_pid'])/'cmdline'
        assert p.exists() and str(ROOT/'run_idea_first_prerequisites_v1.py').encode() in p.read_bytes(),'Specific prerequisite controller missing; no blind restart'
        return x
    started=time.monotonic();process=None;name=None;tick=None
    try:
        guard()
        while True:
            guard();x=current()
            if x['status'].startswith('FAILED') and x['stage'] not in ['cache','Dcheck']:
                raise RuntimeError('Actual teacher prerequisite failed before complete roles; preserve and inspect')
            if x['stage'] in ['cache','Dcheck'] or x['status'].startswith('COMPLETED_'):
                cfg=load(ROOT/'configs/idea_first_ATMS_training_v1.json');base=Path(cfg['output_root'])
                assert all(load(base/role/'completion.json')['status']=='COMPLETED'
                           and len(load(base/role/'completion.json')['subjects'])==10 for role in ['full','oof0','oof1'])
                assert not x.get('GPU_used',False) or x['status'].startswith('COMPLETED_')
                break
            assert time.monotonic()-started<7200 and remaining()>3600
            time.sleep(20)
        def run(phase,script,arguments,cap):
            nonlocal process,name,tick
            guard()
            if remaining()<cap+900:
                state['phase_receipts'].append(dict(phase=phase,status='NOT_RUN_CURRENT_BUDGET',hypothesis_rejected=False));write(state);return False
            name='IDEA1_prepare_'+phase;command=[PY,str(ROOT/'run_gpu_job.py'),'--name',name,'--max-seconds',str(cap),'--',PY,str(ROOT/script),*map(str,arguments)]
            log=ROOT/'logs'/f'{name}.log';assert not log.exists()
            tick=time.monotonic()
            with log.open('xb') as file:
                process=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,stdout=file,stderr=subprocess.STDOUT,start_new_session=True)
                state.update(status='RUNNING',phase=phase,phase_supervisor_pid=process.pid,command=command,log=str(log));write(state)
                while process.poll() is None:
                    if (ROOT/'STOP_IDEA_IMAGE_PREPARATION_V1').exists() or time.monotonic()-tick>cap+50:
                        stop_group(process);recover_gpu_receipt(name,process.returncode,time.monotonic()-tick);break
                    time.sleep(5)
                code=process.wait()
            state['phase_receipts'].append(dict(phase=phase,exit_code=code,seconds=time.monotonic()-tick,log=str(log),
                status='COMPLETED' if code==0 else 'FAILED_OR_RESOURCE_STOPPED_NOT_SCIENTIFIC_REJECTION'))
            if code!=0:recover_gpu_receipt(name,code,time.monotonic()-tick)
            write(state);return code==0
        # Independent phases continue after an engineering failure; there are
        # no automatic parameter revisions/retries and no failed result is zero.
        run('batch_cost','profile_idea_first_batches_v1.py',['--plan',ROOT/'configs/idea_first_batch_cost_probe_v1.json'],360)
        run('T_latents','build_idea_training_latents_v1.py',['--plan',ROOT/'configs/idea_interface_training_recipe_v1.json'],900)
        state.update(status='WAITING_REAL_COMPLETE_NUMERICAL_DCHECK',phase='CPU_Dcheck_prerequisite');write(state)
        while True:
            guard();x=current()
            if x['status'].startswith('FAILED'):
                state.update(status='PREPARATION_COMPLETE_UNET_NOT_RUN_NUMERICAL_STAGE_INCOMPLETE',
                    actual_prerequisite_error=x.get('error'),hypothesis_rejected=False,
                    next_action='Inspect/resume actual failed cache or numerical validation; retain independent preparation outputs')
                write(state);return
            if x['status']=='COMPLETED_REAL_ALL10_OOF_AND_INDEPENDENT_DCHECK_IMAGE_STAGE_REVIEW_REQUIRED':break
            assert remaining()>1800
            time.sleep(20)
        validation=Path(plan['validation_output_root']);summary=load(validation/'summary.json')
        assert summary['subjects']==10 and summary['Dcheck_queries']==8300
        gate=summary['gate']
        if not (gate['native_energy_CI_upper_below_zero_vs_Iso_and_Moment'] and gate['projected_energy_CI_upper_below_zero_vs_Iso_and_Moment']):
            state.update(status='PREPARATION_COMPLETE_UNET_NOT_RUN_CPU_PREREQUISITE_FAILED',
                numerical_validation_summary=str(validation/'summary.json'),numerical_validation_sha256=sha(validation/'summary.json'),
                fixed_error_model_CPU_gate_passed=False,all_EEG_ideas_rejected=False,
                next_action='Inspect actual fixed-channel failure. RealPaired/Clean image controls do not depend on Conditional fidelity.')
            write(state);return
        cache_path=ROOT/'configs/idea_first_native_cache_v1.json'
        success=run('full_UNet_response','diagnose_idea_first_unet_response_v1.py',
            ['--plan',ROOT/'configs/idea_first_unet_response_plan_v1.json','--cache-config',cache_path,
             '--validation-output',validation,'--output',plan['UNet_response_output_root']],5400)
        state.update(status='IMAGE_PREPARATION_TERMINAL_REVIEW_ACTUAL_RECEIPTS',
            numerical_validation_summary=str(validation/'summary.json'),UNet_phase_completed=success,
            next_action='Inspect full UNet response and actual complete lifecycle costs; implement/run all funded matched image arms. No image gain from diagnostics.')
        write(state)
    except Exception as error:
        if process is not None and process.poll() is None:
            stop_group(process);recover_gpu_receipt(name,process.returncode,time.monotonic()-tick)
        state.update(status='FAILED_OR_STOPPED_WITH_OUTPUTS_PRESERVED',error=repr(error),automatic_retry=False,hypothesis_rejected=False)
        write(state);raise


if __name__=='__main__':main()
