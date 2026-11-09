"""Finite per-host AR image queue. Drafts cannot dispatch; no cross-host SSH."""
import argparse,datetime,json,os,subprocess,time,traceback
from pathlib import Path
import AR_anchored_images_contract_CPU_20261029_v1 as C

def original_process_live(process):
    path=Path('/proc')/str(process['pid'])/'stat'
    if not path.exists():return False
    fields=path.read_text().rsplit(')',1)[1].split()
    return int(fields[19])==process['start_ticks'] and fields[0] not in ('Z','X')
def GPU_idle():return not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()

def run(admission_path):
    a,_=C.load_admission(admission_path);C.validate_sources(a,Path(__file__));C.validate_shared_recipe(a);C.validate_panel(a)
    assigned=a['assigned_arms'];assert assigned==a['generation_assignment'][a['host_role']] and len(assigned)>0
    assert a['score_owner_host']=='old' and 0<a['resources']['maximum_queue_seconds']<=172800
    root=Path(a['output_root']);root.mkdir(exist_ok=True);controllers=root/'controllers';controllers.mkdir(exist_ok=True);out=controllers/a['host_role'];assert not out.exists();out.mkdir()
    C.write_once(out/'registration.json',dict(admission=C.ref(admission_path),assigned_arms=assigned,early_generation='Own fixed final only; no inspection/effect before all3 fixed fits+stage closures',GPU_predecessor_controller=a['GPU_predecessor_controller']))
    began=time.monotonic();events=[];phase='WAIT_OWN_FIXED_FINAL_AND_PREDECESSOR_GPU_QUEUE'
    def guard():assert time.monotonic()-began<a['resources']['maximum_queue_seconds']
    def wait_until(predicate,label):
        nonlocal phase;phase=label
        while not predicate():
            guard();pending=out/'progress.pending';value=dict(status=label,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events_completed=len(events),seconds=time.monotonic()-began,score_reads_by_wait_loop=0)
            pending.write_text(json.dumps(value,indent=2)+'\n');pending.replace(out/'progress.json');time.sleep(5)
    def fixed_ready(arm):
        path=Path(a['fit_completion_paths'][arm])
        if not path.is_file():return False
        C.validate_fit(a,arm);return True
    def execute(args,name,GPU=False):
        guard();env=os.environ.copy();env.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
        if GPU:
            wait_until(lambda:not original_process_live(a['GPU_predecessor_controller']) and GPU_idle(),'WAIT_PREDECESSOR_TERMINAL_AND_GPU_IDLE')
            wrapper='run_gpu_job_v2.py' if a['host_role']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
            command=['/root/miniconda3/bin/python','-B',str(Path(a['repository_root'])/wrapper),'--name',name,'--max-seconds','900','--',*args]
        else:command=args;env['CUDA_VISIBLE_DEVICES']=''
        stdout=out/(name+'.stdout.log');stderr=out/(name+'.stderr.log');assert not stdout.exists() and not stderr.exists();start=time.monotonic()
        with stdout.open('xb') as outstream,stderr.open('xb') as errstream:result=subprocess.run(command,stdin=subprocess.DEVNULL,stdout=outstream,stderr=errstream,env=env,cwd=Path(__file__).parent,timeout=930 if GPU else 310)
        item=dict(name=name,command=command,exit_code=result.returncode,seconds=time.monotonic()-start,stdout=C.ref(stdout),stderr=C.ref(stderr));events.append(item);C.write_once(out/(name+'.json'),item);assert result.returncode==0,result.returncode
    def complete_own(stage):
        for arm in assigned:
            wait_until(lambda:fixed_ready(arm),'WAIT_FIXED_FINAL_'+arm)
            start_index=0 if stage==80 else 80
            for lo in range(start_index,stage,20):
                chunk=root/arm/'generation_chunks'/f'{lo:05d}_{lo+20:05d}';done=chunk/'completion.json'
                assert not (chunk/'failure.json').exists()
                if not done.exists():
                    assert not chunk.exists(),'Unclosed prior chunk requires evidence-preserving explicit recovery'
                    job=f'AR_anchored_images_{arm}_gen_{lo:03d}_{lo+20:03d}_v1'
                    command=['/root/miniconda3/bin/python','-B',str(Path(__file__).parent/'AR_anchored_images_generate_GPU_20261029_v1.py'),'--admission',str(admission_path),'--arm',arm,'--stage',str(stage),'--lo',str(lo),'--hi',str(lo+20),'--job-name',job]
                    execute(command,job,True)
                value=json.loads(done.read_bytes());assert value['status']=='COMPLETE_FIXED_AR_EEG_ONLY_IMAGE_CHUNK' and value['query_range']==[lo,lo+20]
            audit=root/arm/'artifacts'/f'stage{stage}.json'
            if not audit.exists():execute(['/root/miniconda3/bin/python','-B',str(Path(__file__).parent/'AR_anchored_images_artifacts_CPU_20261029_v1.py'),'--admission',str(admission_path),'--arm',arm,'--stage',str(stage)],f'AR_anchored_images_{arm}_artifact{stage}_v1')
            assert json.loads(audit.read_bytes())['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT'
            for lo in range(start_index,stage,20):
                chunk=root/arm/'feature_chunks'/f'{lo:05d}_{lo+20:05d}';done=chunk/'completion.json';assert not (chunk/'failure.json').exists()
                if not done.exists():
                    assert not chunk.exists(),'Unclosed feature chunk requires explicit recovery'
                    job=f'AR_anchored_images_{arm}_DINO_{lo:03d}_{lo+20:03d}_v1'
                    execute(['/root/miniconda3/bin/python','-B',str(Path(__file__).parent/'AR_anchored_images_features_GPU_20261029_v1.py'),'--admission',str(admission_path),'--arm',arm,'--stage',str(stage),'--lo',str(lo),'--hi',str(lo+20),'--job-name',job],job,True)
                assert json.loads(done.read_bytes())['status']=='COMPLETE_FIXED_AR_FROZEN_DINO_FEATURE_CHUNK'
        C.write_once(out/f'own_stage{stage}_closure.json',dict(status='OWN_HOST_ALL_ASSIGNED_AR_IMAGES_ARTIFACTS_FEATURES_COMPLETE_NO_EFFECT_READ',stage=stage,arms=assigned,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),cross_host_collection_external=True))
    def all3_boundary_ready(stage):
        for arm in C.ARMS:
            if not fixed_ready(arm):return False
            if not (root/arm/'artifacts'/f'stage{stage}.json').is_file():return False
            if any(not (root/arm/'feature_chunks'/f'{lo:05d}_{lo+20:05d}/completion.json').is_file() for lo in range(0,stage,20)):return False
        return True
    try:
        for stage in (80,400):
            if stage==400:C.validate_expansion(a,stage)
            complete_own(stage)
            summary=root/'screens'/f'stage{stage}/summary.json'
            if a['host_role']==a['score_owner_host']:
                wait_until(lambda:all3_boundary_ready(stage),'WAIT_ALL_THREE_FIXED_FINALS_AND_FULL_STAGE_ARTIFACTS_FEATURES')
                execute(['/root/miniconda3/bin/python','-B',str(Path(__file__).parent/'AR_anchored_images_score_CPU_20261029_v1.py'),'--admission',str(admission_path),'--stage',str(stage)],f'AR_anchored_images_all3_score{stage}_v1')
            else:wait_until(summary.is_file,'WAIT_EXACT_OLD_SCORE_OWNER_DECISION_COPY_NO_LOCAL_EFFECT_READ')
            decision=json.loads(summary.read_bytes());assert decision['status']=='COMPLETE_FIXED_THREE_ARM_AR_IMAGE_SCREEN' and decision['stage']==stage and decision['policy']['sha256']==C.POLICY_SHA
            if stage==80 and decision['decision']['action']=='STOP_INSTANCE':break
            if stage==80:assert decision['decision']['action']=='EXPAND400'
        C.write_once(out/'completion.json',dict(status='COMPLETE_FINITE_AR_IMAGE_LIFECYCLE_ASSIGNED_HOST',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),events=events,seconds=time.monotonic()-began,new_fitted_models=0,source_policy_unchanged=True))
    except BaseException as error:
        C.write_once(out/'failure.json',dict(status='AR_IMAGE_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY_OR_RECIPE_CHANGE',phase=phase,error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-began));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--admission',type=Path,required=True);q=p.parse_args();run(q.admission)
