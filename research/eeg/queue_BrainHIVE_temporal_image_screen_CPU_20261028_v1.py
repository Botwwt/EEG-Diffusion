"""Finite D37 all-control 80->400 image/feature/audit queue on the second GPU."""
import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import brainhive_temporal_image_screen_20261028_v1 as U


def run(spec_path,execution_path):
    started=time.monotonic();spec,prepared,_,_=U.prepared(spec_path);execution=U.read(execution_path)
    assert execution['status']=='D37_IMAGE_EXECUTORS_LOCKED_BEFORE_RESEARCH_IMAGES' and execution['spec']['sha256']==U.sha(spec_path)
    for item in execution['source_pins']:U.check(item)
    folder=U.B/'image_queue_v1';assert not folder.exists();folder.mkdir()
    completed=[]
    def stage(label,command,timeout):
        assert time.monotonic()-started<7000
        U.write(folder/(label+'_started.json'),dict(UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),label=label,command=command))
        t=time.monotonic()
        with (folder/(label+'_stdout.log')).open('x') as out,(folder/(label+'_stderr.log')).open('x') as err:
            result=subprocess.run(command,cwd=U.ROOT,stdin=subprocess.DEVNULL,stdout=out,stderr=err,timeout=timeout)
        value=dict(label=label,exit_code=result.returncode,seconds=time.monotonic()-t,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command)
        U.write(folder/(label+'_terminal.json'),value);completed.append(value);assert result.returncode==0,(label,result.returncode)
    def gpu(label,mode,count,start=0,end=80):
        name='D37_'+label+'_20261028_v1'
        command=[sys.executable,'-B',str(U.ROOT/'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'),'--name',name,'--max-seconds','900','--',sys.executable,'-B',str(U.ROOT/'generate_BrainHIVE_temporal_image_screen_GPU_20261028_v1.py'),
                 '--spec',str(spec_path),'--execution',str(execution_path),'--mode',mode,'--stage',str(count),'--query-start',str(start),'--query-end',str(end),'--job-name',name]
        stage(label,command,945)
    def cpu(label,mode,count):
        stage(label,[sys.executable,'-B',str(U.ROOT/'score_BrainHIVE_temporal_image_screen_CPU_20261028_v1.py'),'--spec',str(spec_path),'--stage',str(count),'--mode',mode],180)
    def independent(count):
        stage('independent'+str(count),[sys.executable,'-B',str(U.ROOT/'audit_BrainHIVE_temporal_image_score_CPU_20261028_v1.py'),'--spec',str(spec_path),'--stage',str(count)],180)
    try:
        gpu('images80','generate',80);cpu('image_audit80','audit_images',80);gpu('features80','encode',80);cpu('score80','score',80);independent(80)
        first=U.read(U.B/'screen80/score/completion.json');final=first
        if first['primary_decision']['action']=='EXPAND400':
            for lo in [80,160,240,320]:gpu('images400_'+str(lo)+'_'+str(lo+80),'generate',400,lo,lo+80)
            cpu('image_audit400','audit_images',400);gpu('features400','encode',400);cpu('score400','score',400);independent(400)
            final=U.read(U.B/'screen400/score/completion.json')
        for item in execution['source_pins']:U.check(item)
        value=dict(status='D37_FIXED_THREE_ARM_IMAGE_SCREEN_ACTUALLY_CLOSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            spec=U.ref(spec_path),execution=U.ref(execution_path),completed_steps=completed,final_stage=final['stage'],
            primary_decision=final['primary_decision'],final_score=U.ref(U.B/f'screen{final["stage"]}/score/completion.json'),
            independent_CPU_audit=U.ref(U.B/f'screen{final["stage"]}/score/independent_CPU_audit.json'),
            all_three_candidate_arms_retained=True,old_ordinary_immutable=True,old_D_resumed=False,
            new_research_images=final['stage']*3,additional_learned_parameters=0,seconds=time.monotonic()-started)
        U.write(folder/'completion.json',value);print(json.dumps(value),flush=True)
    except BaseException as error:
        U.write(folder/'failure.json',dict(status='D37_IMAGE_QUEUE_FAILURE_PRESERVED_NO_AUTOMATIC_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),completed_steps=completed,seconds=time.monotonic()-started))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--execution',type=Path,required=True);a=p.parse_args();run(a.spec,a.execution)
