"""Finite own-final image waiter; no effect reads or training restarts."""
import argparse,datetime,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def write(p,x):
    with p.open('x') as f:json.dump(x,f,indent=2);f.write('\n')
def run(path,stage):
    s=json.loads(Path(path).read_bytes());root=Path(s['queue_root'])/f'stage{stage}';root.mkdir(parents=True);started=time.monotonic();events=[]
    try:
        assert s['status']=='REGISTERED_FINITE_NATIVE_MARGINAL_IMAGE_WAITER_v2' and stage in (80,400)
        for item in s['source_pins']:assert ref(Path(item['path']))==item
        fit=Path(s['fit_root']);training_queue=Path(s['training_queue_root'])
        while not (fit/'completion.json').is_file() or not (training_queue/'completion.json').is_file():
            assert time.monotonic()-started<3600,'Finite wait for original fixed fit exhausted'
            assert not (training_queue/'failure.json').is_file(),'Original fit failed; never restart it automatically'
            launch=s['training_controller'];p=Path('/proc')/str(launch['controller_PID'])/'stat'
            assert p.is_file(),'Original controller missing before closure; inspect it, do not launch replacement'
            fields=p.read_text().rsplit(')',1)[1].split();assert int(fields[19])==launch['start_ticks'] and fields[0] not in ('Z','X')
            time.sleep(5)
        completed=json.loads((training_queue/'completion.json').read_bytes());assert completed['status']=='NEW_ROUTE_FULL_T_FIT_AND_INDEPENDENT_CPU_AUDIT_CLOSED'
        done=json.loads((fit/'completion.json').read_bytes());assert done['epoch']==10 and done['EEG_updates']==2070
        profile_spec=json.loads(Path(s['fit_profile_spec']['path']).read_bytes())
        image_spec=dict(s['image_template'],fit_completion=ref(fit/'completion.json'),fit_CPU_audit=ref(training_queue/'independent_CPU_audit.json'))
        if stage==400:image_spec['stage80_decision']=s['stage80_decision']
        image_path=Path(s['code'])/f"images_{s['host']}_stage{stage}_spec_v2.json";write(image_path,image_spec)
        for lo in range(0 if stage==80 else 80,stage,20):
            assert time.monotonic()-started<7200
            assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU occupied: no competing dispatch'
            job=f"EEG_native_marginal_{s['host']}_images{lo:03d}_20261029_v2";wrapper='run_gpu_job_v2.py' if s['host']=='old' else 'run_BrainHIVE_two_host_GPU_job_20261028_v1.py'
            command=['/root/miniconda3/bin/python','-B',str(Path(s['repository_root'])/wrapper),'--name',job,'--max-seconds','900','--','/root/miniconda3/bin/python','-B',str(Path(s['code'])/'generate_EEG_native_marginal_images_GPU_20261029_v2.py'),'--spec',str(image_path),'--lo',str(lo),'--hi',str(lo+20),'--job-name',job]
            with (root/(job+'.stdout.log')).open('xb') as so,(root/(job+'.stderr.log')).open('xb') as se:
                p=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=so,stderr=se,start_new_session=True,cwd=s['code']);rc=p.wait(timeout=950)
            event=dict(job=job,returncode=rc,range=[lo,lo+20]);events.append(event);assert rc==0,event
            chunk=Path(image_spec['output_root'])/f'chunk_{lo:04d}_{lo+20:04d}'/'completion.json';result=json.loads(chunk.read_bytes())
            assert result['status']=='COMPLETE_NATIVE_MARGINAL_FIXED_FINAL_TWO_ARM_IMAGE_FEATURE_CHUNK' and result['range']==[lo,lo+20]
            assert result['spec']==ref(image_path) and not result['effect_scores_read'] and result['query_target_images_read']==0
        write(root/'completion.json',dict(status='COMPLETE_NEW_ROUTE_FIXED_FINAL_STAGE_IMAGES_AND_DINO_FEATURES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=s['host'],stage=stage,arms=image_spec['arms'],image_spec=ref(image_path),fit_completion=image_spec['fit_completion'],events=events,seconds=time.monotonic()-started,effect_scores_read=False))
    except BaseException as error:
        write(root/'failure.json',dict(status='NEW_ROUTE_IMAGE_WAITER_FAILURE_PRESERVED_NO_RETRY',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),traceback=traceback.format_exc(),events=events,seconds=time.monotonic()-started));raise
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--spec',required=True);ap.add_argument('--stage',type=int,required=True);v=ap.parse_args();run(v.spec,v.stage)
