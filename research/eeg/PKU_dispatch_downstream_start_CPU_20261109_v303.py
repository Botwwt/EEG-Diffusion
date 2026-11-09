"""Follow immutable actual predecessor stages, no GPU waiting reservation."""
import argparse,datetime,json,os,socket,subprocess,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);a=ap.parse_args();assert 1<=a.hop<=160
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_downstream_start_v303';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,v):
    t=p.with_name(p.name+'.pending')
    with t.open('x') as f:json.dump(v,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
    t.replace(p)
def closed(folder):return max([read(p) for p in folder.glob('stage_*.json')],key=lambda v:v['stage'])
def submit(stem,dep,args):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.stdout,p.stderr);return job
try:
    assert not (out/'submitted.json').exists() and not (out/('hop_'+str(a.hop)+'.json')).exists()
    image=R/'runs/EEG_anchor_images_v288';score=R/'runs/EEG_anchor_scores_v288';train=R/'runs/EEG_anchor_v286';closure=R/'runs/EEG_anchor_image_dispatch_v294'
    for d in [image,score,train/'queue',closure]:assert not list(d.glob('failure_*.json')) and not (d/'cap_stop.json').exists()
    r=dict(hop=a.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if (score/'analysis_completion.json').exists():
        assert read(score/'analysis_completion.json')['status'].startswith('COMPLETE_ALL830_20ARMS_');assert (R/'runs/EEG_downstream_engineering_v303/completion.json').exists()
        job=submit('PKU_queue_downstream_GPU_20261109_v303.sbatch',jid,[1,'P_positive']);r.update(status='NINE_CASE_DOWNSTREAM_STUDY_SUBMITTED_AFTER_PREVIOUS_FULL_ANALYSIS',training_job=job);write(out/'submitted.json',r)
    else:
        assert a.hop<160
        if (score/'scoring_completion.json').exists():dep=closed(score/'stages')['analysis_job']
        elif list((score/'stages').glob('stage_*.json')):dep=closed(score/'stages')['next_job_id']
        elif (closure/'scoring_submitted.json').exists():dep=read(closure/'scoring_submitted.json')['scoring_job']
        elif list((image/'stages').glob('stage_*.json')):
            st=closed(image/'stages');dep=read(closure/'progress.json')['next_dispatch_job'] if (image/'completion.json').exists() else st['next_job_id']
        else:dep=read(closure/'progress.json')['next_dispatch_job']
        assert dep.isdigit() and dep!=jid;job=submit('PKU_dispatch_downstream_start_CPU_20261109_v303.sbatch',dep,[a.hop+1]);r.update(status='FOLLOWING_PREVIOUS_SERIAL_CHAIN_GPU_NOT_RESERVED',dependency=dep,next_dispatch_job=job)
    write(out/('hop_'+str(a.hop)+'.json'),r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
