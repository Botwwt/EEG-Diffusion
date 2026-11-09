"""Follow actual training/image tails with bounded CPU dependencies, then audit and score."""
import argparse,datetime,json,os,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);args=ap.parse_args();assert 1<=args.hop<=65
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_downstream_scoring_dispatch_v303';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def write(p,r):
    b=(json.dumps(r,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def submit(stem,dependency,arguments):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in arguments],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.returncode,p.stdout,p.stderr);return job
try:
    path=out/('hop_'+str(args.hop)+'.json');assert not path.exists() and not (out/'submitted.json').exists()
    image=R/'runs/EEG_downstream_images_v303';r=dict(hop=args.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert not list(image.glob('failure_*.json')) and not (image/'cap_stop.json').exists()
    if (image/'completion.json').exists():
        assert json.loads((image/'completion.json').read_text())['images']==7470
        prep=submit('PKU_prepare_downstream_scoring_CPU_20261109_v303.sbatch',jid,[]);score=submit('PKU_score_downstream_images_GPU_20261109_v303.sbatch',prep,[1,'DINO']);r.update(status='ALL7470_IMAGES_COMPLETE_FULL_AUDIT_AND_SCORING_SUBMITTED',preparation_job=prep,scoring_job=score);write(out/'submitted.json',r)
    else:
        assert args.hop<65;st=max([json.loads(p.read_text()) for p in (image/'stages').glob('stage_*.json')],key=lambda v:v['stage']);dependency=st['next_job_id'];job=submit('PKU_dispatch_downstream_scoring_CPU_20261109_v303.sbatch',dependency,[args.hop+1]);r.update(status='FOLLOWING_ACTUAL_IMAGE_PREFIX',dependency=dependency,next_dispatch_job=job)
    write(path,r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
