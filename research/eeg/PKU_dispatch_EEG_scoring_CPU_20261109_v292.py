"""Follow actual training/image tails with bounded CPU dependencies, then audit and score."""
import argparse,datetime,json,os,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);args=ap.parse_args();assert 1<=args.hop<=65
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_scoring_dispatch_v292';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def write(p,r):
    b=(json.dumps(r,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def submit(stem,dependency,arguments):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in arguments],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.returncode,p.stdout,p.stderr);return job
try:
    path=out/('hop_'+str(args.hop)+'.json');assert not path.exists() and not (out/'submitted.json').exists()
    train=R/'runs/EEG_candidates_v267/queue';image=R/'runs/EEG_images_v271';r=dict(hop=args.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    assert not list(train.glob('failure_*.json')) and not (train/'cap_stop.json').exists() and not [p for p in image.glob('failure_*.json') if not (p.name=='failure_5099103.json' and (image/'device_recovery_v290.json').exists())] and not (image/'cap_stop.json').exists() and not list((image/'dispatch_v277').glob('failure_*.json'))
    assert json.loads((R/'runs/EEG_metric_core_v279/completion.json').read_text())['status']=='PASS_SOURCE_DIRECTION_TIES_EXPLICIT_GALLERY_AND_SSIM_FIXTURES'
    if (image/'completion.json').exists():
        assert json.loads((image/'completion.json').read_text())['status']=='COMPLETE_ALL830_QUERIES_14_ARMS_IMAGES_SCORES_PENDING'
        prep=submit('PKU_prepare_EEG_scoring_CPU_20261109_v281.sbatch',jid,[]);score=submit('PKU_score_EEG_images_GPU_20261109_v279.sbatch',prep,[1,'DINO'])
        r.update(status='ALL11620_IMAGES_COMPLETE_FULL_CPU_AUDIT_AND_GPU_SCORING_SUBMITTED',preparation_job=prep,scoring_job=score);write(out/'submitted.json',r)
    else:
        assert args.hop<65
        if not (train/'completion.json').exists():
            closed=[json.loads(p.read_text()) for p in train.glob('stage_*.json')];p=max(closed,key=lambda v:v['stage']);assert p.get('next_job_id');dependency=p['next_job_id'];phase='training'
        elif list((image/'stages').glob('stage_*.json')):
            closed=[json.loads(p.read_text()) for p in (image/'stages').glob('stage_*.json')];p=max(closed,key=lambda v:v['stage']);dependency=p['next_job_id'];phase='images'
        elif (image/'image_queue_submitted.json').exists():
            p=json.loads((image/'image_queue_submitted.json').read_text());dependency=json.loads((image/'device_recovery_v290.json').read_text())['job_id'] if (image/'device_recovery_v290.json').exists() else p['image_job'];phase='first_image_job'
        else:
            p=json.loads((image/'dispatch_v277/progress.json').read_text());dependency=p['next_dispatch_job'];phase='image_dispatch'
        assert dependency.isdigit() and dependency!=jid
        next_job=submit('PKU_dispatch_EEG_scoring_CPU_20261109_v292.sbatch',dependency,[args.hop+1]);r.update(status='FOLLOWING_ACTUAL_TRAINING_OR_IMAGE_TAIL',phase=phase,dependency=dependency,next_dispatch_job=next_job)
    write(path,r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',hop=args.hop,error=repr(e),traceback=traceback.format_exc()));raise
