"""Wait on actual existing finite study tails; admit one new GPU only after full original analysis."""
import argparse,datetime,json,os,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);args=ap.parse_args();assert 1<=args.hop<=100
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_anchor_dispatch_v287';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,v):
    b=(json.dumps(v,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def submit(name,dependency,arguments):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/name)]+[str(v) for v in arguments],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.returncode,p.stdout,p.stderr);return job
def latest_stage(folder):
    rows=[read(p) for p in folder.glob('stage_*.json')];assert rows;return max(rows,key=lambda v:v['stage'])
try:
    assert read(R/'runs/EEG_anchor_engineering_v286/completion.json')['status']=='PASS_REAL_EEG_NATIVE_IDENTITY_GRADIENT_AND_GLOBAL_CAP_ALGEBRA'
    assert not (out/'submitted.json').exists() and not (out/('hop_'+str(args.hop)+'.json')).exists()
    train=R/'runs/EEG_candidates_v267/queue';image=R/'runs/EEG_images_v271';score=R/'runs/EEG_scores_v279';r=dict(hop=args.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for p in [train,image,image/'dispatch_v277',R/'runs/EEG_scoring_dispatch_v281',score]:assert not list(p.glob('*failure_*.json'))
    for p in [train,image,score]:assert not (p/'cap_stop.json').exists()
    if (score/'analysis_completion.json').exists():
        proof=read(score/'analysis_completion.json');assert proof['status']=='COMPLETE_ALL830_14ARMS_DINO_PAPER7_EFFNET_PAIRED_ANALYSIS_SEED_PENDING'
        # No hypothesis selection uses the metric values; only actual closure/source integrity admit this registered study.
        profile=submit('PKU_profile_anchor_GPU_20261109_v286.sbatch',jid,[])
        start=submit('PKU_start_anchor_CPU_20261109_v287.sbatch',profile,[])
        r.update(status='ORIGINAL_FIXED_STUDY_CLOSED_NEW_ANCHOR_PREFIX_AND_CPU_GATE_SUBMITTED',profile_job=profile,continuation_gate_job=start,new_cases=3,updates_each=827,new_images_yet=0)
        write(out/'submitted.json',r)
    else:
        assert args.hop<100
        if not (train/'completion.json').exists():p=latest_stage(train);dependency=p['next_job_id'];phase='original_training'
        elif not (image/'completion.json').exists():
            if list((image/'stages').glob('stage_*.json')):p=latest_stage(image/'stages');dependency=p['next_job_id']
            elif (image/'image_queue_submitted.json').exists():dependency=read(image/'image_queue_submitted.json')['image_job']
            else:dependency=read(image/'dispatch_v277/progress.json')['next_dispatch_job']
            phase='original_images'
        elif (score/'scoring_completion.json').exists():
            p=latest_stage(score/'stages');dependency=p['analysis_job'];phase='original_CPU_analysis'
        elif list((score/'stages').glob('stage_*.json')):p=latest_stage(score/'stages');dependency=p['next_job_id'];phase='original_metrics'
        elif (R/'runs/EEG_scoring_dispatch_v281/submitted.json').exists():dependency=read(R/'runs/EEG_scoring_dispatch_v281/submitted.json')['scoring_job'];phase='original_first_metric'
        else:dependency=read(R/'runs/EEG_scoring_dispatch_v281/progress.json')['next_dispatch_job'];phase='original_score_dispatch'
        assert dependency.isdigit() and dependency!=jid
        job=submit('PKU_dispatch_anchor_CPU_20261109_v287.sbatch',dependency,[args.hop+1]);r.update(status='FOLLOWING_ACTUAL_ORIGINAL_STUDY_TAIL_NO_GPU_RESERVED',phase=phase,dependency=dependency,next_dispatch_job=job)
    write(out/('hop_'+str(args.hop)+'.json'),r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
