"""Finite closure for registered anchored fits, six new image arms and20arm cached comparison."""
import argparse,datetime,json,os,socket,subprocess,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);args=ap.parse_args();assert 1<=args.hop<=125
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_anchor_image_dispatch_v294';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def read(p):return json.loads(p.read_text()) if p.exists() else None
def write(p,r):
    b=(json.dumps(r,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def submit(stem,dependency,arguments):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in arguments],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    jid=p.stdout.strip().split(';')[0];assert p.returncode==0 and jid.isdigit(),(p.returncode,p.stdout,p.stderr);return jid
def closed(folder):return max([read(p) for p in folder.glob('stage_*.json')],key=lambda v:v['stage'])
try:
    path=out/('hop_'+str(args.hop)+'.json');assert not path.exists() and not (out/'scoring_submitted.json').exists()
    train=R/'runs/EEG_anchor_v286';image=R/'runs/EEG_anchor_images_v288';r=dict(hop=args.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    for folder in [train,train/'queue',image,R/'runs/EEG_anchor_dispatch_v293']:assert not list(folder.glob('failure_*.json')) and not (folder/'cap_stop.json').exists()
    if (image/'completion.json').exists():
        assert read(image/'completion.json')['status']=='COMPLETE_ALL830_QUERIES_6_ANCHOR_ARMS_IMAGES_SCORES_PENDING'
        prep=submit('PKU_prepare_anchor_scoring_CPU_20261109_v288.sbatch',jid,[]);score=submit('PKU_score_anchor_images_GPU_20261109_v288.sbatch',prep,[1,'DINO'])
        r.update(status='ALL4980_NEW_IMAGES_COMPLETE_SIX_ARM_SCORING_AND20ARM_ANALYSIS_QUEUED',preparation_job=prep,scoring_job=score,original_features_reused_arms=14);write(out/'scoring_submitted.json',r)
    elif (train/'queue/completion.json').exists() and not (out/'images_submitted.json').exists():
        assert read(train/'queue/completion.json')['status']=='COMPLETE_ALL3_ANCHORED_FIXED_T_MODELS_CPU_IMAGES_PENDING'
        prep=submit('PKU_prepare_anchor_images_CPU_20261109_v288.sbatch',jid,[]);image_job=submit('PKU_generate_anchor_images_GPU_20261109_v291.sbatch',prep,[1])
        next_job=submit('PKU_dispatch_anchor_images_CPU_20261109_v294.sbatch',image_job,[args.hop+1]);r.update(status='ALL3MODELS_COMPLETE_CPU830_REPLAY_AND4980_IMAGES_SUBMITTED',condition_job=prep,image_job=image_job,next_dispatch_job=next_job);write(out/'images_submitted.json',r)
    else:
        assert args.hop<125
        if (out/'images_submitted.json').exists():
            stage=closed(image/'stages');dependency=stage['next_job_id'];phase='new_images'
        elif list((train/'queue').glob('stage_*.json')):
            stage=closed(train/'queue');dependency=stage['next_job_id'];phase='new_training'
        elif (train/'full_training_submitted.json').exists():dependency=read(train/'full_training_submitted.json')['job_id'];phase='new_first_training'
        elif (R/'runs/EEG_anchor_dispatch_v293/submitted.json').exists():dependency=read(R/'runs/EEG_anchor_dispatch_v293/submitted.json')['continuation_gate_job'];phase='new_GPU_prefix_and_CPU_gate'
        else:dependency=read(R/'runs/EEG_anchor_dispatch_v293/progress.json')['next_dispatch_job'];phase='old_study_tail_before_newGPU'
        assert dependency.isdigit() and dependency!=jid
        job=submit('PKU_dispatch_anchor_images_CPU_20261109_v294.sbatch',dependency,[args.hop+1]);r.update(status='FOLLOWING_ACTUAL_SERIAL_CHAIN_NO_EXTRA_GPU',phase=phase,dependency=dependency,next_dispatch_job=job)
    write(path,r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(out/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED',error=repr(e),traceback=traceback.format_exc()));raise
