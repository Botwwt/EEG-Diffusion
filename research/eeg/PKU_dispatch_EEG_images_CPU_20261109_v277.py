"""Follow the actual finite training tail by dependencies, never occupy a waiting GPU."""
import argparse,datetime,json,os,socket,subprocess,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);args=ap.parse_args();assert 1<=args.hop<=13
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_images_v271';out.mkdir(exist_ok=True);folder=out/'dispatch_v277';folder.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']

def write(p,r):
    b=(json.dumps(r,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)

def submit(name,dependency,argument):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/name),str(argument)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
    job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.returncode,p.stdout,p.stderr);return job

try:
    record=folder/('hop_'+str(args.hop)+'.json');assert not record.exists() and not (out/'image_queue_submitted.json').exists()
    queue=R/'runs/EEG_candidates_v267/queue';assert not list(queue.glob('failure_*.json')) and not (queue/'cap_stop.json').exists()
    r=dict(hop=args.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if (queue/'completion.json').exists():
        assert json.loads((queue/'completion.json').read_text())['status']=='COMPLETE_ALL6_FIXED_T_CASES_MODELS_AND_AUDITS_IMAGES_PENDING'
        bank_job=submit('PKU_prepare_EEG_image_conditions_CPU_20261109_v271.sbatch',jid,'0')
        image_job=submit('PKU_generate_EEG_comparison_GPU_20261109_v271.sbatch',bank_job,'1')
        r.update(status='ALL6_MODELS_COMPLETE_CPU_BANK_AND_GPU_IMAGES_SUBMITTED',condition_job=bank_job,image_job=image_job,images_expected=11620)
        write(out/'image_queue_submitted.json',r)
    else:
        closed=[json.loads(p.read_text()) for p in queue.glob('stage_*.json')];progress=max(closed,key=lambda v:v['stage']);assert progress['status']=='NEXT_CASE_OR_PREFIX_SUBMITTED_AFTER_ACTUAL_CPU_PROOF' and progress['next_job_id'].isdigit()
        assert args.hop<13
        next_job=submit('PKU_dispatch_EEG_images_CPU_20261109_v277.sbatch',progress['next_job_id'],args.hop+1)
        r.update(status='FOLLOWING_ACTUAL_REGISTERED_TRAINING_TAIL',training_job=progress['next_job_id'],next_dispatch_job=next_job)
    write(record,r);write(folder/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(folder/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',hop=args.hop,error=repr(e),traceback=traceback.format_exc()));raise
