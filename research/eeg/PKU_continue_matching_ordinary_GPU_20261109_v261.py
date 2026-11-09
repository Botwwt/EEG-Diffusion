"""Finite Slurm chain: same fit, CPU-verified checkpoints, at most five tranches."""
import argparse,datetime,hashlib,json,os,socket,subprocess,sys,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--tranche',type=int,required=True);k=ap.parse_args().tranche;assert 1<=k<=5
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/HIVE_matching_ordinary_sub03_v259';jid=os.environ['SLURM_JOB_ID'];start=time.monotonic()
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
try:
    tag=out/('chain_tranche_'+str(k)+'.json');assert not tag.exists()
    before=json.loads((out/'checkpoint_receipt.json').read_text());assert before['updates']<2481
    if k==1:
        marker=json.loads((R/'manifests/matching_ordinary_state_audit_prefix4_v260.json').read_text())
        checked=json.loads((R/('runs/matching_ordinary_state_audit_'+marker['job_id']+'/completion.json')).read_text())
    else:
        previous=json.loads((out/('chain_tranche_'+str(k-1)+'.json')).read_text());assert previous['next_job_id']==jid and previous['status']=='NEXT_TRANCHE_SUBMITTED_AFTER_VERIFIED_PREFIX'
        checked=json.loads((R/('runs/matching_ordinary_state_audit_'+previous['job_id']+'/completion.json')).read_text())
    assert checked['status']=='PASS_COMMITTED_MATCHING_ORDINARY_MODEL_ADAM_RNG_AND_EXACT_COVERAGE' and checked['checkpoint_sha256']==before['checkpoint_sha256'] and checked['updates']==before['updates']
    stage=dict(tranche=k,job_id=jid,status='RUNNING_SAME_COMMITTED_FIT',starting_updates=before['updates'],starting_checkpoint_sha256=before['checkpoint_sha256'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(out/'chain_progress.json',stage)
    python=R/'envs/torch230_cu121_v235/bin/python'
    subprocess.check_call([str(python),'-B',str(R/'code/PKU_train_matching_ordinary_GPU_20261109_v259.py'),'--resume','--stop-after-updates','2481'])
    subprocess.check_call([str(python),'-B',str(R/'code/PKU_audit_matching_ordinary_CPU_20261109_v260.py')])
    receipt=json.loads((out/'checkpoint_receipt.json').read_text());audit=R/('runs/matching_ordinary_state_audit_'+jid+'/completion.json');checked=json.loads(audit.read_text())
    assert checked['status']=='PASS_COMMITTED_MATCHING_ORDINARY_MODEL_ADAM_RNG_AND_EXACT_COVERAGE' and checked['checkpoint_sha256']==receipt['checkpoint_sha256'] and receipt['updates']>before['updates']
    stage.update(updates=receipt['updates'],exposures=receipt['seen'],checkpoint_sha256=receipt['checkpoint_sha256'],CPU_audit_sha256=sha(audit),seconds=time.monotonic()-start)
    if receipt['updates']==2481:
        assert checked['full_training_complete'] and receipt['seen']==39690
        stage['status']='COMPLETE_FIXED_SINGLE_SUB03_ORDINARY_FIT_AND_CHECKPOINT_AUDIT_PRECALIBRATION';write(out/'chain_completion.json',stage)
    elif k==5:stage['status']='FINITE_FIVE_TRANCHE_CAP_SAFE_PREFIX_NOT_FULL_COMPLETION';write(out/'chain_cap_stop.json',stage)
    else:
        # A unique dispatch claim prevents another owner from submitting this step twice.
        claim=out/('dispatch_after_tranche_'+str(k)+'.json')
        with claim.open('x') as f:f.write(json.dumps(dict(job_id=jid,next_tranche=k+1))+'\n');f.flush();os.fsync(f.fileno())
        p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_continue_matching_ordinary_GPU_20261109_v261.sbatch'),str(k+1)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        next_jid=p.stdout.strip().split(';')[0];assert p.returncode==0 and next_jid.isdigit(),(p.returncode,p.stdout,p.stderr)
        stage.update(status='NEXT_TRANCHE_SUBMITTED_AFTER_VERIFIED_PREFIX',next_job_id=next_jid,next_tranche=k+1)
    stage['no_effect_threshold_or_hyperparameter_change']=True;stage['new_images']=0;write(tag,stage);write(out/'chain_progress.json',stage);print(json.dumps(stage),flush=True)
except BaseException as e:
    write(out/('chain_failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY',tranche=k,job_id=jid,error=repr(e),traceback=traceback.format_exc()));raise
