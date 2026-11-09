"""Bounded six-case queue. Each continuation requires actual checkpoint proof."""
import argparse,datetime,hashlib,json,os,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);ap.add_argument('--case',required=True);args=ap.parse_args()
CASES=['C_positive','C_pair','C_reverse'];assert args.case in CASES and 1<=args.stage<=9
R=Path('/dfs/share/maleiLab/wwt/eeg');root=R/'runs/EEG_anchor_v286';queue=root/'queue';queue.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID'];start=time.monotonic()
def write(p,r):
    b=(json.dumps(r,indent=2,allow_nan=False)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    assert t.read_bytes()==b;t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
try:
    tag=queue/('stage_'+str(args.stage)+'.json');assert not tag.exists()
    if args.stage>1:
        previous=json.loads((queue/('stage_'+str(args.stage-1)+'.json')).read_text());assert previous['next_job_id']==jid and previous['next_case']==args.case
    receipt=json.loads((root/args.case/'checkpoint_receipt.json').read_text());assert receipt['updates']<827
    candidates=[json.loads(p.read_text()) for p in sorted((root/'audits').glob(args.case+'_*.json'))]
    checked=[v for v in candidates if v['checkpoint_sha256']==receipt['checkpoint_sha256'] and v['updates']==receipt['updates']]
    assert checked and all(v['status']=='PASS_ANCHORED_MODEL_ADAM_RNG_ALL_COVERAGE_FROZEN_ANCHOR_AND_CAP' for v in checked)
    stage=dict(stage=args.stage,case=args.case,job_id=jid,status='RUNNING_SAME_REGISTERED_CASE',starting_updates=receipt['updates'],starting_checkpoint_sha256=receipt['checkpoint_sha256'],UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    write(queue/'progress.json',stage);python=R/'envs/torch230_cu121_v235/bin/python'
    subprocess.check_call([str(python),'-B',str(R/'code/PKU_train_anchor_GPU_20261109_v286.py'),'--case',args.case])
    subprocess.check_call([str(python),'-B',str(R/'code/PKU_audit_anchor_CPU_20261109_v302.py'),'--case',args.case])
    proof_path=root/'audits'/(args.case+'_'+jid+'.json');proof=json.loads(proof_path.read_text());assert proof['status']=='PASS_ANCHORED_MODEL_ADAM_RNG_ALL_COVERAGE_FROZEN_ANCHOR_AND_CAP' and proof['updates']>receipt['updates']
    stage.update(updates=proof['updates'],exposures=proof['exposures'],checkpoint_sha256=proof['checkpoint_sha256'],audit_sha256=sha(proof_path),case_complete=proof['full_T_complete'],seconds=time.monotonic()-start)
    index=CASES.index(args.case);all_done=proof['full_T_complete'] and index==len(CASES)-1
    if all_done:
        completions=[json.loads((root/case/'completion.json').read_text()) for case in CASES];assert all(v['updates']==827 and v['exposures']==13230 for v in completions)
        stage.update(status='COMPLETE_ALL3_ANCHORED_FIXED_T_MODELS_CPU_IMAGES_PENDING',total_updates=2481,total_query_exposures=39690,new_images=0);write(queue/'completion.json',stage)
    elif args.stage==9:
        stage['status']='FINITE_9_STAGE_CAP_SAFE_CHECKPOINTS_NOT_FULL_STUDY';write(queue/'cap_stop.json',stage)
    else:
        next_case=CASES[index+1] if proof['full_T_complete'] else args.case
        claim=queue/('dispatch_after_'+str(args.stage)+'.json')
        with claim.open('x') as f:f.write(json.dumps(dict(parent_job=jid,next_case=next_case,next_stage=args.stage+1))+'\n');f.flush();os.fsync(f.fileno())
        p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+jid,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_queue_anchor_GPU_20261109_v302.sbatch'),str(args.stage+1),next_case],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
        next_jid=p.stdout.strip().split(';')[0];assert p.returncode==0 and next_jid.isdigit(),(p.returncode,p.stdout,p.stderr)
        stage.update(status='NEXT_CASE_OR_PREFIX_SUBMITTED_AFTER_ACTUAL_CPU_PROOF',next_job_id=next_jid,next_case=next_case,next_stage=args.stage+1)
    stage['no_effect_gate_or_epoch_selection']=True;write(tag,stage);write(queue/'progress.json',stage);print(json.dumps(stage),flush=True)
except BaseException as e:
    write(queue/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',stage=args.stage,case=args.case,job_id=jid,error=repr(e),traceback=traceback.format_exc()));raise
