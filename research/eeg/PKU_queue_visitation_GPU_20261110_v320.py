"""One serial GPU lifecycle; no failed-step retries or outcome-driven model selection."""
import argparse,datetime,json,os,socket,subprocess,traceback
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);ap.add_argument('--phase',choices=['bank','train'],required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_conditional_visitation_core_20261110_v320 import R,ROOT,CASES,sha,write
queue=ROOT/'queue';queue.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID'];py=str(R/'envs/torch230_cu121_v235/bin/python')
def submit(stem,dep,args):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.stdout,p.stderr);return job
try:
    assert 1<=a.stage<=90 and a.case in CASES and not list(ROOT.glob('failure_*.json')) and not list(queue.glob('failure_*.json'))
    tag=queue/('stage_'+str(a.stage)+'.json');assert not tag.exists()
    if a.stage==1:assert a.phase=='bank' and json.loads((ROOT/'engineering/completion.json').read_text())['status']=='PASS_VISITATION_ALGEBRA_AND_FROZEN_CONTROL'
    else:
        prev=json.loads((queue/('stage_'+str(a.stage-1)+'.json')).read_text());assert prev['next_job_id']==jid and prev['next_phase']==a.phase and prev['next_case']==a.case
    if a.phase=='bank':
        profile=not (ROOT/'bank/progress.json').exists();subprocess.check_call([py,'-B',str(R/'code/PKU_build_visitation_bank_GPU_20261110_v320.py')]+(['--profile'] if profile else []));subprocess.check_call([py,'-B',str(R/'code/PKU_audit_visitation_bank_CPU_20261110_v320.py')]);full=(ROOT/'bank/completion.json').exists();phase='train' if full else 'bank';case=CASES[0];updates=0
    else:
        folder=ROOT/a.case;profile=not (folder/'state.pt').exists()
        if not profile:
            rec=json.loads((folder/'checkpoint_receipt.json').read_text());assert any(json.loads(p.read_text())['checkpoint_sha256']==rec['checkpoint_sha256'] for p in (ROOT/'audits').glob(a.case+'_*.json'))
        subprocess.check_call([py,'-B',str(R/'code/PKU_train_visitation_GPU_20261110_v320.py'),'--case',a.case]+(['--profile'] if profile else []));subprocess.check_call([py,'-B',str(R/'code/PKU_audit_visitation_CPU_20261110_v320.py'),'--case',a.case])
        proof=json.loads((ROOT/'audits'/(a.case+'_'+jid+'.json')).read_text());updates=proof['updates'];full=proof['full_T_complete'];index=CASES.index(a.case);phase='train';case=CASES[index+1] if full and index<len(CASES)-1 else a.case
    r=dict(stage=a.stage,phase=a.phase,case=a.case,job_id=jid,updates=updates,CPU_proof=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if a.phase=='train' and full and a.case==CASES[-1]:
        r['status']='COMPLETE_ALL3_VISITATION_FIXED_T_MODELS_PLUS_FROZEN_NORM_CONTROL';write(queue/'completion.json',r)
        prep=submit('PKU_prepare_visitation_images_CPU_20261110_v320.sbatch',jid,[]);image=submit('PKU_generate_visitation_images_GPU_20261110_v320.sbatch',prep,[1]);dispatch=submit('PKU_dispatch_visitation_scoring_CPU_20261110_v320.sbatch',image,[1]);r.update(image_prepare_job=prep,image_job=image,scoring_dispatch_job=dispatch)
    elif a.stage==90:write(queue/'cap_stop.json',dict(r,status='FINITE_STAGE_CAP_RETAINED'))
    else:
        write(queue/('claim_'+str(a.stage)+'.json'),dict(parent=jid,next_stage=a.stage+1,next_phase=phase,next_case=case));nxt=submit('PKU_queue_visitation_GPU_20261110_v320.sbatch',jid,[a.stage+1,phase,case]);r.update(status='SAFE_PREFIX_NEXT_SUBMITTED',next_job_id=nxt,next_phase=phase,next_case=case)
    write(tag,r);write(queue/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:write(queue/('failure_'+jid+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
