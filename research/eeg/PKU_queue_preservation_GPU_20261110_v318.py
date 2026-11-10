"""Finite train/image/scoring lifecycle, starts only after the previous study closes."""
import argparse,datetime,json,os,socket,subprocess,traceback
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_conditional_preservation_core_20261110_v317 import R,ROOT,CASES,sha,write
assert a.case in CASES and 1<=a.stage<=65;queue=ROOT/'queue';queue.mkdir(parents=True,exist_ok=True);jid=os.environ['SLURM_JOB_ID'];py=str(R/'envs/torch230_cu121_v235/bin/python')
def submit(stem,dep,args):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in args],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.stdout,p.stderr);return job
try:
    tag=queue/('stage_'+str(a.stage)+'.json');assert not tag.exists() and not list(ROOT.glob('failure_*.json'))
    if a.stage==1:
        assert json.loads((R/'runs/EEG_P_intervention_scores_v316/analysis_completion.json').read_text())['status']=='COMPLETE_FULL830_FOUR_NEW_INTERVENTIONS_EIGHT_ARMS_ALL14METRICS'
        assert json.loads((ROOT/'engineering/completion.json').read_text())['status']=='PASS_NATIVE_P_WEIGHT_SHAPE_GRADIENT_AND_PROJECTION_ALGEBRA'
    else:
        prev=json.loads((queue/('stage_'+str(a.stage-1)+'.json')).read_text());assert prev['next_job_id']==jid and prev['next_case']==a.case
    folder=ROOT/a.case;profile=not (folder/'state.pt').exists()
    if not profile:
        rec=json.loads((folder/'checkpoint_receipt.json').read_text());proofs=[json.loads(p.read_text()) for p in (ROOT/'audits').glob(a.case+'_*.json')];assert any(p['checkpoint_sha256']==rec['checkpoint_sha256'] and p['updates']==rec['updates'] for p in proofs)
    cmd=[py,'-B',str(R/'code/PKU_train_preservation_GPU_20261110_v318.py'),'--case',a.case]+(['--profile'] if profile else [])
    subprocess.check_call(cmd);subprocess.check_call([py,'-B',str(R/'code/PKU_audit_preservation_CPU_20261110_v317.py'),'--case',a.case])
    proof=json.loads((ROOT/'audits'/(a.case+'_'+jid+'.json')).read_text());r=dict(stage=a.stage,case=a.case,job_id=jid,real_GPU_prefix=profile,updates=proof['updates'],exposures=proof['exposures'],checkpoint_sha256=proof['checkpoint_sha256'],CPU_proof=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    index=CASES.index(a.case)
    if proof['full_T_complete'] and index==len(CASES)-1:
        r['status']='COMPLETE_ALL5_PRESERVATION_FIXED_T_MODELS';write(queue/'completion.json',r)
        prep=submit('PKU_prepare_preservation_images_CPU_20261110_v317.sbatch',jid,[]);image=submit('PKU_generate_preservation_images_GPU_20261110_v317.sbatch',prep,[1]);r.update(image_prepare_job=prep,image_job=image)
        dispatch=submit('PKU_dispatch_preservation_scoring_CPU_20261110_v317.sbatch',image,[1]);r['scoring_dispatch_job']=dispatch
    elif a.stage==65:r['status']='FINITE_65_STAGE_CAP_RETAINED_PREFIX';write(queue/'cap_stop.json',r)
    else:
        nextcase=CASES[index+1] if proof['full_T_complete'] else a.case
        write(queue/('claim_'+str(a.stage)+'.json'),dict(parent_job=jid,next_stage=a.stage+1,next_case=nextcase));job=submit('PKU_queue_preservation_GPU_20261110_v318.sbatch',jid,[a.stage+1,nextcase]);r.update(status='NEXT_FIXED_CASE_OR_SAME_STATE_PREFIX_SUBMITTED',next_job_id=job,next_case=nextcase)
    write(tag,r);write(queue/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:
    write(queue/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTO_RETRY',stage=a.stage,case=a.case,error=repr(e),traceback=traceback.format_exc()));raise
