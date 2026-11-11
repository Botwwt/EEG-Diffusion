"""Five complete encoders, five own ordinary heads, ten full image arms, one GPU."""
import argparse,datetime,json,os,socket,subprocess,traceback
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--stage',type=int,required=True);ap.add_argument('--phase',choices=['encoder','ordinary'],required=True);ap.add_argument('--case',required=True);ap.add_argument('--role',required=True);a=ap.parse_args()
from PKU_backbone_core_20261110_v322 import R,ROOT,CASES,ROLES,write,sha
queue=ROOT/'queue';queue.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID'];py=str(R/'envs/torch230_cu121_v235/bin/python')
def submit(stem,dep,args):
    p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);j=p.stdout.strip().split(';')[0];assert p.returncode==0 and j.isdigit(),(p.stdout,p.stderr);return j
try:
    assert a.case in CASES and a.role in ROLES and 1<=a.stage<=160 and not list(ROOT.glob('failure_*.json')) and not list(queue.glob('failure_*.json'))
    tag=queue/('stage_'+str(a.stage)+'.json');assert not tag.exists()
    if a.stage==1:assert json.loads((R/'runs/EEG_visitation_scores_v320/analysis_completion.json').read_text())['status']=='COMPLETE_FULL830_THREE_VISITATION_FITS_FROZEN_NORM_CONTROL_NINE_ARMS_ALL14METRICS'
    else:
        previous=json.loads((queue/('stage_'+str(a.stage-1)+'.json')).read_text());assert previous['next_job_id']==jid and previous['next_phase']==a.phase and previous['next_case']==a.case and previous['next_role']==a.role
    phase=a.phase;case=a.case;role=a.role;index=CASES.index(case);finished=False
    if phase=='encoder':
        folder=ROOT/case/'encoders'/role;profile=not (folder/'state.pt').exists()
        subprocess.check_call([py,'-B',str(R/'code/PKU_train_backbone_GPU_20261110_v322.py'),'--case',case,'--role',role]+(['--profile'] if profile else []));subprocess.check_call([py,'-B',str(R/'code/PKU_audit_backbone_CPU_20261110_v322.py'),'--case',case,'--role',role])
        proof=json.loads((ROOT/'audits'/(case+'_'+role+'_'+jid+'.json')).read_text());updates=proof['updates']
        if proof['full_complete']:
            if role!=ROLES[-1]:role=ROLES[ROLES.index(role)+1]
            else:
                subprocess.check_call([py,'-B',str(R/'code/PKU_assemble_backbone_OOF_CPU_20261110_v322.py'),'--case',case])
                if index<len(CASES)-1:case=CASES[index+1];role='full'
                else:phase='ordinary';case=CASES[0];role='full'
    else:
        folder=ROOT/case/'ordinary';resume=(folder/'resume.pt').exists();cmd=[py,'-B',str(R/'code/PKU_train_backbone_ordinary_GPU_20261110_v322.py'),'--case',case,'--stop-after-updates','2481' if resume else '4']+(['--resume'] if resume else []);subprocess.check_call(cmd)
        subprocess.check_call([py,'-B',str(R/'code/PKU_audit_backbone_ordinary_CPU_20261110_v322.py'),'--case',case]);proof=json.loads((ROOT/case/('ordinary_audit_'+jid)/'completion.json').read_text());updates=proof['updates']
        if proof['full_training_complete']:
            subprocess.check_call([py,'-B',str(R/'code/PKU_calibrate_backbone_ordinary_GPU_20261110_v322.py'),'--case',case])
            if index<len(CASES)-1:case=CASES[index+1]
            else:finished=True
    r=dict(stage=a.stage,phase=a.phase,case=a.case,role=a.role,job_id=jid,updates=updates,CPU_proof=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if finished:
        r['status']='COMPLETE_ALL5_THREE_ROLE_BACKBONES_AND_OWN_ORDINARY';write(queue/'completion.json',r)
        prep=submit('PKU_prepare_backbone_images_CPU_20261110_v322.sbatch',jid,[]);image=submit('PKU_generate_backbone_images_GPU_20261110_v322.sbatch',prep,[1]);dispatch=submit('PKU_dispatch_backbone_scoring_CPU_20261110_v322.sbatch',image,[1]);r.update(image_prepare_job=prep,image_job=image,scoring_dispatch_job=dispatch)
    elif a.stage==160:write(queue/'cap_stop.json',dict(r,status='FINITE_STAGE_CAP_RETAINED'))
    else:
        write(queue/('claim_'+str(a.stage)+'.json'),dict(parent_job=jid,next_stage=a.stage+1,next_phase=phase,next_case=case,next_role=role));nxt=submit('PKU_queue_backbones_GPU_20261110_v322.sbatch',jid,[a.stage+1,phase,case,role]);r.update(status='SAFE_PREFIX_NEXT_SUBMITTED',next_job_id=nxt,next_phase=phase,next_case=case,next_role=role)
    write(tag,r);write(queue/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:write(queue/('failure_'+jid+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
