"""Follow actual v316 tails; admit the new serial study only after full closure."""
import argparse,datetime,json,os,socket,subprocess,time,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);a=ap.parse_args();assert 1<=a.hop<=90
R=Path('/dfs/share/maleiLab/wwt/eeg');OUT=R/'runs/EEG_preservation_start_v317';OUT.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def write(p,v):
    t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def submit(stem,dependency,args=[],kind='afterok'):
    cmd=['sbatch','--parsable','--dependency='+kind+':'+dependency,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+[str(v) for v in args]
    p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);job=p.stdout.strip().split(';')[0];assert p.returncode==0 and job.isdigit(),(p.stdout,p.stderr);return job
try:
    assert not (OUT/'submitted.json').exists() and not (OUT/('hop_'+str(a.hop)+'.json')).exists()
    folders=[R/'runs'/k for k in ['EEG_P_intervention_tokens_v316','EEG_P_intervention_images_v316','EEG_P_intervention_scores_v316','EEG_P_intervention_dispatch_v316']]
    assert not any(list(folder.glob('*failure*.json')) for folder in folders),'Prior study failed; preserve and do not start new GPU fits'
    done=R/'runs/EEG_P_intervention_scores_v316/analysis_completion.json'
    v=dict(hop=a.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
    if done.exists():
        assert json.loads(done.read_text())['status']=='COMPLETE_FULL830_FOUR_NEW_INTERVENTIONS_EIGHT_ARMS_ALL14METRICS'
        eng=json.loads((R/'runs/EEG_preservation_v317/engineering/completion.json').read_text())
        assert eng['status']=='PASS_NATIVE_P_WEIGHT_SHAPE_GRADIENT_AND_PROJECTION_ALGEBRA'
        train=submit('PKU_queue_preservation_GPU_20261110_v317.sbatch',jid,[1,'P_abs_free'])
        v.update(status='PREVIOUS_FULL_STUDY_CLOSED_FIVE_CASE_COMPLETE_LIFECYCLE_SUBMITTED',CPU_engineering=eng['CPU_job'],first_GPU_training=train,new_images=4150)
        write(OUT/'submitted.json',v)
    else:
        assert a.hop<90
        # Metadata only: actual own study tails, never another project's job.
        rows=subprocess.check_output(['squeue','-u','HPC_USER','-h','-o','%i|%j'],universal_newlines=True).splitlines()
        ids=[row.split('|')[0] for row in rows if row.split('|')[1].startswith('eeg_P_')]
        if not ids:
            for folder in folders[1:]:
                p=folder/'progress.json'
                if p.exists():
                    progress=json.loads(p.read_text())
                    for key in ['next_job_id','next_dispatch_job','analysis_job','scoring_job']:
                        if key in progress:ids.append(progress[key])
        assert ids,'No live or saved continuation in v316; preserve unknown instead of starting another GPU'
        dep=':'.join(sorted(set(ids)));nxt=submit('PKU_start_preservation_CPU_20261110_v317.sbatch',dep,[a.hop+1],kind='afterany')
        v.update(status='WAITING_ON_ACTUAL_V316_FULL_IMAGE_AND_SCORE_CHAIN',dependency=dep,next_job_id=nxt)
    write(OUT/('hop_'+str(a.hop)+'.json'),v);write(OUT/'progress.json',v);print(json.dumps(v),flush=True)
except BaseException as e:
    write(OUT/('failure_'+jid+'.json'),dict(status='FAILED_PRESERVED_NO_AUTOMATIC_NEW_FIT',error=repr(e),traceback=traceback.format_exc()));raise
