"""Follow only current EEG visitation tails; start after its full all-metric closure."""
import argparse,datetime,json,os,socket,subprocess,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);a=ap.parse_args();assert 1<=a.hop<=160
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_backbone_start_v322';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def write(p,v):
    t=p.with_name(p.name+'.pending');t.write_text(json.dumps(v,indent=2)+'\n');t.replace(p)
def submit(stem,dep,args,after='afterok'):
    p=subprocess.run(['sbatch','--parsable','--dependency='+after+':'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code'/stem)]+list(map(str,args)),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);j=p.stdout.strip().split(';')[0];assert p.returncode==0 and j.isdigit(),(p.stdout,p.stderr);return j
try:
    assert not (out/'submitted.json').exists();root=R/'runs/EEG_backbones_v322';assert json.loads((root/'engineering/completion.json').read_text())['status']=='PASS_ALL5_REAL_EEG_GRADCACHE_GRADIENTS_BN_RNG_AND_MATCHED_JOINT_SEPARATED_ARCHITECTURES'
    old=R/'runs/EEG_visitation_v320';assert not list(old.glob('failure_*.json')) and not list((old/'queue').glob('failure_*.json')) and not (old/'queue/cap_stop.json').exists()
    for name in ['EEG_visitation_images_v320','EEG_visitation_scores_v320']:assert not list((R/'runs'/name).glob('*failure*.json')) and not (R/'runs'/name/'cap_stop.json').exists()
    r=dict(hop=a.hop,job_id=jid,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());complete=R/'runs/EEG_visitation_scores_v320/analysis_completion.json'
    if complete.exists():
        assert json.loads(complete.read_text())['status']=='COMPLETE_FULL830_THREE_VISITATION_FITS_FROZEN_NORM_CONTROL_NINE_ARMS_ALL14METRICS';gpu=submit('PKU_queue_backbones_GPU_20261110_v322.sbatch',jid,[1,'encoder','MLP17','full']);r.update(status='PREVIOUS_FULL_STUDY_CLOSED_BACKBONE_CHAIN_SUBMITTED',first_GPU_job=gpu);write(out/'submitted.json',r)
    else:
        rows=subprocess.check_output(['squeue','-u','HPC_USER','-h','-o','%i|%j'],universal_newlines=True).splitlines();ids=[v.split('|')[0] for v in rows if v.split('|')[1].startswith('eeg_visit_')];assert ids and a.hop<160,rows;dep=':'.join(ids);nxt=submit('PKU_start_backbones_CPU_20261110_v322.sbatch',dep,[a.hop+1],after='afterany');r.update(status='WAITING_ON_ACTUAL_VISITATION_FULL_LIFECYCLE',dependency=dep,next_job_id=nxt)
    write(out/('hop_'+str(a.hop)+'.json'),r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:write(out/('failure_'+jid+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
