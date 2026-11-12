"""Follow only actual original EEG scoring jobs; a CPU-only final table/protocol audit."""
import argparse,datetime,json,os,socket,subprocess,traceback
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower();ap=argparse.ArgumentParser();ap.add_argument('--hop',type=int,required=True);a=ap.parse_args();assert 1<=a.hop<=300
R=Path('/dfs/share/maleiLab/wwt/eeg');out=R/'runs/EEG_backbone_protocol_wait_v330';out.mkdir(exist_ok=True);jid=os.environ['SLURM_JOB_ID']
def write(p,v):
    b=(json.dumps(v,indent=2)+'\n').encode();t=p.with_name(p.name+'.pending')
    with t.open('wb') as f:f.write(b);f.flush();os.fsync(f.fileno())
    t.replace(p)
try:
    root=R/'runs/EEG_backbones_v322';assert not list(root.glob('failure_*.json')) and not list((root/'queue').glob('failure_*.json')) and not (root/'queue/cap_stop.json').exists()
    for suffix in ['images','scores','dispatch']:
        p=R/'runs'/('EEG_backbone_'+suffix+'_v322');assert not list(p.glob('*failure*.json')) and not (p/'cap_stop.json').exists()
    r=dict(job_id=jid,hop=a.hop,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat());complete=R/'runs/EEG_backbone_scores_v322/analysis_completion.json'
    if complete.exists():
        assert json.loads(complete.read_text())['status']=='COMPLETE_FULL830_ALL5_BACKBONES_OWN_ORDINARY_THIRTEEN_ARMS_ALL14METRICS';subprocess.check_call([str(R/'envs/torch230_cu121_v235/bin/python'),'-B',str(R/'code/PKU_backbone_protocol_audit_CPU_20261111_v330.py')]);r['status']='COMPLETE_ALL_IMAGE_TABLE_AND_FRESH_MLP_PROTOCOL_CHECK'
    else:
        rows=subprocess.check_output(['squeue','-u','HPC_USER','-h','-o','%i|%j'],universal_newlines=True).splitlines();ids=[v.split('|')[0] for v in rows if v.split('|')[1].startswith('eeg_bb_')];assert ids and a.hop<300;dep=':'.join(ids)
        p=subprocess.run(['sbatch','--parsable','--dependency=afterany:'+dep,'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_follow_backbone_protocol_CPU_20261111_v330.sbatch'),str(a.hop+1)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);nxt=p.stdout.strip().split(';')[0];assert p.returncode==0 and nxt.isdigit(),(p.stdout,p.stderr);r.update(status='WAITING_ON_ORIGINAL_COMPLETE_IMAGE_AND_SCORING_CHAIN',dependency=dep,next_job_id=nxt)
    write(out/('hop_'+str(a.hop)+'.json'),r);write(out/'progress.json',r);print(json.dumps(r),flush=True)
except BaseException as e:write(out/('failure_'+jid+'.json'),dict(error=repr(e),traceback=traceback.format_exc()));raise
