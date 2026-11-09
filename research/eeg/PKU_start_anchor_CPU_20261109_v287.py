"""Admit complete three-case fits only after the real two-update GPU prefixes and CPU proof."""
import datetime,json,os,socket,subprocess
from pathlib import Path
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
R=Path('/dfs/share/maleiLab/wwt/eeg');root=R/'runs/EEG_anchor_v286';proof=json.loads((root/'audits/profile_completion.json').read_text());assert proof['status']=='PASS_ALL3_ANCHORED_TWO_UPDATE_PREFIXES_CPU'
assert len(proof['cases'])==3 and all(r['updates']==2 and r['exposures']==32 for r in proof['cases']) and not (root/'queue').exists()
p=subprocess.run(['sbatch','--parsable','--dependency=afterok:'+os.environ['SLURM_JOB_ID'],'--chdir='+str(R),'--output='+str(R/'logs/%x-%j.out'),'--error='+str(R/'logs/%x-%j.err'),str(R/'code/PKU_queue_anchor_GPU_20261109_v286.sbatch'),'1','C_positive'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
jid=p.stdout.strip().split(';')[0];assert p.returncode==0 and jid.isdigit(),(p.returncode,p.stdout,p.stderr)
r=dict(status='ANCHORED_ENCODER_FIXED3_FULL_T_FITS_SUBMITTED_AFTER_REAL_PREFIX_CPU_PROOF',job_id=jid,cases=3,updates_each=827,first2count_once=True,UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
p=root/'full_training_submitted.json';assert not p.exists()
with p.open('x') as f:f.write(json.dumps(r,indent=2)+'\n');f.flush();os.fsync(f.fileno())
print(json.dumps(r))
