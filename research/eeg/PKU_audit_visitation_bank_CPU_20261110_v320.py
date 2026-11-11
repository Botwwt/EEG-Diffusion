"""CPU file, tensor, full coverage, timestep and forward-scale checks."""
import json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
import torch
from PKU_conditional_visitation_core_20261110_v320 import ROOT,sha,write
torch.set_num_threads(4);root=ROOT/'bank';start=time.monotonic();cursor=0;rows=[]
for p in sorted((root/'parts').glob('*.json')):
    r=json.loads(p.read_text());f=p.with_suffix('.pt');assert r['lo']==cursor and sha(f)==r['sha256'];v=torch.load(f,map_location='cpu',weights_only=True);n=r['hi']-cursor
    assert v['lo']==cursor and v['hi']==r['hi'] and v['on'].shape==v['forward'].shape==(n,4,4,64,64)
    assert v['on'].dtype==v['forward'].dtype==torch.float16 and torch.isfinite(v['on']).all() and torch.isfinite(v['forward']).all()
    assert v['times'].tolist()==r['times']==[999.,749.,499.,249.] and len(v['seeds'])==n
    rows.append(r);cursor=r['hi']
proof=dict(status='PASS_BANK_PREFIX_OR_COMPLETE',queries=cursor,all4_times_per_query=True,all_file_sha256=True,seconds=time.monotonic()-start,parts=rows,job_id=os.environ['SLURM_JOB_ID'])
write(root/'CPU_proof.json',proof);assert (root/'completion.json').exists()==(cursor==13230);print('PASS_BANK_'+str(cursor),flush=True)
