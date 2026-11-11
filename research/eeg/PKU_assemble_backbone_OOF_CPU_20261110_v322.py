"""Join actual concept-excluded predictions, retaining each parent's native affine."""
import argparse,os,socket,json,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_backbone_core_20261110_v322 import *
torch.set_num_threads(4);p=ROOT/a.case/'encoders';split,T,folds,post=input_contract();OOF=torch.empty(13230,1024);seen=torch.zeros(13230,dtype=torch.int64);locations={r:i for i,r in enumerate(T)}
for role in ROLES:
    done=json.loads((p/role/'completion.json').read_text());assert done['epochs']==25 and sha(p/role/'conditions.pt')==done['conditions_sha256'];v=torch.load(p/role/'conditions.pt',map_location='cpu',weights_only=True)
    assert v['model_sha256']==done['model_sha256']
    if role=='full':continue
    for i,r in enumerate(v['array_rows'].tolist()):
        ix=locations[r];assert int(folds[ix])==int(role[-1]);OOF[ix]=v['condition'][i];seen[ix]+=1
assert (seen==1).all() and torch.isfinite(OOF).all();save(p/'T_true_OOF_conditions.pt',dict(condition=OOF,array_rows=torch.tensor(T),folds=folds,subject='sub-03',case=a.case))
write(p/'OOF_completion.json',dict(status='COMPLETE_TRUE_CONCEPT_EXCLUDED_THREE_ROLE_BANK',queries=13230,roles=ROLES,sha256=sha(p/'T_true_OOF_conditions.pt'),new_updates=0,whole_ordinary_not_OOF=True));print('PASS_REAL_OOF_'+a.case,flush=True)
