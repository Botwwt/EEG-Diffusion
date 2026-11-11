"""Decode every actual prefix: data permissions, 25-pass coverage, Adam/RNG and exports."""
import argparse,math,os,socket,time,json
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);ap.add_argument('--role',required=True);a=ap.parse_args()
from PKU_backbone_core_20261110_v322 import *
torch.set_num_threads(4);start=time.monotonic();p=ROOT/a.case/'encoders'/a.role;r=json.loads((p/'checkpoint_receipt.json').read_text());assert sha(p/'state.pt')==r['checkpoint_sha256']
s=torch.load(p/'state.pt',map_location='cpu',weights_only=False);v=json.loads((p/'provenance.json').read_text());assert sha(p/'provenance.json')==s['provenance_sha256']
model=make(a.case);model.load_state_dict(s['model'],strict=True);assert all(torch.isfinite(t).all() for t in s['model'].values()) and torch.isfinite(s['scale'])
split,T,folds,post=input_contract();allowed=v['train_array_rows'];N=len(allowed);assert set(allowed).isdisjoint(split['image_rows']['D']+split['image_rows']['V'])
if a.role!='full':assert set(allowed)==set(torch.tensor(T)[folds!=int(a.role[-1])].tolist())
g=torch.Generator().manual_seed(1730);orders=[torch.randperm(N,generator=g) for _ in range(25)];assert [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]==v['orders_sha256']
cover=torch.zeros(25,N,dtype=torch.bool);seen=0
for k,e in enumerate(s['history'],1):
    ep=(k-1)//math.ceil(N/1024);off=((k-1)%math.ceil(N/1024))*1024;ix=orders[ep][off:off+1024];assert e['update']==k and e['epoch']==ep+1 and e['offset']==off and e['n']==len(ix) and not cover[ep,ix].any();cover[ep,ix]=True;seen+=len(ix)
    assert e['exposures']==seen and math.isfinite(e['loss']) and e['gradient_norm']>0 and e['peak_GPU_allocated_bytes']<20<<30
assert seen==s['cursor']==r['exposures'] and len(s['history'])==s['updates']==r['updates'] and torch.equal(cover,s['coverage'])
for state in s['optimizer']['state'].values():assert float(state['step'])==s['updates'] and torch.isfinite(state['exp_avg']).all() and torch.isfinite(state['exp_avg_sq']).all()
for state in [s['teacher_RNG'],s['student_RNG']]:assert state['CPU'].dtype==state['CUDA'].dtype==torch.uint8 and state['CUDA'].numel()>=16
complete=seen==25*N
if complete:
    assert cover.all();done=json.loads((p/'completion.json').read_text());assert done['model_sha256']==sha(p/'brain_final.pt') and done['conditions_sha256']==sha(p/'conditions.pt')
    ck=torch.load(p/'brain_final.pt',map_location='cpu',weights_only=True);actual=affine(model,a.case);assert all(torch.equal(ck['affine'][k],t) for k,t in actual.items())
out=ROOT/'audits';out.mkdir(exist_ok=True);proof=dict(status='PASS_BACKBONE_MODEL_ADAM_RNG_PERMISSIONS_AND_EXACT_COVERAGE',case=a.case,role=a.role,updates=s['updates'],exposures=seen,full_complete=complete,checkpoint_sha256=sha(p/'state.pt'),seconds=time.monotonic()-start,job_id=os.environ['SLURM_JOB_ID'],CUDA_training_replayed=False)
write(out/(a.case+'_'+a.role+'_'+os.environ['SLURM_JOB_ID']+'.json'),proof);print(json.dumps(proof),flush=True)
