"""Same original five-strength Dfit rule, separately for each full parent."""
import argparse,os,socket,json,time,csv
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
ap=argparse.ArgumentParser();ap.add_argument('--case',required=True);a=ap.parse_args()
from PKU_backbone_core_20261110_v322 import *
from PKU_native_generator_20261109_v253 import load_native
from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
torch.set_num_threads(4);torch.manual_seed(1729);start=time.monotonic();p=ROOT/a.case/'ordinary';out=ROOT/a.case/'calibration';out.mkdir();assert json.loads((p/'training_completion.json').read_text())['updates']==2481
s=torch.load(p/'resume.pt',map_location='cpu',weights_only=False);assert s['updates']==2481 and s['coverage'].all()
split,T,folds,post=input_contract()
with (R/'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:ids=[int(v['array_row']) for v in csv.DictReader(f) if v['dataset_split']=='train' and v['concept_id'] in split['d_fit_concepts']]
assert len(ids)==820 and set(ids).isdisjoint(T+split['image_rows']['V'])
parent=ROOT/a.case/'encoders/full';v=torch.load(parent/'conditions.pt',map_location='cpu',weights_only=True);ck=torch.load(parent/'brain_final.pt',map_location='cpu',weights_only=True);positions={r:i for i,r in enumerate(v['array_rows'].tolist())};c=v['condition'][[positions[r] for r in ids]].cuda();gamma=ck['affine']['gamma'].cuda().expand(820,-1);beta=ck['affine']['beta'].cuda().expand(820,-1)
target=torch.load(R/'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt',map_location='cpu',weights_only=True)[ids].cuda();pipe,A,P,prompts,host=load_native()
model=BoundedConditionAdapter(predict_risk=False).cuda().eval();model.load_state_dict(s['model'],strict=True);grid=[0.,.25,.5,.75,1.];errors=[]
with torch.no_grad():
    for strength in grid:
        model.radius=.25*strength;parts=[]
        for lo in range(0,820,128):
            sl=slice(lo,lo+128);changed,_=model(c[sl],gamma[sl],beta[sl],gate_mode='fixed')
            with torch.autocast('cuda',dtype=torch.bfloat16):pred=P(A(changed));real=P(A(target[sl]))
            parts.append((1-F.cosine_similarity(pred.float(),real.float(),dim=-1)).mean(-1).cpu())
        errors.append(torch.cat(parts))
totals=[float(x.mean()) for x in errors];selected=min(range(5),key=lambda i:(totals[i],i));calibration=dict(selected_strength=grid[selected],grid=grid,Dfit_token_errors=totals,Dfit_queries=820,Dcheck_or_V_used=False)
save(out/'component_final.pt',dict(model=s['model'],calibration=calibration,parameters=525568,case=a.case,updates=2481,passes=3,seen_queries=39690,full_parent_sha256=sha(parent/'brain_final.pt')))
save(out/'Dfit_errors.pt',dict(array_rows=ids,errors=torch.stack(errors),grid=grid));write(out/'completion.json',dict(status='COMPLETE_MATCHING_ORDINARY_AND_FIXED_DFIT_CALIBRATION',case=a.case,checkpoint_sha256=sha(out/'component_final.pt'),calibration=calibration,seconds=time.monotonic()-start,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated()));print(json.dumps(calibration),flush=True)
