"""Independently replay all5 parents and heads for all830 queries before generation."""
import os,socket,json,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
from PKU_ordinary_component_source_v253 import BoundedConditionAdapter
torch.set_num_threads(4);start=time.monotonic();out=R/'runs/EEG_backbone_images_v322';out.mkdir();assert (ROOT/'queue/completion.json').exists()
old=R/'runs/EEG_images_v271';proof=json.loads((old/'conditions_completion.json').read_text());assert sha(old/'condition_bank.pt')==proof['condition_bank_sha256'];source=torch.load(old/'condition_bank.pt',map_location='cpu',weights_only=True)
split,T,folds,post=input_contract();rows=source['array_rows'];raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);x=raw[rows].clone();del raw
values=[];arms=[];checks=[];max_error=0.
for case in CASES:
    parent=ROOT/case/'encoders/full';s=torch.load(parent/'brain_final.pt',map_location='cpu',weights_only=True);model=make(case).eval().requires_grad_(False);model.load_state_dict(s['model'],strict=True);cache=torch.load(parent/'conditions.pt',map_location='cpu',weights_only=True);where={r:i for i,r in enumerate(cache['array_rows'].tolist())}
    with torch.no_grad():c=torch.cat([encode(model,case,x[i:i+16],post) for i in range(0,830,16)])
    reference=cache['condition'][[where[r] for r in rows]];error=float(((c-reference).abs()/(1+reference.abs())).max());assert error<2e-4,(case,error);max_error=max(max_error,error)
    headpath=ROOT/case/'calibration/component_final.pt';h=torch.load(headpath,map_location='cpu',weights_only=True);assert h['full_parent_sha256']==sha(parent/'brain_final.pt');head=BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False);head.load_state_dict(h['model'],strict=True);head.radius=.25*h['calibration']['selected_strength']
    g=s['affine']['gamma'].expand(830,-1);b=s['affine']['beta'].expand(830,-1)
    with torch.no_grad():corrected=torch.cat([head(c[i:i+64],g[i:i+64],b[i:i+64],gate_mode='fixed')[0] for i in range(0,830,64)])
    for arm,condition in [(case+'_base',c),(case+'_ordinary',corrected)]:
        arms.append(arm);values.append(condition);checks.append(dict(case=case,arm=arm,target_sha256=sha(parent/'brain_final.pt') if arm.endswith('_base') else sha(headpath)))
    del model,head,c,corrected
save(out/'condition_bank.pt',dict(condition=torch.stack(values),arms=arms,array_rows=rows,stimulus_ids=source['stimulus_ids'],model_checks=checks,subject='sub-03',repeat_protocol='R4',normal_inference_target_access=False))
write(out/'conditions_completion.json',dict(status='PASS_ALL830_ALL5_MATCHED_BACKBONES_AND_OWN_ORDINARY_HEADS',condition_bank_sha256=sha(out/'condition_bank.pt'),policy_sha256=sha(R/'code/EEG_backbone_image_policy_v322.json'),independent_CPU_encoder_max_scaled_error=max_error,seconds=time.monotonic()-start,new_images=0));print('PASS_ALL5_FULL830_CPU_FORWARD',flush=True)
