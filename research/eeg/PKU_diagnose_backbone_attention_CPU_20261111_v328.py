"""Diagnose the frozen separated-model export/replay failure without fitting or scoring."""
import json,os,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
torch.set_num_threads(4);start=time.monotonic();out=R/'runs/EEG_backbone_attention_diagnostic_v328';out.mkdir()
split,T,folds,post=input_contract();bank=torch.load(R/'runs/EEG_images_v271/condition_bank.pt',map_location='cpu',weights_only=True);rows=bank['array_rows'];raw=torch.load(R/'data/recovery_20261108_v236/sub-03_train_mean4.pt',map_location='cpu',weights_only=True,mmap=True);x=raw[rows].clone();del raw
parent=ROOT/'Separated63/encoders/full';s=torch.load(parent/'brain_final.pt',map_location='cpu',weights_only=True);model=make('Separated63').eval().requires_grad_(False);model.load_state_dict(s['model'],strict=True);cache=torch.load(parent/'conditions.pt',map_location='cpu',weights_only=True);where={r:i for i,r in enumerate(cache['array_rows'].tolist())};ref=cache['condition'][[where[r] for r in rows]]
def describe(t):
    good=torch.isfinite(t);return dict(shape=list(t.shape),finite=int(good.sum()),nonfinite=int((~good).sum()),nonfinite_queries=torch.where(~good.reshape(len(t),-1).all(1))[0].tolist(),max_finite_abs=float(t[good].abs().max()) if good.any() else None)
def explicit(xx):
    C=model.channels;h=model.patch(xx.reshape(len(xx),C,10,25).transpose(1,2))+model.channel_pos+model.time_pos;h=h.flatten(1,2)
    for i,block in enumerate(model.blocks):
        n,l,d=h.shape;q,k,v=block.qkv(block.norm1(h)).reshape(n,l,3,4,32).permute(2,0,3,1,4).unbind(0);mask=model.space_allowed if i==0 else model.time_allowed
        scores=(q@k.transpose(-2,-1))/math.sqrt(32);scores=scores.masked_fill(~mask,float('-inf'));a=(scores.softmax(-1)@v).transpose(1,2).reshape(n,l,d);h=h+block.out(a);h=h+block.ff(block.norm2(h))
    h=model.input_proj(h.reshape(len(xx),10,C,128).mean(2).reshape(len(xx),1280));return model.out_norm(h+model.res(h))
report=dict(CPU_job=os.environ['SLURM_JOB_ID'],cache_all=describe(cache['condition']),cache_Dcheck=describe(ref),input_Dcheck=describe(x),model_parameters_finite=all(torch.isfinite(v).all() for v in s['model'].values()),masks=dict(space_allowed_min=int(model.space_allowed.sum(1).min()),time_allowed_min=int(model.time_allowed.sum(1).min())),cases=[])
with torch.inference_mode():
    for batch in [1,4,16,64]:
        xx=x[:64];default=torch.cat([model(xx[i:i+batch]) for i in range(0,len(xx),batch)])
        with torch.backends.cuda.sdp_kernel(enable_flash=False,enable_math=True,enable_mem_efficient=False):mathmode=torch.cat([model(xx[i:i+batch]) for i in range(0,len(xx),batch)])
        manual=torch.cat([explicit(xx[i:i+batch]) for i in range(0,len(xx),batch)])
        d=dict(batch=batch,default=describe(default),mathmode=describe(mathmode),explicit=describe(manual));valid=torch.isfinite(manual)&torch.isfinite(ref[:64]);d['explicit_vs_GPU_cache_max_scaled']=float(((manual-ref[:64]).abs()/(1+ref[:64].abs()))[valid].max()) if valid.any() else None;valid=torch.isfinite(manual)&torch.isfinite(mathmode);d['explicit_vs_math_max_scaled']=float(((manual-mathmode).abs()/(1+manual.abs()))[valid].max()) if valid.any() else None;report['cases'].append(d)
report.update(status='FROZEN_ATTENTION_NUMERICAL_DIAGNOSIS_COMPLETE_NO_METHOD_FIT_OR_SCORE',seconds=time.monotonic()-start,trained_model_sha256=sha(parent/'brain_final.pt'),cache_sha256=sha(parent/'conditions.pt'),new_training=0,new_images=0,source_sha256=sha(__file__));write(out/'completion.json',report);print(json.dumps(report),flush=True)
