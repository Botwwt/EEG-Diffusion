"""D36: full-T frozen native teacher-mode/Jensen diagnostic; no neural fitting."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
import argparse
import ast
import datetime
import hashlib
import json
from pathlib import Path
import time
from typing import Dict,List,Tuple,Union
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import brainhive_teacher_moments_20261028_v1 as M

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def checked(x):
    p=Path(x['path']);assert p.is_file() and p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];return p
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024
def module_sha(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):h.update(key.encode());h.update(value.contiguous().numpy().tobytes())
    return h.hexdigest()
def save(p,x):
    with p.open('xb') as f:np.save(f,x,allow_pickle=False)

METRICS=['finite_empirical_Jensen_loss_gap','stochastic_minus_deterministic_loss',
         'stochastic_gradient_shift_ratio','mean_target_gradient_shift_ratio','Jensen_gradient_shift_ratio',
         'split_half_gradient_shift_cosine','MC_standard_error_norm_ratio','stochastic_deterministic_gradient_cosine',
         'cached_deterministic_precision_gradient_ratio','mean_unit_target_concentration',
         'mean_unit_target_deterministic_distance','stochastic_update_positive_alignment','stochastic_gradient_sample_SD_norm_ratio']

def run(spec_path):
    start=time.monotonic();base=rss();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    spec=read(spec_path);assert spec['status']=='D36_NATIVE_TEACHER_MOMENTS_FROZEN_DIAGNOSTIC_LOCKED'
    assert spec['draws']==32 and spec['batch_size']==1024 and spec['seed']==1729
    for item in spec['source_pins']:checked(item)
    folder=Path(spec['output']);assert not folder.exists();folder.mkdir(parents=True)
    def budget():
        assert time.monotonic()-start<850 and rss()-base<10<<30
        assert sum(x.stat().st_size for x in folder.iterdir() if x.is_file())<512<<20
    layers=checked(spec['native_layers']);selected=['BaseMLP','ResNet1DLayer','ResNet1DBlock','FusionEncoder']
    nodes=[x for x in ast.parse(layers.read_text()).body if isinstance(x,ast.ClassDef) and x.name in selected]
    assert [x.name for x in nodes]==selected
    scope=dict(torch=torch,nn=nn,F=F,Dict=Dict,List=List,Tuple=Tuple,Union=Union)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(layers),'exec'),scope)
    config=read(checked(spec['teacher_config']))
    teacher=scope['FusionEncoder'](hidden_size=config['hidden_size'],intermediate_size=config['intermediate_size'],
        proj_meta=config['proj_meta'],num_layers=config['num_layers'],dropout=.1).float().requires_grad_(False)
    # Tensor-only loading avoids changing the shared Diffusers runtime.
    import sys
    sys.path[:0]=['/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages',
                 '/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',
                 '/root/mced_representation_runtime_v1/lib/python3.12/site-packages',
                 '/root/mced_eeg_brainhive_parquet_packages_v1']
    from safetensors.torch import load_file
    weights=load_file(str(checked(spec['teacher_weights'])))
    vision={key[len('vision_model.'):]:value for key,value in weights.items() if key.startswith('vision_model.')}
    assert set(vision)==set(teacher.state_dict());teacher.load_state_dict(vision,strict=True)
    assert sum(x.numel() for x in teacher.parameters())==11015168
    assert len([x for x in teacher.modules() if isinstance(x,nn.Dropout)])==1
    weights_before=module_sha(teacher)
    inputs=torch.load(checked(spec['T_visual_embeddings']),map_location='cpu',mmap=True,weights_only=True)
    assert {k:list(v.shape) for k,v in inputs.items()}=={k:[13230,d] for k,d in config['proj_meta'].items()}
    c=np.load(checked(spec['native_conditions']),mmap_mode='r',allow_pickle=False)
    cached=np.load(checked(spec['cached_T_targets']),mmap_mode='r',allow_pickle=False)
    index=[json.loads(x) for x in checked(spec['T_index']).read_text().splitlines()]
    assert c.shape==cached.shape==(13230,1024) and len(index)==13230
    groups={}
    for i,row in enumerate(index):assert row['row']==i and row['partition']=='T';groups.setdefault(row['concept_id'],[]).append(i)
    assert len(groups)==1323 and all(len(x)==10 for x in groups.values())
    parent=torch.load(checked(spec['parent_checkpoint']),map_location='cpu',weights_only=True)
    temperature=float(parent['logit_scale'].double().exp())
    order=np.random.default_rng(1729).permutation(13230)
    metrics=np.empty((13230,len(METRICS)),np.float64);loss_bank=np.empty((13230,4),np.float64)
    bank_names=['deterministic_FP32_unit_targets','mean_stochastic_unit_targets','mean_stochastic_gradient',
                'deterministic_gradient','mean_target_gradient','cached_target_gradient']
    banks={name:np.lib.format.open_memmap(folder/(name+'.npy'),mode='w+',dtype=np.float32,shape=(13230,1024)) for name in bank_names}
    draw_logs=[];batch_logs=[]
    with torch.inference_mode():
        for bid,lo in enumerate(range(0,13230,1024)):
            rows=order[lo:lo+1024];ids=torch.tensor(rows,dtype=torch.long)
            values={k:v.index_select(0,ids) for k,v in inputs.items()}
            condition=torch.tensor(np.asarray(c[rows]),dtype=torch.float64)
            teacher.eval();det=F.normalize(teacher(**values).double(),dim=-1)
            cache=F.normalize(torch.tensor(np.asarray(cached[rows]),dtype=torch.float64),dim=-1)
            Ldet,gdet=M.symmetric_readout(condition,det,temperature)
            Lcache,gcache=M.symmetric_readout(condition,cache,temperature)
            sumV=torch.zeros_like(det);sumG=torch.zeros_like(gdet);sumL=torch.zeros_like(Ldet)
            squareG=torch.zeros(len(rows),dtype=torch.float64);first=torch.zeros_like(gdet)
            teacher.train();seeds=[]
            for draw in range(32):
                seed=1729+bid*4096+draw;seeds.append(seed)
                torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
                target=F.normalize(teacher(**values).double(),dim=-1)
                loss,gradient=M.symmetric_readout(condition,target,temperature)
                sumV+=target;sumG+=gradient;sumL+=loss;squareG+=gradient.square().sum(1)
                if draw<16:first+=gradient
                draw_logs.append(dict(batch=bid,draw=draw,seed=seed,mean_loss=float(loss.mean())))
                budget()
            vbar=sumV/32;gbar=sumG/32;Lbar=sumL/32
            Lmean,gmean=M.symmetric_readout(condition,vbar,temperature)
            norm=gdet.norm(dim=1).clamp_min(1e-30);shift=gbar-gdet
            half1=first/16-gdet;half2=(sumG-first)/16-gdet
            SE=((squareG-32*gbar.square().sum(1)).clamp_min(0)/(32*31)).sqrt()
            unit=F.normalize(condition,dim=-1);positive=temperature/condition.norm(dim=1,keepdim=True)*(det-unit*(unit*det).sum(1,keepdim=True))
            eta=((gdet-gbar)*positive).sum(1)/positive.square().sum(1).clamp_min(1e-30)
            cosine=lambda a,b:(a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)).clamp_min(1e-30)
            result=torch.stack((Lbar-Lmean,Lbar-Ldet,shift.norm(dim=1)/norm,(gmean-gdet).norm(dim=1)/norm,
                (gbar-gmean).norm(dim=1)/norm,cosine(half1,half2),SE/norm,cosine(gbar,gdet),
                (gcache-gdet).norm(dim=1)/norm,vbar.norm(dim=1),(vbar-det).norm(dim=1),eta,SE*32**.5/norm),dim=-1).numpy()
            assert np.isfinite(result).all() and result[:,0].min()>=-1e-10
            assert result[:,9].max()<=1+1e-10
            metrics[rows]=result;loss_bank[rows]=torch.stack((Lcache,Ldet,Lbar,Lmean),dim=-1).numpy()
            for name,tensor in zip(bank_names,[det,vbar,gbar,gdet,gmean,gcache]):banks[name][rows]=tensor.float().numpy()
            event=dict(batch=bid,rows=len(rows),seeds=seeds,mean_metrics=result.mean(0).tolist(),seconds=time.monotonic()-start)
            batch_logs.append(event)
            print(json.dumps(dict(batch=bid,all32_teacher_draws_complete=True,seconds=time.monotonic()-start)),flush=True)
    for value in banks.values():value.flush()
    concepts=np.stack([metrics[rows].mean(0) for rows in groups.values()]);mean=concepts.mean(0)
    indices=np.random.default_rng(1729).integers(0,1323,size=(1000,1323))
    draws=np.concatenate([concepts[indices[lo:lo+20]].mean(1) for lo in range(0,1000,20)])
    low,high=np.quantile(draws,[.025,.975],axis=0)
    gate=bool(mean[2]>=.01 and mean[2]>2*mean[6] and mean[5]>=.8 and mean[8]<.5*mean[2])
    assert module_sha(teacher)==weights_before and not torch.cuda.is_initialized()
    save(folder/'all13230x13_query_metrics.npy',metrics);save(folder/'all13230x4_losses.npy',loss_bank)
    save(folder/'all1323x13_concept_metrics.npy',concepts);save(folder/'all13230_batch_order.npy',order)
    for name,value in [('batch_readouts.json',batch_logs),('all416_draw_readouts.json',draw_logs)]:
        with (folder/name).open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    files=[ref(p) for p in folder.iterdir() if p.is_file()]
    budget()
    value=dict(status='D36_COMPLETE_NATIVE_TEACHER_FULL_T_FROZEN_MOMENTS',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),spec=ref(spec_path),
        rows=13230,concepts=1323,batches=13,draws_per_batch=32,total_teacher_stochastic_sample_evaluations=13230*32,
        total_stochastic_teacher_batches=416,temperature=temperature,metric_order=METRICS,mean_metrics=mean.tolist(),
        descriptive_concept_95_lower=low.tolist(),descriptive_concept_95_upper=high.tolist(),
        registered_measurable_teacher_gradient_shift_gate=gate,
        decision='MEASURABLE_FROZEN_OBJECTIVE_SHIFT_REQUIRES_MATCHED_CONTINUATION_CONTROL' if gate else 'NOT_PRIORITIZED_BY_CURRENT_FROZEN_OBJECTIVE_SHIFT_GATE',
        teacher_parameters=11015168,teacher_parameter_digest_before=weights_before,teacher_parameter_digest_after=module_sha(teacher),
        compute='CPU FP32 native F forward; float64 normalization, symmetric logits and condition-space gradients; not historical BF16 GPU/dropout-step replay',
        scope='Already-fit sub01 full-T encoder in eval; no held-development effect, no encoder-parameter gradient or generalized architecture conclusion',
        finite_Jensen_identity_uses_unnormalized_mean_unit_targets=True,ordinary_read=False,
        new_models_fit=0,new_SGD_updates=0,new_GPU_jobs=0,new_images=0,automatic_training_or_image_admission=False,
        files=files,seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-base)
    with (folder/'completion.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    print(json.dumps(value),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);args=p.parse_args();run(args.spec)
