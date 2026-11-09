"""Independent functional F rewrite and full-batch autograd replay of D36."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def check(x):
    p=Path(x['path']);assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'];return p
def rss():return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:')))*1024

def autograd_readout(c,target,temperature):
    leaf=c.detach().clone().requires_grad_(True)
    logits=temperature*F.normalize(leaf,dim=-1)@target.T
    labels=torch.arange(len(leaf))
    row=F.cross_entropy(logits,labels,reduction='none')
    col=F.cross_entropy(logits.T,labels,reduction='none')
    losses=.5*(row+col)
    gradient=torch.autograd.grad(losses.sum(),leaf)[0]
    return losses.detach(),gradient

def run(spec_path,primary_path,output):
    start=time.monotonic();base=rss();torch.set_num_threads(2);assert not torch.cuda.is_initialized()
    spec=read(spec_path);done=read(primary_path)
    assert done['status']=='D36_COMPLETE_NATIVE_TEACHER_FULL_T_FROZEN_MOMENTS' and sha(spec_path)==done['spec']['sha256']
    assert not output.exists();output.mkdir(parents=True)
    files={Path(x['path']).stem:x for x in done['files']}
    for item in done['files']:check(item)
    actual=np.load(files['all13230x13_query_metrics']['path'],allow_pickle=False)
    losses=np.load(files['all13230x4_losses']['path'],allow_pickle=False)
    concept_actual=np.load(files['all1323x13_concept_metrics']['path'],allow_pickle=False)
    order=np.load(files['all13230_batch_order']['path'],allow_pickle=False)
    assert np.array_equal(order,np.random.default_rng(1729).permutation(13230))
    inputs=torch.load(check(spec['T_visual_embeddings']),map_location='cpu',mmap=True,weights_only=True)
    condition=np.load(check(spec['native_conditions']),mmap_mode='r',allow_pickle=False)
    cached=np.load(check(spec['cached_T_targets']),mmap_mode='r',allow_pickle=False)
    config=read(check(spec['teacher_config']))
    sys.path[:0]=['/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages','/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages',
                 '/root/mced_representation_runtime_v1/lib/python3.12/site-packages','/root/mced_eeg_brainhive_parquet_packages_v1']
    from safetensors.torch import load_file
    state=load_file(str(check(spec['teacher_weights'])))
    def visual(rows,training):
        projected=[]
        for name in config['proj_meta']:
            prefix='vision_model.proj_maps.'+name
            projected.append(F.linear(inputs[name].index_select(0,rows),state[prefix+'.weight'],state[prefix+'.bias']))
        x=torch.stack(projected,dim=0).sum(dim=0)
        p='vision_model.block.layers.0.'
        h=F.gelu(F.linear(x,state[p+'ff.w1.weight']))
        h=F.dropout(h,p=.1,training=training)
        z=x+F.linear(h,state[p+'ff.w2.weight'])
        z=F.layer_norm(z,(1024,),state[p+'norm.weight'],state[p+'norm.bias'],eps=1e-6)
        return F.normalize(z.double(),dim=-1)
    snapshot=torch.load(check(spec['parent_checkpoint']),map_location='cpu',weights_only=True)
    temp=float(snapshot['logit_scale'].double().exp());assert temp==done['temperature']
    derived=np.empty_like(actual);loss_derived=np.empty_like(losses);bank_error=0.;draw_error=0.
    logs=read(Path(files['all416_draw_readouts']['path']))
    stored_banks={name:np.load(files[name]['path'],mmap_mode='r',allow_pickle=False) for name in [
        'deterministic_FP32_unit_targets','mean_stochastic_unit_targets','mean_stochastic_gradient',
        'deterministic_gradient','mean_target_gradient','cached_target_gradient']}
    for bid,lo in enumerate(range(0,13230,1024)):
        nodes=order[lo:lo+1024];ids=torch.tensor(nodes,dtype=torch.long)
        c=torch.tensor(np.asarray(condition[nodes]),dtype=torch.float64)
        with torch.no_grad():det=visual(ids,False);cache=F.normalize(torch.tensor(np.asarray(cached[nodes]),dtype=torch.float64),dim=-1)
        Ldet,gdet=autograd_readout(c,det,temp);Lcache,gcache=autograd_readout(c,cache,temp)
        sumV=torch.zeros_like(det);sumG=torch.zeros_like(det);sumL=torch.zeros(len(nodes),dtype=torch.float64)
        g2=torch.zeros(len(nodes),dtype=torch.float64);first=torch.zeros_like(det)
        for draw in range(32):
            seed=1729+bid*4096+draw
            torch.random.set_rng_state(torch.Generator(device='cpu').manual_seed(seed).get_state())
            with torch.no_grad():target=visual(ids,True)
            loss,gradient=autograd_readout(c,target,temp)
            record=logs[bid*32+draw];assert record['seed']==seed and record['draw']==draw and record['batch']==bid
            draw_error=max(draw_error,abs(float(loss.mean())-record['mean_loss']))
            sumV+=target;sumG+=gradient;sumL+=loss;g2+=gradient.square().sum(1)
            if draw<16:first+=gradient
            assert time.monotonic()-start<850 and rss()-base<10<<30
        vbar=sumV/32;gbar=sumG/32;Lbar=sumL/32
        Lmean,gmean=autograd_readout(c,vbar,temp)
        a=first/16-gdet;b=(sumG-first)/16-gdet;norm=gdet.norm(dim=1).clamp_min(1e-30)
        SE=((g2-32*gbar.square().sum(1)).clamp_min(0)/(32*31)).sqrt()
        positive_leaf=c.detach().clone().requires_grad_(True)
        positive=torch.autograd.grad((temp*F.normalize(positive_leaf,dim=-1)*det).sum(),positive_leaf)[0]
        cosine=lambda x,y:(x*y).sum(1)/(x.norm(dim=1)*y.norm(dim=1)).clamp_min(1e-30)
        eta=((gdet-gbar)*positive).sum(1)/positive.square().sum(1).clamp_min(1e-30)
        values=torch.stack((Lbar-Lmean,Lbar-Ldet,(gbar-gdet).norm(dim=1)/norm,(gmean-gdet).norm(dim=1)/norm,
            (gbar-gmean).norm(dim=1)/norm,cosine(a,b),SE/norm,cosine(gbar,gdet),(gcache-gdet).norm(dim=1)/norm,
            vbar.norm(dim=1),(vbar-det).norm(dim=1),eta,SE*32**.5/norm),dim=-1)
        derived[nodes]=values.numpy();loss_derived[nodes]=torch.stack((Lcache,Ldet,Lbar,Lmean),dim=-1).numpy()
        for name,x in zip(stored_banks,[det,vbar,gbar,gdet,gmean,gcache]):bank_error=max(bank_error,float(np.max(np.abs(x.float().numpy()-stored_banks[name][nodes]))))
        print(json.dumps(dict(batch=bid,all32_native_teacher_and_autograd_gradients_replayed=True,seconds=time.monotonic()-start)),flush=True)
    error=float(np.max(np.abs(actual-derived)));loss_error=float(np.max(np.abs(losses-loss_derived)))
    assert error<1e-10 and loss_error<1e-10 and draw_error<1e-10 and bank_error<1e-7
    groups={}
    for i,line in enumerate(check(spec['T_index']).read_text().splitlines()):
        item=json.loads(line);assert item['row']==i and item['partition']=='T';groups.setdefault(item['concept_id'],[]).append(i)
    concepts=np.stack([derived[rows].mean(0) for rows in groups.values()]);assert np.max(np.abs(concepts-concept_actual))<1e-10
    inds=np.random.default_rng(1729).integers(0,1323,size=(1000,1323));draws=np.concatenate([concepts[inds[lo:lo+20]].mean(1) for lo in range(0,1000,20)])
    mean=concepts.mean(0);low,high=np.quantile(draws,[.025,.975],axis=0)
    for expected,key in [(mean,'mean_metrics'),(low,'descriptive_concept_95_lower'),(high,'descriptive_concept_95_upper')]:assert np.max(np.abs(expected-done[key]))<1e-10
    gate=bool(mean[2]>=.01 and mean[2]>2*mean[6] and mean[5]>=.8 and mean[8]<.5*mean[2]);assert gate==done['registered_measurable_teacher_gradient_shift_gate']
    result=dict(status='D36_ALL416_STOCHASTIC_NATIVE_TEACHER_BATCHES_AND_FULL_SYMMETRIC_AUTOGRAD_REPLAY_VERIFIED',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),primary_completion_sha256=sha(primary_path),source_sha256=sha(__file__),
        all13230x32_stochastic_teacher_sample_evaluations=True,all13_complete_batches_32_draws_independently_replayed=True,
        functional_F_primitive_rewrite_without_producer_import=True,symmetric_condition_gradients_by_independent_autograd=True,
        all_six_FP32_banks_verified=True,maximum_metric_abs_error=error,maximum_loss_abs_error=loss_error,
        maximum_draw_mean_loss_error=draw_error,maximum_saved_FP32_bank_abs_error=bank_error,
        all1323_concepts_1000_bootstrap_and_gate_recomputed=True,registered_gate_recomputed=gate,
        original_historical_GPU_BF16_training_replayed=False,raw_EEG_encoder_forward_repeated=False,
        new_models_fit=0,new_GPU_jobs=0,new_images=0,CUDA_initialized=torch.cuda.is_initialized(),seconds=time.monotonic()-start,peak_extra_CPU_bytes=rss()-base)
    assert not result['CUDA_initialized']
    with (output/'completion.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--spec',type=Path,required=True);p.add_argument('--completion',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();run(args.spec,args.completion,args.output)
