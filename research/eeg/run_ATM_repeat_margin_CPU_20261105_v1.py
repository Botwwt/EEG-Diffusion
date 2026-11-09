"""Independent CPU64 neural fixture and explicit pair mass reduction."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time
import numpy as np
import atm_repeat_margin_contract_20261105_v1 as C
import atm_repeat_margin_core_20261105_v1 as F
start=time.monotonic();c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical']
assert K.ref(__file__)==q['sources']['CPU'];C.resources(c,True)
done=K.read(out/'scientific_completion.json');assert done['status']=='COMPLETE_FULL_T_FROZEN_REPEAT_MARGIN'
f=np.load(K.checked(done['fixture']));s=np.load(K.checked(done['statistics']));meta=K.read(K.checked(done['parameters']))
assert np.array_equal(s['T'],c['T']) and np.all(s['coverage']==1) and s['coverage'].shape==(4,13230,5)
order=np.random.default_rng(1739).permutation(1323);pos=(order[:,None]*10+np.arange(10)).reshape(-1)
assert np.array_equal(pos,s['positions']) and np.array_equal(f['rows'],s['T'][pos[:160]])
assert np.max(abs(s['numerators'].sum(-1)-s['denominators'][...,None]))<1e-14
assert np.allclose(s['numerators'][:,:,0],s['numerators'][:,:,1]+s['numerators'][:,:,2],atol=1e-14)
assert np.allclose(s['denominators'][:,:,0],s['denominators'][:,:,1]+s['denominators'][:,:,2],atol=1e-14)
assert np.isfinite(s['gradient_grams']).all() and np.isfinite(s['global_gradient_grams']).all()
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
obj=np.load(K.checked(pq['physical_source']),allow_pickle=True);obj=obj if isinstance(obj,dict) else obj.item()
rows=f['rows'];block=obj['preprocessed_eeg_data'][rows].astype(np.float32);del obj
target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True)[rows].double()
Y=torch.nn.functional.normalize(target,dim=1);results={}
for ai,arm in enumerate(F.STATES):
    assert time.monotonic()-start<850
    state=torch.load(K.checked(q['checkpoints'][arm]),map_location='cpu',weights_only=True,mmap=True)
    net=ATMS().double().eval();net.load_state_dict(state['model'],strict=True)
    actual=[];pred=[]
    with torch.no_grad():
        for r in range(4):
            z=net(torch.tensor(block[:,r],dtype=torch.float64),torch.ones(160,dtype=torch.long))
            actual.append((net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T).numpy());pred.append(z[:8].numpy())
    z=net(torch.tensor(block.mean(1),dtype=torch.float64),torch.ones(160,dtype=torch.long))
    logits=net.logit_scale.exp().clamp_max(100)*torch.nn.functional.normalize(z,dim=1)@Y.T
    actual.append(logits.detach().numpy());pred.append(z[:8].detach().numpy());actual=np.stack(actual)
    logerr=float(np.max(abs(actual-f[arm+'_logits'])));prederr=float(np.max(abs(np.stack(pred)-f[arm+'_first8_predictions'])))
    assert logerr<.001 and prederr<.0001,(arm,logerr,prederr)
    ll=F.torch_losses(logits);errors=[];normerrors=[]
    for ki,loss in enumerate(ll):
        g=torch.autograd.grad(loss,tuple(net.parameters()),retain_graph=(ki==0),allow_unused=True)
        active=np.array([v is not None for v in g]);assert np.array_equal(active,f[arm+'_active'])
        flat=torch.cat([(torch.zeros_like(p) if v is None else v).reshape(-1) for p,v in zip(net.parameters(),g)]).numpy()
        ref=f[arm+'_gradient_samples'][ki];err=float(np.max(abs(flat[f['sampled_parameter_indices']]-ref)));assert err<.0001
        norms=np.array([float(flat[a:b]@flat[a:b]) for a,b in meta['slices']]);old=f[arm+'_tensor_squared_norms'][ki]
        relative=float(np.linalg.norm(norms-old)/(np.linalg.norm(old)+1e-30));assert relative<.002
        errors.append(err);normerrors.append(relative)
    # Independent, explicit split and anchor reduction from saved original GPU logits.
    saved=f[arm+'_logits'].astype(np.float64);n=160;s4=saved[4];diag=np.diag(s4);off=~np.eye(n,dtype=bool)
    ex=np.exp(s4-s4.max(1,keepdims=True));row=ex/ex.sum(1,keepdims=True)/(2*n)
    ex=np.exp(s4-s4.max(0,keepdims=True));col=ex/ex.sum(0,keepdims=True)/(2*n)
    weights=[(row,col),(np.exp(-np.logaddexp(0.,diag[:,None]-s4))/(2*n*(n-1)),np.exp(-np.logaddexp(0.,diag[None,:]-s4))/(2*n*(n-1))),
        (np.ones_like(s4)/(2*n*(n-1)),np.ones_like(s4)/(2*n*(n-1)))]
    check=np.zeros((n,3,3,2,4));den=np.zeros((n,3,3));same=np.arange(n)[:,None]//10==np.arange(n)[None,:]//10
    for ci in range(2):
        perm=np.arange(n) if ci==0 else (np.arange(n)//10)*10+(np.arange(n)%10+1)%10
        ss=saved[:4][:,:,perm]
        for orientation in range(2):
            d=np.diagonal(ss,axis1=1,axis2=2);margin=(d[:,:,None] if orientation==0 else d[:,None,:])-ss
            for left,right in F.SPLITS:
                a=(margin[left[0]]+margin[left[1]])/2;b=(margin[right[0]]+margin[right[1]])/2
                cats=((a>0)&(b>0),(a<0)&(b<0),a*b<0,(a==0)|(b==0))
                for ri,mask in enumerate((off,off&same,off&~same)):
                    for mi,pair in enumerate(weights):
                        w=pair[orientation]*mask
                        if ci==0 and left==(0,1):den[:,ri,mi]+=w.sum(1 if orientation==0 else 0)
                        for ki,cat in enumerate(cats):check[:,ri,mi,ci,ki]+=(w*cat).sum(1 if orientation==0 else 0)/3
    err=float(np.max(abs(check-s['numerators'][ai,pos[:160]])));derr=float(np.max(abs(den-s['denominators'][ai,pos[:160]])))
    assert err<1e-14 and derr<1e-14,(err,derr)
    # Report FP precision changes; do not substitute recomputed signs into primary counts.
    re,_=F.measure(actual);near=float(np.max(abs(re-check)))
    results[arm]=dict(CPU64_logit_max_error=logerr,first8_prediction_max_error=prederr,gradient_sample_max_errors=errors,
        tensor_squared_norm_relative_errors=normerrors,independent_mass_max_error=err,independent_denominator_max_error=derr,
        CPU64_vs_original_FP32_mass_max_change=near,original_FP32_primary_retained=True)
    del net,state
K.write(out/'independent_CPU_audit.json',dict(status='PASS_FIXED3200_CPU64_AND_ALL_T_COVERAGE_MASS',results=results,
    CPU_neural_forwards=3200,CPU_parameter_backward_calls=8,all264600_neural_outputs_NOT_replayed=True,
    math=F.math_checks(),seconds=time.monotonic()-start))
C.resources(c)
