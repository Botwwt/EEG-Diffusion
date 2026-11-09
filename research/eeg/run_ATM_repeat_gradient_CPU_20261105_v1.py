"""Original-network CPU64 fixed240 replay and independent autograd affine check."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
import atm_repeat_gradient_contract_20261105_v1 as C
from atm_repeat_gradient_math_20261105_v1 import loss_affine_gradients,compare_gradient_counts,gradient_summary
started=time.monotonic();c=C.setup(sys.argv[1]);q,K,out=c['q'],c['K'],c['out'];pq=c['prior']['q']
assert K.ref(__file__)==q['sources']['CPU'];C.resources(c,starting=True)
from enigma_repeat_task_risk_math_20261105_v1 import SUBSETS,COUNTS,task_metrics,summaries
done=K.read(out/'scientific_completion.json')
assert done['status']=='COMPLETE_ATMS_BOTH_FROZEN_FULL_T_SUBSETS_AND_AFFINE_GRADIENTS'
f=np.load(K.checked(done['fixture']),allow_pickle=False)
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
if not isinstance(obj,dict):obj=obj.item()
assert np.array_equal(obj['preprocessed_eeg_data'][f['rows']],f['raw'])
del obj
targets=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
assert np.array_equal(targets[(f['rows']//10)[:,None]*10+np.arange(10)],f['targets10'])
max_nn=max_autograd=max_metrics=max_count=0.;replayed=0
for arm in q['arms']:
    a=np.load(K.checked(done['arms'][arm]['arrays']),allow_pickle=False)
    assert np.array_equal(a['T'],c['T']) and np.all(a['coverage']==1)
    s,means=summaries(a['per_view'],a['loss_of_mean'],a['dispersion'])
    r=K.read(done['arms'][arm]['readout']['path'])
    assert r['task_risk']==s and r['gradient_risk']==gradient_summary(a['gradient_comparisons'])
    assert np.array_equal(means,a['count_means'])
    state=torch.load(q['checkpoints'][arm]['path'],map_location='cpu',weights_only=True,mmap=True)
    if arm=='nce_continued':state=state['model']
    net=ATMS().double().eval().requires_grad_(False)
    net.load_state_dict({k:v.double() if v.is_floating_point() else v for k,v in state.items()},strict=True)
    gamma=f[arm+'_gamma'];beta=f[arm+'_beta'];gcounts=np.zeros((8,4,3,2048),float)
    for j,subset in enumerate(SUBSETS):
        x=np.ascontiguousarray(f['raw'][:,subset].mean(1)).astype(float)
        with torch.inference_mode():p=net(torch.from_numpy(x),torch.ones(8,dtype=torch.long)).numpy()
        recorded=f[arm+'_prediction'][:,j]
        max_nn=max(max_nn,float(abs(p-recorded).max()));replayed+=8
        metrics=task_metrics(recorded,f['targets10'],f['rows']%10)
        max_metrics=max(max_metrics,float(abs(metrics-a['per_view'][:8,j]).max()))
        loss,g=loss_affine_gradients(recorded,f['targets10'],f['rows']%10,gamma,beta)
        gcounts[:,len(subset)-1]+=g
        # Separate Torch autograd implementation for the actual existing affine coordinates.
        u=torch.from_numpy((recorded.astype(float)-beta)/gamma)
        gg=torch.from_numpy(gamma.copy()).requires_grad_();bb=torch.from_numpy(beta.copy()).requires_grad_()
        z=u*gg+bb;y10=torch.from_numpy(f['targets10'].astype(float));ix=torch.from_numpy(f['rows']%10)
        y=y10[torch.arange(8),ix];v=torch.nn.functional.normalize(y10,dim=-1)
        unit=torch.nn.functional.normalize(z,dim=-1);scores=(unit[:,None]*v).sum(-1)
        losses=[(z-y).square().mean(-1),1-(unit*v[torch.arange(8),ix]).sum(-1),torch.nn.functional.cross_entropy(scores,ix,reduction='none')]
        for k,l in enumerate(losses):
            ag,ab=torch.autograd.grad(l.sum(),(gg,bb),retain_graph=True)
            actual=np.concatenate((ag.numpy(),ab.numpy()))
            max_autograd=max(max_autograd,float(abs(actual-g[:,k].sum(0)).max()))
            assert np.max(abs(l.detach().numpy()-loss[:,k]))<2e-12
    for k,n in enumerate(range(1,5)):gcounts[:,k]/=int((COUNTS==n).sum())
    max_count=max(max_count,float(abs(compare_gradient_counts(gcounts)-a['gradient_comparisons'][:8]).max()))
    del state,net
assert max_nn<5e-5 and max_autograd<2e-11 and max_metrics<2e-12 and max_count<2e-11 and replayed==240
K.write(out/'independent_CPU_audit.json',dict(status='PASS_ATMS_FIXED240_CPU64_AND_AUTOGRAD_ALL_REDUCTIONS',
    CPU64_neural_forwards=240,maximum_neural_error=max_nn,maximum_autograd_gradient_error=max_autograd,
    maximum_fixed_metric_error=max_metrics,maximum_fixed_count_gradient_error=max_count,
    all_T_neural_or_gradient_replay=False,all_T_saved_reductions_recomputed=True,
    scope='Two frozen models; eight fixed T stimuli/model; all15 subsets; existing output LN affine only',
    seconds=time.monotonic()-started))
C.resources(c)
