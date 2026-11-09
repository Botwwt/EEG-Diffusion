"""CPU64 whole-model derivatives on fixed T8, all15 subsets and all3 points."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
import atm_full_repeat_gradient_contract_20261105_v1 as C
import atm_full_repeat_gradient_core_20261105_v1 as F
started=time.monotonic();c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical']
assert K.ref(__file__)==q['sources']['CPU'];C.resources(c,starting=True)
done=K.read(out/'scientific_completion.json');assert done['status']=='COMPLETE_FULL_T_AND_FIXED_NEIGHBORHOOD_WHOLE_ENCODER_GRADIENTS'
f=np.load(K.checked(done['fixture']),allow_pickle=False);a=np.load(K.checked(done['statistics']),allow_pickle=False)
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
torch.set_num_threads(2)
obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
if not isinstance(obj,dict):obj=obj.item()
assert np.array_equal(obj['preprocessed_eeg_data'][f['rows']],f['raw']);del obj
targets=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()
assert np.array_equal(targets[(f['rows']//10)[:,None]*10+np.arange(10)],f['targets10'])
state=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)['model']
net=ATMS().double().eval();net.load_state_dict({k:v.double() if v.is_floating_point() else v for k,v in state.items()},strict=True)
meta=F.metadata(net);assert np.array_equal(meta['indices'],f['sampled_parameter_indices'])
max_output=max_sample=max_norm_relative=max_loss=0.;summary={}
for point,sign in [('base',0),('plus',1),('minus',-1)]:
    if sign:F.perturb(net,meta,state,sign*q['parameter_step_L2']*direction)
    g,r=F.collect(net,meta,f['raw'],f['targets10'],f['rows']%10,'cpu',torch.float64,record=True)
    if point=='base':direction=g[3,2]/torch.linalg.vector_norm(g[3,2])
    max_output=max(max_output,float(abs(r['predictions']-f[point+'_predictions']).max()))
    max_sample=max(max_sample,float(abs(r['gradient_samples']-f[point+'_gradient_samples']).max()))
    max_loss=max(max_loss,float(abs(r['losses']-f[point+'_losses']).max()))
    expected=f[point+'_tensor_squared_norms'];nonzero=expected>1e-8
    relative=abs(r['tensor_squared_norms'][nonzero]-expected[nonzero])/expected[nonzero]
    max_norm_relative=max(max_norm_relative,float(relative.max()))
    assert np.max(abs(r['tensor_squared_norms'][~nonzero]-expected[~nonzero]),initial=0)<1e-8
    ev=F.evidence(g,meta);summary[point]=ev['comparison']
assert max_output<5e-5 and max_sample<2e-5 and max_norm_relative<.005 and max_loss<2e-6
assert np.array_equal(a['T'],c['T']) and np.all(a['coverage']==1)
readout=K.read(done['readout']['path']);gram=a['gradient_gram'];assert gram.shape==(4,3,4,3)
weights=[np.array([0,1/3,0,2/3]),np.array([-1/27,4/9,0,16/27]),np.array([0,0,0,1])]
maximum_gram_error=0.
for k,key in enumerate(F.OBJECTIVES):
    H=gram[:,k,:,k];true=np.eye(4)[2]
    for name,w in zip(F.METHODS,weights):
        d=w-true
        vals=dict(absolute_L2_error=float(np.sqrt(max(0,d@H@d))),relative_L2_error=float(np.sqrt(max(0,d@H@d)/H[2,2])),cosine=float(w@H@true/np.sqrt((w@H@w)*H[2,2])))
        for metric,value in vals.items():
            error=abs(value-readout['comparison'][key][name][metric]);maximum_gram_error=max(maximum_gram_error,error);assert error<2e-9
    assert np.allclose(np.diag(H),a['norms'][:,k]**2,rtol=1e-10,atol=1e-12)
K.write(out/'independent_CPU_audit.json',dict(status='PASS_FIXED360_CPU64_WHOLE_PARAMETER_GRADIENT_EVIDENCE_AND_FULL_T_GRAM',
    CPU64_neural_forwards=360,maximum_output_error=max_output,maximum_sampled_gradient_error=max_sample,
    maximum_nontrivial_tensor_squared_norm_relative_error=max_norm_relative,maximum_loss_error=max_loss,
    maximum_full_T_Gram_reconstruction_error=maximum_gram_error,
    per_tensor_sample_count_maximum=8,all_parameter_tensor_squared_norms_checked=True,
    every_gradient_coordinate_compared=False,full_T_neural_gradients_independently_replayed=False,
    profile_gradient_comparisons_CPU64=summary,seconds=time.monotonic()-started))
C.resources(c)
