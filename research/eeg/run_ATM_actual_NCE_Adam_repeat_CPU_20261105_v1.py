"""Original CPU64 global InfoNCE and saved-state Adam on fixed 160/all15 eval."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']=os.environ['MKL_NUM_THREADS']='2'
import sys,time
from pathlib import Path
import numpy as np
import atm_actual_NCE_Adam_repeat_contract_20261105_v1 as C
started=time.monotonic();c=C.setup(sys.argv[1]);q,K,out,pq=c['q'],c['K'],c['out'],c['physical']
assert K.ref(__file__)==q['sources']['CPU'];C.resources(c,starting=True)
done=K.read(out/'scientific_completion.json');assert done['status']=='COMPLETE_ACTUAL_GLOBAL_NCE_ADAM_ALL_T_TWO_MODES'
f=np.load(K.checked(done['fixture']),allow_pickle=False);a=np.load(K.checked(done['statistics']),allow_pickle=False)
sys.path[:0]=pq['existing_runtime_paths']+['/root/atm_official_20261016']
import torch
from models.atms import ATMS
from atm_conditional_stimulus_objective_20261104_v1 import torch_loss,numpy_loss_gradient
import atm_actual_NCE_Adam_repeat_core_20261105_v1 as F
torch.set_num_threads(2)
obj=np.load(K.checked(pq['physical_source']),allow_pickle=True)
if not isinstance(obj,dict):obj=obj.item()
raw=obj['preprocessed_eeg_data'][f['rows']].astype(np.float32);del obj
target=torch.load(pq['visual_target']['path'],map_location='cpu',weights_only=True,mmap=True).numpy()[f['rows']]
assert np.array_equal(target[:8],f['first8_targets'])
state=torch.load(K.checked(q['checkpoint']),map_location='cpu',weights_only=True,mmap=True)
net=ATMS().double().eval();net.load_state_dict(state['model'],strict=True);meta=F.F.metadata(net)
ref=F.optimizer_reference(meta,state,'cpu')
g,r=F.collect(net,meta,raw,target,state['model'],101729,'cpu',torch.float64,torch_loss,record=True)
ev,spaces=F.measurements(g,meta,r['active_tensors'],ref);mode='eval'
assert np.array_equal(r['active_tensors'],f[mode+'_active_tensors'])
errors={key:float(abs(r[key]-f[mode+'_'+key]).max()) for key in ('losses','first8_predictions','sampled_gradients','R4_logits')}
expected=f[mode+'_tensor_squared_norms'];nz=expected>1e-8
errors['nontrivial_tensor_norm_relative']=float((abs(r['tensor_squared_norms'][nz]-expected[nz])/expected[nz]).max())
assert np.max(abs(r['tensor_squared_norms'][~nz]-expected[~nz]),initial=0)<1e-8
assert errors['losses']<3e-5 and errors['first8_predictions']<5e-5 and errors['sampled_gradients']<2e-5 and errors['R4_logits']<1e-4 and errors['nontrivial_tensor_norm_relative']<.01
update_errors={}
for key,values in ev['samples'].items():
    error=float(abs(values-f[mode+'_'+key+'_samples']).max());update_errors[key]=error
    assert error<(2e-5 if 'gradient' in key else 3e-6)
assert np.array_equal(a['T'],c['T']);order=np.random.default_rng(1739).permutation(1323);assert np.array_equal(order,a['concept_order'])
assert np.array_equal(a['positions'],(order[:,None]*10+np.arange(10)).reshape(-1))
readout=K.read(done['readout']['path']);max_gram=0.
for mode in q['modes']:
    assert np.all(a[mode+'_coverage']==1)
    for key in ('raw_gradient','clipped_gradient','Adam_data_update','Adam_total_update'):
        H=a[mode+'_'+key+'_gram'];assert H.shape==(7,7)
        for j,name in enumerate(F.VARIANTS):
            d=np.eye(7)[j]-np.eye(7)[2]
            expected=dict(norm=float(np.sqrt(H[j,j])),relative_L2_error=float(np.sqrt(max(0,d@H@d)/H[2,2])),cosine=float(H[j,2]/np.sqrt(H[j,j]*H[2,2])))
            for field,value in expected.items():
                error=abs(value-readout['modes'][mode][key][name][field]);max_gram=max(max_gram,error);assert error<2e-8
    manual=numpy_loss_gradient(f[mode+'_R4_logits'],'nce_continued');assert abs(manual['loss']-f[mode+'_losses'][-1])<3e-5
K.write(out/'independent_CPU_audit.json',dict(status='PASS_NATIVE_NCE_FIXED2400_EVAL_CPU64_AND_ADAM_GRAM',
    CPU64_eval_forwards=2400,errors=errors,sampled_update_errors=update_errors,maximum_Gram_error=max_gram,
    original_native_objective_used=True,saved_830_Adam_moments_used=True,
    train_mode_GPU_dropout_NOT_replayed_on_CPU=True,full_T_neural_gradients_NOT_replayed=True,
    all_gradient_coordinates_NOT_compared=True,CPU_profile_summary=ev['summary'],seconds=time.monotonic()-started))
C.resources(c)
