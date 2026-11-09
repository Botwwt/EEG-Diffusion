import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import sys,numpy as np
import atm_repeat_step_profile_contract_20261105_v1 as C
c=C.setup(sys.argv[1]);K,q,out=c['K'],c['q'],c['out'];C.resources(c)
z=K.read(out/'scientific_completion.json');assert z['status']=='PASS_REAL_TRAIN_MODE_REPEAT_STEP_PROFILE'
f=np.load(K.checked(z['fixture']),allow_pickle=False);checks=[]
for batch in ('first160','last110'):
 for arm in ('native','repeat_step','sign_control','norm_control'):
  p=batch+'_'+arm+'_';B=f[p+'B'];tau=f[p+'tau'];b=f[p+'b'];coef=np.linalg.solve(B+tau*np.eye(4),b)
  assert np.max(abs(coef-f[p+'coefficient']))<1e-9
  delta=-f[p+'D']*(f[p+'U']@coef);expected=f[p+'original']+delta
  if arm=='native':expected=f[p+'original']
  if arm=='norm_control':expected=f[p+'original']*f[p+'global_correct_norm']/f[p+'global_native_norm']
  assert np.max(abs(expected-f[p+'chosen']))<1e-10
  assert np.array_equal((f[p+'native_actual']+expected-f[p+'original']).astype(np.float32).astype(np.float64),f[p+'actual'])
  checks.append(dict(batch=batch,arm=arm,coefficient_and_sampled_update_pass=True))
for p in z['proof']:assert all(p[k] is True for k in ['R4_gradient_bitexact','BN_equals_one_R4','native_parameters_bitexact','Adam_state_bitexact','all_temporary_updates_discarded'])
K.write(out/'independent_CPU_audit.json',dict(status='PASS_REAL_TRAIN_STEP_ALGEBRA_AND_NATIVE_IDENTITY',checks=checks,
 full_neural_dropout_gradients_CPU_replayed=False,full_vectors_NumPy_crosscheck_on_GPU_process=True,new_models=0,new_images=0))
