"""Check the fitted joint model against independent full conditioning."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
import brainhive_conditional_innovation_diagnostic_20261027_v1 as V
import brainhive_posterior_response_math_20261027_v1 as M

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists()
start=time.monotonic();torch.set_num_threads(2);torch.manual_seed(1729)
u=torch.randn(600,128,dtype=torch.float64)*torch.linspace(.2,2.,128)
c=u[:,:6]@torch.randn(6,6,dtype=u.dtype)+torch.randn(600,6,dtype=u.dtype)*.4
e=u[:,:8]+torch.randn(600,8,dtype=u.dtype)*.6
model=V.fit_joint(u,c,e);testu=u[:11]+.1;testc=c[:11]+.2;teste=e[:11]-.1
base,joint,_=V.clean_means(model,testc,teste)
z=torch.cat([V.standardize(x)[0] for x in (u,c,e)],1).numpy();s=.9*z.T@z/len(z)+.1*np.eye(z.shape[1])
obs=torch.cat([(testc-model['condition_mean'])/model['condition_std'],(teste-model['eeg_mean'])/model['eeg_std']],1).numpy()
expected=obs@np.linalg.solve(s[128:,128:],s[128:,:128]);expected_cov=s[:128,:128]-s[:128,128:]@np.linalg.solve(s[128:,128:],s[128:,:128])
assert np.allclose(joint.numpy(),expected,atol=1e-10) and np.allclose(model['joint_cov'].numpy(),expected_cov,atol=1e-10)
checks={'Schur_fitting_equals_independent_full_covariance_conditioning':True}
ut=(testu-model['image_mean'])/model['image_std'];noise=torch.randn_like(ut)
for alpha,sigma in ((.01,.99995),(.5,.8660254),(.98,.1989975)):
    noisy=alpha*ut+sigma*noise/model['image_std'];prediction,cov=V.noisy_update(joint,model['joint_cov'],noisy,alpha,sigma,model['image_std'])
    rhs=(noisy-alpha*joint).numpy();prior=model['joint_cov'].numpy();noisecov=np.diag((sigma/model['image_std'].numpy())**2)
    # Independent joint conditioning of [u,w] around each query's EEG posterior.
    gain=np.linalg.solve(alpha**2*prior+noisecov,alpha*prior).T
    direct=joint.numpy()+rhs@gain.T
    assert np.allclose(prediction.numpy(),direct,atol=1e-10)
    expected_cov=prior-alpha*gain@prior
    assert np.allclose(cov.numpy(),expected_cov,atol=1e-10)
    values=V.score(prediction,ut,cov,model['image_std']);assert all(torch.isfinite(x).all() for x in values.values())
checks['all3_noise_levels_respect_nonisotropic_PC_noise_and_full_joint_solution']=True
alpha,sigma=.5,.8;noisy=alpha*ut+sigma*noise/model['image_std']
right,_=V.noisy_update(joint,model['joint_cov'],noisy,alpha,sigma,model['image_std'])
wrong,_=V.noisy_update(joint,model['joint_cov'],noisy,alpha,sigma,torch.ones_like(model['image_std']))
assert float((right-wrong).abs().max())>.01
checks['wrong_isotropic_noise_after_PC_standardization_detected']=True
qs=np.repeat(np.arange(82),10);subjects=[f'sub-{i:02d}' for i in range(1,11)]
scores=np.ones((10,820,4,4,3));scores[:,:,:,1,0]=.98;scores[:,:,:,1,1]=.99
positive=V.readout(scores,qs,subjects);assert positive['conditional_increment_supported_in_registered_subspace']
scores[:,:,:,1,0]=1.02;scores[:,:,:,1,1]=1.02
negative=V.readout(scores,qs,subjects);assert not negative['conditional_increment_supported_in_registered_subspace'] and not negative['automatic_new_image_admission']
checks['fixed_readout_accepts_joint_gain_and_rejects_harm_without_image_admission']=True
assert not torch.cuda.is_initialized()
value=dict(status='CONDITIONAL_INNOVATION_JOINT_FIT_AND_NOISE_MATH_CPU_PASSED',checks=checks,check_count=len(checks),
           production_input_dimensions_not_exercised=True,EEG_fit=False,GPU_jobs=0,new_images=0,seconds=time.monotonic()-start)
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
