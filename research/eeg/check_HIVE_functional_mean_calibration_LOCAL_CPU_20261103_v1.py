"""Independent checks of attention gauge, LS optimum and paired preservation."""
import datetime,json
from pathlib import Path
import numpy as np
from scipy.special import softmax
import brainhive_functional_mean_calibration_CPU_20261103_v1 as G
rng=np.random.default_rng(1729)
z=rng.normal(size=(12,4,2048));teacher=rng.normal(size=z.shape)
wk=rng.normal(size=(24,2048))/np.sqrt(2048);wv=rng.normal(size=(24,2048))/np.sqrt(2048);wo=rng.normal(size=(24,24))/np.sqrt(24)
q=rng.normal(size=(12,3,5,8))
fields=G.calibration_fields(teacher.mean(0),z.mean(0));delta=fields['shared_delta']
def attention(z):
 k=(z@wk.T).reshape(12,4,3,8).transpose(0,2,1,3)
 v=(z@wv.T).reshape(12,4,3,8).transpose(0,2,1,3)
 a=softmax((q@k.swapaxes(-1,-2))/np.sqrt(8),axis=-1)
 return (a@v).transpose(0,2,1,3).reshape(12,5,24)@wo.T,a
original,a=attention(z);changed,b=attention(z+delta)
bias=G.output_bias(delta,wv,wo)
gauge_error=float(np.max(np.abs(a-b)));bias_error=float(np.max(np.abs(changed-original-bias)))
paired_error=float(np.max(np.abs((changed[::2]-changed[1::2])-(original[::2]-original[1::2]))))
assert max(gauge_error,bias_error,paired_error)<1e-12
full,a_full=attention(z+fields['matched_full_delta'])
full_paired_change=float(np.linalg.norm((full[::2]-full[1::2])-(original[::2]-original[1::2])))
assert full_paired_change>1e-4
before=float(np.mean((teacher-z)**2));after=float(np.mean((teacher-z-delta)**2))
ls_identity_error=abs(before-after-float(delta@delta)/2048)
assert ls_identity_error<1e-12
assert np.max(np.abs((z+delta-teacher).mean((0,1))))<1e-12
assert abs(np.linalg.norm(G.wrong_field(bias))-np.linalg.norm(bias))<1e-12
assert np.array_equal(z+np.zeros(2048),z)
result=dict(status='PASS_SYNTHETIC_SHARED_TOKEN_LS_AND_NATIVE_ATTENTION_IDENTITIES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),attention_weight_gauge_error=gauge_error,value_output_bias_identity_error=bias_error,paired_response_preservation_error=paired_error,matched_full_shift_does_change_paired_responses=full_paired_change,LS_error_identity=ls_identity_error,field_proof=fields['proof'],zero_shift_exact=True,new_GPU_jobs=0,new_images=0,real_EEG_means_fitted=False,scope='Exact real-vector attention algebra only; floating FP16 hooks and whole-trajectory effects still need real engineering and pictures.')
out=Path(__file__).resolve().parent/'audit/crossdisciplinary_20261028_v1/functional_mean_calibration_math_checks_20261103_v1.json';out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
