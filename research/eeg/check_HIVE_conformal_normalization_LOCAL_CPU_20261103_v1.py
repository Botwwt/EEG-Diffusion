"""Meaningful inverse/chord/Jacobian/fit tests before any real EEG fit."""
import datetime, hashlib, json
from pathlib import Path
import numpy as np
import brainhive_conformal_normalization_CPU_20261103_v1 as G

root = Path(__file__).resolve().parent
rng = np.random.default_rng(1729)
x = G.unit(rng.normal(size=(256, 9)))
a = G.unit(rng.normal(size=9)) * .62
y = G.mobius(x, a)
norm_error = float(np.max(np.abs(np.linalg.norm(y,axis=1)-1)))
inverse_error = float(np.max(np.abs(G.mobius(y,-a)-x)))
assert norm_error < 1e-12 and inverse_error < 1e-12
assert np.array_equal(G.mobius(x,np.zeros(9)),x)
before = np.linalg.norm(x[::2]-x[1::2],axis=1)**2
after = np.linalg.norm(y[::2]-y[1::2],axis=1)**2
scales = G.conformal_scale(x,a)
chord_error = float(np.max(np.abs(after-before*scales[::2]*scales[1::2])))
assert chord_error < 1e-12
direction = rng.normal(size=x.shape)
direction -= (direction*x).sum(1)[:,None]*x
direction = G.unit(direction)
finite = (G.mobius(G.unit(x+1e-6*direction),a)-G.mobius(G.unit(x-1e-6*direction),a))/2e-6
jacobian_error = float(np.max(np.abs(np.linalg.norm(finite,axis=1)-scales)))
assert jacobian_error < 1e-8
loss, gradient = G.barycenter_potential(a,x)
potential_FD_error=0.
for k in range(9):
 plus,minus=a.copy(),a.copy();plus[k]+=1e-6;minus[k]-=1e-6
 fd=(G.barycenter_potential(plus,x)[0]-G.barycenter_potential(minus,x)[0])/2e-6
 potential_FD_error=max(potential_FD_error,abs(fd-gradient[k]))
assert potential_FD_error < 1e-8
base = np.concatenate((np.eye(9),-np.eye(9)))
known = G.mobius(base,a)
fitted, proof = G.fit_barycenter(known)
known_center_error=float(np.max(np.abs(fitted-a)))
assert proof['converged_center'] and known_center_error < 1e-6
b = G.zero_mean_basis(10)
basis_error=float(max(np.abs(b.sum(0)).max(),np.abs(b.T@b-np.eye(9)).max()))
assert basis_error < 1e-12
permutation=np.arange(len(x)).reshape(-1,2)[:,::-1].reshape(-1)
models, pair_proof=G.fit_pair(x,y,permutation)
assert len(models)==4
reconstruction_error=float(np.max(np.abs(G.transform(x,models['conformal_paired'])-y)))
assert reconstruction_error < 1e-5
crossratio_error=0.
for i in range(0,256,4):
 def ratio(z):
  p,q,r,s=z[i:i+4]
  return np.linalg.norm(p-q)*np.linalg.norm(r-s)/(np.linalg.norm(p-r)*np.linalg.norm(q-s))
 crossratio_error=max(crossratio_error,abs(ratio(y)-ratio(x)))
assert crossratio_error < 1e-12
result=dict(status='PASS_SYNTHETIC_CONFORMAL_MATH_NOT_REAL_EEG',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),norm_error=norm_error,inverse_error=inverse_error,chord_error=chord_error,tangent_jacobian_FD_error=jacobian_error,potential_FD_error=potential_FD_error,known_center_error=known_center_error,basis_error=basis_error,known_pair_transform_error=reconstruction_error,crossratio_error=crossratio_error,barycenter_proof=proof,prior_art='Rezende2020 Eq8; CantarellaSchumacher2020 Eq25. No new theorem; our finite L-BFGS is not their intrinsic Newton algorithm.',real_fit=False,new_GPU_jobs=0,new_images=0)
out=root/'audit/crossdisciplinary_20261028_v1/conformal_normalization_synthetic_math_20261103_v1.json'
out.write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in result.items() if k!='barycenter_proof'}))
