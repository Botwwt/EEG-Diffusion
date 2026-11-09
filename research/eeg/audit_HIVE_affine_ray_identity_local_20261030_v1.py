"""Independent algebra on real cached inputs; no generator, fit or method admission."""
import hashlib,json,os
from pathlib import Path
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
L=Path(__file__).resolve().parent;N=L/'audit/HIVE_FP32_trace_complete_20261030_v1';P=L/'audit/HIVE_Euclidean_F_postfit_registration_20261030_v2'
def read(p):return json.loads(Path(p).read_bytes())
def verify_file(p,digest):
    with Path(p).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==digest
num=read(N/'actual/completion.json');verify_file(N/'actual/T320_fixed_affine_trace.npz',num['T_probe']['sha256']);verify_file(N/'actual/R80_all2000_FP32_trace.npz',num['R80_trace']['sha256'])
manifest=read(P/'receiving_manifest.json');verify_file(P/'numeric/original_R80_input_bank.npz',manifest['numeric/original_R80_input_bank.npz']['sha256'])
probe=np.load(N/'actual/T320_fixed_affine_trace.npz',allow_pickle=False);bank=np.load(P/'numeric/original_R80_input_bank.npz',allow_pickle=False);trace=np.load(N/'actual/R80_all2000_FP32_trace.npz',allow_pickle=False)
def quantile(x):return np.quantile(x,[0,.5,1]).tolist()
def analyze(c,g,b,z):
    c,g,b,z=[x.astype(np.float64) for x in (c,g,b,z)];u=(c-b)/g;radius2=np.square(u).sum(1);origin_energy=np.square(b/g).sum(1)
    assert np.all(origin_energy<radius2)
    a=np.square(c/g).sum(1);other_root=(origin_energy-radius2)/a
    # General positive-ray intersection of a new nonzero z with the same norm
    # ellipsoid. This fixes the norm only: the LN zero-mean constraint is extra.
    az=np.square(z/g).sum(1);bz=((z/g)*(b/g)).sum(1);d=origin_energy-radius2
    factor=(bz+np.sqrt(bz*bz-az*d))/az;assert np.all(factor>0) and np.isfinite(factor).all()
    counterfactual=factor[:,None]*z;after=(counterfactual-b)/g
    norm_error=np.abs(np.square(after).sum(1)-radius2)/radius2
    cos=np.sum(z*counterfactual,axis=1)/(np.linalg.norm(z,axis=1)*np.linalg.norm(counterfactual,axis=1))
    native_a=np.square(c/g).sum(1);native_b=((c/g)*(b/g)).sum(1);native_factor=(native_b+np.sqrt(native_b*native_b-native_a*d))/native_a
    assert norm_error.max()<1e-12 and np.abs(1-cos).max()<1e-12 and np.abs(1-native_factor).max()<1e-12
    return dict(queries=len(c),origin_inside_all=True,beta_over_gamma_energy_to_radius2_quantiles=quantile(origin_energy/radius2),other_root_quantiles=quantile(other_root),positive_norm_only_factor_quantiles=quantile(factor),maximum_relative_squared_norm_error=float(norm_error.max()),maximum_direction_cosine_error=float(np.abs(1-cos).max()),maximum_native_identity_factor_error=float(np.abs(1-native_factor).max()),native_standardized_absmean_over_RMS_quantiles=quantile(np.abs(u.mean(1))/np.sqrt(np.mean(u*u,axis=1))),counterfactual_standardized_absmean_over_RMS_quantiles=quantile(np.abs(after.mean(1))/np.sqrt(np.mean(after*after,axis=1))))
reports=dict(T320=analyze(probe['c'],probe['g'],probe['b'],probe['output']),R80=analyze(bank['conditions'].reshape(2000,1024),np.repeat(bank['gammas'],200,axis=0),np.repeat(bank['betas'],200,axis=0),trace['output']))
out=L/'audit/HIVE_Fcos_scale_and_affine_sphere_CPU_20261030_v1';out.mkdir(exist_ok=True);p=out/'independent_local_ray_identity.json';assert not p.exists()
result=dict(status='PASS_INDEPENDENT_LOCAL_T320_R80_AFFINE_RAY_ALGEBRA',reports=reports,classical_quadratic_geometry_not_novel_theorem=True,norm_only_counterfactual_not_full_LayerNorm_support=True,centering_and_angle_constraints_not_established_by_this_identity=True,algebraic_counterfactual_not_implemented_in_generator=True,new_fits_or_GPU_calls_or_images=0,image_quality_or_causal_gain_not_tested=True)
p.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
