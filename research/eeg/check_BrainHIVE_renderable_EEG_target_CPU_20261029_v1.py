"""Independent analytic invariants of the candidate's classical solver."""
import json
import numpy as np
from brainhive_renderable_EEG_target_20261029_v1 import centered_basis,spherical_quadratic_target,empirical_repeat_precision
rng=np.random.default_rng(1729);d=17;Q=centered_basis(d);checks=[]
def sphere(x):return x/np.linalg.norm(x)*np.sqrt(d)
m=sphere(Q@rng.normal(size=d-1));f=sphere(Q@rng.normal(size=d-1))
H=rng.normal(size=(d,d));H=H.T@H;B=rng.normal(size=(d,d));P=B.T@B+np.eye(d)
u,k=spherical_quadratic_target(m,f,H,P)
assert k['objective']<=min(k['objective_at_EEG'],k['objective_at_visual'])+1e-8;checks.append('KKT_global_certificate_and_cost_no_worse_than_both_feasible_anchors')
z,meta=spherical_quadratic_target(m,f,np.zeros((d,d)),P);assert np.max(abs(z-m))<1e-8;checks.append('no_semantic_constraint_returns_EEG')
z,meta=spherical_quadratic_target(m,m,H,P);assert np.max(abs(z-m))<1e-8;checks.append('matched_EEG_visual_returns_identity')
z,meta=spherical_quadratic_target(m,f,2*np.eye(d),3*np.eye(d));exact=sphere(2*f+3*m);assert np.max(abs(z-exact))<1e-8;checks.append('isotropic_case_equals_normalized_weighted_anchors')
T=Q@np.linalg.qr(rng.normal(size=(d-1,d-1)))[0]@Q.T+np.ones((d,d))/d
z,meta=spherical_quadratic_target(T@m,T@f,T@H@T.T,T@P@T.T);assert np.max(abs(z-T@u))<1e-8;checks.append('joint_centered_orthogonal_coordinate_equivariance')
z,meta=spherical_quadratic_target(m,-m,np.eye(d),np.eye(d));assert meta['hard_case'] and abs(np.linalg.norm(z)-np.sqrt(d))<1e-8;checks.append('antipodal_zero_linear_term_hard_case_preserves_sphere')
P0,meta=empirical_repeat_precision(np.tile(m,(4,1)));assert np.array_equal(P0,np.eye(d)) and meta['zero_dispersion'];checks.append('identical_repeats_have_explicit_isotropic_fallback')
views=rng.normal(size=(4,d));P0,meta=empirical_repeat_precision(views);assert np.linalg.eigvalsh(P0)[0]>0 and abs(np.trace(Q.T@P0@Q)-(d-1))<1e-8;checks.append('shrinkage_positive_definite_and_centered_trace_normalized')
print(json.dumps(dict(status='PASS_CLASSICAL_SPHERICAL_TARGET_CPU_INVARIANTS',checks=checks,new_EEG_or_image_evidence=False,novelty_established=False)))
