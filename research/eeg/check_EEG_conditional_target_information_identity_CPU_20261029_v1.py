"""Exact finite-probability interpretation, not empirical EEG information."""
import datetime,json,time
from pathlib import Path
import numpy as np
from score_EEG_native_marginal_images_CPU_20261029_v2 import decision
started=time.monotonic();checks={}
pxe=np.array([[.22,.03],[.15,.10],[.05,.20],[.08,.17]],dtype=np.float64)
qxz=np.array([[.70,.20,.10],[.40,.30,.30],[.10,.60,.30],[.15,.15,.70]])
pez=np.array([[.60,.25,.15],[.20,.30,.50]])
assert np.allclose(pxe.sum(),1) and np.allclose(qxz.sum(1),1) and np.allclose(pez.sum(1),1)
joint=pxe[:,:,None]*qxz[:,None,:];pe=pxe.sum(0);qe_z=joint.sum(0)/pe[:,None]
R=sum(pxe[x,e]*sum(qxz[x,z]*np.log(qxz[x,z]/pez[e,z]) for z in range(3)) for x in range(4) for e in range(2))
CMI=sum(joint[x,e,z]*np.log(qxz[x,z]/qe_z[e,z]) for x in range(4) for e in range(2) for z in range(3))
mismatch=(pe[:,None]*qe_z*np.log(qe_z/pez)).sum()
checks['crossmodal_KL_equals_conditional_MI_plus_conditional_prior_error']=abs(R-CMI-mismatch)<1e-14
qz=joint.sum((0,1));Ize=(pe[:,None]*qe_z*np.log(qe_z/qz[None,:])).sum()
R0=(joint*np.log(qxz[:,None,:]/qz[None,None,:])).sum()
checks['optimal_EEG_prior_reduces_rate_by_EEG_shared_information']=abs(R0-CMI-Ize)<1e-14 and Ize>0
# Exact categorical alternative on native pretrained condition support. This
# verifies only its variational identity; no new generator or EEG was fitted.
likelihood=np.array([[.18,.04,.09],[.01,.20,.03]],dtype=np.float64)
prior=np.array([[.6,.3,.1],[.2,.3,.5]])
evidence=(likelihood*prior).sum(1);posterior=likelihood*prior/evidence[:,None]
trial=np.array([[.4,.3,.3],[.3,.4,.3]])
objective=(trial*(-np.log(likelihood)+np.log(trial/prior))).sum(1)
extra=(trial*np.log(trial/posterior)).sum(1)
checks['categorical_variational_objective_equals_exact_NLL_plus_posterior_KL']=np.allclose(objective,-np.log(evidence)+extra,atol=1e-14)
minimum=(posterior*(-np.log(likelihood)+np.log(posterior/prior))).sum(1)
checks['closed_posterior_minimizes_E_step_exactly']=np.allclose(minimum,-np.log(evidence),atol=1e-14) and np.all(objective>=minimum)
def contrast(delta,low,high,top=0):return dict(DINO2AFC=dict(difference=delta,descriptive_concept_bootstrap95=[low,high]),within_concept10_top1=dict(difference=top))
cases=[(80,contrast(-.02,-.1,-.001),'STOP_INSTANCE'),(80,contrast(-.02,-.1,0),'EXPAND400'),(80,contrast(.2,.1,.3),'EXPAND400')]
for i,(stage,row,wanted) in enumerate(cases):checks[f'prospective80_boundary_case_{i+1}']=decision({'coupled_minus_visual':row},stage)==wanted
a=contrast(.01,.001,.03);b=contrast(.009999,.001,.03)
checks['full400_no_rounding_practical_threshold']=decision(dict(coupled_minus_visual=a,coupled_minus_marginal=b),400)=='STOP_INSTANCE'
checks={key:bool(value) for key,value in checks.items()}
assert all(checks.values()),checks
result=dict(status='PASS_EXACT_PROBABILITY_INTERPRETATION_AND_PROSPECTIVE_IMAGE_RULE_BOUNDARIES',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),checks=checks,
    exact_example=dict(KL_rate=R,conditional_MI=CMI,conditional_prior_KL_error=mismatch,EEG_shared_MI=Ize,categorical_exact_NLL=(-np.log(evidence)).tolist()),seconds=time.monotonic()-started,
    empirical_EEG_MI_estimated=False,new_GPU_jobs=0,new_models_or_images=0,claimed_mathematical_originality=False,
    applicability='Population identity assumes the visual posterior is a function of image alone, not an extra EEG side channel. A per-image T bank and unique noisy EEG do not establish population MI or its estimate. This is interpretation, not a result proving recovery of EEG-identifiable image information.')
out=Path(__file__).resolve().parent/'audit/EEG_native_conditional_variational_20261029_v1/exact_information_identity_CPU_actual_v1.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
