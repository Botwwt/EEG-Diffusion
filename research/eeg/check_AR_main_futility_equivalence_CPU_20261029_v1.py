import hashlib,json
from pathlib import Path
import numpy as np
from AR_main_futility_80_CPU_20261029_v1 import decision
from AR_anchored_image_screen_math_20261029_v1 import stage_decision
rng=np.random.default_rng(1729);cases=[(-.02,-.03,-.01),(-.02,-.03,0.),(-.019999,-.03,-.01),(0.,-.01,.01),(.009875,.001,.02)]
cases += [(float(mean),float(mean-width),float(mean+width)) for mean,width in zip(rng.uniform(-.2,.2,1000),rng.uniform(0,.1,1000))]
for mean,low,high in cases:
    primary={'DINO2AFC':dict(difference=mean,descriptive_concept_bootstrap95=[low,high])}
    for arbitrary_donor in (-100.,0.,100.):
        contrasts=dict(paired_minus_native=primary,paired_minus_donor={'unused':arbitrary_donor},donor_minus_native={'unused':-arbitrary_donor})
        original=stage_decision(contrasts,80);early=decision(primary)
        assert (original['action']=='STOP_INSTANCE')==(early['action']=='STOP_MAIN_FOR_CLEAR_HARM')
        assert early['positive_method_gain_established'] is False
L=Path(__file__).resolve().parent;out=L/'audit/AR_main_futility_20261029_v1';out.mkdir(exist_ok=True);p=out/'CPU_equivalence_actual_v1.json';assert not p.exists()
v=dict(status='PASS_DONOR_INDEPENDENT_EARLY_HARM_EXACT_ORIGINAL_RULE_EQUIVALENCE',synthetic_contrast_cases=len(cases)*3,thresholds_and_boundary_cases_unchanged=True,
    method_or_mechanism_success_never_admitted_early=True,real_images_or_scores_read=False,new_GPU_jobs=0,
    original_math_source_sha256=hashlib.sha256((L/'AR_anchored_image_screen_math_20261029_v1.py').read_bytes()).hexdigest(),
    early_reader_source_sha256=hashlib.sha256((L/'AR_main_futility_80_CPU_20261029_v1.py').read_bytes()).hexdigest())
p.write_bytes((json.dumps(v,indent=2)+'\n').encode());print(json.dumps(v))
