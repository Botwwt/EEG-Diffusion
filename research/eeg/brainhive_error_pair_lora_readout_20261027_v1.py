"""D20 same400gallery; prospective ordinary reference and matched controls."""
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_error_pair_lora_20261027_v2 import ARMS
SCHEMA='D20_NATIVE_IP_LORA_ERROR_PAIR_IMAGES_20261027_v1'
def score(generated,target,baseline,queries,gallery,stage):
    assert generated.shape==(stage,3,768) and baseline.shape==(400,3,768) and stage in (80,400)
    old={a:U.S.metric_arrays(baseline[:stage,j],target,queries,gallery) for j,a in enumerate(U.ARMS)}
    new={a:U.S.metric_arrays(generated[:,j],target,queries,gallery) for j,a in enumerate(ARMS)}
    contrasts={a:U.S.compare(new['paired_error'],v,queries) for a,v in (old|new).items() if a!='paired_error'}
    decision=U.S.decision(contrasts['ordinary'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        strong=all(U.S.decision(contrasts[a],'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for a in U.ARMS)
        mechanism=all(contrasts[a]['DINO2AFC']['difference']>0 and contrasts[a]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and contrasts[a]['within_concept10_top1']['difference']>=0 for a in ARMS[1:])
        if not(strong and mechanism):decision|={'action':'STOP_INSTANCE','reason':'STRONG_REFERENCE_OR_ERROR_PAIR_CONTROL_NOT_PASSED'}
    return dict(status='D20_COMPLETE_ALL_REGISTERED_ARMS_ACTUAL_IMAGE_READOUT',queries=stage,concepts=stage//10,gallery_targets=400,
        means={a:{k:float(x.mean()) for k,x in v.items()} for a,v in (old|new).items()},paired_minus_each_reference=contrasts,decision=decision,
        primary_reference='old ordinary directly reused',ordinary_reference_fixed_before_any_D20_fit=True,all3_new_arms_reported=True,normal_inference_UNet_NFE=4,
        same_NFE_not_identical_compute=True,exact_Turbo_likelihood_or_posterior=False,independent_confirmation=False,SOTA_established=False)
