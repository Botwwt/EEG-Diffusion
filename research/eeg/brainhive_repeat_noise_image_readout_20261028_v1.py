"""D23 fixed three-arm native image comparison, ordinary primary before images."""
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_repeat_noise_subspace_20261027_v1 import ARMS
SCHEMA='BrainHIVE_repeat_noise_native_subspace_20261028_v1'

def score(generated,targets,baseline,queries,gallery,stage):
    assert stage in (80,400) and generated.shape==(stage,3,768) and baseline.shape==(400,3,768)
    old={arm:U.S.metric_arrays(baseline[:stage,j],targets,queries,gallery) for j,arm in enumerate(U.ARMS)}
    new={arm:U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,arm in enumerate(ARMS)}
    refs=old|{arm:value for arm,value in new.items() if arm!=ARMS[0]};contrasts={arm:U.S.compare(new[ARMS[0]],value,queries) for arm,value in refs.items()}
    decision=U.S.decision(contrasts['ordinary'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(v,'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for v in contrasts.values()):decision=decision|dict(action='STOP_INSTANCE',reason='NOT_PROMISING_VS_EVERY_STRONG_AND_MATCHED_ENCODER_REFERENCE')
    return dict(status='D23_COMPLETE_NATIVE_FIRST_WEIGHT_PAIRED_IMAGE_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={arm:{k:float(v.mean()) for k,v in value.items()} for arm,value in (old|new).items()},repeat_noise_minus_every_reference=contrasts,decision=decision,
        ordinary_primary_fixed_before_D23_images=True,all3_final_first_weight_arms_reported=True,no_prediction_proxy_image_gate=True,normal_inference_target_access=False,
        additional_inference_parameters=0,normal_inference_individual_trials=False,independent_confirmation=False,SOTA_established=False)
