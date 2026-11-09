"""D17 original80-to400 image decisions for three fixed final native encoders."""
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_crossbrain_supervision_20261027_v1 import ARMS,SCHEMA

def score(generated,targets,baseline,queries,gallery,stage):
    assert stage in (80,400) and generated.shape==(stage,3,768) and baseline.shape==(400,3,768)
    old={arm:U.S.metric_arrays(baseline[:stage,j],targets,queries,gallery) for j,arm in enumerate(U.ARMS)}
    new={arm:U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,arm in enumerate(ARMS)}
    references=old|{arm:value for arm,value in new.items() if arm!='crossbrain_soft'}
    contrasts={arm:U.S.compare(new['crossbrain_soft'],value,queries) for arm,value in references.items()}
    decision=U.S.decision(contrasts['risk'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        if not all(U.S.decision(value,'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for value in contrasts.values()):
            decision=decision|dict(action='STOP_INSTANCE',reason='NOT_PROMISING_VS_EVERY_STRONG_AND_MATCHED_ENCODER_REFERENCE')
    return dict(status='D17_COMPLETE_NATIVE_ENCODER_PAIRED_IMAGE_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={arm:{k:float(v.mean()) for k,v in value.items()} for arm,value in (old|new).items()},
        crossbrain_minus_every_reference=contrasts,decision=decision,risk_primary_fixed_before_D17_images=True,
        all3_final_encoder_arms_reported=True,no_prediction_proxy_image_gate=True,normal_inference_target_access=False,
        other_brain_EEG_used_at_normal_inference=False,independent_confirmation=False,SOTA_established=False)
