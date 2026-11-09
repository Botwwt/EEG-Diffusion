"""D18 original80-to400 image policy, frozen strong references and endpoint control."""
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_state_transport_20261027_v1 import ARMS,SCHEMA

def score(generated,targets,baseline,queries,gallery,stage):
    assert stage in (80,400) and generated.shape==(stage,2,768) and baseline.shape==(400,3,768)
    old={arm:U.S.metric_arrays(baseline[:stage,j],targets,queries,gallery) for j,arm in enumerate(U.ARMS)}
    new={arm:U.S.metric_arrays(generated[:,j],targets,queries,gallery) for j,arm in enumerate(ARMS)}
    contrasts={arm:U.S.compare(new[ARMS[0]],value,queries) for arm,value in old.items()}
    matched=U.S.compare(new[ARMS[0]],new[ARMS[1]],queries)
    decision=U.S.decision(contrasts['risk'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        strong=all(U.S.decision(value,'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for value in contrasts.values())
        mechanism=matched['DINO2AFC']['difference']>0 and matched['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and matched['within_concept10_top1']['difference']>=0
        if not(strong and mechanism):decision=decision|dict(action='STOP_INSTANCE',reason='STRONG_REFERENCE_OR_MATCHED_ENDPOINT_CONTROL_NOT_PASSED')
    return dict(status='D18_COMPLETE_PAIRED_SOURCE_TRANSPORT_IMAGE_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={arm:{key:float(value.mean()) for key,value in values.items()} for arm,values in (old|new).items()},
        paired_minus_each_strong_reference=contrasts,paired_minus_sameconcept_wrong_transport=matched,decision=decision,
        risk_primary_fixed_before_D18_images=True,all2_fields_reported=True,no_prediction_proxy_image_gate=True,
        normal_inference_target_access=False,independent_confirmation=False,SOTA_established=False)
