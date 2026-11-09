"""D19 same immutable400gallery, cached strong references and actual edit controls."""
import brainhive_gain_calibration_20261027_v1 as U
from brainhive_image_domain_20261027_v1 import SCHEMA,ARMS

def score(generated,target,baseline,queries,gallery,stage):
    assert stage in (80,400) and generated.shape==(stage,4,768) and baseline.shape==(400,3,768)
    old={arm:U.S.metric_arrays(baseline[:stage,j],target,queries,gallery) for j,arm in enumerate(U.ARMS)}
    new={arm:U.S.metric_arrays(generated[:,j],target,queries,gallery) for j,arm in enumerate(ARMS)}
    contrasts={arm:U.S.compare(new['conditional'],value,queries) for arm,value in (old|new).items() if arm!='conditional'}
    decision=U.S.decision(contrasts['risk'],str(stage))
    if stage==400 and decision['action']=='PROMISING_FOR_FULL_EVALUATION':
        strong=all(U.S.decision(contrasts[arm],'400')['action']=='PROMISING_FOR_FULL_EVALUATION' for arm in ('parent','ordinary','risk','vae_roundtrip'))
        mechanism=all(contrasts[arm]['DINO2AFC']['difference']>0 and contrasts[arm]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and contrasts[arm]['within_concept10_top1']['difference']>=0 for arm in ('unconditional','wrong_condition'))
        if not(strong and mechanism):decision=decision|dict(action='STOP_INSTANCE',reason='STRONG_REFERENCE_ZERO_EDIT_OR_CONDITIONING_CONTROL_NOT_PASSED')
    return dict(status='D19_COMPLETE_ACTUAL_IMAGE_DOMAIN_CORRECTION_READOUT',stage=stage,queries=stage,concepts=stage//10,gallery_targets=400,
        means={arm:{key:float(v.mean()) for key,v in value.items()} for arm,value in (old|new).items()},conditional_minus_each_reference=contrasts,decision=decision,
        risk_primary_fixed_before_D19_images=True,all4_new_arms_reported=True,proper_classifier_loss_not_image_gain_gate=True,
        extra_VAE_and_CNN_gradient_costs_reported_separately=True,exact_diffusion_score_or_posterior=False,independent_confirmation=False,SOTA_established=False)
