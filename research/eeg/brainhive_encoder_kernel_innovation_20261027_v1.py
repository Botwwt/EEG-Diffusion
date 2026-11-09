"""Fixed learned-input-kernel versus matched-random conditional increment test."""
import numpy as np
from brainhive_conditional_innovation_diagnostic_20261027_v1 import bootstrap
ARMS=('c_only','shared_kernel','matched_random','sameconcept_wrong_kernel_train','sameconcept_wrong_kernel_inference')

def readout(scores,concept_ids,subjects):
    assert scores.shape==(10,820,4,5,3) and np.isfinite(scores).all()
    means={a:dict(zip(('NLL_per_dim','MSE_native_PC','standardized_quadratic_per_dim'),scores[:,:,:,i].mean((0,1,2)).tolist())) for i,a in enumerate(ARMS)}
    contrasts={}
    for i,arm in enumerate(ARMS):
        if i==1:continue
        contrasts[arm]={key:bootstrap((scores[:,:,:,i,k]-scores[:,:,:,1,k]).mean(2),concept_ids,subjects)
                        for k,key in enumerate(('NLL_improvement_per_dim','native_PC_MSE_reduction'))}
    base=contrasts['c_only'];relative=base['native_PC_MSE_reduction']['mean']/means['c_only']['MSE_native_PC']
    supported=(relative>=.005 and base['native_PC_MSE_reduction']['subjects_positive']>=7
       and all(contrasts[a]['NLL_improvement_per_dim']['mean']>=.001
               and contrasts[a]['NLL_improvement_per_dim']['descriptive_subject_concept_bootstrap95'][0]>0
               and contrasts[a]['native_PC_MSE_reduction']['mean']>0
               and contrasts[a]['native_PC_MSE_reduction']['descriptive_subject_concept_bootstrap95'][0]>0
               for a in ('c_only','matched_random','sameconcept_wrong_kernel_train'))
       and .5<=means['shared_kernel']['standardized_quadratic_per_dim']<=2)
    return dict(status='ALL10_T30_KERNEL_RANDOM_WRONG_FITS_AND_DFIT8200_COMPLETE',means=means,
        shared_kernel_improvements=contrasts,relative_MSE_improvement_vs_c=relative,
        kernel_specific_conditional_increment_supported=bool(supported),
        action='SUPPORTS_SEPARATELY_FROZEN_DCHECK_DIAGNOSTIC_ONLY' if supported else 'STOP_KERNEL_GAUSSIAN_INSTANCE_NO_IMAGES_OR_DCHECK',
        full16384_image_increment_established=False,actual_HIVE_trajectory_applicability_established=False,
        new_image_method_admitted=False,actual_image_gain_established=False,novelty_established=False,SOTA_established=False,
        exposure_note='Dfit is previously exposed development; kernel basis uses full T trained weights and is not fully OOF')
