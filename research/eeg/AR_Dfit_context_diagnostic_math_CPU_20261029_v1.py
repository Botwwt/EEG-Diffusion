"""Fixed signs/units for the five registered teacher-prefix conditions."""
import numpy as np
CONDITIONS=('own_EEG_own_GT_prefix','same_concept_donor_EEG_own_GT_prefix','zero_EEG_own_GT_prefix','own_EEG_same_concept_donor_prefix','same_concept_donor_EEG_same_concept_donor_prefix')
def summarize(nll,concept_ids):
    a=np.asarray(nll,np.float64);assert a.shape==(len(concept_ids),5,10) and np.isfinite(a).all()
    concepts=sorted(set(concept_ids));assert len(concepts)==20
    groups=[[i for i,c in enumerate(concept_ids) if c==concept] for concept in concepts];assert all(len(x)==10 for x in groups)
    # Positive means own EEG improves own-target log likelihood.
    differences=dict(own_minus_donor_EEG_at_GT_prefix=a[:,1]-a[:,0],own_minus_zero_EEG_at_GT_prefix=a[:,2]-a[:,0],
        own_minus_donor_EEG_at_donor_prefix=a[:,4]-a[:,3],GT_minus_donor_prefix_at_own_EEG=a[:,3]-a[:,0])
    draws=np.random.default_rng(1729).integers(0,20,(1000,20));result={}
    for name,x in differences.items():
        by_concept=np.array([x[g].mean(0) for g in groups]);boot=by_concept[draws].mean(1);interval=np.percentile(boot,[2.5,97.5],axis=0)
        result[name]=dict(mean_log_likelihood_difference_nats_per_token=x.mean(0).tolist(),descriptive_concept_bootstrap95_per_scale=interval.T.tolist())
    return dict(conditions=list(CONDITIONS),mean_NLL_nats_per_token_by_condition_and_scale=a.mean(0).tolist(),contrasts=result,
        inference='Teacher-forced heldout Dfit only; original target images are diagnostic targets/prefixes. Not normal inference or free-generation benefit, conditional mutual information, or causal mechanism proof.',
        multiple_contrasts_or_optional_looks_have_no_nominal_error_control=True,current_training_or_image_policy_changed=False)
