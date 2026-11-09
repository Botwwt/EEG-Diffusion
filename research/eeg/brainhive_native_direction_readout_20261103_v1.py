"""Fixed exposed-development comparisons; every metric is retained when directions differ."""
import numpy as np
from brainhive_native_direction_image_contract_20261103_v1 import CONTRASTS
DINO_METRICS=('target_cosine','DINO2AFC','gallery_top1','within_concept10_top1','within_concept10_two_afc')
LOW_METRICS=('PixCorr_RGB','SSIM_Y','mean_RGB_absolute_error')
METRICS=DINO_METRICS+LOW_METRICS
def paired(array,queries):
 assert array.shape==(len(queries),len(METRICS)) and np.isfinite(array).all();concepts=sorted({q['concept_id'] for q in queries});groups=[[i for i,q in enumerate(queries) if q['concept_id']==c] for c in concepts]
 pc=np.stack([array[g].mean(0) for g in groups]);rng=np.random.default_rng(1729);draws=rng.integers(0,len(groups),(1000,len(groups)));samples=pc[draws].mean(1);ci=np.quantile(samples,[.025,.975],axis=0)
 return {key:dict(difference=float(array[:,j].mean()),descriptive_concept_bootstrap95=ci[:,j].tolist(),positive_concepts=int((pc[:,j]>0).sum()),concepts=len(concepts)) for j,key in enumerate(METRICS)}
def decision(contrasts,n):
 assert n in (80,400);primary=contrasts['directional_minus_old_ordinary'];afc=primary['DINO2AFC'];within=primary['within_concept10_two_afc']
 # The additional within-concept requirement prevents a single adverse metric
 # from terminating expansion; -.02 retains the old v2 harm magnitude.
 joint=(afc['difference']<=-.02 and afc['descriptive_concept_bootstrap95'][1]<0 and within['descriptive_concept_bootstrap95'][1]<0)
 practical=(afc['difference']>=.01 and afc['descriptive_concept_bootstrap95'][0]>0 and within['difference']>=0)
 matched=all(contrasts[name]['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 for name in ('directional_minus_positive_only','directional_minus_wrong_EEG','directional_minus_reencoded_ordinary'))
 action=('STOP_EXPANSION_JOINT_HARM' if joint else 'EXPAND400') if n==80 else ('PROMISING_MATCHED_INCREMENT_REQUIRES_MORE_SUBJECTS_AND_FULL_METRICS' if practical and matched else 'NO_CONFIRMED_MATCHED_INCREMENT_ON_THIS_EXPOSED_PANEL')
 disagreements={name:[key for key in METRICS if (row[key]['difference']>0 if key=='mean_RGB_absolute_error' else row[key]['difference']<0)] for name,row in contrasts.items()}
 return dict(action=action,joint_harm=joint,primary_practical_signal=practical,matched_directional_increment=matched,adverse_direction_metrics=disagreements,
  all_metrics_and_contrasts_reported=True,one_failed_metric_does_not_reject_family=True,thresholds_registered_before_new_images=True,descriptive_only=True,optional_stopping_has_no_nominal_error_control=True,SOTA_established=False)
def summarize(values,queries):
 assert all(v.shape==(len(queries),8) and np.isfinite(v).all() for v in values.values());diff={k:values[a]-values[b] for k,(a,b) in CONTRASTS.items()};contrasts={k:paired(v,queries) for k,v in diff.items()}
 return dict(means={a:{key:float(v[:,j].mean()) for j,key in enumerate(METRICS)} for a,v in values.items()},paired=contrasts,decision=decision(contrasts,len(queries)),comparison_definitions={k:list(v) for k,v in CONTRASTS.items()}),diff
