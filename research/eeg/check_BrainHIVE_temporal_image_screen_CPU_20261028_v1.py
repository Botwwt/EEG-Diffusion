"""Decision-boundary, tie and matched-control checks before D37 images."""
import argparse
import datetime
import json
from pathlib import Path
import numpy as np
import brainhive_temporal_image_screen_20261028_v1 as U
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
import audit_BrainHIVE_temporal_image_score_CPU_20261028_v1 as A


def run(output):
    queries=[dict(query_id=str(i),subject_id='sub-01',partition='Dcheck',concept_id=f'c{i//10:03d}',stimulus_id=f's{i:03d}',array_row=i,image_seed=i+1) for i in range(400)]
    rows,groups=U.queries(dict(Dcheck=queries));donor=U.control_map(groups)
    assert all(donor[i]!=i and rows[donor[i]]['concept_id']==rows[i]['concept_id'] for i in range(400))
    generator=np.random.default_rng(1729);details=generator.normal(size=(400,4,5,16)).astype(np.float32);details-=details.mean(2,keepdims=True)
    rms=np.sqrt(np.square(details,dtype=np.float64).sum(-1).mean(-1));scale=(rms/rms[donor]).astype(np.float32)
    wrong=details[donor]*scale[:,:,None,None];assert np.allclose(np.square(wrong,dtype=np.float64).sum(-1).mean(-1),rms*rms,atol=1e-5,rtol=1e-6)
    gallery=np.eye(400,dtype=np.float32);cases=[gallery[:80],np.ones((80,400),np.float32),np.roll(gallery[:80],1,axis=1)]
    error=0.
    for case in cases:
        reference=S.metric_arrays(case,gallery,queries[:80],queries);independent=A.metrics(case,gallery,80,queries)
        for key in reference:error=max(error,float(np.abs(reference[key]-independent[key]).max()));assert np.array_equal(reference[key],independent[key])
    first=S.metric_arrays(cases[0],gallery,queries[:80],queries);second=S.metric_arrays(cases[2],gallery,queries[:80],queries)
    assert S.compare(first,second,queries[:80])==A.paired(first,second,queries[:80])
    def paired(difference,low,high,cos=.01,top1=0):
        return dict(DINO2AFC=dict(difference=difference,descriptive_concept_bootstrap95=[low,high]),target_cosine=dict(difference=cos),within_concept10_top1=dict(difference=top1))
    assert S.decision(paired(-.03,-.05,-.01),'80')['action']=='STOP_INSTANCE'
    assert S.decision(paired(0,-.01,.01),'80')['action']=='EXPAND400'
    assert S.decision(paired(.009875,.005,.015),'400')['action']=='STOP_INSTANCE'
    assert S.decision(paired(.02,.01,.03,cos=-.001),'400')['action']=='STOP_INSTANCE'
    assert S.decision(paired(.02,.01,.03,top1=-.001),'400')['action']=='STOP_INSTANCE'
    assert S.decision(paired(.02,.01,.03),'400')['action']=='PROMISING_FOR_FULL_EVALUATION'
    value=dict(status='D37_IMAGE_CONTROL_TIES_BOOTSTRAPS_AND_STRICT_DECISION_CHECKS_PASSED',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checks=6,maximum_rank_metric_error=error,all400_donors_deranged_within_concept=True,
        per_token_control_RMS_matched=True,all3_tie_cases_agree=True,all1000_bootstraps_agree=True,
        original_ordinary_point009875_not_rounded_to_admission=True,raw_EEG_or_real_image_scores_loaded=False,
        new_GPU_jobs=0,new_training_updates=0,source=U.ref(__file__))
    U.write(output,value);print(json.dumps(value))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.output)
