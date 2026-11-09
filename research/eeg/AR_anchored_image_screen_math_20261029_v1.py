"""AR-only fixed-panel decisions; never imports HIVE generation contracts.

Metric arrays and whole-concept bootstrap reuse the frozen existing arithmetic.
This module contains no image/model loading or GPU dispatch. A producer must
separately verify actual AR artifacts, all three arms and the fixed400 gallery.
"""
import hashlib
from pathlib import Path
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked

ARMS=('native_global','paired_anchored','donor_anchored')
LOCKED_MATH_SHA='bc9610b97501cdf078789f913b0520b023c0079801e24c37093cbac406189009'

def validate_math_source():
    assert hashlib.sha256(Path(locked.__file__).read_bytes()).hexdigest()==LOCKED_MATH_SHA

def stage_decision(contrasts,stage):
    assert stage in (80,400)
    assert set(contrasts)=={'paired_minus_native','paired_minus_donor','donor_minus_native'}
    primary=contrasts['paired_minus_native'];afc=primary['DINO2AFC'];mean=afc['difference'];low,high=afc['descriptive_concept_bootstrap95']
    if stage==80:
        stopped=mean<=-.02 and high<0
        return dict(action='STOP_INSTANCE' if stopped else 'EXPAND400',reason='PREDECLARED_CLEAR_HARM_TO_AR_NATIVE' if stopped else 'EXPAND_UNLESS_CLEAR_HARM',method_gain_established=False,mechanism_established=False,SOTA_established=False)
    passed={}
    for name in ('paired_minus_native','paired_minus_donor'):
        item=contrasts[name]
        passed[name]=bool(item['DINO2AFC']['difference']>=.01 and item['DINO2AFC']['descriptive_concept_bootstrap95'][0]>0 and item['target_cosine']['difference']>0 and item['within_concept10_top1']['difference']>=0)
    promoted=all(passed.values())
    return dict(action='PROMISING_FOR_COMPLETE_MULTISUBJECT_EVALUATION' if promoted else 'STOP_INSTANCE',reason='BOTH_FIXED_AR_COMPARATORS_PASS' if promoted else 'EXPANDED_PRACTICAL_OR_MECHANISM_GATE_NOT_MET',comparator_gates=passed,descriptive_development_signal_only=True,conditional_information_or_causal_mechanism_proved=False,independent_confirmation=False,SOTA_established=False)

def score_features(features,gallery_features,queries,gallery,stage):
    validate_math_source();assert set(features)==set(ARMS) and len(queries)==stage and len(gallery)==len(gallery_features)==400
    groups={}
    for q in queries:groups[q['concept_id']]=groups.get(q['concept_id'],0)+1
    assert len(groups)*10==stage and all(count==10 for count in groups.values())
    arrays={arm:locked.metric_arrays(features[arm],gallery_features,queries,gallery) for arm in ARMS}
    pairs={'paired_minus_native':('paired_anchored','native_global'),'paired_minus_donor':('paired_anchored','donor_anchored'),'donor_minus_native':('donor_anchored','native_global')}
    contrasts={name:locked.compare(arrays[a],arrays[b],queries) for name,(a,b) in pairs.items()}
    return dict(means={arm:{metric:float(values.mean()) for metric,values in row.items()} for arm,row in arrays.items()},contrasts=contrasts,decision=stage_decision(contrasts,stage)),arrays
