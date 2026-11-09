"""Prospective donor-independent harm look; never admits a positive effect."""
import argparse,datetime,hashlib,json,math,sys,time
from pathlib import Path

def decision(contrast):
    afc=contrast['DINO2AFC'];mean=afc['difference'];low,high=afc['descriptive_concept_bootstrap95']
    assert all(math.isfinite(x) for x in (mean,low,high)) and low<=high
    return dict(action='STOP_MAIN_FOR_CLEAR_HARM' if mean<=-.02 and high<0 else 'CONTINUE_FIXED_CONTROL_AND_FULL_COMPARISON',
        harm_rule='Unchanged original stage80 mean<=-0.02 AND upper<0',positive_method_gain_established=False,mechanism_established=False,SOTA_established=False)

def run(addendum_path):
    addendum_path=Path(addendum_path);addendum=json.loads(addendum_path.read_bytes())
    assert addendum['status']=='PROSPECTIVELY_AUTHORIZED_DONOR_INDEPENDENT_MAIN80_HARM_EXCEPTION_v1'
    assert addendum['original_image_policy_sha256']=='732c5de27b84b11ff6f65884456c0124abe55c26efae90c0cd4d5195fa19fce1'
    assert addendum['registered_before_native_fixed_final_and_any_research_images'] is True
    assert addendum['positive_and400_gates_keep_all_three_original_fixed_fits_and_features'] is True
    sys.path.insert(0,addendum['image_code'])
    import AR_anchored_images_contract_CPU_20261029_v1 as C
    import AR_anchored_images_score_CPU_20261029_v1 as Score
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
    import numpy as np
    a,_=C.load_admission(addendum['image_admission']);C.validate_sources(a);C.validate_shared_recipe(a);_,panel=C.validate_panel(a)
    assert C.sha(Path(locked.__file__))==C.METRIC_SHA
    for arm in ('native_global','paired_anchored'):C.validate_fit(a,arm)
    features={};sources={};all_inodes=[]
    for arm in ('native_global','paired_anchored'):
        features[arm],sources[arm],inodes=Score.load_arm_features(a,arm,80,panel);all_inodes+=inodes
    assert len(set(all_inodes))==160
    cache=json.loads(C.checked(a['target_cache']).read_bytes());assert cache['panel']['sha256']==C.PANEL_SHA and cache['features']['sha256']==a['target_features']['sha256']
    targets=np.load(C.checked(a['target_features']),allow_pickle=False);assert targets.shape==(400,768) and np.isfinite(targets).all()
    arrays={arm:locked.metric_arrays(features[arm],targets,panel['queries'][:80],panel['target_gallery']) for arm in features}
    contrast=locked.compare(arrays['paired_anchored'],arrays['native_global'],panel['queries'][:80])
    independent={arm:Score.independent_arrays(features[arm],targets,panel['queries'][:80],panel['target_gallery']) for arm in features}
    check=Score.independent_contrast(independent['paired_anchored'],independent['native_global'],panel['queries'][:80])
    for metric in contrast:
        assert abs(contrast[metric]['difference']-check[metric]['difference'])<=2e-6
        assert np.allclose(contrast[metric]['descriptive_concept_bootstrap95'],check[metric]['descriptive_concept_bootstrap95'],atol=2e-6,rtol=0)
    assert decision(contrast)==decision(check)
    result=dict(status='COMPLETE_ONCE_ONLY_PROSPECTIVE_MAIN80_FUTILITY_EXCEPTION',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        addendum=C.ref(addendum_path),panel=a['panel'],original_image_policy=a['policy'],required_fixed_fits={arm:C.validate_fit(a,arm)[1] for arm in features},
        input_artifacts=sources,contrast=contrast,decision=decision(contrast),independent_CPU_rescore_passed=True,query_denominator_per_arm=80,
        same_fixed_queries_seeds_gallery_and_metrics=True,positive_result_not_admitted_without_original_all3_gates=True,
        donor_fit_or_scores_not_read=True,ordinary_unchanged=True,training_not_changed_by_this_reader=True,
        future_donor_withdrawal_requires_preserved_model_Adam_RNG_and_actual_process_closure=True)
    output=Path(addendum['output']);assert not output.exists();output.mkdir()
    C.write_once(output/'decision.json',result)
    np.savez_compressed(output/'main80_arrays.npz',**{arm+'__'+name:x for arm,values in arrays.items() for name,x in values.items()})
    print(json.dumps(dict(decision=C.ref(output/'decision.json'),action=result['decision']['action'])))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--addendum',type=Path,required=True);q=p.parse_args();run(q.addendum)
