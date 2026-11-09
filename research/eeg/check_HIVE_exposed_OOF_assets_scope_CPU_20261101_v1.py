"""Full-size metadata rejection checks; never imports a model library."""
import argparse,copy,csv,json
from pathlib import Path
import brainhive_exposed_OOF_assets_contract_CPU_20261101_v1 as G

def checks(s,tasks,images,ids):
    keyed,train=G.validate_scope(s,tasks,images,ids)
    assert len(keyed)==20 and len(train)==16540
    results=['actual_full20_task165400_scope_and_shared_concept_exclusion']
    def reject(name,change):
        a,b,c,d=copy.deepcopy(s),copy.deepcopy(tasks),images,ids.copy()
        change(a,b,c,d)
        try:G.validate_scope(a,b,c,d)
        except (AssertionError,KeyError):results.append(name);return
        raise AssertionError('Accepted invalid full scope: '+name)
    reject('fake_unexposed_benchmark_rejected',lambda a,b,c,d:a.update(official200_previously_exposed=False))
    reject('wrong_prior_direction_flag_rejected',lambda a,b,c,d:a.update(official_results_used_for_previous_direction=False))
    reject('benchmark_payload_permission_rejected',lambda a,b,c,d:a.update(official_test_payloads_allowed=True))
    reject('incomplete_subjects_rejected',lambda a,b,c,d:a['subjects'].pop('sub-10'))
    reject('overlapping_export_row_rejected',lambda a,b,c,d:b['tasks'][0]['exported_OOF_array_rows'].__setitem__(0,b['tasks'][0]['training_array_rows'][0]))
    reject('incomplete25_epoch_budget_rejected',lambda a,b,c,d:b['tasks'][0].update(epochs=24))
    reject('changed_cached_visual_ID_rejected',lambda a,b,c,d:d.__setitem__(0,'wrong_stimulus'))
    reject('transient_or_old_OOF_adoption_rejected',lambda a,b,c,d:b['tasks'][0].update(existing_protected_T_OOF_reused=True))
    reject('wrong_last_batch_rejected',lambda a,b,c,d:b['tasks'][0].update(final_batch_size=80))
    return dict(status='PASS_TRAIN_ONLY_EXPOSURE_AWARE_FULL_SCOPE_CPU_CHECKS',checks=results,count=len(results),new_models=0,new_GPU_jobs=0)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);a=p.parse_args();s=G.read(a.protocol)
    tasks=G.read(G.checked(s['tasks']))
    with G.checked(s['image_manifest']).open(encoding='utf8',newline='') as f:images=list(csv.DictReader(f))
    result=checks(s,tasks,images,G.read(G.checked(s['visual_IDs'])));result['protocol']=G.ref(a.protocol)
    G.atomic(Path(a.protocol).parent/'actual_scope_CPU_checks.json',result)
    print(json.dumps(result))
