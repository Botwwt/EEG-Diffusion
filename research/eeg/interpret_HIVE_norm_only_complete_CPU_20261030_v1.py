"""Apply the already registered mechanism and efficacy gates after full audit.

No training, queue actions, new thresholds, source edits, or SOTA declaration.
"""
import argparse,hashlib,json,math
from pathlib import Path

PAIRS=('Norm_only_minus_Euclidean_F','Norm_only_minus_bounded_F','Norm_only_minus_ordinary','Norm_only_minus_parent')
MECHANISM='Norm_only minus Euclidean_F DINO and SEED positive descriptive95CI; ObjectF1 mean nonnegative'
PERFORMANCE='Norm_only vs BOTH originalordinary andboundedF DINO>=+.01 with positive95CI,SEED positive95CI,ObjectF1 means>=0; otherwise no efficacy promotion'

def conjunction(items):
    return False if False in items else None if None in items else True

def primary_gate(results,pair,minimum_DINO=None):
    metrics=results.get(pair,{})
    values={}
    for name in ('DINO2AFC','SEED','ObjectF1'):
        record=metrics.get(name,{})
        if record.get('status')!='COMPLETE':return None,dict(status='UNKNOWN_PRIMARY_RESULT_FULL_DENOMINATOR_RETAINED',metric=name)
        value=record['paired_left_minus_right'];ci=record['descriptive_subject_concept_bootstrap95']
        assert record['defined']==2000 and len(ci)==2 and ci[0]<=ci[1] and all(math.isfinite(x) for x in [value,*ci])
        values[name]=dict(difference=value,CI95=ci)
    checks=dict(DINO_CI_lower_strictly_positive=values['DINO2AFC']['CI95'][0]>0,SEED_CI_lower_strictly_positive=values['SEED']['CI95'][0]>0,ObjectF1_mean_nonnegative=values['ObjectF1']['difference']>=0)
    if minimum_DINO is not None:checks['DINO_mean_at_least_registered_point01']=values['DINO2AFC']['difference']>=minimum_DINO
    return all(checks.values()),dict(status='COMPLETE',values=values,checks=checks)

def interpret(results):
    assert set(results)==set(PAIRS)
    mechanism,mechanism_detail=primary_gate(results,PAIRS[0])
    efficacy={};details={}
    for pair in PAIRS[1:3]:efficacy[pair],details[pair]=primary_gate(results,pair,.01)
    performance=conjunction(list(efficacy.values()))
    ordinary=results[PAIRS[2]].get('DINO2AFC',{})
    harm=None if ordinary.get('status')!='COMPLETE' else ordinary['descriptive_subject_concept_bootstrap95'][1]<0
    status=('CURRENT_EXPOSED_SINGLE_SEED_BENCHMARK_PROGRESS_GATE_PASS' if performance is True else
            'UNKNOWN_PRIMARY_RESULT_NO_EFFICACY_PROMOTION' if performance is None else
            'MECHANISM_SUPPORT_ONLY_NO_EFFICACY_PROMOTION' if mechanism is True else
            'NO_MECHANISM_OR_EFFICACY_PROMOTION')
    return dict(status=status,mechanism_support=mechanism,mechanism_detail=mechanism_detail,performance_progress_gate_passed=performance,performance_details=details,clear_DINO_harm_versus_ordinary=harm,registered_DINO_threshold=.01,no_rounding_to_pass=True,undefined_primary_keeps_UNKNOWN=True,mechanism_support_does_not_substitute_for_best_baseline_gain=True,original_fullT_FP64_guard_passed=False,single_training_seed=1729,exposed_R80=True,independent_H=False,SOTA_established=False,CCF_A_established=False,automatic_new_training_or_parameter_search=False)

def run(root):
    root=Path(root);receipt=json.loads((root/'receiving_completion.json').read_bytes())
    assert receipt['completion']['status']=='COMPLETE_SINGLE1729_FROZEN_NORM_ONLY_R80_ALL16_FOUR_COMPARISONS_v1' and not receipt['exact_controller_live'] and receipt['failure'] is None
    study=json.loads((root/'protocol/study.json').read_bytes());assert study['mechanism_support']==MECHANISM and study['performance_progress_gate']==PERFORMANCE
    summary=json.loads((root/'actual/joint_comparison/summary.json').read_bytes());audit=json.loads((root/'independent_local_statistics.json').read_bytes())
    assert summary['queries_per_method']==2000 and summary['training_seed']==1729 and summary['all10000_image_file_hashes_and_inodes_verified']
    assert audit['status']=='PASS_FULL_FOUR_CONTRAST_ALL16_MEANS_BOOTSTRAP_AND_EVERY_STORED_RAW_CELL' and audit['maximum_statistic_error']<1e-12 and audit['maximum_raw_cell_error']==0
    assert audit['independently_verified_pairs']==128000-2000*len(audit['undefined_groups_retained'])
    result=interpret(summary['results']);result.update(full128000_audit_complete=not audit['undefined_groups_retained'],actual_audited_pairs=audit['independently_verified_pairs'],source_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),study_SHA256=hashlib.sha256((root/'protocol/study.json').read_bytes()).hexdigest())
    out=root/'registered_gate_interpretation.json';assert not out.exists();out.write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
