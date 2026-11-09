"""Complete source/schema and historical5x16 arithmetic fixtures; no new effects."""
import argparse
import ast
import copy
import csv
import json
from pathlib import Path
import numpy as np
import brainhive_full1654_ordinary_fit_contract_CPU_20261101_v1 as F
import compare_HIVE_full1654_all4_methods_CPU_20261101_v1 as C


def run():
    root=Path(__file__).resolve().parent;checks=[]
    for name in F.REQUIRED_ENTRIES:
        path=root/name;assert path.is_file();ast.parse(path.read_bytes())
    checks.append('all6_actual_lifecycle_sources_present_and_parse_not_admission')
    history=root/'audit/HIVE_Euclidean_F_postfit_complete_20261030_v3'
    loss=root/'audit/HIVE_loss_controls_R80_single_seed_complete_20261030_v1'
    def table(path):
        with path.open(newline='') as stream:rows=list(csv.DictReader(stream))
        assert len(rows)==2000 and len({r['query_id'] for r in rows})==2000
        return {r['query_id']:r for r in rows}
    # Explicit historical-role aliases are fixtures, never new full1654 outcomes.
    aliases=dict(ordinary_full1654='ordinary',parent='parent',bounded_F='bounded_F',Euclidean_F='Euclidean_F')
    data={role:{kind:table(history/'tables'/alias/(kind+'.csv')) for kind in ('common','paper')}
          for role,alias in aliases.items()}
    reference=data['parent']['common'];subjects=sorted({r['subject'] for r in reference.values()});concepts=sorted({r['concept_id'] for r in reference.values()})
    cells={(r['subject'],r['concept_id']):key for key,r in reference.items()};assert len(cells)==2000
    values,differences,undefined,support=C.comparison_values(data,subjects,concepts,cells)
    old=json.loads((history/'actual/joint_comparison/summary.json').read_bytes())['results']
    old_loss=json.loads((loss/'actual/joint_comparison/summary.json').read_bytes())['results']
    expected=dict(ordinary_full1654_minus_parent=old_loss['ordinary_minus_parent'],
        bounded_F_minus_parent=old_loss['F_only_minus_parent'],Euclidean_F_minus_parent=old['Euclidean_F_minus_parent'],
        bounded_F_minus_Euclidean_F=old['bounded_F_minus_Euclidean_F'],ordinary_full1654_minus_bounded_F=old_loss['ordinary_minus_F_only'])
    maximum=0.
    for name,metrics in values.items():
        assert len(metrics)==16
        for metric,result in metrics.items():
            assert result['status']=='COMPLETE'
            for field in ('left_mean','right_mean','paired_left_minus_right','descriptive_subject_concept_bootstrap95','per_subject_difference'):
                error=float(np.max(np.abs(np.asarray(result[field])-np.asarray(expected[name][metric][field]))))
                maximum=max(maximum,error);assert error<1e-12
    assert len(differences)==80 and sum(value.size for value in differences.values())==160000 and not undefined
    assert support['ordinary_full1654_minus_parent'] and not support['ordinary_full1654_minus_bounded_F']
    checks.append('actual_historical80_metric_crossed_bootstraps_all160000_delta_inputs_replayed')
    damaged=copy.deepcopy(data);key=next(iter(damaged['ordinary_full1654']['common']))
    damaged['ordinary_full1654']['common'][key]['DINO2AFC']='nan'
    bad,deltas,unknown,gate=C.comparison_values(damaged,subjects,concepts,cells)
    assert len(unknown)==2 and len(deltas)==78 and sum(value.size for value in deltas.values())==156000
    assert all(not value for value in gate.values())
    assert bad['ordinary_full1654_minus_parent']['DINO2AFC']['expected']==2000
    checks.append('one_nan_keeps_full2000_metric_UNKNOWN_not1999_query_success')
    identical=copy.deepcopy(data)
    for kind,metrics in [('common',('DINO2AFC','DINO_cosine','DINO_Top1','DINO_Top5','EffNet','CapSim','ObjectF1','SEED')),
                         ('paper',('eval_pixcorr','eval_ssim','eval_alex2','eval_alex5','eval_inception','eval_clip','eval_effnet','eval_swav'))]:
        for key in reference:
            for metric in metrics:identical['ordinary_full1654'][kind][key][metric]=identical['parent'][kind][key][metric]
    equal,unused,unused,gate=C.comparison_values(identical,subjects,concepts,cells)
    assert equal['ordinary_full1654_minus_parent']['DINO2AFC']['paired_left_minus_right']==0 and not gate['ordinary_full1654_minus_parent']
    checks.append('zero_primary_difference_never_promoted_by_control_results')
    candidates=list(history.glob('actual/R80/evaluation/low_level/summary.json'))
    assert len(candidates)==1
    low=json.loads(candidates[0].read_bytes())
    assert low['status']=='COMPLETED' and low['expected_images']==low['verified_and_decoded']==2000 and 'queries' not in low
    checks.append('actual_locked_low_level_schema_uses_expected_images_not_invented_queries')
    return dict(status='PASS_FUTURE_FULL_ORDINARY_LIFECYCLE_LOCAL_CPU_PREPARATION',checks=checks,
                source_pins=[F.K.ref(root/name) for name in F.REQUIRED_ENTRIES],
                actual_historical_metric_comparisons=80,actual_historical_delta_input_cells=160000,
                maximum_historical_statistics_error=maximum,new_full1654_method_effects_read=0,
                actual_GPU_profile_or_training_or_generation_executed=False,actual_complete_lifecycle_admission=False,
                historical_role_aliases_for_fixtures=aliases)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--receipt',required=True,type=Path);args=parser.parse_args()
    result=run();args.receipt.parent.mkdir(parents=True,exist_ok=True);assert not args.receipt.exists()
    args.receipt.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
