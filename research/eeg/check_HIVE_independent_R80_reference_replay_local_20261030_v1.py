"""Exercise the future auditor on actual OLD O/native data, not new controls."""
import argparse,copy,json
from pathlib import Path
from brainhive_independent_R80_pair_recompute_CPU_20261030_v1 import load_table,recompute,compare_published,check_pair_identity

parser=argparse.ArgumentParser();parser.add_argument('--source',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
root=Path(args.source);candidate={kind:load_table(root/'candidate'/(kind+'.csv')) for kind in ('common','paper')};parent={kind:load_table(root/'baseline'/(kind+'.csv')) for kind in ('common','paper')}
original=json.loads((root/'metadata/paired_summary.json').read_bytes());expected={}
for metric,item in original['results'].items():
    expected[metric]=dict(status=item['status'],left_mean=item['ordinary_mean'],right_mean=item['parent_mean'],paired_left_minus_right=item['paired_ordinary_minus_parent'],descriptive_subject_concept_bootstrap95=item['descriptive_subject_concept_bootstrap95'],defined=item['defined'],direction=item['direction'])
actual=recompute(candidate,parent);maximum=compare_published(actual,expected)
key=next(iter(parent['common']));bad=copy.deepcopy(parent['common']);bad[key]['score_target_path']='deliberately_wrong_engineering_target'
try:check_pair_identity(candidate['common'],bad)
except AssertionError:pass
else:raise AssertionError('Wrongtargetadmitted')
partial=dict(parent['common']);partial.pop(key)
try:check_pair_identity(candidate['common'],partial)
except AssertionError:pass
else:raise AssertionError('Incompletequerysetadmitted')
result=dict(status='PASS_INDEPENDENT_REAL_OLD_ORDINARY_R80_16METRIC_REFERENCE_REPLAY_ONLY',queries=2000,paired_scalars=32000,metrics=16,maximum_mean_or_interval_error=maximum,SEED_candidate=actual['SEED_left'],SEED_parent=actual['SEED_right'],wrong_target_and_missing_query_rejected=True,production_statistics_module_imported=False,new_control_results_used=False,new_images_or_GPU_calls=0,new_method_effect_assessed=False)
path=Path(args.output);assert not path.exists();path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
