"""After full receipt only: independent 4method/16metric/5contrast arithmetic."""
import argparse,hashlib,json
from pathlib import Path
from brainhive_independent_R80_pair_recompute_CPU_20261030_v1 import load_table,recompute,compare_published,check_pair_identity

parser=argparse.ArgumentParser();parser.add_argument('--received',required=True);args=parser.parse_args();root=Path(args.received)
received=json.loads((root/'receiving_completion.json').read_bytes());assert not received['exact_controller_live']
assert received['completion']['status']=='COMPLETE_SINGLE_SEED_R80_TWO_LOSS_CONTROLS_AND_FIVE_ALL16_COMPARISONS'
assert received['failure'] is None and all(received['arms'][arm]['image_records']==2000 for arm in ('F_only','token_only'))
manifest=json.loads((root/'receiving_manifest.json').read_bytes())
for name,item in manifest.items():
    raw=(root/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
roles={role:{kind:load_table(root/'tables'/role/(kind+'.csv')) for kind in ('common','paper')} for role in ('parent','ordinary','F_only','token_only')}
for parts in roles.values():check_pair_identity(roles['parent']['common'],parts['common'])
published=json.loads((root/'actual/joint_comparison/summary.json').read_bytes());assert published['all8000_image_file_hashes_and_inodes_verified'] and published['all4000_control_noise_observations_equal_original_ordinary']
assert published['training_seed']==1729 and not published['training_seed_stability_assessed']
comparisons=(('ordinary','F_only'),('ordinary','token_only'),('F_only','parent'),('token_only','parent'),('ordinary','parent'));records={};maximum=0.
assert set(published['results'])=={a+'_minus_'+b for a,b in comparisons}
for left,right in comparisons:
    key=left+'_minus_'+right;replay=recompute(roles[left],roles[right]);error=compare_published(replay,published['results'][key]);records[key]=dict(replay=replay,maximum_error=error);maximum=max(maximum,error)
defined=sum(value['defined'] for record in records.values() for value in record['replay']['metrics'].values() if value['status']=='COMPLETE')
result=dict(status='PASS_LOCAL_INDEPENDENT_FULL2000_ALL4METHODS_16METRICS_5CONTRASTS_RECOMPUTATION' if defined==160000 else 'PASS_ALL_QUERIES_WITH_UNDEFINED_METRICS_RETAINED_UNKNOWN',queries_per_method=2000,methods=4,metrics=16,comparisons=5,paired_metric_scalars=defined,expected_paired_metric_scalars=160000,maximum_mean_or_interval_error=maximum,records=records,all_received_file_SHA_count=len(manifest),production_statistics_imported=False,independent_matrix_count_bootstrap_and_manual_quantiles=True,new_GPU_calls=0,not_independent_visual_feature_or_optimizer_replay=True,training_seed_stability_assessed=False,SOTA_established=False)
path=root/'independent_local_statistics.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(dict(status=result['status'],maximum_error=maximum,paired_scalars=defined)))
