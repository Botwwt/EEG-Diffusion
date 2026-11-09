"""Independent full-official metadata arithmetic after benchmark exposure.

This is not a method lock, resource reservation, payload audit or fit admission.
It never imports model libraries, opens tensors, edits legacy gates or launches.
"""
import collections,csv,gzip,hashlib,json,math,random
from pathlib import Path
L=Path(__file__).resolve().parent;D=L/'audit/official_fullfit_metadata_plan_20261022_v1'
def read(p):return json.loads(p.read_bytes())
def ref(p):
    data=p.read_bytes();return dict(path=str(p.relative_to(L)).replace('\\','/'),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
plan=read(D/'official_fullfit_plan.json');folds=read(D/'official_fullfit_two_folds.json');tasks=read(D/'official_fullfit_encoder_tasks.json');split=read(L/'configs/split_manifest.json')
with (L/'audit/image_manifest.csv').open(newline='',encoding='utf8') as f:images=list(csv.DictReader(f))
train={int(r['array_row']):r for r in images if r['dataset_split']=='train'};test=[r for r in images if r['dataset_split']=='test']
assert len(train)==16540 and set(train)==set(range(16540)) and len(test)==200
concepts=sorted({r['concept_id'] for r in train.values()});assert len(concepts)==1654
assert len({r['stimulus_id'] for r in train.values()})==16540
old={k:set(v) for k,v in split['image_rows'].items()};assert {k:len(v) for k,v in old.items()}=={'T':13230,'D':1650,'V':1660}
assert not old['T']&old['D'] and not old['T']&old['V'] and not old['D']&old['V'] and set.union(*old.values())==set(train)
shuffled=concepts.copy();random.Random(1729).shuffle(shuffled);independent_fold={c:i%2 for i,c in enumerate(shuffled)}
assert folds['fold_by_concept']==independent_fold and folds['training_seed']==1729 and folds['official_train_only']
assert len(tasks['tasks'])==tasks['task_count']==20
subjects=[f'sub-{n:02d}' for n in range(1,11)];keyed={t['task_id']:t for t in tasks['tasks']};assert len(keyed)==20
per_task={};rows_by_task={};export_position={}
for sub in subjects:
    both=[]
    for held in (0,1):
        task=keyed[sub+'/official_oof'+str(held)];fit=set(task['training_array_rows']);export=set(task['exported_OOF_array_rows'])
        assert task['subject']==sub and task['held_fold']==held and task['training_seed']==1729
        assert len(fit)==len(export)==8270 and not fit&export and fit|export==set(train)
        fc={train[i]['concept_id'] for i in fit};ec={train[i]['concept_id'] for i in export};assert len(fc)==len(ec)==827 and not fc&ec
        assert fc==set(task['training_concepts']) and ec==set(task['held_concepts'])
        assert all(independent_fold[train[i]['concept_id']]==held for i in export)
        assert (task['epochs'],task['batch_size'],task['expected_updates'],task['expected_training_exposures'],task['final_batch_size'])==(25,1024,225,206750,78)
        assert all(task[k] is None for k in ('checkpoint_path','checkpoint_sha256','condition_cache_path','condition_cache_sha256'))
        assert task['parameter_updates']==0 and not task['existing_protected_T_OOF_reused']
        both.extend(export);rows_by_task[task['task_id']]=export;export_position[task['task_id']]={r:i for i,r in enumerate(task['exported_OOF_array_rows'])}
        per_task[task['task_id']]=dict(fit_images=8270,export_images=8270,fit_concepts=827,held_concepts=827,updates=225,exposures=206750)
    assert len(both)==len(set(both))==16540
seen=set();groups=collections.Counter();counts=collections.Counter();linehash=hashlib.sha256()
with gzip.open(D/'official_fullfit_OOF_query_map.jsonl.gz','rb') as f:
    for i,line in enumerate(f):
        q=json.loads(line);row=q['array_row'];task_id=q['encoder_task_id'];key=q['subject'],row
        assert key not in seen and q['query_index']==i and q['subject'] in subjects and row in train
        assert q['training_seed']==1729 and q['dataset_split']=='official_train' and not q['official_benchmark_query']
        assert q['concept_id']==train[row]['concept_id'] and q['stimulus_id']==train[row]['stimulus_id']
        assert task_id==q['subject']+'/official_oof'+str(independent_fold[q['concept_id']])
        assert row in rows_by_task[task_id] and q['future_OOF_cache_row']==export_position[task_id][row]
        assert q['EEG_repeat_protocol']=='R4' and q['physical_repeat_count']==4 and not q['actual_condition_available']
        assert all(q[k] is None for k in ('condition_cache_path','condition_cache_sha256','encoder_checkpoint_path','encoder_checkpoint_sha256'))
        group=next(k for k,v in old.items() if row in v);assert q['previous_development_group']==group
        seen.add(key);groups[group]+=1;counts[q['subject']]+=1;linehash.update(line)
assert len(seen)==165400 and all(counts[sub]==16540 for sub in subjects)
assert dict(groups)=={'T':132300,'D':16500,'V':16600}
ct=plan['component_training'];assert (ct['unique_training_queries'],ct['passes'],ct['batch_size'],ct['final_batch_size'],ct['expected_updates'],ct['expected_exposures'])==(165400,3,16,8,31014,496200)
legacy=(L/'brainhive_official_execution_contract_CPU_v1.py').read_text(encoding='utf8');trainer=(L/'train_BrainHIVE_official_encoder_v1.py').read_text(encoding='utf8')
assert "method['official_result_access_used_for_lock'] is False" in legacy and "MIN_ROLES = {'parent', 'O', 'P', 'PN', 'S'}" in legacy
assert 'MIN_OUTPUT_BYTES = 5 << 29' in trainer
result=dict(status='PASS_INDEPENDENT_OFFICIAL1654_METADATA_SCOPE_EXPOSURE_AND_UPDATE_ACCOUNTING_NO_ADMISSION',source=ref(Path(__file__)),inputs=[ref(D/name) for name in ('official_fullfit_plan.json','official_fullfit_two_folds.json','official_fullfit_encoder_tasks.json','official_fullfit_OOF_query_map.jsonl.gz')]+[ref(L/'configs/split_manifest.json'),ref(L/'audit/image_manifest.csv'),ref(L/'brainhive_official_execution_contract_CPU_v1.py'),ref(L/'train_BrainHIVE_official_encoder_v1.py')],official_training_concepts=1654,official_training_images=16540,official_OOF_queries=165400,subjects=10,training_seeds=[1729],OOF_tasks=20,OOF_updates_per_task=225,OOF_total_updates=4500,OOF_total_exposures=4135000,per_component_updates=31014,per_component_exposures=496200,per_component_last_batch=8,former_development_group_queries=dict(groups),query_map_uncompressed_SHA256=linehash.hexdigest(),per_task=per_task,legacy_executor_requires_official_result_access_used_for_lock_false=True,current_actual_official_result_access_used_for_direction_selection=True,legacy_executor_usable_unchanged=False,legacy_PN_P_S_minimum_roles_do_not_match_current_candidate_program=True,legacy_gate_not_edited=True,legacy_per_encoder_minimum_output_bytes=5<<29,legacy20_encoder_minimum_output_bytes=20*(5<<29),legacy20_encoder_minimum_output_GiB=50,new_whole_lifecycle_resource_review_required=True,actual_server_space_or_payload_availability_checked=False,new_method_or_role_selection_locked=False,new_fit_or_export_admitted=False,new_training_models_or_updates_or_images=0,model_libraries_imported=False,current_norm_only_experiment_source_modified=False,independent_H=False,SOTA_established=False)
out=L/'audit/HIVE_single_seed_full1654_scope_review_20261030_v1';assert not out.exists();out.mkdir();(out/'completion.json').write_bytes((json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode());print(json.dumps({k:v for k,v in result.items() if k not in ('inputs','per_task')},indent=2))
