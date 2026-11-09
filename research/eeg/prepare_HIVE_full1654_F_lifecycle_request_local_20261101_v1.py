"""Package reviewed source bytes; local scope checks do not admit remote jobs."""
import ast,base64,hashlib,importlib.util,json,shutil,sys
from pathlib import Path
root=Path(__file__).resolve().parent;A=root/'audit/crossdisciplinary_20261028_v1';old=root/'audit/HIVE_loss_controls_R80_single_seed_registration_20261030_v1/protocol'
preparation=root/'audit/HIVE_full1654_F_lifecycle_preparation_20261101_v2';assert not preparation.exists();code=preparation/'code';code.mkdir(parents=True)
new=['brainhive_full1654_F_study_contract_20261101_v1.py','train_HIVE_full1654_F_heads_GPU_20261101_v1.py','audit_HIVE_full1654_F_heads_CPU_20261101_v1.py','run_HIVE_full1654_F_R80_GPU_20261101_v1.py','audit_HIVE_full1654_R80_CPU_20261101_v1.py','compare_HIVE_full1654_F_R80_CPU_20261101_v1.py','queue_HIVE_full1654_F_lifecycle_CPU_20261101_v1.py','check_HIVE_full1654_F_lifecycle_CPU_20261101_v1.py','brainhive_exposed_fullfit_component_inputs_CPU_20261101_v1.py','brainhive_bounded_component_v1.py','brainhive_ordinary_geometry_control_20261030_v1.py']
frozen=['profile_HIVE_loss_control_R80_GPU_20261030_v1.py','score_HIVE_loss_control_R80_author_GPU_20261030_v1.py','run_HIVE_R80_evaluation_guarded_20261029_v1.py','brainhive_loss_control_R80_statistics_CPU_20261030_v1.py','brainhive_owned_process_CPU_20261029_v1.py']
sources={}
for name in new+frozen:
    source=(root if name in new else old)/name;raw=source.read_bytes();ast.parse(raw.decode('utf8'));(code/name).write_bytes(raw)
    sources[name]=dict(base64=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest())
profile=json.loads((root/'audit/HIVE_exposed_full1654_F_geometry_profile_20261101_v1/code/profile_protocol.json').read_bytes())
completion_file=root/'audit/HIVE_exposed_full1654_F_geometry_profile_20261101_v1/code/profile_completion.json'
def remote_ref(local,path):
    raw=Path(local).read_bytes();return dict(path=path,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
actual_profile=remote_ref(completion_file,profile['profile_output'])
protocol=json.loads((old/'F_only_protocol.json').read_bytes());st=json.loads((old/'study.json').read_bytes());template=st['execution_template']
evaluation={k:protocol[k] for k in ('input_completion','baseline_inference','baseline_scoring_manifest','gallery_manifest')}
evaluation.update(ordinary_manifest=st['reference_manifests']['ordinary'],parent_CSVs=st['reference_CSVs']['parent'],gallery_features=template['gallery_features'])
evalpins=list(evaluation[k] for k in ('input_completion','baseline_inference','baseline_scoring_manifest','gallery_manifest','ordinary_manifest'))+list(evaluation['parent_CSVs'].values())+template['gallery_feature_files']+[template['gallery_feature_completion'],template['author_metric_asset_receipt']]
R='/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1'
names=['brainhive_paired_study_contract_20261025_v1.py','brainhive_native_models_v1.py','brainhive_component_generator_v1.py','run_BrainHIVE_frozen_function_low_memory_20261025_v3.py','generate_BrainHIVE_neural_clean_fastscreen_20261027_v1.py','brainhive_functional_training_contract_v2.py','extract_image_features.py','score_full_manifest.py','score_low_level.py','locked_metrics.py','semantic_evaluator.py','brainhive_original_metrics_runtime_v1.py']
existing=[remote_ref(root/name,R+'/'+name) for name in names]
q=dict(host='old',sources=sources,existing_source_pins=existing,evaluation=evaluation,evaluation_pins=evalpins,actual_profile=actual_profile,
    consumer_review=profile['consumer_review'],asset_protocol=profile['asset_protocol'],input_bank=profile['input_bank'],runtime_manifest=profile['runtime_manifest'],ordinary=next(p for p in profile['source_pins'] if p['path'].endswith('ordinary/component_final.pt')))
sys.path.insert(0,str(code));import brainhive_full1654_F_study_contract_20261101_v1 as G
fixture=dict(status=G.STATUS,host='old',GPU_UUID=G.GPU,arms=list(G.ARMS),official200_previously_exposed=True,official_results_used_for_previous_direction=True,independent_H=False,
    method=dict(seed=1729,queries=165400,images=16540,subjects=10,passes=3,batch=16,updates=31014,exposures=496200,last_batch=8,parameters=525568,lr=1e-4,weight_decay=.01,gradient_clip=1.,strength=1.,radius=.25,objective='.1 mean(1-cos(raw_corrected_c,F))',fixed_last=True),
    R80=dict(queries=2000,subjects=10,candidates=200,resolution=512,UNet_steps=4,CFG=0,IP_scale=1,images_per_query=1),maximum_output_bytes=12<<30,legacy_floor_bytes=50<<30,disk_buffer_bytes=8<<30,minimum_CPU_headroom_bytes=24<<30,maximum_extra_CPU_bytes=10<<30,maximum_GPU_bytes=20<<30,numerical_guard=dict(atol=1e-5,rtol=5e-6),input_bank=q['input_bank'],source_pins=[G.ref(code/name) for name in new+frozen])
import check_HIVE_full1654_F_lifecycle_CPU_20261101_v1 as K
checks=K.run(fixture);checks['local_schema_fixture_not_execution_admission']=True
(preparation/'actual_local_CPU_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='utf8')
(A/'HIVE_full1654_F_lifecycle_prepare_launch_request_20261101_v2.json').write_text(json.dumps(q,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status='PASS_LOCAL_COMPLETE_LIFECYCLE_SOURCE_AND11_SCOPE_CHECKS',checks=checks['count'],new_source_files=len(new),byte_identical_frozen_helpers=len(frozen),actual_remote_registration_still_required=True)))
