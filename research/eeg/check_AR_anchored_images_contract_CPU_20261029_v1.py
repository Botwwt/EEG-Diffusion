"""Actual-panel CPU integration with explicitly synthetic model/image fixtures."""
import copy,hashlib,json,os,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import AR_anchored_images_contract_CPU_20261029_v1 as C
import AR_anchored_images_artifacts_CPU_20261029_v1 as Audit
import AR_anchored_images_score_CPU_20261029_v1 as Score
import AR_anchored_image_screen_math_20261029_v1 as Math

L=Path(__file__).resolve().parent;result_dir=L/'audit/AR_anchored_images_20261029_v1';result_dir.mkdir(exist_ok=True);checks={}
def expect(name,call):
    try:call()
    except (AssertionError,ValueError,FileNotFoundError,KeyError):checks[name]=True;return
    raise AssertionError(name+' accepted invalid fixture')
root=Path(tempfile.mkdtemp(prefix='CPU_explicitly_synthetic_fixture_',dir=result_dir));output=root/'AR_anchored_images_v1'
metadata=L/'audit/AR_anchored_images_20261029_v1/fixed_metadata_v1';panel_path=metadata/'panel.json';policy_path=L/'configs/AR_anchored_image_evaluation_DRAFT_20261029_v1/policy.json';panel=json.loads(panel_path.read_bytes())
assert C.sha(panel_path)==C.PANEL_SHA and C.sha(policy_path)==C.POLICY_SHA
training=dict(epochs=10,updates_per_arm=8270,seed=1729,logical_batch=16,micro_batch=2,query_exposures_per_arm=132300)
science=dict(status='LOCKED_SHARED_THREE_ARM_AR_SCIENCE_BEFORE_FORMAL_FIT_v1',arms=list(C.ARMS),assigned_hosts={'new':['native_global'],'old':['paired_anchored','donor_anchored']},training=training,initial_generator_tensor_digest=C.INITIAL,image_policy=C.ref(policy_path),panel=C.ref(panel_path),image_binding=C.ref(metadata/'panel_completion.json'),donor_training_only=True,CPU_fixture_only_not_real_training=True)
science_path=root/'SYNTHETIC_shared_science.json';C.write_once(science_path,science)
training_refs={};done_paths={};model_paths={}
for arm in C.ARMS:
    train=root/(arm+'_SYNTHETIC_training_admission.json');C.write_once(train,dict(status='FORMAL_ANCHORED_AR_MICRO2_ADMITTED_v1',arms=[arm],training=training,shared_science=C.ref(science_path),initial_generator_tensor_digest=C.INITIAL,image_policy=C.ref(policy_path),data=dict(EEG_completion={'sha256':C.EEG_SHA},token_completion={'sha256':C.TOKEN_SHA}),CPU_fixture_only=True))
    training_refs[arm]=C.ref(train);model=root/(arm+'_DUMMY_NOT_A_TORCH_MODEL.pt');model.write_bytes(b'CPU metadata fixture; never a model');model_paths[arm]=str(model)
    done=root/(arm+'_SYNTHETIC_completion.json');C.write_once(done,dict(status='COMPLETE_FIXED_T_10_EPOCH_ANCHORED_AR_ARM',arm=arm,admission=C.ref(train),epochs=10,updates=8270,query_exposures=132300,new_research_images=0,old_ordinary_unchanged=True,native_reads_local_memory_file=arm!='native_global',committed_log_chain=[dict(start_update=1,end_update=8270,CPU_fixture_only=True)],final_model=C.ref(model),CPU_fixture_only=True));done_paths[arm]=str(done)
source_names=['AR_anchored_images_contract_CPU_20261029_v1.py','AR_anchored_images_generate_GPU_20261029_v1.py','AR_anchored_images_artifacts_CPU_20261029_v1.py','AR_anchored_images_features_GPU_20261029_v1.py','AR_anchored_images_score_CPU_20261029_v1.py','AR_anchored_images_queue_CPU_20261029_v1.py','AR_anchored_image_screen_math_20261029_v1.py','score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2.py','EEG_generative_backend_loader_20261029_v1.py','EEG_anchored_attention_20261029_v1.py','eeg_multiscale_prefix_20261028_v1.py','image_evaluation.py','semantic_evaluator.py']
a=dict(status='AR_ANCHORED_IMAGES_ADMITTED_v1',host_role='old',GPU_UUID=C.HOST_UUID['old'],output_root=str(output),policy=C.ref(policy_path),panel=C.ref(panel_path),shared_science=C.ref(science_path),training_admissions=training_refs,fit_completion_paths=done_paths,model_paths=model_paths,
    generation_assignment={'new':['native_global','paired_anchored'],'old':['donor_anchored']},assigned_arms=['donor_anchored'],GPU_predecessor_controller={'pid':1,'start_ticks':1,'CPU_FIXTURE_ONLY':True},
    early_generation_gate='OWN_FIXED_FINAL_AND_SHARED_FIXED_THREE_ARM_TRAINING_RECIPE',early_generation_limit=80,inspection_and_effect_gate='ALL_THREE_FIXED_FITS_AND_ALL_THREE_STAGE_ARTIFACTS_AND_FEATURES',expansion_gate='LOCKED_ALL_THREE_80_DECISION_EXPAND400',
    resources=dict(GPU_job_seconds=900,body_hard_seconds=840,maximum_GPU_allocated_bytes=20<<30,maximum_extra_CPU_bytes=10<<30,maximum_queries_per_chunk=20,unreserved_disk_buffer_bytes=2<<30,maximum_CPU_audit_seconds=300,maximum_CPU_score_seconds=300),generation_cost=dict(native_scale_steps=10,CFG_branches=2,image_tokens_per_query=680,UNet_forwards=0,equivalent_to_HIVE_4_NFE=False),
    source_pins=[C.ref(L/name) for name in source_names],DINO_weight={'path':'/CPU_fixture_no_weights_or_GPU/DINO.pth','bytes':0,'sha256':C.DINO_SHA},image_evaluation_sha256=C.sha(L/'image_evaluation.py'),stage80_decision_path=str(output/'screens/stage80/summary.json'),target_cache=C.ref(metadata/'target_cache.json'),
    target_features=C.ref(L/'audit/spatial_innovation_20261028_v1/final400_received_v1/27c833cf67_targets400.npy'),CPU_SYNTHETIC_FIXTURE_NOT_EXECUTION_ADMISSION=True)
admission=root/'SYNTHETIC_image_admission.json';C.write_once(admission,a);C.load_admission(admission);C.validate_sources(a,L/'AR_anchored_images_score_CPU_20261029_v1.py');C.validate_shared_recipe(a);C.validate_panel(a)
checks['actual_400_panel_and_fixed_seed_join']=True
checks['shared_three_arm_frozen_recipe_join']=True
bad_admission=root/'BAD_PID.json';bad_spec=copy.deepcopy(a);bad_spec['GPU_predecessor_controller']['pid']=None;C.write_once(bad_admission,bad_spec)
expect('unbound_GPU_predecessor_is_rejected',lambda:C.load_admission(bad_admission))
bad_admission=root/'DUPLICATED_ARM.json';bad_spec=copy.deepcopy(a);bad_spec['generation_assignment']['old'].append('paired_anchored');C.write_once(bad_admission,bad_spec)
expect('duplicate_generation_assignment_is_rejected',lambda:C.load_admission(bad_admission))
for arm in C.ARMS:C.validate_fit(a,arm,model=True)
checks['all_three_final_metadata_source_joins']=True
strip=C.stripped_query(panel['queries'][0]);checks['inference_query_whitelist_excludes_concept_target_and_gallery']=not ({'concept_id','target_image','target_index','original_exposed_panel_HIVE_seed'}&set(strip))
for stage,lo,hi in [(80,0,20),(80,60,80),(400,80,100),(400,380,400)]:C.chunk_bounds(stage,lo,hi)
checks['fixed20_chunks_and400_excludes80_regeneration']=True
expect('no_stage400_before_locked80_decision',lambda:C.validate_expansion(a,400))
for name,args in [('cannot_regenerate80_in400',(400,0,20)),('no_partial19_query_chunk',(80,0,19)),('no_outside400_chunk',(400,400,420))]:expect(name,lambda args=args:C.chunk_bounds(*args))
draft=root/'DRAFT.json';C.write_once(draft,dict(status='DRAFT',output_root=str(root/'MUST_NOT_EXIST')))
expect('draft_admission_rejected_before_outputs',lambda:C.load_admission(draft));assert not (root/'MUST_NOT_EXIST').exists()
process=subprocess.run([sys.executable,'-B',str(L/'AR_anchored_images_generate_GPU_20261029_v1.py'),'--admission',str(draft),'--arm','native_global','--stage','80','--lo','0','--hi','20','--job-name','CPU_DRAFT_REJECTION_ONLY'],capture_output=True,text=True)
checks['real_GPU_CLI_draft_exits64_without_torch_or_outputs']=process.returncode==64 and 'NOT_ADMITTED_' in process.stdout and not (root/'MUST_NOT_EXIST').exists()

# Three complete synthetic image/feature ledgers exercise production joins.
# These are labelled CPU fixtures, NOT real generation/feature extraction.
output.mkdir();targets=np.load(C.checked(a['target_features']),allow_pickle=False)
for arm in C.ARMS:
    arm_root=output/arm
    for name in ['images','records','generation_chunks','feature_chunks','artifacts']:(arm_root/name).mkdir(parents=True)
    for lo in range(0,80,20):
        chunk=arm_root/'generation_chunks'/f'{lo:05d}_{lo+20:05d}';chunk.mkdir();binding=chunk/'binding.json'
        C.write_once(binding,dict(schema='AR_ANCHORED_GENERATION_CHUNK_BINDING_v1',arm=arm,range=[lo,lo+20],admission=C.ref(admission),policy=a['policy'],panel=a['panel'],fit_completion=C.ref(Path(done_paths[arm])),model=C.ref(Path(model_paths[arm])),CPU_SYNTHETIC_FIXTURE=True))
        refs=[]
        for index in range(lo,lo+20):
            row=panel['queries'][index];image,record=C.image_paths(a,arm,index);y,x=np.indices((256,256));pixels=np.stack((x,y,np.full_like(x,index*3)),axis=-1).astype(np.uint8);Image.fromarray(pixels).save(image)
            C.write_once(record,dict(schema='AR_ANCHORED_SINGLE_FIXED_IMAGE_v1',arm=arm,query_index=index,query=C.stripped_query(row),sampling_seed=row['image_seed'],policy_sha256=C.POLICY_SHA,panel_sha256=C.PANEL_SHA,
                inference_inputs='own_global_c' if arm=='native_global' else 'own_global_c_and_own_local_H',normal_inference_reads_query_image_or_label_or_gallery=False,best_of_N=1,image=C.ref(image),resolution=[256,256],raw_pixels_dtype='uint8_RGB',raw_pixel_SHA256=hashlib.sha256(pixels.tobytes()).hexdigest(),generation_binding=C.ref(binding),fit_completion=C.ref(Path(done_paths[arm])),model_sha256=C.sha(Path(model_paths[arm])),CPU_SYNTHETIC_FIXTURE=True));refs.append(C.ref(record))
        C.write_once(chunk/'completion.json',dict(status='COMPLETE_FIXED_AR_EEG_ONLY_IMAGE_CHUNK',records=refs,CPU_SYNTHETIC_FIXTURE=True))
    result=Audit.audit(a,arm,80,panel);assert result['status']=='COMPLETE_FIXED_AR_ALL_QUERY_ARTIFACT_AUDIT';C.write_once(arm_root/'artifacts/stage80.json',result)
    for lo in range(0,80,20):
        chunk=arm_root/'feature_chunks'/f'{lo:05d}_{lo+20:05d}';chunk.mkdir();rows=[]
        for index in range(lo,lo+20):
            record,rr=C.validate_record(a,arm,index,panel['queries'][index]);rows.append(dict(query_index=index,query_id=panel['queries'][index]['query_id'],record=rr,image=record['image'],raw_pixel_SHA256=record['raw_pixel_SHA256']))
        C.write_once(chunk/'rows.json',rows);np.save(chunk/'features.npy',targets[lo:lo+20].astype(np.float32),allow_pickle=False)
        C.write_once(chunk/'completion.json',dict(status='COMPLETE_FIXED_AR_FROZEN_DINO_FEATURE_CHUNK',arm=arm,range=[lo,lo+20],stage=80,queries=20,feature_dimension=768,DINO_weight=a['DINO_weight'],image_evaluation_source=C.ref(L/'image_evaluation.py'),target_features_or_scores_read=False,effect_metrics_computed=0,rows=C.ref(chunk/'rows.json'),features=C.ref(chunk/'features.npy'),CPU_SYNTHETIC_FEATURE_FIXTURE_NO_GPU=True))
checks['full80_artifact_byte_pixel_identity_schema_interop_all3']=True
record_path=C.image_paths(a,'native_global',0)[1];original=record_path.read_bytes();wrong=json.loads(original);wrong['sampling_seed']+=1;record_path.write_text(json.dumps(wrong))
expect('wrong_query_seed_is_rejected',lambda:C.validate_record(a,'native_global',0,panel['queries'][0]));record_path.write_bytes(original)
image=C.image_paths(a,'native_global',0)[0];original=image.read_bytes();image.write_bytes(b'CORRUPT_CPU_FIXTURE')
bad=Audit.audit(a,'native_global',80,panel);checks['corrupt_query_keeps_full80_denominator']=bad['expected_queries']==80 and bad['valid_queries']==79 and bad['invalid_queries']==1;image.write_bytes(original)
donor_done=Path(done_paths['donor_anchored']);original=donor_done.read_bytes();donor_done.unlink()
expect('no_effect_before_third_fixed_fit',lambda:Score.run(admission,80));assert not (output/'screens').exists();donor_done.write_bytes(original)
missing=output/'paired_anchored/feature_chunks/00000_00020/completion.json';original=missing.read_bytes();missing.unlink()
expect('no_effect_before_every_arm80_features_close',lambda:Score.run(admission,80));assert not (output/'screens').exists();missing.write_bytes(original)
source=C.image_paths(a,'native_global',0)[0];destination=C.image_paths(a,'donor_anchored',0)[0];original=destination.read_bytes();destination.unlink();os.link(source,destination)
expect('cross_arm_hardlinked_image_is_rejected_before_scores',lambda:Score.run(admission,80));assert not (output/'screens').exists();destination.unlink();destination.write_bytes(original)
Score.run(admission,80);summary=json.loads((output/'screens/stage80/summary.json').read_bytes());checks['actual_all3_score_schema_and_independent_rescore_interop']=summary['status']=='COMPLETE_FIXED_THREE_ARM_AR_IMAGE_SCREEN' and summary['decision']['action']=='EXPAND400'
assert summary['full_query_denominator_per_arm']==80 and len(summary['contrasts'])==3 and summary['SOTA_established'] is False
checks['all3_fixed_contrasts_no_winner_selection_and_no_SOTA']=True
C.validate_expansion(a,400);checks['locked80_expansion_reuses_prefix_sources']=True
try:Score.run(admission,80)
except (AssertionError,FileExistsError):checks['score_outputs_are_immutable']=True
else:raise AssertionError('score output overwritten')
assert all(checks.values())
receipt=dict(status='PASS_AR_IMAGE_CONTRACT_ACTUAL_PANEL_AND_SYNTHETIC_ARTIFACT_FEATURE_CPU_INTEGRATION',checks=checks,actual_panel_SHA256=C.PANEL_SHA,fixture_directory=str(root),fixture_scope='Actual immutable400querypanel; dummy final metadata, synthetic PNGs and target-as-candidate feature fixtures only. NO real models, generation, DINO encoder or research effect.',new_GPU_jobs=0,new_research_images=0,temporary_fixture_images_not_research_images=240,production_sources={name:C.ref(L/name) for name in source_names[:6]})
path=result_dir/'CPU_contract_actual_v2.json';assert not path.exists();C.write_once(path,receipt);print(json.dumps(dict(status=receipt['status'],checks=len(checks),receipt=C.ref(path))))
