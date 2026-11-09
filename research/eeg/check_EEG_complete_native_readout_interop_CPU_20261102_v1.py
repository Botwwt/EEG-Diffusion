"""Execute all400 CPU prediction and ten-arm stratified80 audit/score schemas.

All EEG weights, images, uniforms and features here are SYNTHETIC fixtures.
This checks real program interoperation, never pretrained efficacy/profile.
"""
import argparse,hashlib,json,os,tempfile,time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from PIL import Image
import EEG_complete_native_study_contract_CPU_20261102_v1 as C
import EEG_complete_native_study_core_20261102_v1 as M
import EEG_complete_native_runtime_20261102_v1 as R
import audit_EEG_complete_native_predictions_images_CPU_20261102_v1 as I
import score_EEG_complete_native_study_CPU_20261102_v1 as S
from check_EEG_complete_native_study_CPU_20261102_v1 import SmallEEG

L=Path(__file__).resolve().parent
def write(path,value):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(value)+'\n',encoding='utf8');return C.ref(path)
def run(output):
 started=time.monotonic();rng=np.random.default_rng(1729);torch.manual_seed(1729);torch.set_num_threads(2);checks={}
 audit=L/'audit/VAR_full_EEG_study_preparation_20261102_v1';audit.mkdir(exist_ok=True)
 with tempfile.TemporaryDirectory(prefix='EXPLICIT_SYNTHETIC_READOUT_',dir=audit) as temporary:
  root=Path(temporary).resolve();assert root.is_relative_to(audit.resolve());code=root/'code';out=root/'out';code.mkdir();out.mkdir();protocol=code/'protocol.json';write(protocol,dict(purpose='EXPLICIT_SYNTHETIC_PROGRAM_INTEROP_NOT_REAL_PROFILE_OR_METHOD'))
  q=dict(CPU_prediction_seconds=900,CPU_prediction_tolerance=dict(atol=5e-4,rtol=1e-4),CPU_artifact_seconds=1800,CPU_score_seconds=1800,inference=dict(SYNTHETIC_INTERFACE_TEST=True))
  p_ref=C.ref(protocol);panel=[dict(query_id=f'q{i}',stimulus_id=f's{i}',array_row=i,concept_id=i//10) for i in range(400)];selected=[i for i in range(400) if i%10<2]
  ppanel=write(root/'panel.json',dict(queries=panel,target_gallery=panel));joined=write(root/'joined.json',dict(stage80=selected,T_rows=panel,Dcheck_rows=panel))
  normalized=rng.normal(size=(400,4)).astype(np.float32);np.save(root/'EEG.npy',normalized)
  write(out/'inputs/completion.json',dict(status='PASS_COLD_COMPLETE_T13230_DCHECK400_REAL_EEG_INPUT_JOIN_TRANSFORM',identities=joined,files=dict(Dcheck=C.ref(root/'EEG.npy'))))
  import image_evaluation
  evaluation=C.ref(image_evaluation.__file__);image=dict(panel=ppanel,DINO_weight=dict(sha256=image_evaluation.DINO_SHA256),image_evaluation=evaluation);q['legacy_image_spec']=write(root/'image.json',image)
  import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as locked
  import EEG_complete_native_independent_scores_CPU_20261102_v1 as alternate
  q['metric_sources']=[C.ref(v.__file__) for v in (locked,alternate)]
  targets=rng.normal(size=(400,768)).astype(np.float32);targets/=np.linalg.norm(targets,axis=-1,keepdims=True);np.save(root/'target.npy',targets);q['target_features']=C.ref(root/'target.npy')
  # Real cache and generated evaluator use two verified paths with equal bytes.
  cache_eval=dict(evaluation,path='/historical/verified/equal_content/image_evaluation.py')
  q['target_cache']=write(root/'cache.json',dict(status='EXISTING_LOCKED_TARGET_FEATURES_REUSED_WITH_COMPLETE_AR_EEG_IDENTITY_JOIN',panel=ppanel,features=q['target_features'],encoder=dict(weight_sha256=image_evaluation.DINO_SHA256,image_evaluation=cache_eval)))
  def factory(q,arm,device):return (M.ImagePrior() if arm=='image_prior_full' else M.EEGDistribution(SmallEEG(),arm)).to(device).eval()
  prediction_files={};model_refs={};conditions={};predroot=out/'tasks/predict_final400';predroot.mkdir(parents=True)
  for arm in C.ARMS:
   model=factory(q,arm,'cpu')
   with torch.no_grad():
    if arm=='image_prior_full':model.logits.copy_(torch.linspace(-1,1,1000))
    else:model.encoder.head.weight.normal_(0,.05);model.encoder.head.bias.normal_(0,.02)
    logits,stay=model(torch.from_numpy(normalized),input_chans=[0]);v=logits[:,0:1].numpy().copy() if model.family=='global' else logits.numpy().copy();z=stay.numpy().copy()
   fit=out/'fits'/arm;fit.mkdir(parents=True);torch.save(dict(purpose='FORMAL_FIXED_FULL_T_FINAL_ONLY',protocol=p_ref,arm=arm,updates=2070,model=R.cpu_tree(model.state_dict())),fit/'model_final.pt');pin=C.ref(fit/'model_final.pt');model_refs[arm]=pin
   write(fit/'completion.json',dict(status='COMPLETE_FIXED10_FULL_T_EEG_ARM',arm=arm,epoch=10,updates=2070,protocol=p_ref,final_model=pin));write(fit/'fit10_CPU_audit.json',dict(status='PASS_ALL_FIXED_EEG_EPOCH_STATE_ADAM_COVERAGE',arm=arm,epoch=10,updates=2070,protocol=p_ref,final_model=pin))
   np.savez_compressed(predroot/(arm+'.npz'),logits=v,stay=z);prediction_files[arm]=C.ref(predroot/(arm+'.npz'));conditions[arm]=(np.repeat(v,10,axis=1) if v.shape[1]==1 else v,z)
  write(predroot/'completion.json',dict(status='COMPLETE_ALL6_FINAL_EEG_ONLY_DCHECK400_PREDICTIONS',protocol=p_ref,files=prediction_files,models=model_refs,query_images_or_labels_read_by_predictor=False,EEG_input=C.ref(root/'EEG.npy')))
  with ExitStack() as patches:
   patches.enter_context(patch.object(C,'CODE',code));patches.enter_context(patch.object(C,'OUT',out));patches.enter_context(patch.object(C,'setup',lambda *a,**k:q));patches.enter_context(patch.object(C,'resources',lambda *a,**k:{}));patches.enter_context(patch.object(C,'roots',lambda:{}));patches.enter_context(patch.object(I,'guard_factory',lambda *a,**k:lambda:None));patches.enter_context(patch.object(S,'guard_factory',lambda *a,**k:lambda:None));patches.enter_context(patch.object(R,'model',factory));patches.enter_context(patch.object(R,'inputs',lambda *a,**k:normalized));patches.enter_context(patch.object(R,'channels',lambda *a,**k:[0]));patches.enter_context(patch.dict(os.environ,CUDA_VISIBLE_DEVICES=''))
   I.predictions(protocol);cpu=C.read(out/'final_predictions_CPU_audit.json');assert cpu['all_prediction_cells']==9600000+21600
   assert cpu['noEEG_reversed_and_scaled_input_invariance'] and all(v['max_logits_error']<2e-6 for v in cpu['errors'].values());checks['all6_real_prediction_audit_with_synthetic400_models_and_inputs']=dict(cells=cpu['all_prediction_cells'],new_real_models=False)
   same,different=I.donor_maps(panel)
   for part in range(4):
    work=out/'tasks'/f'images80_{part:02d}';work.mkdir();ids=selected[part*20:(part+1)*20];record_refs={};feature_refs={}
    for arm in I.all_image_arms():
     base,intervention=I.split_arm(arm);directory=work/arm;directory.mkdir();records=[];v,z=conditions[base]
     for index in ids:
      row=panel[index];source=same[index] if intervention=='same' else different[index] if intervention=='different' else index
      private=np.random.default_rng(I.seed(row['stimulus_id'],'latent'));uniforms=dict(initial_uniform=private.random((1,1)).tolist(),stay_uniform=private.random((1,9)).tolist(),refresh_uniform=private.random((1,9)).tolist());chain,_=I.independent_chain(v[source],z[source],arm,uniforms)
      raw=np.zeros((256,256,3),np.uint8);raw[:]=[index%256,(index*3)%256,200];path=directory/f'{index:04d}.png';Image.fromarray(raw).save(path)
      trace=[];prefix=0
      for j,side in enumerate((1,2,3,4,5,6,8,10,13,16)):
       prefix+=side*side;trace.append(dict(scale=j,classes=[chain[j]],tokens=side*side,prefix_tokens=prefix,prefix_recomputed_under_current_class=True,sampled_codes_SHA256='b'*64,logits_SHA256='c'*64))
      records.append(dict(arm=arm,base_arm=base,subject='sub-01',query_index=index,stimulus_id=row['stimulus_id'],array_row=row['array_row'],EEG_prediction_row=None if base=='image_prior_full' else source,EEG_intervention=intervention,model_source=model_refs[base],latent_seed=I.seed(row['stimulus_id'],'latent'),token_seed=I.seed(row['stimulus_id'],'token'),latent_uniforms=uniforms,latent_class_chain=chain,latent_rng_SHA256=dict(uniform_draws_per_query=19,no_best_of_N=True,private_latent_rng_before='d'*64,private_latent_rng_after='e'*64),prior_logits_SHA256=hashlib.sha256(v[source:source+1].tobytes()).hexdigest(),prior_stay_SHA256=hashlib.sha256(z[source:source+1].tobytes()).hexdigest(),image=C.ref(path),pixel_SHA256=hashlib.sha256(raw.tobytes()).hexdigest(),native_table_SHA256='a'*64,renderer_trace=dict(scales=10,actual_generated_tokens=680,CFG=0,full_prefix_transformer_calls=10,one_image_per_query=True,native_cached_sampling_used=False,normal_inference_target_access=False,extra_learned_parameters=0,records=trace),normal_inference_target_or_label_access=False,labels_only_for_registered_donor_permutations=intervention!='correct',generation=q['inference']))
     record_refs[arm]=write(directory/'records.json',records);np.save(directory/'features.npy',targets[ids]);feature_refs[arm]=C.ref(directory/'features.npy')
    write(work/'completion.json',dict(status='COMPLETE_ALL10_MATCHED_EEG_METHOD_AND_DONOR_IMAGE_CHUNK',protocol=p_ref,arms=I.all_image_arms(),frozen_VAR_VQ_unchanged=True,actual_native_calls_per_image=10,engineering_only=False,stage=80,part=part,query_indices=ids,new_research_images=200,records=record_refs,features=feature_refs))
   I.artifacts(protocol,'stage80');art=C.read(out/'artifacts_stage80_CPU_audit.json');assert art['images']==800 and art['spatially_constant_images']==800 and art['duplicate_pixel_groups'] and art['all_samples_retained'];checks['all800_actual_synthetic_PNG_identity_latent_and_spatial_constant_audit']=True
   S.run(protocol,80);summary=C.read(out/'screens/stage80/summary.json');assert summary['all_query_metric_cells']==4000 and summary['expand_any'] and all(v['DINO2AFC']==1. for v in summary['means'].values());checks['all10_80query_all40concepts_full_rank_and_bootstrap_interop']=True
   # Changing a donor identity must fail before any new audit can be committed.
   path=out/'tasks/images80_00/global_full_same/records.json';records=C.read(path);records[0]['EEG_prediction_row']=0;write(path,records)
   caught=False
   try:I.artifacts(protocol,'stage80')
   except AssertionError:caught=True
   assert caught;checks['altered_record_file_or_donor_row_rejected']=True
  assert not torch.cuda.is_initialized()
 result=dict(status='PASS_SYNTHETIC_ALL400_PREDICTION_ALL800_IMAGE_AND_ALL10_SCORE_PROGRAM_INTEROP',checks=checks,actual_pretrained_or_GPU_profile_verified=False,actual_new_scientific_images=0,actual_new_scientific_models=0,seconds=time.monotonic()-started)
 output=Path(output);assert not output.exists();temp=output.with_suffix('.pending');temp.write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');temp.replace(output);print(json.dumps(result))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);run(p.parse_args().output)
