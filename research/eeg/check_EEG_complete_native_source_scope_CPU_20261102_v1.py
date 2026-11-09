"""Cheap source/metadata checks suitable while the original GPU bank is live.

No numpy, torch, model checkpoint, normalized EEG or likelihood array is loaded.
The real-profile gate remains a later, separate formal admission.
"""
import ast,hashlib,json,sys
from pathlib import Path
import EEG_complete_native_study_contract_CPU_20261102_v1 as C

REQUIRED={'EEG_complete_native_study_core_20261102_v1.py','EEG_full_native_chain_objective_20261102_v1.py','EEG_complete_native_bank_reader_CPU_20261102_v1.py','EEG_complete_native_study_contract_CPU_20261102_v1.py','prepare_EEG_complete_native_inputs_CPU_20261102_v1.py','EEG_complete_native_runtime_20261102_v1.py','train_EEG_complete_native_study_GPU_20261102_v1.py','audit_EEG_complete_native_study_CPU_20261102_v1.py','generate_EEG_complete_native_study_GPU_20261102_v1.py','audit_EEG_complete_native_predictions_images_CPU_20261102_v1.py','score_EEG_complete_native_study_CPU_20261102_v1.py','queue_EEG_complete_native_study_CPU_20261102_v1.py','check_EEG_complete_native_study_CPU_20261102_v1.py','check_EEG_complete_native_readout_interop_CPU_20261102_v1.py','check_EEG_complete_native_source_scope_CPU_20261102_v1.py'}

REQUIRED.add('EEG_complete_native_independent_scores_CPU_20261102_v1.py')
def run(protocol):
 q=C.validate(C.read(protocol));assert {Path(p['path']).name for p in q['source_pins']}==REQUIRED and len(q['source_pins'])==len(REQUIRED)
 assert all(Path(p['path']).parent==C.CODE for p in q['source_pins'])
 for pin in q['source_pins']:
  path=C.checked(pin);ast.parse(path.read_text(encoding='utf8'),filename=str(path))
 for pin in q['inherited_sources']+q['metric_sources']:C.checked(pin)
 base,a,image,stats=C.source_inputs(q,binary=False)
 assert a['public_VAR']['bytes']>0 and a['public_VQ']['bytes']>0 and base['full_EEG']['model']['bytes']==24165170
 assert base['full_EEG']['completion']==q['warm_completion'] and base['full_EEG']['normalization']['bytes']==275116
 warm=C.read(C.checked(q['warm_completion']));assert warm['role']=='full' and warm['epochs']==50 and warm['updates']==5200 and warm['query_exposures']==661500
 cache=C.read(C.checked(q['target_cache']));assert cache['panel']==image['panel'] and cache['features']==q['target_features'] and cache['encoder']['feature_dimension']==768
 assert cache['encoder']['weight_sha256']==image['DINO_weight']['sha256'] and all(cache['encoder']['image_evaluation'][k]==image['image_evaluation'][k] for k in ('bytes','sha256'))
 proof=C.read(C.checked(q['local_checks']['gradients_interfaces']))
 assert proof['status']=='PASS_REAL_T8_ALL6_TARGET_GRADIENTS_AND_EXPLICIT_SYNTHETIC_INTERFACE_LIFECYCLE_CHECKS' and proof['checks']['all6_real_saved_T8_targets_and_independent_gradients']['class_and_stay_gradient_cells']==480432
 proof=C.read(C.checked(q['local_checks']['readout_interop']));assert proof['status']=='PASS_SYNTHETIC_ALL400_PREDICTION_ALL800_IMAGE_AND_ALL10_SCORE_PROGRAM_INTEROP' and not proof['actual_pretrained_or_GPU_profile_verified']
 assert not any(name=='torch' or name.startswith('torch.') or name=='numpy' or name.startswith('numpy.') for name in sys.modules)
 return dict(status='PASS_COMPLETE_EEG_ALL_SOURCE_AND_REAL_SMALL_METADATA_SCOPE_NOT_GPU_PROFILE',source_count=len(REQUIRED),inherited_sources_checked=len(q['inherited_sources']),all6_profiles_and_formal_fit_not_admitted=True,no_large_binary_or_CUDA_loaded=True,local_math_and_synthetic_interop_proofs_bound=True,new_updates=0,new_scientific_images=0)

if __name__=='__main__':print(json.dumps(run(sys.argv[1])))
