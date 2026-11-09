"""Independent local concrete-source and formal-calibration/policy scope review."""
import argparse,ast,hashlib,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();E=a.directory;read=lambda p:json.loads(Path(p).read_bytes());manifest=read(E/'received_manifest.json')['manifest']
for name,item in manifest.items():raw=(E/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
s=read(E/'code/study.json');review=read(E/'code/actual_CPU_admission_review.json');execution=read(E/'code/image_execution.json');assert s['status']=='REGISTERED_SINGLE1729_ORDINARY_COMMON_RELATIVE_AP_FULL_LIFECYCLE_v1'
assert s['arms']==['joint_same','relative_only','AP_only','joint_mismatched_teacher'] and s['training_seed']==1729 and not s['multiple_seeds_allowed'] and s['R80_used_for_direction_selection']
assert review['actual_CPU_checks']==20 and len(review['checks'])==20 and review['CPU_CUDA_initialized'] is False and review['OLD_GPU_compute']==''
assert hashlib.sha256((E/'code/study.json').read_bytes()).hexdigest()==review['study']['sha256'] and execution['training_protocol']==review['study']
checks=[]
for p in (E/'code').glob('*.py'):
    ast.parse(p.read_text(encoding='utf8'));pin=next(v for v in s['source_pins'] if v['path'].endswith('/'+p.name) and '/HIVE_common_continuation_code_' in v['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==pin['sha256'];checks.append(p.name)
train=(E/'code/train_HIVE_common_continuation_GPU_20261030_v1.py').read_text();image=(E/'code/run_HIVE_common_continuation_images_20261030_v1.py').read_text();audit=(E/'code/audit_HIVE_common_continuation_CPU_20261030_v1.py').read_text();queue=(E/'code/queue_HIVE_common_continuation_CPU_20261030_v1.py').read_text()
assert 'selected_strength' not in train+image and 'AdamW(' not in train
assert "initial_checkpoint=s['ordinary']" in train and "model.radius=.25" in image and "selected_parameter_fraction" in audit and "MANUAL_NORMALIZED_SGD" in audit
assert "env['CUDA_VISIBLE_DEVICES']='0'" in queue and 'EXACT_ORIGINAL_ORDINARY_PARAMETER_IDENTITY_CANNOT_ADD_GAIN' in queue
assert s['passes']==3 and s['batch']==16 and s['last_batch']==12 and s['per_arm_additional_updates']==24807 and s['per_arm_additional_exposures']==396900
assert s['max_new_images']==1600 and s['finite_GPU_seconds']==900 and s['finite_main_seconds']==21600 and s['output_cap_bytes']==2<<30
assert 'no240scores' in s['image_policy'] and not read(E/'actual_observation.json')['new_image_effect_values_read']
r=dict(status='PASS_LOCAL_CONCRETE_COMMON_LIFECYCLE_SOURCE_PINS_SGD_PARAMETER_CALIBRATION_AND_FIXED_POLICY',files=len(manifest),new_source_files_checked=checks,actual_server_CPU_checks=20,
    remaining_legacy_source_pins_verified_on_server_not_all_transferred_locally=True,server_real_input_reviews_not_claimed_local_reexecution=True,fit_not_started_at_review=read(E/'actual_observation.json')['main'] is None,
    no_full_fit_image_or_SOTA_claim=True)
(E/'independent_local_registration_review.json').write_text(json.dumps(r,indent=2)+'\n',encoding='utf8');print(json.dumps(r))
