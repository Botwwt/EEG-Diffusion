"""Run the sealed image-preparation body with the verified CPU SDPA math backend."""
import json,os,runpy,socket,time
assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
from PKU_backbone_core_20261110_v322 import *
assert not torch.cuda.is_available();start=time.monotonic();old=R/'code/PKU_prepare_backbone_images_CPU_20261110_v322.py';pins=json.loads((R/'code/EEG_backbone_pins_v322.json').read_text());assert sha(old)==pins[old.name]
diag=json.loads((R/'runs/EEG_backbone_attention_diagnostic_v328/completion.json').read_text());assert diag['cache_all']['nonfinite']==diag['input_Dcheck']['nonfinite']==0 and diag['model_parameters_finite']
for v in diag['cases']:
    assert v['default']['nonfinite']>0 and v['mathmode']['nonfinite']==v['explicit']['nonfinite']==0 and v['explicit_vs_GPU_cache_max_scaled']<2e-4 and v['explicit_vs_math_max_scaled']<2e-4
paths=[ROOT/case/'encoders'/role/'brain_final.pt' for case in CASES for role in ROLES]+[ROOT/case/'calibration/component_final.pt' for case in CASES];before={str(p.relative_to(R)):sha(p) for p in paths}
torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
runpy.run_path(str(old),run_name='__main__')
out=R/'runs/EEG_backbone_images_v322';state=torch.load(out/'condition_bank.pt',map_location='cpu',weights_only=True);assert state['condition'].shape==(10,830,1024) and torch.isfinite(state['condition']).all()
after={str(p.relative_to(R)):sha(p) for p in paths};assert before==after
proof=json.loads((out/'conditions_completion.json').read_text());assert proof['independent_CPU_encoder_max_scaled_error']<2e-4
write(out/'CPU_backend_recovery_v329.json',dict(status='PASS_ALL8300_CONDITIONS_WITH_MATH_CPU_SDPA_NO_RETRAINING',CPU_job=os.environ['SLURM_JOB_ID'],diagnosis_sha256=sha(R/'runs/EEG_backbone_attention_diagnostic_v328/completion.json'),original_preparation_sha256=sha(old),wrapper_sha256=sha(__file__),condition_bank_sha256=sha(out/'condition_bank.pt'),original_tolerance=2e-4,CPU_vs_GPU_max_scaled=proof['independent_CPU_encoder_max_scaled_error'],trained_20_checkpoint_hashes_before_and_after=before,new_training_updates=0,new_images=0,numerical_execution_backend_changed='CPU SDPA default to math only; same sealed formulas and trained parameters',seconds=time.monotonic()-start));print('PASS_ALL8300_CONDITIONS_MATH_CPU_BACKEND',flush=True)
