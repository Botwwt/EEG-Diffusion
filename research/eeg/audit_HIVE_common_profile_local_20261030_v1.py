"""Review exact profile source pins, arithmetic/coverage and finite job closure."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
L=Path(__file__).resolve().parent;E=L/'audit/HIVE_common_descent_profile_terminal_20261030_v2';read=lambda p:json.loads(Path(p).read_bytes());manifest=read(E/'received_manifest.json')
for name,item in manifest['manifest'].items():
    raw=(E/name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
s=read(E/'actual/completion.json');protocol=read(E/'code/protocol.json');CPU=read(E/'code/CPU_review.json');wrapper=read(E/'actual/GPU_wrapper_receipt.json');controller=read(E/'code/controller_completion.json')
assert s['status']=='PASS_ACTUAL_FOUR_ARM_COMMON_DESCENT_FP32_GPU_ENGINEERING' and CPU['checks_passed']==11 and len(set(CPU['checks']))==11
for item in protocol['source_pins']:
    raw=(E/'code'/Path(item['path']).name).read_bytes();assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256']
assert wrapper['exit_code']==0 and wrapper['status']=='COMPLETED' and wrapper['max_seconds']==300 and controller['followon_queue'] is None
assert not manifest['observation']['main']['exact_live'] and not manifest['observation']['child']['exact_live']
assert s['temporary_updates']==128 and s['scientific_updates']==0 and s['checkpoints_saved']==0 and s['new_research_images']==0 and s['generator_UNet_forwards']==0
maximum=0.;counts={};timings={}
for arm,a in s['arms'].items():
    assert sorted(x['role'] for x in a['all20_role_gradients'])==list(range(20));assert len(a['temporary_step_records'])==a['temporary_steps']==32
    cells=a['all20_role_gradients']+a['temporary_step_records'];counts[arm]=dict(cells=52,both_predicted_derivatives_negative=sum(x['relative_derivative']<0 and x['AP_derivative']<0 for x in cells))
    times=[x['step_seconds'] for x in a['temporary_step_records']];median=float(np.median(times));p95=float(np.quantile(times,.95));estimate=p95*24807
    maximum=max(maximum,abs(median-a['median_step_seconds']),abs(p95-a['p95_step_seconds']),abs(estimate-a['estimated_24807_steps_p95_seconds']));assert maximum<1e-10
    assert a['original_and_final_fraction_state_exact'] and a['temporary_weights_discarded'];timings[arm]=dict(median=median,p95=p95,extrapolated_steps_only_seconds=estimate)
assert set(s['arms'])==set(protocol['future_candidate_DRAFT']['arms']) and sum(a['temporary_steps'] for a in s['arms'].values())==128
assert abs(s['fixed_parameter_step']-s['initial_parameter_norm']*1e-4)<1e-15
assert all(x['relative_error']<2e-3 and x['scaled_error']<2e-5 for x in s['CPU64_to_actual_GPUFP32'].values())
result=dict(status='PASS_LOCAL_PROFILE_SOURCE_HASHES_80_ROLE_CELLS_128_DISCARDED_STEPS_AND_TIMING_ARITHMETIC',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=len(manifest['manifest']),CPU_checks=11,
    gradient_cells=80,temporary_steps=128,counts=counts,timings=timings,maximum_arithmetic_error=maximum,wrapper_seconds=wrapper['charged_seconds'],controller_seconds=controller['seconds'],GPU_program_seconds=s['seconds'],
    packed_GPU_gradients_not_independently_replayed_locally=True,initial_CPU_packed_gradients_verified_separately=True,NOT_full_training_or_image_results=True)
(E/'independent_local_profile_review.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result))
