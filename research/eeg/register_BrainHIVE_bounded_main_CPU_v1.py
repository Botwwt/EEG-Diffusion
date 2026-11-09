"""Register one hypothesis and reserve three complete matched main lifecycles."""
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import time

from brainhive_native_models_v1 import ROOT,WORK,sha,write


def main():
    recipe_path=ROOT/'configs/BrainHIVE_bounded_main_recipe_v1.json'
    storage_path=ROOT/'audit/BrainHIVE_bounded_three_main_lifecycle_reservation_v1.json'
    assert not recipe_path.exists() and not storage_path.exists()
    native=json.loads((ROOT/'audit/BrainHIVE_native_R80_full_closure_20261020_v10.json').read_text())
    assert native['status']=='ALL6000_NATIVE_IMAGES_COMMON_AND_AUTHOR_EIGHT_QUERY_RESULTS_CLOSED_CPU'
    profile=json.loads((WORK/'engineering/bounded_component_profile_v2_batch16/completion.json').read_text())
    assert profile['status']=='ENGINEERING_IMAGE_GRADIENT_AND_FIXED_T_IMAGE_PATH_COMPLETE'
    means={}
    for arm in ['correct','permuted','missing']:
        manifest=WORK/'images'/('native_R80_'+arm+'_v1')/'manifest.jsonl'
        rows=[json.loads(x) for x in manifest.read_text().splitlines()];assert len(rows)==2000
        means[arm]=statistics.mean(Path(r['generated_path']).stat().st_size for r in rows)
    per_image=math.ceil(max(means.values())/10000)*10000
    assert per_image==330000
    roots=[WORK/'allocated_images_v1',Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/images/BrainHIVE_bounded_20261020_v1')]
    for root in roots:root.mkdir(parents=True,exist_ok=True)
    assert os.stat(roots[0]).st_dev!=os.stat(roots[1]).st_dev,'Do not double-count one filesystem'
    free=[shutil.disk_usage(r).free for r in roots]
    reserve=[(512<<20)+650000000+300000000,(512<<20)]
    remaining=[a-b for a,b in zip(free,reserve)]
    total_images=3*16600;required=total_images*per_image+sum(reserve)
    funded=sum(free)>=required
    storage=dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),all_three_main_lifecycles_reserved=funded,
        formal_queries_per_arm=16600,main_image_arms=['risk','ordinary','parent'],total_saved_images_planned=49800,
        actual_native_mean_bytes=means,reserved_bytes_per_image=per_image,
        estimate_method='Worst complete native arm mean rounded up to10000 bytes, plus explicit evaluator/metadata budgets and512MiB free buffer per volume',
        required_persistent_bytes=required,actual_free_by_volume=free,non_image_reservation_by_volume=reserve,
        allowed_private_roots=[str(r) for r in roots],C_D_large_files=False,shared_or_public_mounts_used=False,
        initial_V_gallery_bytes=sum(p.stat().st_size for p in (WORK/'gallery/V').rglob('*') if p.is_file() and not p.is_symlink()) if (WORK/'gallery/V').exists() else 0,
        forecast_is_not_a_guarantee_of_future_file_sizes=True,stop_before_private_buffer_exhaustion=True,
        new_parameter_checkpoint_bytes_included_in_metadata_reservation=True,
        contingent_mechanism_controls_not_yet_reserved=['same-trained fixed-mean gate','shuffled-risk control','R4 EEG interventions'],
        main_comparison_alone_cannot_complete_risk_mechanism_or_SOTA_goal=True)
    if not funded:
        write(storage_path,storage);print(json.dumps(storage),flush=True);raise RuntimeError('Main lifecycle storage is not funded; no formal candidate may start')
    segments={}
    for arm in ['risk','ordinary','parent']:
        parts=[]
        for start in range(0,16600,16):
            stop=min(start+16,16600);need=(stop-start)*per_image
            volume=max(range(2),key=lambda i:remaining[i]);assert remaining[volume]>=need
            parts.append(dict(start=start,stop=stop,root=str(roots[volume])));remaining[volume]-=need
        segments[arm]=parts
    storage.update(segments=segments,unallocated_bytes_after_all_reservations=remaining)
    write(storage_path,storage)
    source_names=['brainhive_native_models_v1.py','brainhive_bounded_component_v1.py','brainhive_component_generator_v1.py',
        'brainhive_component_data_v1.py','brainhive_lifecycle_storage_v1.py','train_BrainHIVE_bounded_component_v1.py','generate_BrainHIVE_bounded_V_v1.py']
    count=dict(ordinary=profile['profile']['ordinary']['parameters'],risk=profile['profile']['risk']['parameters'])
    assert count==dict(ordinary=525568,risk=525825)
    estimate={arm:profile['profile'][arm]['steady_seconds_per_query']*396900 for arm in ['ordinary','risk']}
    recipe=dict(version='One_BrainHIVE_bounded_direction_and_calibrated_risk_v1',training_seed=1729,
        fixed_arm_order=['risk','ordinary','parent'],new_main_candidate_configurations=1,
        inherited_candidate_cap=12,inherited_structural_revision_cap=2,candidate_cap_not_reset=True,
        subjects=10,unique_T_images=13230,unique_OOF_training_queries=132300,passes=3,
        total_training_query_exposures_per_trainable_arm=396900,expected_updates_per_trainable_arm=24807,
        batch_size=16,learning_rate=.0001,weight_decay=.01,optimizer='AdamW default betas/eps, constant LR',gradient_clip=1.,
        maximum_standardized_direction_radius=.25,trainable_parameters=count,
        host='Exact frozen BrainHIVE full-T parent + native pretrained A/P/IP + pinned SDXL-Turbo',
        image_objective='Actual frozen native Turbo epsilon MSE over uniform training timesteps0..999',
        auxiliary_objective='0.1 token angular error; risk arm additionally0.1 prediction risk MSE',
        condition_drop_probability=0.,new_external_training_images=0,additional_raw_EEG_features=0,
        ordinary_global_strength_calibration='Only Dfit token error, fixed grid[0,.25,.5,.75,1], choose smaller strength on ties',
        risk_calibration='Only Dfit, nonnegative affine slope/intercept fitted to already-audited base token angular error; clip[0,2], gate=1-risk/2',
        Dcheck_policy='Single post-training diagnostic of risk calibration; no checkpoint, model or strength revision',
        normal_inference_inputs='EEG-derived raw condition and the host output affine only; no query target, class, caption or gallery',
        checkpoint_selection='Fixed final3 passes, no V or official benchmark score selection',
        generator=dict(resolution=512,steps=4,CFG=0,IP_scale=1,images_per_query=1),
        run_names={arm:'protected_R4_'+arm+'_bounded_v1' for arm in ['parent','ordinary','risk']},
        formal_V_queries_per_arm=16600,protected_metrics_unchanged=True,
        storage_lifecycle_receipt=str(storage_path),storage_receipt_sha256=sha(storage_path),
        input_contract_sha256=sha(WORK/'data/component_input_contract_v1.json'),
        latent_receipt=str(WORK/'audit/component_T_latent_bank_v2.json'),
        source_sha256={n:sha(ROOT/n) for n in source_names},
        queue_source_sha256=sha(ROOT/'queue_BrainHIVE_bounded_full_lifecycle_v1.py'),
        comparison_source_sha256=sha(ROOT/'compare_BrainHIVE_bounded_V_CPU_v1.py'),
        training_job_seconds=43200,maximum_exact_resume_attempts_per_arm=1,total_training_seconds_cap_per_arm=50400,
        actual_training_seconds_linear_estimate=estimate,expected_main_phase_GPU_hours_range=[30,36],
        generation_job_seconds=14400,evaluator_job_seconds=14400,
        no_new_hyperparameter_search=True,one_training_seed=True,independent_H=False,
        risk_mechanism_not_confirmed_by_main_three_arm_result=True,quality_noninferiority_not_assumed=True,
        positive_signal_requires='DINO >=+0.01 and lower pairedCI>0 against both references, SEED same direction and lowerCI>0, ObjectF1 nonnegative; still provisional until registered mechanism/quality controls',
        KAN_and_stage_routing='Deferred; no second parent architecture in this phase')
    write(recipe_path,recipe)
    print(json.dumps(dict(status='ONE_CANDIDATE_AND_ALL_THREE_MAIN_LIFECYCLES_REGISTERED',
        required_bytes=required,free_bytes=sum(free),remaining=remaining,estimated_training_hours={k:v/3600 for k,v in estimate.items()},
        recipe_path=str(recipe_path),recipe_sha256=sha(recipe_path))),flush=True)


if __name__=='__main__':main()
