"""Register exactly two requested ideas from real server sources and input SHAs.

Registration is not storage admission, profiling, training or an image result.
"""
import csv
import json
import time
from pathlib import Path
from brainhive_two_ideas_storage_v1 import ROOT,WORK,sha


def write_new(path,value):
    assert not path.exists(),str(path)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def main():
    files=['brainhive_native_models_v1.py','brainhive_bounded_component_v1.py',
        'brainhive_component_data_v1.py','brainhive_component_generator_v1.py',
        'brainhive_lifecycle_storage_v1.py','train_BrainHIVE_bounded_component_v1.py',
        'brainhive_brainkan_v1.py','train_BrainHIVE_BrainKAN_v1.py',
        'brainhive_response_rank_v1.py','train_BrainHIVE_response_rank_v1.py',
        'brainhive_two_ideas_storage_v1.py','generate_BrainHIVE_two_ideas_V_v1.py',
        'audit_BrainHIVE_two_ideas_image_arm_CPU_v1.py','compare_BrainHIVE_two_ideas_CPU_v1.py',
        'extract_image_features.py','score_full_manifest.py','score_low_level.py',
        'locked_metrics.py','semantic_evaluator.py','run_gpu_job_v2.py']
    sources={name:sha(ROOT/name) for name in files}
    basic=[WORK/'data/actual_input_bindings.json',ROOT/'configs/split_manifest.json',
        ROOT/'audit/image_manifest.csv',WORK/'data/train_visual_embeddings.pt',WORK/'audit/input_preparation_complete.json']
    response=[ROOT/'configs/split_manifest.json',ROOT/'audit/image_manifest.csv',
        WORK/'data/component_input_contract_v1.json',WORK/'audit/component_T_latent_bank_v2.json']
    common=dict(training_seed=1729,subjects=10,formal_V_queries_per_arm=16600,
        generation_inputs_sha256={str(WORK/'data/generation_inputs/V.json'):sha(WORK/'data/generation_inputs/V.json'),
            str(WORK/'data/generation_inputs/V_targets.jsonl'):sha(WORK/'data/generation_inputs/V_targets.jsonl')},
        generator=dict(resolution=512,steps=4,CFG=0,IP_scale=1,images_per_query=1),
        reserved_bytes_per_image=330000,feature_budget_bytes=220000000,metadata_budget_bytes=150000000,
        training_job_seconds=86400,engineering_profile_max_updates=2,
        source_sha256=sources,allowed_private_roots=[],image_segments=[],
        storage_admission_receipt=None,storage_admission_sha256=None,
        no_final_test_tuning=True,checkpoint_selection='fixed_final_epoch',
        normal_inference_target_access=False,new_raw_EEG_acquisition=False,
        same_query_noise_as_native=True,one_training_seed=True,
        inherited_candidate_cap=12,inherited_structural_revision_cap=2,
        caps_not_reset=True,additional_configurations_explicitly_requested=2,
        no_unregistered_hyperparameter_expansion=True,
        practical_gain_rule='DINO2AFC delta>=.01 with paired lowerCI>0, SEED delta>0 with lowerCI>0, ObjectF1 delta>=0; provisional exposed V only',
        image_quality_noninferiority_assumed=False,mechanism_confirmed=False,
        independent_confirmation=False,cross_architecture_claim=False)
    kan=dict(common,candidate='brainkan_full_v1',role='full',
        run_name='protected_R4_brainkan_two_ideas_v1',
        training_entry='train_BrainHIVE_BrainKAN_v1.py',output_directory=str(WORK/'runs/brainkan_full_v1'),
        encoder_output_directory=str(WORK/'runs/brainkan_full_v1'),
        input_sha256={str(p):sha(p) for p in basic},training_artifact_budget_bytes=1400000000,
        epochs=25,batch_size=1024,learning_rate=.0005,AdamW_betas=[.9,.995],weight_decay=.05,
        gradient_clip=1.,input_dim=4250,hidden_size=1024,spline_basis_count=8,spline_rank=60,
        trainable_encoder_parameters=6456512,native_BrainMLP_parameters=6452224,
        native_visual_training_dropout=.1,encoder_dropout=.1,
        training_updates_per_subject=325,train_images=13230,training_exposures_per_subject=330750,
        architecture='4250 raw physical linear path + low-rank additive cubic splines(rank60,bases8) -> native-eps LayerNorm1024',
        hypothesis='Matched-size physical-coordinate nonlinear encoder can preserve useful EEG structure with exact pre-LN attribution',
        statistic_fit_scope='Only exact T rows: coordinate median/MAD; native linear branch unchanged',
        matched_controls=['protected_R4_parent_bounded_v1'],
        interpretability_claim_limit='Exact17electrode x10time-block contribution only before LayerNorm; downstream latents and physiology are not additive or causal',
        required_contingent_validation=['Prespecified deletion/insertion vs random and low-importance, matched nativeMLP attribution comparator',
            'Same protocol actual-image EEG controls and independent frozen confirmation'],
        novelty_limit='KAN/EEG combination alone is not a first-use or conference-level innovation claim')
    rank=dict(common,candidate='response_rank_v1',run_name='protected_R4_response_rank_two_ideas_v1',
        training_entry='train_BrainHIVE_response_rank_v1.py',
        output_directory=str(WORK/'components/two_ideas_v1/response_rank_v1'),
        component_output_directory=str(WORK/'components/two_ideas_v1/response_rank_v1'),
        encoder_output_directory=str(WORK/'runs/full_source_dropout_v2'),
        input_sha256={str(p):sha(p) for p in response},training_artifact_budget_bytes=64000000,
        input_contract_sha256=sha(WORK/'data/component_input_contract_v1.json'),
        latent_receipt=str(WORK/'audit/component_T_latent_bank_v2.json'),
        unique_T_images=13230,unique_OOF_training_queries=132300,passes=3,
        batch_size=16,learning_rate=.0001,weight_decay=.01,gradient_clip=1.,expected_updates=24807,
        trainable_parameters=525568,maximum_standardized_direction_radius=.25,
        response_rank_weight=.1,token_angular_weight=.1,response_maximum_query_positions=16,
        response_expected_layers=70,response_rank_epsilon=1e-6,engineering_profile_batch_size=16,
        response_strength_policy='Dfit_only_global_strength_calibration',
        checkpoint_selection='fixed_final_three_passes',
        architecture='Same525568parameter bounded ordinary head; paired native-IP response discrimination training',
        hypothesis='High common token components may hide stimulus distinctions; constrain functional positive/negative responses in frozen native IP space',
        response_objective='epsilonMSE+.1tokenAngular+.1meanLayer softplus((Dpos-Dneg)/(stopgrad teacherSeparation+1e-6))',
        negative_policy='Fixed seed1729 bijection of T image positions, verified different concept; never D/V targets',
        response_Q_policy='Detach identical actual native Q for student and both teachers; <=16spread positions/layer; no native processor replacement',
        inference_extra_UNet_NFE=0,training_extra_full_UNet_NFE=0,
        matched_controls=['protected_R4_parent_bounded_v1','protected_R4_ordinary_bounded_v1'],
        near_neighbor_limit='Old A1 UNet-response MSE is negative evidence; paired different-concept IP discrimination is a new falsifiable change, not proven novel',
        required_contingent_validation=['Shuffled positive/negative teacher control at fixed trained branch',
            'Same protocol EEG interventions and independent frozen confirmation'])
    configs=ROOT/'configs'
    paths=[configs/'BrainHIVE_BrainKAN_two_ideas_template_v1.json',configs/'BrainHIVE_response_rank_two_ideas_template_v1.json']
    for path,recipe in zip(paths,[kan,rank]):write_new(path,recipe)
    sequence=configs/'BrainHIVE_two_ideas_sequence_v1.json'
    write_new(sequence,dict(version='two_fixed_requested_ideas_v1',
        user_request='Replace BrainMLP by KAN; select two checked promising hypotheses actually adapted to HIVE',
        registration_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        fixed_order=['brainkan_full_v1','response_rank_v1'],template_paths=[str(p) for p in paths],
        template_sha256={str(p):sha(p) for p in paths},maximum_wait_seconds=72*3600,
        source_sha256={name:sha(ROOT/name) for name in files+['queue_BrainHIVE_two_ideas_v1.py']},
        existing_main_queue_not_modified=True,exactly_two_new_candidate_configurations=True,
        storage_admission_complete=False,formal_new_training_started=False,
        historical_count_not_reconstructed_as_zero=True,no_automatic_further_search=True,
        contingent_mechanism_control_images_not_funded_by_this_main_comparison=True))
    receipt=dict(status='TWO_FIXED_CANDIDATES_REGISTERED_NOT_YET_ADMITTED_OR_TRAINED',
        sequence=str(sequence),sequence_sha256=sha(sequence),templates={str(p):sha(p) for p in paths},
        source_sha256=sources,history_preserved=True,local_large_files_created=False)
    write_new(ROOT/'audit/BrainHIVE_two_ideas_registration_v1.json',receipt)
    with (ROOT/'experiment_ledger.jsonl').open('a') as log:
        log.write(json.dumps(dict(event_id='BrainHIVE_two_requested_ideas_registered_v1',type='immutable_fixed_registration',
            receipt=str(ROOT/'audit/BrainHIVE_two_ideas_registration_v1.json'),new_candidate_configurations=2,
            new_GPU_jobs=0,formal_new_training_started=False,component_image_gain_established=False))+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
