"""Issue actual four-arm engineering-only recipes after inspecting frozen evidence.

This does not modify the frozen v2 trainer/contract or admit formal fitting.
The fixed reader's UNKNOWN label is retained. Pairing and nondegeneracy below
are narrower, explicitly quantified prerequisites for a two-update GPU profile.
"""
import hashlib
import json
import pathlib
import shutil
import subprocess
import time

import brainhive_functional_training_contract_v2 as contract

ROOT = pathlib.Path(__file__).resolve().parent
WORK = pathlib.Path(str(contract.WORK))
BASE = ROOT / 'audit/functional_profiles_preparation_20261022_v1'
CONFIG = ROOT / 'configs/functional_profiles_20261022_v1'
DRAFT = ROOT / 'configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json'
DRAFT_SHA = 'fb3b20f01ab92565dcc362a27e79daaa2d9fa145590c8cbc15624dcbf672064f'
EVIDENCE = ROOT / 'audit/BrainHIVE_native_T32_evidence_actual_20261022_v1.json'


def write(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def bound(path):
    return dict(path=str(path), sha256=contract.sha(path))


def recipe_for(arm, sources, inputs):
    return dict(schema='BrainHIVE_functional_alignment_v2_recipe', arm=arm,
        candidate='functional_alignment_v2_' + arm, training_seed=1729, subjects=10,
        unique_T_images=13230, unique_OOF_training_queries=132300, passes=3,
        total_training_query_exposures=396900, expected_updates=24807, batch_size=16,
        learning_rate=.0001, weight_decay=.01, gradient_clip=1., trainable_parameters=525568,
        predict_risk=False, gate_policy='fixed_one', maximum_standardized_direction_radius=.25,
        timestep_policy='uniform_integer_0_999', reference_Q_policy='frozen_unmodified_parent_same_query_t_noise_detached',
        training_UNet_forwards_per_update=2, token_angular_weight=.1, functional_auxiliary_weight=.1,
        generator_asset_root=str(contract.GENERATOR_ASSET_ROOT), functional_epsilon=1e-6,
        maximum_query_positions=16, expected_native_IP_layers=70,
        minimum_nondegenerate_teacher_fraction=.5, engineering_profile_updates=2,
        engineering_profile_batch_size=16, checkpoint_selection='fixed_final_three_passes',
        strength_policy='Dfit_only_global_strength_calibration',
        output_directory=str(WORK / 'components/functional_alignment_v2' / arm),
        profile_output=str(WORK / 'audit/functional_alignment_v2' / arm / 'actual_GPU_profile.json'),
        generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1),
        training_job_seconds=86400, maximum_cumulative_training_seconds_per_arm=172800,
        profile_job_seconds=1800, profile_job_name='BrainHIVE_functional_v2_' + arm + '_profile_20261022_v1',
        formal_job_names=['BrainHIVE_functional_v2_' + arm + '_fit_20261022_v1',
                          'BrainHIVE_functional_v2_' + arm + '_exact_resume1_20261022_v1'],
        source_sha256=sources, latent_receipt=str(WORK / 'audit/component_T_latent_bank_v2.json'),
        input_contract_sha256=inputs[str(WORK / 'data/component_input_contract_v1.json')],
        input_sha256=inputs)


def main():
    assert ROOT == pathlib.Path(str(contract.REPO))
    assert contract.sha(DRAFT) == DRAFT_SHA
    assert not BASE.exists() and not CONFIG.exists()
    draft = contract.read(DRAFT)
    assert draft['status'] == 'DRAFT_NOT_ADMITTED_NOT_EXECUTABLE'
    assert not draft['automatic_launch_enabled']
    for name, digest in (draft['source_sha256'] | draft['additional_engineering_source_sha256']).items():
        assert contract.sha(ROOT / name) == digest, name
    evidence = contract.read(EVIDENCE)
    assert evidence['numerical_integrity'] == 'PASS_COMPLETE32_BY4_NATIVE70_SHA_AND_IDENTITY'
    e = evidence['evidence']
    assert e['status'] == 'UNKNOWN' and e['automatic_admission'] is False
    assert e['teacher_valid_fraction_across_all_query_time_layer_entries'] == 1.
    # This only establishes paired EEG distinctions, not all four directional
    # criteria, bottleneck diagnosis, statistical significance, or image benefit.
    assert all(row['EEG_J_minus_permuted_EEG_J'] > 0 and
               row['permuted_d_positive_minus_EEG_d_positive'] > 0 and
               row['wrong_F_epsilon_MSE_minus_correct_F'] > 0
               for row in e['all_four_timestep_details'])
    assert e['aggregate']['missing_d_positive_minus_EEG_d_positive'] < 0
    assert e['parent_Q_v2_training_validated'] is False
    main_state = contract.read(str(contract.MAIN))
    diag_state = contract.read(str(contract.DIAG))
    assert main_state['status'] == 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
    assert diag_state['status'] == 'FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING'
    assert not contract.process_live(main_state['pid']) and not contract.process_live(diag_state['pid'])
    pixel = contract.read(WORK / 'audit/functional_T_pixel_replay_v2/summary.json')
    assert pixel['status'] == 'FUNCTIONAL_T_PIXEL_REPLAY_V2_EXACT_THREE_PATHS_PASSED'
    assert all(row['pixel_byteexact'] for row in pixel['comparisons'].values())
    assert not subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                              check=True, capture_output=True, text=True).stdout.strip()
    assert contract.resource_headroom()['effective_headroom_bytes'] >= 8 << 30
    assert shutil.disk_usage(WORK).free >= 128 << 20
    before = contract.sha(ROOT / 'experiment_ledger.jsonl')
    BASE.mkdir(); CONFIG.mkdir()
    snapshot = BASE / 'experiment_ledger_before_profiles.jsonl'
    snapshot.write_bytes((ROOT / 'experiment_ledger.jsonl').read_bytes())
    accounting = contract.read(ROOT / 'audit/BrainHIVE_candidate_accounting_reconciliation_CPU_20261021_v1.json')
    assert accounting['counts']['conservative_all_history_commitments_before_v2'] == 11
    ledger = BASE / 'candidate_review.json'
    write(ledger, dict(status='INHERITED_LIMITS_RECONCILED_FOUR_ARM_PLAN_RECORDED',
        candidate_limit=12, structural_revision_limit=2, candidate_limit_not_reset=True,
        structural_revision_index=1, used_before=11, used_after=12,
        arms=[dict(arm=a, counts_as_new_candidate=a == 'PN') for a in contract.ARMS],
        ledger_snapshot=bound(snapshot), accounting=bound(ROOT / 'audit/BrainHIVE_candidate_accounting_reconciliation_CPU_20261021_v1.json'),
        scope='PN is the twelfth conservative commitment; profiles are disposable engineering updates, not formal fits.'))
    diag = WORK / 'audit/native_function_T32_v2'
    review = BASE / 'functional_prerequisite_review.json'
    write(review, dict(status='FROZEN_NATIVE_FIXED_Q_FUNCTIONAL_PREREQUISITE_REVIEWED',
        actual_GPU_diagnostic=True, native_fixed_Q_reused=True, reference_condition='missing_raw_condition',
        teacher_response_nondegenerate=True, EEG_pairing_signal_present=True,
        engineering_prerequisite_is_not_image_gain=True,
        actual_diagnostic_summary=bound(diag / 'summary.json'),
        actual_diagnostic_provenance=bound(diag / 'provenance.json'),
        actual_diagnostic_completion=bound(diag / 'completion.json'),
        fixed_reader_evidence=bound(EVIDENCE), fixed_reader_overall_label='UNKNOWN',
        exact_aggregate=e['aggregate'], per_time=e['all_four_timestep_details'],
        narrow_boolean_interpretation='EEG versus permuted distinguishes paired conditions in both J and positive MSE at all four fixed times. All teacher entries are nondegenerate. This does not convert overall UNKNOWN to directional support.',
        conflict_preserved='Missing condition has lower positive-teacher MSE than correct EEG, while EEG has better relative J. Both proximity and discrimination must be checked; no claim of complete response collapse.',
        formal_fitting_admitted=False, parent_Q_validated=False, perceptual_or_semantic_gain=False,
        decision='Permit only the already registered two-update engineering profiles; require actual parent-Q validity/gradient/throughput checks before separate resource-funded formal admission.'))
    latent = contract.read(WORK / 'audit/component_T_latent_bank_v2.json')
    names = set(contract.SOURCES) | set(draft['source_sha256'])
    sources = {name: contract.sha(ROOT / name) for name in sorted(names)}
    paths = set(contract.FIXED_INPUTS) | {str(WORK / 'audit/component_T_latent_bank_v2.json'), latent['path']}
    paths |= {str(p) for p in pathlib.Path(str(contract.GENERATOR_ASSET_ROOT / 'sdxl_config')).rglob('*') if p.is_file()}
    inputs = {path: contract.sha(path) for path in sorted(paths)}
    assert inputs[latent['path']] == latent['sha256']
    items = []
    for arm in contract.ARMS:
        recipe = recipe_for(arm, sources, inputs)
        core = contract.validate_recipe(recipe)
        assert not pathlib.Path(recipe['profile_output']).exists()
        admission = BASE / (arm + '_profile_admission.json')
        now = time.time()
        write(admission, dict(schema='BrainHIVE_functional_alignment_v2_reviewed_admission',
            status='ENGINEERING_PROFILE_ONLY_ADMITTED', purpose='profile',
            issuer='verified_internal_engineering_audit', candidate=recipe['candidate'],
            recipe_core_sha256=core, source_sha256=sources,
            sha_bound_review_not_cryptographic_signature=True, issued_epoch=now, expires_epoch=now + 86400,
            terminal_prerequisites=[bound(pathlib.Path(str(p))) for p in (contract.MAIN, contract.DIAG)],
            functional_prerequisite_review=bound(review), ledger_review=bound(ledger),
            resource_limits=dict(minimum_cgroup_headroom_bytes=8 << 30,
                minimum_free_GPU_bytes=16 << 30, profile_metadata_budget_bytes=1 << 20,
                minimum_retained_disk_buffer_bytes=64 << 20),
            original_DRAFT_unchanged=True, formal_fit_or_full_V_admitted=False))
        recipe.update(admission_receipt=str(admission), admission_sha256=contract.sha(admission))
        contract.verify_admission(recipe, 'profile')
        path = CONFIG / (arm + '.json'); write(path, recipe)
        items.append(dict(arm=arm, **bound(path), recipe_core_sha256=core,
                          profile_job_name=recipe['profile_job_name'], profile_output=recipe['profile_output']))
    assert contract.sha(DRAFT) == DRAFT_SHA and contract.sha(ROOT / 'experiment_ledger.jsonl') == before
    write(BASE / 'completion.json', dict(status='ALL_FOUR_ENGINEERING_ONLY_RECIPES_ACTUALLY_BOUND_AND_CHECKED',
        recipes=items, source=bound(pathlib.Path(__file__)), draft=bound(DRAFT),
        evidence=bound(EVIDENCE), new_GPU_jobs=0, formal_training_enabled=False,
        actual_parent_Q_profile_complete=False, full_lifecycle_storage_admitted=False,
        maximum_profile_jobs=4, maximum_seconds_per_job=1800,
        maximum_total_profile_seconds=7200, maximum_new_metadata_bytes=64 << 20))
    print(json.dumps(dict(status='PROFILE_RECIPES_READY_ONLY', recipes=items, formal_training_enabled=False)))


if __name__ == '__main__':
    main()
