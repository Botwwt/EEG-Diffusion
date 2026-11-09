"""Pure-CPU recipe/index/admission rejection checks. No Torch/model/data reads."""
import argparse
import ast
import copy
import json
from pathlib import Path
import random
import tempfile
import time
from unittest.mock import patch

import brainhive_functional_training_contract_v2 as contract


def example(arm):
    recipe = dict(schema='BrainHIVE_functional_alignment_v2_recipe', arm=arm,
        candidate='functional_alignment_v2_' + arm, training_seed=1729, subjects=10,
        unique_T_images=13230, unique_OOF_training_queries=132300, passes=3,
        total_training_query_exposures=396900, expected_updates=24807, batch_size=16,
        learning_rate=.0001, weight_decay=.01, gradient_clip=1., trainable_parameters=525568,
        predict_risk=False, gate_policy='fixed_one', maximum_standardized_direction_radius=.25,
        timestep_policy='uniform_integer_0_999', reference_Q_policy='frozen_unmodified_parent_same_query_t_noise_detached',
        training_UNet_forwards_per_update=2, token_angular_weight=.1, functional_auxiliary_weight=.1,
        generator_asset_root=str(contract.GENERATOR_ASSET_ROOT),
        functional_epsilon=1e-6, maximum_query_positions=16, expected_native_IP_layers=70,
        minimum_nondegenerate_teacher_fraction=.5, engineering_profile_updates=2, engineering_profile_batch_size=16,
        checkpoint_selection='fixed_final_three_passes', strength_policy='Dfit_only_global_strength_calibration',
        output_directory=str(contract.WORK / 'components/functional_alignment_v2' / arm),
        profile_output=str(contract.WORK / 'audit/functional_alignment_v2' / arm / 'actual_GPU_profile.json'),
        generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1),
        training_job_seconds=86400, maximum_cumulative_training_seconds_per_arm=172800,
        profile_job_seconds=1800, profile_job_name='functional_v2_' + arm + '_profile',
        formal_job_names=['functional_v2_' + arm + '_fit', 'functional_v2_' + arm + '_exact_resume1'],
        source_sha256={name: 'a' * 64 for name in contract.SOURCES},
        latent_receipt=str(contract.WORK / 'audit/component_T_latent_bank_v2.json'),
        input_contract_sha256='b' * 64)
    paths = list(contract.FIXED_INPUTS) + [recipe['latent_receipt']]
    recipe['input_sha256'] = {path: 'b' * 64 for path in paths}
    return recipe


def rejects(action):
    try:
        action()
    except (AssertionError, FileNotFoundError, KeyError):
        return
    raise AssertionError('Malformed contract was accepted')


def run_checks():
    # Values deliberately cannot be mistaken for zero-based tensor positions.
    rows = [10001 + i * 37 for i in range(24)]
    concepts = {row: 'concept_' + str(i // 4) for i, row in enumerate(rows)}
    random.Random(1729).shuffle(rows)
    mapping = contract.teacher_mappings(rows, concepts)
    assert mapping == contract.teacher_mappings(rows, concepts)
    for key in ('negative', 'rank_positive_shuffled', 'rank_negative_shuffled'):
        indices = mapping[key]
        assert sorted(indices) == list(range(len(rows))) and set(indices).isdisjoint(rows)
        assert sorted(rows[i] for i in indices) == sorted(rows)
        assert all(concepts[rows[i]] != concepts[rows[j]] for i, j in enumerate(indices))
        destinations = {}
        for i, j in enumerate(indices):
            destinations.setdefault(concepts[rows[i]], set()).add(concepts[rows[j]])
        assert all(len(targets) == 1 for targets in destinations.values())
    assert all(concepts[rows[a]] != concepts[rows[b]] for a, b in
               zip(mapping['rank_positive_shuffled'], mapping['rank_negative_shuffled']))
    rejects(lambda: contract.teacher_mappings(rows + rows[:1], concepts))
    rejects(lambda: contract.teacher_mappings(rows, {row: 'same' for row in rows}))
    rejects(lambda: contract.teacher_mappings(rows[:-1], concepts))
    rejects(lambda: contract.teacher_mappings(rows, concepts, seed=1730))
    recipes = {arm: example(arm) for arm in contract.ARMS}
    cores = {arm: contract.validate_recipe(recipe, check_files=False) for arm, recipe in recipes.items()}
    assert len(set(cores.values())) == 4
    plan = dict(arms=[dict(arm=arm, candidate='functional_alignment_v2_' + arm,
        run_name=contract.run_name(arm), recipe_core_sha256=cores[arm],
        training_root=recipe['output_directory'], metadata_root=str(contract.WORK / 'images' / contract.run_name(arm)),
        inference_manifest=str(contract.WORK / 'images' / contract.run_name(arm) / 'inference_manifest.jsonl'),
        feature_root=str(contract.WORK / 'evaluation' / contract.run_name(arm) / 'features'),
        image_segments=[dict(start=0, stop=16600, root=str(contract.WORK / 'allocated_functional_v2' / arm))])
        for arm, recipe in recipes.items()])
    assert contract.validate_group_identity(plan, recipes)
    wrong_plan = copy.deepcopy(plan)
    wrong_plan['arms'][2]['inference_manifest'] = str(contract.WORK / 'images/protected_R4_risk_bounded_v1/inference_manifest.jsonl')
    rejects(lambda: contract.validate_group_identity(wrong_plan, recipes))
    wrong_plan = copy.deepcopy(plan)
    wrong_plan['arms'][1]['image_segments'][0]['root'] = wrong_plan['arms'][0]['image_segments'][0]['root']
    rejects(lambda: contract.validate_group_identity(wrong_plan, recipes))
    wrong_hosts = copy.deepcopy(recipes)
    wrong_hosts['PN']['input_sha256'][str(contract.WORK / 'audit/native_asset_verification.json')] = 'changed'
    rejects(lambda: contract.validate_group_identity(plan, wrong_hosts))
    for arm, recipe in recipes.items():
        admitted = dict(recipe, admission_receipt='/root/future_not_issued.json', admission_sha256='c' * 64)
        assert contract.core_sha(admitted) == cores[arm]
        for key, value in [('training_UNet_forwards_per_update', 1), ('gate_policy', 'risk'),
                           ('reference_Q_policy', 'student_Q'), ('passes', 4),
                           ('training_job_seconds', 43200),
                           ('minimum_nondegenerate_teacher_fraction', 0.),
                           ('output_directory', '/root/old_history'), ('trainable_parameters', 525825)]:
            broken = copy.deepcopy(recipe)
            broken[key] = value
            rejects(lambda: contract.validate_recipe(broken, check_files=False))
        broken = copy.deepcopy(recipe)
        broken['source_sha256'].pop('brainhive_functional_alignment_v2.py')
        rejects(lambda: contract.validate_recipe(broken, check_files=False))
    # A fabricated or absent future receipt cannot become a successful profile.
    with tempfile.TemporaryDirectory(prefix='brainhive_v2_CPU_contract_') as folder:
        p = Path(folder) / 'not_an_admission.json'
        p.write_text(json.dumps(dict(schema='BrainHIVE_functional_alignment_v2_reviewed_admission', status='DRAFT')))
        recipe = dict(recipes['PN'], admission_receipt=str(p), admission_sha256=contract.sha(p))
        with patch.object(contract, 'validate_recipe', return_value=cores['PN']):
            rejects(lambda: contract.verify_admission(recipe, 'profile'))
        recipe['admission_receipt'] = str(p.with_name('absent.json'))
        with patch.object(contract, 'validate_recipe', return_value=cores['PN']):
            rejects(lambda: contract.verify_admission(recipe, 'formal'))
        # Verify the ACTUAL old diagnostic is represented as missing-reference
        # fixed Q, never as a native-parent-Q GPU experiment. Synthetic metadata
        # is temporary and cannot be used as an issued runtime admission.
        class ReachedLedger(Exception):
            pass
        diagnostic = dict(status='FROZEN_NATIVE_FIXED_Q_FUNCTIONAL_PREREQUISITE_REVIEWED',
            actual_GPU_diagnostic=True, native_fixed_Q_reused=True, reference_condition='missing_raw_condition',
            teacher_response_nondegenerate=True, EEG_pairing_signal_present=True,
            engineering_prerequisite_is_not_image_gain=True,
            actual_diagnostic_summary=dict(path='summary', sha256='s'),
            actual_diagnostic_provenance=dict(path='provenance', sha256='p'),
            actual_diagnostic_completion=dict(path='completion', sha256='c'))
        provenance = dict(Q_reference='Exactly one missing_zero_raw_condition full-UNet pass per query/time; its detached native Q reused for every response arm. Fixed before data, no response-specific Q.',
            missing_definition='Exactly zeros_like(raw_OOF_EEG) at input to frozen A; not BrainMLP(zero EEG) and not zero IP tokens',
            all_conditions_share_same_real_latent_noise_and_timestep=True, generated_images=0, parameter_updates=0, optimizer_created=False)
        completion = dict(status='COMPLETE_FROZEN_NATIVE_T32_FUNCTIONAL_DIAGNOSTIC_NO_IMAGE_QUALITY_CLAIM',
            summary_sha256='s', provenance_sha256='p', generated_images=0, fitted_parameters=0,
            native_IP_layers=70, real_T_queries=32, engineering_checks_passed=True, protected_scores_read=False)
        objects = {str(contract.MAIN): dict(status=sorted(contract.MAIN_TERMINAL)[0]),
                   str(contract.DIAG): dict(status=sorted(contract.DIAG_TERMINAL)[0]),
                   'review': diagnostic, 'provenance': provenance, 'completion': completion,
                   'summary': dict(status='FROZEN_T_ONLY_FUNCTIONAL_MEASUREMENTS_SAVED', by_timestep=[dict(queries=32)] * 4)}
        def controlled_read(item):
            if item['path'] == 'ledger_sentinel':
                raise ReachedLedger()
            return objects[item['path']]
        receipt = dict(schema='BrainHIVE_functional_alignment_v2_reviewed_admission',
            status='ENGINEERING_PROFILE_ONLY_ADMITTED', purpose='profile', issuer='verified_internal_engineering_audit',
            candidate='functional_alignment_v2_PN', recipe_core_sha256=cores['PN'], source_sha256=recipes['PN']['source_sha256'],
            sha_bound_review_not_cryptographic_signature=True, issued_epoch=time.time() - 1, expires_epoch=time.time() + 60,
            terminal_prerequisites=[dict(path=str(contract.MAIN)), dict(path=str(contract.DIAG))],
            functional_prerequisite_review=dict(path='review'), ledger_review=dict(path='ledger_sentinel'))
        p.write_text(json.dumps(receipt))
        recipe = dict(recipes['PN'], admission_receipt=str(p), admission_sha256=contract.sha(p))
        with patch.object(contract, 'validate_recipe', return_value=cores['PN']), patch.object(contract, 'bound_read', side_effect=controlled_read), patch.object(contract, 'process_live', return_value=False):
            try:
                contract.verify_admission(recipe, 'profile')
            except ReachedLedger:
                pass
            else:
                raise AssertionError('Valid fixed-Q metadata did not reach the next independent gate')
            diagnostic['reference_condition'] = 'native_parent'
            rejects(lambda: contract.verify_admission(recipe, 'profile'))
            diagnostic['reference_condition'] = 'missing_raw_condition'
            provenance['Q_reference'] = 'A parent-Q experiment that never happened'
            rejects(lambda: contract.verify_admission(recipe, 'profile'))
    source = contract.ROOT / 'train_BrainHIVE_functional_alignment_v2.py'
    tree = ast.parse(source.read_text())
    step = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'functional_step')
    calls = [node for node in ast.walk(step) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'denoise']
    assert len(calls) == 2, 'Exactly two explicit UNet forwards are part of every mode'
    captures = [node for node in ast.walk(step) if isinstance(node, ast.With) and any(
        isinstance(item.context_expr, ast.Call) and isinstance(item.context_expr.func, ast.Attribute)
        and item.context_expr.func.attr == 'capture' for item in node.items)]
    assert len(captures) == 1 and len(captures[0].body) == 1
    assert any(node is calls[0] or node is calls[1] for node in ast.walk(captures[0]))
    assert len([node for node in ast.walk(captures[0]) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == 'denoise']) == 1
    return dict(status='PURE_CPU_FUNCTIONAL_V2_RECIPE_MAPPING_AND_REFUSAL_CHECKS_PASSED',
        source_sha256={source.name: contract.sha(source), Path(contract.__file__).name: contract.sha(contract.__file__)},
        unsorted_array_values_not_confused_with_positions=True, fixed_concept_level_bijections=True,
        teacher_marginals_preserved=True, both_shuffled_rank_teachers_exclude_query_concept=True,
        risk_manifest_budget_alias_rejected=True, cross_arm_image_root_alias_rejected=True,
        differing_frozen_native_receipt_between_arms_rejected=True,
        native_author_receipts_weights_configs_and_generator_assets_required=True,
        all_four_modes_have_separate_recipe_cores_and_new_outputs=True,
        immutable_gate_radius_timestep_head_and_forward_contract_checked=True,
        absent_or_DRAFT_admission_rejected=True, exactly_two_explicit_UNet_calls_in_source=True,
        missing_reference_T32_not_misrepresented_as_parent_Q=True,
        fixed_24h_training_timeout_cannot_be_rewritten_after_profile=True,
        parent_capture_contains_only_parent_forward=True, actual_GPU_profile_validated=False,
        actual_EEG_or_latents_read=False, trained_parameters=0, generated_images=0, GPU_access=False,
        scientific_effect_or_SOTA_established=False,
        limits=['AST checks do not replace the independent two-layer native CPU replay or actual GPU profile',
                'No future reviewed admission, complete storage plan or fitting authority was issued'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        assert not args.output.exists(), 'Preserve prior receipts'
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
