"""CPU metadata/AST checks only; no image, model, Torch or GPU execution."""
import argparse
import ast
import copy
import json
from pathlib import Path

from brainhive_functional_training_contract_v2 import WORK, ARMS, sha, run_name, query_identity, core_sha
from check_BrainHIVE_functional_training_contract_CPU_v2 import example, rejects
from generate_BrainHIVE_functional_alignment_V_v2 import generation_contract, validate_blind_inputs, row_identity


def run_checks():
    for arm in ARMS:
        recipe = example(arm)
        recipe['source_sha256']['generate_BrainHIVE_functional_alignment_V_v2.py'] = 'a' * 64
        recipe.update(generation_job_names=['functional_v2_' + arm + '_images', 'functional_v2_' + arm + '_images_resume1'],
            generation_job_seconds=14400, maximum_cumulative_generation_seconds=28800,
            generation_gallery_manifest=str(WORK / 'gallery/V/manifest.jsonl'))
        recipe['input_sha256'][recipe['generation_gallery_manifest']] = 'b' * 64
        generation_contract(recipe, check_files=False)
        query = dict(stimulus_id='test_object_03.jpg', array_row=15001)
        identity, image_seed = query_identity('sub-03', query, 'test_protocol')
        row = dict(query_id=identity, image_seed=image_seed, subject='sub-03', global_query_index=99,
            stimulus_id=query['stimulus_id'], EEG_source_stimulus_id=query['stimulus_id'], array_row=query['array_row'],
            repeat_protocol='R4', candidate=recipe['candidate'], arm=arm, recipe_core_sha256=core_sha(recipe),
            run_name=run_name(arm), target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False)
        row_identity(row, 'sub-03', query, 99, recipe, 'test_protocol')
        for key, value in [('array_row', 3), ('image_seed', image_seed + 1), ('global_query_index', 100),
                           ('EEG_source_stimulus_id', 'wrong.jpg'), ('candidate', 'risk'), ('run_name', 'protected_R4_risk_bounded_v1')]:
            rejects(lambda: row_identity(dict(row, **{key: value}), 'sub-03', query, 99, recipe, 'test_protocol'))
        broken = copy.deepcopy(recipe)
        broken['source_sha256'].pop('generate_BrainHIVE_functional_alignment_V_v2.py')
        rejects(lambda: generation_contract(broken, check_files=False))
    queries = [dict(stimulus_id='stimulus_' + str(i) + '.jpg', array_row=10000 + i) for i in range(1660)]
    inputs = dict(expected_queries=16600, subjects=['sub-' + str(i) for i in range(10)],
        repeat_protocol='R4', target_image_or_class_fields_present=False, inputs=queries)
    protocol = dict(image_rows=dict(V=[row['array_row'] for row in queries]))
    validate_blind_inputs(inputs, protocol)
    bad = copy.deepcopy(inputs)
    bad['inputs'][0]['concept_id'] = 'should_not_be_a_condition'
    rejects(lambda: validate_blind_inputs(bad, protocol))
    source = Path(__file__).resolve().parent / 'generate_BrainHIVE_functional_alignment_V_v2.py'
    tree = ast.parse(source.read_text())
    assert not any(isinstance(node, (ast.Import, ast.ImportFrom)) and
        ((isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torch')) or
         (isinstance(node, ast.Import) and any(item.name.startswith('torch') for item in node.names))) for node in tree.body)
    main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
    samples = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'pipe']
    assert len(samples) == 1
    keywords = {item.arg: item.value for item in samples[0].keywords}
    assert all(isinstance(keywords[key], ast.Constant) and keywords[key].value == value
               for key, value in dict(height=512, width=512, num_inference_steps=4, guidance_scale=0.).items())
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('auxiliary', 'capture') for node in ast.walk(main))
    assert len([node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'attach_targets_after_all_inference']) == 1
    return dict(status='PURE_CPU_FUNCTIONAL_V2_GENERATION_IDENTITY_AND_SCHEMA_CHECKS_PASSED',
        generation_source_sha256=sha(source), all_four_arms_have_own_names_and_recipe_identity=True,
        mismatched_stimulus_array_seed_arm_and_old_run_rejected=True,
        blind_inputs_reject_target_class_fields=True,
        fixed_native_512_four_step_CFG0_source_checked=True, no_auxiliary_or_reference_Q_in_sampling=True,
        target_join_requires_all16600_inference_records=True,
        actual_GPU_used=False, actual_images_generated=0, actual_checkpoints_loaded=False,
        actual_lossless_pixel_replay_or_full_resume_run_verified=False,
        limits=['CPU metadata/AST checks cannot validate actual pipeline hooks, GPU memory, images or final saved artifacts',
                'No formal recipe/admission or checkpoint is issued by this check'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = run_checks()
    if args.output:
        assert not args.output.exists()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
