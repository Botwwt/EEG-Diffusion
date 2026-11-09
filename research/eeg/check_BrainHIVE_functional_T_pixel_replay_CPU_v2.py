"""Small metadata/AST checks; does not import Torch or execute the pixel probe."""
import argparse
import ast
import copy
import json
from pathlib import Path

from diagnose_BrainHIVE_functional_T_pixel_replay_v2 import (
    ROOT, OUT, NAME, CASES, T32_REGISTRATION, REQUIRED_SOURCES, NATIVE_INPUTS,
    validate_config, reject_active_terminal, select_fixed_queries, sha,
)


def rejects(action):
    try:
        action()
    except (AssertionError, KeyError):
        return
    raise AssertionError('An invalid engineering registration was accepted')


def run_checks():
    config = dict(schema='BrainHIVE_functional_T_pixel_replay_v2_registration',
        status='REVIEWED_ENGINEERING_PROBE_READY', diagnostic=NAME, seed=1729,
        fixed_T32_query_index=0, cases=list(CASES), T32_registration_path=T32_REGISTRATION,
        output_directory=str(OUT), generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1),
        formal_training_enabled=False, optimizer_created=False, visual_targets_or_scores_enabled=False,
        automatic_launch_enabled=False, job_name='T_pixel_replay_v2_engineering', maximum_seconds=1200,
        maximum_probe_bytes=16 << 20, minimum_CPU_headroom_bytes=2 << 30,
        minimum_free_GPU_bytes=16 << 30, minimum_retained_disk_bytes=64 << 20,
        source_sha256={name: 'a' * 64 for name in REQUIRED_SOURCES},
        input_sha256={path: 'b' * 64 for path in set(NATIVE_INPUTS) | {T32_REGISTRATION}})
    assert validate_config(config, check_files=False)
    for key, value in [('status', 'DRAFT_NOT_ADMITTED'), ('fixed_T32_query_index', 1),
                       ('seed', 1730), ('formal_training_enabled', True), ('optimizer_created', True),
                       ('maximum_probe_bytes', 0), ('automatic_launch_enabled', True)]:
        rejects(lambda: validate_config(dict(config, **{key: value}), check_files=False))
    changed = copy.deepcopy(config)
    changed['generator']['steps'] = 8
    rejects(lambda: validate_config(changed, check_files=False))
    assert not any('/data/fused_train_targets.pt' in name or '/generation_inputs/V' in name for name in NATIVE_INPUTS)
    terminal = dict(status='COMPLETE', pid=12345)
    dead = dict(exists=False, live=False, state=None, command=None)
    assert reject_active_terminal(terminal, expected_states={'COMPLETE'}, snapshot=dead)['pid'] == 12345
    rejects(lambda: reject_active_terminal(dict(terminal, status='RUNNING'), expected_states={'COMPLETE'}, snapshot=dead))
    live = dict(exists=True, live=True, state='S', command='the actual controller')
    rejects(lambda: reject_active_terminal(terminal, expected_states={'COMPLETE'}, snapshot=live))
    # The exact first pre-existing input is selected, not the smallest array row
    # or the nicest image. A nonzero first array row catches position confusion.
    allowed = [10001 + i * 13 for i in range(40)]
    metadata = {row: dict(stimulus_id='stimulus_' + str(row) + '.jpg', dataset_split='train', concept_id='concept_' + str(i))
                for i, row in enumerate(allowed)}
    ordered = [dict(stimulus_id=metadata[row]['stimulus_id'], array_row=row) for row in allowed[7:39]]
    selection = select_fixed_queries(ordered, allowed, metadata)
    assert selection[0]['array_row'] == ordered[0]['array_row'] and selection[0]['oracle_query_index'] == 0
    assert selection[0]['T_position'] == 7 and selection[0]['array_row'] != 7
    wrong = copy.deepcopy(ordered)
    wrong[0]['stimulus_id'] = 'a_different_target.jpg'
    rejects(lambda: select_fixed_queries(wrong, allowed, metadata))
    source = ROOT / 'diagnose_BrainHIVE_functional_T_pixel_replay_v2.py'
    tree = ast.parse(source.read_text())
    run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'run_probe')
    calls = [node for node in ast.walk(run) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'pipe']
    assert len(calls) == 1
    keywords = {item.arg: item.value for item in calls[0].keywords}
    assert all(isinstance(keywords[key], ast.Constant) and keywords[key].value == value
               for key, value in dict(height=512, width=512, num_inference_steps=4, guidance_scale=0.).items())
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('backward', 'step', 'zero_grad') for node in ast.walk(run))
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torch') or
                   isinstance(node, ast.Import) and any(item.name.startswith('torch') for item in node.names)
                   for node in tree.body)
    return dict(status='CPU_T_PIXEL_REPLAY_V2_REGISTRATION_AND_SOURCE_CONTRACT_PASSED',
        source_sha256=sha(source), first_fixed_T_query_not_selected_by_quality=True,
        array_row_and_tensor_position_mapping_checked=True, RUNNING_or_live_terminal_controller_rejected=True,
        native_asset_source_pins_required=True, DRAFT_or_changed_seed_steps_query_index_rejected=True,
        three_cases=list(CASES), same_native_four_step_sampling_rule_source_checked=True,
        no_optimizer_or_training_calls_in_probe=True, no_four_arm_storage_or_positive_functional_requirement=True,
        actual_GPU_used=False, actual_models_loaded=False, actual_images_generated=0,
        actual_pixel_or_lossless_replay_passed=False,
        limits=['CPU source/metadata checks are not an actual image replay',
                'Exclusive GPU lock, native asset availability, exact pixels and runtime resources remain unverified'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = run_checks()
    if args.output:
        assert not args.output.exists()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
