"""Pure CPU one-shot observer contracts; no /proc, GPU, model or launch call."""
import argparse
import ast
import copy
import json
from pathlib import Path
from unittest.mock import patch

import queue_BrainHIVE_functional_T_pixel_replay_v2 as queue
from diagnose_BrainHIVE_functional_T_pixel_replay_v2 import (
    NAME, CASES, T32_REGISTRATION, REQUIRED_SOURCES, NATIVE_INPUTS,
)


def rejects(action):
    try:
        action()
    except (AssertionError, KeyError):
        return
    raise AssertionError('An unsafe one-shot observer contract was accepted')


def example_registration():
    template = dict(schema='BrainHIVE_functional_T_pixel_replay_v2_registration',
        status='PENDING_TERMINAL_FACTS_NOT_EXECUTABLE', diagnostic=NAME, seed=1729,
        fixed_T32_query_index=0, cases=list(CASES), T32_registration_path=T32_REGISTRATION,
        output_directory=str(queue.OUT), generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1),
        formal_training_enabled=False, optimizer_created=False, visual_targets_or_scores_enabled=False,
        automatic_launch_enabled=False, terminal_state_bindings=[],
        job_name='T_pixel_replay_v2_once', maximum_seconds=1200, maximum_probe_bytes=16 << 20,
        minimum_CPU_headroom_bytes=2 << 30, minimum_free_GPU_bytes=16 << 30, minimum_retained_disk_bytes=64 << 20,
        source_sha256={name: 'a' * 64 for name in REQUIRED_SOURCES},
        input_sha256={path: 'b' * 64 for path in set(NATIVE_INPUTS) | {T32_REGISTRATION}})
    return dict(schema='BrainHIVE_functional_T_pixel_replay_v2_wait_registration',
        status='REGISTERED_CPU_WAIT_ONLY_NO_GPU_LAUNCHED', automatic_retry=False,
        polling_seconds=1200, maximum_wait_seconds=7 * 86400, maximum_probe_seconds=1200,
        maximum_launch_attempts=1, reviewed_config_path=queue.CONFIG_OUT,
        queue_state_path=queue.STATE, queue_log_path=queue.LOG, queue_lock_path=queue.LOCK,
        formal_training_enabled=False, full_V_generation_enabled=False, probe_template=template,
        source_sha256=dict(template['source_sha256'], **{Path(queue.__file__).name: 'a' * 64}))


def run_checks():
    registration = example_registration()
    queue.validate_registration(registration, check_files=False)
    for key, value in [('automatic_retry', True), ('maximum_launch_attempts', 2),
                       ('polling_seconds', 60), ('maximum_wait_seconds', 8 * 86400),
                       ('maximum_probe_seconds', 1800), ('formal_training_enabled', True),
                       ('full_V_generation_enabled', True)]:
        rejects(lambda: queue.validate_registration(dict(registration, **{key: value}), check_files=False))
    changed = copy.deepcopy(registration)
    changed['probe_template']['status'] = 'REVIEWED_ENGINEERING_PROBE_READY'
    rejects(lambda: queue.validate_registration(changed, check_files=False))
    changed = copy.deepcopy(registration)
    changed['source_sha256'][queue.PROBE] = 'changed'
    rejects(lambda: queue.validate_registration(changed, check_files=False))
    main = dict(status=next(iter(queue.MAIN_TERMINAL)), pid=101)
    diag = dict(status=next(iter(queue.DIAG_TERMINAL)), pid=202)
    dead = dict(live=False, exists=False)
    live = dict(live=True, exists=True, command='either the actual controller or a reused PID')
    ready = lambda a, ap, b, bp, others=(), idle=True: queue.readiness(a, ap, b, bp, other_controllers=others, GPU_idle=idle)
    assert ready(main, dead, diag, dead)['ready']
    assert not ready(dict(main, status='RUNNING'), dead, diag, dead)['ready']
    assert not ready(main, live, diag, dead)['ready']
    assert not ready(main, dead, diag, live)['ready']
    assert not ready(main, dead, diag, dead, others=[dict(pid=303)])['ready']
    assert not ready(main, dead, diag, dead, idle=False)['ready']
    assert not ready(main, None, diag, dead)['ready']
    with patch.object(queue.Path, 'exists', return_value=True):
        rejects(lambda: queue.no_existing_probe_artifacts(registration))
    source = Path(queue.__file__)
    tree = ast.parse(source.read_text())
    spawns = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
              and node.func.attr == 'Popen']
    assert len(spawns) == 1, 'There is only one launch boundary'
    assert not any(isinstance(node, ast.Attribute) and node.attr in ('kill', 'killpg', 'terminate') for node in ast.walk(tree))
    assert not any(isinstance(node, ast.ImportFrom) and (node.module or '').startswith('torch') or
        isinstance(node, ast.Import) and any(item.name.startswith('torch') for item in node.names) for node in tree.body)
    return dict(status='CPU_ONE_SHOT_T_PIXEL_WAIT_CONTRACT_PASSED', queue_source_sha256=queue.sha(source),
        preregistered_static_template_not_executable=True, exactly_one_probe_launch_site=True,
        finite_1200_second_polling_and_seven_day_wait=True, maximum_probe_job_seconds=1200,
        RUNNING_live_PID_reused_PID_other_controller_or_busy_GPU_never_ready=True,
        missing_process_identity_never_promoted=True, existing_artifacts_rejected_without_overwrite=True,
        no_kill_restart_fit_or_full_V_actions=True, actual_probe_launched=False,
        actual_GPU_or_model_used=False, reviewed_config_created=False,
        limits=['Actual static server pins, /proc observations, lock and resource availability remain runtime prerequisites',
                'This CPU fixture does not prove three-image pixel equality or launch a waiting controller'])


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
