"""T-only source/OOF contract and inherited resource commitments."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from enigma_repeat_task_risk_math_20261105_v1 import SUBSETS, METRICS


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def ref(path):
    p = Path(path)
    return dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))


def checked(pin):
    actual = ref(pin['path'])
    assert actual['sha256'] == pin['sha256'], (pin['path'], actual)
    if 'bytes' in pin:
        assert actual['bytes'] == pin['bytes'], (pin['path'], actual)
    return Path(pin['path'])


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    p = Path(path)
    tmp = p.with_name(p.name + '.partial')
    tmp.write_bytes((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
    tmp.replace(p)


def process(pid, start_ticks=None):
    p = Path('/proc') / str(int(pid)) / 'stat'
    try:
        parts = p.read_text().rsplit(')', 1)[1].split()
        state, ticks = parts[0], int(parts[19])
        return dict(PID=int(pid), state=state, start_ticks=ticks,
                    live=state != 'Z' and (start_ticks is None or start_ticks == ticks))
    except FileNotFoundError:
        return dict(PID=int(pid), state='MISSING', live=False)


def allocated(path):
    return sum(p.stat().st_blocks * 512 for p in Path(path).rglob('*') if p.is_file()) if Path(path).exists() else 0


def setup(path):
    q = read(path)
    assert q['status'] == 'LOCKED_T_ONLY_FROZEN_ENCODER_OOF_REPEAT_TASK_RISK'
    assert q['host'] == 'old' and q['seed'] == 1729 and q['subject'] == 'sub-01'
    assert q['maximum_output_bytes'] == 32 << 20 and q['maximum_GPU_job_seconds'] == 900
    assert q['new_models'] == q['new_updates'] == q['new_images'] == 0
    assert q['subsets'] == [list(s) for s in SUBSETS] and q['metrics'] == list(METRICS)
    for pin in q['sources'].values():
        checked(pin)
    for key in ('reference_policy', 'prepared', 'physical_source', 'physical_source_completion',
                'mean_cache_replay_proof', 'cache_builder', 'EEG', 'visual_target', 'visual_ordering',
                'text_ordering', 'legacy_inputs', 'original_models', 'original_training', 'author_math',
                'OOF_conditions', 'resource_contract', 'prior_image_policy', 'prior_image_completion',
                'prior_image_launch', 'owned_process', 'GPU_wrapper'):
        checked(q[key])
    for pin in q['checkpoints'].values():
        checked(pin)
    rp = read(q['reference_policy']['path'])
    prepared = read(q['prepared']['path'])
    assert prepared['OOF_conditions'] == q['OOF_conditions']
    assert prepared['whole_pipeline_OOF_NOT_established']
    for role in ('oof0', 'oof1'):
        assert prepared['models'][role]['checkpoint'] == q['checkpoints'][role]
    for key in ('EEG', 'visual_target', 'visual_ordering', 'legacy_inputs', 'original_models', 'original_training'):
        assert rp[key]['path'] == q[key]['path'] and rp[key]['sha256'] == q[key]['sha256']
    proof = read(q['mean_cache_replay_proof']['path'])
    assert proof['status'] == 'PASS_ALL_T_BIT_EXACT_ACTUAL_CAST_THEN_FLOAT32_MEAN'
    assert proof['mismatched_values'] == 0 and proof['values_checked'] == 208372500
    assert proof['source'] == q['physical_source'] and proof['mean_cache']['sha256'] == q['EEG']['sha256']
    sys.path.insert(0, str(Path(q['author_math']['path']).parent))
    from enigma_native_reference_math_20261104_v1 import role_rows, canonical_concept_join
    legacy = read(q['legacy_inputs']['path'])
    T = np.asarray(legacy['T_array_rows'], np.int64)
    folds = np.asarray(legacy['T_heldout_folds'], np.int64)
    assert len(T) == 13230 and np.array_equal(T, np.sort(T)) and len(set(T)) == 13230
    assert np.all(T.reshape(-1, 10) % 10 == np.arange(10))
    split = read(checked(legacy['split']))
    assert np.array_equal(T, np.asarray(split['image_rows']['T'], np.int64))
    order = next(v for v in read(q['text_ordering']['path']) if v['split'] == 'train')
    visual = sorted((v for v in read(q['visual_ordering']['path']) if v['split'] == 'train'), key=lambda v: v['index'])
    assert len(visual) == len(order['filenames']) == 16540
    assert [v['index'] for v in visual] == list(range(16540))
    assert [Path(v['member']).name for v in visual] == order['filenames']
    assert all(Path(v['member']).parent.name == order['class_folders'][i // 10] for i, v in enumerate(visual))
    join = canonical_concept_join(order['class_folders'], split['train_concepts'])
    canonical = [r['canonical_split_concept'] for r in join]
    assert set(canonical[int(v) // 10] for v in T) == set(split['train_concepts']['T'])
    assert all(f == split['oof_fold_by_concept'][canonical[int(v) // 10]] for v, f in zip(T, folds))
    roles = {r: role_rows(T, folds, r) for r in ('oof0', 'oof1')}
    assert sorted(len(v[1]) for v in roles.values()) == [6610, 6620]
    positions = np.full(16540, -1, np.int64)
    positions[T] = np.arange(len(T))
    return dict(q=q, T=T, folds=folds, roles=roles, positions=positions, out=Path(q['output_root']))


def resources(c, starting=False):
    q = c['q']
    spec = importlib.util.spec_from_file_location('pinned_R80_resource_contract', checked(q['resource_contract']))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    imageq = read(q['prior_image_policy']['path'])
    done = read(q['prior_image_completion']['path'])
    launch = read(q['prior_image_launch']['path'])
    assert done['status'] == 'COMPLETE_FOUR_ARM_R80_COUNT_IMAGE_STUDY'
    assert not process(launch['PID'], launch['start_ticks'])['live']
    assert done['policy'] == q['prior_image_policy']
    for event in done['events']:
        if event['GPU']:
            job = read(checked(event['receipt']))
            assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
            assert not process(job['pid'])['live'] and not process(job['wrapper_pid'])['live']
    K = SimpleNamespace(checked=checked, read=read, process=process, ref=ref)
    prior = module.resources(dict(K=K, q=imageq, ledger=read(checked(imageq['reserve_ledger']))), starting=starting)
    used = allocated(q['project_root'])
    assert used <= q['maximum_output_bytes']
    required = prior['required_including_this_project'] + q['maximum_output_bytes'] - used
    assert prior['free_bytes'] >= required, dict(free=prior['free_bytes'], required=required)
    return dict(prior=prior, free_bytes=prior['free_bytes'], required_including_this_project=required,
                own_used_bytes=used, own_remaining_bytes=q['maximum_output_bytes'] - used,
                no_additional_old_allowance_settlement=True, no_files_deleted=True,
                image384MiB_and_measurement64MiB_remaining_and_all_other_forecasts_still_charged=True)
