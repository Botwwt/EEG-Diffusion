"""All 13230 T stimuli x 15 physical subsets, original frozen concept-OOF models."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse
import fcntl
import hashlib
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import enigma_repeat_task_risk_contract_20261105_v1 as C
from enigma_repeat_task_risk_math_20261105_v1 import SUBSETS, COUNTS, METRICS, task_metrics, summaries

ap = argparse.ArgumentParser()
ap.add_argument('--policy', required=True)
args = ap.parse_args()
started = time.monotonic()
rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
c = C.setup(args.policy)
q = c['q']
out = c['out']
assert C.ref(__file__) == q['sources']['GPU']
before = C.resources(c, starting=True)
repo = Path(q['GPU_wrapper']['path']).parent
jobpath = repo / 'audit' / (q['job_prefix'] + '_measure_job.json')
job = C.read(jobpath)
assert job['pid'] == os.getpid() and job['status'] == 'RUNNING'
lock = (repo / 'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == q['GPU_UUID']
sys.path[:0] = q['existing_runtime_paths']
import torch
from enigma_native_reference_math_20261104_v1 import author_namespace
torch.set_num_threads(2)
torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(q['maximum_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory)


def guard():
    assert time.monotonic() - started < q['maximum_GPU_job_seconds'] - 25
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < q['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated() < q['maximum_GPU_bytes']
    used = C.allocated(q['project_root'])
    assert used < q['maximum_output_bytes']
    stat = os.statvfs(q['project_root'])
    assert stat.f_bavail * stat.f_frsize >= before['required_including_this_project'] + before['own_used_bytes'] - used


try:
    C.write(out / 'progress.json', dict(phase='LOAD_ONLY_PERMITTED_T_SOURCES'))
    obj = np.load(C.checked(q['physical_source']), allow_pickle=True)
    if not isinstance(obj, dict):
        obj = obj.item()
    raw = obj['preprocessed_eeg_data']
    assert raw.shape == (16540, 4, 63, 250) and raw.dtype == np.float32
    mean4 = torch.load(q['EEG']['path'], map_location='cpu', weights_only=True, mmap=True)
    targets = torch.load(q['visual_target']['path'], map_location='cpu', weights_only=True, mmap=True).numpy()
    assert mean4.shape == (16540, 63, 250) and targets.shape == (16540, 1024)
    bank = np.load(q['OOF_conditions']['path'], allow_pickle=False, mmap_mode='r')
    assert bank.shape == (13230, 1024)
    scope = author_namespace(Path(q['original_models']['path']).read_text(), Path(q['original_training']['path']).read_text())
    values = np.full((13230, 15, 5), np.nan, np.float64)
    mean_values = np.full((13230, 4, 5), np.nan, np.float64)
    dispersion = np.full((13230, 4), np.nan, np.float64)
    coverage = np.zeros((13230, 15), np.uint8)
    fixture = dict(raw=[], predictions=[], targets10=[], rows=[], positions=[], roles=[])
    role_evidence = {}
    forwards = 0
    for role in ('oof0', 'oof1'):
        fit, held = c['roles'][role]
        checkpoint = C.checked(q['checkpoints'][role])
        # These are this thread's own SHA-pinned trainer states with a OneCycle callable.
        assert checkpoint.name == 'epoch_150.pt' and checkpoint.parent.name == role
        state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
        assert state['role'] == role and state['epoch'] == 150 and state['updates'] == 1950
        assert state['fit_rows_SHA256'] == hashlib.sha256(fit.tobytes()).hexdigest()
        assert state['policy'] == q['reference_policy'] and state['scratch_initialization_seed'] == 1729
        assert not set(fit // 10) & set(held // 10)
        net = scope['ENIGMA'](63, 250, ['sub-01']).cuda().eval().requires_grad_(False)
        net.load_state_dict(state['model'], strict=True)
        assert sum(p.numel() for p in net.parameters()) == 1405242
        max_replay = 0.
        with torch.inference_mode():
            for lo in range(0, len(held), 64):
                guard()
                rows = held[lo:lo + 64]
                pos = c['positions'][rows]
                block = raw[rows].astype(np.float32)
                assert np.array_equal(block.mean(1), mean4[rows].numpy())
                group_targets = targets[(rows // 10)[:, None] * 10 + np.arange(10)]
                accum = np.zeros((len(rows), 4, 1024), np.float64)
                energy = np.zeros((len(rows), 4), np.float64)
                saved = []
                for j, subset in enumerate(SUBSETS):
                    x = np.ascontiguousarray(block[:, subset].mean(1))
                    pred = net(torch.from_numpy(x).cuda(), ['sub-01'] * len(rows)).cpu().numpy()
                    assert pred.dtype == np.float32 and np.isfinite(pred).all()
                    values[pos, j] = task_metrics(pred, group_targets, rows % 10)
                    k = len(subset) - 1
                    accum[:, k] += pred.astype(np.float64)
                    energy[:, k] += np.square(pred.astype(np.float64)).mean(1)
                    coverage[pos, j] += 1
                    forwards += len(rows)
                    if len(subset) == 4:
                        err = float(np.max(abs(pred - bank[pos])))
                        max_replay = max(max_replay, err)
                        assert err <= q['R4_OOF_replay_max_absolute_error']
                    if lo == 0:
                        saved.append(pred[:8].copy())
                for k, n in enumerate((1, 2, 3, 4)):
                    number = int((COUNTS == n).sum())
                    average_pred = accum[:, k] / number
                    mean_values[pos, k] = task_metrics(average_pred, group_targets, rows % 10)
                    dispersion[pos, k] = energy[:, k] / number - np.square(average_pred).mean(1)
                if lo == 0:
                    fixture['raw'].append(block[:8].copy())
                    fixture['predictions'].append(np.stack(saved, axis=1))
                    fixture['targets10'].append(group_targets[:8].copy())
                    fixture['rows'].append(rows[:8].copy())
                    fixture['positions'].append(pos[:8].copy())
                    fixture['roles'].append(np.full(8, int(role[-1]), np.int64))
                if lo % 1024 == 0:
                    C.write(out / 'progress.json', dict(phase='COMPLETE_T_SUBSET_TASK_FORWARD', role=role,
                        held_rows_completed=min(lo + 64, len(held)), held_rows_total=len(held),
                        total_frozen_forwards=forwards, maximum_R4_OOF_replay_error=max_replay))
        for key, value in net.state_dict().items():
            assert torch.equal(value.cpu(), state['model'][key]), key
        role_evidence[role] = dict(checkpoint=q['checkpoints'][role], fit_rows=len(fit), held_rows=len(held),
            fit_rows_SHA256=state['fit_rows_SHA256'], held_rows_SHA256=hashlib.sha256(held.tobytes()).hexdigest(),
            held_concepts_not_in_encoder_training=True, all_weights_and_BN_buffers_unchanged=True,
            maximum_R4_original_OOF_bank_error=max_replay)
        del net, state
        torch.cuda.empty_cache()
    assert forwards == 198450 and np.all(coverage == 1)
    assert np.isfinite(values).all() and np.isfinite(mean_values).all()
    readout, counts = summaries(values, mean_values, dispersion)
    np.savez_compressed(out / 'all_T_task_metrics.npz', per_view=values, loss_of_mean=mean_values,
                        dispersion=dispersion, count_means=counts, coverage=coverage, T=c['T'], folds=c['folds'])
    np.savez_compressed(out / 'fixed16_all15_fixture.npz', **{k: np.concatenate(v) for k, v in fixture.items()})
    C.write(out / 'readout.json', dict(status='COMPLETE_T_ONLY_FROZEN_OOF_ALL_REPEAT_TASK_RISKS',
        policy=C.ref(args.policy), metrics=list(METRICS), subsets=[list(s) for s in SUBSETS], **readout,
        descriptive_only_single_subject_single_training_seed=True,
        shared_recordings_between_subsets_not_independent_trials=True,
        encoder_OOF_only_whole_preprocessing_chain_OOF_unproved=True,
        held_R3_is_not_validation_of_R80_extrapolation=True,
        no_R80_EEG_target_or_image_array_loaded=True, no_model_or_formula_selected_from_official_R80=True))
    guard()
    C.write(out / 'scientific_completion.json', dict(status='COMPLETE_ALL13230_T_ALL15_SUBSETS_FROZEN_OOF_TASK_RISK',
        policy=C.ref(args.policy), metrics=C.ref(out / 'all_T_task_metrics.npz'),
        fixture=C.ref(out / 'fixed16_all15_fixture.npz'), readout=C.ref(out / 'readout.json'),
        roles=role_evidence, stimuli=13230, concepts=1323, frozen_EEG_forwards=forwards,
        all_T_actual_mean4_cache_bit_exact=True, no_new_models=True, new_updates=0, new_images=0,
        seconds=time.monotonic() - started, peak_GPU_bytes=torch.cuda.max_memory_allocated(),
        extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0))
    C.write(out / 'progress.json', dict(phase='COMPLETE_ALL_T_REPEAT_TASK_RISK', forwards=forwards))
except BaseException as exc:
    C.write(out / 'failure.json', dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY', error=repr(exc),
        traceback=traceback.format_exc(), seconds=time.monotonic() - started))
    raise
