"""Measure native first-BN repeat variance on complete T; no parameter training."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse, fcntl, hashlib, json, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--policy', required=True); args = ap.parse_args()
q = json.loads(Path(args.policy).read_bytes()); sys.path.insert(0, str(Path(q['reference_policy']['path']).parent))
import enigma_native_reference_contract_20261104_v1 as C
c = C.setup(q['reference_policy']['path']); K = c['K']; repo = c['repo']; out = Path(q['output_root'])
assert K.ref(__file__) == q['sources']['GPU']
for item in q['sources'].values(): K.checked(item)
before = C.resources(c, starting=True); cap = q['maximum_output_bytes']
assert cap == 64 << 20 and q['new_models'] == q['new_updates'] == q['new_images'] == 0
assert before['free_bytes'] >= before['required_including_this_project'] + cap
started = time.monotonic(); rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
job = K.read(repo / 'audit' / (q['job_prefix'] + '_measure_job.json'))
assert job['pid'] == os.getpid() and job['status'] == 'RUNNING'
lock = (repo / 'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == c['parent']['GPU_UUID']
sys.path[:0] = c['q']['existing_runtime_paths']
import torch
import torch.nn.functional as F
from enigma_native_reference_math_20261104_v1 import author_namespace
sys.path.insert(0, str(Path(args.policy).parent))
from eeg_measurement_count_variance_math_20261104_v1 import predicted_variance, finite_pool_moments
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory)

def guard():
    assert time.monotonic() - started < 850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < c['parent']['maximum_extra_CPU_bytes']
    assert torch.cuda.max_memory_allocated() < c['parent']['maximum_GPU_bytes']
    used = sum(p.stat().st_blocks * 512 for p in Path(q['project_root']).rglob('*') if p.is_file())
    assert used < cap
    stat = os.statvfs(q['project_root'])
    assert stat.f_bavail * stat.f_frsize >= before['required_including_this_project'] + cap - used

def progress(phase, **items): K.write(out / 'progress.json', dict(phase=phase, **items))
try:
    source = K.checked(q['physical_source']); assert source.stat().st_size == 4168083005
    progress('LOAD_SHA_PINNED_REAL_R4')
    obj = np.load(source, allow_pickle=True)
    if not isinstance(obj, dict): obj = obj.item()
    raw = obj['preprocessed_eeg_data']; assert raw.shape == (16540, 4, 63, 250) and raw.dtype == np.float32
    mean_cache = torch.load(K.checked(c['q']['EEG']), map_location='cpu', weights_only=True, mmap=True)
    checkpoint = K.checked(q['checkpoint'])
    assert checkpoint.is_relative_to(Path(c['q']['project_root'])) and checkpoint.name == 'epoch_150.pt'
    state = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert state['role'] == 'full' and state['epoch'] == 150 and state['updates'] == 3900
    scope = author_namespace(K.checked(c['q']['original_models']).read_text(), K.checked(c['q']['original_training']).read_text())
    net = scope['ENIGMA'](63, 250, ['sub-01']).cuda().eval().requires_grad_(False)
    net.load_state_dict(state['model'], strict=True)
    assert sum(p.numel() for p in net.parameters()) == 1405242
    linear = net.subject_wise_linear['sub-01']; conv = net.tsencoder.tsconv[0]; pool = net.tsencoder.tsconv[1]
    bn = net.tsencoder.tsconv[2]; assert isinstance(bn, torch.nn.BatchNorm2d) and bn.num_features == 40
    original_mean = bn.running_mean.detach().clone(); original_var = bn.running_var.detach().clone()
    w, b = linear.weight.double(), linear.bias.double()
    cw, cb = conv.weight.double(), conv.bias.double()
    def prefix(x):
        a = F.linear(x, w, b).reshape(-1, 1, 63, 250)
        a = F.conv2d(a, cw, cb, stride=conv.stride, padding=conv.padding)
        return F.avg_pool2d(a, pool.kernel_size, pool.stride, pool.padding)
    total = torch.zeros((3, 40), dtype=torch.float64, device='cuda')
    total_positions = 0; max_mean_error = 0.; prefix_commutation_error = 0.; finite_pool_checks = []
    T = c['T']; assert len(T) == 13230
    with torch.inference_mode():
        for lo in range(0, len(T), 32):
            guard(); ids = T[lo:lo+32]; batch = torch.from_numpy(np.ascontiguousarray(raw[ids])).cuda().double()
            n = len(ids); mean_raw = batch.mean(1)
            error = float((mean_raw.float().cpu() - mean_cache[ids]).abs().max())
            max_mean_error = max(max_mean_error, error)
            # Explicitly bound float32-trial versus float64-source mean rounding; do not replace cache inputs.
            assert error < 4e-6, dict(source_mean_cache_error=error, positions=ids.tolist())
            a = prefix(batch).reshape(n, 4, 40, 63, -1); am = a.mean(1)
            total[0] += am.sum((0, 2, 3)); total[1] += am.square().sum((0, 2, 3))
            total[2] += (a - am[:, None]).square().sum((0, 1, 3, 4)) / 3
            total_positions += n * am.shape[2] * am.shape[3]
            if lo == 0:
                err = float((prefix(mean_raw) - am).abs().max()); prefix_commutation_error = err
                assert err < 2e-10
                small = a[:8].cpu().numpy().reshape(8, 4, 40, -1)
                measured = finite_pool_moments(small)
                for count in (1, 2, 3, 4):
                    expected = measured['variance4'] + (1/count - .25) * measured['contrast_variance']
                    delta = float(abs(expected - measured['subset_variances'][count]).max())
                    assert delta < 2e-10
                    finite_pool_checks.append(dict(count=count, all40_variance_identity_max_error=delta))
                K.write(out / 'first8_prefix_fixture.json', dict(T_rows=ids[:8].tolist(), mean=measured['mean'].tolist(),
                    variance4=measured['variance4'].tolist(), contrast_variance=measured['contrast_variance'].tolist(),
                    subset_variances={str(k):v.tolist() for k,v in measured['subset_variances'].items()}))
            if lo % 1024 == 0: progress('ALL_T_FIRST_BN_REPEAT_MOMENTS', completed=min(lo+32, len(T)), total=len(T))
            del batch, a, am, mean_raw
    moments = total.cpu().numpy() / total_positions
    mu = moments[0]; variance4 = moments[1] - mu**2; nu = moments[2]
    assert np.all(variance4 > 0) and np.all(nu >= 0)
    em = original_mean.cpu().numpy().astype(float); ev = original_var.cpu().numpy().astype(float)
    forecasts = {}; invalid = {}
    for name, v in (('EMA', ev), ('T_population', variance4)):
        candidate = v + (1/80 - .25) * nu
        invalid[name] = np.flatnonzero(candidate <= 0).tolist()
        forecasts[name] = candidate.tolist()
    stats = dict(status='COMPLETE_REAL_ALL_T_FIRST_BN_MEASUREMENT_MOMENTS', policy=K.ref(args.policy),
        physical_source=q['physical_source'], original_mean4_source=c['q']['EEG'], source_dtype='float32',
        cached_mean4_was_from_original_float64_source=True, all13230_mean_cache_max_rounding_error=max_mean_error,
        T_rows_SHA256=hashlib.sha256(T.tobytes()).hexdigest(), stimuli=13230, concepts=1323,
        prefix_linear_through_subject_linear_temporal_conv_and_average_pool_only=True,
        prefix_mean_commutation_error=prefix_commutation_error, finite_pool_identity_checks=finite_pool_checks,
        mean_T=mu.tolist(), variance_R4_T=variance4.tolist(), repeat_contrast_variance=nu.tolist(),
        EMA_mean=em.tolist(), EMA_variance=ev.tolist(), forecast_R80=forecasts, nonpositive_forecast_channels=invalid,
        exchangeable_repeat_covariance_and_transfer_to_R80_are_assumptions=True,
        no_claim_repeat_variation_is_only_neural_noise=True, finite_pool_identity_not_validation_of_R80_extrapolation=True,
        no_test_EEG_or_images_used_for_statistics=True)
    K.write(out / 'all_T_first_BN_moments.json', stats)
    assert K.sha(source) == q['physical_source']['sha256']
    if any(invalid.values()):
        K.write(out / 'completion.json', dict(status='MEASURED_PHYSICAL_FORECAST_INADMISSIBLE_NO_CLAMP_NO_EXPORT',
            policy=K.ref(args.policy), moments=K.ref(out/'all_T_first_BN_moments.json'), new_updates=0, new_images=0,
            seconds=time.monotonic()-started, nonpositive_forecast_channels=invalid))
    else:
        r80 = torch.load(K.checked(q['EEG_R80']), map_location='cpu', weights_only=True, mmap=True)
        assert r80.shape == (200, 63, 250)
        configs = [('EMA_original', em, ev), ('EMA_count', em, np.asarray(forecasts['EMA'])),
                   ('T_population', mu, variance4), ('T_count', mu, np.asarray(forecasts['T_population']))]
        all_conditions = np.empty((200, 4, 1024), np.float32)
        # Exact R4 identity for the count-only EMA path uses cached inputs and unchanged original buffers.
        with torch.inference_mode():
            ids = T[:8]; before_r4 = net(mean_cache[ids].cuda(), ['sub-01']*8)
            same_var = predicted_variance(ev, nu, 4)
            bn.running_var.copy_(torch.tensor(same_var, device='cuda', dtype=bn.running_var.dtype))
            after_r4 = net(mean_cache[ids].cuda(), ['sub-01']*8)
            assert torch.equal(original_var, bn.running_var) and torch.equal(before_r4, after_r4)
            for j, (name, mm, vv) in enumerate(configs):
                progress('EXPORT_FROZEN_R80_CONDITIONS', arm=name)
                bn.running_mean.copy_(torch.tensor(mm, device='cuda', dtype=bn.running_mean.dtype))
                bn.running_var.copy_(torch.tensor(vv, device='cuda', dtype=bn.running_var.dtype))
                for lo in range(0, 200, 32):
                    guard(); x = r80[lo:lo+32].cuda(); all_conditions[lo:lo+len(x), j] = net(x, ['sub-01']*len(x)).cpu().numpy()
            bn.running_mean.copy_(original_mean); bn.running_var.copy_(original_var)
            replay = net(mean_cache[ids].cuda(), ['sub-01']*8)
            assert torch.equal(replay, before_r4)
        for key, value in state['model'].items(): assert torch.equal(net.state_dict()[key].cpu(), value)
        np.save(out / 'R80_200_conditions.npy', all_conditions, allow_pickle=False)
        K.write(out / 'completion.json', dict(status='COMPLETE_ALL_T_MEASUREMENT_CALIBRATION_AND_FROZEN_R80_4ARMS',
            policy=K.ref(args.policy), moments=K.ref(out/'all_T_first_BN_moments.json'),
            conditions=K.ref(out/'R80_200_conditions.npy'), arms=[v[0] for v in configs], EEG_R80=q['EEG_R80'],
            queries=200, all200_original_test_cache_order=True, labels_or_images_not_read=True,
            only_first_BN_running_mean_variance_modified_then_all_state_restored=True,
            exact_R4_EMA_count_identity=True, all_original_checkpoint_state_restored=True,
            model_parameters=1405242, new_models=0, new_updates=0, new_images=0,
            no_R80_scores_or_method_efficacy_claim=True, no_new_published_or_originality_claim=True,
            seconds=time.monotonic()-started, peak_GPU_bytes=torch.cuda.max_memory_allocated(),
            extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024-rss0))
    guard(); progress('COMPLETE_MEASUREMENT_COUNT_ENGINEERING')
except BaseException as exc:
    K.write(out/'failure.json', dict(status='FAILED_PREFIX_RETAINED_NO_AUTO_RETRY', error=repr(exc),
        traceback=traceback.format_exc(), seconds=time.monotonic()-started)); raise
