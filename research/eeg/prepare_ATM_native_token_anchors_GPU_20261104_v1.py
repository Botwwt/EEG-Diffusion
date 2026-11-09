"""Frozen T-only native token means; no optimizer, UNet or research image."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse, fcntl, hashlib, json, resource, subprocess, sys, time, traceback
from pathlib import Path
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--policy', required=True); a = ap.parse_args()
path = Path(a.policy); q = json.loads(path.read_bytes())
assert q['host'] == 'old' and q['status'] == 'LOCKED_T_ONLY_NATIVE_TOKEN_ANCHOR_PREPARATION_NO_IMAGES'
assert q['maximum_output_bytes'] == 128 << 20 and q['maximum_GPU_job_seconds'] == 900
sys.path.insert(0, str(Path(q['image_policy']['path']).parent))
import atm_token_split_image_contract_20261104_v1 as C
c = C.setup(q['image_policy']['path']); K = c['K']; main = c['up']; out = Path(q['output_root'])
started = time.monotonic(); rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
assert K.ref(__file__) == q['sources']['GPU']
for key in ('image_policy', 'image_completion', 'projection_weights', 'native_P400', 'old_new_conditions400', 'T8_conditions', 'reference_T_CPU_statistics'): K.checked(q[key])
done = K.read(q['image_completion']['path']); assert done['status'] == 'COMPLETE_EIGHT_ARM_NATIVE_IP_SPLIT_FULL400_IMAGE_STUDY'
launch = K.read(Path(q['image_policy']['path']).parent / 'actual_launch.json'); assert not K.process(launch['PID'], launch['start_ticks'])['live']
for e in done['events']:
    if e['GPU']:
        j = K.read(K.checked(e['receipt'])); assert j['status'] == 'COMPLETED' and j['exit_code'] == 0 and not K.process(j['pid'])['live'] and not K.process(j['wrapper_pid'])['live']
before = C.resources(c, starting=True)
job = K.read(c['repo'] / 'audit' / (q['job_name'] + '_job.json')); assert job['pid'] == os.getpid() and job['status'] == 'RUNNING'
lock = (c['repo'] / 'audit/TRIALSET_GPU_INTERLEAVE.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == c['parent']['GPU_UUID']
assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
import torch
torch.set_num_threads(2); torch.manual_seed(1729); torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
torch.cuda.set_per_process_memory_fraction(c['parent']['maximum_GPU_bytes'] / torch.cuda.get_device_properties(0).total_memory)
from brainhive_native_direction_contract_20261103_v1 import tensor_digest

def used(): return c['up']['cv']['ctx']['base']['ctx']['C'].allocated(q['project_root'])
def guard():
    assert time.monotonic() - started < 850 and used() < q['maximum_output_bytes']
    fs = os.statvfs(q['project_root']); assert fs.f_bavail * fs.f_frsize >= before['required_including_this_project'] + q['maximum_output_bytes'] - used()
    assert torch.cuda.max_memory_allocated() < c['parent']['maximum_GPU_bytes']
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < c['parent']['maximum_extra_CPU_bytes']

try:
    class ExactNativeProjection(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.image_embeds = torch.nn.Linear(1024, 8192); self.norm = torch.nn.LayerNorm(2048, eps=1e-5)
        def forward(self, x):
            return self.norm(self.image_embeds(x).reshape(x.shape[0], 4, 2048))
    weights = np.load(q['projection_weights']['path'], allow_pickle=False)
    p = ExactNativeProjection().half().cuda().eval().requires_grad_(False)
    p.load_state_dict({k: torch.from_numpy(weights[k].copy()) for k in weights.files}, strict=True)
    assert all(np.array_equal(v.detach().cpu().numpy(), weights[k]) for k, v in p.state_dict().items())
    p_digest = tensor_digest(p, half=True)
    conditions = np.load(q['old_new_conditions400']['path'], allow_pickle=False)
    expected_P = np.load(q['native_P400']['path'], allow_pickle=False)
    assert conditions.shape == (400, 2, 1024) and expected_P.shape == (400, 2, 4, 2048)
    with torch.inference_mode():
        for i in range(400):
            if i % 32 == 0: guard()
            for j in range(2):
                actual = p(torch.tensor(conditions[i, j][None].copy(), device='cuda').half()).cpu().numpy()[0]
                assert np.array_equal(actual, expected_P[i, j]), (i, j, 'native P replay')
    sys.path.insert(0, '/root/atm_official_20261016')
    from models.atms import ATMS
    raw = torch.load(K.checked(main['q']['EEG']), map_location='cpu', weights_only=True, mmap=True)
    visual = torch.load(K.checked(main['q']['visual_target']), map_location='cpu', weights_only=True, mmap=True)
    T = main['T']; assert len(T) == 13230 and T.reshape(-1, 10).shape == (1323, 10)
    assert raw.shape == (16540, 63, 250) and visual.shape == (16540, 1024)
    models = {}; model_pins = {}; digests = {}
    for role in ('old', 'new'):
        pin = main['q']['encoder'] if role == 'old' else K.read(main['out'] / 'fits/nce_continued/completion.json')['checkpoint']
        loaded = torch.load(K.checked(pin), map_location='cpu', weights_only=True, mmap=True)
        state = loaded if role == 'old' else loaded['model']
        model = ATMS().cuda().eval().requires_grad_(False); model.load_state_dict(state, strict=True)
        assert sum(v.numel() for v in model.parameters()) == 3200413
        models[role] = model; model_pins[role] = pin; digests[role] = tensor_digest(model, half=False)
        del loaded, state
    reference8 = np.load(q['T8_conditions']['path'], allow_pickle=False)
    assert np.array_equal(reference8['T_rows'], T.reshape(-1, 10)[:8, 0])
    c_sums = np.empty((3, 1323, 1024), float); token_sums = np.empty((3, 1323, 2048), float)
    token_square_sum = np.zeros((3, 2048)); errors8 = {}; seen = []
    with torch.inference_mode():
        for lo in range(0, 1323, 16):
            guard(); rows = T.reshape(-1, 10)[lo:lo + 16].reshape(-1); seen.extend(rows.tolist()); n = len(rows); count = n // 10
            e = raw[rows].cuda(); subject = torch.ones(n, device='cuda', dtype=torch.long)
            values = [models['old'](e, subject).float(), models['new'](e, subject).float(), visual[rows].cuda().float()]
            for j, value in enumerate(values):
                assert value.shape == (n, 1024) and torch.isfinite(value).all()
                if lo == 0 and j < 2:
                    role = ('old', 'new')[j]; ref = reference8[role]
                    err = float(abs(value[::10][:8].cpu().numpy() - ref).max()); assert err < 5e-5; errors8[role] = err
                tokens = p(value.half()); assert torch.isfinite(tokens).all()
                m = tokens.float().mean(dim=1).cpu().numpy().astype(float)
                v = value.cpu().numpy().astype(float)
                c_sums[j, lo:lo + count] = v.reshape(count, 10, 1024).sum(axis=1)
                token_sums[j, lo:lo + count] = m.reshape(count, 10, 2048).sum(axis=1)
                token_square_sum[j] += (m * m).sum(axis=0)
            K.write(out / 'progress.json', dict(T_concepts_complete=lo + count, T_stimuli_complete=len(seen), total_T=13230, new_optimizer_updates=0, research_images=0))
    assert np.array_equal(np.array(seen), T)
    means = token_sums.sum(axis=1) / 13230; c_means = c_sums.sum(axis=1) / 13230
    previous = np.load(q['reference_T_CPU_statistics']['path'], allow_pickle=False); c_errors = {}
    for j, key in enumerate(('direct', 'nce_continued', 'visual')):
        error = float(abs(c_means[j] - previous[key + '__mean']).max()); assert error < 5e-5; c_errors[key] = error
    variance = token_square_sum / 13230 - means * means; assert variance.min() > -1e-10
    # All three fixed anchors are estimated only from T. The Dcheck bank is used
    # only to replay the frozen native P and prepare future, unscored inputs.
    base = torch.tensor(expected_P[:, 1].copy(), device='cuda').float()
    with torch.inference_mode():
        anchors = torch.stack([(base - base.mean(-2, keepdim=True) + torch.tensor(mu.astype(np.float32), device='cuda')[None, None]).half() for mu in means], dim=1).cpu().numpy()
    assert anchors.shape == (400, 3, 4, 2048) and np.isfinite(anchors).all()
    np.savez_compressed(out / 'T_anchor_statistics.npz', T_array_rows=T, concept_c_sums=c_sums, concept_token_mean_sums=token_sums,
        token_mean_square_sum=token_square_sum, token_means=means, c_means=c_means, token_mean_variance=variance)
    K.save_array(out / 'fixed_anchor_tokens400.npy', anchors); guard()
    assert tensor_digest(p, half=True) == p_digest and all(tensor_digest(models[k], half=False) == digests[k] for k in models)
    result = dict(status='COMPLETE_T_ONLY_NATIVE_TOKEN_ANCHORS_NO_FIT_OR_IMAGES', UTC=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
        policy=K.ref(path), source=K.ref(__file__), T_concepts=1323, T_stimuli=13230, frozen_EEG_GPU_queries=26460,
        T_projector_queries=39690, all800_existing_native_P_replays_BIT_EXACT=True, T8_existing_EEG_GPU_replay_errors=errors8,
        T_all_global_c_means_vs_previous_CPU_errors=c_errors, EEG_models=model_pins, projection_weights=q['projection_weights'],
        all_frozen_weights_unchanged=True, new_optimizer_updates=0, new_models=0, new_research_images=0, UNet_calls=0,
        statistics=K.ref(out / 'T_anchor_statistics.npz'), fixed_future_tokens=K.ref(out / 'fixed_anchor_tokens400.npy'),
        anchor_names=['old_EEG_T_mean', 'new_EEG_T_mean', 'visual_T_mean'], means_fitted_T_only=True,
        Dcheck_targets_features_and_scores_NOT_used_for_moment_fit=True, Dcheck_existing_EEG_P_only_engineering_replay=True,
        future_inference='current EEG through new encoder and P; subtract within-query token mean, add fixed T mean; one EEG encoder and four UNet, no second query EEG encoder',
        no_future_image_study_admitted=True, classical_marginal_calibration_NOT_claimed_novel=True,
        seconds=time.monotonic() - started, peak_GPU_bytes=torch.cuda.max_memory_allocated(), extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)
    K.write(out / 'completion.json', result); print(json.dumps(result), flush=True)
except BaseException as exc:
    K.write(out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc(), partial_evidence_retained=True, no_automatic_retry=True)); raise
