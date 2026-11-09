"""Original CPU64 neural replay of fixed 16 x 15 inputs and all-T reductions."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import sys, time
from pathlib import Path
import numpy as np
import enigma_repeat_task_risk_contract_20261105_v1 as C
from enigma_repeat_task_risk_math_20261105_v1 import SUBSETS, task_metrics, summaries

started = time.monotonic()
c = C.setup(sys.argv[1])
q, out = c['q'], c['out']
assert C.ref(__file__) == q['sources']['CPU']
C.resources(c, starting=True)
done = C.read(out / 'scientific_completion.json')
assert done['status'] == 'COMPLETE_ALL13230_T_ALL15_SUBSETS_FROZEN_OOF_TASK_RISK'
a = np.load(C.checked(done['metrics']), allow_pickle=False)
f = np.load(C.checked(done['fixture']), allow_pickle=False)
assert a['per_view'].shape == (13230, 15, 5) and np.all(a['coverage'] == 1)
assert np.array_equal(a['T'], c['T']) and np.array_equal(a['folds'], c['folds'])
summary, means = summaries(a['per_view'], a['loss_of_mean'], a['dispersion'])
assert np.array_equal(means, a['count_means'])
saved = C.read(done['readout']['path'])
assert all(saved[k] == v for k, v in summary.items())
sys.path[:0] = q['existing_runtime_paths']
import torch
from enigma_native_reference_math_20261104_v1 import author_namespace
torch.set_num_threads(2)
scope = author_namespace(Path(q['original_models']['path']).read_text(), Path(q['original_training']['path']).read_text())
raw = np.load(q['physical_source']['path'], allow_pickle=True).item()['preprocessed_eeg_data']
assert np.array_equal(raw[f['rows']], f['raw'])
del raw
targets = torch.load(q['visual_target']['path'], map_location='cpu', weights_only=True, mmap=True).numpy()
assert np.array_equal(targets[(f['rows'] // 10)[:, None] * 10 + np.arange(10)], f['targets10'])
maximum = 0.
metric_error = 0.
replayed = 0
for role in ('oof0', 'oof1'):
    take = np.flatnonzero(f['roles'] == int(role[-1]))
    assert len(take) == 8
    assert np.all(c['folds'][f['positions'][take]] == int(role[-1]))
    state = torch.load(q['checkpoints'][role]['path'], map_location='cpu', weights_only=False, mmap=True)
    net = scope['ENIGMA'](63, 250, ['sub-01']).double().eval().requires_grad_(False)
    net.load_state_dict({k: v.double() if v.is_floating_point() else v for k, v in state['model'].items()}, strict=True)
    with torch.inference_mode():
        for j, subset in enumerate(SUBSETS):
            # Match the recorded source builder's FP32 average before CPU64 neural execution.
            x = np.ascontiguousarray(f['raw'][take][:, subset].mean(1)).astype(np.float64)
            p = net(torch.from_numpy(x), ['sub-01'] * len(take)).numpy()
            maximum = max(maximum, float(np.max(abs(p - f['predictions'][take, j]))))
            replayed += len(take)
            metrics = task_metrics(f['predictions'][take, j], f['targets10'][take], f['rows'][take] % 10)
            metric_error = max(metric_error, float(np.max(abs(metrics - a['per_view'][f['positions'][take], j]))))
    del state, net
assert maximum <= q['CPU64_fixture_max_absolute_error']
assert metric_error < 2e-12 and replayed == 240
C.write(out / 'independent_CPU_audit.json', dict(status='PASS_FIXED240_ORIGINAL_CPU64_FORWARDS_AND_ALL_T_REDUCTIONS',
    policy=C.ref(sys.argv[1]), scientific_completion=C.ref(out / 'scientific_completion.json'),
    CPU64_neural_forwards=replayed, neural_replay_is_fixed16_NOT_all198450=True,
    maximum_CPU64_condition_error=maximum, maximum_fixed1200_scalar_metric_error=metric_error,
    fixed_inputs_replayed_against_actual_SHA_pinned_physical_recordings=True,
    all_T_coverage_means_Jensen_risk_predictions_and_bootstraps_recomputed=True,
    all_T_neural_forward_replayed_on_CPU=False, original_weights_unchanged=True,
    seconds=time.monotonic() - started))
C.resources(c, starting=False)
