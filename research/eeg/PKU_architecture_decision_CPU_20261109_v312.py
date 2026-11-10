"""Full-T encoder-compression / visual-fusion decisions. Slurm CPU only.

Classical kernel regressions are controlled diagnostic instruments, not a new
architecture or a deployable visual-oracle method. No held-out answers are fit.
"""
import argparse
import csv
import gzip
import hashlib
import json
import os
import shutil
import socket
import time
import traceback
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
ap = argparse.ArgumentParser()
ap.add_argument('--experiment', choices=['compression', 'fusion'], required=True)
args = ap.parse_args()
R = Path('/dfs/share/maleiLab/wwt/eeg')
OUT = R / ('runs/EEG_architecture_' + args.experiment + '_v312')
OUT.mkdir(exist_ok=False)
START = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    p = OUT / name
    tmp = p.with_name(p.name + '.pending')
    with tmp.open('xb') as f:
        f.write((json.dumps(value, indent=2, allow_nan=False) + '\n').encode())
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


def guard():
    assert time.monotonic() - START < 1100, 'Finite CPU budget; retain all prefix files'
    assert shutil.disk_usage(OUT).free > 100 << 30
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < 2 << 30


try:
    import numpy as np
    import scipy.linalg as la
    import torch

    torch.set_num_threads(4)
    assert not torch.cuda.is_available(), 'This diagnostic must not reserve a GPU'
    spec_path = R / 'code/EEG_architecture_decision_v312.json'
    spec = json.loads(spec_path.read_text())
    assert spec['queries'] == 13230 and spec['seed'] == 1729
    grid = np.asarray(spec['ridge_grid'], dtype=np.float64)
    parent = R / 'runs/HIVE_native_sub03_5096542'
    bank_path = parent / 'T_true_OOF_conditions.pt'
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    split_path = R / 'data/recovery_20261108_v236/current_split_manifest.json'
    split = json.loads(split_path.read_text())
    rows = np.asarray(bank['array_rows'].tolist(), dtype=np.int64)
    roles = bank['folds'].numpy().astype(np.int64)
    assert rows.tolist() == split['image_rows']['T'] and len(rows) == 13230
    c = bank['condition'].numpy().copy()
    del bank
    raw_path = R / 'data/recovery_20261108_v236/sub-03_train_mean4.pt'
    assert sha(raw_path) == '85a25128f6b94e3739478c60e089d33f437a8dcbeb31cf136ce9772d2cc1615d'
    layout_path = R / 'data/recovery_20261108_v236/sub-03_layout.json'
    layout = json.loads(layout_path.read_text())
    native_policy = json.loads((R / 'code/HIVE_native_sub03_protocol_v239.json').read_text())
    electrodes = [layout['channels'].index(k) for k in native_policy['electrodes']]
    raw = torch.load(raw_path, weights_only=True, map_location='cpu', mmap=True)
    eeg = raw[rows][:, electrodes, :].reshape(13230, 4250).numpy().copy()
    del raw
    visual_path = R / 'data/native_training_v239/train_visual_embeddings.pt'
    vis = torch.load(visual_path, weights_only=True, map_location='cpu')
    views = {k: v[rows].numpy().copy() for k, v in sorted(vis.items())}
    del vis
    assert len(views) == 3 and sorted(v.shape[1] for v in views.values()) == [512, 1024, 1024]
    fused_path = parent / 'deterministic_F_all16540.pt'
    fused = torch.load(fused_path, weights_only=True, map_location='cpu')[rows].numpy().copy()
    assert c.shape == fused.shape == (13230, 1024)
    if args.experiment == 'compression':
        views['native_fused_F'] = fused
    manifest_path = R / 'data/full_T_recovery_20261108_v238/image_manifest.csv'
    with manifest_path.open() as f:
        lookup = {int(v['array_row']): v for v in csv.DictReader(f) if v['dataset_split'] == 'train'}
    concepts = np.asarray([lookup[int(r)]['concept_id'] for r in rows])
    keys = sorted(set(concepts))
    assert len(keys) == 1323 and all((concepts == k).sum() == 10 for k in keys)
    groups = {k: np.flatnonzero(concepts == k) for k in keys}
    same_indices = np.asarray([groups[k] for k in concepts])
    assignment = np.full(13230, -1, dtype=np.int64)
    for role in [0, 1]:
        role_keys = sorted(set(concepts[roles == role]), key=lambda k: hashlib.sha256(('1729|' + k).encode()).hexdigest())
        for index, key in enumerate(role_keys):
            ix = groups[key]
            assert np.all(roles[ix] == role)
            assignment[ix] = index % 3
    assert np.all(assignment >= 0)
    inputs = dict(c=c, raw17=eeg, F=fused)
    if args.experiment == 'compression':
        recipes = [('c', [('c', 'linear', False)], None),
                   ('raw17', [('raw17', 'linear', False)], None),
                   ('c_raw17', [('c', 'linear', False), ('raw17', 'linear', False)], None),
                   ('c_wrong', [('c', 'linear', True)], 'c'),
                   ('raw17_wrong', [('raw17', 'linear', True)], 'raw17')]
        contrasts = [('raw17', 'c'), ('c_raw17', 'c'), ('c', 'c_wrong'), ('raw17', 'raw17_wrong')]
    else:
        recipes = [('F_linear', [('F', 'linear', False)], None),
                   ('F_RBF', [('F', 'RBF', False)], None),
                   ('F_RBF_c', [('F', 'RBF', False), ('c', 'linear', False)], None),
                   ('F_RBF_raw17', [('F', 'RBF', False), ('raw17', 'linear', False)], None),
                   ('F_RBF_c_wrong', [('F', 'RBF', False), ('c', 'linear', True)], 'F_RBF_c'),
                   ('F_RBF_raw17_wrong', [('F', 'RBF', False), ('raw17', 'linear', True)], 'F_RBF_raw17')]
        contrasts = [('F_RBF', 'F_linear'), ('F_RBF_c', 'F_RBF'), ('F_RBF_raw17', 'F_RBF'),
                     ('F_RBF_c', 'F_RBF_c_wrong'), ('F_RBF_raw17', 'F_RBF_raw17_wrong'), ('F_RBF_raw17', 'F_RBF_c')]
    cases = [v[0] for v in recipes]
    view_names = list(views)
    dims = [views[n].shape[1] for n in view_names]
    boundaries = np.cumsum([0] + dims)
    y = np.concatenate([views[k] for k in view_names], axis=1).astype(np.float64)
    unit_views = {k: v.astype(np.float64) / np.maximum(np.linalg.norm(v.astype(np.float64), axis=1, keepdims=True), 1e-12) for k, v in views.items()}
    metric_names = ['MSE', 'variance_normalized_MSE', 'cosine', 'same_concept_margin', 'same_concept_strict_Top1', 'same_concept_2AFC']
    metrics = np.zeros((len(cases), len(views), 13230, len(metric_names)), dtype=np.float64)
    predictions = np.lib.format.open_memmap(OUT / 'all_T_predictions.npy', mode='w+', dtype=np.float32, shape=(len(cases), 13230, y.shape[1]))
    seen = np.zeros(13230, dtype=bool)

    def wrong_permutation(indices):
        permutation = np.arange(len(indices))
        local = concepts[indices]
        for key in sorted(set(local)):
            selected = np.flatnonzero(local == key)
            assert len(selected) == 10
            permutation[selected] = np.roll(selected, -1)
        assert np.all(concepts[indices][permutation] == local) and np.all(permutation != np.arange(len(indices)))
        return permutation

    def make_components(fit, test):
        cache = {}
        required = sorted({(source, kind) for _, recipe, _ in recipes for source, kind, _ in recipe})
        for source, kind in required:
            guard()
            a = inputs[source][fit].astype(np.float64)
            b = inputs[source][test].astype(np.float64)
            mean = a.mean(0)
            scale = np.maximum(a.std(0), 1e-3)
            a = (a - mean) / scale
            b = (b - mean) / scale
            k = (a @ a.T) / a.shape[1]
            kt = (b @ a.T) / a.shape[1]
            width = None
            if kind == 'RBF':
                aa = np.sum(a * a, axis=1) / a.shape[1]
                bb = np.sum(b * b, axis=1) / a.shape[1]
                d = np.maximum(aa[:, None] + aa[None, :] - 2 * k, 0)
                dt = np.maximum(bb[:, None] + aa[None, :] - 2 * kt, 0)
                probe = np.linspace(0, len(fit) - 1, min(512, len(fit)), dtype=int)
                square = d[np.ix_(probe, probe)]
                positive = square[square > 1e-12]
                assert len(positive)
                width = float(np.median(positive))
                k = np.exp(-d / width)
                kt = np.exp(-dt / width)
            cache[(source, kind)] = (k, kt, width)
        return cache

    def decomposition(recipe, components, permutation):
        k = None
        kt = None
        for source, kind, wrong in recipe:
            a, b, _ = components[(source, kind)]
            if wrong:
                a, b = a[np.ix_(permutation, permutation)], b[:, permutation]
            k = a.copy() if k is None else k + a
            kt = b.copy() if kt is None else kt + b
        k /= len(recipe)
        kt /= len(recipe)
        mean = k.mean(0)
        grand = float(mean.mean())
        k -= mean[:, None]
        k -= mean[None, :]
        k += grand
        kt = kt - kt.mean(1, keepdims=True) - mean[None, :] + grand
        k = (k + k.T) / 2
        eigenvalues, eigenvectors = la.eigh(k, check_finite=False, driver='evd', overwrite_a=True)
        assert eigenvalues.min() > -1e-7 * max(1., eigenvalues.max())
        return np.maximum(eigenvalues, 0), eigenvectors, kt

    # Independent finite fixture: centered linear-kernel and primal ridge agree.
    rng = np.random.default_rng(1729)
    xa = rng.normal(size=(13, 4)); xb = rng.normal(size=(7, 4)); ya = rng.normal(size=(13, 3))
    xm = xa.mean(0); ym = ya.mean(0); xac = xa - xm
    lam = .2
    primal = (xb - xm) @ la.solve(xac.T @ xac + 13 * lam * np.eye(4), xac.T @ (ya - ym)) + ym
    dual = ((xb - xm) @ xac.T) @ la.solve(xac @ xac.T + 13 * lam * np.eye(13), ya - ym) + ym
    fixture_error = float(np.max(np.abs(primal - dual)))
    assert fixture_error < 1e-12
    records = []
    for role in [0, 1]:
        for fold in range(3):
            guard()
            train = np.flatnonzero((roles == role) & (assignment != fold))
            test = np.flatnonzero((roles == role) & (assignment == fold))
            train_keys = sorted(set(concepts[train]), key=lambda k: hashlib.sha256(('inner312|1729|' + k).encode()).hexdigest())
            inner_keys = set(train_keys[1::4])
            iv = np.asarray([i for i in train if concepts[i] in inner_keys])
            it = np.asarray([i for i in train if concepts[i] not in inner_keys])
            assert set(concepts[it]).isdisjoint(concepts[iv]) and set(concepts[train]).isdisjoint(concepts[test])
            assert len(it) + len(iv) == len(train) and not seen[test].any()
            selected = {}
            inner_losses = {}
            for phase, fit, evaluate in [('inner', it, iv), ('outer', train, test)]:
                components = make_components(fit, evaluate)
                permutation = wrong_permutation(fit)
                ym = y[fit].mean(0)
                centered = y[fit] - ym
                view_scales = [max(float(np.mean(centered[:, boundaries[j]:boundaries[j+1]] ** 2)), 1e-12) for j in range(len(views))]
                for ci, (case, recipe, inherit) in enumerate(recipes):
                    if phase == 'inner' and inherit:
                        selected[case] = selected[inherit].copy()
                        continue
                    guard()
                    ev, u, kt = decomposition(recipe, components, permutation)
                    rhs = u.T @ centered
                    transfer = kt @ u
                    if phase == 'inner':
                        losses = np.zeros((len(grid), len(views)))
                        for gi, ridge in enumerate(grid):
                            p = ym + transfer @ (rhs / (ev[:, None] + len(fit) * ridge))
                            for vi in range(len(views)):
                                s = slice(boundaries[vi], boundaries[vi+1])
                                losses[gi, vi] = np.mean((p[:, s] - y[evaluate, s]) ** 2) / view_scales[vi]
                        # Exact ties choose the larger penalty, never the outer answer.
                        best = [int(np.flatnonzero(losses[:, vi] == losses[:, vi].min())[-1]) for vi in range(len(views))]
                        selected[case] = [float(grid[j]) for j in best]
                        inner_losses[case] = losses.tolist()
                    else:
                        for vi, name in enumerate(view_names):
                            s = slice(boundaries[vi], boundaries[vi+1])
                            ridge = selected[case][vi]
                            p = (ym[s] + transfer @ (rhs[:, s] / (ev[:, None] + len(fit) * ridge))).astype(np.float32)
                            assert np.isfinite(p).all()
                            predictions[ci, test, s] = p
                            p = p.astype(np.float64)
                            error = np.mean((p - y[test, s]) ** 2, axis=1)
                            unit = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
                            truth = unit_views[name]
                            cosine = np.sum(unit * truth[test], axis=1)
                            candidate = truth[same_indices[test]]
                            scores = np.einsum('nd,nkd->nk', unit, candidate)
                            correct_slot = np.argmax(same_indices[test] == test[:, None], axis=1)
                            mask = np.ones_like(scores, dtype=bool)
                            mask[np.arange(len(test)), correct_slot] = False
                            actual = scores[np.arange(len(test)), correct_slot]
                            assert np.max(np.abs(actual - cosine)) < 1e-12
                            negatives = scores[mask].reshape(len(test), 9)
                            margin = actual - negatives.max(1)
                            afc = ((actual[:, None] > negatives) + .5 * (actual[:, None] == negatives)).mean(1)
                            metrics[ci, vi, test] = np.stack([error, error / view_scales[vi], cosine, margin, (margin > 0).astype(float), afc], axis=1)
                        write('progress.json', dict(experiment=args.experiment, role=role, fold=fold, case=case, phase=phase, complete_test_queries=int(seen.sum()), current_outer_queries=len(test), seconds=time.monotonic()-START, image_effect=False))
                    del ev, u, kt, rhs, transfer
                if phase == 'outer':
                    widths = {s + ':' + k: w for (s, k), (_, _, w) in components.items()}
                del components
            predictions.flush()
            seen[test] = True
            record = dict(role=role, fold=fold, outer_fit_rows=rows[train].tolist(), outer_test_rows=rows[test].tolist(), inner_fit_rows=rows[it].tolist(), inner_validation_rows=rows[iv].tolist(), lambda_by_case_view=selected, inner_losses_by_lambda_view=inner_losses, outer_RBF_widths=widths, seconds=time.monotonic()-START)
            write('role' + str(role) + '_fold' + str(fold) + '.json', record)
            records.append(record)
            print(json.dumps(dict(role=role, fold=fold, covered=int(seen.sum()), seconds=time.monotonic()-START)), flush=True)
    assert seen.all() and np.isfinite(metrics).all() and len(records) == 6
    predictions.flush()
    with gzip.open(OUT / 'all_T_metrics.csv.gz', 'wt', newline='') as f:
        w = csv.writer(f)
        w.writerow(['case', 'view', 'array_row', 'concept_id', 'source_encoder_role', 'outer_fold'] + metric_names)
        for ci, case in enumerate(cases):
            for vi, view in enumerate(view_names):
                for i in range(13230):
                    w.writerow([case, view, int(rows[i]), concepts[i], int(roles[i]), int(assignment[i])] + metrics[ci, vi, i].tolist())
    means = {case: {view: {metric: float(metrics[ci, vi, :, mi].mean()) for mi, metric in enumerate(metric_names)} for vi, view in enumerate(view_names)} for ci, case in enumerate(cases)}
    rng = np.random.default_rng(1729)
    resamples = rng.integers(0, len(keys), size=(1000, len(keys)))
    differences = []
    for left, right in contrasts:
        delta = metrics[cases.index(left)] - metrics[cases.index(right)]
        for vi, view in enumerate(view_names):
            grouped = np.asarray([delta[vi, groups[key]].mean(0) for key in keys])
            intervals = np.quantile(grouped[resamples].mean(1), [.025, .975], axis=0)
            for mi, name in enumerate(metric_names):
                differences.append(dict(left=left, right=right, view=view, metric=name, mean=float(delta[vi, :, mi].mean()), concept95=[float(v) for v in intervals[:, mi]]))
    guard()
    completion = dict(status='COMPLETE_ALL13230_SIX_SOURCE_ROLE_CONCEPT_ISOLATED_FITS_' + args.experiment.upper(), experiment=args.experiment, CPU_job=os.environ['SLURM_JOB_ID'], seconds=time.monotonic()-START, queries=13230, concepts=1323, fits=6, subjects=1, repeat='R4', cases=cases, view_dimensions={k:int(v.shape[1]) for k,v in views.items()}, means=means, comparisons=differences,
        fixture_primal_dual_error=fixture_error, source_role_model_not_fit_on_test_concepts=True, no_test_normalization_or_lambda_selection=True, full_gallery_benchmark=False, privileged_F_input=(args.experiment=='fusion'), new_GPU_jobs=0, new_images=0, new_neural_components=0, ordinary_changed=False, novel_method_claim=False, information_theorem_claim=False,
        input_SHA256={n:sha(p) for n,p in [('OOF',bank_path),('raw17_source',raw_path),('visual_inputs',visual_path),('fused_F',fused_path),('split',split_path),('image_identity_manifest',manifest_path),('policy',spec_path),('source',Path(__file__))]}, table_SHA256=sha(OUT/'all_T_metrics.csv.gz'), predictions_SHA256=sha(OUT/'all_T_predictions.npy'), output_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()), multiplicity='All predefined contrasts reported; intervals descriptive, no significance-selected architecture or automatic follow-up fits')
    write('completion.json', completion)
    print(json.dumps(dict(status=completion['status'], means=means, seconds=completion['seconds'])), flush=True)
except BaseException as e:
    write('failure.json', dict(status='FAILED_PRESERVED_NO_AUTO_RETRY', error=repr(e), traceback=traceback.format_exc(), seconds=time.monotonic()-START))
    raise
