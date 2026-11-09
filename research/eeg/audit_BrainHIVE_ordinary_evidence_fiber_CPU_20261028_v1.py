"""Independent CPU replay of D29 banks, spherical constraints and all metrics.

No EEG model fitting, raw EEG forward pass, generator, CUDA operation, Dcheck or
test-value access occurs here.  The protected endpoint is re-derived with NumPy
QR and a separate spherical-cap construction, without importing its producer.
"""
import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import numpy as np

N, D, K = 13230, 1024, 32
ARMS = ('evidence_fiber', 'permuted_fiber', 'old_ordinary')
METRICS = ('F_MSE', 'same_concept_Top1', 'same_concept_2AFC')
CP_SHA = 'a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
ORDINARY_SOURCE_SHA = '2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7'
ALPHA = math.atan(.25)


def need(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_bytes())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(value):
    need(isinstance(value, dict) and 'path' in value and 'sha256' in value, 'Missing path/SHA source reference')
    p = Path(value['path'])
    need(p.is_file() and sha(p) == value['sha256'], 'Source SHA mismatch: ' + str(p))
    if 'bytes' in value:
        need(p.stat().st_size == value['bytes'], 'Source byte count mismatch')
    return p


def json_new(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def peak_rss():
    status = Path('/proc/self/status')
    need(status.exists(), 'Real source audit requires the Linux peak-RSS interface')
    for line in status.read_text().splitlines():
        if line.startswith('VmHWM:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('Missing Linux VmHWM')


def groups_for(index, partition):
    group = {}
    for i, row in enumerate(index):
        need(row['row'] == i and row['partition'] == partition and row['subject_id'] == 'sub-01', 'Incorrect compact row/partition/subject')
        group.setdefault(row['concept_id'], []).append(i)
    need(len({row['array_row'] for row in index}) == len(index), 'Duplicate physical row')
    need(len({row['stimulus_id'] for row in index}) == len(index), 'Duplicate stimulus identity')
    need(all(len(rows) == 10 for rows in group.values()), 'Incomplete ten-stimulus concept')
    if partition == 'T':
        need(len(index) == N and len(group) == 1323, 'Incomplete T denominator')
        need(all(len({index[i]['fold'] for i in rows}) == 1 for rows in group.values()), 'T concept crosses folds')
        need({row['fold'] for row in index} == {0, 1}, 'Missing OOF fold')
    else:
        need(len(index) == 820 and len(group) == 82, 'Incomplete Dfit denominator')
    return group


def angles(left, right):
    den = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    cosine = np.ones(len(left), np.float64)
    np.divide(np.sum(left * right, axis=1), den, out=cosine, where=den > 0.)
    return np.arccos(np.clip(cosine, -1., 1.))


def qr_basis(H):
    """QR supplies an independent basis rather than the producer's row SVD."""
    rows = np.concatenate((np.asarray(H, np.float64), np.ones((1, D)) / math.sqrt(D)))
    columns, upper = np.linalg.qr(rows.T, mode='reduced')
    need(columns.shape == (D, K + 1) and np.min(np.abs(np.diag(upper))) > 1e-9, 'Protected H plus mean lacks rank33')
    return columns


def independently_project(c, old, weight, bias, H):
    """Nearest fiber endpoint with the documented non-nearest antipodal fallback."""
    need(c.dtype == old.dtype == np.float32, 'Expected stored float32 endpoint banks')
    u32 = (c - bias) / weight
    old32 = (old - bias) / weight
    u, target = u32.astype(np.float64), old32.astype(np.float64)
    basis = qr_basis(H)
    fixed = (u @ basis) @ basis.T
    base_null = u - fixed
    target_null = target - (target @ basis) @ basis.T
    radius = np.linalg.norm(u, axis=1)
    r = np.linalg.norm(base_null, axis=1)
    rt = np.linalg.norm(target_null, axis=1)
    scale = np.maximum(radius, 1.)
    tiny = np.finfo(np.float64).tiny
    q0 = base_null / np.maximum(r[:, None], tiny)
    qt = target_null / np.maximum(rt[:, None], tiny)
    dot = np.clip(np.sum(q0 * qt, axis=1), -1., 1.)
    desired = np.arccos(dot)
    max_null = np.arccos(np.clip(1. - radius**2 * (1. - math.cos(ALPHA)) / np.maximum(r**2, tiny), -1., 1.))
    tangent = qt - dot[:, None] * q0
    tangent_length = np.linalg.norm(tangent, axis=1)
    unit_tangent = tangent / np.maximum(tangent_length[:, None], tiny)
    constrained = desired > max_null + 1e-12
    arc = np.minimum(desired, max_null)
    result_direction = np.where(constrained[:, None], np.cos(arc)[:, None] * q0 + np.sin(arc)[:, None] * unit_tangent, qt)
    v = fixed + r[:, None] * result_direction
    degenerate = ((radius <= 1e-12) | (r <= 1e-12 * scale) | (rt <= 1e-12 * scale))
    antipodal = constrained & (tangent_length <= 1e-12) & ~degenerate
    already = np.linalg.norm((target - u) @ basis, axis=1) <= 2e-8 * scale
    already &= np.abs(np.linalg.norm(target, axis=1) - radius) <= 2e-8 * scale
    already &= angles(u, target) <= ALPHA + 2e-8
    unchanged = np.all(old32 == u32, axis=1)
    kept = degenerate | antipodal | unchanged
    v32 = v.astype(np.float32)
    v32[already] = old32[already]
    v32[kept] = u32[kept]
    result = c + weight * (v32 - u32)
    result[already] = old[already]
    result[kept] = c[kept]
    return result, dict(non_nearest_antipodal_rows=int(antipodal.sum()), kept_original_rows=int(kept.sum()))


def independently_ordinary(c, weight, bias, state, torch):
    """Use primitive CPU linear/GELU math, never the trained component class."""
    F = torch.nn.functional
    x = torch.from_numpy(np.array(c, copy=True))
    w, b = torch.from_numpy(np.array(weight, copy=True)), torch.from_numpy(np.array(bias, copy=True))
    with torch.inference_mode():
        u = (x - b) / w
        h = F.gelu(F.linear(u, state['trunk.0.weight'], state['trunk.0.bias']))
        d = F.linear(h, state['direction.weight'], state['direction.bias'])
        d = d - d.mean(-1, keepdim=True)
        norm = torch.linalg.vector_norm(u, dim=-1, keepdim=True)
        r2 = norm.square()
        d = d - (d * u).sum(-1, keepdim=True) / r2 * u
        step = .25 * r2.sqrt() * d / (r2 + d.square().sum(-1, keepdim=True)).sqrt()
        rescale = norm / torch.linalg.vector_norm(u + step, dim=-1, keepdim=True)
        return (x + w * ((rescale - 1.) * u + rescale * step)).numpy()


def audit_endpoints(native, bank, weight, bias, H, Hperm, ordinary_state, torch, tick):
    worst = dict(old_ordinary_relative_L2=0., projected_max_absolute=0., protected_absolute=0., norm_absolute=0., global_angle=0.)
    flags = [dict(non_nearest_antipodal_rows=0, kept_original_rows=0) for _ in range(2)]
    per_query = np.empty((len(native), 2, 3), np.float64)
    bases = (qr_basis(H), qr_basis(Hperm))
    for lo in range(0, len(native), 256):
        tick()
        c = np.asarray(native[lo:lo + 256])
        actual = np.asarray(bank[lo:lo + 256])
        old = independently_ordinary(c, weight, bias, ordinary_state, torch)
        relative = np.linalg.norm((old - actual[:, 2]).astype(np.float64), axis=1) / np.maximum(np.linalg.norm(actual[:, 2].astype(np.float64), axis=1), 1e-12)
        worst['old_ordinary_relative_L2'] = max(worst['old_ordinary_relative_L2'], float(relative.max()))
        need(relative.max() < 3e-5, 'Independent CPU old ordinary replay mismatch')
        u = ((c - bias) / weight).astype(np.float64)
        for arm, Hitem in enumerate((H, Hperm)):
            expected, notes = independently_project(c, actual[:, 2], weight, bias, Hitem)
            discrepancy = np.abs(expected.astype(np.float64) - actual[:, arm]).max()
            worst['projected_max_absolute'] = max(worst['projected_max_absolute'], float(discrepancy))
            need(discrepancy < 3e-5, 'Independent spherical endpoint replay mismatch')
            v = ((actual[:, arm] - bias) / weight).astype(np.float64)
            protected = np.linalg.norm((v - u) @ bases[arm], axis=1)
            norm_error = np.abs(np.linalg.norm(v, axis=1) - np.linalg.norm(u, axis=1))
            angle = angles(u, v)
            need(protected.max() < 2e-4 and norm_error.max() < 2e-4 and angle.max() < ALPHA + 3e-5, 'Stored endpoint violates fixed spherical constraints')
            per_query[lo:lo + len(c), arm] = np.stack((protected, norm_error, angle), axis=1)
            worst['protected_absolute'] = max(worst['protected_absolute'], float(protected.max()))
            worst['norm_absolute'] = max(worst['norm_absolute'], float(norm_error.max()))
            worst['global_angle'] = max(worst['global_angle'], float(angle.max()))
            for key in notes:
                flags[arm][key] += notes[key]
    return dict(maxima=worst, fallbacks=flags, rows=len(native)), per_query


def metrics_one_concept(bank, target):
    pred, target = np.asarray(bank, np.float64), np.asarray(target, np.float64)
    pn, tn = np.linalg.norm(pred, axis=-1, keepdims=True), np.linalg.norm(target, axis=-1, keepdims=True)
    unit_pred = np.divide(pred, pn, out=np.zeros_like(pred), where=pn > 0.)
    unit_target = np.divide(target, tn, out=np.zeros_like(target), where=tn > 0.)
    score = np.einsum('iad,jd->iaj', unit_pred, unit_target, optimize=True)
    truth = score[np.arange(10), :, np.arange(10)]
    maximum = score.max(-1)
    top1 = (truth == maximum) / (score == maximum[..., None]).sum(-1)
    afc = ((truth[..., None] > score).sum(-1) + .5 * (truth[..., None] == score).sum(-1) - .5) / 9.
    mse = ((pred - target[:, None])**2).mean(-1)
    return np.stack((mse, top1, afc), axis=-1)


def complete_metrics(bank, target, group, saved, tick):
    query = np.empty((len(bank), 3, 3), np.float64)
    for ci, rows in enumerate(group.values()):
        if ci % 32 == 0:
            tick()
        query[rows] = metrics_one_concept(bank[rows], target[rows])
    concept = np.stack([query[rows].mean(0) for rows in group.values()])
    difference = np.stack((concept[:, 0] - concept[:, 2], concept[:, 0] - concept[:, 1]), axis=1)
    rng = np.random.default_rng(1729)
    draws = np.stack([difference[rng.integers(0, len(group), size=len(group))].mean(0) for _ in range(1000)])
    lower, upper = np.quantile(draws, [.025, .975], axis=0)
    arrays = dict(per_query=query, per_concept=concept, means=concept.mean(0), comparison_means=difference.mean(0), comparison_descriptive95_lower=lower, comparison_descriptive95_upper=upper)
    errors = {key: float(np.abs(value - saved[key]).max()) for key, value in arrays.items()}
    need(max(errors.values()) < 1e-12, 'Saved query/concept/bootstrap metrics mismatch: ' + repr(errors))
    return dict(metrics=list(METRICS), arms=list(ARMS), concepts=len(group), queries=len(bank), means=arrays['means'].tolist(), comparison_order=['evidence_minus_old_ordinary', 'evidence_minus_permuted'], comparison_means=arrays['comparison_means'].tolist(), comparison_descriptive95_lower=lower.tolist(), comparison_descriptive95_upper=upper.tolist(), maximum_saved_metric_error=max(errors.values())), dict(**arrays, bootstrap_paired=draws)


def self_check():
    truth = np.eye(10, D)
    p = np.repeat(truth[:, None], 3, axis=1)
    perfect = metrics_one_concept(p, truth)
    opposite = metrics_one_concept(-p, truth)
    ties = metrics_one_concept(np.zeros_like(p), truth)
    need(np.all(perfect[..., 0] == 0.) and np.all(perfect[..., 1:] == 1.), 'Perfect metric identity failed')
    need(np.all(opposite[..., 1:] == 0.), 'Opposite metric identity failed')
    need(np.all(ties[..., 1] == .1) and np.all(ties[..., 2] == .5), 'Tie metric identity failed')
    rng = np.random.default_rng(17)
    H = np.linalg.qr(rng.normal(size=(D, K)))[0].T
    u = rng.normal(size=(3, D)).astype(np.float32)
    # A feasible positive rotation in ker([H;mean]) has a known exact endpoint.
    basis = qr_basis(H)
    pfix = (u.astype(np.float64) @ basis) @ basis.T
    n0 = u - pfix
    tangent = rng.normal(size=u.shape)
    tangent -= (tangent @ basis) @ basis.T
    tangent -= (tangent * n0).sum(1)[:, None] / (n0 * n0).sum(1)[:, None] * n0
    tangent *= np.linalg.norm(n0, axis=1)[:, None] / np.linalg.norm(tangent, axis=1)[:, None]
    old = (pfix + math.cos(.05) * n0 + math.sin(.05) * tangent).astype(np.float32)
    result, notes = independently_project(u, old, np.ones(D, np.float32), np.zeros(D, np.float32), H)
    need(np.max(np.abs(result - old)) < 1e-5 and notes['non_nearest_antipodal_rows'] == 0, 'Feasible spherical endpoint replay failed')
    return dict(perfect=True, opposite=True, fractional_ties=True, independently_constructed_feasible_sphere=True)


def run(completion_path, output, maximum_seconds=180):
    start = time.monotonic()
    base_rss = peak_rss()
    need(not output.exists(), 'Audit output already exists; preserve old attempts')
    output.mkdir(parents=True)
    stage = 'INPUTS'
    try:
        done_ref = ref(completion_path)
        done = read(completion_path)
        need(done['status'] == 'D29_ALL3_ROLE_H32_FIT_AND_TRUE_OOF_T_DFIT_CONDITIONS_EXPORTED', 'Incomplete D29 producer')
        for key in ('all_native_and_ordinary_weights_frozen', 'all_original_affines_unchanged'):
            need(done.get(key) is True, 'Missing frozen-source declaration')
        for key in ('original_ordinary_checkpoint_changed', 'formal_D_resumed', 'automatic_image_admission', 'image_admission_written', 'Dcheck_conditions_or_target_values_read'):
            need(done.get(key) is False, 'Unexpected permission or mutation: ' + key)
        need(tuple(done['arms']) == ARMS and done['H_rank'] == K and done['protected_rank_including_mean'] == K + 1 and done['SGD_updates'] == 0, 'D29 fixed arms/rank changed')
        need(0. <= done['max_mean4_cache_error'] < 3e-5, 'Native cache replay tolerance failed')
        keys = ('source', 'spec', 'H_bundle', 'ordinary_freeze', 'native_input_contract', 'trials', 'index', 'split', 'manifest', 'targets', 'OOF_T_conditions', 'OOF_T_native', 'Dfit_conditions', 'Dfit_native', 'Dfit_index', 'target_T', 'target_Dfit', 'coordinate_permutation', 'metrics', 'reproducibility', 'endpoint_proofs')
        paths = {key: checked(done[key]) for key in keys}
        spec = read(paths['spec'])
        need(spec.get('status') == 'D29_TONLY_EEG_EVIDENCE_FIBER_ORDINARY_VERSION_LOCKED_BEFORE_FIT' and spec.get('fixed_rank') == K, 'Registered fixed D29 hypothesis differs')
        pins = spec['source_pins']
        for item in pins:
            checked(item)
        pinned = {(str(Path(item['path']).resolve()), item['sha256']) for item in pins}
        for key in ('source', 'native_input_contract', 'trials', 'index', 'split', 'manifest', 'targets'):
            need((str(paths[key].resolve()), done[key]['sha256']) in pinned, 'Unregistered source: ' + key)
        for key in ('ordinary_freeze', 'native_input_contract', 'trials', 'index'):
            need(done[key] == spec[key], 'Producer input binding differs from registered spec: ' + key)
        frozen = read(paths['ordinary_freeze'])
        need(frozen['checkpoint']['sha256'] == CP_SHA and frozen['source']['sha256'] == ORDINARY_SOURCE_SHA and frozen['selected_strength'] == 1., 'Original ordinary baseline identity changed')
        checked(frozen['checkpoint']); checked(frozen['source'])
        # Only CPU tensor deserialisation is used.  Never import producer code.
        import torch
        torch.set_num_threads(2)
        need(not torch.cuda.is_initialized(), 'CUDA must not be initialised by this CPU audit')
        def tick():
            need(time.monotonic() - start < maximum_seconds, 'Finite CPU audit deadline')
            need(not torch.cuda.is_initialized(), 'Unexpected CUDA initialisation')
            need(peak_rss() - base_rss < 10 << 30, 'Finite additional CPU memory budget')
        checks = self_check()
        payload = torch.load(frozen['checkpoint']['path'], map_location='cpu', weights_only=False)
        need(payload['arm'] == 'ordinary' and payload['calibration']['selected_strength'] == 1., 'Ordinary payload identity')
        state = payload['model']
        need(set(state) == {'trunk.0.weight', 'trunk.0.bias', 'direction.weight', 'direction.bias'}, 'Ordinary state shape or risk branch changed')
        bundle = torch.load(paths['H_bundle'], map_location='cpu', weights_only=True)
        need(bundle['rank'] == K and set(bundle['roles']) == {'full', 'oof0', 'oof1'}, 'Wrong analytic role bundle')
        index = [json.loads(line) for line in paths['index'].read_text(encoding='utf-8').splitlines()]
        dindex = [json.loads(line) for line in paths['Dfit_index'].read_text(encoding='utf-8').splitlines()]
        tg, dg = groups_for(index, 'T'), groups_for(dindex, 'Dfit')
        contract, split = read(paths['native_input_contract']), read(paths['split'])
        metadata = {int(r['array_row']): r for r in csv.DictReader(paths['manifest'].open(encoding='utf-8')) if r['dataset_split'] == 'train'}
        need({r['array_row'] for r in index} == set(split['image_rows']['T']), 'T index differs from original concept split')
        for row in index + dindex:
            original_row = metadata[row['array_row']]
            need(row['concept_id'] == original_row['concept_id'] and row['stimulus_id'] == original_row['stimulus_id'], 'Saved index concept/stimulus identity differs from original manifest')
        need(set(dg) == set(split['d_fit_concepts']), 'Dfit index differs from original concept split')
        need(contract['subjects'][0]['subject'] == 'sub-01', 'Original subject route mismatch')
        sources = {s['role']: s for s in contract['subjects'][0]['sources']}
        need(set(sources) == set(bundle['roles']), 'Original checkpoint role set mismatch')
        all_native = np.load(paths['OOF_T_native'], mmap_mode='r', allow_pickle=False)
        all_bank = np.load(paths['OOF_T_conditions'], mmap_mode='r', allow_pickle=False)
        dnative = np.load(paths['Dfit_native'], mmap_mode='r', allow_pickle=False)
        dbank = np.load(paths['Dfit_conditions'], mmap_mode='r', allow_pickle=False)
        target_t = np.load(paths['target_T'], mmap_mode='r', allow_pickle=False)
        target_d = np.load(paths['target_Dfit'], mmap_mode='r', allow_pickle=False)
        need(all_native.shape == target_t.shape == (N, D) and all_bank.shape == (N, 3, D), 'Incomplete T banks')
        need(dnative.shape == target_d.shape == (820, D) and dbank.shape == (820, 3, D), 'Incomplete Dfit banks')
        for arr in (all_native, all_bank, dnative, dbank, target_t, target_d):
            need(arr.dtype == np.float32 and np.isfinite(arr).all(), 'Nonfinite or wrong-dtype condition/target bank')
        permutation = np.load(paths['coordinate_permutation'], allow_pickle=False)
        need(np.array_equal(permutation, np.random.default_rng(1729).permutation(D)), 'Fixed coordinate permutation changed')
        target_original = torch.load(paths['targets'], map_location='cpu', weights_only=True, mmap=True)
        need(np.array_equal(target_t, target_original[[r['array_row'] for r in index]].numpy()), 'T target source order differs')
        need(np.array_equal(target_d, target_original[[r['array_row'] for r in dindex]].numpy()), 'Dfit target source order differs')
        del target_original
        proof, repro_audits = {}, {}
        t_preserve = np.empty((N, 2, 3), np.float64)
        d_preserve = None
        coverage = np.zeros(N, np.int8)
        repro_doc = read(paths['reproducibility'])
        for role in ('full', 'oof0', 'oof1'):
            stage = 'ROLE_' + role; tick()
            row, source = bundle['roles'][role], sources[role]
            for kind in ('checkpoint', 'condition_cache', 'parent_registration'):
                expected = dict(path=source[kind + '_path'], sha256=source[kind + '_sha256'])
                need(row[kind] == expected and done['source_models'][role][kind] == expected, 'Changed source identity: ' + role + '/' + kind)
                checked(expected)
            parent = read(row['parent_registration']['path'])
            fit = list(range(N)) if role == 'full' else [r['row'] for r in index if r['fold'] != int(role[-1])]
            held = [] if role == 'full' else [r['row'] for r in index if r['fold'] == int(role[-1])]
            need(row['fit_positions'] == fit and row['held_positions'] == held, 'Role-private fit/held rows changed')
            need(set(parent['training_array_rows']) == {index[i]['array_row'] for i in fit} and set(parent['excluded_oof_rows']) == {index[i]['array_row'] for i in held}, 'H fitting permission differs from native role')
            original = torch.load(row['checkpoint']['path'], map_location='cpu', weights_only=True)['model']
            need(torch.equal(original['block.layers.0.norm.weight'], row['affine_weight']) and torch.equal(original['block.layers.0.norm.bias'], row['affine_bias']), 'Actual output affine changed')
            weight, bias = row['affine_weight'].numpy(), row['affine_bias'].numpy()
            need(np.all(np.abs(weight) > 1e-6), 'Invalid original affine')
            H, Hp = row['H'].numpy(), row['H_permuted'].numpy()
            need(H.shape == Hp.shape == (K, D) and np.array_equal(Hp, H[:, permutation]), 'Protected/permuted basis differs')
            need(np.max(np.abs(H @ H.T - np.eye(K))) < 1e-9, 'H rows are not orthonormal')
            signal, total = row['fit_signal_cross_moment'].numpy(), row['fit_total_moment'].numpy()
            eig = row['signal_eigenvalues_descending'].numpy()
            need(signal.shape == total.shape == (D, D) and eig.shape == (D,), 'Moment/spectrum shape mismatch')
            need(np.isfinite(signal).all() and np.isfinite(total).all() and np.isfinite(eig).all(), 'Nonfinite moments')
            need(np.max(np.abs(signal - signal.T)) < 1e-10 and np.max(np.abs(total - total.T)) < 1e-10, 'Asymmetric saved moments')
            need(np.all(eig[:-1] >= eig[1:]) and np.all(eig[:K] > 0.), 'Not fixed32 positive leading eigenvectors')
            eigen_residual = float(np.max(np.abs(H @ signal - eig[:K, None] * H)))
            need(eigen_residual < 1e-8, 'H does not diagonalise the registered leading signal directions')
            need(np.isclose(eig.sum(), np.trace(signal), rtol=1e-10, atol=1e-8) and np.isclose((eig * eig).sum(), (signal * signal).sum(), rtol=1e-10, atol=1e-8), 'Stored full spectrum is inconsistent')
            for key, h in (('Q_with_mean', H), ('Q_permuted_with_mean', Hp)):
                q = row[key].numpy(); qr = qr_basis(h)
                need(q.shape == (K + 1, D) and np.max(np.abs(q @ q.T - np.eye(K + 1))) < 1e-9, 'Protected mean basis rank/orthogonality mismatch')
                need(np.linalg.norm(q.T @ q - qr @ qr.T) < 1e-7, 'Independent QR and saved protected subspace differ')
            cache = torch.load(row['condition_cache']['path'], map_location='cpu', weights_only=True, mmap=True)
            need(cache['model_sha256'] == row['checkpoint']['sha256'], 'Original condition cache model mismatch')
            if held:
                selected = held
                routes = source['groups']['component_train']
                route = {r['array_row']: r['cache_row'] for r in routes}
                need(set(route) == {index[i]['array_row'] for i in held}, 'Original OOF route coverage mismatch')
                expected = cache['condition'][[route[index[i]['array_row']] for i in held]].numpy()
                need(all(int(cache['array_rows'][route[index[i]['array_row']]]) == index[i]['array_row'] for i in held), 'OOF original cache row metadata differs')
                native = np.asarray(all_native[held]); bank = np.asarray(all_bank[held])
                relative = np.linalg.norm((native - expected).astype(np.float64), axis=1) / np.maximum(np.linalg.norm(expected.astype(np.float64), axis=1), 1e-12)
                need(relative.max() < 3e-5, 'Complete CPU OOF cache identity replay failed')
                coverage[held] += 1
                endpoint, preservation = audit_endpoints(native, bank, weight, bias, H, Hp, state, torch, tick)
                t_preserve[held] = preservation
                item = repro_doc[role]
                need(item['scope'] == 'held_concepts_only', 'OOF reproducibility must use held concepts')
                raw_repro = np.load(checked(item['all_concepts']), allow_pickle=False)
                cross_total = raw_repro['per_concept_cross_and_total']
                need(cross_total.shape == (len(held) // 10, 2, 2), 'Held reproducibility denominator mismatch')
                expected_concepts = list(dict.fromkeys(index[i]['concept_id'] for i in held))
                need(raw_repro['concepts'].tolist() == row['held_concepts'] == expected_concepts, 'Held concept identity/order differs')
                ratio = np.divide(cross_total[..., 0], cross_total[..., 1], out=np.zeros(cross_total.shape[:2]), where=cross_total[..., 1] > 0.)
                need(np.allclose(ratio, raw_repro['per_concept_ratios'], rtol=1e-13, atol=1e-13), 'Held ratios do not match energies')
                held_signal, held_total = row['held_signal_cross_moment'].numpy(), row['held_total_moment'].numpy()
                trace_values = np.array([[(h * (h @ held_signal)).sum(), (h * (h @ held_total)).sum()] for h in (H, Hp)])
                need(np.allclose(trace_values, cross_total.mean(0), rtol=1e-10, atol=1e-9), 'Held per-concept and full-moment traces differ')
                rng = np.random.default_rng(1729)
                delta = ratio[:, 0] - ratio[:, 1]
                boot = np.array([delta[rng.integers(0, len(delta), size=len(delta))].mean() for _ in range(1000)])
                repro_audits[role] = dict(scope='held_concepts_only', concepts=len(delta), candidate_measurement_not_physiology=True, mean_cross_total=trace_values.tolist(), correct_minus_permuted_mean_concept_ratio=float(delta.mean()), descriptive95=np.quantile(boot, [.025, .975]).tolist())
            else:
                selected = []
                routes = source['groups']['calibration_only_Dfit']
                need([r['array_row'] for r in routes] == [r['array_row'] for r in dindex], 'Dfit source identity/order differs')
                expected = cache['condition'][[r['cache_row'] for r in routes]].numpy()
                need(all(int(cache['array_rows'][r['cache_row']]) == r['array_row'] for r in routes), 'Dfit original cache row metadata differs')
                need(np.array_equal(dnative, expected), 'Dfit native conditions differ from original cache bytes')
                endpoint, d_preserve = audit_endpoints(dnative, dbank, weight, bias, H, Hp, state, torch, tick)
                need(repro_doc[role]['scope'] == 'FIT_ONLY_NO_HELD_T_CONCEPTS', 'Full role cannot claim held T evidence')
                repro_audits[role] = dict(scope='FIT_ONLY_NO_HELD_T_CONCEPTS', independent_confirmation=False)
            proof[role] = dict(**endpoint, eigenvector_max_absolute_residual=eigen_residual, H_fit_queries=len(fit), H_held_queries=len(held), all_original_source_SHA_verified=True, native_cache_identity_rows=len(dindex) if role == 'full' else len(held))
            del cache, original
        need(np.all(coverage == 1), 'Every T stimulus must be routed exactly once')
        stage = 'ALL_METRICS_AND_FIXED_BOOTSTRAP'; tick()
        reported = read(paths['metrics'])
        recomputed, compact = {}, {}
        for name, bank, target, group in (('T', all_bank, target_t, tg), ('Dfit', dbank, target_d, dg)):
            saved = np.load(checked(reported[name]['all_values']), allow_pickle=False)
            result, arrays = complete_metrics(bank, target, group, saved, tick)
            need(reported[name]['concept_order'] == list(group), 'Saved concept order differs')
            for key in ('means', 'comparison_means', 'comparison_descriptive95_lower', 'comparison_descriptive95_upper'):
                need(np.allclose(np.asarray(reported[name][key]), np.asarray(result[key]), atol=1e-12, rtol=1e-12), 'JSON metric differs from raw bank')
            recomputed[name] = result
            compact.update({name + '_' + key: value for key, value in arrays.items()})
        dfit = recomputed['Dfit']
        means, lower, upper = (np.asarray(dfit[key]) for key in ('comparison_means', 'comparison_descriptive95_lower', 'comparison_descriptive95_upper'))
        top1_gate = bool(np.all(means[:, 1] > 0.) and np.all(lower[:, 1] > 0.))
        no_clear_afc_harm = bool(np.all(upper[:, 2] >= 0.))
        direction = top1_gate and no_clear_afc_harm
        stage = 'FINAL_SOURCE_RECHECK'; tick()
        for item in pins + [done[key] for key in keys] + [frozen['checkpoint'], frozen['source']]:
            checked(item)
        need(ref(completion_path) == done_ref and not torch.cuda.is_initialized(), 'Producer completion changed or CUDA activated')
        compact.update(T_spherical_errors=t_preserve, Dfit_spherical_errors=d_preserve)
        with (output / 'all_metrics_bootstraps_and_endpoint_constraints.npz').open('xb') as f:
            np.savez_compressed(f, **compact)
        summary = dict(status='D29_ALL14050_ENDPOINTS_AND_T_DFIT_METRICS_CPU_INDEPENDENTLY_VERIFIED', UTC=datetime.now(timezone.utc).isoformat(), source=ref(__file__), completion=done_ref, axes=dict(arm=list(ARMS), metric=list(METRICS)), T_queries=N, Dfit_queries=820, T_concepts=1323, Dfit_concepts=82, rank=K, role_proofs=proof, held_reproducibility=repro_audits, recomputed_metrics=recomputed, Dfit_direction_gate=dict(Top1_positive_mean_and_lower95_against_both=top1_gate, no_clear_2AFC_harm=no_clear_afc_harm, passed=direction), decision=('DFIT_DIRECTION_GATE_PASSED_HELD_EEG_SOURCE_REVIEW_REQUIRED_NO_AUTOMATIC_IMAGES' if direction else 'STOP_THIS_INSTANCE_NO_IMAGE_ADMISSION'), ordinary_old_checkpoint_changed=False, formal_D_resumed=False, new_images=0, GPU_initialized=False, automatic_image_admission=False, image_admission_written=False, whole_ordinary_pipeline_independent_OOF=False, old_ordinary_previously_fitted_on_all_T=True, independent_arithmetic='NumPy float64 QR/spherical endpoints and metric arithmetic; original ordinary replay via primitive CPU Torch linear/GELU, no producer class imports', verification_limits=['Saved FIT/held moment algebra is checked, but raw EEG/model forward moment construction is not independently rerun.', 'Preserving a fitted H does not prove physiology, full EEG sufficiency, new image benefit, or novelty.', 'Clipped antipodal fallback can be feasible without being the nearest ordinary endpoint.', 'T and Dfit are repeatedly used development data, not independent confirmation.'], mathematical_self_check=checks, all_source_SHA256_reverified=True, peak_RSS_bytes=peak_rss(), peak_extra_CPU_bytes=peak_rss()-base_rss, maximum_extra_CPU_bytes=10 << 30, compact_results=ref(output / 'all_metrics_bootstraps_and_endpoint_constraints.npz'), seconds=time.monotonic() - start)
        json_new(output / 'summary.json', summary)
        print(json.dumps(dict(status=summary['status'], decision=summary['decision'], seconds=summary['seconds']), allow_nan=False), flush=True)
        return summary
    except BaseException as error:
        json_new(output / 'failure.json', dict(status='D29_CPU_AUDIT_FAILURE_PRESERVED', stage=stage, error=repr(error), traceback=traceback.format_exc(), seconds=time.monotonic() - start, no_automatic_retry=True))
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--completion', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-seconds', type=int, default=180)
    a = p.parse_args()
    need(30 <= a.max_seconds <= 300, 'Finite CPU budget must be30..300 seconds')
    run(a.completion, a.output, a.max_seconds)


if __name__ == '__main__':
    main()
