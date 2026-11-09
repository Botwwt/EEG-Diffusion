"""D29: fit three role-private EEG evidence subspaces, never fit old weights.

H32 is an analytic measurement hypothesis from native mean2-repeat conditions.
All old encoders, affines and ordinary weights stay frozen. No Dcheck input,
image generation, optimizer, automatic effect gate or image admission exists.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_key, '2')
import argparse
import csv
import gc
import json
from pathlib import Path
import signal
import time
import traceback
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_ordinary_evidence_fiber_20261028_v1 as V

ROOT = Path(__file__).resolve().parent
ARMS = ('evidence_fiber', 'permuted_fiber', 'old_ordinary')
PARTITIONS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
RANK, DIM, CONCEPT_BATCH = 32, 1024, 32


def cpu_tree(value):
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [cpu_tree(v) for v in value]
    return value


def grouped_positions(index, selected):
    selected = set(int(i) for i in selected)
    groups = {}
    for row in index:
        if row['row'] in selected:
            groups.setdefault(row['concept_id'], []).append(row['row'])
    assert groups and all(len(v) == 10 for v in groups.values())
    assert sum(map(len, groups.values())) == len(selected)
    return list(groups), np.array(list(groups.values()), dtype=np.int64)


def metric_bank(predictions, target, concepts):
    """All native-coordinate F metrics, grouped by actual stimulus metadata."""
    assert predictions.shape == (len(target), 3, DIM) and target.shape[1] == DIM
    group = {}
    for i, concept in enumerate(concepts):
        group.setdefault(concept, []).append(i)
    assert all(len(rows) == 10 for rows in group.values())
    per_query = np.empty((len(target), 3, 3), dtype=np.float64)
    for rows in group.values():
        pred = np.asarray(predictions[rows], np.float64)
        truth = np.asarray(target[rows], np.float64)
        pred_norm = np.linalg.norm(pred, axis=-1, keepdims=True)
        target_norm = np.linalg.norm(truth, axis=-1, keepdims=True)
        assert np.all(pred_norm > 0.) and np.all(target_norm > 0.)
        similarity = np.einsum('iad,jd->aij', pred/pred_norm, truth/target_norm)
        diagonal = np.diagonal(similarity, axis1=1, axis2=2)
        maxima = similarity.max(axis=2)
        ties = (similarity == maxima[:, :, None]).sum(axis=2)
        top1 = np.where(diagonal == maxima, 1./ties, 0.)
        within2 = ((diagonal[:, :, None] > similarity).sum(axis=2)
                   + .5*(diagonal[:, :, None] == similarity).sum(axis=2)-.5)/9.
        per_query[rows, :, 0] = np.square(pred-truth[:, None]).mean(axis=2)
        per_query[rows, :, 1] = top1.T
        per_query[rows, :, 2] = within2.T
    per_concept = np.stack([per_query[rows].mean(axis=0) for rows in group.values()])
    comparisons = np.stack((per_concept[:, 0]-per_concept[:, 2],
                            per_concept[:, 0]-per_concept[:, 1]), axis=1)
    rng = np.random.default_rng(1729)
    draws = np.empty((1000, 2, 3), dtype=np.float64)
    for i in range(1000):
        draws[i] = comparisons[rng.integers(0, len(group), size=len(group))].mean(axis=0)
    low, high = np.quantile(draws, [.025, .975], axis=0)
    return dict(metrics=['F_MSE', 'same_concept_Top1', 'same_concept_2AFC'],
                arms=list(ARMS), concept_order=list(group), per_query=per_query,
                per_concept=per_concept, means=per_concept.mean(axis=0),
                comparison_order=['evidence_minus_old_ordinary', 'evidence_minus_permuted'],
                comparison_means=comparisons.mean(axis=0),
                comparison_descriptive95_lower=low, comparison_descriptive95_upper=high)


def main():
    parser = argparse.ArgumentParser()
    for name in ('spec', 'native-registration', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--gpu-job-name', required=True)
    parser.add_argument('--max-seconds', type=int, required=True)
    args = parser.parse_args()
    assert args.max_seconds == 600 and not args.output.exists()
    args.output.mkdir(parents=True)
    started, stopped, stage = time.monotonic(), [False], 'INPUTS'
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    try:
        spec = U.read(args.spec)
        for reference in spec['source_pins']:
            U.checked(reference)
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(C.read(args.native_registration))
        import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
        from brainhive_native_models_v1 import classes
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        registration = C.read(args.native_registration)
        D.torch = torch
        document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=600,
                        maximum_output_bytes=512 << 20, maximum_extra_CPU_bytes=10 << 30,
                        maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
                        resource_authorization=registration['resource_authorization'])
        wrapper, budget = D.verify_wrapper(document), D.Budget(document)
        def check():
            assert not stopped[0] and time.monotonic()-started < 540
            budget.check()
            assert sum(p.stat().st_size for p in args.output.rglob('*') if p.is_file()) < (512 << 20)-(16 << 20)
        def log(value):
            value = dict(value, seconds=time.monotonic()-started)
            with (args.output/'progress.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(value, allow_nan=False)+'\n')
                stream.flush()
                os.fsync(stream.fileno())
            print(json.dumps(value, allow_nan=False), flush=True)
        torch.set_num_threads(2)
        torch.manual_seed(1729)
        torch.cuda.set_device(0)
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        assert V.RHO == .25
        contract = U.read(U.checked(spec['native_input_contract']))
        assert contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT'
        subject = contract['subjects'][0]
        assert subject['subject'] == 'sub-01'
        source_by_role = {s['role']: s for s in subject['sources'] if s['role'] in ('full', 'oof0', 'oof1')}
        assert set(source_by_role) == {'full', 'oof0', 'oof1'}
        split_ref = dict(path=str(ROOT/'configs/split_manifest.json'), sha256=contract['split_manifest_sha256'])
        split = U.read(U.checked(split_ref))
        manifest_ref = dict(path=str(ROOT/'audit/image_manifest.csv'), sha256=split['image_manifest_sha256'])
        metadata = {int(row['array_row']): row for row in csv.DictReader(U.checked(manifest_ref).open(encoding='utf-8')) if row['dataset_split'] == 'train'}
        index = [json.loads(line) for line in U.checked(spec['index']).read_text(encoding='utf-8').splitlines()]
        assert len(index) == 13230 and {r['array_row'] for r in index} == set(split['image_rows']['T'])
        assert all(r['row'] == i and r['target_row'] == i and r['subject_id'] == 'sub-01' and r['partition'] == 'T'
                   and r['concept_id'] == metadata[r['array_row']]['concept_id']
                   and r['stimulus_id'] == metadata[r['array_row']]['stimulus_id']
                   and r['fold'] == split['oof_fold_by_concept'][r['concept_id']] for i, r in enumerate(index))
        trials = np.load(U.checked(spec['trials']), mmap_mode='r', allow_pickle=False)
        assert trials.shape == (13230, 4, 17, 250) and trials.dtype == np.float32
        target_ref = dict(path=contract['target_cache_path'], sha256=contract['target_cache_sha256'])
        fbank = torch.load(U.checked(target_ref), weights_only=True, mmap=True, map_location='cpu')
        assert torch.is_tensor(fbank) and tuple(fbank.shape) == (16540, DIM)
        T_rows = [row['array_row'] for row in index]
        target_T = fbank[T_rows].float().numpy().copy()
        assert target_T.shape == (13230, DIM) and np.isfinite(target_T).all()
        freeze = U.read(U.checked(spec['ordinary_freeze']))
        U.checked(freeze['source'])
        assert freeze['selected_strength'] == 1. and freeze['radius_rho_before_strength'] == .25
        payload = torch.load(U.checked(freeze['checkpoint']), weights_only=False, map_location='cpu')
        assert payload['arm'] == 'ordinary' and payload['calibration']['selected_strength'] == 1.
        ordinary = BoundedConditionAdapter().eval().requires_grad_(False).cuda()
        ordinary.load_state_dict(payload['model'], strict=True)
        del payload
        BrainMLP = classes()['BrainMLP']
        permutation = np.random.default_rng(1729).permutation(DIM)
        permutation_gpu = torch.as_tensor(permutation, device='cuda')
        outputs = np.lib.format.open_memmap(args.output/'OOF_T_conditions.npy', mode='w+', dtype=np.float32, shape=(13230, 3, DIM))
        native_outputs = np.lib.format.open_memmap(args.output/'OOF_T_native.npy', mode='w+', dtype=np.float32, shape=(13230, DIM))
        covered = np.zeros(13230, np.int8)
        by_array_row = {r['array_row']: r['row'] for r in index}
        bundles, source_proofs, reproduction = {}, {}, {}
        global_max_cache_error = 0.
        endpoint_proofs = []
        Dfit_conditions = Dfit_native = Dfit_index = target_Dfit = None
        for role in ('full', 'oof0', 'oof1'):
            stage = 'ROLE_'+role
            check()
            source = source_by_role[role]
            cp_ref = dict(path=source['checkpoint_path'], sha256=source['checkpoint_sha256'])
            cache_ref = dict(path=source['condition_cache_path'], sha256=source['condition_cache_sha256'])
            parent_ref = dict(path=source['parent_registration_path'], sha256=source['parent_registration_sha256'])
            parent = U.read(U.checked(parent_ref))
            cache = torch.load(U.checked(cache_ref), weights_only=True, mmap=True, map_location='cpu')
            assert cache['model_sha256'] == cp_ref['sha256'] and cache['role'] == role
            assert cache['condition'].ndim == 2 and cache['condition'].shape[1] == DIM
            cache_rows = {int(row): i for i, row in enumerate(cache['array_rows'])}
            assert len(cache_rows) == len(cache['array_rows'])
            fit_positions = list(range(13230)) if role == 'full' else [r['row'] for r in index if r['fold'] != int(role[-1])]
            held_positions = [] if role == 'full' else [r['row'] for r in index if r['fold'] == int(role[-1])]
            assert set(parent['training_array_rows']) == {T_rows[i] for i in fit_positions}
            assert set(parent['excluded_oof_rows']) == {T_rows[i] for i in held_positions}
            if role != 'full':
                routes = source['groups']['component_train']
                assert {r['array_row'] for r in routes} == {T_rows[i] for i in held_positions}
                assert all(int(cache['array_rows'][r['cache_row']]) == r['array_row'] for r in routes)
            model = BrainMLP(4250, DIM, num_layers=1, dropout=.1).eval().requires_grad_(False).cuda()
            original = torch.load(U.checked(cp_ref), weights_only=True, map_location='cpu')
            model.load_state_dict(original['model'], strict=True)
            del original
            weight = model.block.layers[0].norm.weight
            bias = model.block.layers[0].norm.bias
            assert torch.all(weight.abs() > 1e-6)

            def moments(positions, protected=None):
                names, groups = grouped_positions(index, positions)
                signal = torch.zeros((DIM, DIM), dtype=torch.float64, device='cuda')
                total = torch.zeros_like(signal)
                projected = np.zeros((len(groups), 2, 2), np.float64) if protected is not None else None
                for lo in range(0, len(groups), CONCEPT_BATCH):
                    check()
                    current = groups[lo:lo+CONCEPT_BATCH].reshape(-1)
                    block = torch.from_numpy(np.array(trials[current], copy=True)).cuda()
                    assert torch.isfinite(block).all()
                    for left, right in PARTITIONS:
                        a = block[:, left].mean(dim=1).reshape(-1, 4250)
                        b = block[:, right].mean(dim=1).reshape(-1, 4250)
                        values = model(torch.cat((a, b), dim=0))
                        values = ((values-bias)/weight).double()
                        A, B = values[:len(current)], values[len(current):]
                        A = A-A.reshape(-1, 10, DIM).mean(1, keepdim=True).expand(-1, 10, -1).reshape(-1, DIM)
                        B = B-B.reshape(-1, 10, DIM).mean(1, keepdim=True).expand(-1, 10, -1).reshape(-1, DIM)
                        cross = A.T@B
                        signal.add_(.5*(cross+cross.T))
                        total.add_(.5*(A.T@A+B.T@B))
                        if protected is not None:
                            for arm, H in enumerate(protected):
                                ha, hb = A@H.T, B@H.T
                                cross_energy = (ha*hb).sum(1).reshape(-1, 10).mean(1)
                                total_energy = .5*(ha.square()+hb.square()).sum(1).reshape(-1, 10).mean(1)
                                projected[lo:lo+len(current)//10, arm, 0] += cross_energy.cpu().numpy()/3.
                                projected[lo:lo+len(current)//10, arm, 1] += total_energy.cpu().numpy()/3.
                    del block, a, b, values, A, B, cross
                denominator = 3.*len(positions)
                signal.div_(denominator)
                total.div_(denominator)
                assert torch.isfinite(signal).all() and torch.isfinite(total).all()
                return signal, total, names, projected

            with torch.inference_mode():
                signal, total, fit_concepts, _ = moments(fit_positions)
                eig, vectors = torch.linalg.eigh(signal)
                positive = int((eig > 0.).sum().item())
                assert positive >= RANK, 'Fewer than32 positive signal eigenvalues; stop without rank scan'
                H = vectors[:, -RANK:].T.flip(0).contiguous()
                pivot = H.abs().argmax(dim=1)
                signs = torch.sign(H[torch.arange(RANK, device='cuda'), pivot])
                H = H*signs[:, None]
                H_permuted = H[:, permutation_gpu].contiguous()
                assert torch.linalg.vector_norm(H@H.T-torch.eye(RANK, device='cuda', dtype=torch.float64)) < 1e-10
                assert torch.allclose(H@H.T, H_permuted@H_permuted.T, atol=1e-12, rtol=1e-12)
                Q = V.build_protected_basis_torch(H)
                Q_permuted = V.build_protected_basis_torch(H_permuted)
                assert len(Q) == len(Q_permuted) == RANK+1
                bundle = dict(H=H, H_permuted=H_permuted, Q_with_mean=Q, Q_permuted_with_mean=Q_permuted,
                              signal_eigenvalues_descending=eig.flip(0), positive_eigenvalue_count=positive,
                              fit_signal_cross_moment=signal, fit_total_moment=total,
                              fit_positions=fit_positions, held_positions=held_positions,
                              fit_concepts=fit_concepts, affine_weight=weight.clone(), affine_bias=bias.clone(),
                              checkpoint=cp_ref, condition_cache=cache_ref, parent_registration=parent_ref)
                if held_positions:
                    held_signal, held_total, held_concepts, projected = moments(held_positions, (H, H_permuted))
                    ratios = np.divide(projected[:, :, 0], projected[:, :, 1], out=np.zeros(projected.shape[:2]), where=projected[:, :, 1] > 0.)
                    aggregate = {}
                    for arm, label in enumerate(('true_H', 'permuted_H')):
                        ce, te = projected[:, arm].mean(axis=0)
                        aggregate[label] = dict(cross_energy=float(ce), total_energy=float(te),
                                                cross_over_total=float(ce/te) if te > 0. else 0.,
                                                mean_of_concept_ratios=float(ratios[:, arm].mean()))
                    repro_path = args.output/(role+'_held_reproducibility.npz')
                    with repro_path.open('xb') as stream:
                        np.savez(stream, per_concept_cross_and_total=projected, per_concept_ratios=ratios,
                                 concepts=np.asarray(held_concepts), arms=np.asarray(['true_H', 'permuted_H']))
                    reproduction[role] = dict(scope='held_concepts_only', held_concepts=len(held_concepts),
                                               queries=len(held_positions), aggregate=aggregate,
                                               all_concepts=U.ref(repro_path), physiological_iid_established=False)
                    bundle.update(held_signal_cross_moment=held_signal, held_total_moment=held_total,
                                  held_concepts=held_concepts)
                else:
                    aggregate = {}
                    for label, basis in (('true_H', H), ('permuted_H', H_permuted)):
                        ce = float((basis*(basis@signal)).sum())
                        te = float((basis*(basis@total)).sum())
                        aggregate[label] = dict(cross_energy=ce, total_energy=te, cross_over_total=ce/te if te > 0. else 0.)
                    reproduction[role] = dict(scope='FIT_ONLY_NO_HELD_T_CONCEPTS', queries=len(fit_positions),
                                               aggregate=aggregate, not_independent_confirmation=True,
                                               physiological_iid_established=False)

                def apply_component(native):
                    corrected = ordinary(native, weight.expand(len(native), -1), bias.expand(len(native), -1))[0]
                    main, main_proof = V.restore_condition_torch(native, corrected, weight, bias, H)
                    permuted, permutation_proof = V.restore_condition_torch(native, corrected, weight, bias, H_permuted)
                    proof = dict(role=role, rows=len(native),
                                 main_protected_max=float(main_proof['protected_error'].max()),
                                 permuted_protected_max=float(permutation_proof['protected_error'].max()),
                                 main_norm_error_max=float(main_proof['norm_error'].max()),
                                 permuted_norm_error_max=float(permutation_proof['norm_error'].max()),
                                 main_global_angle_max=float(main_proof['actual_global_angle'].max()),
                                 permuted_global_angle_max=float(permutation_proof['actual_global_angle'].max()),
                                 main_kept_original=int(main_proof['kept_original'].sum()),
                                 permuted_kept_original=int(permutation_proof['kept_original'].sum()),
                                 main_non_nearest_antipodal_fallback=int(main_proof['antipodal_clipped_fallback'].sum()),
                                 permuted_non_nearest_antipodal_fallback=int(permutation_proof['antipodal_clipped_fallback'].sum()))
                    assert proof['main_protected_max'] < 1e-4 and proof['permuted_protected_max'] < 1e-4
                    assert proof['main_norm_error_max'] < 1e-4 and proof['permuted_norm_error_max'] < 1e-4
                    assert proof['main_global_angle_max'] <= V.ANGULAR_LIMIT+2e-5 and proof['permuted_global_angle_max'] <= V.ANGULAR_LIMIT+2e-5
                    endpoint_proofs.append(proof)
                    return torch.stack((main, permuted, corrected), dim=1).cpu().numpy()

                if role != 'full':
                    held = np.asarray(held_positions, np.int64)
                    native_parts = []
                    for lo in range(0, len(held), 256):
                        check()
                        positions = held[lo:lo+256]
                        physical_mean = np.asarray(trials[positions]).mean(axis=1, dtype=np.float32)
                        actual = model(torch.from_numpy(physical_mean.copy().reshape(-1, 4250)).cuda()).cpu()
                        expected = cache['condition'][[cache_rows[T_rows[i]] for i in positions]]
                        relative = torch.linalg.vector_norm((actual-expected).double(), dim=1)/torch.linalg.vector_norm(expected.double(), dim=1)
                        error = float(relative.max())
                        assert error < 3e-5
                        global_max_cache_error = max(global_max_cache_error, error)
                        native_parts.append(actual)
                    native = torch.cat(native_parts).cuda()
                    conditions = apply_component(native)
                    assert np.all(covered[held] == 0)
                    outputs[held] = conditions
                    native_outputs[held] = native.cpu().numpy()
                    covered[held] += 1
                    del native_parts, native, conditions
                else:
                    df_routes = source['groups']['calibration_only_Dfit']
                    df_rows = [r['array_row'] for r in df_routes]
                    assert len(df_rows) == len(set(df_rows)) == 820
                    assert all(metadata[i]['concept_id'] in set(split['d_fit_concepts']) for i in df_rows)
                    assert all(int(cache['array_rows'][r['cache_row']]) == r['array_row'] for r in df_routes)
                    Dfit_index = [dict(row=i, array_row=r, subject_id='sub-01', partition='Dfit',
                                       concept_id=metadata[r]['concept_id'], stimulus_id=metadata[r]['stimulus_id']) for i, r in enumerate(df_rows)]
                    native = cache['condition'][[r['cache_row'] for r in df_routes]].clone().cuda()
                    Dfit_native = native.cpu().numpy()
                    Dfit_conditions = apply_component(native)
                    target_Dfit = fbank[df_rows].float().numpy().copy()
                    del native
                bundles[role] = cpu_tree(bundle)
            source_proofs[role] = dict(checkpoint=cp_ref, condition_cache=cache_ref, parent_registration=parent_ref,
                                       H_fit_queries=len(fit_positions), H_fit_concepts=len(fit_concepts),
                                       H_held_queries=len(held_positions), rank=RANK, protected_rank_including_mean=len(Q),
                                       only_role_FIT_used_for_H=True, affine_unchanged=True)
            log(dict(stage='ROLE_ANALYTIC_H_AND_CONDITIONS_COMPLETE', role=role,
                     FIT=len(fit_positions), held=len(held_positions), positive_eigenvalues=positive,
                     reproducibility_scope=reproduction[role]['scope']))
            del model, cache, H, H_permuted, Q, Q_permuted, bundle, signal, total, eig, vectors
            gc.collect()
            torch.cuda.empty_cache()
        assert np.all(covered == 1) and np.isfinite(outputs).all() and np.isfinite(native_outputs).all()
        outputs.flush()
        native_outputs.flush()
        assert Dfit_conditions is not None and np.isfinite(Dfit_conditions).all()
        stage = 'ALL_T_AND_FINAL_DFIT_READOUT'
        check()
        T_metrics = metric_bank(outputs, target_T, [row['concept_id'] for row in index])
        df_metrics = metric_bank(Dfit_conditions, target_Dfit, [row['concept_id'] for row in Dfit_index])
        summaries = {}
        for name, values in (('T', T_metrics), ('Dfit', df_metrics)):
            path = args.output/(name+'_all_metrics.npz')
            with path.open('xb') as stream:
                np.savez(stream, per_query=values['per_query'], per_concept=values['per_concept'],
                         means=values['means'], comparison_means=values['comparison_means'],
                         comparison_descriptive95_lower=values['comparison_descriptive95_lower'],
                         comparison_descriptive95_upper=values['comparison_descriptive95_upper'])
            summaries[name] = {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                               for key, value in values.items() if key not in ('per_query', 'per_concept')}
            summaries[name]['all_values'] = U.ref(path)
        for name, value in (('Dfit_conditions.npy', Dfit_conditions), ('Dfit_native.npy', Dfit_native),
                            ('target_T.npy', target_T), ('target_Dfit.npy', target_Dfit),
                            ('coordinate_permutation.npy', permutation)):
            U.save_array(args.output/name, value)
        with (args.output/'Dfit_index.jsonl').open('x', encoding='utf-8') as stream:
            for row in Dfit_index:
                stream.write(json.dumps(row, allow_nan=False)+'\n')
        torch.save(dict(roles=bundles, rank=RANK, partitions=PARTITIONS,
                        signal_formula='sum_3 sym(XA.T@XB)/(3*N); within10concept centered mean2 native u',
                        total_formula='sum_3 (XA.T@XA+XB.T@XB)/(6*N)',
                        original_affines_preserved=True), args.output/'H_bundle.pt')
        U.atomic(args.output/'reproducibility.json', reproduction)
        U.atomic(args.output/'endpoint_proofs.json', dict(proofs=endpoint_proofs, affine_coordinates='original role-specific LN affine undone'))
        U.atomic(args.output/'final_metrics.json', dict(summaries, automatic_image_admission=False,
                                                     image_admission_written=False, MSE_is_not_an_image_effect=True))
        del outputs, native_outputs, fbank
        for reference in spec['source_pins']:
            U.checked(reference)
        U.checked(freeze['checkpoint'])
        U.checked(freeze['source'])
        check()
        done = dict(status='D29_ALL3_ROLE_H32_FIT_AND_TRUE_OOF_T_DFIT_CONDITIONS_EXPORTED',
                    spec=U.ref(args.spec), source=U.ref(__file__), wrapper=wrapper,
                    H_bundle=U.ref(args.output/'H_bundle.pt'), ordinary_freeze=spec['ordinary_freeze'],
                    native_input_contract=spec['native_input_contract'], trials=spec['trials'], index=spec['index'],
                    split=split_ref, manifest=manifest_ref, targets=target_ref,
                    OOF_T_conditions=U.ref(args.output/'OOF_T_conditions.npy'),
                    OOF_T_native=U.ref(args.output/'OOF_T_native.npy'),
                    Dfit_conditions=U.ref(args.output/'Dfit_conditions.npy'), Dfit_native=U.ref(args.output/'Dfit_native.npy'),
                    Dfit_index=U.ref(args.output/'Dfit_index.jsonl'),
                    target_T=U.ref(args.output/'target_T.npy'), target_Dfit=U.ref(args.output/'target_Dfit.npy'),
                    coordinate_permutation=U.ref(args.output/'coordinate_permutation.npy'),
                    metrics=U.ref(args.output/'final_metrics.json'), reproducibility=U.ref(args.output/'reproducibility.json'),
                    endpoint_proofs=U.ref(args.output/'endpoint_proofs.json'), source_models=source_proofs,
                    arms=ARMS, T_queries=13230, Dfit_queries=820, H_rank=RANK, protected_rank_including_mean=RANK+1,
                    max_mean4_cache_error=global_max_cache_error, all_native_and_ordinary_weights_frozen=True,
                    all_original_affines_unchanged=True, ordinary_strength=1., ordinary_radius=.25,
                    new_trainable_parameters=0, analytic_subspaces_fitted=3,
                    analytic_H_scalars_per_role=RANK*DIM,
                    fixed_inference_basis_rows_including_mean=RANK+1, SGD_updates=0,
                    new_images=0, new_UNet_calls=0, Dcheck_conditions_or_target_values_read=False,
                    normal_inference_reads_only_current_c_and_fixed_H=True,
                    normal_inference_reads_physical_trials_or_query_targets=False,
                    whole_ordinary_pipeline_independent_OOF=False,
                    old_ordinary_was_previously_fitted_on_all_T=True,
                    H_held_reproducibility_has_physiological_iid_guarantee=False,
                    automatic_image_admission=False, image_admission_written=False,
                    original_ordinary_checkpoint_changed=False, formal_D_resumed=False,
                    seconds=time.monotonic()-started, peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,
                    peak_GPU_bytes=budget.peak_GPU_bytes)
        U.atomic(args.output/'completion.json', done)
        log(dict(stage='COMPLETE', status=done['status'], maximum_mean4_cache_error=global_max_cache_error))
    except BaseException as error:
        U.atomic(args.output/'failure.json', dict(stage=stage, error=repr(error), traceback=traceback.format_exc(),
                                                 seconds=time.monotonic()-started, no_automatic_retry=True))
        raise


if __name__ == '__main__':
    main()
