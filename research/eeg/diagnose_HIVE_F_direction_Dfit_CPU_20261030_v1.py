"""Frozen all-Dfit direction diagnostic. Concept labels are scoring-only.

No fit, strength selection, generator, or CUDA. Dfit has already been used for
calibration; these descriptive diagnostics cannot create an independent test.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'


def ref(path):
    path = Path(path)
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def check(item):
    assert ref(item['path']) == item
    return Path(item['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def run(protocol):
    start = time.monotonic()
    s = json.loads(Path(protocol).read_bytes())
    assert s['status'] == 'REGISTERED_FROZEN_DFIT_DIRECTION_DIAGNOSTIC_v1'
    assert s['host'] == 'old' and s['training_seed'] == 1729
    assert s['new_training_updates'] == s['new_images'] == 0
    out = Path(s['output_root'])
    out.mkdir(exist_ok=False, parents=True)
    phase = 'SOURCE_AND_INPUT_VERIFICATION'
    try:
        for item in s['source_pins']:
            check(item)
        baseline_study = json.loads(check(s['norm_only_study']).read_bytes())
        R = Path(baseline_study['repository_root'])
        sys.path.insert(0, str(R))
        sys.path.insert(0, str(Path(s['module_root'])))
        import brainhive_paired_study_contract_20261025_v1 as C
        C.setup_runtime(dict(runtime_manifest=baseline_study['runtime_manifest']))
        import torch
        import numpy as np
        from brainhive_component_data_v1 import load_development
        from brainhive_bounded_component_v1 import BoundedConditionAdapter
        from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
        from brainhive_Euclidean_F_norm_only_control_20261030_v1 import positive_ray_norm_projection
        torch.set_num_threads(2)
        torch.manual_seed(1729)
        assert not torch.cuda.is_initialized()
        contract_path = Path('/root/brainhive_reproduction_20261020_v1/data/component_input_contract_v1.json')
        contract = json.loads(contract_path.read_bytes())
        target_path = Path(contract['target_cache_path'])
        assert ref(target_path)['sha256'] == contract['target_cache_sha256']
        split_path = R / 'configs/split_manifest.json'
        split = json.loads(split_path.read_bytes())
        Trows = np.asarray(split['image_rows']['T'], dtype=np.int64)
        first_full = next(x for x in contract['subjects'][0]['sources'] if x['role']=='full')
        Drows = np.asarray([x['array_row'] for x in first_full['groups']['calibration_only_Dfit']], dtype=np.int64)
        assert len(Trows) == 13230 and len(Drows) == 820 and not set(Trows) & set(Drows)
        full_target = torch.load(target_path, weights_only=True, map_location='cpu', mmap=True)
        assert full_target.shape == (16540, 1024)
        tv = full_target[Trows.tolist()].double().numpy()
        tv /= np.linalg.norm(tv, axis=1, keepdims=True)
        population_average = tv.mean(axis=0)
        population = population_average / np.linalg.norm(population_average)
        # Independently partition T into 13 fixed chunks: no Dfit target enters m.
        chunk_sum = np.zeros(1024, dtype=np.float64)
        for ids in np.array_split(Trows, 13):
            x = full_target[ids.tolist()].double()
            chunk_sum += (x / x.norm(dim=-1, keepdim=True)).sum(dim=0).numpy()
        independent_population = chunk_sum / np.linalg.norm(chunk_sum)
        assert np.max(np.abs(independent_population-population)) < 2e-14
        data = load_development('Dfit', 'cpu')
        n = len(data['queries'])
        assert n == 8200
        metadata_path = R / 'audit/image_manifest.csv'
        metadata = {int(r['array_row']): r for r in csv.DictReader(metadata_path.open()) if r['dataset_split'] == 'train'}
        subjects = sorted({q['subject'] for q in data['queries']})
        concepts = sorted({metadata[int(i)]['concept_id'] for i in Drows})
        assert len(subjects) == 10 and len(concepts) == 82
        assert concepts == sorted(split['d_fit_concepts'])
        assert set(Drows) <= set(split['image_rows']['D'])
        subject_ids = np.asarray([subjects.index(q['subject']) for q in data['queries']], dtype=np.int64)
        rows = np.asarray([q['array_row'] for q in data['queries']], dtype=np.int64)
        concept_ids = np.asarray([concepts.index(metadata[int(r)]['concept_id']) for r in rows], dtype=np.int64)
        for subject in range(10):
            sel = subject_ids == subject
            assert np.array_equal(rows[sel], rows[subject_ids == 0])
            for concept in range(82):
                assert np.sum(sel & (concept_ids == concept)) == 10
        target = data['target'].double().numpy()
        assert np.array_equal(target, full_target[rows.tolist()].double().numpy())
        target /= np.linalg.norm(target, axis=1, keepdims=True)
        target_perp = target - np.outer(target @ population, population)
        target_perp /= np.linalg.norm(target_perp, axis=1, keepdims=True)
        np.savez_compressed(out / 'identity_population.npz', subject_ids=subject_ids, concept_ids=concept_ids,
                            array_rows=rows, population=population, population_average=population_average,
                            T_array_rows=Trows, Dfit_array_rows=Drows)
        phase = 'FROZEN_CPU_HEAD_FORWARD_ALL8200'
        model_pins = {name: baseline_study[key] for name, key in (
            ('ordinary', 'immutable_ordinary'), ('bounded_F', 'bounded_F_component'), ('Euclidean_F', 'fixed_component'))}
        for pin in model_pins.values():
            check(pin)
        outputs = {'parent': data['condition'].numpy().copy()}
        model_details = {}
        witness_ids = np.linspace(0, n-1, 64, dtype=np.int64)
        witnesses = dict(condition=data['condition'][witness_ids].numpy(), weight=data['weight'][witness_ids].numpy(),
                         bias=data['bias'][witness_ids].numpy(), query_indices=witness_ids)
        from scipy.special import erf
        for name in ('ordinary', 'bounded_F', 'Euclidean_F'):
            state = torch.load(model_pins[name]['path'], weights_only=True, map_location='cpu')
            assert state['updates'] == 24807 and state['seen_queries'] == 396900
            model = EuclideanConditionControl() if name == 'Euclidean_F' else BoundedConditionAdapter()
            model.load_state_dict(state['model'], strict=True)
            model.eval().requires_grad_(False)
            assert sum(p.numel() for p in model.parameters()) == 525568
            model.radius = .25
            result = []
            with torch.inference_mode():
                for begin in range(0, n, 512):
                    sl = slice(begin, begin+512)
                    result.append(model(data['condition'][sl], data['weight'][sl], data['bias'][sl])[0].numpy())
            outputs[name] = np.concatenate(result)
            # Independent FP64 witness equations; numerical differences are
            # reported rather than substituting for the old failed T guard.
            params = {k: v.double().numpy() for k, v in state['model'].items()}
            c, w, b = (witnesses[k].astype(np.float64) for k in ('condition', 'weight', 'bias'))
            u = (c-b)/w
            h = u @ params['trunk.0.weight'].T + params['trunk.0.bias']
            h = h*.5*(1+erf(h/np.sqrt(2)))
            d = h @ params['direction.weight'].T + params['direction.bias']
            if name == 'Euclidean_F':
                ideal = c + .25*w*d
            else:
                d -= d.mean(axis=-1, keepdims=True)
                r2 = (u*u).sum(axis=-1, keepdims=True)
                d -= (d*u).sum(axis=-1, keepdims=True)/r2*u
                delta = .25*np.sqrt(r2)*d/np.sqrt(r2+(d*d).sum(axis=-1, keepdims=True))
                scale = np.sqrt(r2)/np.linalg.norm(u+delta, axis=-1, keepdims=True)
                ideal = c+w*((scale-1)*u+scale*delta)
            err = outputs[name][witness_ids].astype(np.float64)-ideal
            model_details[name] = dict(checkpoint=model_pins[name], parameters=525568, fixed_strength=1,
                CPU_forward_queries=n, independent_FP64_witness_queries=64,
                witness_max_abs=float(np.abs(err).max()), witness_relative_L2=float(np.linalg.norm(err)/np.linalg.norm(ideal)),
                original_fullT_guard_replaced=False)
            witnesses[name+'_actual'] = outputs[name][witness_ids]
            witnesses[name+'_independent64'] = ideal
            for k,v in params.items():
                witnesses[name+'__'+k] = v
        norm_rows = []
        with torch.inference_mode():
            for begin in range(0, n, 512):
                sl = slice(begin, begin+512)
                value,_ = positive_ray_norm_projection(data['condition'][sl], torch.from_numpy(outputs['Euclidean_F'][sl]), data['weight'][sl], data['bias'][sl])
                norm_rows.append(value.numpy())
        outputs['Norm_only'] = np.concatenate(norm_rows)
        outputs['T_population_null'] = np.repeat(population[None,:], n, axis=0)
        np.savez_compressed(out/'head_witnesses64.npz', **witnesses)
        phase = 'ALL_QUERY_DIRECTION_METRICS'
        methods = s['methods']
        score_bank = {}
        query_metrics = {}
        concentrations = {}
        eps = 1e-12 # Tie comparison only; never changes head output or training.
        for name in methods:
            z = outputs[name].astype(np.float64)
            z /= np.linalg.norm(z, axis=1, keepdims=True)
            perp = z - np.outer(z @ population, population)
            perp_norm = np.linalg.norm(perp, axis=1)
            matched = np.einsum('ij,ij->i', z, target)
            common_part = (z @ population)*(target @ population)
            residual_part = np.einsum('ij,ij->i', perp, target)
            assert np.max(np.abs(matched-common_part-residual_part)) < 2e-14
            concentrations[name] = dict(unit_prediction_population_energy=float(np.square(z.mean(axis=0)).sum()),
                unit_prediction_variance=float(np.square(z-z.mean(axis=0)).sum(axis=1).mean()),
                per_subject_population_energy=[float(np.square(z[subject_ids==i].mean(axis=0)).sum()) for i in range(10)])
            assert abs(concentrations[name]['unit_prediction_population_energy']+concentrations[name]['unit_prediction_variance']-1)<1e-12
            metrics = dict(F_cosine=matched, T_population_cosine=z@population,
                           F_common_projection=common_part, F_residual_projection=residual_part,
                           native_direction_cosine=np.einsum('ij,ij->i',z,outputs['parent'].astype(np.float64)/np.linalg.norm(outputs['parent'].astype(np.float64),axis=1,keepdims=True)))
            same = np.empty((n,10),dtype=np.float64)
            perp_same = np.empty_like(same)
            true_slots = np.empty(n,dtype=np.int64)
            for si in range(10):
                ids = np.flatnonzero(subject_ids==si)
                gallery = target[ids]
                sims = z[ids] @ gallery.T
                truth = np.diag(sims)
                gt = sims > truth[:,None]+eps
                ties = np.abs(sims-truth[:,None])<=eps
                metrics.setdefault('all820_F_Top1_tie_fraction',np.empty(n))[ids] = np.where(gt.sum(axis=1)==0,1/ties.sum(axis=1),0)
                metrics.setdefault('all820_F_2AFC',np.empty(n))[ids] = ((sims<truth[:,None]-eps).sum(axis=1)+.5*(ties.sum(axis=1)-1))/819
                different = concept_ids[ids,None] != concept_ids[None,ids]
                metrics.setdefault('different_concept_F_2AFC',np.empty(n))[ids] = (((sims<truth[:,None]-eps)&different).sum(axis=1)+.5*(ties&different).sum(axis=1))/810
                for ci in range(82):
                    pos = np.flatnonzero(concept_ids[ids]==ci)
                    ix = ids[pos]
                    same[ix] = sims[np.ix_(pos,pos)]
                    true_slots[ix] = np.arange(10)
                    if name == 'T_population_null':
                        perp_same[ix] = 0.
                    else:
                        unit_perp = perp[ix]/perp_norm[ix,None]
                        perp_same[ix] = unit_perp @ target_perp[ix].T
            score_bank[name+'_same_scores'] = same
            if name != 'T_population_null':
                score_bank[name+'_perp_same_scores'] = perp_same
            for prefix, scores in (('same_concept_F',same),('same_concept_perp_F',perp_same)):
                if name == 'T_population_null' and 'perp' in prefix:
                    continue # Undefined residual direction, never impute as 0.
                truth = scores[np.arange(n),true_slots]
                other = np.ones((n,10),dtype=bool)
                other[np.arange(n),true_slots] = False
                gt = (scores > truth[:,None]+eps)&other
                ties = (np.abs(scores-truth[:,None])<=eps)&other
                metrics[prefix+'_Top1_tie_fraction'] = np.where(gt.sum(axis=1)==0,1/(1+ties.sum(axis=1)),0)
                metrics[prefix+'_2AFC'] = (((scores<truth[:,None]-eps)&other).sum(axis=1)+.5*ties.sum(axis=1))/9
                metrics[prefix+'_mean_margin'] = truth-(scores*other).sum(axis=1)/9
                metrics[prefix+'_best_wrong_margin'] = truth-np.where(other,scores,-np.inf).max(axis=1)
            query_metrics[name] = metrics
        score_bank['true_slots'] = true_slots
        np.savez_compressed(out/'same_concept_score_bank.npz',**score_bank)
        flat = {name+'__'+key:value for name,metrics in query_metrics.items() for key,value in metrics.items()}
        np.savez_compressed(out/'all_query_metrics.npz', **flat)
        phase = 'FIXED_DESCRIPTIVE_SUBJECT_CONCEPT_BOOTSTRAP'
        # Fixed 1000 cross subject/concept resamples, preserving all ten stimuli
        # per concept. Not a training-seed interval or fresh-test inference.
        rng = np.random.default_rng(1729)
        resamples = [(rng.integers(0,10,10),rng.integers(0,82,82)) for _ in range(1000)]
        def summary(x):
            group = np.asarray([[x[(subject_ids==si)&(concept_ids==ci)].mean() for ci in range(82)] for si in range(10)])
            boot = np.asarray([group[np.ix_(ss,cc)].mean() for ss,cc in resamples])
            return dict(mean=float(x.mean()),descriptive_CI95=np.quantile(boot,[.025,.975]).tolist(),
                        subject_means=group.mean(axis=1).tolist())
        means = {name:{key:summary(x) for key,x in metrics.items()} for name,metrics in query_metrics.items()}
        comparisons = {}
        for first,second in s['fixed_comparisons']:
            common = sorted(set(query_metrics[first])&set(query_metrics[second]))
            comparisons[first+'_minus_'+second] = {key:summary(query_metrics[first][key]-query_metrics[second][key]) for key in common}
        null = means['T_population_null']
        assert abs(null['same_concept_F_Top1_tie_fraction']['mean']-.1)<1e-12
        assert abs(null['same_concept_F_2AFC']['mean']-.5)<1e-12
        norm_direction_error = float(np.max(np.abs(query_metrics['Norm_only']['F_cosine']-query_metrics['Euclidean_F']['F_cosine'])))
        assert norm_direction_error < 2e-7
        np.savez_compressed(out/'summary_group_grids.npz',**{name+'__'+key:np.asarray([[x[(subject_ids==si)&(concept_ids==ci)].mean() for ci in range(82)] for si in range(10)]) for name,metrics in query_metrics.items() for key,x in metrics.items()})
        write(out/'summary.json',dict(status='COMPLETE_ALL8200_FIXED_DIRECTION_DIAGNOSTICS',methods=means,comparisons=comparisons,
            concentrations=concentrations,norm_only_minus_raw_max_Fcos_error=norm_direction_error,labels_evaluation_only=True,
            not_fresh_independent_test=True,not_image_efficacy=True,no_automatic_new_fit=True))
        for pin in model_pins.values():
            check(pin)
        assert not torch.cuda.is_initialized()
        completion = dict(status='COMPLETE',UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),seconds=time.monotonic()-start,
            protocol=ref(protocol),input_contract=ref(contract_path),split=ref(split_path),image_metadata=ref(metadata_path),target_cache=ref(target_path),
            T_images=13230,Dfit_queries=8200,Dfit_subjects=10,Dfit_concepts=82,methods=6,
            model_details=model_details,independent_T_population_max_error=float(np.max(np.abs(independent_population-population))),
            original_fullT_guard_passed=False,new_training_updates=0,new_images=0,CUDA_initialized=False,NEW_contacted=False,
            files={p.name:ref(p) for p in out.iterdir() if p.is_file()})
        write(out/'completion.json',completion)
        print(json.dumps(dict(status='COMPLETE',completion=ref(out/'completion.json'),seconds=completion['seconds'])))
    except BaseException as e:
        write(out/'failure.json',dict(phase=phase,type=type(e).__name__,error=str(e),traceback=traceback.format_exc(),seconds=time.monotonic()-start))
        raise


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--protocol',required=True);args=p.parse_args();run(args.protocol)
