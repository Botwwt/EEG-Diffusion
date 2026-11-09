"""Bounded full10 CPU feasibility diagnosis; never trains or launches images.

The old completed Validator supplies independently hashed T/D cache loading.
Its v1 scientific outputs are preserved. V and official test queries are unused.
Dcheck is already exposed, so all results remain exploratory development evidence.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
import torch.nn.functional as F
import numpy as np
from validate_idea_first_error_channel_v1 import Validator
from direct_mean_error_channel_v2 import designs, fit_mean, predict, calibration, sample, tensor_sha
from native_error_channel import NativeErrorChannel, stream_seed
from error_channel_validation_metrics import energy_score, classifier_features, fit_two_sample, confusion_js
from locked_metrics import fixed_gallery_metrics
from paired_analysis import paired_summary

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def write(path, obj):
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2)+'\n')
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text())
    assert plan['version'] == 'direct_mean_channel_exploratory_v2'
    assert plan['relative_ridge'] == .01 and plan['draws'] == 16 and plan['seed'] == 1729
    assert plan['CPU_only'] and not plan['may_launch_GPU_training']
    for name, digest in plan['source_sha256'].items():
        assert sha(ROOT/name) == digest, name
    torch.set_num_threads(2)
    os.nice(10)
    started = time.monotonic()
    output = Path(plan['output'])
    data_root = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
    assert output.is_relative_to(data_root/'error_validation') and not output.exists()
    import shutil
    assert shutil.disk_usage(output.parent).free >= plan['minimum_free_bytes_before_start']
    v = Validator(SimpleNamespace(plan=str(ROOT/'configs/idea_first_error_validation_plan_v1.json'),
        cache_config=str(ROOT/'configs/idea_first_native_cache_v1.json'), output=str(output)))
    assert v.chain == 'REF_REG'
    write(output/'direct_mean_plan_receipt.json', dict(plan=str(plan_path), sha256=sha(plan_path),
        old_validator_input_provenance_sha256=sha(output/'input_provenance.json'),
        Dcheck_already_exposed=True, scientific_goal_complete=False))
    tci = v.banks['T']['C_I']
    dci = v.banks['D']['C_I']
    tfolds = torch.tensor([v.protocol['oof_fold_by_concept'][c] for c in v.banks['T']['concept_ids']])
    design = designs(tci, tfolds, plan['relative_ridge'])
    old_root = Path(plan['v1_completed_validation'])
    old_proof = json.loads((old_root/'input_provenance.json').read_text())
    assert old_proof['cache_config_sha256'] == sha(ROOT/'configs/idea_first_native_cache_v1.json')
    previous = {}
    with (old_root/'per_query.csv').open(newline='') as file:
        for r in csv.DictReader(file):
            if r['role'] == 'Dcheck':
                previous[(r['arm'], r['query_id'])] = r
    assert len(previous) == 4*8300
    scalar_rows, subject_receipts, real_features, sim_features = [], [], [], []
    for index, subject in enumerate(v.subjects):
        assert time.monotonic()-started < plan['max_seconds']-60, 'Preserve partial CPU diagnosis'
        t = v.load(subject, 'T_oof', 'T')
        d = v.load(subject, 'D_full', 'D')
        ce = d['C_E']
        models = {role: fit_mean(value, t['C_E']) for role, value in design.items()}
        oof_prediction = torch.empty_like(t['C_E'])
        for fold in [0, 1]:
            mask = tfolds == fold
            oof_prediction[mask] = predict(models[f'holdout{fold}'], tci[mask])
            assert not (design[f'holdout{fold}']['mask'] & mask).any()
        residual = t['C_E']-oof_prediction
        mean_d = predict(models['full'], dci)
        state = calibration(mean_d, ce, v.fit, residual)
        old_model = torch.load(old_root/subject/'error_channel.pt', weights_only=True, map_location='cpu')
        assert old_model['actual_scientific_fit'] and old_model['T_source_sha256'] == sha(v.root/subject/'T_oof.pt')
        model = NativeErrorChannel(old_model['state'])
        neighbors = torch.load(old_root/subject/'D_nearest_other_T_concepts.pt', weights_only=True, map_location='cpu')
        assert neighbors.shape == (1650, 32)
        for row, concept in zip(neighbors.tolist(), d['concept_ids']):
            selected_concepts = [v.banks['T']['concept_ids'][j] for j in row]
            assert len(set(selected_concepts)) == 32 and concept not in selected_concepts
        simulated = []
        for draw in range(plan['draws']):
            # Same existing draw assignment and source-neighbor stream as v1.
            seeds = [stream_seed(1729, v.plan['version'],
                     subject, stim, 'R4', 'error-draw', draw) for stim in d['stimulus_ids']]
            value, _ = sample(mean_d, residual, neighbors, seeds, state)
            simulated.append(value)
        sim = torch.stack(simulated, 1)
        first = sim[:, 0]
        real_projected = v.project(ce)
        native, projected = [], []
        for begin in range(0, len(dci), 32):
            x = sim[begin:begin+32]
            native.append(energy_score(x, ce[begin:begin+32]))
            p = v.project(x.flatten(0, 1)).view(len(x), 16, -1)
            projected.append(energy_score(p, real_projected[begin:begin+32]))
        native, projected = torch.cat(native), torch.cat(projected)
        rank = fixed_gallery_metrics(first, dci, d['stimulus_ids'], d['stimulus_ids'])
        actual = fixed_gallery_metrics(ce, dci, d['stimulus_ids'], d['stimulus_ids'])
        rtop = (F.normalize(ce, dim=-1)@F.normalize(dci, dim=-1).T).argmax(1)
        stop = (F.normalize(first, dim=-1)@F.normalize(dci, dim=-1).T).argmax(1)
        confusion = confusion_js(rtop[v.check], stop[v.check],
            [c for c, use in zip(d['concept_ids'], v.check) if use], d['concept_ids'])
        real_features.append(classifier_features(dci, ce, model.state['pca'], index))
        sim_features.append(classifier_features(dci, first, model.state['pca'], index))
        folder = output/subject
        folder.mkdir()
        saved = dict(models=models, calibration=state, plan_sha256=sha(plan_path),
            T_source_sha256=sha(v.root/subject/'T_oof.pt'),
            residual_sha256=tensor_sha(residual), fit_roles=['T_concept_crossfit', 'Dfit'],
            residual_storage='Recompute from saved inner-fold means and existing T cache; no duplicate large residual bank')
        torch.save(saved, folder/'mean_models_and_calibration.pt')
        for i, query in enumerate(d['query_ids']):
            if not v.check[i]:
                continue
            old = previous[('Conditional', query)]
            close_actual = float(actual['two_afc'][i])
            assert math.isclose(close_actual, float(old['real_native_2AFC']), abs_tol=1e-7)
            scalar_rows.append(dict(query_id=query, subject=subject, stimulus_id=d['stimulus_ids'][i],
                concept_id=d['concept_ids'][i], native_energy_score=float(native[i]),
                projected_energy_score=float(projected[i]), sim_native_Top1=int(rank['top1'][i]),
                real_native_Top1=int(actual['top1'][i]), sim_native_Top5=int(rank['top5'][i]),
                real_native_Top5=int(actual['top5'][i]), sim_native_2AFC=float(rank['two_afc'][i]),
                real_native_2AFC=close_actual, sim_norm=float(first[i].norm()), real_norm=float(ce[i].norm()),
                mean_only_squared_error=float((mean_d[i]+state['bias']-ce[i]).square().mean())))
        subject_receipts.append(dict(subject=subject, queries=830,
            wrong_concept_pattern=confusion, v1_wrong_concept_pattern=next(x['wrong_concept_pattern']
                for x in json.loads((old_root/'summary.json').read_text())['per_subject_distributions']
                if x['subject']==subject and x['arm']=='Conditional'),
            calibration={k: value for k, value in state.items() if k not in ['residual_mean', 'bias']},
            model_sha256=sha(folder/'mean_models_and_calibration.pt')))
        write(output/'progress.json', dict(status='RUNNING_CPU_ONLY_EXPLORATORY',
            completed_subjects=index+1, expected_subjects=10, Dcheck_queries=len(scalar_rows),
            seconds=time.monotonic()-started, GPU_used=False))
        print(json.dumps(dict(stage='CPU_direct_mean_residual_v2', subject=subject,
            completed_subjects=index+1, seconds=time.monotonic()-started)), flush=True)
    assert len(scalar_rows) == len({r['query_id'] for r in scalar_rows}) == 8300
    with (output/'per_query.csv').open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(scalar_rows[0]))
        writer.writeheader()
        writer.writerows(scalar_rows)
    real_x, sim_x = torch.cat(real_features), torch.cat(sim_features)
    mask = v.fit.repeat(10)
    classification, classifier_states = fit_two_sample(real_x[mask], sim_x[mask],
        real_x[~mask], sim_x[~mask], v.plan['classifier'])
    torch.save(classifier_states, output/'classifier_states.pt')
    means = {k: float(np.mean([r[k] for r in scalar_rows])) for k in scalar_rows[0]
             if k not in ['query_id', 'subject', 'stimulus_id', 'concept_id']}
    comparisons = {}
    for arm in ['Conditional', 'Moment', 'Iso']:
        fields = ['native_energy_score', 'projected_energy_score']
        diff = np.array([[r[k]-float(previous[(arm,r['query_id'])][k]) for k in fields] for r in scalar_rows])
        stats = paired_summary(diff, [r['subject'] for r in scalar_rows],
                               [r['concept_id'] for r in scalar_rows], 1000, 1729)
        comparisons[arm] = {k: dict(delta=stats['mean'][j], CI95=stats['CI95'][j]) for j,k in enumerate(fields)}
    old_summary = json.loads((ROOT/'audit/idea_first_Dcheck_summary_v1.json').read_text())
    auc = classification['mlp64']['Dcheck']['AUC_real_positive']
    gate = dict(Top1_absolute_gap_within_3pp=abs(means['sim_native_Top1']-means['real_native_Top1'])<=.03,
        Top5_absolute_gap_within_5pp=abs(means['sim_native_Top5']-means['real_native_Top5'])<=.05,
        norm_ratio_within10percent=abs(means['sim_norm']/means['real_norm']-1)<=.1,
        confusion_JS_lower_in_all10=all(x['wrong_concept_pattern']['jensen_shannon_nats'] <
            x['v1_wrong_concept_pattern']['jensen_shannon_nats'] for x in subject_receipts),
        controlled_MLP_AUC_lower_than_v1=auc<old_summary['results']['Conditional']['Dcheck_classifiers']['mlp64']['AUC_real_positive'],
        native_and_projected_energy_better_than_Moment=all(comparisons['Moment'][k]['CI95'][1]<0
            for k in ['native_energy_score','projected_energy_score']))
    assert not torch.cuda.is_initialized()
    summary = dict(status='COMPLETED_FULL10_CPU_EXPLORATORY_DIAGNOSIS', queries=8300, subjects=10,
        T_queries=132300, Dfit_queries=8200, Dcheck_queries=8300, condition_chain='REF_REG',
        means=means, per_subject=subject_receipts, classifiers=classification,
        relative_energy_comparisons=comparisons, engineering_checks=gate,
        all_registered_engineering_checks_passed=all(gate.values()),
        plan_sha256=sha(plan_path), per_query_sha256=sha(output/'per_query.csv'),
        GPU_used=False, seconds=time.monotonic()-started,
        formal_images_generated=0, scientific_goal_complete=False, may_launch_GPU_training=False,
        limits=['Dcheck previously exposed; exploratory feasibility, no new independent validation.',
                'Fixed tolerances are engineering checks, not proofs of distributional equality.',
                'First fixed draw retrieval;16 draws energy; all10 subjects and every Dcheck query.',
                'No image generation, image gain, convergence, reliable confidence or SOTA claim.'])
    write(output/'summary.json', summary)
    write(output/'progress.json', dict(status=summary['status'], completed_subjects=10,
        expected_subjects=10, Dcheck_queries=8300, seconds=summary['seconds'], GPU_used=False))
    print(json.dumps({k:value for k,value in summary.items() if k not in ['per_subject','classifiers']}), flush=True)


if __name__ == '__main__':
    main()
