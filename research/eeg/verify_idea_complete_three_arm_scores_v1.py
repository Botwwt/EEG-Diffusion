"""CPU closure check of completed Frozen/RealPaired/Conditional scores, without rescoring.

Checks every saved score and identity row and the saved paired differences.
It never generates images, changes metrics, selects checkpoints, or tunes a model.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
from safetensors import safe_open
from verify_completed_idea_V_inputs_v1 import FIELDS
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
METRICS = ['DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5',
           'EffNet', 'CapSim', 'ObjectF1', 'SEED']


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def indexed(path, kind='csv'):
    with Path(path).open(newline='') as file:
        values = list(csv.DictReader(file)) if kind == 'csv' else [json.loads(s) for s in file]
    result = {r['query_id']: r for r in values}
    assert len(result) == len(values) == 16600, ('Incomplete or duplicate queries', str(path))
    return result


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-12), (a, b)


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    os.nice(10)
    output = ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json'
    assert not output.exists(), 'Preserve the earlier closure receipt'
    sources = {}
    arms = {}
    for arm in ['Frozen', 'RealPaired', 'Conditional']:
        evaluation = DATA/'evaluation/V'/f'IDEA1_{arm}_correct_R4'
        image = DATA/'images/IDEA1_interface_v1/V'/f'IDEA1_{arm}_correct_R4'
        paths = dict(manifest=image/'manifest.jsonl', metrics=evaluation/'metrics/per_query.csv',
                     identity=evaluation/'identity_after/per_query.csv',
                     summary=evaluation/'metrics/summary.json',
                     identity_summary=evaluation/'identity_after/summary.json',
                     failures=evaluation/'identity_after/failures.json',
                     low_level=evaluation/'low_level/summary.json',
                     completion=image/'completion.json', report=evaluation/'REPORT_receipt.json')
        summary, identity = load(paths['summary']), load(paths['identity_summary'])
        assert summary['status'] == 'COMPLETED' and summary['queries'] == 16600
        assert identity['verified_queries'] == identity['score_rows_checked'] == 16600
        assert identity['failed_queries'] == 0 and load(paths['failures']) == []
        assert summary['metric_source_sha256'] == sha(ROOT/'locked_metrics.py')
        manifest = indexed(paths['manifest'], 'json')
        metrics, identities = indexed(paths['metrics']), indexed(paths['identity'])
        assert manifest.keys() == metrics.keys() == identities.keys()
        assert collections.Counter(m['subject'] for m in manifest.values()) == {f'sub-{i:02}': 1660 for i in range(1, 11)}
        cells = collections.Counter((m['subject'], m['concept_id']) for m in manifest.values())
        assert len(cells) == 1660 and set(cells.values()) == {10}
        assert sha(paths['manifest']) == identity['manifest_sha256'] == load(paths['completion'])['image_manifest_sha256']
        receipt = load(paths['report'])
        assert receipt['source_metric_sha256'] == sha(paths['summary']) and receipt['manifest_sha256'] == sha(paths['manifest'])
        completion = load(paths['completion'])
        assert completion['reused_query_outputs'] == 0
        assert completion['counts'] == dict(calls=83000, sample_evaluations=664000)
        generation_job = ROOT/'audit'/f'IDEA1_{arm}_generate_job.json'
        assert load(generation_job)['status'] == 'COMPLETED' and load(generation_job)['exit_code'] == 0
        sources[arm+'_generation_job'] = dict(path=str(generation_job), sha256=sha(generation_job))
        if arm == 'Frozen':
            asset = Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')
            validation_plan = load(ROOT/'configs/idea_first_error_validation_plan_v1.json')
            assert sha(asset) == validation_plan['native_projector_asset_sha256']
            with safe_open(str(asset), framework='pt', device='cpu') as file:
                deployed = {'encoder_hid_proj.image_projection_layers.0.'+new:file.get_tensor(old).to(torch.bfloat16)
                    for new,old in [('image_embeds.weight','image_proj.proj.weight'),
                                    ('image_embeds.bias','image_proj.proj.bias'),
                                    ('norm.weight','image_proj.norm.weight'),('norm.bias','image_proj.norm.bias')]}
            checkpoint_digest = None
        else:
            training_path = DATA/'runs'/f'IDEA1_interface_{arm}_v1/completion.json'
            training = load(training_path)
            assert training['status'] == 'COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS'
            assert training['seen_training_queries'] == 132300 and training['updates'] == 8269
            assert sha(training['checkpoint']) == training['checkpoint_sha256']
            state = torch.load(training['checkpoint'], weights_only=True, map_location='cpu')
            assert state['arm'] == arm
            deployed = {name:value.to(torch.bfloat16) for name,value in state['state_dict'].items()}
            checkpoint_digest = training['checkpoint_sha256']
        assert len(deployed) == 4 and sum(v.numel() for v in deployed.values()) == 8400896
        projection_hashes = {k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in deployed.items()}
        assert completion['actual_deployed_projection_tensor_sha256'] == projection_hashes
        for key, m in manifest.items():
            assert m['projection_checkpoint_sha256'] == checkpoint_digest
            assert m['actual_deployed_projection_tensor_sha256'] == projection_hashes
            assert m['UNet_NFE'] == 40 and not m['target_image_or_visual_feature_read_at_inference']
            s, a = metrics[key], identities[key]
            assert a['status'] == 'OK' and a['native_cache_verified'] == 'True' and not a['error']
            for field in ['subject', 'stimulus_id', 'concept_id', 'EEG_source_stimulus_id', 'generated_path']:
                assert str(m[field]) == s[field] == a[field], (arm, key, field)
            assert int(a['array_row']) == m['array_row'] and int(s['image_seed']) == m['image_seed']
            assert s['target_image_path'] == s['score_target_path'] == a['score_target_path']
            assert a['target_zip'] == m['target_image_zip'] and a['target_member'] == m['target_image_member']
            assert a['target_sha256'] == m['target_image_sha256']
            close(float(s['SEED']), sum(float(s[f]) for f in ['ObjectF1', 'EffNet', 'CapSim'])/3)
        for metric in METRICS:
            close(math.fsum(float(s[metric]) for s in metrics.values())/16600, summary['summary'][metric]['mean'])
        sources[arm] = {k: dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for k, p in paths.items()}
        arms[arm] = (manifest, metrics, summary)
    comparisons = []
    for reference_arm,candidate_arm in [('Frozen','RealPaired'),('Frozen','Conditional'),('RealPaired','Conditional')]:
        reference, candidate = arms[reference_arm], arms[candidate_arm]
        assert reference[0].keys() == candidate[0].keys()
        pair = DATA/'evaluation/paired'/f'IDEA1_{candidate_arm}_minus_{reference_arm}_V_R4'
        paired, summary = indexed(pair/'paired_per_query.csv'), load(pair/'summary.json')
        assert summary['status'] == 'COMPLETED_PAIRED_FULL_PARTITION' and summary['queries'] == 16600
        assert paired.keys() == reference[0].keys()
        assert summary['plan_sha256'] == sha(ROOT/'configs/paired_statistics_plan.json')
        assert summary['source_sha256'] == sha(ROOT/'paired_analysis.py')
        for name, source in [('reference_manifest', sources[reference_arm]['manifest']),
                             ('candidate_manifest', sources[candidate_arm]['manifest']),
                             ('reference_metrics', sources[reference_arm]['metrics']),
                             ('candidate_metrics', sources[candidate_arm]['metrics'])]:
            assert summary['inputs'][name] == {k: source[k] for k in ['path', 'sha256']}
        for key, row in paired.items():
            for field in FIELDS:
                assert reference[0][key][field] == candidate[0][key][field], (reference_arm,candidate_arm,key,field)
            for field in ['subject', 'stimulus_id', 'concept_id']:
                assert row[field] == reference[1][key][field] == candidate[1][key][field]
            assert reference[1][key]['score_target_path'] == candidate[1][key]['score_target_path']
            assert reference[1][key]['generated_path'] != candidate[1][key]['generated_path']
            for metric in METRICS:
                a, b = float(reference[1][key][metric]), float(candidate[1][key][metric])
                close(float(row[metric+'_reference']), a)
                close(float(row[metric+'_candidate']), b)
                close(float(row[metric+'_delta']), b-a)
        for metric in METRICS:
            s = summary['metrics'][metric]
            close(math.fsum(float(r[metric+'_delta']) for r in paired.values())/16600, s['paired_delta'])
            close(s['reference'], reference[2]['summary'][metric]['mean'])
            close(s['candidate'], candidate[2]['summary'][metric]['mean'])
            assert len(s['CI95_subject_concept']) == 2 and all(math.isfinite(v) for v in s['CI95_subject_concept'])
        sources[f'{candidate_arm}_minus_{reference_arm}'] = {p.name: dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in pair.iterdir() if p.is_file()}
        comparisons.append(dict(reference=reference_arm,candidate=candidate_arm,queries=16600))
    result = dict(status='PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC',
                  queries=16600, subjects=10, concepts=166, metrics_checked=METRICS,
                  source_sha256=sha(__file__), sources=sources, GPU_used=False,
                  completed_arms=list(arms), comparisons=comparisons, identical_input_fields=FIELDS,
                  pretrained_and_trained_deployed_projection_hashes_verified=True,
                  seconds=time.monotonic()-started, bootstrap_recomputed=False,
                  scientific_goal_complete=False, independent_increment_established=False,
                  limits=['Checks saved completed evidence; does not rerun feature extraction or image decoding.',
                          'The original independent identity audit checked every JPEG, native EEG condition and score target.',
                          'Bootstrap intervals are locked-plan descriptive V intervals, not seed variance or independent confirmation.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(stage='complete_Frozen_RealPaired_Conditional_V_score_closure',
                                  GPU_used=False, seconds=result['seconds'], receipt=str(output),
                                  receipt_sha256=sha(output), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
