"""CPU closure check of completed Conditional/RealPaired scores, without rescoring.

Checks every saved score and identity row and the saved paired differences.
It never generates images, changes metrics, selects checkpoints, or tunes a model.
"""
from pathlib import Path
import collections
import csv
import hashlib
import json
import math
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
    output = ROOT/'audit/idea_Conditional_RealPaired_full_score_closure_v1.json'
    assert not output.exists(), 'Preserve the earlier closure receipt'
    sources = {}
    arms = {}
    for arm in ['RealPaired', 'Conditional']:
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
        for key, m in manifest.items():
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
    reference, candidate = arms['RealPaired'], arms['Conditional']
    assert reference[0].keys() == candidate[0].keys()
    pair = DATA/'evaluation/paired/IDEA1_Conditional_minus_RealPaired_V_R4'
    paired, summary = indexed(pair/'paired_per_query.csv'), load(pair/'summary.json')
    assert summary['status'] == 'COMPLETED_PAIRED_FULL_PARTITION' and summary['queries'] == 16600
    assert paired.keys() == reference[0].keys()
    assert summary['plan_sha256'] == sha(ROOT/'configs/paired_statistics_plan.json')
    assert summary['source_sha256'] == sha(ROOT/'paired_analysis.py')
    for name, source in [('reference_manifest', sources['RealPaired']['manifest']),
                         ('candidate_manifest', sources['Conditional']['manifest']),
                         ('reference_metrics', sources['RealPaired']['metrics']),
                         ('candidate_metrics', sources['Conditional']['metrics'])]:
        assert summary['inputs'][name] == {k: source[k] for k in ['path', 'sha256']}
    for key, row in paired.items():
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
    sources['paired'] = {p.name: dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in pair.iterdir() if p.is_file()}
    result = dict(status='PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC',
                  queries=16600, subjects=10, concepts=166, metrics_checked=METRICS,
                  source_sha256=sha(__file__), sources=sources, GPU_used=False,
                  seconds=time.monotonic()-started, bootstrap_recomputed=False,
                  scientific_goal_complete=False, independent_increment_established=False,
                  limits=['Checks saved completed evidence; does not rerun feature extraction or image decoding.',
                          'The original independent identity audit checked every JPEG, native EEG condition and score target.',
                          'Bootstrap intervals are locked-plan descriptive V intervals, not seed variance or independent confirmation.'])
    output.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(stage='complete_Conditional_RealPaired_V_score_closure',
                                  GPU_used=False, seconds=result['seconds'], receipt=str(output),
                                  receipt_sha256=sha(output), UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'sources'}))


if __name__ == '__main__':
    main()
