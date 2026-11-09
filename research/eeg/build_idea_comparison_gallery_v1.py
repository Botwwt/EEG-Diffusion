"""CPU-only full-query materials for completed, matched IDEA1 image arms.

This entry point does not train, generate, score, choose a checkpoint, or launch
itself while results are pending. It reuses the inspected offline gallery builder.
Missing controls remain visible; a gallery of available arms is not a full study.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import html
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
ARMS = ['Frozen', 'RealPaired', 'Conditional', 'Moment', 'Iso', 'Clean']
CONDITIONS = ['correct', 'permuted', 'missing']
REQUIRED = [f'IDEA1_{a}_correct_R4' for a in ARMS] + [
    'IDEA1_Conditional_permuted_R4', 'IDEA1_Conditional_missing_R4']
COMMON_FIELDS = ['subject', 'stimulus_id', 'concept_id', 'array_row',
    'repeat_protocol', 'partition', 'image_seed', 'prior_seed',
    'initial_image_noise_sha256', 'target_image_sha256', 'target_image_zip',
    'target_image_member', 'encoder_sha256', 'generator_revision', 'steps',
    'CFG', 'IP_scale', 'UNet_NFE', 'image_interface_config_sha256',
    'full_training_config_sha256', 'recipe_sha256']


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def rows(path, kind):
    with Path(path).open(newline='') as file:
        data = list(csv.DictReader(file)) if kind == 'csv' else [json.loads(s) for s in file]
    indexed = {r['query_id']: r for r in data}
    assert len(indexed) == len(data), ('Repeated query ID', str(path))
    return indexed


def inspect_source(arm, condition, runtime_path, recipe_path):
    runtime_sha = sha(runtime_path)
    recipe_sha = sha(recipe_path)
    name = f'IDEA1_{arm}_{condition}_R4'
    image = DATA/'images/IDEA1_interface_v1/V'/name
    evaluation = DATA/'evaluation/V'/name
    required = [image/'completion.json', image/'manifest.jsonl',
        evaluation/'metrics/summary.json', evaluation/'metrics/per_query.csv',
        evaluation/'identity_after/summary.json', evaluation/'low_level/summary.json']
    absent = [str(p) for p in required if not p.is_file()]
    if absent:
        return dict(name=name, status='PENDING_COMPLETE_IMAGES_METRICS_AND_AUDITS', missing=absent), None
    complete = load(image/'completion.json')
    identity = load(evaluation/'identity_after/summary.json')
    metric = load(evaluation/'metrics/summary.json')
    low = load(evaluation/'low_level/summary.json')
    assert complete['status'] == metric['status'] == low['status'] == 'COMPLETED', name
    assert complete['queries'] == metric['queries'] == identity['verified_queries'] == 16600, name
    assert identity['expected_queries'] == identity['score_rows_checked'] == 16600, name
    assert identity['failed_queries'] == 0 and low['verified_and_decoded'] == 16600, name
    assert low['missing_count'] == low['corrupt_or_mismatched_count'] == 0, name
    assert complete['subjects'] == 10 and complete['partition'] == 'V', name
    assert complete['intervention'] == condition, name
    assert complete['image_interface_config_sha256'] == identity['runtime_sha256'] == runtime_sha, name
    assert complete['recipe_sha256'] == recipe_sha, name
    full_sha = load(runtime_path)['full_training_config_sha256']
    assert complete['full_training_config_sha256'] == identity['full_training_config_sha256'] == full_sha, name
    assert sha(image/'manifest.jsonl') == complete['image_manifest_sha256'] == identity['manifest_sha256'] == low['manifest_sha256'], name
    manifest = rows(image/'manifest.jsonl', 'json')
    scores = rows(evaluation/'metrics/per_query.csv', 'csv')
    assert len(manifest) == 16600 and manifest.keys() == scores.keys(), name
    assert collections.Counter(r['subject'] for r in manifest.values()) == {f'sub-{i:02}': 1660 for i in range(1, 11)}, name
    assert len({r['concept_id'] for r in manifest.values()}) == 166, name
    assert set(collections.Counter((r['subject'], r['concept_id']) for r in manifest.values()).values()) == {10}, name
    for key, row in manifest.items():
        score = scores[key]
        assert row['method'] == f'IDEA1_{arm}' and row['intervention'] == condition, (name, key)
        assert row['partition'] == 'V' and row['repeat_protocol'] == 'R4', (name, key)
        assert row['full_training_config_sha256'] == full_sha and row['recipe_sha256'] == recipe_sha, (name, key)
        assert row['image_interface_config_sha256'] == runtime_sha, (name, key)
        assert not row['target_image_or_visual_feature_read_at_inference'], (name, key)
        assert row['generated_path'] == score['generated_path'] and row['stimulus_id'] == score['stimulus_id'], (name, key)
        assert row['concept_id'] == score['concept_id'] and row['image_seed'] == int(score['image_seed']), (name, key)
        assert score['target_image_path'] == score['score_target_path'], (name, key)
    if arm != 'Frozen':
        training = load(DATA/'runs'/f'IDEA1_interface_{arm}_v1'/'completion.json')
        assert training['status'] == 'COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS', name
        assert training['seen_training_queries'] == 132300 and training['updates'] == 8269, name
        assert training['each_subject_query_coverage'] == [13230]*10, name
        assert training['recipe_sha256'] == recipe_sha, name
        assert sha(training['checkpoint']) == training['checkpoint_sha256'], name
        assert all(r['projection_checkpoint_sha256'] == training['checkpoint_sha256'] for r in manifest.values()), name
    entry = dict(name=name, arm=arm, condition=condition, status='COMPLETE_MATCHED_SOURCE',
        manifest=str(image/'manifest.jsonl'), manifest_sha256=sha(image/'manifest.jsonl'),
        metrics=str(evaluation/'metrics/per_query.csv'), metrics_sha256=sha(evaluation/'metrics/per_query.csv'),
        identity_sha256=sha(evaluation/'identity_after/summary.json'),
        low_level_sha256=sha(evaluation/'low_level/summary.json'),
        per_subject=metric['per_subject'], summary=metric['summary'])
    return entry, (manifest, scores)


def collect():
    runtime_path = ROOT/'configs/idea_interface_runtime_v1.json'
    recipe_path = ROOT/'configs/idea_interface_training_recipe_v1.json'
    sources = []
    baseline = None
    baseline_name = None
    all_paths = set()
    for arm in ARMS:
        for condition in CONDITIONS:
            entry, data = inspect_source(arm, condition, runtime_path, recipe_path)
            if data is not None:
                manifest, scores = data
                paths = {row['generated_path'] for row in manifest.values()}
                assert len(paths) == 16600 and not (paths & all_paths), ('Cross-method output path reuse', entry['name'])
                all_paths.update(paths)
                if baseline is None:
                    baseline = data
                    baseline_name = entry['name']
                else:
                    assert manifest.keys() == baseline[0].keys(), entry['name']
                    for key, row in manifest.items():
                        for field in COMMON_FIELDS:
                            assert row[field] == baseline[0][key][field], ('Unequal comparison contract', entry['name'], key, field)
                        assert scores[key]['score_target_path'] == baseline[1][key]['score_target_path'], (entry['name'], key)
                    entry['identical_saved_JPEG_hash_comparison'] = dict(reference=baseline_name,
                        queries=16600, identical_queries=sum(manifest[k]['generated_sha256'] == baseline[0][k]['generated_sha256'] for k in manifest),
                        limit='Byte identity only; different hashes do not establish perceptually different objects.')
            sources.append(entry)
    ready = [s for s in sources if s['status'] == 'COMPLETE_MATCHED_SOURCE']
    missing = [n for n in REQUIRED if n not in {s['name'] for s in ready}]
    return sources, ready, missing


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-ready', action='store_true')
    parser.add_argument('--output')
    args = parser.parse_args()
    os.nice(10)
    started = time.monotonic()
    sources, ready, missing = collect()
    initial = dict(status='SOURCES_CHECKED_GALLERY_NOT_BUILT', complete_sources=[s['name'] for s in ready],
        missing_required_sources=missing, scientific_goal_complete=False, human_ratings_performed=False,
        source_sha256=sha(__file__), GPU_used=False)
    if args.check_ready:
        print(json.dumps(initial), flush=True)
        return
    assert ready, 'No completed matched source: wait for actual full-image evaluation'
    assert args.output, 'Choose a fresh versioned server-side output directory'
    output = Path(args.output).resolve()
    assert output.is_relative_to((DATA/'galleries').resolve()) and output != (DATA/'galleries').resolve()
    assert output.parent.is_dir(), 'Use an existing server gallery parent directory'
    assert not output.exists(), 'Preserve every earlier gallery and failed attempt'
    command = [sys.executable, str(ROOT/'build_offline_gallery.py')]
    for source in ready:
        command += ['--arm', source['name'], source['manifest'], source['metrics']]
    command += ['--output', str(output)]
    free = shutil.disk_usage(output.parent).free
    # The original full JPEGs are hard linked. Reserve thumbnail/HTML space using
    # the maximum actual thumbnail size from the already verified old gallery.
    previous = load(ROOT/'audit/existing_REG_gallery_CPU_state_v1.json')
    thumbnails = list((Path(previous['output'])/'blinded/thumbs').glob('*.jpg'))
    assert thumbnails, 'Need a measured thumbnail storage bound before building'
    max_thumb = max(p.stat().st_size for p in thumbnails)
    estimate = int(2 * max_thumb * (1660 + 16600*len(ready)) + 256*1024**2)
    assert free >= estimate + 1024**3, ('Insufficient server disk for material export plus1GiB reserve', free, estimate)
    env = os.environ | dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    subprocess.run(command, cwd=ROOT, env=env, check=True, timeout=1800)
    completed = load(output/'completion.json')
    assert completed['queries'] == 16600 and completed['verified_generated_images'] == 16600*len(ready)
    assert completed['missing_images'] == completed['corrupt_images'] == 0 and completed['thumbnail_pages'] == 415
    initial.update(status='AVAILABLE_MATCHED_FULL_QUERY_GALLERY_COMPLETED',
        full_required_image_materials_complete=not missing, required_source_names=REQUIRED,
        sources=sources, gallery_completion_sha256=sha(output/'completion.json'),
        builder_sha256=sha(ROOT/'build_offline_gallery.py'), seconds=time.monotonic()-started,
        measured_storage_estimate_bytes=estimate, free_bytes_before_export=free,
        browser_UI_QA_performed=False, storage='Exact full JPEG hard links; thumbnail bytes remain on server',
        limits=['Missing source arms are visible and are not completed negative results.',
            'Quality noninferiority, independent confirmation, training convergence and SOTA are not established by material completeness.',
            'No human ratings; browser interaction QA is a separate later step.'])
    notice = ('同一新编码器的完整 V：每列16,600 query。尚缺必需列：' +
        ('、'.join(missing) if missing else '本入口要求的列均已出图和评价；其余科学门槛仍需审查。'))
    page = output/'index.html'
    extra = '<p class="notice">'+html.escape(notice)+'</p><p><a href="per_subject_metric_means.csv">每被试指标</a> · <a href="case_indices.html">完整提升/下降案例索引</a></p>'
    page.write_text(page.read_text().replace('<body>', '<body>'+extra, 1))
    with (output/'per_subject_metric_means.csv').open('x', newline='') as file:
        names = ['method', 'subject', 'DINO2AFC', 'SEED', 'ObjectF1', 'CapSim', 'EffNet']
        writer = csv.DictWriter(file, fieldnames=names)
        writer.writeheader()
        for source in ready:
            for subject, values in sorted(source['per_subject'].items()):
                writer.writerow(dict(method=source['name'], subject=subject, **{k: values[k] for k in names[2:]}))
    by_name = {s['name']: s for s in ready}
    linked = []
    pending_pairs = []
    for reference in ['Frozen', 'RealPaired', 'Moment', 'Iso', 'Clean']:
        pair_name = f'IDEA1_Conditional_minus_{reference}_V_R4'
        pair = DATA/'evaluation/paired'/pair_name
        candidate_key, reference_key = 'IDEA1_Conditional_correct_R4', f'IDEA1_{reference}_correct_R4'
        if not (pair/'summary.json').is_file() or candidate_key not in by_name or reference_key not in by_name:
            pending_pairs.append(pair_name)
            continue
        pair_summary = load(pair/'summary.json')
        assert pair_summary['status'] == 'COMPLETED_PAIRED_FULL_PARTITION' and pair_summary['queries'] == 16600
        assert pair_summary['inputs']['candidate_manifest']['sha256'] == by_name[candidate_key]['manifest_sha256']
        assert pair_summary['inputs']['reference_manifest']['sha256'] == by_name[reference_key]['manifest_sha256']
        for direction in ['largest_gain', 'largest_decline']:
            source = pair/f'{direction}_all_cases.csv'
            relative = f'{pair_name}_{direction}_all_cases.csv'
            os.link(source, output/relative)
            linked.append(dict(label=f'Conditional − {reference}: {direction} (全部16,600行)', path=relative, sha256=sha(source)))
    body = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>完整差值案例索引</title><h1>完整配对案例索引</h1><p>仅为诊断排序，不选择生成种子，也不改变评价分母。</p><ul>'
    body += ''.join('<li><a href="'+html.escape(x['path'])+'">'+html.escape(x['label'])+'</a></li>' for x in linked)
    body += '</ul><p>待完成配对比较：'+html.escape('、'.join(pending_pairs) or '无')+'</p><a href="index.html">返回主画廊</a></html>'
    (output/'case_indices.html').write_text(body)
    initial.update(paired_case_indices=linked, pending_paired_case_indices=pending_pairs)
    (output/'comparison_context.json').write_text(json.dumps(initial, indent=2)+'\n')
    print(json.dumps({k:v for k,v in initial.items() if k != 'sources'}), flush=True)


if __name__ == '__main__':
    main()
