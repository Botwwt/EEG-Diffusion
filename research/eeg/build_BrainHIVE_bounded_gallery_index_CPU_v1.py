"""Index the completed three-arm R4 gallery; reuse its server files and old UI."""
import argparse
import collections
import csv
import hashlib
import html
import io
import json
import math
import os
from pathlib import Path
import shutil
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ARMS = ('parent', 'ordinary', 'risk')
METRICS = ('DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5', 'EffNet', 'CapSim', 'ObjectF1', 'SEED')
MAIN_DONE = 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
COMPARISON_DONE = 'ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE'
MAX_BYTES, MAX_SECONDS, BUFFER = 128 << 20, 300, 512 << 20
NOTICE = ('服务器文件依赖：必须保留原图库 assets；这不是便携离线包。full/thumb 均引用已有原图，'
    '缩略尺寸由浏览器显示，没有复制、解码或新建缩略图。blank 是旧全 RGB 标准差代理；'
    'DINO/ObjectF1 筛选是指标代理，不是人工对象错误判定。R4 打乱/缺失 EEG 对照尚未运行，不能混入 R80。'
    '全量改善/退步排序仅供诊断，不调参、不选种子、不重评分；未实施人工质量评分。')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path, pins, kind='json'):
    path = Path(path)
    content = path.read_bytes()
    pins[str(path)] = hashlib.sha256(content).hexdigest()
    text = content.decode('utf-8')
    if kind == 'csv':
        return list(csv.DictReader(io.StringIO(text, newline='')))
    return [json.loads(line) for line in text.splitlines()] if kind == 'jsonl' else json.loads(text)


def indexed(rows, key='query_id'):
    result = {row[key]: row for row in rows}
    assert len(result) == len(rows), 'Duplicate source identity'
    return result


def process_live(pid):
    assert os.name == 'posix' and Path('/proc').is_dir(), 'Actual Linux /proc evidence required'
    try:
        return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()[0] not in ('Z', 'X')
    except (FileNotFoundError, ProcessLookupError):
        return False


def prerequisites(work=WORK):
    paths = [work / 'comparison/bounded_v1/summary.json', work / 'gallery/bounded_component_V_v1/coverage.json',
             work / 'audit/main_storage_continuation_20261021_v1.json']
    pins = {}
    if not paths[2].is_file():
        return None
    main = read(paths[2], pins)
    if main.get('status') != MAIN_DONE or not main.get('pid') or process_live(main['pid']):
        return None
    assert set(main['completed']) == set(ARMS) and not main['failures']
    if not paths[1].is_file():
        return None
    coverage = read(paths[1], pins)
    if coverage.get('rows') != 16600 or coverage.get('generated_images') != 49800 or coverage.get('pages') != 166:
        return None
    if not paths[0].is_file():
        return None
    comparison = read(paths[0], pins)
    if comparison.get('status') != COMPARISON_DONE:
        return None
    return comparison, main, pins


def asset_url(path, output, old, roots, target=False):
    source = Path(path)
    if target:
        asset = old / 'assets/targets' / source.name
    else:
        matches = [(i, source.relative_to(root)) for i, root in enumerate(roots) if source.is_relative_to(root)]
        assert len(matches) == 1, 'Generated file must belong to one registered private volume'
        index, relative = matches[0]
        asset = old / 'assets' / f'volume{index}' / relative
    assert asset.is_file() and source.is_file() and os.path.samefile(asset, source), 'Old asset URL resolves to a different file'
    return quote(Path(os.path.relpath(asset, output)).as_posix(), safe='/')


def join(manifests, scores, targets, summaries, comparison, urls, sources, expected_n=16600):
    keys = sorted(manifests['parent'])
    assert len(keys) == expected_n
    assert len(comparison['table']) == 3 and {row['arm'] for row in comparison['table']} == set(ARMS)
    for arm in ARMS:
        assert set(keys) == set(manifests[arm]) == set(scores[arm])
    assert set(targets) == {manifests['parent'][key]['stimulus_id'] for key in keys}
    subjects = sorted({manifests['parent'][key]['subject'] for key in keys})
    if expected_n == 16600:
        assert len(subjects) == 10 and len(targets) == 1660
        assert collections.Counter(row['subject'] for row in manifests['parent'].values()) == dict.fromkeys(subjects, 1660)
        assert len({str(row['concept_id']) for row in manifests['parent'].values()}) == 166
        assert set(collections.Counter((row['subject'], str(row['concept_id'])) for row in manifests['parent'].values()).values()) == {10}
    rows, trace, cases, subject_rows, all_paths = [], [], {'parent': [], 'ordinary': []}, [], set()
    for index, key in enumerate(keys):
        first = manifests['parent'][key]
        target = targets[first['stimulus_id']]
        subject, stimulus, concept = first['subject'], first['stimulus_id'], str(target['concept_id'])
        assert key == f'{subject}/{stimulus}/R4' and int(first['array_row']) == int(target['array_row'])
        page = f'../bounded_component_V_v1/page_{index // 100:03d}.html'
        target_path = str(WORK / 'gallery/V' / Path(stimulus).name)
        target_url = urls(target_path, True)
        record = dict(id=key, subject=subject, concept=concept, target=target_url, targetFull=target_url, pageLocator=page, cells=[])
        provenance = dict(query_id=key, subject=subject, stimulus_id=stimulus, concept_id=concept,
            array_row=first['array_row'], image_seed=first['image_seed'], target_image_zip=target['image_zip'],
            target_image_member=target['image_member'], target_image_sha256=target['image_sha256'],
            score_target_path=target_path, page_locator=page, position_in_old_page=index % 100 + 1)
        query_paths = []
        for arm in ARMS:
            row, score = manifests[arm][key], scores[arm][key]
            for field in ('query_id', 'subject', 'stimulus_id', 'image_seed', 'array_row'):
                assert row[field] == first[field], (key, arm, field)
            assert str(row['concept_id']) == concept and row['target_image_sha256'] == target['image_sha256']
            assert row['EEG_source_stimulus_id'] == stimulus and row['repeat_protocol'] == 'R4' and row['intervention'] == 'correct'
            assert row['score_target_path'] == score['target_image_path'] == score['score_target_path'] == target_path
            for field in ('query_id', 'subject', 'stimulus_id', 'EEG_source_stimulus_id', 'generated_path'):
                assert score[field] == row[field], (key, arm, field)
            assert str(score['concept_id']) == concept and int(score['image_seed']) == row['image_seed']
            for field in ('array_row', 'target_image_sha256'):
                if field in score:
                    assert str(score[field]) == str(row[field])
            assert row['generated_path'] not in all_paths
            all_paths.add(row['generated_path']); query_paths.append(row['generated_path'])
            assert len(row['generated_sha256']) == 64 and score['blank_proxy'] in ('True', 'False')
            values = {metric: float(score[metric]) for metric in METRICS}
            assert all(math.isfinite(value) for value in values.values())
            url = urls(row['generated_path'], False)
            record['cells'].append(dict(label=arm.title(), full=url, thumb=url, dino=values['DINO2AFC'],
                seed=values['SEED'], object=values['ObjectF1'], blank=score['blank_proxy'] == 'True'))
            provenance.update({arm + '_' + name: value for name, value in dict(generated_path=row['generated_path'],
                generated_sha256=row['generated_sha256'], score_target_path=score['score_target_path'], **sources[arm]).items()})
        assert not any(os.path.samefile(a, b) for a, b in itertools_pairs(query_paths)), 'Cross-arm file reuse'
        rows.append(record); trace.append(provenance)
        for reference in cases:
            cases[reference].append(dict(query_id=key, subject=subject, stimulus_id=stimulus, concept_id=concept,
                page_locator=page, **{metric + '_delta': float(scores['risk'][key][metric]) - float(scores[reference][key][metric])
                for metric in ('DINO2AFC', 'SEED', 'ObjectF1')}, risk_generated_path=manifests['risk'][key]['generated_path'],
                reference_generated_path=manifests[reference][key]['generated_path']))
    for arm in ARMS:
        summary = summaries[arm]
        assert summary['status'] == 'COMPLETED' and summary['queries'] == expected_n
        assert summary['repeat_protocol'] == 'R4' and summary['intervention'] == 'correct'
        assert set(summary['per_subject']) == set(subjects)
        compared = next(row for row in comparison['table'] if row['arm'] == arm)
        for metric in METRICS:
            average = math.fsum(float(scores[arm][key][metric]) for key in keys) / expected_n
            assert abs(average - summary['summary'][metric]['mean']) <= 2e-6, (arm, metric)
            if metric in compared:
                assert abs(average - compared[metric]) <= 2e-6
        for subject in subjects:
            selected = [key for key in keys if manifests[arm][key]['subject'] == subject]
            values = summary['per_subject'][subject]
            for metric in METRICS:
                assert abs(math.fsum(float(scores[arm][key][metric]) for key in selected) / len(selected) - values[metric]) <= 2e-6
            subject_rows.append(dict(arm=arm, subject=subject, **{metric: values[metric] for metric in METRICS}))
    return rows, trace, subject_rows, cases


def itertools_pairs(values):
    return [(values[i], values[j]) for i in range(len(values)) for j in range(i)]


def csv_bytes(rows):
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader(); writer.writerows(rows)
    return output.getvalue().encode('utf-8')


def build(recipe_path, output):
    started = time.monotonic()
    gate = prerequisites()
    if gate is None:
        print(json.dumps(dict(status='PREREQUISITES_NOT_COMPLETE', scores_opened=False, output_created=False)), flush=True)
        return None
    comparison, main_state, pins = gate
    output, old = Path(output).resolve(), WORK / 'gallery/bounded_component_V_v1'
    assert output.parent == (WORK / 'gallery').resolve() and output.parent.is_dir()
    assert not output.exists(), 'Fresh direct child of the server gallery directory required'
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    recipe = read(recipe_path, pins)
    assert main_state['recipe_sha256'] == main_state['recipe_sha256_at_exit'] == pins[str(Path(recipe_path))]
    assert recipe['formal_V_queries_per_arm'] == 16600 and set(recipe['run_names']) == set(ARMS)
    assert recipe['training_seed'] == 1729 and recipe['generator'] == dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1)
    for name, digest in recipe['source_sha256'].items():
        assert Path(name).name == name and sha(ROOT / name) == digest
        pins[str(ROOT / name)] = digest
    for name, digest in [('compare_BrainHIVE_bounded_V_CPU_v1.py', recipe['comparison_source_sha256'])]:
        assert sha(ROOT / name) == digest; pins[str(ROOT / name)] = digest
    for name in (Path(__file__).name, 'build_offline_gallery.py', 'score_full_manifest.py'):
        pins[str(ROOT / name)] = sha(ROOT / name)
    storage = read(recipe['storage_lifecycle_receipt'], pins)
    assert pins[recipe['storage_lifecycle_receipt']] == recipe['storage_receipt_sha256']
    roots = [Path(path) for path in storage['allowed_private_roots']]
    targets = indexed(read(WORK / 'data/generation_inputs/V_targets.jsonl', pins, 'jsonl'), 'stimulus_id')
    manifests, scores, summaries, sources = {}, {}, {}, {}
    for arm in ARMS:
        run = recipe['run_names'][arm]; assert Path(run).name == run
        image, evaluation = WORK / 'images' / run, WORK / 'evaluation' / run / 'scores'
        manifest, score_csv = image / 'manifest.jsonl', evaluation / 'per_query.csv'
        proof = read(image / 'pairing_audit.json', pins)
        assert proof['status'] == 'ALL_SAVED_IMAGES_TARGETS_AND_STIMULUS_ROWS_JOIN_VERIFIED' and proof['queries'] == 16600
        manifests[arm] = indexed(read(manifest, pins, 'jsonl')); scores[arm] = indexed(read(score_csv, pins, 'csv'))
        assert pins[str(manifest)] == proof['manifest_sha256']
        assert Path(proof['gallery_manifest']) == WORK / 'gallery/V/manifest.jsonl'
        assert sha(proof['gallery_manifest']) == proof['gallery_manifest_sha256']
        pins[proof['gallery_manifest']] = proof['gallery_manifest_sha256']
        summaries[arm] = read(evaluation / 'summary.json', pins)
        sources[arm] = dict(manifest_path=str(manifest), manifest_sha256=pins[str(manifest)],
            score_csv_path=str(score_csv), score_csv_sha256=pins[str(score_csv)])
    for index in range(166):
        path = old / f'page_{index:03d}.html'; pins[str(path)] = sha(path)
    urls = lambda path, target: asset_url(path, output, old, roots, target)
    rows, trace, subject_rows, cases = join(manifests, scores, targets, summaries, comparison, urls, sources)
    from build_offline_gallery import app_page
    files = {'data.js': ('window.GALLERY=' + json.dumps(dict(labels=[a.title() for a in ARMS], rows=rows, blinded=False),
        separators=(',', ':'), ensure_ascii=False) + ';\n').encode(), 'query_provenance.csv': csv_bytes(trace),
        'per_subject_metric_means.csv': csv_bytes(subject_rows), 'README.txt': (NOTICE + '\n').encode()}
    links = ['<a href="query_provenance.csv">全量来源与旧页定位</a>', '<a href="per_subject_metric_means.csv">30行被试均值</a>']
    for reference, values in cases.items():
        for direction, sign in (('improvement_descending', -1), ('regression_ascending', 1)):
            name = f'risk_minus_{reference}_{direction}_all16600.csv'
            files[name] = csv_bytes(sorted(values, key=lambda row: (sign * row['DINO2AFC_delta'], row['query_id'])))
            links.append(f'<a href="{name}">risk − {reference}: {direction} 全16600行</a>')
    locator = '<p><input id="old-query" placeholder="完整query ID"><button id="old-jump">跳转原100行分页</button><span id="old-result"></span></p>'
    script = '''<script>document.getElementById('old-jump').onclick=()=>{const q=document.getElementById('old-query').value;const r=window.GALLERY.rows.find(x=>x.id===q);const e=document.getElementById('old-result');e.textContent=r?'':'未找到完整query ID';if(r)window.location.href=r.pageLocator;};</script>'''
    page = app_page(False).replace('<body>', '<body><p class="notice">' + html.escape(NOTICE) + '</p><p>' + ' · '.join(links) + '</p>' + locator, 1)
    files['index.html'] = page.replace('</body>', script + '</body>', 1).encode()
    completion = dict(status='COMPLETED_FULL_BOUNDED_R4_GALLERY_INDEX', queries=16600, generated_images=49800,
        subject_metric_rows=len(subject_rows), old_pages_preserved=166, controls='R4 permuted/missing not run; R80 separate',
        source_and_input_sha256=pins, sources=sources, all_asset_URLs_samefile_checked=True,
        copied_decoded_or_linked_images=False, primary_metrics_changed=False, new_training_or_scores=False,
        browser_UI_QA_performed=False, notice=NOTICE, file_sha256={name: hashlib.sha256(data).hexdigest() for name, data in files.items()})
    free = shutil.disk_usage(output.parent).free
    completion['storage'] = dict(fresh_free_bytes=free, main_remaining_lifecycle_bytes=0,
        main_remaining_zero_reason='All three original lifecycles and comparison saved; authoritative main terminal and PID inactive.',
        extra_temporary_factor=2, retained_buffer_bytes=BUFFER, maximum_output_bytes=MAX_BYTES)
    completion['seconds_before_write'] = time.monotonic() - started
    files['completion.json'] = (json.dumps(completion, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode()
    encoded = sum(map(len, files.values()))
    assert encoded <= MAX_BYTES and free >= 2 * encoded + BUFFER
    assert time.monotonic() - started < MAX_SECONDS
    for path, digest in pins.items():
        assert sha(path) == digest, 'Input/source drift before output'
    output.mkdir(exist_ok=False)
    for name, data in files.items():
        if name != 'completion.json':
            with (output / name).open('xb') as stream: stream.write(data)
    for path, digest in pins.items():
        assert sha(path) == digest, 'Input/source drift after output; incomplete artifacts preserved'
    assert time.monotonic() - started < MAX_SECONDS
    with (output / 'completion.json').open('xb') as stream: stream.write(files['completion.json'])
    print(json.dumps(dict(status=completion['status'], output=str(output), encoded_bytes=encoded)), flush=True)
    return completion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recipe', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); build(args.recipe, args.output)


if __name__ == '__main__':
    main()
