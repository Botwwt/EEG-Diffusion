"""Tiny metadata delivery fixtures; no real gallery scores or image payloads."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import quote

import verify_BrainHIVE_bounded_gallery_delivery_CPU_20261022_v1 as entry


def raw(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)


def save(path, value):
    raw(path, (json.dumps(value, indent=2) + '\n').encode())


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root):
    work = root / 'work'; gallery = work / 'gallery/delivery'; old = work / 'gallery/bounded_component_V_v1'
    image_root = root / 'private_images'; source_recipe = root / 'recipe.json'; storage_path = root / 'storage.json'
    storage = dict(allowed_private_roots=[str(image_root)])
    save(storage_path, storage)
    recipe = dict(run_names={a: 'protected_R4_' + a + '_bounded_v1' for a in entry.ARMS},
        storage_lifecycle_receipt=str(storage_path), storage_receipt_sha256=sha(storage_path))
    save(source_recipe, recipe)
    save(work / 'audit/main_storage_continuation_20261021_v1.json', dict(
        status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED', completed=list(entry.ARMS), failures={}, pid=99999999))
    save(work / 'comparison/bounded_v1/summary.json', dict(status='ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE'))
    save(old / 'coverage.json', dict(rows=2, generated_images=6, pages=1))
    raw(old / 'page_000.html', b'<html>Tiny old saved page</html>')
    targets, origins, rows, traces = [], [], [], []
    sources, scores_by_arm, manifests = {}, {}, {}
    for arm in entry.ARMS:
        scores_by_arm[arm], manifests[arm] = [], []
    for index, stimulus in enumerate(('q0.jpg', 'q1 #.jpg')):
        qid = 'sub-01/' + stimulus + '/R4'; target = work / 'gallery/V' / stimulus
        raw(target, b'Placeholder target, never opened by verifier')
        target_alias = old / 'assets/targets' / stimulus
        target_alias.parent.mkdir(parents=True, exist_ok=True); target_alias.hardlink_to(target)
        targets.append(dict(stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
            image_sha256=sha(target), image_path=str(target)))
        origins.append(dict(stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
            image_sha256=sha(target), image_zip=str(root / 'not_opened.zip'), image_member=stimulus))
        target_url = quote(Path(os.path.relpath(target_alias, gallery)).as_posix(), safe='/')
        row = dict(id=qid, subject='sub-01', concept='concept', target=target_url, targetFull=target_url,
            pageLocator='../bounded_component_V_v1/page_000.html', cells=[])
        trace = dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
            image_seed=1000 + index, target_image_zip=str(root / 'not_opened.zip'), target_image_member=stimulus,
            target_image_sha256=sha(target), score_target_path=str(target), page_locator=row['pageLocator'], position_in_old_page=index + 1)
        for number, arm in enumerate(entry.ARMS):
            run = recipe['run_names'][arm]
            path = image_root / run / 'sub-01' / (Path(stimulus).stem + '.webp')
            raw(path, ('Never decoded placeholder ' + arm + stimulus).encode())
            alias = old / 'assets/volume0' / path.relative_to(image_root)
            alias.parent.mkdir(parents=True, exist_ok=True); alias.hardlink_to(path)
            url = quote(Path(os.path.relpath(alias, gallery)).as_posix(), safe='/')
            value = .2 + .1 * number + .01 * index
            row['cells'].append(dict(label=arm.title(), full=url, thumb=url, dino=value, seed=value, object=value, blank=False))
            values = {k: (0 if k == 'DINO_Top1' else 1 if k == 'DINO_Top5' else value) for k in entry.METRICS}
            scored = dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept',
                EEG_source_stimulus_id=stimulus, target_image_path=str(target), generated_path=str(path), score_target_path=str(target),
                image_seed=1000 + index, **values)
            scores_by_arm[arm].append(scored)
            manifests[arm].append(dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
                image_seed=1000 + index, target_image_sha256=sha(target), score_target_path=str(target), generated_path=str(path),
                generated_sha256=sha(path), repeat_protocol='R4', intervention='correct', EEG_source_stimulus_id=stimulus))
            trace.update({arm + '_generated_path': str(path), arm + '_generated_sha256': sha(path), arm + '_score_target_path': str(target)})
        rows.append(row); traces.append(trace)
    def jsonl(path, values):
        raw(path, ''.join(json.dumps(v) + '\n' for v in values).encode())
    jsonl(work / 'gallery/V/manifest.jsonl', targets)
    jsonl(work / 'data/generation_inputs/V_targets.jsonl', origins)
    means = []
    for arm in entry.ARMS:
        run = recipe['run_names'][arm]
        manifest = work / 'images' / run / 'manifest.jsonl'; scores = work / 'evaluation' / run / 'scores/per_query.csv'
        jsonl(manifest, manifests[arm]); csv_write(scores, scores_by_arm[arm])
        sources[arm] = dict(manifest_path=str(manifest), manifest_sha256=sha(manifest), score_csv_path=str(scores), score_csv_sha256=sha(scores))
        for trace in traces:
            trace.update({arm + '_' + k: v for k, v in sources[arm].items()})
        means.append(dict(arm=arm, subject='sub-01', **{k: sum(float(r[k]) for r in scores_by_arm[arm]) / 2 for k in entry.METRICS}))
        save(work / 'evaluation' / run / 'low_level/summary.json', dict(status='COMPLETED', expected_images=2, actual_present=2,
            verified_and_decoded=2, missing_count=0, corrupt_or_mismatched_count=0, duplicate_hash_groups=0,
            duplicate_images_beyond_first=0, manifest_sha256=sha(manifest), gallery_manifest_sha256=sha(work / 'gallery/V/manifest.jsonl'),
            source_sha256='9d7c4a4e0304490b7934dbdaa32271556572227507ba166ab5a9bbda01fe8b7f'))
    raw(gallery / 'data.js', ('window.GALLERY=' + json.dumps(dict(labels=[a.title() for a in entry.ARMS], blinded=False, rows=rows)) + ';\n').encode())
    raw(gallery / 'index.html', b'<html>Tiny static UI; no browser QA</html>'); raw(gallery / 'README.txt', b'Tiny fixture only')
    csv_write(gallery / 'query_provenance.csv', traces); csv_write(gallery / 'per_subject_metric_means.csv', means)
    for reference in ('parent', 'ordinary'):
        cases = []
        for row, trace in zip(rows, traces):
            cases.append({k: trace[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'page_locator')} |
                {metric + '_delta': row['cells'][2][field] - row['cells'][entry.ARMS.index(reference)][field]
                 for metric, field in (('DINO2AFC', 'dino'), ('SEED', 'seed'), ('ObjectF1', 'object'))} |
                dict(risk_generated_path=trace['risk_generated_path'], reference_generated_path=trace[reference + '_generated_path']))
        for direction, sign in (('improvement_descending', -1), ('regression_ascending', 1)):
            csv_write(gallery / f'risk_minus_{reference}_{direction}_all16600.csv', sorted(cases, key=lambda r: (sign * r['DINO2AFC_delta'], r['query_id'])))
    save(gallery / 'completion.json', dict(status='COMPLETED_FULL_BOUNDED_R4_GALLERY_INDEX', queries=2, generated_images=6,
        subject_metric_rows=3, old_pages_preserved=1, browser_UI_QA_performed=False, new_training_or_scores=False,
        primary_metrics_changed=False, copied_decoded_or_linked_images=False,
        file_sha256={name: sha(gallery / name) for name in entry.OUTPUT_NAMES},
        source_and_input_sha256={str(entry.ROOT / 'build_BrainHIVE_bounded_gallery_index_CPU_v1.py'): entry.BUILDER_SHA,
            str(entry.ROOT / 'build_offline_gallery.py'): entry.UI_SHA, str(old / 'page_000.html'): sha(old / 'page_000.html')}, sources=sources))
    return gallery, dict(work=work, count=2, subjects=1, concepts=1, per_concept=2, recipe=source_recipe)


def verify(item):
    return entry.verify_gallery(item[0], entry.Metadata(time.monotonic() + 30), _fixture=item[1])


def reseal(gallery, name):
    path = gallery / 'completion.json'; done = json.loads(path.read_text()); done['file_sha256'][name] = sha(gallery / name); save(path, done)


def rejects(call, text):
    try:
        call()
    except (ValueError, KeyError, OSError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Invalid metadata accepted: ' + text)


def check():
    checks = []
    with tempfile.TemporaryDirectory(prefix='hive_delivery_CPU_') as temporary:
        base = Path(temporary)
        item = fixture(base / 'valid'); report = verify(item)
        assert report['status'] == 'CPU_TINY_FIXTURE_ONLY' and report['actual_full16600_delivery_verified'] is False
        assert report['rows'] == 2 and report['generated_files'] == 6 and report['subject_mean_rows'] == 3
        assert report['browser_UI_QA_performed'] is report['images_decoded'] is report['original_image_payloads_rehashed'] is False
        checks.append('real_builder_output_schema_sources_URLs_cases_means_lowlevel_tiny_delivery')
        item = fixture(base / 'hash'); raw(item[0] / 'index.html', b'changed')
        rejects(lambda: verify(item), 'SHA mismatch'); checks.append('output_SHA_drift_refused')
        item = fixture(base / 'new_V_scope')
        completion_path = item[0] / 'completion.json'; value = json.loads(completion_path.read_text())
        forbidden = item[1]['work'] / 'images/protected_R4_functional_alignment_v2_O/scores/per_query.csv'
        value['source_and_input_sha256'][str(forbidden)] = '0' * 64
        save(completion_path, value)
        rejects(lambda: verify(item), 'New functional score metadata')
        assert not forbidden.exists()
        checks.append('actual_new_V_run_prefix_refused_before_even_missing_file_open')
        for mode in ('sort', 'labels', 'path', 'seed', 'delta', 'low'):
            item = fixture(base / mode); gallery, options = item
            if mode in ('sort', 'labels', 'path'):
                data = json.loads((gallery / 'data.js').read_text()[len('window.GALLERY='):-2])
                if mode == 'sort': data['rows'].reverse()
                elif mode == 'labels': data['labels'][0] = 'Other'
                else: data['rows'][0]['cells'][0]['full'] = data['rows'][0]['cells'][0]['thumb'] = '../wrong.webp'
                raw(gallery / 'data.js', ('window.GALLERY=' + json.dumps(data) + ';\n').encode()); reseal(gallery, 'data.js')
                expected = {'sort': 'stable order', 'labels': 'columns', 'path': 'samefile'}[mode]
            elif mode in ('seed', 'delta'):
                name = 'query_provenance.csv' if mode == 'seed' else 'risk_minus_parent_improvement_descending_all16600.csv'
                with (gallery / name).open(newline='') as stream: rows = list(csv.DictReader(stream))
                rows[0]['image_seed' if mode == 'seed' else 'DINO2AFC_delta'] = '999'
                csv_write(gallery / name, rows); reseal(gallery, name)
                expected = 'identity differs' if mode == 'seed' else 'Case delta differs'
            else:
                low = options['work'] / 'evaluation/protected_R4_risk_bounded_v1/low_level/summary.json'
                value = json.loads(low.read_text()); value['missing_count'] = 1; save(low, value); expected = 'coverage/source'
            rejects(lambda item=item: verify(item), expected)
        checks.append('stable_sort_labels_path_seed_case_delta_and_incomplete_lowlevel_refused')
        identity = dict(controller=dict(pid=1, startticks=2, argv=['controller']), training=dict(pid=3, startticks=4, argv=['train']),
            wrapper=dict(pid=5, startticks=6, argv=['wrapper']), plan=dict(path='fixed', sha256='x'))
        import copy
        entry.same_runtime(identity, copy.deepcopy(identity))
        altered = copy.deepcopy(identity); altered['training']['startticks'] += 1
        rejects(lambda: entry.same_runtime(identity, altered), 'process identity changed')
        checks.append('PID_reuse_or_runtime_identity_change_refused')
        rejects(lambda: entry.Metadata(time.monotonic() - 1).raw(item[0] / 'completion.json'), 'time bound')
        rejects(lambda: entry.Metadata(time.monotonic() + 30).raw(base / 'forbidden.webp'), 'Payload hashing')
        checks.append('finite_deadline_and_image_payload_hash_read_refused')
        assert 'torch' not in sys.modules and 'PIL' not in sys.modules
    return dict(status='PASS_TINY_DELIVERY_SCHEMA_ONLY_NO_ACTUAL_SERVER_OR_GALLERY_READ', checks=checks,
        source_sha256={Path(entry.__file__).name: sha(entry.__file__), Path(__file__).name: sha(__file__)},
        actual_full16600_delivery_verified=False, actual_processes_or_formal_remaining_checked=False,
        real_V_scores_read=False, GPU_used=False, images_decoded_or_real_payloads_read=False, browser_UI_QA_performed=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt', type=Path); args = parser.parse_args()
    if args.receipt: assert not args.receipt.exists()
    result = check()
    if args.receipt: raw(args.receipt, (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
