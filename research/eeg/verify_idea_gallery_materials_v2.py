"""Check every completed paired-gallery row against immutable scored sources."""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import json
import os
import time


ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def read_rows(path, jsonl=False):
    with Path(path).open(newline='') as file:
        rows = [json.loads(line) for line in file] if jsonl else list(csv.DictReader(file))
    indexed = {row['query_id']: row for row in rows}
    assert len(indexed) == len(rows)
    return indexed


def js(path):
    text = path.read_text().strip()
    assert text.startswith('window.GALLERY=') and text.endswith(';')
    return json.loads(text[len('window.GALLERY='):-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', required=True)
    args = parser.parse_args()
    gallery = Path(args.gallery).resolve()
    assert gallery.is_relative_to(DATA/'galleries')
    proof_path = gallery/'independent_materials_check_v1.json'
    assert not proof_path.exists()
    os.nice(10)
    started = time.monotonic()
    context = json.loads((gallery/'comparison_context.json').read_text())
    completed = json.loads((gallery/'completion.json').read_text())
    assert sha(gallery/'completion.json') == context['gallery_completion_sha256']
    shown, blinded = js(gallery/'data.js'), js(gallery/'blinded/data.js')
    labels = context['complete_sources']
    assert labels == ['IDEA1_RealPaired_correct_R4', 'IDEA1_Conditional_correct_R4']
    assert shown['labels'] == completed['arms'] == labels and not shown['blinded']
    assert blinded['labels'] == ['A', 'B'] and blinded['blinded']
    assert len(shown['rows']) == len(blinded['rows']) == completed['queries'] == 16600
    manifests, scores = {}, {}
    for source in context['sources']:
        if source['name'] not in labels:
            continue
        assert sha(source['manifest']) == source['manifest_sha256']
        assert sha(source['metrics']) == source['metrics_sha256']
        manifests[source['name']] = read_rows(source['manifest'], True)
        scores[source['name']] = read_rows(source['metrics'])
    expected = sorted(manifests[labels[0]])
    assert [row['id'] for row in shown['rows']] == [row['id'] for row in blinded['rows']] == expected
    assert collections.Counter(row['subject'] for row in shown['rows']) == {f'sub-{i:02}':1660 for i in range(1,11)}
    assert len({row['concept'] for row in shown['rows']}) == 166
    with (gallery/'unblinding_and_pairing.csv').open(newline='') as file:
        unblinding = list(csv.DictReader(file))
    keyed = {(row['query_id'], row['method_label']):row for row in unblinding}
    assert len(keyed) == len(unblinding) == 33200
    assets = set()

    def asset(relative_full, relative_thumb, source_path, digest):
        token = hashlib.sha256((source_path+'|'+digest).encode()).hexdigest()[:32]
        assert relative_full == 'blinded/full/'+token+'.jpg'
        assert relative_thumb == 'blinded/thumbs/'+token+'.jpg'
        full, thumb = gallery/relative_full, gallery/relative_thumb
        own, original = full.stat(), Path(source_path).stat()
        assert own.st_ino == original.st_ino and own.st_dev == original.st_dev
        assert own.st_size > 0 and thumb.stat().st_size > 0
        assets.add(token)

    for row, blind in zip(shown['rows'], blinded['rows']):
        key = row['id']
        assert len(row['cells']) == len(blind['cells']) == 2
        assert [c['label'] for c in row['cells']] == labels
        assert [c['label'] for c in blind['cells']] == ['A','B']
        reference = manifests[labels[0]][key]
        target = scores[labels[0]][key]['score_target_path']
        assert row['subject'] == blind['subject'] == reference['subject']
        assert row['concept'] == blind['concept'] == reference['concept_id']
        assert row['targetFull'] == 'blinded/'+blind['targetFull']
        assert row['target'] == 'blinded/'+blind['target']
        asset(row['targetFull'], row['target'], target, reference['target_image_sha256'])
        for cell, label in zip(row['cells'], labels):
            source, score, pair = manifests[label][key], scores[label][key], keyed[(key,label)]
            assert score['target_image_path'] == score['score_target_path'] == target
            for field in ['stimulus_id','EEG_source_stimulus_id','generated_path']:
                assert pair[field] == source[field]
            assert pair['stimulus_id'] == pair['EEG_source_stimulus_id']
            assert pair['target_image_path'] == pair['scoring_target_path'] == target
            assert pair['target_sha256'] == source['target_image_sha256'] == reference['target_image_sha256']
            assert pair['generated_sha256'] == source['generated_sha256']
            assert int(pair['image_seed']) == source['image_seed']
            for output, metric in [('dino','DINO2AFC'),('seed','SEED'),('object','ObjectF1')]:
                assert cell[output] == float(score[metric])
            assert cell['blank'] == (score['blank_proxy']=='True')
            asset(cell['full'], cell['thumb'], source['generated_path'], source['generated_sha256'])
            counterpart = next(c for c in blind['cells'] if c['label']==pair['blind_label'])
            assert cell['full'] == 'blinded/'+counterpart['full']
            assert cell['thumb'] == 'blinded/'+counterpart['thumb']
            assert set(counterpart) == {'label','full','thumb'}
    assert len(assets) == completed['assets'] == 34860
    assert len(list((gallery/'blinded/full').glob('*.jpg'))) == len(assets)
    assert len(list((gallery/'blinded/thumbs').glob('*.jpg'))) == len(assets)
    assert len(list((gallery/'pages').glob('page_*.html'))) == completed['thumbnail_pages'] == 415
    with (gallery/'blinded/human_ratings_blank.csv').open(newline='') as file:
        assert list(csv.DictReader(file)) == []
    assert not (gallery/'blinded/unblinding_and_pairing.csv').exists()
    for case in context['paired_case_indices']:
        path = gallery/case['path']
        assert sha(path) == case['sha256']
        cases = read_rows(path)
        assert set(cases) == set(expected)
    assert len(context['paired_case_indices']) == 2
    proof = dict(status='COMPLETE16600_PAIRED_OFFLINE_AND_BLINDED_MATERIALS_VERIFIED',
        queries=16600, subjects=10, pages=415, assets=34860, pairing_rows=33200,
        complete_sources=labels, source_gallery_completion_sha256=sha(gallery/'completion.json'),
        comparison_context_sha256=sha(gallery/'comparison_context.json'),
        blinded_data_sha256=sha(gallery/'blinded/data.js'), source_sha256=sha(__file__),
        full_assets_verified_by='Exact source device/inode hardlink equality; original bytes previously fully hashed by completed gallery builder and identity audits',
        human_ratings_performed=False, browser_UI_QA_performed=False,
        scientific_goal_complete=False, missing_required_sources=context['missing_required_sources'],
        seconds=time.monotonic()-started, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    audit = ROOT/'audit/IDEA1_Conditional_RealPaired_gallery_materials_check_v1.json'
    assert not audit.exists()
    audit.write_text(proof_path.read_text())
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(stage='independent_full_paired_gallery_materials_check',
            seconds=proof['seconds'], GPU_used=False, receipt=str(audit), receipt_sha256=sha(audit)))+'\n')
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
