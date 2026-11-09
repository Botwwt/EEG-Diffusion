"""Inventory actual full-subject inputs and verify every image-ID join on server."""
from pathlib import Path
import csv
import hashlib
import json
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/datasets')
FULL = Path('/root/mced_eeg_full_montage_cache_v3')
MULTI = Path('/root/mced_eeg_multisource_cache_v1')
IMAGES = Path('/root/autodl-tmp/mced_research_bundle/external/things_eeg2_direct_v1')
FEATURES = Path('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2')


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def csv_write(p, rows):
    assert rows
    with p.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    started = time.monotonic(); torch.set_num_threads(2)
    metadata = np.load(IMAGES / 'image_metadata.npy', allow_pickle=True).item()
    receipts = json.loads((FEATURES / 'image_receipts.json').read_text())
    receipts = {(r['split'], r['index']): r for r in receipts}
    images = []
    concepts_by_split = {}
    for split, prefix in [('train', 'training_images'), ('test', 'test_images')]:
        files = metadata[split + '_img_files']; concepts = metadata[split + '_img_concepts']
        global_concepts = metadata[split + '_img_concepts_THINGS']
        assert len(files) == (16540 if split == 'train' else 200)
        assert len(set(files)) == len(files)
        concepts_by_split[split] = set(global_concepts)
        with zipfile.ZipFile(IMAGES / (prefix + '.zip')) as z:
            members = sorted(n for n in z.namelist() if n.lower().endswith(('.jpg', '.png', '.jpeg')))
            assert len(members) == len(files)
            for index, (file, concept, global_concept) in enumerate(zip(files, concepts, global_concepts)):
                member = f'{prefix}/{concept}/{file}'
                assert members[index] == member, (split, index, members[index], member)
                receipt = receipts[(split, index)]
                assert receipt['member'] == member
                raw = z.read(member)
                digest = hashlib.sha256(raw).hexdigest()
                assert digest == receipt['sha256'] and len(raw) == receipt['bytes']
                images.append(dict(dataset_split=split, array_row=index, stimulus_id=file,
                                   concept_id=str(global_concept), local_concept_id=str(concept),
                                   image_zip=str(IMAGES / (prefix + '.zip')), image_member=member,
                                   image_bytes=len(raw), image_sha256=digest,
                                   alignment='author_release_order_plus_metadata_and_zip_receipt',
                                   independent_event_label_replay=False))
        print(json.dumps(dict(stage='images_verified', split=split, count=len(files))), flush=True)
    assert not concepts_by_split['train'] & concepts_by_split['test']
    csv_write(ROOT / 'audit/image_manifest.csv', images)
    rows = []; summary = []
    for subject_n in range(1, 11):
        subject = f'sub-{subject_n:02d}'
        for split, repeat_n in [('train', 4), ('test', 1), ('test', 4), ('test', 80)]:
            filename = f'{subject}_{split}_mean{repeat_n}.pt'
            candidates = [DATA / filename, FULL / filename, MULTI / filename]
            found = [p for p in candidates if p.exists()]
            row = dict(subject=subject, dataset_split=split, repeat_protocol=f'R{repeat_n}',
                       path='', resolved_path='', status='MISSING', shape='', dtype='', bytes=0,
                       sha256='', expected_sha256='', source_asset='', source_sha256='',
                       source_whole_sha_verified=False, physical_repeat_count=repeat_n,
                       channels=63, timepoints=250, sampling_hz=250,
                       time_start_s=0, time_end_s=.996, finite=False,
                       independent_event_label_replay=False)
            if found:
                p = found[0]
                layout_candidates = [p.with_name(p.stem + '.layout.json'),
                                     p.parent / f'{subject}_{split}_layout.json',
                                     p.parent / f'{subject}_layout.json',
                                     FULL / f'{subject}_{split}_layout.json',
                                     MULTI / f'{subject}_layout.json']
                layout = next((json.loads(q.read_text()) for q in layout_candidates if q.exists()), {})
                expected = layout.get('output_identity', {}).get('sha256', '')
                # A test R4 cache cannot borrow the identity of the R80 derivative.
                if layout.get('output') != p.name:
                    expected = ''
                digest = sha(p)
                if expected:
                    assert digest == expected, (p, digest, expected)
                x = torch.load(p, weights_only=True, mmap=True)
                assert x.shape == (16540 if split == 'train' else 200, 63, 250), (p, x.shape)
                assert x.dtype == torch.float32
                finite = all(torch.isfinite(x[i:i+256]).all().item() for i in range(0, len(x), 256))
                assert finite
                raw_shape = layout.get('raw_shape', [])
                assert not raw_shape or raw_shape == ([16540, 4, 63, 250] if split == 'train' else [200, 80, 63, 250])
                source = layout.get('source_identity', layout.get('source', {}))
                asset = source.get('asset', '')
                if isinstance(asset, dict):
                    source_sha = asset.get('sha256', ''); asset = asset.get('path', '')
                else:
                    source_sha = source.get('sha256', '')
                row.update(path=str(p), resolved_path=str(p.resolve()),
                           status='VERIFIED_READY' if expected else 'TENSOR_VALID_PROVENANCE_PENDING',
                           shape='x'.join(map(str, x.shape)), dtype=str(x.dtype), bytes=p.stat().st_size,
                           sha256=digest, expected_sha256=expected, finite=finite,
                           source_asset=asset, source_sha256=source_sha,
                           source_whole_sha_verified=source.get('author_whole_SHA_verified', source.get('whole_official_LFS_SHA_verified', False)))
                del x
            rows.append(row)
        summary.append(dict(subject=subject, available=[r['repeat_protocol'] + '_' + r['dataset_split']
                            for r in rows if r['subject'] == subject and r['status'] == 'VERIFIED_READY']))
        print(json.dumps(dict(stage='subject_audited', **summary[-1])), flush=True)
    csv_write(ROOT / 'audit/data_manifest.csv', rows)
    result = dict(status='DATA_AUDIT_COMPLETED', complete_for_formal_training=all(
                  any(r['subject'] == s['subject'] and r['dataset_split'] == 'train' and r['status'] == 'VERIFIED_READY' for r in rows)
                  for s in summary),
                  training_subjects_ready=sum(r['dataset_split'] == 'train' and r['status'] == 'VERIFIED_READY' for r in rows),
                  r4_benchmark_subjects_ready=sum(r['dataset_split'] == 'test' and r['repeat_protocol'] == 'R4' and r['status'] == 'VERIFIED_READY' for r in rows),
                  images_verified=len(images), train_concepts=len(concepts_by_split['train']),
                  test_concepts=len(concepts_by_split['test']), train_test_concepts_disjoint=True,
                  input_preprocessing='author-released full 63x250 MVNN arrays, float32 mean of four physical training repetitions; no new normalization',
                  event_alignment_limit='Original acquisition/event stimulus IDs are absent from released dictionaries; semantic joins verified by published array order and full image metadata, not independent raw-event replay',
                  subjects=summary, missing=[{k:r[k] for k in ['subject','dataset_split','repeat_protocol']} for r in rows if r['status'] != 'VERIFIED_READY'],
                  image_manifest_sha256=sha(ROOT / 'audit/image_manifest.csv'),
                  data_manifest_sha256=sha(ROOT / 'audit/data_manifest.csv'), seconds=time.monotonic()-started)
    (ROOT / 'audit/data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
