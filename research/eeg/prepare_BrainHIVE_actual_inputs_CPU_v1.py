"""Verify actual public payloads, IDs and physical-repeat EEG before training."""
import csv
import hashlib
import json
import os
from pathlib import Path
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
from brainhive_native_models_v1 import *


def main():
    started = time.monotonic()
    torch.set_num_threads(2)
    plan = json.loads((ASSETS / 'transfer_plan.json').read_text())
    verified = {}
    for item in plan['assets']:
        if item['path'].startswith('prior_H14/attn_adapter/'):
            continue  # Independent image-attention payload is unnecessary for EEG training.
        path = ASSETS / item['path']
        assert Path(str(path) + '.ready.json').exists(), 'Training payload still transferring: ' + str(path)
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], str(path)
        verified[str(path)] = {'sha256': item['sha256'], 'bytes': item['bytes']}
    write(WORK / 'audit/native_asset_verification.json', dict(status='ALL14_TRAINING_PAYLOADS_NATIVE_SHA256_VERIFIED_ATTENTION_SEPARATE', files=verified))
    images = list(csv.DictReader((ROOT / 'audit/image_manifest.csv').open()))
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    assert sha(ROOT / 'audit/image_manifest.csv') == protocol['image_manifest_sha256']
    import duckdb
    import numpy as np
    connection = duckdb.connect(':memory:')
    connection.execute('SET autoinstall_known_extensions=false')
    connection.execute('SET autoload_known_extensions=false')
    cfg = json.loads((ASSETS / 'prior_H14/fusion_encoder/config.json').read_text())
    schemas = []
    for split, count in [('train', 16540), ('test', 200)]:
        ordered = sorted([x for x in images if x['dataset_split'] == split], key=lambda x: int(x['array_row']))
        assert len(ordered) == count
        ids = [Path(x['stimulus_id']).stem for x in ordered]
        assert len(set(ids)) == count
        positions = {value: i for i, value in enumerate(ids)}
        data = {}
        for key, dim in cfg['proj_meta'].items():
            paths = sorted((ASSETS / 'features/things-eeg').glob('things_' + split + '_' + key + '-part-*.parquet'))
            assert len(paths) == 2
            tensor = torch.empty((count, dim))
            seen = set()
            for path in paths:
                cursor = connection.execute('SELECT image_id, emb FROM read_parquet(?)', [str(path)])
                n = 0
                while rows := cursor.fetchmany(256):
                    names = [str(row[0]) for row in rows]
                    assert len(set(names)) == len(names) and not set(names) & seen
                    assert set(names) <= positions.keys()
                    array = np.asarray([row[1] for row in rows], dtype=np.float32).reshape(len(rows), -1)
                    assert array.shape == (len(rows), dim) and np.isfinite(array).all()
                    tensor[[positions[value] for value in names]] = torch.from_numpy(array)
                    seen.update(names)
                    n += len(rows)
                schemas.append(dict(path=str(path), split=split, encoder=key, rows=n, dim=dim))
            assert seen == set(ids)
            data[key] = tensor
        save_tensor(WORK / 'data' / (split + '_visual_embeddings.pt'), data)
        write(WORK / 'data' / (split + '_image_ids.json'), ids)
    connection.close()
    inventory = list(csv.DictReader((ROOT / 'audit/data_manifest.csv').open()))
    existing = json.loads((ROOT / 'audit/author_reproduction_local_inputs_inventory_20261020_v1.json').read_text())
    # These paths come from the actual verified inventory, never generated substitutes.
    training = existing['EEG_training_inputs']
    assert len(training) == 10
    native_meta = json.loads(Path('/root/mced_eeg_full_montage_cache_v3/sub-01_train_layout.json').read_text())
    selected = ['P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2']
    indices = [native_meta['channels'].index(x) for x in selected]
    bindings = {}
    # The old inventory already binds every full train SHA. Rehash each payload now.
    for item in training:
        path = Path(item['path'])
        assert sha(path) == item['sha256'] and str(item['physical_repeat_count']) == '4'
        tensor = torch.load(path, weights_only=True, mmap=True)
        assert tensor.shape == (16540, 63, 250) and tensor.dtype == torch.float32
        subject = item['subject']
        candidates = [x for x in inventory if x['subject'] == subject and x['dataset_split'] == 'test'
                      and x['repeat_protocol'] == 'R80' and Path(x['path']).exists()]
        if not candidates:
            # Existing dataset files were inventoried earlier; inspect their saved layout.
            folder = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/datasets')
            names = {x.name: x for x in folder.iterdir() if x.is_file()}
            name = subject + '_test_mean80.pt'
            assert name in names and subject + '_test_mean80.layout.json' in names
            test_path = names[name]
            layout_path = names[subject + '_test_mean80.layout.json']
            layout = json.loads(layout_path.read_text())
            assert layout['repeat_protocol'] == 'R80'
            assert layout['physical_repeat_indices'] == list(range(80))
            assert layout['source_identity']['author_whole_SHA_verified']
            assert layout['channels'] == native_meta['channels']
            assert sha(test_path) == layout['output_identity']['sha256']
            test_binding = dict(path=str(test_path), sha256=sha(test_path), layout_path=str(layout_path),
                                layout_sha256=sha(layout_path), layout=layout)
        else:
            test_path = Path(candidates[0]['path'])
            assert sha(test_path) == candidates[0]['sha256']
            test_binding = dict(path=str(test_path), sha256=candidates[0]['sha256'])
        test = torch.load(test_path, weights_only=True, mmap=True)
        assert test.shape == (200, 63, 250) and torch.isfinite(test).all()
        bindings[subject] = dict(train_path=str(path), train_sha256=item['sha256'], train_repeats=4,
                                 test_R80=test_binding, test_repeats=80)
        print(json.dumps(dict(stage='ACTUAL_EEG_VERIFIED', subject=subject)), flush=True)
    write(WORK / 'data/actual_input_bindings.json', dict(subjects=bindings, selected_channels=selected,
        selected_indices=indices, channels_source_sha256=sha('/root/mced_eeg_full_montage_cache_v3/sub-01_train_layout.json'),
        train_stimuli=16540, test_stimuli=200, additional_per_channel_zscore=False,
        input_preprocessing='Author-released MVNN arrays; select actual posterior channel names; physical mean4/mean80',
        image_manifest_sha256=sha(ROOT / 'audit/image_manifest.csv'),
        split_manifest_sha256=sha(ROOT / 'configs/split_manifest.json')))
    models = prior_models('cpu')
    with torch.inference_mode():
        embeddings = torch.load(WORK / 'data/train_visual_embeddings.pt', weights_only=True)
        probe = {k: v[:2] for k, v in embeddings.items()}
        fused = models['fusion_encoder'].vision_model(**probe)
        mapped = models['fusion_encoder'].adapter(fused)
        tokens = models['proj'](mapped)
        assert fused.shape == mapped.shape == (2, 1024) and tokens.shape == (2, 4, 2048)
        assert all(torch.isfinite(x).all() for x in (fused, mapped, tokens))
    write(WORK / 'audit/input_preparation_complete.json', dict(status='ALL10_EEG_AND_PUBLIC_H14_FEATURES_VERIFIED',
        subjects=10, feature_schemas=schemas, actual_F_A_P_CPU_forward_finite=True,
        F_norm_mean=float(fused.norm(dim=-1).mean()), A_F_norm_mean=float(mapped.norm(dim=-1).mean()),
        token_shape=list(tokens.shape), default_MLP_signature_fix_required=True,
        no_pretrained_EEG_checkpoint_claimed=True, additional_normalization=False,
        GPU_work=False, seconds=time.monotonic() - started))
    print('INPUT_PREPARATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
