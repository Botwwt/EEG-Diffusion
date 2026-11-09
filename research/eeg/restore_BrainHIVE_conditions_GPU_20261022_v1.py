"""Re-export 30 missing frozen full/OOF caches; require original exact file SHA.

No optimizer, learned-weight changes, checkpoint selection, metrics or images.
Reuse the original author's export batching and the pinned native EEG classes.
"""
import json
import os
from pathlib import Path
import time

from brainhive_native_models_v1 import ROOT, WORK, classes, sha, save_tensor, write, torch
from train_BrainHIVE_parent_v2 import infer

OUT = ROOT / 'audit/runtime_30_condition_caches_restored_20261022_v1.json'
PROGRESS = ROOT / 'audit/runtime_condition_restore_progress_20261022_v1.json'
STAGING = Path('/dev/shm/BrainHIVE_private_derived_cache_20261020_v2/condition_reexport_staging')


def main():
    assert b'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    assert not OUT.exists()
    started = time.monotonic()
    torch.set_num_threads(2); torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    contract_path = WORK / 'data/component_input_contract_v1.json'
    contract = json.loads(contract_path.read_text())
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    bindings_path = WORK / 'data/actual_input_bindings.json'
    bindings = json.loads(bindings_path.read_text())
    recipe = json.loads((ROOT / 'configs/functional_profiles_20261022_v1/O.json').read_text())
    assert sha(contract_path) == recipe['input_contract_sha256']
    for name, digest in recipe['source_sha256'].items():
        assert sha(ROOT / name) == digest, name
    scope = classes()
    records = []; total_forwards = 0
    torch.cuda.reset_peak_memory_stats()
    for subject in contract['subjects']:
        item = bindings['subjects'][subject['subject']]
        raw = None
        for source in subject['sources']:
            role = source['role']
            assert role in ('oof0', 'oof1', 'full')
            original = Path(source['condition_cache_path'])
            target = original.resolve(strict=False)
            assert original.is_symlink() and target.is_relative_to('/dev/shm')
            expected = source['condition_cache_sha256']
            assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
            if original.exists():
                assert sha(original) == expected
                records.append(dict(path=str(original), sha256=expected, reused=True)); continue
            if raw is None:
                raw = torch.load(item['train_path'], weights_only=True, mmap=True)
                assert tuple(raw.shape) == (16540, 63, 250)
            if role == 'full':
                export_rows = list(range(16540))
            else:
                # Contract rows retain exactly the original held-out export order.
                rows = source['groups']['component_train']
                assert sorted(x['cache_row'] for x in rows) == list(range(len(rows)))
                export_rows = [x['array_row'] for x in sorted(rows, key=lambda x: x['cache_row'])]
                assert len(export_rows) == len(set(export_rows)) and set(export_rows) <= set(protocol['image_rows']['T'])
            model = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).cuda()
            state = torch.load(source['checkpoint_path'], weights_only=True, map_location='cpu')
            model.load_state_dict(state['model'], strict=True)
            eeg = raw[export_rows][:, bindings['selected_indices'], :].clone().cuda()
            assert tuple(eeg.shape) == (len(export_rows), 17, 250)
            condition = infer(model, eeg)
            assert torch.isfinite(condition).all() and tuple(condition.shape) == (len(export_rows), 1024)
            staged = STAGING / role / subject['subject'] / 'conditions.pt'
            staged.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            assert not staged.exists() and not staged.with_name('conditions.pt.pending').exists()
            save_tensor(staged, dict(condition=condition, array_rows=torch.tensor(export_rows),
                model_sha256=source['checkpoint_sha256'], role=role, normalization='raw LayerNorm output'))
            assert sha(staged) == expected, 'Preserve staged re-export; never adopt a nonidentical cache'
            target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            assert not target.exists()
            staged.rename(target)
            assert sha(original) == expected
            assert sha(source['checkpoint_path']) == source['checkpoint_sha256']
            forwards = (len(export_rows) + 255) // 256
            total_forwards += forwards
            record = dict(path=str(original), resolved_path=str(target), sha256=expected,
                query_rows=len(export_rows), frozen_MLP_forwards=forwards,
                checkpoint_sha256=source['checkpoint_sha256'], original_file_bytes_identical=True, reused=False)
            records.append(record)
            write(PROGRESS, dict(completed=len(records), expected=30, last=record,
                seconds=time.monotonic() - started, no_fitting=True))
            print(json.dumps(record), flush=True)
            del model, state, eeg, condition
            torch.cuda.empty_cache()
        del raw
    assert len(records) == 30
    assert all(sha(row['path']) == row['sha256'] for row in records)
    write(OUT, dict(status='ALL30_FULL_AND_OOF_CACHES_ORIGINAL_EXACT_SHA_RESTORED',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=records,
        source_sha256=sha(__file__), original_export_source_sha256=sha(ROOT / 'train_BrainHIVE_parent_v2.py'),
        input_contract_sha256=sha(contract_path), input_bindings_sha256=sha(bindings_path),
        MLP_forwards=total_forwards, UNet_forwards=0, training_updates=0,
        final_models_modified=False, images_generated=0, metrics_read=False,
        peak_GPU_bytes=torch.cuda.max_memory_allocated(), seconds=time.monotonic() - started,
        RAM_is_volatile=True, formal_functional_training_started=False))


if __name__ == '__main__':
    main()
