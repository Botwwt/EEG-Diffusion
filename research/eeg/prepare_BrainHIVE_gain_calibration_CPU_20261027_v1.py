"""D09 CPU admission, existing model verification, and blind calibration panel."""
import argparse
import json
from pathlib import Path
import time
import zipfile
import numpy as np
import torch
from brainhive_bounded_component_v1 import BoundedConditionAdapter
import brainhive_gain_calibration_20261027_v1 as U
import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('spec', 'baseline-completion', 'output'): parser.add_argument('--' + key, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    spec = U.read(args.spec); assert spec['schema'] == U.SCHEMA
    assert spec['calibration_queries'] == 200 and spec['check_gallery_queries'] == 400 and spec['check_first_queries'] == 80
    baseline = U.read(args.baseline_completion); assert baseline['status'] == 'COMPLETE_RAPID_NATIVE_BASELINE_CACHE'
    old_panel = U.read(U.checked(baseline['panel']))
    split_path = U.checked(old_panel['split']); split = U.read(split_path)
    metadata_path = U.checked(old_panel['image_manifest']); metadata = G.metadata(metadata_path)
    assert U.sha(metadata_path) == split['image_manifest_sha256']
    assert len(split['train_concepts']) == 1323 and len(split['d_fit_concepts']) == 82 and len(split['d_check_concepts']) == 83
    fit = U.choose_queries(split, metadata, 'Dfit', 20); check = U.choose_queries(split, metadata, 'Dcheck', 40)
    assert {q['concept_id'] for q in fit}.isdisjoint(q['concept_id'] for q in check)
    assert {q['array_row'] for q in fit + check}.issubset(set(split['image_rows']['D']))
    assert set(split['image_rows']['T']).isdisjoint(q['array_row'] for q in fit + check)
    assert set(split['image_rows']['V']).isdisjoint(q['array_row'] for q in fit + check)
    targets = {partition: [q | {k: metadata[q['array_row']][k] for k in ('image_zip', 'image_member', 'image_sha256', 'image_bytes')} for q in rows]
               for partition, rows in (('Dfit', fit), ('Dcheck', check))}
    containers = {}
    for items in targets.values():
        for row in items: containers.setdefault(row['image_zip'], []).append(row)
    for path, members in containers.items():
        with zipfile.ZipFile(path) as archive:
            for row in members: assert archive.getinfo(row['image_member']).file_size == int(row['image_bytes'])
    torch.set_num_threads(2)
    cache_path = WORK / 'runs/full_source_dropout_v2/sub-01/conditions.pt'
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    checkpoint_path = WORK / 'runs/full_source_dropout_v2/sub-01/brain_final.pt'
    assert cache['condition'].shape == (16540, 1024) and torch.isfinite(cache['condition']).all()
    assert set(cache['array_rows'].tolist()) == set(range(16540))
    assert cache['model_sha256'] == U.sha(checkpoint_path)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)['model']
    assert state['block.layers.0.norm.weight'].shape == state['block.layers.0.norm.bias'].shape == (1024,)
    assert (state['block.layers.0.norm.weight'].abs() > 1e-6).all()
    components = {}
    for arm in ('ordinary', 'risk'):
        completion_path = WORK / 'components/bounded_v1' / arm / 'completion.json'
        completion = U.read(completion_path); assert completion['status'] == 'COMPLETE_ALL10_REAL_OOF_THREE_PASSES_AND_FIXED_FINAL_COMPONENT'
        path = Path(completion['checkpoint_path']); assert U.sha(path) == completion['checkpoint_sha256']
        saved = torch.load(path, weights_only=True, map_location='cpu')
        assert saved['updates'] == 24807 and saved['seen_queries'] == 396900 and saved['arm'] == arm
        model = BoundedConditionAdapter(predict_risk=arm == 'risk').eval().requires_grad_(False)
        model.load_state_dict(saved['model'], strict=True)
        components[arm] = dict(checkpoint=U.ref(path), completion=U.ref(completion_path), calibration=saved['calibration'])
    native_binding = U.read(U.checked(baseline['native_parent_binding']))
    native_path = Path(native_binding['predictions_path']); assert U.sha(native_path) == native_binding['predictions_sha256']
    native_rows = G.rows(U.checked(baseline['manifest'])); first = native_rows[0]
    index = G.rows(U.checked(old_panel['index'])); position = next(i for i, q in enumerate(index) if q['row'] == first['row'])
    parity_raw = np.array(np.load(native_path, mmap_mode='r', allow_pickle=False)[position:position + 1], copy=True)
    assert parity_raw.shape == (1, 1024)
    import hashlib
    assert hashlib.sha256(parity_raw.tobytes()).hexdigest() == first['condition_sha256']
    U.save_array(args.output / 'parity_condition.npy', parity_raw)
    parity = {k: first[k] for k in ('query_id', 'image_seed', 'condition_sha256', 'condition_FP16_sha256', 'tokens_sha256',
        'initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'generated_path', 'generated_sha256', 'generated_pixel_sha256')}
    U.atomic(args.output / 'panel.json', dict(schema=U.SCHEMA, status='FIXED_D09_DFIT200_DCHECK400_BLIND_PANEL',
        Dfit=fit, Dcheck=check, T_rows=split['image_rows']['T'], V_read=False, subject='sub-01',
        selected_before_new_image_outcomes=True, development_only=True, independent_confirmation=False))
    U.atomic(args.output / 'target_metadata.json', targets)
    result = dict(status='D09_REAL_CPU_INPUTS_AND_FROZEN_MODELS_VERIFIED', schema=U.SCHEMA, panel=U.ref(args.output / 'panel.json'),
        target_metadata=U.ref(args.output / 'target_metadata.json'), native_cache=U.ref(cache_path), native_checkpoint=U.ref(checkpoint_path),
        components=components, parity_condition=U.ref(args.output / 'parity_condition.npy'), parity_reference=parity,
        baseline_completion=U.ref(args.baseline_completion), expected_host=baseline['host'], split=U.ref(split_path), image_manifest=U.ref(metadata_path),
        spec=U.ref(args.spec), source=U.ref(__file__), CPU_only=True, CUDA_initialized=torch.cuda.is_initialized(),
        Dfit_image_targets_read=False, Dcheck_image_targets_read=False, V_fit_permission_changed=False,
        sampling_scope='sub01,20/82 Dfit concepts for calibration,8/83 first Dcheck concepts; all10 stimuli each; not formal full fit/evaluation',
        seconds=time.monotonic() - started)
    assert result['CUDA_initialized'] is False
    U.atomic(args.output / 'completion.json', result)
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], calibration_queries=200, check_gallery_queries=400,
        first_check_queries=80, strict_loaded_components=2, CUDA_initialized=False))); return 0


if __name__ == '__main__': raise SystemExit(main())
