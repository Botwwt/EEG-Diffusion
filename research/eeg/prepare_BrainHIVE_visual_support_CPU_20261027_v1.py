"""Export fixed D10 conditions from EEG and T-only dictionary; no target access."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import csv
from pathlib import Path
import resource
import shutil
import time
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_visual_support_20261027_v1 as V


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--d09-completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); invoked = time.monotonic()
    assert args.output.resolve().is_relative_to(Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1'))
    assert shutil.disk_usage(args.output.parent).free > 2 << 30
    assert not args.output.exists(); args.output.mkdir(parents=True)
    torch.set_num_threads(2); assert not torch.cuda.is_initialized()
    spec = U.read(args.spec); assert spec['schema'] == V.SCHEMA and spec['neighbors'] == 1
    for source in spec['source_pins']: U.checked(source)
    assert U.ref(args.d09_completion) == spec['D09_completion']
    previous = U.read(args.d09_completion); assert previous['status'] == 'COMPLETE_D09_STAGE400'
    assert previous['readout']['decision']['action'] == 'STOP_INSTANCE'
    inputs = U.read(U.checked(previous['inputs'])); panel = U.read(U.checked(inputs['panel']))
    split = U.read(U.checked(inputs['split'])); t_rows = sorted(panel['T_rows'])
    assert t_rows == sorted(split['image_rows']['T']) and len(t_rows) == 13230
    assert set(t_rows).isdisjoint(split['image_rows']['D'] + split['image_rows']['V'])
    metadata = {int(row['array_row']): row for row in csv.DictReader(U.checked(inputs['image_manifest']).open()) if row['dataset_split'] == 'train'}
    t_concepts = {metadata[row]['concept_id'] for row in t_rows}
    check_concepts = {query['concept_id'] for query in panel['Dcheck']}
    assert len(t_concepts) == 1323 and len(check_concepts) == 40 and t_concepts.isdisjoint(check_concepts)
    work = Path('/root/brainhive_reproduction_20261020_v1')
    contract_path = work/'data/component_input_contract_v1.json'; contract = U.read(contract_path)
    target_path = Path(contract['target_cache_path']); assert U.sha(target_path) == contract['target_cache_sha256']
    targets = torch.load(target_path, weights_only=True, map_location='cpu', mmap=True)
    assert targets.shape == (16540, 1024)
    # Only T entries are indexed. Dfit/Dcheck/V F targets are never selected or scored here.
    bank = targets[t_rows].numpy().copy(); assert bank.dtype == np.float32 and np.isfinite(bank).all()
    U.save_array(args.output/'model_dictionary_T.npy', bank)
    cache = torch.load(U.checked(inputs['native_cache']), weights_only=True, map_location='cpu', mmap=True)
    assert cache['model_sha256'] == inputs['native_checkpoint']['sha256']
    lookup = {int(row): i for i, row in enumerate(cache['array_rows'])}
    ids = [lookup[q['array_row']] for q in panel['Dcheck']]
    assert cache['array_rows'][ids].tolist() == [q['array_row'] for q in panel['Dcheck']]
    conditions = cache['condition'][ids].numpy().copy()
    selection_started = time.monotonic()
    choices, cosine, margin = V.nearest_support(conditions, bank)
    permutation = V.control_permutation(panel['Dcheck'])
    selected = bank[choices]
    output = np.stack([selected, selected[permutation]], axis=1)
    assert output.shape == (400, 2, 1024) and output.dtype == np.float32
    assert np.array_equal(output[:, 0], bank[choices]) and np.array_equal(output[:, 1], output[permutation, 0])
    U.save_array(args.output/'conditions_all400.npy', output)
    counts = np.bincount(choices, minlength=len(bank)); support_rows = np.asarray(t_rows)[choices]
    lock = dict(status='D10_ALL400_CORRECT_AND_SHUFFLED_EEG_CONDITIONS_LOCKED_BEFORE_D10_IMAGES',
        query_ids=[q['query_id'] for q in panel['Dcheck']], T_array_rows=t_rows, chosen_dictionary_indices=choices.tolist(),
        chosen_T_array_rows=support_rows.tolist(), shuffled_EEG_query_permutation=permutation.tolist(),
        best_training_support_cosine=cosine.tolist(), first_minus_second_cosine=margin.tolist(),
        model_dictionary=U.ref(args.output/'model_dictionary_T.npy'), conditions=U.ref(args.output/'conditions_all400.npy'),
        query_target_F_features_used=False, query_target_images_used=False, query_concepts_used_in_normal_selection=False,
        target_rank_or_DINO_used_for_choices=False, target_concepts_absent_from_dictionary=True,
        selection='float64 normalized EEG/F cosine, single argmax, lowest sorted T row tie; output exact raw F',
        new_fitted_weights=0, generator_or_EEG_weights_updated=False, normal_inference_only_EEG_and_fixed_training_dictionary=True,
        original_D_resumed=False, development_only=True, independent_confirmation=False)
    U.atomic(args.output/'all400_condition_lock.json', lock)
    U.atomic(args.output/'math_checks.json', V.self_test())
    result = dict(status='D10_T_ONLY_SUPPORT_AND_ALL400_CONDITIONS_CPU_COMPLETE', schema=V.SCHEMA,
        source=U.ref(__file__), spec=U.ref(args.spec), D09_completion=U.ref(args.d09_completion),
        D09_inputs=previous['inputs'], panel=inputs['panel'], lock=U.ref(args.output/'all400_condition_lock.json'),
        raw_T_F_cache=U.ref(target_path), component_contract=U.ref(contract_path), native_EEG_cache=inputs['native_cache'],
        native_checkpoint=inputs['native_checkpoint'], baseline_generated_features=previous['check_generated_features'],
        target_features=previous['check_target_features'], baseline_manifest=previous['check_manifest'],
        T_dictionary_float32_bytes=bank.nbytes, T_dictionary_entries=13230, dictionary_concepts=1323,
        unique_selected_training_images=int((counts > 0).sum()), maximum_queries_selecting_same_image=int(counts.max()),
        paired_correct_and_shuffled_conditions_identical=bool(np.array_equal(output[:, 0], output[:, 1])),
        median_nearest_cosine=float(np.median(cosine)), median_top2_margin=float(np.median(margin)),
        CPU_selection_seconds=time.monotonic()-selection_started, seconds=time.monotonic()-invoked,
        CUDA_initialized=torch.cuda.is_initialized(), query_targets_used_for_selection=False,
        earlier_Dcheck_exposure='D09 and older condition/risk diagnostics; not independent H',
        source_archived_F_cache_contains_other_partitions_but_only_T_rows_read_as_dictionary=True,
        peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        new_training_or_external_images=0, no_automatic_retry_or_sweep=True)
    assert result['CUDA_initialized'] is False
    assert result['peak_CPU_RSS_bytes'] < 10 << 30
    assert sum(path.stat().st_size for path in args.output.rglob('*') if path.is_file()) < 128 << 20
    U.atomic(args.output/'completion.json', result)
    print({k: result[k] for k in ('status','seconds','unique_selected_training_images','maximum_queries_selecting_same_image','median_nearest_cosine','median_top2_margin','T_dictionary_float32_bytes')})


if __name__ == '__main__': main()
