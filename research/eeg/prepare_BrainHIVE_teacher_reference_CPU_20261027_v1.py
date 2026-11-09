"""D11 cache identity and CPU F replay, then privileged teacher condition lock.

Public feature parquet files were retired earlier. This checks the historically
hashed assembled public-feature cache and all stimulus IDs, not a fresh visual
encoder execution from the JPEGs or a new download/rejoin of the public shards.
"""
import argparse
from contextlib import ExitStack
import csv
from pathlib import Path
import resource
import shutil
import time
import zipfile
import numpy as np
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_teacher_reference_20261027_v1 as V
from brainhive_native_models_v1 import prior_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--d10-completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert args.output.resolve().is_relative_to(Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/rapid_iteration_20261027_v1'))
    assert shutil.disk_usage(args.output.parent).free > 2 << 30
    assert not args.output.exists(); args.output.mkdir(parents=True)
    torch.set_num_threads(2); assert not torch.cuda.is_initialized()
    spec = U.read(args.spec); assert spec['schema'] == V.SCHEMA and spec['diagnostic_only'] is True
    for ref in spec['source_pins']: U.checked(ref)
    assert U.ref(args.d10_completion) == spec['D10_completion']
    d10 = U.read(args.d10_completion); assert d10['status'] == 'COMPLETE_D10_STAGE80'
    prepared = U.read(U.checked(d10['inputs'])); d09 = U.read(U.checked(prepared['D09_completion']))
    inputs = U.read(U.checked(prepared['D09_inputs'])); panel = U.read(U.checked(inputs['panel']))
    protocol = U.read(U.checked(inputs['split'])); metadata_path = U.checked(inputs['image_manifest'])
    metadata = sorted([row for row in csv.DictReader(metadata_path.open()) if row['dataset_split'] == 'train'], key=lambda x: int(x['array_row']))
    assert [int(row['array_row']) for row in metadata] == list(range(16540))
    work = Path('/root/brainhive_reproduction_20261020_v1')
    ids_path = work/'data/train_image_ids.json'; ids = U.read(ids_path)
    assert ids == [Path(row['stimulus_id']).stem for row in metadata] and len(set(ids)) == 16540
    visual_path = work/'data/train_visual_embeddings.pt'
    assert U.sha(visual_path) == spec['historical_visual_cache_expected_sha256']
    source_features = torch.load(visual_path, map_location='cpu', mmap=True, weights_only=True)
    fused_path = U.checked(prepared['raw_T_F_cache']); fused = torch.load(fused_path, map_location='cpu', mmap=True, weights_only=True)
    assert fused.shape == (16540, 1024)
    rows = [q['array_row'] for q in panel['Dcheck']]
    assert len(rows) == len(set(rows)) == 400 and set(rows).issubset(protocol['image_rows']['D'])
    assert set(rows).isdisjoint(protocol['image_rows']['T'] + protocol['image_rows']['V'])
    targets_metadata = U.read(U.checked(inputs['target_metadata']))['Dcheck']
    with ExitStack() as stack:
        archives = {}
        for i, query in enumerate(panel['Dcheck']):
            assert ids[query['array_row']] == Path(query['stimulus_id']).stem
            item = targets_metadata[i]
            for key,value in query.items(): assert item[key] == value
            if item['image_zip'] not in archives:
                archives[item['image_zip']] = stack.enter_context(zipfile.ZipFile(item['image_zip']))
            raw = archives[item['image_zip']].read(item['image_member'])
            import hashlib
            assert hashlib.sha256(raw).hexdigest() == item['image_sha256'] and len(raw) == int(item['image_bytes'])
    # An independent CPU full-precision replay of the original fused F cache.
    # Its saved source used GPU BF16; small arithmetic differences are expected.
    models = prior_models('cpu'); model = models['fusion_encoder'].vision_model
    expected_dims = U.read(work/'assets/prior_H14/fusion_encoder/config.json')['proj_meta']
    assert set(source_features) == set(expected_dims)
    for key, dim in expected_dims.items(): assert source_features[key].shape == (16540, dim)
    predicted = []
    with torch.inference_mode():
        for lo in range(0, 400, 64):
            predicted.append(model(**{key: value[rows[lo:lo+64]].clone() for key,value in source_features.items()}).float())
    predicted = torch.cat(predicted); cached = fused[rows].clone()
    relative = (predicted-cached).norm(dim=1)/cached.norm(dim=1)
    cos = torch.nn.functional.cosine_similarity(predicted, cached, dim=1)
    numerical_pass = bool((relative <= .02).all() and (cos >= .999).all())
    proof = dict(status='ALL400_CPU_F_REPLAY_NUMERICALLY_MATCHES_SAVED_GPU_BF16' if numerical_pass else 'CPU_F_REPLAY_UNRESOLVED_NO_GPU_ADMISSION',
        query_count=400, all16540_ID_rows_match_manifest=True, source_visual_cache=U.ref(visual_path),
        IDs=U.ref(ids_path), source_fused_cache=U.ref(fused_path), minimum_cosine=float(cos.min()),
        maximum_relative_L2_error=float(relative.max()), maximum_absolute_error=float((predicted-cached).abs().max()),
        stored_GPU_BF16_vs_CPU_FP32_tolerance=dict(maximum_relative_L2=.02, minimum_cosine=.999),
        raw_JPEG_feature_encoder_independently_replayed=False, fresh_public_parquet_rejoin=False,
        original_feature_preparation_receipt=U.ref(work/'audit/input_preparation_complete.json'),
        fusion_weights=U.ref(work/'assets/prior_H14/fusion_encoder/diffusion_pytorch_model.safetensors'),
        fusion_config=U.ref(work/'assets/prior_H14/fusion_encoder/config.json'),
        public_source_provenance='historically hashed assembled cache from public image_id keyed parquet; original preparation source verified separately',
        JPEG_file_bytes_verified=400, model_or_cache_modified=False, CUDA_initialized=torch.cuda.is_initialized())
    U.atomic(args.output/'F_source_replay.json', proof)
    assert numerical_pass, 'CPU F replay is unresolved; retain evidence, do not promote or rewrite cache'
    U.save_array(args.output/'CPU_F_replay_all400.npy', predicted.numpy())
    mapping = V.source_rows(panel['Dcheck']); raw_F = cached.numpy()
    d10_lock = U.read(U.checked(d10['condition_lock']))
    assert mapping[:, 2].tolist() == d10_lock['shuffled_EEG_query_permutation']
    conditions = np.stack([raw_F[mapping[:, j]] for j in range(3)], axis=1)
    assert conditions.shape == (400, 3, 1024) and conditions.dtype == np.float32
    U.save_array(args.output/'teacher_conditions_all400.npy', conditions)
    source_array_rows = np.asarray(rows)[mapping]
    lock = dict(status='D11_ALL400_PRIVILEGED_TEACHER_INTERVENTIONS_FROZEN_BEFORE_GENERATION',
        query_ids=[q['query_id'] for q in panel['Dcheck']], source_query_indices=mapping.tolist(),
        source_array_rows=source_array_rows.tolist(), conditions=U.ref(args.output/'teacher_conditions_all400.npy'),
        source_F_cache=U.ref(fused_path), arms=list(V.ARMS), diagnostic_only=True,
        oracle_uses_query_visual_supervision=True, query_DINO_metrics_used_for_condition_selection=False,
        normal_inference_supported=False, automatic_candidate_promotion=False, model_or_cache_modified=False)
    U.atomic(args.output/'all400_teacher_lock.json', lock)
    assert not torch.cuda.is_initialized()
    result = dict(status='D11_TEACHER_SOURCE_AND_ALL400_CONDITIONS_CPU_COMPLETE', schema=V.SCHEMA,
        source=U.ref(__file__), spec=U.ref(args.spec), D10_completion=U.ref(args.d10_completion),
        D09_completion=prepared['D09_completion'], D09_inputs=prepared['D09_inputs'], panel=prepared['panel'],
        baseline_generated_features=prepared['baseline_generated_features'], target_features=prepared['target_features'],
        baseline_manifest=prepared['baseline_manifest'], d10_generated_features=d10['generated_features'],
        lock=U.ref(args.output/'all400_teacher_lock.json'), F_source_replay=U.ref(args.output/'F_source_replay.json'),
        F_cache_numeric_replay_passed=True,
        diagnostic_only=True, normal_inference_supported=False, automatic_candidate_promotion=False,
        oracle_uses_query_visual_supervision=True, new_fitted_weights=0, original_D_resumed=False,
        peak_CPU_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, seconds=time.monotonic()-started,
        independent_confirmation=False, SOTA_established=False)
    assert result['peak_CPU_RSS_bytes'] < 10 << 30
    U.atomic(args.output/'completion.json', result)
    print({key:result[key] for key in ('status','seconds','peak_CPU_RSS_bytes')}, proof)


if __name__ == '__main__': main()
