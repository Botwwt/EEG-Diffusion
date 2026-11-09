"""All3200 image provenance and eight metrics; no fitting or effect-based selection."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = '2'
import argparse, hashlib, io, json, resource, sys, time, zipfile
from pathlib import Path
import numpy as np
from PIL import Image
import atm_token_split_image_contract_20261104_v1 as C
ap = argparse.ArgumentParser(); ap.add_argument('--policy', required=True); ap.add_argument('--mode', choices=('prepare', 'audit', 'score'), required=True); ap.add_argument('--stage', type=int, default=400); a = ap.parse_args()
started = time.monotonic(); rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
c = C.setup(a.policy); K, q, out = c['K'], c['q'], c['out']; C.resources(c)
assert K.ref(__file__) == q['sources']['CPU']

def guard():
    assert time.monotonic() - started < 850
    assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0 < c['parent']['maximum_extra_CPU_bytes']

def sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()

if a.mode == 'prepare':
    C.prepare(c, a.policy); C.resources(c); print(json.dumps({'status': 'COMPLETE_INPUT_PREPARATION'})); raise SystemExit(0)
assert a.stage == 400
prepared = K.read(out / 'prepared.json'); assert prepared['policy'] == K.ref(a.policy)
values = np.load(K.checked(prepared['conditions']), allow_pickle=False)
baseline_values = np.load(K.checked(prepared['baseline_conditions']), allow_pickle=False)
donors = np.load(K.checked(prepared['donors']), allow_pickle=False); assert np.array_equal(donors, C.donor_indices(c['queries']))
bank = K.read(out / 'token_bank.json'); native = np.load(K.checked(bank['native']), allow_pickle=False); mixed = np.load(K.checked(bank['mixed']), allow_pickle=False)
assert native.shape == (400, 2, 4, 2048) and mixed.shape == (400, 4, 4, 2048) and native.dtype == mixed.dtype == np.float16
old, new = native[:, 0].astype(np.float32), native[:, 1].astype(np.float32)
expected = [new - new.mean(1, keepdims=True) + old.mean(1, keepdims=True),
            old - old.mean(1, keepdims=True) + new.mean(1, keepdims=True),
            .5 * (old + new),
            new - new.mean(1, keepdims=True) + old[donors].mean(1, keepdims=True)]
assert all(np.array_equal(mixed[:, j], v.astype(np.float16)) for j, v in enumerate(expected))
queries = c['queries']; records = C.records(c, 400)
assert all([r['query_index'] for r in records[arm]] == list(range(400)) for arm in C.ARMS)
ordered = [records[arm][i] for i in range(400) for arm in C.ARMS]
if a.mode == 'audit':
    assert not (out / 'image_audit400.json').exists(); paths = set(); inodes = set(); duplicates = {}; constant = 0
    for i in range(400):
        guard()
        for arm in C.ARMS:
            row = records[arm][i]; assert row['arm'] == arm and all(row[k] == queries[i][k] for k in C.IDENTITY)
            path = K.checked(row['image']); stat = path.stat(); inode = stat.st_dev, stat.st_ino
            assert str(path) not in paths and inode not in inodes; paths.add(str(path)); inodes.add(inode)
            pixels = np.asarray(Image.open(path).convert('RGB')).copy()
            assert pixels.shape == (512, 512, 3) and sha(pixels) == row['pixels_sha256']
            assert row['noise'] == records['direct'][i]['noise'] and len(row['noise']['initial']) == 1 and len(row['noise']['ancestral']) == 4
            if arm in C.BASES:
                condition = baseline_values[i, C.BASES.index(arm)]
            else:
                j = C.MODELS.index(arm); base = 0 if arm == 'new_mean_old_center' else 1; condition = values[i, base]
                assert sha(mixed[i, j]) == row['tokens_FP16_sha256'] and row['projection_calls_per_UNet_trajectory'] == 4
                assert row['old_condition_FP32_sha256'] == sha(values[i, 0]) and row['new_condition_FP32_sha256'] == sha(values[i, 1])
                donor = int(donors[i]) if arm == 'wrong_old_mean' else i
                assert row['mean_donor_index'] == donor and row['mean_donor_EEG_SHA'] == sha(values[donor, 0])
                assert row['normal_inference_only_EEG'] == (arm != 'wrong_old_mean')
                assert row['query_concept_used_only_for_privileged_control'] == (arm == 'wrong_old_mean')
                assert not row['query_image_read_at_inference'] and row['UNet_NFE'] == 4 and row['EEG_encoder_forward_cost'] == 2
                assert row['token_bank'] == K.ref(out / 'token_bank.json') and row['prepared'] == K.ref(out / 'prepared.json')
            assert sha(condition) == row['condition_FP32_sha256'] and sha(condition.astype(np.float16)) == row['condition_FP16_sha256']
            duplicates.setdefault(row['pixels_sha256'], []).append([arm, i]); constant += int(np.max(pixels.astype(float).std(axis=(0, 1))) < 2.55)
    K.write(out / 'ordered_images400.json', ordered)
    K.write(out / 'image_audit400.json', dict(status='PASS_ALL_NATIVE_IP_SPLIT_IMAGE_PROVENANCE', stage=400, images=3200,
        ordered_records=K.ref(out / 'ordered_images400.json'), all_conditions_noise_identity_pixels_verified=True,
        all1600_mixed_P_FP16_tokens_independently_rebuilt=True, near_constant_count=constant,
        exact_duplicate_pixel_groups=[v for v in duplicates.values() if len(v) > 1],
        wrong_mean_is_privileged_control_not_method=True))
else:
    assert not (out / 'readout400.json').exists() and K.read(out / 'image_audit400.json')['status'] == 'PASS_ALL_NATIVE_IP_SPLIT_IMAGE_PROVENANCE'
    import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S
    import score_low_level as Low
    assert K.ref(S.__file__) == q['source_metrics'] and K.ref(Low.__file__) == q['source_low_level']
    features = np.load(K.checked(K.read(out / 'encode_400_completion.json')['features']), allow_pickle=False); assert features.shape == (400, 8, 768)
    target = np.load(K.checked(prepared['target_feature_ref']), allow_pickle=False); selfcheck = S.metric_arrays(target, target, queries, queries)
    assert np.all(selfcheck['DINO2AFC'] == 1) and np.all(selfcheck['gallery_top1'] == 1)
    previous = np.load(K.checked(q['baseline_metrics']), allow_pickle=False)
    arrays = {arm: {key: previous[arm + '__' + key].copy() for key in C.METRICS} for arm in C.BASES}
    arrays.update({arm: S.metric_arrays(features[:, C.ARMS.index(arm)], target, queries, queries) for arm in C.MODELS})
    metadata = K.read(K.checked(prepared['target_metadata_ref']))['Dcheck']; archives = {}
    for arm in C.MODELS:
        for key in C.METRICS[5:]: arrays[arm][key] = np.empty(400)
    try:
        for i in range(400):
            guard(); item = metadata[i]; assert all(item[k] == queries[i][k] for k in C.IDENTITY)
            if item['image_zip'] not in archives: archives[item['image_zip']] = zipfile.ZipFile(item['image_zip'])
            raw = archives[item['image_zip']].read(item['image_member']); assert hashlib.sha256(raw).hexdigest() == item['image_sha256']
            trgb, ty = Low.pixels(io.BytesIO(raw)); moments = Low.moments(ty)
            for arm in C.MODELS:
                rgb, y = Low.pixels(K.checked(records[arm][i]['image']))
                arrays[arm]['PixCorr_RGB'][i] = Low.pixcorr(rgb, trgb)
                arrays[arm]['SSIM_Y'][i] = Low.ssim_y(y, ty, moments)
                arrays[arm]['mean_RGB_absolute_error'][i] = float(abs(rgb.mean((0, 1)) - trgb.mean((0, 1))).mean())
    finally:
        for archive in archives.values(): archive.close()
    pairs = C.paired(arrays, queries); path = out / 'metrics400.npz'; temporary = path.with_name(path.name + '.pending')
    with temporary.open('xb') as f: np.savez_compressed(f, **{arm + '__' + key: value for arm, row in arrays.items() for key, value in row.items()})
    temporary.replace(path)
    K.write(out / 'readout400.json', dict(status='COMPLETE_EIGHT_ARM_NATIVE_IP_SPLIT_FULL400_PAIRED_READOUT', stage=400, queries=400, gallery=400,
        arms=list(C.ARMS), means={arm: {key: float(value.mean()) for key, value in row.items()} for arm, row in arrays.items()},
        fixed_comparisons=pairs, interpretation=C.interpret(pairs, 400), arrays=K.ref(path), bootstrap=dict(seed=1729, unit='concept', replicates=1000),
        metrics=list(C.METRICS), source_metrics=q['source_metrics'], source_low_level=q['source_low_level'],
        new_training_updates=0, new_adapter_parameters=0, two_EEG_encoder_cost_disclosed=True,
        all_results_single_seed_development=True, second_independent_method_or_SOTA=False))
guard(); C.resources(c)
print(json.dumps(dict(status='COMPLETE_' + a.mode, stage=400, seconds=time.monotonic() - started,
    extra_CPU_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 - rss0)), flush=True)
