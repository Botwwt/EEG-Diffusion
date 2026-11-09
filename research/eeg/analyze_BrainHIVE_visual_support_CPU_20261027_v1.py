"""D10 complete identity/condition/pixel audit and secondary F-space diagnosis."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import brainhive_gain_calibration_20261027_v1 as U
import brainhive_visual_support_20261027_v1 as V


def F_metrics(predicted, targets, stage):
    q = predicted[:stage].astype(np.float64); t = targets.astype(np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True); t /= np.linalg.norm(t, axis=1, keepdims=True)
    scores = q @ t.T; positive = scores[np.arange(stage), np.arange(stage)]
    within = []; gaps = []
    for i in range(stage):
        indices = np.arange(i//10*10, i//10*10+10); wrong = indices[indices != i]
        within.append(indices[scores[i, indices].argmax()] == i)
        gaps.append(positive[i]-scores[i, wrong].mean())
    return dict(target_F_cosine=float(positive.mean()),
        F_gallery2AFC=float((((scores < positive[:, None]).sum(1)+.5*((scores == positive[:, None]).sum(1)-1))/399).mean()),
        F_gallery_top1=float((scores.argmax(1) == np.arange(stage)).mean()),
        F_within_concept_top1=float(np.mean(within)), F_correct_minus_other9_cosine=float(np.mean(gaps)),
        condition_mean_energy_fraction=float(np.square(predicted[:stage].mean(0)).sum()/np.square(predicted[:stage]).sum(1).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True); torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    done = U.read(args.completion); stage = int(done['stage']); assert done['status'] == f'COMPLETE_D10_STAGE{stage}'
    prepared = U.read(U.checked(done['inputs'])); lock_path = U.checked(done['condition_lock']); lock = U.read(lock_path)
    inputs = U.read(U.checked(prepared['D09_inputs'])); panel = U.read(U.checked(prepared['panel']))
    model = np.load(U.checked(lock['model_dictionary']), allow_pickle=False)
    bank = np.load(U.checked(lock['conditions']), allow_pickle=False); assert bank.shape == (400, 2, 1024)
    original_cache = torch.load(U.checked(prepared['native_EEG_cache']), map_location='cpu', weights_only=True, mmap=True)
    lookup = {int(row): i for i, row in enumerate(original_cache['array_rows'])}
    c = original_cache['condition'][[lookup[q['array_row']] for q in panel['Dcheck']]].numpy()
    choices, cosine, margin = V.nearest_support(c, model)
    assert choices.tolist() == lock['chosen_dictionary_indices']
    assert np.array_equal(cosine, np.asarray(lock['best_training_support_cosine']))
    assert np.array_equal(margin, np.asarray(lock['first_minus_second_cosine']))
    permutation = V.control_permutation(panel['Dcheck']); assert permutation.tolist() == lock['shuffled_EEG_query_permutation']
    assert np.array_equal(bank[:, 0], model[choices]) and np.array_equal(bank[:, 1], bank[permutation, 0])
    raw_F = torch.load(U.checked(prepared['raw_T_F_cache']), weights_only=True, mmap=True, map_location='cpu')
    assert np.array_equal(model, raw_F[lock['T_array_rows']].numpy())
    records = [json.loads(line) for line in U.checked(done['manifest']).read_text().splitlines()]
    baseline_rows = [json.loads(line) for line in U.checked(prepared['baseline_manifest']).read_text().splitlines()]
    assert len(records) == 2*stage and len(baseline_rows) == 1200
    paths = set(); inodes = set(); hashes = set(); pixels = set(); constant = 0
    for ordinal, row in enumerate(records):
        i, j = divmod(ordinal, 2); query = panel['Dcheck'][i]
        for key, value in query.items(): assert row[key] == value
        assert row['arm'] == V.ARMS[j] and row['query_index'] == i
        expected_i = i if j == 0 else int(permutation[i])
        assert row['selected_training_array_row'] == lock['chosen_T_array_rows'][expected_i]
        assert row['UNet_NFE'] == row['per_query_policy_NFE'] == 4
        assert row['native_IP_K_calls'] == row['native_IP_V_calls'] == 280
        assert row['scheduler_step_signature_unmodified'] and row['extra_UNet_evaluations'] == 0
        assert row['normal_inference_target_access'] is False
        for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations'):
            assert row[key] == baseline_rows[i*3][key]
        raw = np.ascontiguousarray(bank[i, j:j+1])
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['condition_sha256']
        assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest() == row['condition_FP16_sha256']
        path = Path(row['generated_path']); stat = path.stat()
        assert path.resolve() not in paths and (stat.st_dev, stat.st_ino) not in inodes
        assert lock_path.stat().st_mtime_ns < stat.st_mtime_ns
        paths.add(path.resolve()); inodes.add((stat.st_dev, stat.st_ino))
        assert U.sha(path) == row['generated_sha256']; hashes.add(row['generated_sha256'])
        with Image.open(path) as image:
            image = image.convert('RGB'); assert image.size == (512, 512)
            pixel = np.asarray(image); h = hashlib.sha256(pixel.tobytes()).hexdigest()
            assert h == row['generated_pixel_sha256']; pixels.add(h); constant += int(np.all(pixel == pixel[0, 0]))
    if stage == 400:
        previous = U.read(U.checked(done['previous']))
        prefix = [json.loads(line) for line in U.checked(previous['manifest']).read_text().splitlines()]
        assert records[:160] == prefix
        old = np.load(U.checked(previous['generated_features']), allow_pickle=False)
        assert np.array_equal(np.load(U.checked(done['generated_features']))[:80], old)
    features = np.load(U.checked(done['generated_features']), allow_pickle=False)
    targets = np.load(U.checked(prepared['target_features']), allow_pickle=False)
    baseline = np.load(U.checked(prepared['baseline_generated_features']), allow_pickle=False)
    assert V.score(features, targets, baseline, panel['Dcheck'][:stage], panel['Dcheck'], stage) == done['readout']
    # These F targets enter a post-result diagnostic only; the pre-image lock stays unchanged.
    F_targets = raw_F[[q['array_row'] for q in panel['Dcheck']]].numpy()
    F_diagnostics = {'native_parent': F_metrics(c, F_targets, stage)}
    for j, arm in enumerate(V.ARMS): F_diagnostics[arm] = F_metrics(bank[:, j], F_targets, stage)
    metadata = U.read(U.checked(inputs['target_metadata']))['Dcheck']; sheets = []
    selected = list(range(0, stage, 10))
    for page in range((len(selected)+9)//10):
        indices = selected[page*10:page*10+10]
        canvas = Image.new('RGB', (6*144, 34+164*len(indices)), 'white'); draw = ImageDraw.Draw(canvas)
        for j, label in enumerate(('target', 'parent', 'ordinary', 'risk', 'T support', 'shuffled EEG')): draw.text((j*144+3, 6), label, fill='black')
        for line, i in enumerate(indices):
            item = metadata[i]
            with zipfile.ZipFile(item['image_zip']) as archive: payload = archive.read(item['image_member'])
            assert hashlib.sha256(payload).hexdigest() == item['image_sha256'] and len(payload) == int(item['image_bytes'])
            target = Image.open(io.BytesIO(payload)).convert('RGB'); y = 34+line*164
            image_paths = [None]+[Path(row['generated_path']) for row in baseline_rows[i*3:i*3+3]]+[Path(row['generated_path']) for row in records[i*2:i*2+2]]
            for j, path in enumerate(image_paths):
                if path is None: image = target
                else:
                    with Image.open(path) as opened: image = opened.convert('RGB')
                canvas.paste(ImageOps.contain(image, (140, 140)), (j*144+2, y))
                if j == 0: draw.text((j*144+2, y+141), item['stimulus_id'][:23], fill='black')
        path = args.output/f'fixed_first_stimulus_page{page+1}.png'; canvas.save(path); sheets.append(U.ref(path))
    summary = dict(status='COMPLETE_D10_ALL_ROWS_INPUT_CONDITION_PIXEL_AND_READOUT_CPU_AUDIT', completion=U.ref(args.completion),
        source=U.ref(__file__), stage=stage, query_count=stage, generated_images=len(records), unique_paths=len(paths),
        unique_inodes=len(inodes), unique_file_hashes=len(hashes), unique_pixel_hashes=len(pixels), exact_constant_images=constant,
        all400_EEG_only_choices_replayed_exactly=True, raw_F_T_dictionary_bitwise_equal=True,
        full_stage_denominator_and_original_noise_verified=True, unchanged_reused_prefix_rows=160 if stage == 400 else 0,
        F_diagnostics=F_diagnostics, F_metrics_are_not_image_DINO=True, F_targets_only_accessed_after_frozen_image_results=True,
        no_model_or_choices_modified=True, GPU_used=False, contact_sheets=sheets,
        sheet_selection='first registered stimulus of every concept, no score sorting',
        complete_human_object_accuracy_not_measured=True, perceptual_noninferiority_established=False,
        seconds=time.monotonic()-started)
    U.atomic(args.output/'summary.json', summary); print(json.dumps(summary))


if __name__ == '__main__': main()
