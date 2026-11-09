"""Read-only D09 identity, condition, noise and decoded-pixel audit.

Contact sheets use the first stimulus of every registered check concept.
This audit does not classify objects or establish perceptual noninferiority.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import brainhive_gain_calibration_20261027_v1 as U


def pixel_sha(image):
    return hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    done = U.read(args.completion); stage = int(done['stage'])
    assert done['status'] == f'COMPLETE_D09_STAGE{stage}' and stage in (80, 400)
    inputs = U.read(U.checked(done['inputs'])); panel = U.read(U.checked(inputs['panel']))
    metadata = U.read(U.checked(inputs['target_metadata']))
    lock_path = U.checked(done['policy_lock']); lock = U.read(lock_path)
    model_dir = U.checked(lock['model']).parent
    check = [json.loads(line) for line in U.checked(done['check_manifest']).read_text().splitlines()]
    fit = [json.loads(line) for line in (model_dir/'Dfit_inference_manifest.jsonl').read_text().splitlines()]
    assert len(fit) == 600 and len(check) == 3 * stage
    counters = {}; saved = {}; exact_constant = {}; reused_prefix = 0
    for partition, records, bank in [('Dfit', fit, np.load(model_dir/'conditions_Dfit.npy', allow_pickle=False)),
                                    ('Dcheck', check, np.load(U.checked(done['check_conditions']), allow_pickle=False))]:
        count = 200 if partition == 'Dfit' else stage
        assert bank.shape[1:] == (3, 1024) and bank.dtype == np.float32
        seen_paths = set(); seen_inodes = set(); hashes = set(); pixels = set(); constant = 0
        for i in range(count):
            query = panel[partition][i]; noise = None
            for arm_index, arm in enumerate(U.ARMS):
                row = records[3*i+arm_index]
                assert row['arm'] == arm and row['query_index'] == i
                for key, value in query.items(): assert row[key] == value, (key, i, arm)
                assert row['UNet_NFE'] == row['per_query_policy_NFE'] == 4
                assert row['scheduler_step_signature_unmodified'] and row['extra_UNet_evaluations'] == 0
                assert row['native_IP_K_calls'] == row['native_IP_V_calls'] == 280
                assert not row['normal_inference_target_access']
                raw = np.ascontiguousarray(bank[i, arm_index:arm_index+1])
                assert hashlib.sha256(raw.tobytes()).hexdigest() == row['condition_sha256']
                assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest() == row['condition_FP16_sha256']
                observed = {key: row[key] for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations')}
                assert len(observed['ancestral_noise_observations']) == 4
                if noise is None: noise = observed
                else: assert observed == noise
                path = Path(row['generated_path']); stat = path.stat()
                assert path.resolve() not in seen_paths and (stat.st_dev, stat.st_ino) not in seen_inodes
                seen_paths.add(path.resolve()); seen_inodes.add((stat.st_dev, stat.st_ino))
                assert U.sha(path) == row['generated_sha256']
                with Image.open(path) as opened:
                    image = opened.convert('RGB'); assert image.size == (512, 512)
                    assert pixel_sha(image) == row['generated_pixel_sha256']
                    a = np.asarray(image); constant += int(np.all(a == a[0, 0]))
                hashes.add(row['generated_sha256']); pixels.add(row['generated_pixel_sha256'])
                if partition == 'Dcheck':
                    assert lock_path.stat().st_mtime_ns <= stat.st_mtime_ns
                    saved[i, arm] = path
            item = metadata[partition][i]
            for key, value in query.items(): assert item[key] == value
            with zipfile.ZipFile(item['image_zip']) as archive: payload = archive.read(item['image_member'])
            assert hashlib.sha256(payload).hexdigest() == item['image_sha256'] and len(payload) == int(item['image_bytes'])
        counters[partition] = dict(queries=count, records=len(records), unique_paths=len(seen_paths), unique_inodes=len(seen_inodes),
            unique_file_hashes=len(hashes), unique_pixel_hashes=len(pixels), exact_constant_images=constant,
            all_identity_condition_FP16_noise_and_file_pixel_checks_passed=True)
    if stage == 400:
        previous = U.read(U.checked(done['previous']))
        prefix = [json.loads(line) for line in U.checked(previous['check_manifest']).read_text().splitlines()]
        assert check[:240] == prefix and len(prefix) == 240
        reused_prefix = 240
    sheets = []
    # The deterministic first stimulus per registered concept, without score sorting.
    selected = list(range(0, stage, 10))
    for page in range((len(selected)+9)//10):
        indices = selected[page*10:(page+1)*10]
        canvas = Image.new('RGB', (6*144, 34+len(indices)*164), 'white'); draw = ImageDraw.Draw(canvas)
        for j, title in enumerate(('target', 'parent', 'ordinary', 'risk', 'gain policy', 'shuffled policy')):
            draw.text((j*144+3, 6), title, fill='black')
        for line, i in enumerate(indices):
            query = panel['Dcheck'][i]; y = 34+line*164
            item = metadata['Dcheck'][i]
            with zipfile.ZipFile(item['image_zip']) as archive:
                import io
                target = Image.open(io.BytesIO(archive.read(item['image_member']))).convert('RGB')
            paths = [None] + [saved[i, arm] for arm in U.ARMS] + [saved[i, U.ARMS[lock['choices'][i]]], saved[i, U.ARMS[lock['shuffled_choices'][i]]]]
            for j, path in enumerate(paths):
                if path is None: image = target
                else:
                    with Image.open(path) as opened: image = opened.convert('RGB')
                canvas.paste(ImageOps.contain(image, (140, 140)), (144*j+2, y))
                label = query['stimulus_id'] if j == 0 else (U.ARMS[lock['choices'][i]] if j == 4 else U.ARMS[lock['shuffled_choices'][i]] if j == 5 else '')
                draw.text((144*j+2, y+141), label[:23], fill='black')
        path = args.output/f'fixed_first_stimulus_page{page+1}.png'; canvas.save(path); sheets.append(U.ref(path))
    result = dict(status='COMPLETE_D09_ALL_IMAGES_CPU_IDENTITY_PIXEL_NOISE_AUDIT', completion=U.ref(args.completion),
        source=U.ref(__file__), counts=counters, unchanged_reused_check_prefix_rows=reused_prefix, contact_sheets=sheets,
        deterministic_sheet_selection='first registered stimulus of each concept in locked panel order',
        contact_sheet_is_not_complete_human_quality_evaluation=True, GPU_used=False, model_or_choices_modified=False,
        perceptual_noninferiority_established=False, seconds=time.monotonic()-started)
    U.atomic(args.output/'summary.json', result); print(json.dumps(result))


if __name__ == '__main__': main()
