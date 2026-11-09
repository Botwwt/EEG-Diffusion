"""Independent D11 source/condition/noise/pixel/readout audit; diagnostic only."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
from contextlib import ExitStack
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
import brainhive_teacher_reference_20261027_v1 as V


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); start = time.monotonic(); torch.set_num_threads(2)
    assert not args.output.exists(); args.output.mkdir(parents=True)
    assert not torch.cuda.is_initialized()
    done = U.read(args.completion); assert done['status'] == 'COMPLETE_D11_TEACHER_DIAGNOSTIC80'
    assert done['diagnostic_only'] is True and done['normal_inference_supported'] is False
    assert done['uses_Dcheck_visual_supervision'] is True and done['EEG_efficacy_or_method_gain'] is False
    prepared = U.read(U.checked(done['inputs'])); panel = U.read(U.checked(prepared['panel']))
    inputs = U.read(U.checked(prepared['D09_inputs']))
    lock_path = U.checked(done['condition_lock']); lock = U.read(lock_path)
    mapping = V.source_rows(panel['Dcheck'])
    assert mapping.tolist() == lock['source_query_indices']
    rows = np.asarray([q['array_row'] for q in panel['Dcheck']])[mapping]
    assert rows.tolist() == lock['source_array_rows']
    bank = np.load(U.checked(lock['conditions']), allow_pickle=False)
    raw_F = torch.load(U.checked(lock['source_F_cache']), weights_only=True, mmap=True, map_location='cpu')
    assert bank.shape == (400, 3, 1024) and bank.dtype == np.float32
    assert np.array_equal(bank, raw_F[rows.reshape(-1)].numpy().reshape(400, 3, 1024))
    assert lock['normal_inference_supported'] is False and lock['oracle_uses_query_visual_supervision'] is True
    source = U.read(U.checked(prepared['F_source_replay']))
    assert source['status'] == 'ALL400_CPU_F_REPLAY_NUMERICALLY_MATCHES_SAVED_GPU_BF16'
    assert source['all16540_ID_rows_match_manifest'] is True
    assert source['raw_JPEG_feature_encoder_independently_replayed'] is False
    records = [json.loads(x) for x in U.checked(done['manifest']).read_text().splitlines()]
    baseline = [json.loads(x) for x in U.checked(prepared['baseline_manifest']).read_text().splitlines()]
    d10 = U.read(U.checked(prepared['D10_completion']))
    support = [json.loads(x) for x in U.checked(d10['manifest']).read_text().splitlines()]
    assert len(records) == 240 and len(baseline) == 1200 and len(support) == 160
    paths = set(); inodes = set(); file_hashes = set(); pixel_hashes = set(); constant = 0
    def check_image(row):
        path = Path(row['generated_path']); st = path.stat()
        assert path.resolve() not in paths and (st.st_dev, st.st_ino) not in inodes
        paths.add(path.resolve()); inodes.add((st.st_dev, st.st_ino))
        assert U.sha(path) == row['generated_sha256']; file_hashes.add(row['generated_sha256'])
        with Image.open(path) as image:
            assert image.mode == 'RGB' and image.size == (512, 512)
            pixels = np.asarray(image); h = hashlib.sha256(pixels.tobytes()).hexdigest()
            assert h == row['generated_pixel_sha256']; pixel_hashes.add(h)
            return int(np.all(pixels == pixels[0, 0]))
    for ordinal, row in enumerate(records):
        i, j = divmod(ordinal, 3); query = panel['Dcheck'][i]
        for key, value in query.items(): assert row[key] == value
        assert row['arm'] == V.ARMS[j] and row['query_index'] == i
        assert row['source_F_array_row'] == int(rows[i, j])
        assert row['source_F_stimulus_id'] == panel['Dcheck'][mapping[i, j]]['stimulus_id']
        assert row['uses_Dcheck_visual_supervision'] is True and row['normal_inference_supported'] is False
        assert row['diagnostic_only'] is True and row['per_diagnostic_image_NFE'] == row['UNet_NFE'] == 4
        assert row['native_IP_K_calls'] == row['native_IP_V_calls'] == 280
        assert row['scheduler_step_signature_unmodified'] is True and row['extra_UNet_evaluations'] == 0
        for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256', 'ancestral_noise_observations'):
            assert row[key] == baseline[i*3][key]
        raw = np.ascontiguousarray(bank[i, j:j+1])
        assert hashlib.sha256(raw.tobytes()).hexdigest() == row['condition_sha256']
        assert hashlib.sha256(raw.astype(np.float16).tobytes()).hexdigest() == row['condition_FP16_sha256']
        assert lock_path.stat().st_mtime_ns < Path(row['generated_path']).stat().st_mtime_ns
        constant += check_image(row)
    parity = U.read(U.checked(done['pixel_parity']))
    assert parity['status'] == 'BOTH_T_NATIVE_AND_FULL_DC_PARENT_EXACT_BYTES_PIXELS_PASSED'
    expected = [inputs['parity_reference'], baseline[0]]
    for row, reference in zip(parity['rows'], expected):
        assert row['image']['generated_sha256'] == reference['generated_sha256']
        assert row['image']['generated_pixel_sha256'] == reference['generated_pixel_sha256']
        for key in ('condition_sha256', 'condition_FP16_sha256', 'tokens_sha256'):
            assert row['proof'][key] == reference[key]
        for key in ('initial_noise_latents_sha256', 'initial_UNet_input_sha256'):
            assert row['observations'][key] == reference[key]
        if 'ancestral_noise_observations' in reference:
            assert row['observations']['ancestral_noise_observations'] == reference['ancestral_noise_observations']
        else:
            assert row['name'] == 'T_native'  # The older T receipt never recorded these arrays.
        check_image(row['image'])
    features = np.load(U.checked(done['generated_features']), allow_pickle=False)
    targets = np.load(U.checked(prepared['target_features']), allow_pickle=False)
    base_features = np.load(U.checked(prepared['baseline_generated_features']), allow_pickle=False)
    d10_features = np.load(U.checked(prepared['d10_generated_features']), allow_pickle=False)
    assert V.score(features, targets, base_features, panel['Dcheck'][:80], panel['Dcheck'], 80, d10_features) == done['readout']
    metadata = U.read(U.checked(inputs['target_metadata']))['Dcheck']
    canvas = Image.new('RGB', (8*144, 34+8*164), 'white'); draw = ImageDraw.Draw(canvas)
    labels = ('target', 'parent', 'ordinary', 'risk', 'T support', 'correct F', 'same wrong F', 'other F')
    for j,label in enumerate(labels): draw.text((j*144+3, 6), label, fill='black')
    with ExitStack() as stack:
        archives = {}
        for line, i in enumerate(range(0, 80, 10)):
            item = metadata[i]
            if item['image_zip'] not in archives: archives[item['image_zip']] = stack.enter_context(zipfile.ZipFile(item['image_zip']))
            payload = archives[item['image_zip']].read(item['image_member'])
            assert hashlib.sha256(payload).hexdigest() == item['image_sha256']
            with Image.open(io.BytesIO(payload)) as image: target = image.convert('RGB')
            image_rows = [None]+baseline[i*3:i*3+3]+[support[i*2]]+records[i*3:i*3+3]
            for j,row in enumerate(image_rows):
                if row is None: image = target
                else:
                    with Image.open(row['generated_path']) as opened: image = opened.convert('RGB')
                y = 34+line*164; canvas.paste(ImageOps.contain(image, (140, 140)), (j*144+2, y))
                if j == 0: draw.text((2, y+141), item['stimulus_id'][:23], fill='black')
    sheet = args.output/'fixed_first_stimulus_all8concepts.png'; canvas.save(sheet)
    summary = dict(status='COMPLETE_D11_ALL240_SOURCE_CONDITION_NOISE_PIXEL_AND_READOUT_CPU_AUDIT',
        source=U.ref(__file__), completion=U.ref(args.completion), query_count=80,
        diagnostic_research_images=240, parity_images=2, unique_paths=len(paths), unique_inodes=len(inodes),
        unique_file_hashes=len(file_hashes), unique_pixel_hashes=len(pixel_hashes), exact_constant_research_images=constant,
        all400_teacher_rows_and_raw_F_replayed_bitwise=True, all240_original_noises_and_query_IDs_verified=True,
        both_original_native_parities_exact=True, Dcheck_parity_ancestral_noises_exact=True,
        legacy_T_reference_has_ancestral_noise_hashes=False,
        readout_recomputed_exactly=True, diagnostic_only=True,
        normal_inference_supported=False, scientific_EEG_method_gain=False, new_fit=0, original_D_resumed=False,
        CUDA_initialized=torch.cuda.is_initialized(), source_replay=prepared['F_source_replay'],
        fresh_JPEG_feature_encoder_or_public_parquet_rejoin=False,
        contact_sheet=U.ref(sheet), sheet_selection='first registered stimulus of every concept; no score sorting',
        complete_human_object_accuracy_measured=False, perceptual_noninferiority_established=False,
        seconds=time.monotonic()-start)
    U.atomic(args.output/'summary.json', summary); print(json.dumps(summary))


if __name__ == '__main__': main()
