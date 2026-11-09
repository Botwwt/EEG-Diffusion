"""CPU check of a completed projector and every fixed T-readiness image.

No training, image regeneration, score computation, or checkpoint selection.
"""
from pathlib import Path
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
import sys
import time
import zipfile

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import torch
from PIL import Image, ImageDraw, ImageOps
from safetensors import safe_open

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def tensor_sha(value):
    return hashlib.sha256(value.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', required=True, choices=['RealPaired', 'Conditional', 'Moment', 'Iso', 'Clean'])
    args = parser.parse_args()
    os.nice(10)
    torch.set_num_threads(1)
    started = time.monotonic()
    arm = args.arm
    training = DATA/'runs'/f'IDEA1_interface_{arm}_v1'
    images = DATA/'images/IDEA1_interface_v1/T_readiness'/f'IDEA1_{arm}_correct_R4'
    report = load(training/'completion.json')
    complete = load(images/'completion.json')
    recipe = ROOT/'configs/idea_interface_training_recipe_v1.json'
    runtime_path = ROOT/'configs/idea_interface_runtime_v1.json'
    runtime = load(runtime_path)
    full_path = ROOT/'configs/idea_first_ATMS_training_v1.json'
    full = load(full_path)
    protocol = load(ROOT/'configs/split_manifest.json')
    assert report['status'] == 'COMPLETED_REGISTERED_MATCHED_FULL_QUERY_PASS'
    assert report['arm'] == arm and report['seed'] == 1729 and report['subjects'] == 10
    assert report['updates'] == 8269 and report['seen_training_queries'] == 132300
    assert report['each_subject_query_coverage'] == [13230]*10
    assert report['recipe_sha256'] == complete['recipe_sha256'] == sha(recipe)
    assert complete['image_interface_config_sha256'] == sha(runtime_path)
    assert sha(full_path) == complete['full_training_config_sha256'] == runtime['full_training_config_sha256']
    assert sha(training/'input_provenance.json') == report['source_input_provenance_sha256']
    assert complete['status'] == 'COMPLETED' and complete['queries'] == 40 and complete['subjects'] == 10
    assert complete['partition'] == 'T_readiness' and not complete['formal_evaluation']
    assert complete['counts'] == dict(calls=200, sample_evaluations=1600)
    assert complete['reused_query_outputs'] == 0
    checkpoint = Path(report['checkpoint'])
    assert sha(checkpoint) == report['checkpoint_sha256']
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert saved['arm'] == arm and saved['recipe_sha256'] == sha(recipe)
    assert saved['training_queries'] == 132300 and saved['updates'] == 8269
    final = saved['state_dict']
    assert len(final) == 4 and sum(x.numel() for x in final.values()) == 8400896
    assert all(x.dtype == torch.float32 and torch.isfinite(x).all() for x in final.values())
    asset = Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')
    assert sha(asset) == load(ROOT/'configs/idea_first_error_validation_plan_v1.json')['native_projector_asset_sha256']
    initial = {}
    with safe_open(str(asset), framework='pt', device='cpu') as file:
        for tail, key in [('image_embeds.weight', 'image_proj.proj.weight'),
                          ('image_embeds.bias', 'image_proj.proj.bias'),
                          ('norm.weight', 'image_proj.norm.weight'),
                          ('norm.bias', 'image_proj.norm.bias')]:
            initial['encoder_hid_proj.image_projection_layers.0.'+tail] = file.get_tensor(key).to(torch.bfloat16).float()
    assert final.keys() == initial.keys()
    init_hash = {k: tensor_sha(v) for k, v in initial.items()}
    assert init_hash == report['initialization_tensor_sha256'] == saved['initialization_tensor_sha256']
    relative = math.sqrt(sum(float((final[k]-initial[k]).double().square().sum()) for k in final)
        / sum(float(v.double().square().sum()) for v in initial.values()))
    assert abs(relative-report['relative_parameter_L2_change']) < 1e-12
    deployed = {k: tensor_sha(v.to(torch.bfloat16)) for k, v in final.items()}
    assert deployed == complete['actual_deployed_projection_tensor_sha256']
    changed = {k: int((v.to(torch.bfloat16) != initial[k].to(torch.bfloat16)).sum()) for k, v in final.items()}
    assert all(n > 0 for n in changed.values())
    assert sha(images/'manifest.jsonl') == complete['image_manifest_sha256']
    manifest = [json.loads(s) for s in (images/'manifest.jsonl').read_text().splitlines()]
    assert len(manifest) == len({x['query_id'] for x in manifest}) == 40
    assert collections.Counter(x['subject'] for x in manifest) == {s: 4 for s in protocol['subjects']}
    assert sha(ROOT/'audit/query_manifest_R4.csv') == runtime['query_manifest_sha256']
    training_queries = collections.defaultdict(list)
    with (ROOT/'audit/query_manifest_R4.csv').open() as f:
        for row in csv.DictReader(f):
            if row['role'] == 'T':
                training_queries[row['subject_id']].append(row)
    expected = {}
    for subject in protocol['subjects']:
        seen = set()
        for q in sorted(training_queries[subject], key=lambda q: int(q['array_row'])):
            if q['concept_id'] in seen:
                continue
            expected[q['query_id']] = q
            seen.add(q['concept_id'])
            if len(seen) == 4:
                break
    assert len(expected) == 40 and set(expected) == {x['query_id'] for x in manifest}
    banks = {}
    for subject in protocol['subjects']:
        proof = load(Path(full['output_root'])/'full'/subject/'completion.json')
        assert sha(proof['encoder']['checkpoint']) == proof['encoder']['checkpoint_sha256']
        assert sha(proof['condition_cache']) == proof['condition_cache_sha256']
        bank = torch.load(proof['condition_cache'], map_location='cpu', weights_only=True, mmap=True)
        assert bank['encoder_sha256'] == proof['encoder']['checkpoint_sha256']
        assert torch.equal(bank['array_rows'], torch.arange(16540))
        banks[subject] = (bank['condition'], proof['encoder']['checkpoint_sha256'])
    canvas = Image.new('RGB', (1280, 2045), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 5), f'{arm}: all40 fixed T-readiness queries | target / generated | unscored; not formal V', fill='black')
    targets = {}
    records = []
    for i, row in enumerate(manifest):
        q = expected[row['query_id']]
        assert row['subject'] == q['subject_id'] and row['stimulus_id'] == q['stimulus_id']
        assert row['concept_id'] == q['concept_id'] and row['array_row'] == int(q['array_row'])
        assert row['EEG_source_stimulus_id'] == row['stimulus_id'] and row['intervention'] == 'correct'
        assert row['projection_checkpoint_sha256'] == report['checkpoint_sha256']
        assert row['actual_deployed_projection_tensor_sha256'] == deployed
        assert row['image_interface_config_sha256'] == sha(runtime_path)
        assert not row['target_image_or_visual_feature_read_at_inference']
        assert row['steps'] == 20 and row['CFG'] == 5. and row['IP_scale'] == 1. and row['UNet_NFE'] == 40
        assert row['image_seed'] == int(q['generation_seed'])
        condition, encoder_sha = banks[row['subject']]
        assert row['encoder_sha256'] == encoder_sha
        assert tensor_sha(condition[row['array_row']]) == row['EEG_native_condition_sha256'] == row['deployed_native_condition_sha256']
        for image_field, source_field in [('target_image_zip', 'image_zip'), ('target_image_member', 'image_member'), ('target_image_sha256', 'image_sha256')]:
            assert row[image_field] == q[source_field]
        key = row['target_image_zip'], row['target_image_member']
        if key not in targets:
            with zipfile.ZipFile(key[0]) as archive:
                value = archive.read(key[1])
            assert hashlib.sha256(value).hexdigest() == row['target_image_sha256']
            with Image.open(io.BytesIO(value)) as im:
                targets[key] = im.convert('RGB').copy()
        assert sha(row['generated_path']) == row['generated_sha256']
        with Image.open(row['generated_path']) as im:
            im.load()
            assert im.mode == 'RGB' and im.size == (512, 512)
            generated = im.copy()
        x = (i % 4)*320
        y = 40+(i//4)*200
        draw.text((x+3, y), row['subject']+' '+row['stimulus_id'], fill='black')
        for column, image in enumerate([targets[key], generated]):
            thumb = ImageOps.contain(image, (156, 170), Image.Resampling.LANCZOS)
            canvas.paste(thumb, (x+column*160+(160-thumb.width)//2, y+18+(170-thumb.height)//2))
        records.append({k: row[k] for k in ['query_id', 'subject', 'stimulus_id', 'array_row',
            'target_image_zip', 'target_image_member', 'target_image_sha256', 'generated_path', 'generated_sha256', 'image_seed']})
    output = DATA/'diagnostics'/f'IDEA1_{arm}_T40_QA_v1'
    assert not output.exists(), 'Preserve completed audits and earlier failed attempts'
    output.mkdir(parents=True)
    canvas.save(output/'all40_target_generated.jpg', quality=90, subsampling=0)
    with (output/'pairing.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    assert not torch.cuda.is_initialized()
    result = dict(status='CHECKPOINT_AND_ALL40_FIXED_T_IMAGES_INDEPENDENTLY_VERIFIED', arm=arm,
        training_queries=132300, updates=8269, each_subject_training_coverage=[13230]*10,
        checkpoint_sha256=sha(checkpoint), trainable_parameters=8400896,
        relative_parameter_L2_change=relative, changed_BF16_values=changed,
        actual_deployed_projection_tensor_sha256=deployed, verified_images=40, subjects=10,
        manifest_sha256=sha(images/'manifest.jsonl'), pairing_sha256=sha(output/'pairing.csv'),
        contact_sheet=str(output/'all40_target_generated.jpg'), contact_sheet_sha256=sha(output/'all40_target_generated.jpg'),
        human_ratings_performed=False, visual_inspection_performed=False, GPU_used=False,
        training_or_generation_source_modified=False, formal_V_image_gain_established=False,
        seconds=time.monotonic()-started, source_sha256=sha(__file__),
        limits=['All40 fixed T cases, not full V performance or independent confirmation.',
            'Parameter change is not output correction strength or image quality.',
            'Image display uses resized thumbnails; source JPEG and target bytes were independently hashed.',
            'Release metadata and condition-cache ownership checked, not independent acquisition-event replay.'])
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
