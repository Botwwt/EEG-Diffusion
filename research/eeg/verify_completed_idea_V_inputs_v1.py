"""Compare complete Conditional/RealPaired inference inputs, without scoring.

This CPU check does not read feature scores or choose examples. Per-arm image
identity checks must already have passed. Different JPEG hashes are not evidence
of different objects or an image-quality gain.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
FIELDS = [
    'query_id', 'subject', 'stimulus_id', 'concept_id', 'array_row',
    'repeat_protocol', 'partition', 'intervention', 'EEG_source_stimulus_id',
    'missing_input_definition', 'target_image_zip', 'target_image_member',
    'target_image_sha256', 'image_seed', 'prior_seed',
    'initial_image_noise_sha256', 'EEG_native_condition_sha256',
    'deployed_native_condition_sha256', 'native_norm', 'encoder_checkpoint',
    'encoder_sha256', 'prior_checkpoint', 'prior_sha256', 'prior_prediction_type',
    'generator_revision', 'steps', 'CFG', 'IP_scale', 'UNet_NFE',
    'image_interface_version', 'image_interface_config_sha256',
    'projected_native_image_tokens_per_query', 'full_training_config_sha256',
    'recipe_sha256', 'target_image_or_visual_feature_read_at_inference',
    'formal_evaluation', 'compression',
]


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    os.nice(10)
    started = time.monotonic()
    output = ROOT/'audit/idea_interface_RealPaired_Conditional_V_matched_inputs_v1.json'
    assert not output.exists(), 'Preserve the previous audit attempt'
    sources = {}
    rows = {}
    for arm in ['Conditional', 'RealPaired']:
        image = DATA/'images/IDEA1_interface_v1/V'/f'IDEA1_{arm}_correct_R4'
        manifest = image/'manifest.jsonl'
        completion = load(image/'completion.json')
        identity = DATA/'evaluation/V'/f'IDEA1_{arm}_correct_R4'/'identity_before/summary.json'
        proof = load(identity)
        job = ROOT/'audit'/f'IDEA1_{arm}_generate_job.json'
        receipt = load(job)
        training_path = DATA/'runs'/f'IDEA1_interface_{arm}_v1/completion.json'
        training = load(training_path)
        assert completion['status'] == 'COMPLETED' and completion['queries'] == 16600
        assert completion['subjects'] == 10 and completion['reused_query_outputs'] == 0
        assert completion['image_manifest_sha256'] == sha(manifest)
        assert completion['counts'] == dict(calls=83000, sample_evaluations=664000)
        assert receipt['status'] == 'COMPLETED' and receipt['exit_code'] == 0
        assert proof['verified_queries'] == proof['native_EEG_cache_hash_verified_queries'] == 16600
        assert proof['failed_queries'] == 0 and proof['manifest_sha256'] == sha(manifest)
        assert sha(training['checkpoint']) == training['checkpoint_sha256']
        with manifest.open() as file:
            records = [json.loads(line) for line in file]
        by_id = {row['query_id']: row for row in records}
        assert len(records) == len(by_id) == 16600
        counts = Counter(row['subject'] for row in records)
        assert len(counts) == 10 and set(counts.values()) == {1660}
        for row in records:
            assert row['method'] == 'IDEA1_'+arm and row['intervention'] == 'correct'
            assert row['projection_checkpoint_sha256'] == training['checkpoint_sha256']
            assert row['actual_deployed_projection_tensor_sha256'] == completion['actual_deployed_projection_tensor_sha256']
            assert row['EEG_source_stimulus_id'] == row['stimulus_id']
            assert row['UNet_NFE'] == 40 and not row['target_image_or_visual_feature_read_at_inference']
        rows[arm] = by_id
        sources[arm] = dict(manifest=str(manifest), manifest_sha256=sha(manifest),
            completion_sha256=sha(image/'completion.json'), identity_before=str(identity),
            identity_before_sha256=sha(identity), generate_job_sha256=sha(job),
            training_completion_sha256=sha(training_path),
            projection_checkpoint_sha256=training['checkpoint_sha256'],
            per_subject_counts=dict(sorted(counts.items())))
    a, b = rows['Conditional'], rows['RealPaired']
    assert a.keys() == b.keys()
    for key in a:
        for field in FIELDS:
            assert a[key][field] == b[key][field], (key, field)
    left = {row['generated_path'] for row in a.values()}
    right = {row['generated_path'] for row in b.values()}
    assert len(left) == len(right) == 16600 and not left & right
    identical = sum(a[key]['generated_sha256'] == b[key]['generated_sha256'] for key in a)
    report = dict(status='PASSED_COMPLETE16600_MATCHED_INFERENCE_INPUTS_NO_SCORES_READ',
        queries=16600, subjects=10, fields_equal_for_every_query=FIELDS, sources=sources,
        cross_method_output_path_overlap=0, identical_saved_JPEG_hash_queries=identical,
        source_sha256=sha(__file__), seconds=time.monotonic()-started, GPU_used=False,
        image_gain_established=False, scientific_goal_complete=False,
        limits=['Only inference inputs and completed per-arm prerequisites are compared here.',
            'JPEG identity does not measure object fidelity or perceptual quality.',
            'Full metric and actual score-target pairing checks remain required.'],
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
    output.write_text(json.dumps(report, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(stage='complete_Conditional_RealPaired_V_input_match',
            seconds=report['seconds'], GPU_used=False, actual_status=report['status'],
            receipt=str(output), source_sha256=report['source_sha256']))+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
