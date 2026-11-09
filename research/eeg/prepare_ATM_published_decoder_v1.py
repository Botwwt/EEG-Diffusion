"""Register one complete, fixed public-feature decoder replication on exposed data.

No final-test unlock is created. The user's master prompt section 4.3 permits
complete exposed-benchmark reporting when an independent H is unavailable.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import time
import zipfile

ROOT = Path(__file__).resolve().parent
OUT = Path('/root/atm_author_decoder_20261020_v1')
ASSETS = Path('/root/atm_author_pretrained_20261020_v2')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write(path, value):
    assert not path.exists(), path
    path.write_text(json.dumps(value, indent=2)+'\n')


def main():
    registration = ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'
    assert not registration.exists() and not OUT.exists()
    assert shutil.disk_usage('/root').free > 2000*1024**2+300*1024**2+1024**3
    cp_proof = json.loads((ROOT/'audit/ATM_author_checkpoint_CPU_preflight_20261020_v1.json').read_text())
    assert cp_proof['status'] == 'PASSED_ALL10_AUTHOR_PRIOR_STRICT_LOAD_AND_FEATURE_FORWARD'
    query_file = ROOT/'audit/query_manifest_R4.csv'
    image_file = ROOT/'audit/image_manifest.csv'
    queries = [q for q in csv.DictReader(query_file.open()) if q['dataset_split'] == 'test']
    originals = {int(q['array_row']):q for q in csv.DictReader(image_file.open()) if q['dataset_split'] == 'test'}
    assert len(queries) == 2000 and set(originals) == set(range(200))
    subjects = sorted({q['subject_id'] for q in queries})
    assert subjects == [f'sub-{i:02d}' for i in range(1, 11)]
    required = ['generate_ATM_published_decoder_v1.py','audit_ATM_published_decoder_v1.py',
                'run_ATM_published_decoder_v1.py','extract_image_features.py',
                'score_full_manifest.py','score_low_level.py','image_evaluation.py',
                'semantic_evaluator.py','locked_metrics.py','run_gpu_job.py',
                'prepare_ATM_published_decoder_v1.py']
    hashes = {str(ROOT/p):sha(ROOT/p) for p in required}
    source = ROOT/'reproduction_sources/ATM_main_49cb3a20'
    for f in ['diffusion_prior.py','custom_pipeline.py']:
        path=source/'Generation'/f
        hashes[str(path)] = sha(path)
    old = ROOT.parent/'eeg_diffusion_next_20261018_v1/asset_restore_results.json'
    generator_assets = [x for x in json.loads(old.read_text())['assets']
                        if x['name'] in ['sd_xl_turbo_1.0_fp16.safetensors','sdxl_models/ip-adapter_sdxl_vit-h.safetensors']]
    assert len(generator_assets) == 2 and all(x['verified'] for x in generator_assets)
    model_root = Path('/dev/shm/eeg_diffusion_next_20261018_v1')
    for p in (model_root/'sdxl_config').rglob('*'):
        if p.is_file():
            hashes[str(p)] = sha(p)
    OUT.mkdir()
    (OUT/'gallery/images').mkdir(parents=True)
    targets, inference, gallery = [], [], []
    archives = {}
    try:
        for i in range(200):
            q = originals[i]
            path = q['image_zip']
            if path not in archives:
                archives[path] = zipfile.ZipFile(path)
            raw = archives[path].read(q['image_member'])
            assert hashlib.sha256(raw).hexdigest() == q['image_sha256']
            target = OUT/'gallery/images'/f'{i:08d}.jpg'
            target.write_bytes(raw)
            gallery.append(dict(stimulus_id=q['stimulus_id'],concept_id=q['concept_id'],array_row=i,
                                image_path=str(target),image_sha256=q['image_sha256'],
                                source_zip=q['image_zip'],source_member=q['image_member']))
        for subject in subjects:
            selected = sorted([q for q in queries if q['subject_id'] == subject], key=lambda q:int(q['array_row']))
            assert [int(q['array_row']) for q in selected] == list(range(200))
            for q in selected:
                row = int(q['array_row'])
                original = originals[row]
                for key in ['stimulus_id','concept_id','image_zip','image_member','image_sha256']:
                    assert q[key] == original[key], (subject,row,key)
                sid = q['stimulus_id']
                query_id = f'{subject}/{sid}/R80_author_published'
                def seed(kind):
                    return int(hashlib.sha256(f'1729|ATM-public-decoder-v1|{subject}|{sid}|{kind}'.encode()).hexdigest()[:16],16)%(2**63-1)
                item = dict(query_id=query_id,subject=subject,stimulus_id=sid,array_row=row,
                            image_seed=seed('image'),prior_seed=seed('prior'))
                inference.append(item)
                targets.append(item | dict(concept_id=q['concept_id'],local_concept_id=original['local_concept_id'],
                    target_image_zip=q['image_zip'],target_image_member=q['image_member'],target_image_sha256=q['image_sha256'],
                    score_target_path=gallery[row]['image_path'],dataset_role='benchmark_exposed',
                    EEG_feature_row_identity_basis='Author released-feature row-order contract; no independent encoder replay',
                    physical_repeat_count_independently_verified=False))
    finally:
        for archive in archives.values():
            archive.close()
    for path,rows in [(OUT/'inference_inputs.jsonl',inference),(OUT/'target_mapping.jsonl',targets),(OUT/'gallery/manifest.jsonl',gallery)]:
        path.write_text(''.join(json.dumps(x)+'\n' for x in rows))
        hashes[str(path)] = sha(path)
    caps=dict(generate=2400,gallery_features=300,features=1500,join=180,score=300,lowlevel=180,verify=180,report=120)
    native={}
    for row in cp_proof['rows']:
        native[row['subject']]=dict(features=row['feature'],features_sha256=row['feature_sha256'],
                                   prior=row['checkpoint'],prior_sha256=row['checkpoint_sha256'])
    plan=dict(scope='ALL2000_PUBLISHED_FEATURE_DECODER_REPRODUCTION_ONLY',seed=1729,query_count=2000,
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),output_root=str(OUT),subjects=native,
        author_source_root=str(source),author_source_commit='49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d',
        inference_manifest=str(OUT/'inference_inputs.jsonl'),target_mapping=str(OUT/'target_mapping.jsonl'),
        gallery_manifest=str(OUT/'gallery/manifest.jsonl'),
        generator_asset_root=str(model_root),generator_assets=[dict(path=x['restored_path'],sha256=x['sha256'],repo=x['repo'],revision=x['revision']) for x in generator_assets],
        python_paths=['/root/autodl-tmp/atm_cpu_addons_20261016_v2/packages','/root/autodl-tmp/atm_generation_cpu_addons_20261016_v1/packages','/root/mced_representation_runtime_v1/lib/python3.12/site-packages'],
        source_sha256=hashes,
        inference_source_sha256={k:v for k,v in hashes.items() if k not in [str(OUT/'target_mapping.jsonl'),str(OUT/'gallery/manifest.jsonl')]},
        phase_caps_seconds=caps,reserve_seconds=900,
        required_prior_closure=str(ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json'),
        required_prior_closure_status='PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC',
        generation=dict(prior_steps=50,prior_CFG=5,prior_clip_sample=True,Turbo_steps=4,CFG=0,IP_scale=1,resolution=512,format='PNG'),
        authority='User requested author reproduction first; master prompt section4.3 line183 permits full exposed benchmark reporting, section4.4 requires separate R80 track.',
        creates_final_test_unlock=False,unblocks_legacy_S1_prior_gate=False,
        author_EEG_encoder_weights_available=False,independent_EEG_feature_identity_replayed=False,
        independent_H=False,component_training_authorized_by_this_registration=False,
        author_pretraining_seed='UNKNOWN; no new training in this replication',
        fixed_published_method_no_checkpoint_or_seed_selection=True,
        local_large_files_allowed=False,total_GPU_hour_cap=40)
    write(registration,plan)
    print(json.dumps(dict(status='REGISTERED_COMPLETE_PUBLIC_FEATURE_DECODER_LIFECYCLE_NOT_STARTED',registration=str(registration),sha256=sha(registration),queries=len(inference),phase_caps=caps)))


if __name__ == '__main__':
    main()
