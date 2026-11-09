"""Strictly inspect saved checkpoints on CPU; do not train or generate images."""
from pathlib import Path
import csv
import hashlib
import inspect
import json
import os
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
ROOT = Path(__file__).resolve().parent


def load(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def main():
    started = time.monotonic()
    output = ROOT/'audit/architecture_inventory_current_20261020_v1.csv'
    proof_path = ROOT/'audit/current_model_registry_CPU_20261020_v1.json'
    assert not output.exists() and not proof_path.exists()
    # Importing the trainer defines classes only; Trainer is never instantiated.
    from train_full_reference import ATMS, DiffusionPriorUNet, torch
    from safetensors import safe_open
    torch.set_num_threads(2)
    cfg = load(ROOT/'configs/idea_first_ATMS_training_v1.json')
    assert sha(inspect.getfile(ATMS)) == cfg['official_atms_source_sha256']
    assert sha(inspect.getfile(DiffusionPriorUNet)) == cfg['official_prior_source_sha256']
    eeg = ATMS().cpu().eval().requires_grad_(False)
    prior = DiffusionPriorUNet(cond_dim=1024, dropout=.1).cpu().eval().requires_grad_(False)
    assert sum(p.numel() for p in eeg.parameters()) == 3200413
    assert sum(p.numel() for p in prior.parameters()) == 9675648
    structures = dict(ATMS=str(eeg), DiffusionPriorUNet=str(prior),
                      ATMS_signature=str(inspect.signature(ATMS)),
                      ATMS_source=str(inspect.getfile(ATMS)),
                      ATMS_source_sha256=sha(inspect.getfile(ATMS)),
                      prior_source=str(inspect.getfile(DiffusionPriorUNet)),
                      prior_source_sha256=sha(inspect.getfile(DiffusionPriorUNet)))
    structure_path = ROOT/'audit/current_model_structures_CPU_20261020_v1.json'
    assert not structure_path.exists()
    structure_path.write_text(json.dumps(structures, indent=2)+'\n')
    legacy = list(csv.DictReader((ROOT/'audit/architecture_inventory.csv').open()))
    fields = list(legacy[0]) + ['record_scope', 'training_updates', 'training_gradient_active_parameters',
        'inference_frozen', 'evidence_path', 'evidence_sha256', 'load_verification', 'notes']
    rows = []
    for row in legacy:
        row.update(record_scope='HISTORICAL_PILOT_ORIGINAL_FORENSIC_RECORD',
                   notes='Original columns retained verbatim; trainable flag describes original training, not this CPU export. Not a current matched baseline.')
        rows.append(row)
    checked = []

    def check(model, path, expected):
        assert sha(path) == expected, path
        state = torch.load(path, weights_only=True, map_location='cpu')
        result = model.load_state_dict(state, strict=True)
        assert not result.missing_keys and not result.unexpected_keys
        assert all(torch.isfinite(x).all() for x in state.values() if torch.is_floating_point(x))
        checked.append(dict(path=str(path), sha256=expected, missing_keys=[], unexpected_keys=[], finite=True))
        del state

    def base(module, checkpoint, digest, evidence, class_name, parameters):
        return dict(module=module, source_repo='https://github.com/ncclab-sustech/EEG_Image_decode',
            source_commit=cfg['official_source_commit'], class_name=class_name, checkpoint=str(checkpoint),
            sha256=digest, total_params=parameters, trainable_params=0, trainable=False,
            inference_frozen=True, precision='float32', pretraining_data='none; trained from scratch in this campaign',
            load_missing_keys=0, load_unexpected_keys=0, evidence_path=str(evidence),
            evidence_sha256=sha(evidence), load_verification='CPU strict load and finite tensor check in this export',
            depth='see actual class structure JSON', width='see actual class structure JSON', heads='see actual class structure JSON')

    for role in ['full', 'oof0', 'oof1']:
        for subject in cfg['subjects']:
            evidence = Path(cfg['output_root'])/role/subject/'completion.json'
            receipt = load(evidence)
            assert receipt['status'] == 'COMPLETED' and receipt['training_seed'] == 1729 and receipt['prior'] is None
            c = receipt['encoder']; check(eeg, c['checkpoint'], c['checkpoint_sha256'])
            row = base('current_ATMS_patience50_'+role+'_'+subject, c['checkpoint'], c['checkpoint_sha256'], evidence, 'models.atms.ATMS()', 3200413)
            row.update(input_shape='B,63,250', output_shape='B,1024', record_scope='CURRENT_DIRECT_NATIVE_CONDITION_'+role,
                normalization='raw native CLIP-H coordinates; learned output affine LayerNorm; no inference L2 normalization',
                prediction_type='direct native visual embedding', training_updates=c['updates'],
                training_gradient_active_parameters=c['gradient_active_parameters'],
                training_data=f'{subject}; concept-disjoint T {c["unique_training_rows"]} mean4 rows; role={role}',
                notes='Exact default author class; max100/patience50; selected V loss. All parameters frozen during image generation. Not complete ATM prior/Turbo reproduction.')
            rows.append(row)

    for item in csv.DictReader((ROOT/'audit/full_reference_training_summary.csv').open()):
        evidence = ROOT/'audit/full_reference_training_summary.csv'
        check(eeg, item['encoder_checkpoint'], item['encoder_sha256'])
        row = base('older_ATMS_patience20_'+item['subject'], item['encoder_checkpoint'], item['encoder_sha256'], evidence, 'models.atms.ATMS()', 3200413)
        row.update(input_shape='B,63,250', output_shape='B,1024', record_scope='OLDER_FULL10_REFERENCE_PATIENCE20',
            training_updates=item['encoder_updates'], training_gradient_active_parameters=item['encoder_gradient_tensor_active_parameters'],
            training_data='all T rows per subject; genuine mean4; V selection',
            normalization='raw native CLIP-H coordinates; output learned LayerNorm', prediction_type='direct native visual embedding',
            notes='Preserved older encoder; not the encoder used in the new three-arm comparison.')
        rows.append(row)
        check(prior, item['prior_checkpoint'], item['prior_sha256'])
        row = base('older_epsilon_prior_'+item['subject'], item['prior_checkpoint'], item['prior_sha256'], evidence, 'DiffusionPriorUNet', 9675648)
        row.update(input_shape='noisy_visual:B,1024; condition:B,1024; t:B', output_shape='epsilon:B,1024',
            record_scope='OLDER_FULL10_EPSILON_PRIOR_READINESS_FAILURE', training_updates=item['prior_updates'],
            training_data='all T mean4 encoder conditions;400epochs;V fixed-noise selection',
            normalization='native unnormalized visual targets; DDPM clip_sample=False', prediction_type='epsilon',
            notes='Specific configuration failed corrected T image readiness; no full V image result. Width1024/512/256/128/64.')
        rows.append(row)
    for item in csv.DictReader((ROOT/'audit/prior_velocity_training_summary.csv').open()):
        evidence = ROOT/'audit/prior_velocity_training_summary.csv'
        check(prior, item['prior_checkpoint'], item['prior_sha256'])
        row = base('older_velocity_repair_'+item['subject'], item['prior_checkpoint'], item['prior_sha256'], evidence, 'DiffusionPriorUNet', 9675648)
        row.update(input_shape='noisy_visual:B,1024; condition:B,1024; t:B', output_shape='velocity:B,1024',
            record_scope='OLDER_FULL10_V_REPAIR_FULL_V_IMAGES_NOT_RUN', training_updates=item['updates'],
            training_data='all T mean4 conditions from reused patience20 encoders;400epochs;V selection',
            normalization='native unnormalized visual targets; DDPM clip_sample=False', prediction_type='v_prediction',
            notes='Same prior capacity; T operational readiness only. Not an innovation or full-V improvement.')
        rows.append(row)

    author_proof = ROOT/'audit/ATM_author_checkpoint_CPU_preflight_20261020_v1.json'
    published = load(author_proof)
    assert published['status'] == 'PASSED_ALL10_AUTHOR_PRIOR_STRICT_LOAD_AND_FEATURE_FORWARD'
    for item in published['rows']:
        row = base('published_ATM_prior_'+item['subject'], item['checkpoint'], item['checkpoint_sha256'], author_proof, 'DiffusionPriorUNet', 9675648)
        row.update(source_commit='49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d',
            input_shape='noisy_visual:B,1024; matched published EEG:B,1024; t:B', output_shape='epsilon:B,1024',
            record_scope='PUBLISHED_MATCHED_PRIOR_DECODER_NOT_RUN', pretraining_data='author pretrained; seed/update history unknown',
            training_data='author release; not trained in this campaign', training_updates='UNKNOWN_AUTHOR_PLAIN_STATE_DICT',
            normalization='author native CLIP-H; default DDPM clip_sample=True', prediction_type='epsilon',
            load_verification='Original author class strict CPU load and finite forward in referenced proof; not repeated here',
            notes='Matching released EEG features200x1024 per subject; upstream encoder checkpoint unavailable; no decoder images generated.')
        rows.append(row)

    for arm in ['RealPaired', 'Conditional']:
        evidence = DATA/'runs'/('IDEA1_interface_'+arm+'_v1/completion.json')
        item = load(evidence); assert sha(item['checkpoint']) == item['checkpoint_sha256']
        state = torch.load(item['checkpoint'], weights_only=True, map_location='cpu')['state_dict']
        assert len(state) == 4 and sum(x.numel() for x in state.values()) == 8400896
        assert all(torch.isfinite(x).all() for x in state.values())
        row = base('current_'+arm+'_native_projection', item['checkpoint'], item['checkpoint_sha256'], evidence, 'diffusers.models.embeddings.ImageProjection', 8400896)
        row.update(source_repo='https://github.com/tencent-ailab/IP-Adapter', source_commit='pretrained asset revision018e402774aeeddd60609b4ecdb7e298259dc729',
            input_shape='B,1024', output_shape='B,4,2048', depth=1, width='1024->8192;reshape4x2048;LayerNorm2048', heads='not an attention module',
            record_scope='CURRENT_MATCHED_COMPLETE_IMAGE_ADAPTATION', pretraining_data=item['initialization'],
            training_data='all132300 T subject-query rows; each mean4;one pass', training_updates=item['updates'],
            training_gradient_active_parameters=8400896, prediction_type='image tokens;epsilon denoising training objective',
            normalization='existing pretrained LayerNorm; raw native input',
            load_verification='All four checkpoint tensors counted/finite here; actual deployed BF16 tensor hashes independently checked in complete three-arm closure',
            notes='8.4M projection only;340787200 image-attention parameters and main generator frozen; not full349188096-parameter IP training.')
        rows.append(row)
    asset = Path('/dev/shm/eeg_diffusion_next_20261018_v1/adapter.safetensors')
    with safe_open(str(asset), framework='pt', device='cpu') as file:
        names = ['image_proj.proj.weight','image_proj.proj.bias','image_proj.norm.weight','image_proj.norm.bias']
        assert sum(file.get_tensor(n).numel() for n in names) == 8400896
    evidence = ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json'
    row = base('current_Frozen_native_projection', asset, 'ebf05d918348aec7abb02a5e9ecef77e0aaea6914a5c4ea13f50d45eb1681831', evidence, 'diffusers.models.embeddings.ImageProjection', 8400896)
    row.update(source_repo='https://github.com/tencent-ailab/IP-Adapter', source_commit='018e402774aeeddd60609b4ecdb7e298259dc729',
        input_shape='B,1024', output_shape='B,4,2048', depth=1, width='1024->8192;reshape4x2048;LayerNorm2048', heads='not attention',
        pretraining_data='existing public IP-Adapter weights', training_data='no current projection training', training_updates=0,
        record_scope='CURRENT_BEST_MATCHED_IMAGE_REFERENCE', normalization='pretrained LayerNorm; raw native input',
        load_verification='Selected public tensor sizes inspected here; deployed BF16 values verified by referenced closure',
        notes='Frozen applies to image projection/generator; EEG encoder was trained in this campaign.')
    rows.append(row)
    with output.open('x', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    proof = dict(status='EXPORTED_CURRENT_AND_HISTORICAL_REGISTRY_WITH_CPU_STRICT_CHECKS',
        UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), source_sha256=sha(__file__),
        rows=len(rows), CPU_strict_loaded_checkpoints=checked, strict_loaded_count=len(checked),
        existing_legacy_rows_preserved=len(legacy), public_prior_previous_load_proof_sha256=sha(author_proof),
        inventory=str(output), inventory_sha256=sha(output), structures_sha256=sha(structure_path),
        GPU_used=False, GPU_initialized=torch.cuda.is_initialized(), new_training=False, new_images=0,
        seconds=time.monotonic()-started,
        limits=['Parameter and load checks do not establish convergence, capacity cause or image success.',
                'Published prior forward proof reused transparently; no new author decoder inference.',
                'Original architecture_inventory.csv retained as a historical snapshot; use this versioned current registry.'])
    assert not proof['GPU_initialized']
    proof_path.write_text(json.dumps(proof,indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='current_model_registry_CPU_20261020_v1',GPU_used=False,
            seconds=proof['seconds'],receipt=str(proof_path),receipt_sha256=sha(proof_path)))+'\n')
    print(json.dumps({k:v for k,v in proof.items() if k!='CPU_strict_loaded_checkpoints'}))


DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')

if __name__ == '__main__':
    main()
