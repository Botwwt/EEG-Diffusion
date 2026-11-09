"""Fixed native-objective ten-subject full-T BrainKAN encoder experiment.

No protected image target, V score, test image, or best-checkpoint selection is
read here. Exported conditions use EEG only. Formal training requires a future
parent-issued complete train/generate/score storage admission. --profile is a
two-update engineering job, never a substitute for the complete experiment.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import time

from brainhive_native_models_v1 import *
from brainhive_brainkan_v1 import BrainKAN, fixed_electrode_time_groups


CANDIDATE = 'brainkan_full_v1'


def core_recipe_sha256(recipe):
    core = {k: v for k, v in recipe.items()
            if k not in {'storage_admission_receipt', 'storage_admission_sha256'}}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_recipe(recipe):
    fixed = dict(candidate=CANDIDATE, role='full', training_seed=1729, epochs=25,
                 batch_size=1024, learning_rate=.0005, AdamW_betas=[.9, .995],
                 weight_decay=.05, gradient_clip=1., subjects=10,
                 input_dim=4250, hidden_size=1024, spline_basis_count=8,
                 spline_rank=60, trainable_encoder_parameters=6456512,
                 native_visual_training_dropout=.1, encoder_dropout=.1,
                 checkpoint_selection='fixed_final_epoch', no_final_test_tuning=True)
    for key, value in fixed.items():
        assert recipe[key] == value, (key, recipe.get(key), value)
    for filename in ['brainhive_brainkan_v1.py', 'train_BrainHIVE_BrainKAN_v1.py',
                     'brainhive_native_models_v1.py', 'brainhive_two_ideas_storage_v1.py',
                     'brainhive_lifecycle_storage_v1.py']:
        assert recipe['source_sha256'][filename] == sha(ROOT / filename), filename
    required_inputs = {
        str(WORK / 'data/actual_input_bindings.json'),
        str(ROOT / 'configs/split_manifest.json'),
        str(ROOT / 'audit/image_manifest.csv'),
        str(WORK / 'data/train_visual_embeddings.pt'),
        str(WORK / 'audit/input_preparation_complete.json'),
    }
    assert set(recipe['input_sha256']) == required_inputs
    for filename in sorted(required_inputs):
        assert recipe['input_sha256'][filename] == sha(filename), filename
    assert recipe['output_directory'] == str(WORK / 'runs' / CANDIDATE)
    assert recipe['generator'] == dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1)
    return core_recipe_sha256(recipe)


def storage_admission(recipe):
    path = Path(recipe['storage_admission_receipt'])
    assert path.is_absolute() and sha(path) == recipe['storage_admission_sha256']
    receipt = json.loads(path.read_text())
    assert receipt['status'] == 'FULL_TRAIN_GENERATE_SCORE_ADMITTED'
    assert receipt['candidate'] == CANDIDATE
    assert receipt['recipe_core_sha256'] == core_recipe_sha256(recipe)
    assert set(receipt['scope']) >= {'training', 'actual_images', 'per_query_evaluation'}
    assert receipt['output_directory'] == recipe['output_directory']
    assert int(receipt['max_training_artifact_bytes']) >= 1 << 30
    roots = [ROOT.resolve(), WORK.resolve(), Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1').resolve()]
    reservations = []
    for item in receipt['reservation_paths']:
        reserve = Path(item['path'])
        assert reserve.is_absolute() and not reserve.is_symlink() and reserve.is_file()
        resolved = reserve.resolve()
        assert any(resolved.is_relative_to(root) for root in roots)
        details = reserve.stat()
        assert stat.S_IMODE(details.st_mode) == 0o600
        assert details.st_blocks * 512 >= int(item['minimum_allocated_bytes']) > 0
        reservations.append(dict(path=str(resolved), allocated=details.st_blocks * 512))
    assert reservations
    # These small, allocated files prove private writable storage ownership.
    # They are not a physical reservation of the whole future experiment. The
    # shared guard freshly accounts for every admitted arm and the main queue.
    from brainhive_two_ideas_storage_v1 import verify_candidate_remaining
    remaining = verify_candidate_remaining(recipe, CANDIDATE)
    return dict(receipt_sha256=sha(path), reservations=reservations, remaining=remaining)


def initialize_subject(item, subject, allowed, protocol, bindings):
    assert item['train_repeats'] == 4
    input_path = Path(item['train_path'])
    assert sha(input_path) == item['train_sha256']
    raw = torch.load(input_path, weights_only=True, mmap=True)
    assert raw.shape == (16540, 63, 250) and raw.dtype == torch.float32
    eeg_cpu = raw[:, bindings['selected_indices'], :].contiguous()
    del raw
    assert eeg_cpu.shape == (16540, 17, 250) and torch.isfinite(eeg_cpu).all()
    torch.manual_seed(1729)
    torch.cuda.manual_seed_all(1729)
    model = BrainKAN(dropout=.1)
    statistics = model.fit_training_statistics(eeg_cpu[allowed], dict(
        split='T', rows=len(allowed), array_rows=allowed, subject=subject,
        source_path=str(input_path), source_sha256=item['train_sha256'], physical_repeats=4,
        selected_channels=bindings['selected_channels'], selected_indices=bindings['selected_indices'],
        split_manifest_sha256=sha(ROOT / 'configs/split_manifest.json'),
        image_manifest_sha256=protocol['image_manifest_sha256'],
        forbidden_statistic_fit_rows=protocol['image_rows']['D'] + protocol['image_rows']['V']))
    assert set(statistics['array_rows']).isdisjoint(statistics['forbidden_statistic_fit_rows'])
    return model.cuda(), eeg_cpu.cuda(), statistics


def optimizer_for(model):
    logit_scale = nn.Parameter(torch.tensor(2.6592, device='cuda'))
    norm_ids = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}
    decay = [p for name, p in model.named_parameters() if not name.endswith('bias') and id(p) not in norm_ids] + [logit_scale]
    no_decay = [p for name, p in model.named_parameters() if name.endswith('bias') or id(p) in norm_ids]
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05),
                                  dict(params=no_decay, weight_decay=0.)], lr=5e-4, betas=(.9, .995))
    return optimizer, logit_scale, list(model.parameters()) + [logit_scale]


def objective(model, eeg, ids, training_F, visual_embeddings, logit_scale):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        brain = F.normalize(model(eeg[ids]), dim=-1)
        with torch.no_grad():
            target = training_F(**{key: value[ids] for key, value in visual_embeddings.items()})
        visual = F.normalize(target.to(brain.dtype), dim=-1)
        logits = brain @ visual.T * logit_scale.exp()
        labels = torch.arange(len(ids), device='cuda')
        loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
    assert torch.isfinite(loss)
    return loss


@torch.inference_mode()
def export_conditions(model, eeg, folder, model_sha256, statistics):
    model.eval()
    destination = folder / 'conditions.pt'
    conditions = torch.cat([model(part).float().cpu() for part in eeg.split(256)])
    assert conditions.shape == (16540, 1024) and torch.isfinite(conditions).all()
    save_tensor(destination, dict(condition=conditions, array_rows=torch.arange(16540), role='full',
        model_sha256=model_sha256, source_class='brainhive_brainkan_v1.BrainKAN',
        normalization='raw LayerNorm output; no extra L2 normalization',
        statistics_provenance_sha256=statistics['provenance_sha256'],
        source_input='EEG only; no protected visual targets read for export'))
    return dict(path=str(destination), sha256=sha(destination), shape=list(conditions.shape),
                mean_norm=float(conditions.norm(dim=-1).mean()),
                note='Export includes all official-training EEG rows for the fixed T/D/V split; no score computed')


def engineering_profile(recipe, bindings, protocol, allowed, models, embeddings):
    assert recipe.get('engineering_profile_max_updates') == 2
    subject = sorted(bindings['subjects'])[0]
    started = time.monotonic()
    model, eeg, statistics = initialize_subject(bindings['subjects'][subject], subject, allowed, protocol, bindings)
    model.train()
    optimizer, logit_scale, parameters = optimizer_for(model)
    ids = torch.tensor(allowed[:1024], device='cuda')
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    update_started = time.monotonic()
    losses, gradients = [], []
    for _ in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = objective(model, eeg, ids, models['fusion_encoder'].vision_model, embeddings, logit_scale)
        loss.backward()
        assert model.spline_coefficients.grad is not None and torch.isfinite(model.spline_coefficients.grad).all()
        gradients.append(float(model.spline_coefficients.grad.norm()))
        assert gradients[-1] > 0
        torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        losses.append(float(loss.detach()))
    torch.cuda.synchronize()
    update_seconds = time.monotonic() - update_started
    model.eval()
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
        condition = model(eeg[ids[:2]])
        mapped = models['fusion_encoder'].adapter(condition)
        tokens = models['proj'](mapped)
    assert condition.shape == mapped.shape == (2, 1024) and tokens.shape == (2, 4, 2048)
    assert all(torch.isfinite(x).all() for x in (condition, mapped, tokens))
    assert all(p.grad is None for module in models.values() for p in module.parameters())
    result = dict(status='ENGINEERING_ONLY_TWO_ACTUAL_INPUT_UPDATES', candidate=CANDIDATE,
        actual_subject=subject, actual_batch=1024, updates=2, losses=losses,
        spline_gradient_norms=gradients, seconds_per_update=update_seconds / 2,
        peak_allocated_GPU_bytes=torch.cuda.max_memory_allocated(), native_token_shape=list(tokens.shape),
        frozen_native_parameter_gradients=False, statistics=statistics,
        total_seconds=time.monotonic()-started, recipe_core_sha256=core_recipe_sha256(recipe),
        formal_training=False, generated_images=0, heldout_score_read=False)
    path = WORK / 'audit/brainkan_actual_input_GPU_profile_v1.json'
    assert not path.exists(), 'Preserve the previous profile; create a reviewed new version instead'
    write(path, result)
    print(json.dumps({key: value for key, value in result.items() if key != 'statistics'}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', type=Path, required=True)
    parser.add_argument('--profile', action='store_true')
    args = parser.parse_args()
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in (Path('/proc') / str(os.getppid()) / 'cmdline').read_bytes()
    recipe = json.loads(args.recipe.read_text())
    recipe_core = validate_recipe(recipe)
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    bindings = json.loads((WORK / 'data/actual_input_bindings.json').read_text())
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    assert bindings['split_manifest_sha256'] == sha(ROOT / 'configs/split_manifest.json')
    assert bindings['image_manifest_sha256'] == protocol['image_manifest_sha256'] == sha(ROOT / 'audit/image_manifest.csv')
    assert len(bindings['subjects']) == 10 and len(bindings['selected_indices']) == 17
    allowed = list(protocol['image_rows']['T'])
    assert len(allowed) == len(set(allowed)) == 13230
    assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    assert set(allowed) | set(protocol['image_rows']['D']) | set(protocol['image_rows']['V']) == set(range(16540))
    if not args.profile:
        storage_admission(recipe)
    models = prior_models('cuda')
    training_F = models['fusion_encoder'].vision_model
    for module in training_F.modules():
        if isinstance(module, nn.Dropout):
            module.p = .1
    training_F.train().requires_grad_(False)
    embeddings = {key: value.cuda() for key, value in torch.load(WORK / 'data/train_visual_embeddings.pt', weights_only=True).items()}
    assert all(not p.requires_grad for module in models.values() for p in module.parameters())
    if args.profile:
        engineering_profile(recipe, bindings, protocol, allowed, models, embeddings)
        return
    started = time.monotonic()
    directory = Path(recipe['output_directory'])
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'completion.json').exists():
        assert json.loads((directory / 'completion.json').read_text())['recipe_core_sha256'] == recipe_core
        print('ALREADY_COMPLETED_NO_RETRAIN', CANDIDATE, flush=True)
        return
    total_updates = 25 * math.ceil(len(allowed) / 1024)
    registration = dict(candidate=CANDIDATE, recipe_core_sha256=recipe_core, recipe_sha256=sha(args.recipe),
        subjects=sorted(bindings['subjects']), epochs=25, updates_per_subject=total_updates,
        training_array_rows=allowed, frozen_visual_training_dropout=.1,
        source_class='brainhive_brainkan_v1.BrainKAN', parameters=6456512,
        original_BrainMLP_parameters=6452224, capacity_increase_claimed=False,
        objective='source-matched symmetric cosine InfoNCE against actual stochastic frozen F(image)',
        inference='raw LN1024 -> native A -> native P4x2048 -> learned IP -> native SDXL-Turbo',
        shared_component_or_OOF_prediction_reuse=False, pretrained_EEG_weights=False,
        frozen_F_A_P_IP_UNet=True, checkpoint_selection='fixed final25epochs',
        protected_visual_target_or_score_read=False,
        model_limitations=['Additive input nonlinearities replace hidden-space interactions',
                           'Pre-LN named-coordinate attribution is not causal physiology or final-pixel attribution'])
    registration_path = directory / 'registration.json'
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write(registration_path, registration)
    completed = []
    for subject in sorted(bindings['subjects']):
        admission = storage_admission(recipe)
        folder = directory / subject
        folder.mkdir(exist_ok=True)
        complete_path = folder / 'training_complete.json'
        if complete_path.exists():
            record = json.loads(complete_path.read_text())
            assert record['recipe_core_sha256'] == recipe_core
            assert sha(folder / 'brain_final.pt') == record['model_sha256']
            assert sha(folder / 'conditions.pt') == record['conditions']['sha256']
            completed.append(record)
            continue
        model, eeg, statistics = initialize_subject(bindings['subjects'][subject], subject, allowed, protocol, bindings)
        provenance_path = folder / 'statistics_provenance.json'
        if provenance_path.exists():
            assert json.loads(provenance_path.read_text()) == statistics
        else:
            write(provenance_path, statistics)
        optimizer, logit_scale, parameters = optimizer_for(model)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 * (1 + math.cos(math.pi * step / total_updates)))
        shuffle = torch.Generator().manual_seed(1730)
        epoch_start, updates = 0, 0
        resume = folder / 'resume.pt'
        if resume.exists():
            state = torch.load(resume, weights_only=False, map_location='cpu')
            assert state['registration_sha256'] == sha(registration_path)
            model.load_state_dict(state['model'], strict=True)
            assert state['statistics_provenance_sha256'] == statistics['provenance_sha256']
            logit_scale.data.copy_(state['logit_scale'].to('cuda'))
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            shuffle.set_state(state['shuffle_rng'])
            epoch_start, updates = state['epoch'], state['updates']
            del state
        train_ids = torch.tensor(allowed, dtype=torch.long, device='cuda')
        with (folder / 'epochs.jsonl').open('a', buffering=1) as log:
            for epoch in range(epoch_start, 25):
                storage_admission(recipe)
                model.train()
                order = train_ids[torch.randperm(len(train_ids), generator=shuffle).cuda()]
                seen, sum_loss = 0, 0.
                for ids in order.split(1024):
                    optimizer.zero_grad(set_to_none=True)
                    loss = objective(model, eeg, ids, training_F, embeddings, logit_scale)
                    loss.backward()
                    grad = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                    optimizer.step()
                    scheduler.step()
                    updates += 1
                    seen += len(ids)
                    sum_loss += float(loss.detach()) * len(ids)
                assert seen == len(allowed)
                assert all(p.grad is None for module in models.values() for p in module.parameters())
                save_tensor(resume, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                    logit_scale=logit_scale.detach().cpu(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), shuffle_rng=shuffle.get_state(),
                    epoch=epoch + 1, updates=updates, registration_sha256=sha(registration_path),
                    statistics_provenance_sha256=statistics['provenance_sha256']))
                row = dict(stage='BRAINHIVE_BRAINKAN_ALIGNMENT', subject=subject, epoch=epoch + 1,
                    updates=updates, expected_updates=total_updates, mean_train_InfoNCE=sum_loss / seen,
                    seen_training_rows=seen, grad_norm_last=float(grad), seconds=time.monotonic() - started)
                log.write(json.dumps(row) + '\n')
                write(folder / 'progress.json', row)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print(json.dumps(row), flush=True)
        assert updates == total_updates
        final = folder / 'brain_final.pt'
        if not final.exists():
            save_tensor(final, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                logit_scale=logit_scale.detach().cpu(), epochs=25, updates=updates, seed=1729,
                registration_sha256=sha(registration_path), recipe_core_sha256=recipe_core,
                statistics=statistics, class_name='brainhive_brainkan_v1.BrainKAN'))
        else:
            state = torch.load(final, weights_only=True, map_location='cpu')
            assert state['recipe_core_sha256'] == recipe_core and state['updates'] == total_updates
            model.load_state_dict(state['model'], strict=True)
        conditions = export_conditions(model, eeg, folder, sha(final), statistics)
        record = dict(subject=subject, candidate=CANDIDATE, updates=updates, epochs=25,
            unique_training_images=len(allowed), training_exposures=25 * len(allowed),
            parameters=6456512, trainable_logit_parameters=1, model_path=str(final), model_sha256=sha(final),
            conditions=conditions, statistics_provenance_sha256=statistics['provenance_sha256'],
            recipe_core_sha256=recipe_core, checkpoint_selection='fixed final25epochs',
            additional_generated_images=0, score_or_protected_visual_targets_read=False)
        write(complete_path, record)
        completed.append(record)
        assert sha(final) == record['model_sha256']
        resume.unlink()
        del model, eeg, optimizer, scheduler, logit_scale, parameters
        torch.cuda.empty_cache()
    write(directory / 'completion.json', dict(status='ALL10_FIXED_EPOCH_BRAINKAN_FULL_T_ALIGNMENTS_AND_CONDITION_EXPORTS_COMPLETE',
        recipe_core_sha256=recipe_core, recipe_sha256=sha(args.recipe), models=completed,
        subjects=10, seconds=time.monotonic() - started, generated_images=0,
        component_gain_established=False, physiological_explanation_established=False,
        next_required='Actual matched native image generation and full per-query evaluation'))
    print('BRAINKAN_FULL_T_ENCODERS_COMPLETE_NO_IMAGE_GAIN_CLAIM', flush=True)


if __name__ == '__main__':
    main()
