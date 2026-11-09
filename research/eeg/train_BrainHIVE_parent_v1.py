"""Full ten-subject, fixed-epoch native Brain-HIVE MLP alignment.

published: all 16540 official train images, R80 benchmark after training.
full/oof0/oof1: protected T and complete concept cross-fitting for our component.
No checkpoint selection on training loss or the exposed official benchmark.
"""
import argparse
import csv
import json
import math
import os
from pathlib import Path
import time

from brainhive_native_models_v1 import *


@torch.inference_mode()
def infer(model, x, batch=256):
    model.eval()
    return torch.cat([model(batch_x).float().cpu() for batch_x in x.split(batch)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--role', choices=['published', 'full', 'oof0', 'oof1'], required=True)
    args = parser.parse_args()
    assert str(ROOT / 'run_gpu_job_v2.py').encode() in Path('/proc') .joinpath(str(os.getppid()), 'cmdline').read_bytes()
    assert (WORK / 'audit/input_preparation_complete.json').exists()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    started = time.monotonic()
    role = args.role
    directory = WORK / 'runs' / role
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'completion.json').exists():
        print('ALREADY_COMPLETE_NO_RETRAIN', role, flush=True)
        return
    bindings = json.loads((WORK / 'data/actual_input_bindings.json').read_text())
    protocol = json.loads((ROOT / 'configs/split_manifest.json').read_text())
    rows_by_index = {int(x['array_row']): x for x in csv.DictReader((ROOT / 'audit/image_manifest.csv').open())
                     if x['dataset_split'] == 'train'}
    allowed = list(range(16540)) if role == 'published' else protocol['image_rows']['T']
    held = []
    if role.startswith('oof'):
        fold = int(role[-1])
        held = [i for i in allowed if protocol['oof_fold_by_concept'][rows_by_index[i]['concept_id']] == fold]
        allowed = [i for i in allowed if i not in set(held)]
        assert set(allowed).isdisjoint(held)
        assert set(allowed) | set(held) == set(protocol['image_rows']['T'])
    if role != 'published':
        assert set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V'])
    target_path = WORK / 'data/fused_train_targets.pt'
    if not target_path.exists():
        models = prior_models('cuda')
        embeddings = torch.load(WORK / 'data/train_visual_embeddings.pt', weights_only=True)
        outputs = []
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            for lo in range(0, 16540, 256):
                result = models['fusion_encoder'].vision_model(**{k: v[lo:lo+256].cuda() for k, v in embeddings.items()})
                assert torch.isfinite(result).all()
                outputs.append(result.float().cpu())
        save_tensor(target_path, torch.cat(outputs))
        write(WORK / 'audit/fused_target_contract.json', dict(
            source='Frozen H14 FusionEncoderModel.vision_model before its residual adapter',
            shape=[16540, 1024], forward_autocast='bfloat16', stored_dtype='float32',
            target_sha256=sha(target_path), additional_L2_normalization_at_inference=False))
        del models, embeddings, outputs
        torch.cuda.empty_cache()
    target = torch.load(target_path, weights_only=True).cuda()
    assert target.shape == (16540, 1024) and torch.isfinite(target).all()
    scope = classes()
    train_ids = torch.tensor(allowed, dtype=torch.long, device='cuda')
    total_updates = 25 * math.ceil(len(allowed) / 1024)
    registration = dict(role=role, training_seed=1729, subjects=list(bindings['subjects']),
        epochs=25, global_batch=1024, updates_per_subject=total_updates,
        training_array_rows=allowed, excluded_oof_rows=held,
        learning_rate=5e-4, AdamW_betas=[.9, .995], weight_decay=.05,
        exclude_bias_and_LayerNorm_from_decay=True, warmup_steps=0,
        scheduler='cosine, optimizer before scheduler', gradient_clip=1.,
        source_commit='f06db69fcc4a029439e369c5c3dba6099e57560e',
        source_class='main.models_brain.BrainMLP', source_forward_unchanged=True,
        backbone='4250->1024; one residual1024->1024->1024 GELU; LayerNorm',
        pretrained_brain_weights=False, frozen_F_A_P_IP_UNet=True,
        preprocessing='physical mean4, actual17posterior channels, no added z-score',
        source_alignment_target='F(image), raw output; symmetric cosine InfoNCE',
        raw_brain_to_generator_contract='P(A(brain)); no unit-norm replacement',
        checkpoint_rule='fixed final25epochs; no best-of-validation or benchmark selection',
        deviations=['Single train seed1729 instead of author2025 repeated seeds',
                    'One generated image/query instead of10',
                    'Direct reviewed class and native Torch loop avoid optional wrapper/dependencies',
                    'All final batches retained; seed fixed before model initialization',
                    'Warmup0 follows released YAML; paper prose states10 steps'],
        training_scope='official full train' if role == 'published' else 'protected full allowed T with concept OOF where applicable',
        benchmark_status='previously exposed; exploratory replication, never independent confirmation')
    registration_path = directory / 'registration.json'
    if registration_path.exists():
        assert json.loads(registration_path.read_text()) == registration
    else:
        write(registration_path, registration)
    all_rows = []
    for subject, item in bindings['subjects'].items():
        folder = directory / subject
        folder.mkdir(exist_ok=True)
        complete = folder / 'training_complete.json'
        if complete.exists():
            all_rows.append(json.loads(complete.read_text()))
            continue
        torch.manual_seed(1729)
        torch.cuda.manual_seed_all(1729)
        generator = torch.Generator().manual_seed(1730)
        model = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).cuda()
        logit_scale = nn.Parameter(torch.tensor(2.6592, device='cuda'))
        assert sum(p.numel() for p in model.parameters()) == 6452224
        norm_ids = {id(p) for m in model.modules() if isinstance(m, nn.LayerNorm) for p in m.parameters()}
        decay = [p for n, p in model.named_parameters() if not n.endswith('bias') and id(p) not in norm_ids] + [logit_scale]
        no_decay = [p for n, p in model.named_parameters() if n.endswith('bias') or id(p) in norm_ids]
        parameters = list(model.parameters()) + [logit_scale]
        optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05), dict(params=no_decay, weight_decay=0.)],
                                     lr=5e-4, betas=(.9, .995))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .5 * (1 + math.cos(math.pi * step / total_updates)))
        raw = torch.load(item['train_path'], weights_only=True, mmap=True)
        eeg = raw[:, bindings['selected_indices'], :].clone().cuda()
        del raw
        assert eeg.shape == (16540, 17, 250) and torch.isfinite(eeg).all()
        resume = folder / 'resume.pt'
        epoch_start, updates = 0, 0
        if resume.exists():
            state = torch.load(resume, weights_only=False, map_location='cpu')
            assert state['registration_sha256'] == sha(registration_path)
            model.load_state_dict(state['model'], strict=True)
            logit_scale.data.copy_(state['logit_scale'].to('cuda'))
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
            torch.set_rng_state(state['torch_rng'])
            torch.cuda.set_rng_state_all(state['cuda_rng'])
            generator.set_state(state['shuffle_rng'])
            epoch_start, updates = state['epoch'], state['updates']
            del state
        log = (folder / 'epochs.jsonl').open('a', buffering=1)
        for epoch in range(epoch_start, 25):
            model.train()
            sum_loss, seen = 0., 0
            order = train_ids[torch.randperm(len(train_ids), generator=generator, device='cpu').cuda()]
            for ids in order.split(1024):
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    brain = F.normalize(model(eeg[ids]), dim=-1)
                    visual = F.normalize(target[ids].to(brain.dtype), dim=-1)
                    logits = brain @ visual.T * logit_scale.exp()
                    labels = torch.arange(len(ids), device='cuda')
                    loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
                assert torch.isfinite(loss)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                optimizer.step()
                scheduler.step()
                updates += 1
                sum_loss += float(loss.detach()) * len(ids)
                seen += len(ids)
            assert seen == len(allowed)
            save_tensor(resume, dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
                logit_scale=logit_scale.detach().cpu(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), shuffle_rng=generator.get_state(),
                epoch=epoch + 1, updates=updates, registration_sha256=sha(registration_path)))
            row = dict(stage='BRAINHIVE_ALIGNMENT', role=role, subject=subject, epoch=epoch+1,
                       updates=updates, expected_updates=total_updates, full_epoch_rows=seen,
                       mean_train_InfoNCE=sum_loss/seen, grad_norm_last=float(norm),
                       elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n')
            if epoch == 0 or (epoch+1) % 5 == 0:
                print(json.dumps(row), flush=True)
        log.close()
        assert updates == total_updates
        save_tensor(folder / 'brain_final.pt', dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},
            logit_scale=logit_scale.detach().cpu(), seed=1729, epochs=25, updates=updates,
            registration_sha256=sha(registration_path)))
        report = dict(subject=subject, role=role, updates=updates, epochs=25, unique_train_stimuli=len(allowed),
                      training_exposures=25*len(allowed), model_sha256=sha(folder/'brain_final.pt'),
                      parameters=6452224, trainable_logit_parameters=1, checkpoint_selection='fixed final epoch',
                      objective='symmetric cosine InfoNCE', model_path=str(folder/'brain_final.pt'))
        write(complete, report)
        all_rows.append(report)
        # Only this run's rolling optimizer buffer is retired after its final model is verified.
        assert sha(folder/'brain_final.pt') == report['model_sha256']
        resume.unlink()
        del eeg, model, optimizer, scheduler, parameters, decay, no_decay, logit_scale
        torch.cuda.empty_cache()
    # All ten fits finish before any official test feature/EEG evaluation.
    retrieval = []
    if role == 'published':
        models = prior_models('cuda')
        embeddings = torch.load(WORK/'data/test_visual_embeddings.pt', weights_only=True)
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
            test_target = models['fusion_encoder'].vision_model(**{k:v.cuda() for k,v in embeddings.items()}).float()
        del models, embeddings
    for subject, item in bindings['subjects'].items():
        folder = directory / subject
        export = folder / 'conditions.pt'
        if export.exists():
            continue
        model = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).cuda()
        state = torch.load(folder/'brain_final.pt', weights_only=True)
        model.load_state_dict(state['model'], strict=True)
        if role == 'published':
            raw = torch.load(item['test_R80']['path'], weights_only=True, mmap=True)
            export_rows = list(range(200))
        else:
            raw = torch.load(item['train_path'], weights_only=True, mmap=True)
            export_rows = held if role.startswith('oof') else list(range(16540))
        eeg = raw[export_rows][:, bindings['selected_indices'], :].clone().cuda()
        del raw
        condition = infer(model, eeg)
        save_tensor(export, dict(condition=condition, array_rows=torch.tensor(export_rows),
            model_sha256=sha(folder/'brain_final.pt'), role=role, normalization='raw LayerNorm output'))
        if role == 'published':
            with torch.inference_mode():
                similarities = F.normalize(condition.cuda(), dim=-1) @ F.normalize(test_target, dim=-1).T
                labels = torch.arange(200, device='cuda')
                top = similarities.topk(5, dim=-1).indices
                retrieval.append(dict(subject=subject, repeat_count=80, correct_target_gallery=200,
                    stimulus_Top1=float((top[:,0] == labels).float().mean()),
                    stimulus_Top5=float((top == labels[:,None]).any(-1).float().mean()),
                    raw_condition_norm=float(condition.norm(dim=-1).mean()),
                    target_F_norm=float(test_target.norm(dim=-1).mean()),
                    correct_cosine=float(similarities.diag().mean()),
                    note='Retrieval diagnostics do not establish a correct generation interface or image quality'))
        del model, eeg, condition
        torch.cuda.empty_cache()
    write(directory/'retrieval_diagnostics.json', dict(rows=retrieval, benchmark_exposed=True))
    write(directory/'completion.json', dict(status='ALL10_FIXED_EPOCH_ALIGNMENTS_AND_CONDITION_EXPORTS_COMPLETE',
        role=role, subjects=10, models=all_rows, seconds=time.monotonic()-started, generated_images=0,
        paper_table_exact_reproduction=False, component_gain_established=False))
    print('BRAINHIVE_PARENT_STAGE_COMPLETE', role, flush=True)


if __name__ == '__main__':
    main()
