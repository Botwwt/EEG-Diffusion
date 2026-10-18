"""SERVER ONLY: normalization statistics ablation on a frozen strong EEG encoder.

Known point-MSE prior. Only two supplied scalar statistics differ. Source-only
validation and fixed image budget; no novelty or generation claim in this job.
"""
import ast
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

assert sys.platform == 'linux'
for k in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    os.environ[k] = '2'
B = Path('/root/autodl-tmp/mced_bootstrap_20261011')
OUT = Path('/root/mced_eeg_norm_stats_point_priors_v1')
QUERY = Path('/root/mced_eeg_norm_stats_conditions_v1')
FEATURE = Path('/root/autodl-tmp/mced_research_bundle/studies/eeg_clip_visual_feature_cache_v2')
AUDIT = Path('/root/mced_eeg_atm_prior_source_audit_v1')
FDS = []
START = time.monotonic()
SEEDS = [2026111801, 2026111802]
EEG_SEEDS = [2026109501, 2026109502]
ARMS = ['no_stats', 'real_stats', 'shuffled_stats']
COUNTS = dict(train_calls=0, train_states=0, backward_calls=0, optimizer_updates=0,
              eval_calls=0, eval_states=0, EEG_calls=0, UNet_calls=0)
# Reuse only stdlib storage helpers; never execute the preparer's top-level work.
helper = B / 'prepare_eeg_crossfit_conditions_v1.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'ff9a2e7ba44e56e6af1882beccd4eb8fab724e6292145c1fd6927d20f220ae76'
import ctypes
import fcntl
names = ['identity', 'write', 'verify', 'adopt', 'export']
nodes = [n for n in ast.parse(helper.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name in names]
assert [n.name for n in nodes] == names
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(helper), 'exec'), globals())
assert not OUT.exists()
OUT.mkdir()


def run():
    import numpy as np
    import torch
    from torch import nn
    import torch.nn.functional as F
    from diffusers.models.embeddings import Timesteps, TimestepEmbedding
    from diffusers.optimization import get_cosine_schedule_with_warmup
    assert torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 5 << 30
    saved = (torch.get_rng_state(), torch.cuda.get_rng_state_all(), torch.get_num_threads(),
             torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        verify(QUERY)
        verify(AUDIT)
        verify(FEATURE, ['train_image_embeds.pt'])
        ready = json.loads((QUERY / 'results.json').read_text())
        os.kill(ready['pid'], 0)
        qr = next(r for r in ready['exports'] if r['name'] == 'private_norm_stats_conditions.pt')
        packet = torch.load(adopt(qr), weights_only=True)
        assert packet['condition_embedding_dimension'] == 1026 and packet['prior_target_dimension'] == 1024
        fit = packet['source_fit_indices']; val = packet['validation_indices']
        assert len(fit) == 15260 and len(val) == 1280
        assert not (set((fit // 10).tolist()) & set((val // 10).tolist()))
        final = torch.load(FEATURE / 'train_image_embeds.pt', weights_only=True).float()
        assert torch.equal(final[fit], packet['fit_targets'])
        target = packet['fit_targets'].float().cuda()
        path = AUDIT / 'develop/Generation/diffusion_prior.py'
        assert identity(path) == dict(bytes=9681, sha256='7bccd517b5fa5d20c7f2738efdc417e785bcaf4a586713e95ca582ad8cd9ce82')
        nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionPriorUNet']
        assert len(nodes) == 1
        scope = dict(torch=torch, nn=nn, Timesteps=Timesteps, TimestepEmbedding=TimestepEmbedding)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), scope)

        def make():
            return scope['DiffusionPriorUNet'](embed_dim=1024, cond_dim=1026, dropout=.1).cuda()

        torch.manual_seed(2026111799)
        probe = make()
        c = packet['queries'][str(EEG_SEEDS[0])]['fit']['real_stats'][:4].float().cuda().detach().clone().requires_grad_(True)
        xx = torch.zeros((4, 1024), device='cuda', requires_grad=True)
        yy = probe(xx, torch.zeros(4, device='cuda', dtype=torch.long), c)
        loss = yy.square().mean(); loss.backward()
        assert yy.shape == (4, 1024) and torch.isfinite(loss)
        assert c.grad[:, :1024].norm() > 0 and c.grad[:, 1024:].norm() > 0 and torch.isfinite(c.grad).all()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in probe.parameters())
        parameters_count = sum(p.numel() for p in probe.parameters())
        write('gradient_preflight.json', dict(complete=True, parameters=parameters_count,
            condition_gradient_norm=float(c.grad.norm()), statistic_gradient_norm=float(c.grad[:, 1024:].norm()), optimizer_updates=0,
            actual_public_class_zero_input_zero_time_backward_passed=True))
        del probe, c, xx, yy, loss
        torch.cuda.empty_cache()
        rows, exports, matched = [], [], {}
        for seed, eeg_seed in zip(SEEDS, EEG_SEEDS):
            source = packet['queries'][str(eeg_seed)]
            conditions = {arm: source['fit'][arm].float().cuda() for arm in ARMS}
            assert all(x.shape == (15260, 1026) and torch.isfinite(x).all() for x in conditions.values())
            assert all(torch.equal(conditions[arm][:, :1024], conditions['no_stats'][:, :1024]) for arm in ARMS)
            assert torch.count_nonzero(conditions['no_stats'][:, 1024:]) == 0
            for arm in ARMS:
                torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
                model = make()
                initial = hashlib.sha256()
                for name, p in model.state_dict().items():
                    initial.update(name.encode()); initial.update(p.detach().cpu().numpy().tobytes())
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=500, num_training_steps=2250)
                order = torch.Generator().manual_seed(seed + 100)
                masks = torch.Generator().manual_seed(seed + 200)
                sequence, dropout_trace, mask_trace = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
                loss_total, updates, dropped_count = 0., 0, 0
                model.train()
                for epoch in range(150):
                    for pos in torch.randperm(len(fit), generator=order).split(1024):
                        sequence.update(fit[pos].numpy().tobytes())
                        dropped = bool(torch.rand((), generator=masks) < .1)
                        dropped_count += int(dropped); mask_trace.update(bytes([int(dropped)]))
                        condition = None if dropped else conditions[arm][pos]
                        output = model(torch.zeros((len(pos), 1024), device='cuda'),
                            torch.zeros(len(pos), device='cuda', dtype=torch.long), condition)
                        dropout_trace.update(torch.cuda.get_rng_state().numpy().tobytes())
                        loss = F.mse_loss(output, target[pos]); assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True); loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step(); scheduler.step()
                        updates += 1; loss_total += float(loss.detach())
                        COUNTS['train_calls'] += 1; COUNTS['train_states'] += len(pos)
                        COUNTS['backward_calls'] += 1; COUNTS['optimizer_updates'] += 1
                    if epoch == 0 or (epoch + 1) % 25 == 0:
                        print(json.dumps(dict(stage='MATCHED_NORM_STATS_PRIOR', arm=arm, seed=seed,
                            EEG_seed=eeg_seed, epoch=epoch + 1, updates=updates,
                            mean_loss=loss_total / updates, counts=COUNTS)), flush=True)
                assert updates == 2250
                row = dict(arm=arm, seed=seed, EEG_seed=eeg_seed, updates=updates,
                    trainable_parameters=parameters_count, initial_state_sha256=initial.hexdigest(),
                    input_sequence_sha256=sequence.hexdigest(), dropout_stream_sha256=dropout_trace.hexdigest(),
                    condition_drop_mask_sha256=mask_trace.hexdigest(), condition_dropped_batches=dropped_count,
                    mean_training_loss=loss_total / updates)
                if seed in matched:
                    for k in ['updates', 'trainable_parameters', 'initial_state_sha256', 'input_sequence_sha256',
                              'dropout_stream_sha256', 'condition_drop_mask_sha256', 'condition_dropped_batches']:
                        assert row[k] == matched[seed][k], k
                else:
                    matched[seed] = row
                cp = export(f'point_{arm}_{eeg_seed}.pt', dict(state={k: p.cpu() for k, p in model.state_dict().items()},
                    arm=arm, seed=seed, EEG_seed=eeg_seed, embed_dim=1024, cond_dim=1026, updates=updates))
                exports.append(cp); row['checkpoint'] = cp; rows.append(row)
                write(f'{arm}_{eeg_seed}_ready.json', dict(complete=True, **row, validation_not_evaluated=True))
                del model, optimizer, scheduler, output, loss, condition
                torch.cuda.empty_cache()
            del conditions
        assert COUNTS['optimizer_updates'] == 13500 and COUNTS['train_states'] == 13734000
        write('all_six_priors_fixed_before_validation.json', dict(complete=True, counts=dict(COUNTS),
            matching_checks_passed=True, official_test_not_read=True))
        truth = final[val].float().cuda()
        radius = target.double().norm(dim=-1).mean().float()
        predictions, evaluations, arrays = {}, [], {}
        with torch.no_grad():
            for eeg_seed in EEG_SEEDS:
                predictions[str(eeg_seed)] = {}; arrays[eeg_seed] = {}
                for arm in ARMS:
                    query = packet['queries'][str(eeg_seed)]['validation'][arm].float().cuda()
                    assert query.shape == (1280, 1026)
                    cp = next(r['checkpoint'] for r in rows if r['EEG_seed'] == eeg_seed and r['arm'] == arm)
                    state = torch.load(cp['memfd_path'], weights_only=True)
                    model = make().eval().requires_grad_(False); model.load_state_dict(state['state'], strict=True)
                    fixed = list(model.parameters()) + list(model.buffers()); versions = [p._version for p in fixed]
                    output = model(torch.zeros((1280, 1024), device='cuda'), torch.zeros(1280, device='cuda', dtype=torch.long), query)
                    COUNTS['eval_calls'] += 1; COUNTS['eval_states'] += len(query)
                    assert torch.isfinite(output).all() and output.shape == (1280, 1024)
                    assert versions == [p._version for p in fixed] and all(p.grad is None for p in fixed)
                    predictions[str(eeg_seed)]['point_' + arm] = output.cpu()
                    predictions[str(eeg_seed)]['point_' + arm + '_fit_radius'] = (F.normalize(output, dim=-1) * radius).cpu()
                    scores = F.normalize(output.double(), dim=-1) @ F.normalize(truth.double(), dim=-1).T
                    rank = (scores >= scores.diag()[:, None]).sum(1) - 1
                    correct = (rank < 1).double().cpu().numpy(); arrays[eeg_seed][arm] = correct
                    row = dict(EEG_seed=eeg_seed, arm=arm, source1280_top1=float(correct.mean()),
                        source1280_top5=float((rank < 5).double().mean()), mean_cosine=float(scores.diag().mean()),
                        raw_MSE=float((output.double() - truth.double()).square().mean()), mean_norm=float(output.double().norm(dim=-1).mean()))
                    evaluations.append(row); print(json.dumps(row), flush=True)
                    del model, state, fixed, output
                    torch.cuda.empty_cache()
        assert COUNTS['eval_calls'] == 6 and COUNTS['eval_states'] == 7680
        concepts = np.asarray((val // 10).tolist())
        members = np.stack([np.flatnonzero(concepts == c) for c in np.unique(concepts)])
        assert members.shape == (128, 10)
        rng = np.random.default_rng(2026111803); draws = rng.integers(0, 128, (20000, 128))
        comparisons = []
        for arm, control in [('real_stats', 'no_stats'), ('real_stats', 'shuffled_stats'), ('shuffled_stats', 'no_stats')]:
            delta = np.mean([arrays[s][arm] - arrays[s][control] for s in EEG_SEEDS], axis=0)
            by_concept = delta[members].mean(1)
            comparisons.append(dict(arm=arm, control=control, mean_top1_difference=float(delta.mean()),
                paired_concept_bootstrap95=np.quantile(by_concept[draws].mean(1), [.025, .975]).tolist()))
        exports.append(export('private_norm_stats_prior_predictions.pt', dict(predictions=predictions,
            source_indices=val, planned_image_indices=packet['planned_image_indices'],
            fit_mean=target.mean(0).cpu(), fit_mean_norm=radius.cpu(),
            target_space='Original_CLIP_H_final_raw1024', official_test_read=False)))
        write('input_bindings.json', dict(query_commit=identity(QUERY / 'final_commit.json'), query_export=qr,
            source_feature_commit=identity(FEATURE / 'final_commit.json'), prior_public_source=identity(path),
            source_prior_images=15260, source_heldout_images=1280, same_first1024_validation_query_in_all_arms=True,
            only_last2_condition_columns_differ=True, statistics_are_latent_not_physical_EEG_amplitude=True,
            shuffled_training='Concept-deranged permutation of TRAIN statistics',
            shuffled_validation='Independent fixed draws from TRAIN statistics; no other validation query involved'))
        return dict(complete=True, pid=os.getpid(), utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            seconds=time.monotonic() - START, counts=COUNTS, training=rows, evaluations=evaluations,
            comparisons=comparisons, exports=exports, all_matching_checks_passed=True,
            planned_image_indices=packet['planned_image_indices'], source_only=True, official_test_read=False,
            generation_not_yet_measured=True, novelty_established=False, SOTA=False)
    finally:
        cpu, cuda, threads, tf32, cudnn = saved
        torch.set_rng_state(cpu); torch.cuda.set_rng_state_all(cuda); torch.set_num_threads(threads)
        torch.backends.cuda.matmul.allow_tf32 = tf32; torch.backends.cudnn.allow_tf32 = cudnn


try:
    write('plan.json', dict(pid=os.getpid(), script=identity(Path(__file__)),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), arms=ARMS, seeds=SEEDS, EEG_seeds=EEG_SEEDS,
        architecture='Pinned ATM DiffusionPriorUNet(embed1024,cond1026,dropout.1), zero noisy input/time: deterministic point-MSE control',
        primary='real_stats versus no_stats; negative control shuffled_stats. Frozen identical EEG query first1024, matched prior architecture/init/order/dropout/targets/compute',
        statistics='Two source-fit-standardized scalars: pre-LayerNorm latent mean and log sqrt(var+eps); finite precision reconstruction checked by condition preparer',
        negative_control='Concept-deranged TRAIN permutation; validation draws fixed independently from TRAIN statistics',
        training='150epochs,B1024,2250updates/model; Adam1e-3,wd0,500warmup/cosine2250,clip1,10percent batch condition dropout; one raw-final-CLIP MSE objective',
        expected_updates=13500, expected_train_states=13734000,
        all_six_models_fixed_before_validation_evaluation=True,
        image_plan_locked_in_query_preparer=True, primary_image_outputs_are_raw_not_radius_selected=True,
        source_only=True, normalization_statistics_not_novel_by_itself=True, SOTA=False))
    while not (QUERY / 'final_commit.json').exists():
        if (QUERY / 'failure.json').exists():
            raise RuntimeError('Condition preparation failed')
        if (QUERY / 'plan.json').exists():
            os.kill(json.loads((QUERY / 'plan.json').read_text())['pid'], 0)
        assert time.monotonic() - START < 7200
        time.sleep(10)
    result = run()
    write('results.json', result)
    write('final_commit.json', dict(complete=True,
        files={p.name: identity(p) for p in OUT.glob('*.json')}, volatile_exports=result['exports']))
    print('READY_MATCHED_NORM_STATS_POINT_PRIORS', flush=True)
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()
    until = time.monotonic() + 86400
    while time.monotonic() < until:
        write('keeper_status.json', dict(pid=os.getpid(), ready=True,
            utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            lease_seconds_remaining=until - time.monotonic()), True)
        time.sleep(30)
except BaseException:
    write('failure.json', dict(traceback=traceback.format_exc(), counts=COUNTS))
    raise
finally:
    for fd in FDS:
        os.close(fd)
