"""Fixed1729 whole-geometry control at the existing .1 F-angle objective.

Engineering and full fit have separate outputs. The full fit starts fresh and
requires its actual engineering receipt. No withdrawn multi-seed CLI is used.
"""
import argparse, datetime, gc, hashlib, json, os, shutil, signal, subprocess, sys, time, traceback
from pathlib import Path


def read(p): return json.loads(Path(p).read_bytes())
def ref(p):
    p = Path(p)
    with p.open('rb') as stream: digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(p), bytes=p.stat().st_size, sha256=digest)
def checked(item): assert ref(item['path']) == item; return Path(item['path'])
def atomic(p, value):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True); pending = p.with_name(p.name + '.pending')
    with pending.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(pending, p)
def memory(): return int(next(x.split()[1] for x in Path('/proc/self/status').read_text().splitlines() if x.startswith('VmHWM:'))) * 1024


def run(path, mode):
    s = read(path)
    assert s['status'] == 'REGISTERED_SINGLE1729_EUCLIDEAN_F_GEOMETRY_FULL_LIFECYCLE'
    assert s['host'] == 'old' and s['training_seed'] == 1729 and not s['multiple_training_seeds_allowed']
    assert mode in ('profile', 'train') and s['R80_exposure_acknowledged']
    for item in s['source_pins'] + s['input_pins']: checked(item)
    assert 'run_gpu_job_v2.py' in (Path('/proc') / str(os.getppid()) / 'cmdline').read_text()
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True).strip() == s['GPU_UUID']
    assert not subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'], text=True).strip()
    root = Path(s['output_root']); out = root / ('fit_profile' if mode == 'profile' else 'fit')
    assert not out.exists(); out.mkdir(parents=True)
    started = time.monotonic(); base = memory(); phase = 'RUNTIME'; stopped = [False]
    def request_stop(*unused): stopped[0] = True
    signal.signal(signal.SIGTERM, request_stop); signal.signal(signal.SIGINT, request_stop)
    sys.path.insert(0, s['repository_root'])
    import brainhive_paired_study_contract_20261025_v1 as C
    C.setup_runtime(dict(runtime_manifest=s['runtime_manifest']))
    import torch
    import torch.nn.functional as F
    import brainhive_native_models_v1 as N
    from brainhive_ordinary_geometry_control_20261030_v1 import EuclideanConditionControl
    from brainhive_component_data_v1 import load_training
    from train_BrainHIVE_bounded_component_v1 import seed, development
    from brainhive_functional_training_contract_v2 import resource_headroom
    torch.set_num_threads(2); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(min(1., float(20 << 30) / torch.cuda.get_device_properties(0).total_memory))
    assert resource_headroom()['effective_headroom_bytes'] >= 10 << 30
    def budget():
        if mode == 'profile' and stopped[0]: raise InterruptedError('Own engineering profile stop; temporary weights discarded')
        assert time.monotonic() - started < 830 and memory() - base < 10 << 30
        assert torch.cuda.max_memory_allocated() < 20 << 30 and shutil.disk_usage(root).free > 2 << 30
        assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file()) < 64 << 20
    def digest(m):
        h = hashlib.sha256()
        for k, v in m.state_dict().items(): h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()
    def save(p, value):
        pending = p.with_name(p.name + '.pending'); torch.save(value, pending); os.replace(pending, p)
    model = optimizer = coverage = None; updates = seen = 0; total = 0.; tick = None
    try:
        phase = 'ALL10_TRUE_OOF_AND_FROZEN_AP'
        inputs = load_training(); assert inputs['input_contract_sha256'] == s['input_contract']['sha256']
        assert inputs['condition'].shape == (132300, 1024) and inputs['weight'].shape == (20, 1024)
        prior = N.prior_models('cuda'); adapter = prior['fusion_encoder'].adapter.half().eval().requires_grad_(False)
        projector = prior['proj'].half().eval().requires_grad_(False); del prior; gc.collect(); torch.cuda.empty_cache()
        frozen = dict(A=digest(adapter), P=digest(projector))
        orders = [torch.randperm(132300, generator=torch.Generator().manual_seed(seed('pass' + str(i)))) for i in range(3)]
        order_SHA = [hashlib.sha256(x.numpy().tobytes()).hexdigest() for x in orders]
        original = read(checked(s['original_ordinary_provenance'])); assert order_SHA == original['orders_sha256']
        bounded = read(checked(s['bounded_F_training_provenance'])); assert order_SHA == bounded['orders_sha256']
        def fresh():
            torch.manual_seed(1729); m = EuclideanConditionControl().cuda()
            assert sum(p.numel() for p in m.parameters()) == 525568
            assert digest(m) == bounded['initial_model_tensor_sha256']
            return m, torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=.01)
        def step(m, opt, indices, update=True):
            idx = indices.cuda(); aid = inputs['affine_ids'][idx]
            c = inputs['condition'][idx]; w = inputs['weight'][aid]; b = inputs['bias'][aid]
            opt.zero_grad(set_to_none=True); changed, info = m(c, w, b)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = .1 * (1 - F.cosine_similarity(changed.float(), inputs['target'][(indices % 13230).cuda()].float(), dim=-1)).mean()
            assert torch.isfinite(loss); loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(m.parameters(), 1., error_if_nonfinite=True)
            assert all(p.grad is None for p in list(adapter.parameters()) + list(projector.parameters()))
            result = dict(loss=float(loss), gradient_norm=float(norm), direction_gradient=float(m.direction.weight.grad.norm()),
                trunk_gradient=float(m.trunk[0].weight.grad.norm()), correction_RMS=float(info['correction'].square().mean().sqrt()))
            if update: opt.step()
            return result
        model, optimizer = fresh(); initial = digest(model); budget()
        selected = torch.stack([torch.nonzero(inputs['affine_ids'] == i)[0, 0] for i in range(20)])
        with torch.no_grad():
            assert torch.equal(model(inputs['condition'][selected], inputs['weight'], inputs['bias'])[0], inputs['condition'][selected])
            assert torch.equal(model(inputs['condition'][selected], inputs['weight'], inputs['bias'], enabled=False)[0], inputs['condition'][selected])
        if mode == 'profile':
            phase = 'REAL_F_OBJECTIVE_ENGINEERING_TEMPORARY_UPDATES_DISCARDED'
            temporary = [step(model, optimizer, orders[0][i*16:(i+1)*16]) for i in range(4)]
            assert temporary[0]['direction_gradient'] > 0 and temporary[0]['trunk_gradient'] == 0
            assert temporary[-1]['trunk_gradient'] > 0
            groups = []
            for i in range(20):
                idx = torch.nonzero(inputs['affine_ids'] == i).flatten()[:16].cpu()
                x = step(model, optimizer, idx, update=False); assert x['direction_gradient'] > 0
                groups.append(dict(affine_id=i, queries=len(idx), **x)); budget()
            assert frozen == dict(A=digest(adapter), P=digest(projector))
            for item in (s['immutable_ordinary'], s['bounded_F_component']): checked(item)
            atomic(out / 'completion.json', dict(status='PASS_REAL_F_OBJECTIVE_EUCLIDEAN_COMPONENT_ENGINEERING_ONLY',
                UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), protocol=ref(path), parameters=525568,
                initial_head_sha256=initial, orders_sha256=order_SHA, zero_head_all20_affines_exact=True,
                real_F_gradient_affine_groups=groups, temporary_updates=4, temporary_update_records=temporary,
                temporary_weights_not_saved=True, scientific_updates=0, UNet_training_calls=0, research_images=0,
                native_generator_pixel_profile_verified=False, formal_fit_admitted=False,
                frozen_AP_unchanged=True, seconds=time.monotonic()-started,
                peak_GPU_allocated=torch.cuda.max_memory_allocated(), extra_CPU_peak=memory()-base)); return
        engineering = read(root / 'fit_profile/completion.json')
        assert engineering['status'] == 'PASS_REAL_F_OBJECTIVE_EUCLIDEAN_COMPONENT_ENGINEERING_ONLY'
        assert engineering['protocol'] == ref(path) and engineering['initial_head_sha256'] == initial
        phase = 'FULL132300_THREE_PASSES_FIXED_FINAL_F_OBJECTIVE'
        coverage = torch.zeros(3, 132300, dtype=torch.bool); tick = time.monotonic(); last_save = tick
        def checkpoint():
            save(out / 'resume.pt', dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                optimizer=optimizer.state_dict(), updates=updates, seen=seen, coverage=coverage,
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(), orders_sha256=order_SHA,
                protocol_sha256=ref(path)['sha256'], seed=1729, total_loss=total, training_seconds=time.monotonic()-tick))
        with (out / 'training.jsonl').open('x', buffering=1) as log:
            for epoch in range(3):
                for start in range(0, 132300, 16):
                    if stopped[0]: checkpoint(); raise InterruptedError('Own user stop at an exact completed-update boundary')
                    budget()
                    indices = orders[epoch][start:start+16]; x = step(model, optimizer, indices)
                    assert not coverage[epoch, indices].any(); coverage[epoch, indices] = True
                    updates += 1; seen += len(indices); total += x['loss'] * len(indices)
                    if updates % 64 == 0 or start + len(indices) == 132300:
                        event = dict(updates=updates, seen_queries=seen, pass_number=epoch+1,
                            cumulative_mean_loss=total/seen, training_seconds=time.monotonic()-tick, **x)
                        log.write(json.dumps(event)+'\n'); atomic(out / 'progress.json', event)
                    if time.monotonic()-last_save > 120: checkpoint(); last_save=time.monotonic()
        torch.cuda.synchronize(); training_seconds=time.monotonic()-tick
        assert coverage.all() and seen == 396900 and updates == 24807; checkpoint()
        state = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False)
        assert int(state['coverage'].sum()) == 396900 and state['updates'] == 24807
        assert {int(v['step']) for v in state['optimizer']['state'].values()} == {24807}
        assert all(torch.equal(v.cpu(), state['model'][k]) for k,v in model.state_dict().items()); del state
        phase = 'DFIT_ONLY_FIXED_FIVE_ALPHA_CALIBRATION'
        calibration = development(model.eval(), adapter, projector, 'ordinary'); atomic(out / 'calibration.json', calibration)
        assert calibration['selected_strength'] in [0., .25, .5, .75, 1.]
        final = out / 'component_final.pt'; save(final, dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
            arm='Euclidean_F', geometry='UNBOUNDED_AFFINE_RESIDUAL', seed=1729, updates=24807, seen_queries=396900,
            passes=3, calibration=calibration, protocol_sha256=ref(path)['sha256']))
        assert frozen == dict(A=digest(adapter), P=digest(projector)); budget()
        for item in (s['immutable_ordinary'], s['bounded_F_component']): checked(item)
        atomic(out / 'completion.json', dict(status='COMPLETE_ALL10_TRUE_OOF_THREE_PASSES_EUCLIDEAN_F_FIXED_FINAL',
            UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(), protocol=ref(path), component=ref(final),
            calibration=ref(out/'calibration.json'), checkpoint=ref(out/'resume.pt'), parameters=525568,
            seed=1729, updates=24807, query_exposures=396900, orders_sha256=order_SHA, initial_head_sha256=initial,
            frozen_AP_unchanged=True, original_baselines_unchanged=True, UNet_training_calls=0,
            training_seconds=training_seconds, seconds=time.monotonic()-started,
            peak_GPU_allocated=torch.cuda.max_memory_allocated(), extra_CPU_peak=memory()-base,
            research_images=0, fixed_final_not_selected_by_test=True))
    except BaseException as error:
        atomic(out / 'failure.json', dict(phase=phase, error=repr(error), traceback=traceback.format_exc(),
            updates=updates, seen=seen, seconds=time.monotonic()-started, automatic_science_retry=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--protocol', required=True); p.add_argument('--mode', choices=['profile','train'], required=True)
    args = p.parse_args(); run(args.protocol, args.mode)
