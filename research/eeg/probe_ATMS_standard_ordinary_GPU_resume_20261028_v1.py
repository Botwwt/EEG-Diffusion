"""Finite, separate-output real-GPU recovery probe; never a scientific fit.

An unchanged run_gpu_job_v2 parent owns the GPU lock and one900s deadline.
Three fresh worker processes run continuous2 versus split1 + restored1, for
four discarded engineering updates. Only real concept-OOF T rows are selected.
The running X01 controller, six locked sources and formal outputs are untouched.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import signal
import struct
import subprocess
import sys
import time
import traceback
START = time.monotonic()
sys.path.insert(0, '/root/mced_representation_runtime_v1/lib/python3.12/site-packages')
import numpy as np
import train_ATMS_standard_SDXL_ordinary_20261028_v1 as T

ROOT = Path(__file__).resolve().parent
SCHEMA = 'ATMS_standard_SDXL_real_GPU_recovery_probe_20261028_v1'
TRAINER_SHA = '01df320704f0a4d2f3f8ac56f58868717639ec11abaee582c331c2b190cc6f00'
ROLES = ('continuous2', 'split_first1', 'split_restored1')
MAX_UPDATES = 4
MAX_OUTPUT = 128 << 20
torch = None
sha, ref, read, checked, atomic = T.sha, T.reference, T.read, T.checked, T.atomic_json


def tensor_bits(value):
    value = value.detach().cpu().contiguous()
    return hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def equal_tree(left, right, path='root', counts=None):
    """Compare serialized state bits, including dtype and signed-zero bits."""
    counts = dict(tensors=0, arrays=0, scalars=0) if counts is None else counts
    if torch is not None and torch.is_tensor(left):
        assert torch.is_tensor(right), path
        assert left.dtype == right.dtype and left.shape == right.shape and left.layout == right.layout, path
        assert tensor_bits(left) == tensor_bits(right), 'Tensor bits changed: ' + path
        counts['tensors'] += 1
    elif isinstance(left, np.ndarray):
        assert isinstance(right, np.ndarray) and left.dtype == right.dtype and left.shape == right.shape, path
        assert left.tobytes() == right.tobytes(), 'Array bits changed: ' + path
        counts['arrays'] += 1
    elif isinstance(left, dict):
        assert isinstance(right, dict) and list(left) == list(right), path
        assert [type(k) for k in left] == [type(k) for k in right], path
        for key in left:
            equal_tree(left[key], right[key], path + '/' + str(key), counts)
    elif isinstance(left, (tuple, list)):
        assert type(left) is type(right) and len(left) == len(right), path
        for i, (a, b) in enumerate(zip(left, right)):
            equal_tree(a, b, path + '/' + str(i), counts)
    elif isinstance(left, float):
        assert type(right) is float and struct.pack('>d', left) == struct.pack('>d', right), path
        counts['scalars'] += 1
    else:
        assert type(left) is type(right) and left == right, path
        counts['scalars'] += 1
    return counts


def verify_spec(path):
    """Verify consumed T-only dependencies; never open the D-condition payload."""
    spec = read(path)
    assert spec['status'] == 'ATMS_STANDARD_SDXL_EQUIVALENT_ORDINARY_LOCKED_BEFORE_EFFECTS'
    assert spec['seed'] == 1729 and spec['passes'] == 3 and spec['batch_size'] == 16
    assert spec['maximum_GPU_bytes'] == 20 << 30 and spec['maximum_extra_CPU_bytes'] == 10 << 30
    assert sha(T.__file__) == TRAINER_SHA, 'Use the actual immutable X01 trainer'
    pins = {str(Path(item['path']).resolve()): item for item in spec['source_pins']}
    consumed = []
    for name in (Path(T.__file__), ROOT / 'profile_standard_generator.py', ROOT / 'brainhive_bounded_component_v1.py'):
        binding = pins[str(name.resolve())]; checked(binding); consumed.append(binding)
    for key in ('split', 'image_manifest', 'ATMS_export_completion', 'ATMS_input_completion',
                'CLIP_H_targets', 'standard_latents_completion'):
        binding = spec[key]
        assert pins[str(Path(binding['path']).resolve())]['sha256'] == binding['sha256']
        checked(binding); consumed.append(binding)
    assert spec['CLIP_H_targets']['sha256'] == '9fad7215e04dad0e03d5d95304109974d00b8ed43599a9bbd4eac5f52e0e15eb'
    export = read(spec['ATMS_export_completion']['path'])
    for key in ('T_ATMS_OOF', 'affines'):
        checked(export[key]); consumed.append(export[key])
    assert export['T_each_query_from_model_excluding_its_concept'] and export['T_queries'] == 132300
    assert export['inputs']['sha256'] == spec['ATMS_input_completion']['sha256']
    # export['D_ATMS_full'] is metadata only and is deliberately NOT checked or loaded.
    return spec, consumed


def T_metadata(spec):
    split = read(checked(spec['split']))
    assert split['subjects'] == [f'sub-{i:02d}' for i in range(1, 11)]
    assert set(split['train_concepts']) == {'T', 'D', 'V'}
    rows = split['image_rows']['T']; assert len(rows) == len(set(rows)) == 13230
    with checked(spec['image_manifest']).open(encoding='utf-8') as stream:
        metadata = {int(row['array_row']): row for row in csv.DictReader(stream) if row['dataset_split'] == 'train'}
    assert len(metadata) == 16540
    assert {metadata[row]['concept_id'] for row in rows} == set(split['train_concepts']['T'])
    assert not set(split['train_concepts']['T']).intersection(split['d_fit_concepts'] + split['d_check_concepts'])
    upstream = read(checked(spec['ATMS_input_completion']))
    assert upstream['status'] == 'D15_ALL30_STRICT_MODELS_ORIGINAL_R4_INPUTS_AND_OOF_ROLES_CPU_VERIFIED'
    assert upstream['T_array_rows'] == rows
    folds = np.array([split['oof_fold_by_concept'][metadata[row]['concept_id']] for row in rows], np.int64)
    for subject in split['subjects']:
        for role in ('oof0', 'oof1'):
            expected = {row for row, fold in zip(rows, folds) if fold == int(role[-1])}
            assert set(upstream['subjects'][subject]['models'][role]['export_array_rows']) == expected
    return split, rows, metadata, upstream, folds


def T_data(spec):
    split, rows, metadata, upstream, folds = T_metadata(spec)
    export = read(checked(spec['ATMS_export_completion']))
    conditions = np.load(checked(export['T_ATMS_OOF']), mmap_mode='r', allow_pickle=False)
    assert conditions.shape == (132300, 1024) and conditions.dtype == np.float32
    affines = torch.load(checked(export['affines']), map_location='cpu', weights_only=True)
    weights = []; biases = []; proofs = []
    for subject in split['subjects']:
        for role in ('oof0', 'oof1'):
            binding = upstream['subjects'][subject]['models'][role]['checkpoint']
            state = torch.load(checked(binding), map_location='cpu', weights_only=True, mmap=True)
            affine = affines[subject + '/' + role]
            assert torch.equal(state['proj_eeg.2.weight'], affine['weight'])
            assert torch.equal(state['proj_eeg.2.bias'], affine['bias'])
            assert affine['weight'].shape == affine['bias'].shape == (1024,)
            assert affine['weight'].abs().min() > 1e-6 and affine['eps'] > 0.
            weights.append(affine['weight']); biases.append(affine['bias'])
            proofs.append(dict(subject=subject, role=role, checkpoint=binding)); del state
    teacher = torch.load(checked(spec['CLIP_H_targets']), map_location='cpu', weights_only=True, mmap=True)
    assert teacher.shape == (16540, 1024) and teacher.dtype == torch.float32
    return dict(split=split, T_rows=rows, metadata=metadata, conditions=conditions, folds=folds,
        weight=torch.stack(weights), bias=torch.stack(biases), target_cache=teacher, role_proofs=proofs)


def bootstrap():
    global torch
    from profile_standard_generator import torch as native_torch
    import torch.nn.functional as F
    torch = native_torch; T.torch = torch; T.F = F
    torch.set_num_threads(2); torch.cuda.set_device(0); torch.cuda.reset_peak_memory_stats()
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def check_output(output):
    assert sum(path.stat().st_size for path in Path(output).rglob('*') if path.is_file()) <= MAX_OUTPUT


def check_worker_control(path, args):
    control = read(path)
    assert control['schema'] == SCHEMA and control['status'] == 'FOUR_DISCARDED_UPDATES_CONTROLLED_BEFORE_GPU'
    assert control['source'] == ref(__file__) and control['training_spec'] == ref(args.training_spec)
    assert control['outer_pid'] == os.getppid() and os.getpgrp() == control['outer_process_group']
    assert Path(control['output']).resolve() == args.output.resolve()
    parent = Path('/proc') / str(control['outer_pid']) / 'cmdline'
    assert parent.is_file() and Path(__file__).name.encode() in parent.read_bytes()
    assert control['wrapper_pid'] == control['wrapper']['wrapper_pid']
    command = (Path('/proc') / str(control['wrapper_pid']) / 'cmdline').read_bytes()
    assert b'run_gpu_job_v2.py' in command
    assert control['maximum_seconds'] == 900 and control['maximum_engineering_updates'] == MAX_UPDATES
    return control


def checkpoint(model, optimizer, noise_rng, orders, coverage, cursor, updates, loss_sum, log, provenance):
    T.verify_cursor(cursor, updates, cursor, coverage, orders)
    torch.cuda.synchronize()
    return dict(status='COMMITTED_COMPLETE_UPDATE_BOUNDARY', model=T.cpu_tree(model.state_dict()),
        optimizer=T.cpu_tree(optimizer.state_dict()), RNG=T.rng_state(noise_rng),
        orders=[order.clone() for order in orders], orders_sha256=[T.order_sha(order) for order in orders],
        coverage=coverage.clone(), cursor=cursor, updates=updates, seen=cursor, loss_sum=loss_sum,
        log_prefix=T.prefix_ref(log), log_path=str(log.resolve()), probe_provenance=provenance)


def load_checkpoint(path, model, optimizer, noise_rng, orders, provenance):
    binding = ref(path); saved = torch.load(checked(binding), map_location='cpu', weights_only=False)
    assert saved['status'] == 'COMMITTED_COMPLETE_UPDATE_BOUNDARY' and saved['probe_provenance'] == provenance
    assert saved['orders_sha256'] == [T.order_sha(value) for value in orders]
    equal_tree(saved['orders'], orders, 'restored_order')
    T.verify_cursor(saved['cursor'], saved['updates'], saved['seen'], saved['coverage'], orders)
    tail = T.verify_prefix(saved['log_path'], saved['log_prefix'])
    model.load_state_dict(saved['model'], strict=True); optimizer.load_state_dict(saved['optimizer'])
    T.restore_rng(saved['RNG'], noise_rng)
    equal_tree(saved['model'], T.cpu_tree(model.state_dict()), 'immediate_restored_model')
    equal_tree(saved['optimizer'], T.cpu_tree(optimizer.state_dict()), 'immediate_restored_AdamW')
    equal_tree(saved['RNG'], T.rng_state(noise_rng), 'immediate_restored_all_rng')
    return saved, binding, tail


def worker(args):
    control = check_worker_control(args.control, args)
    directory = args.output / args.worker; assert not directory.exists(); directory.mkdir()
    bootstrap(); spec, selected = verify_spec(args.training_spec)
    budget = T.Resources(900, spec); budget.started = control['started_monotonic']
    def check(stage):
        budget.check(stage); assert not budget.stopping(), 'Shared finite900s recovery-probe budget'
        check_output(args.output)
    try:
        check('LOAD_REAL_T_AND_ORIGINAL_HOST')
        data = T_data(spec); host = T.setup_host(spec, budget)
        target, latents = T.setup_training_data(spec, data, host, budget)
        model = T.new_adapter(); model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=.01)
        gradients = T.configure_gradients(host)
        noise_rng = torch.Generator(device='cuda').manual_seed(T.seed('shared_training_noise'))
        orders = T.make_orders(); coverage = torch.zeros((3, 132300), dtype=torch.bool)
        cursor = updates = 0; loss_sum = 0.
        batch_records = []
        for index in orders[0][:32].tolist():
            subject_index, position = divmod(index, 13230); row = data['T_rows'][position]
            batch_records.append(dict(global_query_index=index, subject_id=data['split']['subjects'][subject_index],
                T_position=position, array_row=row, concept_id=data['metadata'][row]['concept_id'],
                stimulus_id=data['metadata'][row]['stimulus_id'], role='oof' + str(data['folds'][position])))
        batch_hashes = []
        for offset in (0, 16):
            batch = T.training_batch(data, orders[0][offset:offset + 16], target, latents)
            batch_hashes.append(dict(condition=tensor_bits(batch[0]), real_gamma=tensor_bits(batch[1]),
                real_beta=tensor_bits(batch[2]), raw_T_CLIP_H=tensor_bits(batch[3]), native_T_latent=tensor_bits(batch[4])))
        provenance = dict(source=ref(__file__), trainer=ref(T.__file__), training_spec=ref(args.training_spec),
            host_tensor_digests=host['original'], runtime=host['runtime'], actual_T_batch_records=batch_records,
            batches=batch_hashes, orders_sha256=[T.order_sha(value) for value in orders],
            selected_source_pins=selected, original_ATMS_role_proofs=data['role_proofs'])
        atomic(directory / 'input_provenance.json', provenance)
        if args.worker == 'continuous2':
            log = args.output / 'continuous_training.jsonl'
            random.seed(1729); np.random.seed(1729); torch.manual_seed(1729); torch.cuda.manual_seed_all(1729)
            initial = checkpoint(model, optimizer, noise_rng, orders, coverage, cursor, updates, loss_sum, log, provenance)
            T.atomic_torch(args.output / 'initial_state.pt', initial)
            planned = 2
        else:
            restore = args.output / ('initial_state.pt' if args.worker == 'split_first1' else 'split_after_step1.pt')
            saved = torch.load(restore, map_location='cpu', weights_only=False)
            # Provenance contains only stable dependencies and the same real batches, not worker/PID labels.
            equal_tree(provenance, saved['probe_provenance'], 'worker_input_provenance')
            saved, restored_ref, tail = load_checkpoint(restore, model, optimizer, noise_rng, orders, provenance)
            cursor, updates, loss_sum, coverage = saved['cursor'], saved['updates'], saved['loss_sum'], saved['coverage']
            log = args.output / 'split_training.jsonl'
            expected = 0 if args.worker == 'split_first1' else 1
            assert updates == expected and cursor == 16 * expected
            if args.worker == 'split_restored1':
                assert tail['bytes'] == 0, 'The actual split1 checkpoint must have no uncommitted journal tail'
            else:
                # The initial reference snapshot has an empty continuous-path
                # prefix; that other path has already advanced to step2. Its
                # preserved log tail is not part of the new split path.
                assert saved['log_prefix']['bytes'] == 0
            if args.worker == 'split_restored1':
                T.append_event(log, dict(event='EXACT_CHECKPOINT_RESUME', checkpoint=restored_ref,
                    updates=updates, cursor=cursor, committed_log_prefix=saved['log_prefix'],
                    uncommitted_tail=tail, actual_fresh_worker_pid=os.getpid()))
            planned = 1
        events = []
        for _ in range(planned):
            check('REAL_T_DISCARDED_UPDATE_' + str(updates + 1))
            ids = orders[0][cursor:cursor + 16]
            before_rng = T.rng_state(noise_rng)
            event = T.update(model, optimizer, host, T.training_batch(data, ids, target, latents),
                noise_rng, verify_epsilon_gradient=True)
            cursor += 16; updates += 1; loss_sum += event['loss'] * 16; coverage[0, ids] = True
            event = dict(event='REAL_DISCARDED_ENGINEERING_UPDATE', update=updates, cursor=cursor,
                query_indices_sha256=T.order_sha(ids),
                before_CPU_RNG_sha256=tensor_bits(before_rng['CPU']),
                before_CUDA_RNG_sha256=[tensor_bits(value) for value in before_rng['CUDA']],
                before_noise_RNG_sha256=tensor_bits(before_rng['noise']), **event)
            T.append_event(log, event); events.append(event)
            state = checkpoint(model, optimizer, noise_rng, orders, coverage, cursor, updates, loss_sum, log, provenance)
            name = ('continuous_after_step' + str(updates) if args.worker == 'continuous2'
                else ('split_after_step1' if args.worker == 'split_first1' else 'split_after_step2'))
            T.atomic_torch(args.output / (name + '.pt'), state)
            check('COMPLETE_UPDATE_CPU_SNAPSHOT_CLOSED')
            print(json.dumps(dict(worker=args.worker, completed_discarded_updates=len(events),
                on_path_updates=updates, epsilon_only_gradient=event['epsilon_only_gradient_probe_norm'])), flush=True)
        assert T.host_digests(host['pipe']) == host['original']
        assert all(p.grad is None for p in host['pipe'].unet.parameters())
        check('ALL_FROZEN_HOST_BYTES_AND_COMMITTED_STATE_CLOSED')
        completion = dict(schema=SCHEMA, status='REAL_GPU_RECOVERY_WORKER_COMPLETE_UPDATES_DISCARDED',
            worker=args.worker, pid=os.getpid(), process_group=os.getpgrp(), source=ref(__file__), trainer=ref(T.__file__),
            training_spec=ref(args.training_spec), control=ref(args.control), on_path_updates=updates,
            engineering_updates=len(events), final_state=ref(args.output / (name + '.pt')),
            input_provenance=ref(directory / 'input_provenance.json'), events=events,
            all_frozen_host_tensor_digests_unchanged=True, host_tensor_digests=host['original'], runtime=host['runtime'],
            gradient_checkpointing=gradients, actual_calls=host['counts'],
            D_ATMS_full_payload_opened=False, Dfit_Dcheck_tensor_rows_selected=0, T_only_latents=True,
            teacher_cache_checksum_covers_full_original_file=True, teacher_tensor_rows_used_only_T=True,
            scientific_updates=0, new_images=0, weights_not_eligible_for_scientific_use=True, **budget.result())
        atomic(directory / 'completion.json', completion)
        return 0
    except BaseException as exc:
        atomic(directory / 'failure.json', dict(schema=SCHEMA, status='RECOVERY_PROBE_WORKER_FAILED_NO_RECOVERY_CLAIM',
            worker=args.worker, error=repr(exc), traceback=traceback.format_exc(), no_automatic_retry=True, **budget.result()))
        raise


def compare_states(output):
    global torch
    from profile_standard_generator import torch as native_torch
    torch = native_torch; T.torch = torch
    checkpoints = {}
    for name in ('continuous_after_step1', 'continuous_after_step2', 'split_after_step1', 'split_after_step2'):
        binding = ref(output / (name + '.pt'))
        checkpoints[name] = (binding, torch.load(checked(binding), map_location='cpu', weights_only=False))
    compared = {}
    fields = ('model', 'optimizer', 'RNG', 'orders', 'orders_sha256', 'coverage', 'cursor', 'updates', 'seen', 'loss_sum', 'probe_provenance')
    for number in (1, 2):
        left = checkpoints['continuous_after_step' + str(number)][1]
        right = checkpoints['split_after_step' + str(number)][1]
        assert left['updates'] == right['updates'] == number
        assert left['cursor'] == right['cursor'] == 16 * number
        compared[str(number)] = {key: equal_tree(left[key], right[key], f'step{number}/{key}') for key in fields}
        for value in (left, right):
            T.verify_prefix(value['log_path'], value['log_prefix'])
            T.verify_cursor(value['cursor'], value['updates'], value['seen'], value['coverage'], value['orders'])
    return compared, {name: value[0] for name, value in checkpoints.items()}


def orchestrate(args):
    assert args.max_seconds == 900 and not args.output.exists(), 'Use separate, new probe output; never X01 output'
    wrapper = T.wrapper_proof(); args.output.mkdir(parents=True)
    spec, consumed = verify_spec(args.training_spec)
    T_metadata(spec)  # Real split/OOF/input identities are checked before any worker can initialize CUDA.
    control = dict(schema=SCHEMA, status='FOUR_DISCARDED_UPDATES_CONTROLLED_BEFORE_GPU',
        source=ref(__file__), trainer=ref(T.__file__), training_spec=ref(args.training_spec), output=str(args.output.resolve()),
        outer_pid=os.getpid(), outer_process_group=os.getpgrp(), wrapper_pid=os.getppid(), wrapper=wrapper,
        started_monotonic=START, deadline_monotonic=START + 900, maximum_seconds=900,
        maximum_engineering_updates=MAX_UPDATES, maximum_GPU_bytes=20 << 30,
        maximum_extra_CPU_bytes=10 << 30, maximum_output_bytes=MAX_OUTPUT,
        worker_roles=list(ROLES), selected_T_dependencies=consumed)
    control_path = args.output / 'control.json'; atomic(control_path, control)
    stopped = [False]; active = None
    signal.signal(signal.SIGTERM, lambda *_: stopped.__setitem__(0, True))
    signal.signal(signal.SIGINT, lambda *_: stopped.__setitem__(0, True))
    worker_bindings = []
    try:
        for role in ROLES:
            assert not stopped[0] and time.monotonic() < control['deadline_monotonic'] - 150
            command = [sys.executable, str(Path(__file__).resolve()), '--training-spec', str(args.training_spec.resolve()),
                '--output', str(args.output.resolve()), '--max-seconds', '900', '--worker', role,
                '--control', str(control_path.resolve())]
            with (args.output / (role + '.log')).open('xb') as stream:
                active = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
                print(json.dumps(dict(stage='RUNNING_SEPARATE_RECOVERY_PROBE_WORKER', worker=role,
                    pid=active.pid, maximum_total_seconds=900)), flush=True)
                while active.poll() is None:
                    assert not stopped[0] and time.monotonic() < control['deadline_monotonic'] - 30
                    time.sleep(.5)
                assert active.returncode == 0, 'Keep failed worker log; no automatic retry: ' + role
            binding = ref(args.output / role / 'completion.json'); item = read(checked(binding))
            assert item['status'] == 'REAL_GPU_RECOVERY_WORKER_COMPLETE_UPDATES_DISCARDED' and item['worker'] == role
            assert item['engineering_updates'] == (2 if role == 'continuous2' else 1)
            worker_bindings.append(binding); active = None; check_output(args.output)
        workers = [read(checked(binding)) for binding in worker_bindings]
        assert len({item['pid'] for item in workers}) == 3, 'Recovery must cross actual OS processes'
        assert [item['on_path_updates'] for item in workers] == [2, 1, 2]
        assert sum(item['engineering_updates'] for item in workers) == MAX_UPDATES
        assert sum(item['actual_calls']['UNet_forward_calls'] for item in workers) == MAX_UPDATES
        assert all(event['epsilon_only_gradient_probe_norm'] > 0 for item in workers for event in item['events'])
        assert workers[0]['events'][0] == workers[1]['events'][0]
        assert workers[0]['events'][1] == workers[2]['events'][0]
        compared, checkpoints = compare_states(args.output)
        for binding in (control['source'], control['trainer'], control['training_spec'], *control['selected_T_dependencies']):
            checked(binding)
        assert sha(T.__file__) == TRAINER_SHA
        assert time.monotonic() < control['deadline_monotonic']; check_output(args.output)
        done = dict(schema=SCHEMA, status='REAL_GPU_SEPARATE_PROCESS_ONE_PLUS_ONE_RECOVERY_BITWISE_PASSED',
            source=ref(__file__), trainer=ref(T.__file__), training_spec=ref(args.training_spec), control=ref(control_path),
            workers=worker_bindings, checkpoints=checkpoints, exact_compared_state_fields=compared,
            distinct_GPU_worker_processes=3, engineering_updates=MAX_UPDATES, engineering_updates_all_discarded=True,
            scientific_updates=0, new_images=0, same_real_T_batches_and_host=True,
            step1_and_step2_losses_timesteps_and_epsilon_only_gradients_exact=True,
            model_AdamW_CPU_CUDA_noise_RNG_order_coverage_all_bits_equal=True,
            Dfit_Dcheck_tensor_rows_selected=0, D_ATMS_full_payload_opened=False,
            teacher_cache_checksum_covers_full_original_file=True, teacher_tensor_rows_used_only_T=True,
            original_running_X01_controller_or_sources_changed=False, formal_X01_checkpoints_loaded=False,
            full24807_training_recovery_stability_established=False, image_gain_established=False,
            maximum_GPU_bytes=max(item['peak_GPU_bytes'] for item in workers),
            maximum_worker_extra_CPU_bytes=max(item['peak_extra_CPU_bytes'] for item in workers),
            seconds=time.monotonic() - START, no_automatic_retry=True)
        atomic(args.output / 'completion.json', done); print(json.dumps(done), flush=True)
        return 0
    except BaseException as exc:
        atomic(args.output / 'failure.json', dict(schema=SCHEMA, status='REAL_GPU_RECOVERY_PROBE_NOT_ESTABLISHED',
            source=ref(__file__), error=repr(exc), traceback=traceback.format_exc(),
            completed_workers=worker_bindings, seconds=time.monotonic() - START,
            no_automatic_retry=True, original_X01_pipeline_not_modified=True))
        raise
    finally:
        if active is not None and active.poll() is None:
            active.terminate()
            try:
                active.wait(timeout=10)
            except subprocess.TimeoutExpired:
                active.kill(); active.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=int, default=900)
    parser.add_argument('--worker', choices=ROLES)
    parser.add_argument('--control', type=Path)
    args = parser.parse_args(); assert args.max_seconds == 900 and bool(args.worker) == bool(args.control)
    return worker(args) if args.worker else orchestrate(args)


if __name__ == '__main__':
    raise SystemExit(main())
