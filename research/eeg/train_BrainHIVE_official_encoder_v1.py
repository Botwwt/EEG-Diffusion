"""One complete admitted official subject/fold: 8270 images, 25 epochs, 225 updates.

There is currently no sealed official method/resource execution review. The
first operation therefore rejects before Torch/native imports, input payloads
or training-directory writes. This file does not create a review/admission.
After a future real gate passes it runs the native stochastic-teacher training
engine. Condition export and official inference/scoring are separate entries.
"""
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import sys
import time

sys.dont_write_bytecode = True
import brainhive_official_execution_contract_CPU_v1 as G
import brainhive_official_encoder_job_contract_CPU_v1 as J
import brainhive_official_encoder_training_v1 as T

ROOT = Path(__file__).resolve().parent
MIN_OUTPUT_BYTES = 5 << 29  # Retain every model+Adam+RNG epoch state, not one rolling file.
MIN_CPU_HEADROOM_BYTES = 8 << 30


def directory_allocated(root):
    """Unique real inodes; own immutable compatibility hardlinks count once."""
    seen = set(); total = 0
    if not root.exists():
        return 0
    for path in root.rglob('*'):
        T.require(not path.is_symlink(), 'No aliases in the private training-output tree')
        if path.is_file():
            value = path.stat(); identity = (value.st_dev, value.st_ino)
            if identity not in seen:
                total += value.st_blocks * 512 if hasattr(value, 'st_blocks') else value.st_size
                seen.add(identity)
    return total


def resources(context, root):
    G.require_context(context)
    limits = context['resource_limits']
    threshold = limits['minimum_effective_CPU_headroom_bytes']
    T.require(type(threshold) is int and threshold >= MIN_CPU_HEADROOM_BYTES,
        'The official CPU-memory floor may not be below8GiB')
    from brainhive_functional_training_contract_v2 import resource_headroom
    actual = resource_headroom()
    T.require(actual['effective_headroom_bytes'] >= threshold, 'Fresh reviewed CPU headroom unavailable')
    used = directory_allocated(root)
    T.require(type(context['maximum_output_bytes']) is int and context['maximum_output_bytes'] >= MIN_OUTPUT_BYTES
        and used <= context['maximum_output_bytes'], 'Whole immutable-epoch output budget unavailable/exceeded')
    return dict(memory=actual, output_allocated_bytes=used)


def immutable_json(path, value):
    raw = T.json_bytes(value)
    if path.exists():
        T.require(not path.is_symlink() and path.read_bytes() == raw, 'Existing immutable metadata differs; preserve and stop')
    else:
        T.write_once(path, raw)


def source_check(context):
    for name, expected in context['source_sha256'].items():
        T.require(G.sha(ROOT / name) == expected, 'Admitted training/native source changed: ' + name)


def run_admitted(context):
    """The only payload/compute path; a private successful gate token is required."""
    G.require_context(context)
    T.require(context['action'] == 'train', 'Only the train action can enter the official encoder training engine')
    job = context['job']; J.validate_job(context['bundle'], job)
    reg = job['registration']; root = Path(job['output_root'])
    T.require((reg['epochs'], reg['global_batch'], reg['updates_per_subject'], len(reg['training_array_rows'])) ==
        (25, 1024, 225, 8270), 'Only the full official fixed-budget native task is implemented')
    T.require(math.ceil(8270 / 1024) == 9 and 8270 % 1024 == 78, 'Keep all native final batches')
    before = resources(context, root); source_check(context)
    payload = G.verify_payload_files(context, include_visual=True)
    # No model library is imported until the actual source/method/resource/
    # wrapper gate and complete input-file hashes have passed.
    import brainhive_native_models_v1 as N
    torch = N.torch
    T.require(os.name == 'posix' and torch.cuda.is_available() and torch.cuda.is_bf16_supported(), 'Admitted existing CUDA/BF16 runtime required')
    T.require(torch.__version__ == context['payload_contract']['runtime']['torch_version'], 'Actual Torch runtime differs')
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda')
    # Seed before any native module initialization. The second identical seed
    # below preserves the original parent's seed immediately before BrainMLP.
    T.seed_everything(torch)
    scope = N.classes()
    priors = N.prior_models('cuda')
    teacher = priors['fusion_encoder'].vision_model
    T.require(dict(teacher.input_meta) == payload['proj_meta'], 'Native F projection contract differs from actual training visuals')
    T.require(sum(p.numel() for p in teacher.parameters()) == 11015168, 'Use the real H14 FusionEncoder before A')
    T.frozen_stochastic_teacher(torch, teacher)
    del priors; gc.collect(); torch.cuda.empty_cache()
    raw_eeg = torch.load(payload['EEG_path'], weights_only=True, map_location='cpu', mmap=True)
    raw_visuals = torch.load(payload['visual_path'], weights_only=True, map_location='cpu', mmap=True)
    tensor_report = G.validate_training_tensors(context, raw_eeg, raw_visuals, payload['proj_meta'])
    ids = torch.tensor(reg['training_array_rows'], dtype=torch.long)
    channels = torch.tensor(reg['EEG_input']['selected_indices'], dtype=torch.long)
    eeg = raw_eeg[ids[:, None], channels[None, :], :].contiguous().to(device)
    visuals = {name: value.index_select(0, ids).to(device) for name, value in raw_visuals.items()}
    del raw_eeg, raw_visuals, ids, channels; gc.collect()
    T.require(tuple(eeg.shape) == (8270, 17, 250) and all(value.shape[0] == 8270 for value in visuals.values()), 'No train row truncation or duplicate filling')
    T.seed_everything(torch)
    model = scope['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).to(device)
    T.require(sum(p.numel() for p in model.parameters()) == 6452224, 'Native full BrainMLP required')
    logit_scale = torch.nn.Parameter(torch.tensor(2.6592, device=device))
    optimizer, scheduler = T.make_optimizer(torch, model, logit_scale, 225)
    shuffle_rng = torch.Generator(device='cpu').manual_seed(1730)
    resources(context, root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    T.require(root.resolve(strict=True) == root and not root.is_symlink(), 'Private official task output required')
    immutable_json(root / 'registration.json', reg)
    registration_sha = T.sha(root / 'registration.json')
    producer_sha = context['source_sha256'][Path(__file__).name]
    rows, pending = T.scan_committed_epochs(root, registration_sha, job['registration_core_sha256'],
        producer_source_sha256=producer_sha, subject=job['subject'], role=job['role'], updates_per_epoch=9, total_epochs=25, full_epoch_rows=8270)
    epoch, updates = 0, 0
    if context['resume']:
        T.require(rows and rows[-1]['commit_ref'] == context['resume_commit_ref'], 'Resume only the latest complete source-reviewed epoch transaction')
        state = torch.load(Path(rows[-1]['directory']) / 'resume.pt', weights_only=True, map_location='cpu')
        epoch, updates = T.restore(torch, state, model, logit_scale, optimizer, scheduler, shuffle_rng,
            registration_sha, job['registration_core_sha256'], producer_sha, device, 9, 225)
        del state
    else:
        T.require(not rows and not pending and not any((root / name).exists() for name in
            ('brain_final.pt', 'resume.pt', 'epochs.jsonl', 'training_complete.json')), 'Prior training evidence requires a separate exact source-bound recovery review')
    # Every operation after RNG restoration and before the next update is
    # deterministic metadata/resource work; no random diagnostic forward.
    input_proof = dict(schema='BrainHIVE_official_encoder_input_provenance_v1',
        status='ACTUAL_SOURCE_BOUND_TRAIN_INPUT_PAYLOADS_VERIFIED', task_id=job['task_id'],
        subject=job['subject'], role=job['role'], registration_sha256=registration_sha,
        registration_core_sha256=job['registration_core_sha256'],
        payload_contract_sha256=context['payload_contract_sha256'], source_sha256=context['source_sha256'],
        verified_files=sorted(payload['files_verified'], key=lambda item: item['path']), EEG_path=payload['EEG_path'], EEG_sha256=payload['EEG_sha256'],
        visual_path=payload['visual_path'], visual_sha256=payload['visual_sha256'], proj_meta=payload['proj_meta'],
        tensor_report=tensor_report, producer_source_sha256=producer_sha,
        original_teacher='fusion_encoder.vision_model, frozen weights, train mode, every dropout .1',
        normalization='native raw BrainMLP output; InfoNCE uses temporary F.normalize only',
        actual_GPU=True, official_test_or_target_payloads_loaded=False, V_or_official_scores_read=False)
    input_path = root / 'input_provenance.json'
    immutable_json(input_path, input_proof)
    immutable_json(root / ('input_attempt_' + str(os.getpid()) + '.json'), dict(
        execution_review=context['execution_review_ref'], method_lock=context['method_lock_ref'],
        admission=context['admission_ref'], wrapper_job_snapshot=context['wrapper_job_ref'],
        initial_resources=before, resume_commit_ref=context['resume_commit_ref'],
        restored_epoch=epoch, restored_updates=updates, retained_pending_epoch_directories=pending))
    deadline = context['deadline_monotonic']; started = time.monotonic()
    for next_epoch in range(epoch + 1, 26):
        row, updates = T.run_epoch(torch, model, teacher, logit_scale, optimizer, scheduler, shuffle_rng,
            eeg, visuals, epoch=next_epoch, updates=updates, total_updates=225, batch_size=1024,
            subject=job['subject'], role=job['role'], device=device, deadline=deadline,
            autocast_bfloat16=True, observer=lambda _stage: resources(context, root))
        row['this_attempt_elapsed_seconds'] = time.monotonic() - started
        value = T.snapshot(torch, model, logit_scale, optimizer, scheduler, shuffle_rng, next_epoch, updates,
            registration_sha, job['registration_core_sha256'], producer_sha, device)
        published = T.publish_epoch(torch, root, value, row); del value
        rows.append(dict(directory=str(Path(published['path']).parent),
            commit={key: value for key, value in published.items() if key not in ('path', 'sha256')},
            row=row, commit_ref={key: published[key] for key in ('path', 'sha256')}))
        print(json.dumps(row, allow_nan=False), flush=True)
        resources(context, root)
    # Verify the full immutable history once at closure, rather than reheating
    # every older optimizer snapshot after every epoch.
    rows, pending = T.scan_committed_epochs(root, registration_sha, job['registration_core_sha256'],
        producer_source_sha256=producer_sha, subject=job['subject'], role=job['role'], updates_per_epoch=9, total_epochs=25, full_epoch_rows=8270)
    T.require(updates == 225 and len(rows) == 25, 'Incomplete fixed official encoder fit')
    source_check(context); final = T.publish_final(torch, root, model, logit_scale,
        registration_sha256=registration_sha, registration_core_sha256=job['registration_core_sha256'],
        producer_source_sha256=producer_sha, committed=rows, total_epochs=25, total_updates=225,
        subject=job['subject'], role=job['role'])
    state = torch.load(final['model_path'], weights_only=True, map_location='cpu')
    own_affine = {key: state['model'][key] for key in J.AFFINE_KEYS}
    T.require(all(tuple(value.shape) == (1024,) and torch.isfinite(value).all().item() for value in own_affine.values()), 'Actual final encoder own-affine shape/finite check failed')
    affine = dict(checkpoint_sha256=final['model_sha256'], keys=J.AFFINE_KEYS, shape=[1024],
        tensor_sha256={key: hashlib.sha256(value.detach().contiguous().numpy().tobytes()).hexdigest() for key, value in own_affine.items()})
    del state, own_affine
    T.require(G.verify_payload_files(context, include_visual=True) == payload, 'Training source payloads changed')
    final.update(status='OFFICIAL_SINGLE_ENCODER_ALL8270_FIXED25_EPOCHS_COMPLETE', training_seed=1729,
        unique_train_stimuli=8270, training_exposures=206750, parameters=6452224, trainable_logit_parameters=1,
        checkpoint_selection='fixed final epoch', own_affine=affine, actual_GPU=True,
        condition_export_complete=False, generated_images=0, official_test_inputs_loaded=False,
        official_retrieval_computed=False, official_scores_read=False, input_provenance=G.ref(input_path),
        payload_contract_sha256=context['payload_contract_sha256'],
        actual_wrapper_job_name=context['wrapper_evidence']['job_name'], actual_training_pid=os.getpid(),
        actual_wrapper_pid=os.getppid(), execution_review=context['execution_review_ref'],
        runtime=dict(torch=torch.__version__, CUDA=torch.version.cuda), sources=context['source_sha256'],
        method_lock=context['method_lock_ref'], admission=context['admission_ref'],
        resource_after=resources(context, root), retained_pending_epoch_directories=pending)
    J.validate_fit_metadata(job, final, [record['row'] for record in rows])
    immutable_json(root / 'training_complete.json', final)
    return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-directory', type=Path, required=True)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--execution-review', type=Path, default=ROOT / 'audit/official_fullfit_execution_review_v1.json')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    # This must precede even importing the native helper/Torch or touching the
    # task output namespace. A caller-supplied Boolean is not an admission.
    context = G.preflight(args.plan_directory, args.task_id, args.execution_review,
        action='train', entry_name=Path(__file__).name, resume=args.resume)
    G.require_context(context)
    old_handler = signal.getsignal(signal.SIGALRM)
    def expire(_signal, _frame):
        raise TimeoutError('Finite official encoder execution boundary; preserve all committed and pending states')
    signal.signal(signal.SIGALRM, expire)
    signal.setitimer(signal.ITIMER_REAL, max(.001, context['deadline_monotonic'] - time.monotonic()))
    try:
        result = run_admitted(context)
        print(json.dumps(dict(status=result['status'], task_id=context['task_id'], model_sha256=result['model_sha256'])), flush=True)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0); signal.signal(signal.SIGALRM, old_handler)


if __name__ == '__main__':
    main()
