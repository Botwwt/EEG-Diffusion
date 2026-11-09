"""Future admitted official OOF export; currently rejects without real review.

No Torch/native import, EEG/checkpoint read, GPU allocation or output creation
occurs before the separate CPU preflight. Exports only each new encoder's8270
held official TRAIN rows, never the official200/test/targets/retrieval. Existing
files are adopted only after full actual identity/tensor/own-affine validation;
an incomplete/orphan cache is retained and refused, never exists->skip.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True
import brainhive_official_execution_contract_CPU_v1 as G

E, require = G.E, G.require
CACHE_KEYS = {'condition', 'array_rows', 'affine_weight', 'affine_bias', 'affine_keys',
    'subject', 'role', 'model_sha256', 'registration_sha256', 'registration_core_sha256',
    'exporter_source_sha256', 'training_producer_source_sha256', 'EEG_input_sha256',
    'normalization', 'additional_L2_normalization', 'affine_source_checkpoint_sha256'}


def tensor_sha(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def validate_checkpoint(state, job, fit):
    import torch
    expected = dict(seed=1729, epochs=25, updates=225,
        registration_sha256=fit['registration_sha256'], registration_core_sha256=job['registration_core_sha256'],
        producer_source_sha256=fit['producer_source_sha256'])
    require(set(state) == {'model', 'logit_scale'} | set(expected)
        and all(state[k] == v for k, v in expected.items()), 'Checkpoint registration/source/fixed25epoch225update chain differs')
    require(isinstance(state['model'], dict) and state['model']
        and all(isinstance(v, torch.Tensor) and bool(torch.isfinite(v).all()) for v in state['model'].values()), 'Nonfinite or invalid strict model state')
    require(isinstance(state['logit_scale'], torch.Tensor) and state['logit_scale'].numel() == 1
        and bool(torch.isfinite(state['logit_scale']).all()), 'Invalid trained logit scale')
    return expected


def infer_held_conditions(model, eeg, job, *, context=None, device='cpu', batch_size=256, _fixture=None):
    """Real tensor inference. Private tiny injection changes dimensions only.

    Production has no subset/count switch: actual shape is16540x63x250,
    source rows8270, channels46..62, output8270x1024, unchanged raw finalLN.
    """
    if _fixture is None:
        G.require_context(context)
        require(job == context['job'], 'Export context/job differs')
    import torch
    shape = (16540, 63, 250) if _fixture is None else tuple(_fixture['EEG_shape'])
    width = 1024 if _fixture is None else _fixture['width']
    channels = list(range(46, 63)) if _fixture is None else _fixture['selected_indices']
    count = 8270 if _fixture is None else _fixture['rows']
    require(type(batch_size) is int and 1 <= batch_size <= 256, 'Finite native export batches required')
    G.validate_training_tensors(context, eeg, _fixture=_fixture)
    rows = job['expected_export_rows']
    require(len(rows) == count and len(set(rows)) == count and all(type(i) is int and 0 <= i < shape[0] for i in rows), 'Export all and only the ordered held global array rows')
    require(job['role'] in ('official_oof0', 'official_oof1'), 'Protected-T/full/published caches cannot replace new official OOF')
    if _fixture is None:
        require(sum(p.numel() for p in model.parameters()) == 6452224, 'Use the actual unchanged full native BrainMLP')
    require(all(k in model.state_dict() for k in E.AFFINE_KEYS), 'Native output own-affine keys missing')
    model = model.to(device=device, dtype=torch.float32).eval().requires_grad_(False)
    values = []
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            if _fixture is None: G.require_context(context)
            ids = rows[start:start + batch_size]
            # Global array_row indexes the full EEG tensor, never the position
            # inside this fold's cache. All original held ordering is retained.
            x = eeg[ids][:, channels, :].to(device=device, dtype=torch.float32)
            result = model(x)
            require(result.dtype == torch.float32 and tuple(result.shape) == (len(ids), width)
                and bool(torch.isfinite(result).all()), 'Native raw-LN export shape/dtype/finite failure')
            values.append(result.detach().cpu())
        info = dict(model_training_mode=model.training, gradients_enabled=torch.is_grad_enabled(),
            all_model_parameters_frozen=all(not p.requires_grad for p in model.parameters()),
            actual_full8270_export=_fixture is None, fixture_only=_fixture is not None)
    state = model.state_dict()
    own = [state[k].detach().cpu().float().clone() for k in E.AFFINE_KEYS]
    require(all(tuple(v.shape) == (width,) and bool(torch.isfinite(v).all()) for v in own), 'Own output affine shape/finite failure')
    return dict(condition=torch.cat(values).contiguous(), array_rows=torch.tensor(rows, dtype=torch.long),
        affine_weight=own[0], affine_bias=own[1]), info


def validate_cache_tensors(cache, checkpoint_model, job, fit, exporter_sha256, *, _fixture=None):
    """Check ALL rows/values and checkpoint-own affine, not a sampled cache."""
    import torch
    width, count = (1024, 8270) if _fixture is None else (_fixture['width'], _fixture['rows'])
    require(set(cache) == CACHE_KEYS, 'Export cache exact schema differs')
    expected = dict(subject=job['subject'], role=job['role'], model_sha256=fit['model_sha256'],
        registration_sha256=fit['registration_sha256'], registration_core_sha256=job['registration_core_sha256'],
        exporter_source_sha256=exporter_sha256, training_producer_source_sha256=fit['producer_source_sha256'],
        EEG_input_sha256=job['registration']['EEG_input']['sha256'], normalization='raw LayerNorm output',
        additional_L2_normalization=False, affine_source_checkpoint_sha256=fit['model_sha256'], affine_keys=E.AFFINE_KEYS)
    require(all(cache[k] == v for k, v in expected.items()), 'Cache subject/fold/EEG/registration/model/source chain differs')
    condition, rows = cache['condition'], cache['array_rows']
    require(isinstance(condition, torch.Tensor) and condition.dtype == torch.float32 and tuple(condition.shape) == (count, width), 'Full cache condition shape/dtype differs')
    require(isinstance(rows, torch.Tensor) and rows.dtype == torch.long and torch.equal(rows, torch.tensor(job['expected_export_rows'], dtype=torch.long)), 'Cache global array_rows/order differs')
    for chunk in condition.split(256): require(bool(torch.isfinite(chunk).all()), 'Nonfinite cached conditions')
    for name, key in zip(('affine_weight', 'affine_bias'), E.AFFINE_KEYS):
        value, own = cache[name], checkpoint_model[key].detach().cpu()
        require(value.dtype == torch.float32 and tuple(value.shape) == (width,) and own.dtype == torch.float32
            and torch.equal(value, own), 'Cache affine is not this exact checkpoint own final-LN affine')
    return dict(status='CPU_TINY_REAL_TENSOR_FIXTURE_ONLY' if _fixture else 'ALL8270_ACTUAL_CACHE_VALUES_ROWS_AND_OWN_AFFINE_VERIFIED',
        actual_full8270_verified=_fixture is None, shape=list(condition.shape), array_rows=list(job['expected_export_rows']),
        condition_tensor_sha256=tensor_sha(condition), affine_weight_sha256=tensor_sha(cache['affine_weight']),
        affine_bias_sha256=tensor_sha(cache['affine_bias']))


def make_cache(tensors, job, fit, exporter_sha):
    return dict(tensors, affine_keys=E.AFFINE_KEYS, subject=job['subject'], role=job['role'],
        model_sha256=fit['model_sha256'], registration_sha256=fit['registration_sha256'],
        registration_core_sha256=job['registration_core_sha256'], exporter_source_sha256=exporter_sha,
        training_producer_source_sha256=fit['producer_source_sha256'], EEG_input_sha256=job['registration']['EEG_input']['sha256'],
        normalization='raw LayerNorm output', additional_L2_normalization=False, affine_source_checkpoint_sha256=fit['model_sha256'])


def scan_training_epochs(job, fit, *, _fixture=None):
    """Read the real producer's immutable epoch transactions.

    Official callers always use25 epochs,9 updates/epoch,8270 rows. The private
    CPU checker may shrink transaction counts; no execution CLI exposes it.
    """
    import brainhive_official_encoder_training_v1 as training  # stdlib only.
    budget = dict(updates_per_epoch=9, total_epochs=25, full_epoch_rows=8270)
    if _fixture is not None:
        require(set(_fixture) == set(budget) and all(type(v) is int and v > 0 for v in _fixture.values()),
            'Private CPU transaction fixture requires explicit positive counts')
        budget = _fixture
    return training.scan_committed_epochs(job['output_root'], fit['registration_sha256'], job['registration_core_sha256'],
        producer_source_sha256=fit['producer_source_sha256'], subject=job['subject'], role=job['role'], **budget)


def load_fit_chain(context):
    G.require_context(context); job = context['job']; paths = job['outputs']
    registration = G.read(paths['registration']); fit = G.read(paths['fit_completion'])
    require(registration == job['registration'] and E.core_sha(registration) == job['registration_core_sha256'], 'Actual fitted registration differs from the official task')
    require(G.sha(paths['registration']) == fit['registration_sha256'] and G.sha(paths['final_checkpoint']) == fit['model_sha256'], 'Actual registration/checkpoint file SHA differs')
    require(fit['producer_source_sha256'] == context['source_sha256']['train_BrainHIVE_official_encoder_v1.py'], 'Fitted checkpoint came from another training producer')
    fit_review = G.bound(fit['execution_review'])
    require(fit_review['status'] == G.REVIEW_STATUS and fit_review['source_sha256'] == context['source_sha256']
        and E.core_sha(fit_review['payload_contract']) == context['payload_contract_sha256'], 'Actual training execution review/source/input chain differs')
    job_name = fit['actual_wrapper_job_name']
    require(Path(job_name).name == job_name and job_name in fit_review['tasks'][job['task_id']]['train_job_names'], 'Unregistered actual fit wrapper job')
    actual_job_path = G.ROOT / 'audit' / (job_name + '_job.json'); actual_job = G.read(actual_job_path)
    command = actual_job['command']; entry = str(G.ROOT / 'train_BrainHIVE_official_encoder_v1.py')
    require(actual_job['status'] == 'COMPLETED' and actual_job['exit_code'] == 0
        and actual_job['pid'] == fit['actual_training_pid'] and actual_job['wrapper_pid'] == fit['actual_wrapper_pid']
        and actual_job['charged_seconds'] > 0 and entry in command
        and '--task-id' in command and command[command.index('--task-id') + 1] == job['task_id']
        and '--execution-review' in command and command[command.index('--execution-review') + 1] == fit['execution_review']['path'],
        'Actual fit wrapper has not closed successfully for this exact task/source/review')
    provenance_path = Path(job['output_root']) / 'input_provenance.json'
    require(fit['input_provenance']['path'] == str(provenance_path), 'Fitted input provenance belongs to another namespace')
    provenance = G.bound(fit['input_provenance'])
    require(fit['payload_contract_sha256'] == provenance['payload_contract_sha256'] == context['payload_contract_sha256']
        and provenance['schema'] == 'BrainHIVE_official_encoder_input_provenance_v1'
        and provenance['status'] == 'ACTUAL_SOURCE_BOUND_TRAIN_INPUT_PAYLOADS_VERIFIED'
        and provenance['task_id'] == job['task_id'] and provenance['subject'] == job['subject'] and provenance['role'] == job['role']
        and provenance['registration_sha256'] == fit['registration_sha256']
        and provenance['registration_core_sha256'] == job['registration_core_sha256']
        and provenance['source_sha256'] == context['source_sha256'], 'Actual fitted EEG/visual/native-source payload provenance differs')
    epochs = [json.loads(line) for line in Path(paths['epoch_log']).read_text().splitlines()]
    E.validate_fit_metadata(job, fit, epochs)
    committed, pending = scan_training_epochs(job, fit)
    require(len(committed) == 25 and [item['row'] for item in committed] == epochs
        and fit['final_epoch_commit'] == committed[-1]['commit_ref'], 'Full immutable25epoch training identity chain is incomplete')
    return fit, epochs, dict(final_epoch_commit=committed[-1]['commit_ref'], input_provenance=fit['input_provenance'],
        actual_fit_wrapper_job=G.ref(actual_job_path),
        retained_pending_transactions=pending, all25_committed_epoch_payload_SHAs_verified=True)


def sync_directory(directory):
    if os.name == 'posix':
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(fd)
        finally: os.close(fd)


def write_once(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    sync_directory(Path(path).parent)


def execute(context):
    G.require_context(context); require(context['action'] == 'export', 'Export-only real preflight context required')
    G.check_output_budget(context)
    started = time.monotonic(); job = context['job']; fit, epochs, fit_chain = load_fit_chain(context)
    source_sha = G.sha(__file__)
    require(source_sha == context['source_sha256'][Path(__file__).name], 'Exporter source changed after review')
    paths = job['outputs']; output = Path(job['output_root'])
    require(output.is_dir() and not output.is_symlink(), 'Only the existing private official training task namespace may receive export')
    cache_path, receipt_path = Path(paths['condition_cache']), Path(paths['export_receipt'])
    require(cache_path.exists() == receipt_path.exists(), 'Orphan/partial export is retained; explicit reviewed repair required, never exists->skip')
    payloads = G.verify_payload_files(context, include_visual=False)
    from brainhive_native_models_v1 import torch, classes
    require(torch.__version__ == context['payload_contract']['runtime']['torch_version'], 'Actual reviewed Torch runtime changed')
    torch.set_num_threads(2)
    state = torch.load(paths['final_checkpoint'], weights_only=True, map_location='cpu', mmap=True)
    validate_checkpoint(state, job, fit)
    model = classes()['BrainMLP'](4250, 1024, num_layers=1, dropout=.1).float()
    require(sum(p.numel() for p in model.parameters()) == 6452224, 'Exact native BrainMLP parameter count differs')
    model.load_state_dict(state['model'], strict=True)
    eeg = torch.load(payloads['EEG_path'], weights_only=True, map_location='cpu', mmap=True)
    G.validate_training_tensors(context, eeg)
    if cache_path.exists():
        require(not cache_path.is_symlink() and not receipt_path.is_symlink(), 'Never adopt export aliases')
        receipt = G.read(receipt_path)
        require(receipt['status'] == 'OFFICIAL_OOF_ALL8270_ACTUAL_CONDITIONS_AND_OWN_AFFINE_EXPORTED'
            and receipt['exporter_source_sha256'] == source_sha and receipt['cache_sha256'] == G.sha(cache_path)
            and receipt['payload_contract_sha256'] == context['payload_contract_sha256']
            and receipt['runtime'] == context['payload_contract']['runtime'], 'Existing export receipt/source/cache SHA differs')
        E.validate_export_metadata(job, fit, epochs, receipt)
        cache = torch.load(cache_path, weights_only=True, map_location='cpu', mmap=True)
        check = validate_cache_tensors(cache, state['model'], job, fit, source_sha)
        require(all(receipt[k] == check[k] for k in ('condition_tensor_sha256', 'affine_weight_sha256', 'affine_bias_sha256')), 'Existing complete cache tensor/affine receipt differs')
        return dict(status='ALREADY_COMPLETE_ONLY_AFTER_ACTUAL_FULL_CACHE_AND_TRAINING_CHAIN_VERIFICATION',
            task_id=job['task_id'], cache=G.ref(cache_path), export_receipt=G.ref(receipt_path), verified=check, new_files_written=0)
    require(torch.cuda.is_available(), 'Actual separately admitted GPU export is required')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    tensors, mode = infer_held_conditions(model, eeg, job, context=context, device='cuda')
    cache = make_cache(tensors, job, fit, source_sha)
    check = validate_cache_tensors(cache, state['model'], job, fit, source_sha)
    G.require_context(context)
    tensor_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    # Pending and canonical paths share one inode; enforce the same unique
    # allocated-byte cap used by the training executor, including new metadata.
    G.check_output_budget(context, additional_bytes=tensor_bytes + (2 << 20))
    temporary = cache_path.with_name(cache_path.name + '.pending.' + str(os.getpid()))
    with temporary.open('xb') as stream:
        torch.save(cache, stream); stream.flush(); os.fsync(stream.fileno())
    saved = torch.load(temporary, weights_only=True, map_location='cpu', mmap=True)
    require(validate_cache_tensors(saved, state['model'], job, fit, source_sha) == check, 'Saved actual export roundtrip differs')
    os.link(temporary, cache_path); sync_directory(output)  # Atomic no-replace; retained pending evidence is never deleted.
    G.check_output_budget(context)
    record = dict(status='OFFICIAL_OOF_ALL8270_ACTUAL_CONDITIONS_AND_OWN_AFFINE_EXPORTED',
        subject=job['subject'], role=job['role'], registration_core_sha256=job['registration_core_sha256'],
        checkpoint_path=fit['model_path'], checkpoint_sha256=fit['model_sha256'], cache_path=str(cache_path), cache_sha256=G.sha(cache_path),
        EEG_input_path=payloads['EEG_path'], EEG_input_sha256=payloads['EEG_sha256'], physical_repeat_count=4,
        array_rows=job['expected_export_rows'], condition_shape=[8270, 1024], condition_dtype='float32',
        normalization='raw LayerNorm output', additional_L2_normalization=False, all_condition_values_finite=True,
        model_training_mode=mode['model_training_mode'], gradients_enabled=mode['gradients_enabled'], all_model_parameters_frozen=mode['all_model_parameters_frozen'],
        affine_source_checkpoint_sha256=fit['model_sha256'], affine_keys=E.AFFINE_KEYS, affine_shape=[1024],
        official_test_inputs_loaded=False, target_image_or_concept_used_for_export=False, official_retrieval_computed=False, model_fitted_during_export=False,
        exporter_source_sha256=source_sha, training_producer_source_sha256=fit['producer_source_sha256'], fit_chain=fit_chain,
        condition_tensor_sha256=check['condition_tensor_sha256'], affine_weight_sha256=check['affine_weight_sha256'], affine_bias_sha256=check['affine_bias_sha256'],
        execution_review=context['execution_review_ref'], admission=context['admission_ref'], wrapper_job=context['wrapper_job_ref'],
        payload_contract_sha256=context['payload_contract_sha256'], runtime=context['payload_contract']['runtime'],
        seconds=time.monotonic() - started, actual_full8270_export=True, retained_temporary_cache=str(temporary))
    E.validate_export_metadata(job, fit, epochs, record)
    G.require_context(context)
    write_once(receipt_path, (json.dumps(record, indent=2, allow_nan=False) + '\n').encode())
    G.check_output_budget(context)
    require(G.sha(__file__) == source_sha, 'Exporter changed during execution')
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-directory', required=True, type=Path)
    parser.add_argument('--task-id', required=True)
    parser.add_argument('--execution-review', required=True, type=Path)
    args = parser.parse_args()
    context = G.preflight(args.plan_directory, args.task_id, args.execution_review,
        action='export', entry_name=Path(__file__).name)
    result = execute(context)
    print(json.dumps({k: result[k] for k in ('status', 'subject', 'role') if k in result}), flush=True)


if __name__ == '__main__':
    main()
