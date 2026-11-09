"""Real native InfoNCE training math and immutable epoch-boundary resume.

No Torch/model import, input load, job admission or directory creation happens
on import. The caller must supply already admitted tensors/native modules.
Each committed epoch owns its resume.pt and true epoch.json. Both plus their
SHA manifest are published by one same-filesystem directory rename. Pending
transactions and every previous checkpoint are retained.
"""
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

RESUME_FIELDS = ('model', 'logit_scale', 'optimizer', 'scheduler', 'torch_rng', 'cuda_rng',
    'shuffle_rng', 'epoch', 'updates', 'registration_sha256', 'registration_core_sha256', 'producer_source_sha256')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tick(deadline):
    if not time.monotonic() < deadline:
        raise TimeoutError('Finite admitted encoder-job deadline; committed epoch states retained')


def sha(path):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Hash only a regular owned file')
    fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), 'Regular file required')
        with os.fdopen(os.dup(fd), 'rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = os.fstat(fd)
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'File changed during hash')
        return digest
    finally:
        try:
            if os.name == 'posix' and hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)


def sync_directory(path):
    if os.name == 'posix':
        fd = os.open(path, os.O_RDONLY | getattr(os, 'O_DIRECTORY', 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def json_bytes(value):
    return (json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def write_once(path, raw):
    """Exclusive immutable publication; any interrupted pending file stays."""
    path = Path(path)
    require(not path.exists() and not path.is_symlink(), 'Never overwrite existing final/resume/metadata')
    pending = path.with_name(path.name + '.pending.' + str(os.getpid()))
    with pending.open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    # Link is an atomic no-replace publication. Keep the pending hardlink too;
    # deleting an earlier snapshot is never part of this engine.
    os.link(pending, path)
    sync_directory(path.parent)


def seed_everything(torch, seed=1729):
    require(seed == 1729, 'The training seed is fixed')
    torch.manual_seed(seed)
    if torch.cuda.is_initialized():
        torch.cuda.manual_seed_all(seed)


def frozen_stochastic_teacher(torch, teacher):
    teacher.requires_grad_(False)
    require(not any(isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
        for module in teacher.modules()), 'This native F contract has no mutable BatchNorm running state')
    dropouts = [module for module in teacher.modules() if isinstance(module, torch.nn.Dropout)]
    require(dropouts, 'Native FusionEncoder dropout must exist')
    for module in dropouts:
        module.p = .1
    teacher.train()
    require(all(not p.requires_grad for p in teacher.parameters()), 'Teacher weights must remain frozen')
    return teacher


def make_optimizer(torch, model, logit_scale, total_updates):
    require(total_updates > 0, 'Positive fixed total updates required')
    norm_ids = {id(p) for module in model.modules() if isinstance(module, torch.nn.LayerNorm)
        for p in module.parameters()}
    decay = [p for name, p in model.named_parameters() if not name.endswith('bias') and id(p) not in norm_ids] + [logit_scale]
    no_decay = [p for name, p in model.named_parameters() if name.endswith('bias') or id(p) in norm_ids]
    require(len({id(p) for p in decay + no_decay}) == len(list(model.parameters())) + 1, 'Every student/logit parameter exactly once')
    optimizer = torch.optim.AdamW([dict(params=decay, weight_decay=.05),
        dict(params=no_decay, weight_decay=0.)], lr=5e-4, betas=(.9, .995))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
        lambda step: .5 * (1. + math.cos(math.pi * step / total_updates)))
    return optimizer, scheduler


def cpu_tree(torch, value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_tree(torch, item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_tree(torch, item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_tree(torch, item) for item in value)
    return value


def snapshot(torch, model, logit_scale, optimizer, scheduler, shuffle_rng, epoch, updates,
        registration_sha256, registration_core_sha256, producer_source_sha256, device):
    value = dict(model=cpu_tree(torch, model.state_dict()), logit_scale=cpu_tree(torch, logit_scale),
        optimizer=cpu_tree(torch, optimizer.state_dict()), scheduler=cpu_tree(torch, scheduler.state_dict()),
        torch_rng=torch.get_rng_state().clone(), cuda_rng=[x.clone() for x in torch.cuda.get_rng_state_all()]
            if str(device).startswith('cuda') else [],
        shuffle_rng=shuffle_rng.get_state().clone(), epoch=epoch, updates=updates,
        registration_sha256=registration_sha256, registration_core_sha256=registration_core_sha256,
        producer_source_sha256=producer_source_sha256)
    require(all(field in value for field in RESUME_FIELDS), 'Complete exact-resume state required')
    return value


def restore(torch, value, model, logit_scale, optimizer, scheduler, shuffle_rng,
        registration_sha256, registration_core_sha256, producer_source_sha256, device, updates_per_epoch, total_updates):
    require(set(value) == set(RESUME_FIELDS), 'Exact immutable epoch resume schema required')
    require(value['registration_sha256'] == registration_sha256 and
        value['registration_core_sha256'] == registration_core_sha256 and value['producer_source_sha256'] == producer_source_sha256,
        'Resume registration or producer source differs')
    require(isinstance(value['epoch'], int) and value['epoch'] >= 1 and
        value['updates'] == value['epoch'] * updates_per_epoch and value['updates'] <= total_updates,
        'Resume must be a full committed epoch inside the fixed budget')
    expected_lr = 5e-4 * .5 * (1. + math.cos(math.pi * value['updates'] / total_updates))
    groups = value['optimizer']['param_groups']
    require(len(groups) == 2 and len(value['optimizer']['state']) == len(list(model.parameters())) + 1,
        'All native student/logit Adam states must exist')
    identifiers = [identifier for group in groups for identifier in group['params']]
    require(len(set(identifiers)) == len(identifiers) and set(identifiers) == set(value['optimizer']['state']),
        'Every optimizer state must map to exactly one native parameter')
    for group, actual_group, decay in zip(groups, optimizer.param_groups, (.05, 0.)):
        require(group['weight_decay'] == decay and tuple(group['betas']) == (.9, .995) and group['eps'] == 1e-8
            and math.isclose(group['lr'], expected_lr, rel_tol=0., abs_tol=1e-16)
            and len(group['params']) == len(actual_group['params']), 'Restored AdamW settings differ')
        for identifier, parameter in zip(group['params'], actual_group['params']):
            state = value['optimizer']['state'][identifier]
            require(set(state) == {'step', 'exp_avg', 'exp_avg_sq'} and float(state['step']) == value['updates'],
                'A native parameter is missing its complete Adam step/moments')
            require(all(tuple(state[name].shape) == tuple(parameter.shape) and state[name].dtype == parameter.dtype
                and torch.isfinite(state[name]).all().item() for name in ('exp_avg', 'exp_avg_sq')), 'Invalid Adam moments')
    schedule = value['scheduler']
    require(schedule['last_epoch'] == value['updates'] and schedule['_step_count'] == value['updates'] + 1
        and schedule['base_lrs'] == [5e-4, 5e-4]
        and all(math.isclose(lr, expected_lr, rel_tol=0., abs_tol=1e-16) for lr in schedule['_last_lr']),
        'Restored cosine scheduler differs from the fixed update boundary')
    model.load_state_dict(value['model'], strict=True)
    with torch.no_grad():
        logit_scale.copy_(value['logit_scale'].to(device))
    optimizer.load_state_dict(value['optimizer']); scheduler.load_state_dict(value['scheduler'])
    require(all(torch.isfinite(p).all().item() for p in model.parameters()) and
        torch.isfinite(logit_scale).all().item(), 'Nonfinite restored parameters')
    # This is last: construction/loading/inspection must precede RNG restore.
    # The caller must not perform a random probe between here and run_epoch().
    shuffle_rng.set_state(value['shuffle_rng']); torch.set_rng_state(value['torch_rng'])
    if str(device).startswith('cuda'):
        require(len(value['cuda_rng']) == torch.cuda.device_count(), 'CUDA RNG device coverage differs')
        torch.cuda.set_rng_state_all(value['cuda_rng'])
    else:
        require(value['cuda_rng'] == [], 'CPU synthetic resume cannot claim CUDA RNG coverage')
    return value['epoch'], value['updates']


def run_epoch(torch, model, teacher, logit_scale, optimizer, scheduler, shuffle_rng,
        eeg, visual_embeddings, *, epoch, updates, total_updates, batch_size, subject, role,
        device, deadline, autocast_bfloat16=False, observer=None):
    """The actual native symmetric InfoNCE update, including every last row."""
    import torch.nn.functional as F
    tick(deadline)
    count = len(eeg)
    require(count > 1 and batch_size > 1 and all(len(v) == count for v in visual_embeddings.values()), 'Complete same-row EEG/visual arrays required')
    require(teacher.training and all(not p.requires_grad for p in teacher.parameters()), 'Use live stochastic frozen F.vision_model')
    model.train(); parameters = list(model.parameters()) + [logit_scale]
    order = torch.randperm(count, generator=shuffle_rng, device='cpu').to(device)
    seen = 0; summed_loss = 0.; last_norm = None
    context = (lambda: torch.autocast('cuda', dtype=torch.bfloat16)) if autocast_bfloat16 else nullcontext
    for ids in order.split(batch_size):
        tick(deadline)
        if observer:
            observer('before_update')
        optimizer.zero_grad(set_to_none=True)
        with context():
            brain = F.normalize(model(eeg[ids]), dim=-1)
            with torch.no_grad():
                actual_F = teacher(**{name: values[ids] for name, values in visual_embeddings.items()})
            visual = F.normalize(actual_F.to(brain.dtype), dim=-1)
            require(brain.shape == visual.shape and brain.ndim == 2, 'Native student/F dimensions differ')
            logits = brain @ visual.T * logit_scale.exp()
            labels = torch.arange(len(ids), device=device)
            loss = .5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))
        require(torch.isfinite(loss).item(), 'Nonfinite native InfoNCE')
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); updates += 1
        summed_loss += float(loss.detach()) * len(ids); seen += len(ids)
        last_norm = float(norm)
        require(updates <= total_updates, 'No updates beyond the fixed final epoch')
        if observer:
            observer('after_update')
    require(seen == count and all(p.grad is None for p in teacher.parameters()), 'Teacher gradients or dropped rows')
    return dict(stage='BRAINHIVE_ALIGNMENT', subject=subject, role=role, epoch=epoch,
        updates=updates, expected_updates=total_updates, full_epoch_rows=seen,
        mean_train_InfoNCE=summed_loss / seen, grad_norm_last=last_norm), updates


def publish_epoch(torch, root, value, row, *, fault=None):
    """Publish resume+real log atomically; fault injection is CPU-check-only."""
    root = Path(root); epochs = root / 'epochs'
    require(root.is_absolute() and root.resolve(strict=True) == root and not root.is_symlink(), 'Private resolved epoch output root required')
    require(type(value['epoch']) is int and 1 <= value['epoch'] <= 25, 'Bounded integer epoch required')
    epochs.mkdir(exist_ok=True)
    final = epochs / ('epoch-' + str(value['epoch']).zfill(2))
    require(not final.exists() and not final.is_symlink(), 'Never overwrite a committed epoch')
    pending = Path(tempfile.mkdtemp(prefix='epoch-' + str(value['epoch']).zfill(2) + '.pending.', dir=epochs))
    require(pending.stat().st_dev == epochs.stat().st_dev, 'Epoch publication must stay on one filesystem')
    state = pending / 'resume.pt'
    with state.open('xb') as stream:
        torch.save(value, stream); stream.flush(); os.fsync(stream.fileno())
    with (pending / 'epoch.json').open('xb') as stream:
        stream.write(json_bytes(row)); stream.flush(); os.fsync(stream.fileno())
    if fault:
        fault('payloads_written_before_commit')
    commit = dict(schema='BrainHIVE_official_encoder_epoch_transaction_v1', epoch=value['epoch'], updates=value['updates'],
        registration_sha256=value['registration_sha256'], registration_core_sha256=value['registration_core_sha256'],
        producer_source_sha256=value['producer_source_sha256'],
        resume_sha256=sha(state), epoch_record_sha256=sha(pending / 'epoch.json'))
    with (pending / 'commit.json').open('xb') as stream:
        stream.write(json_bytes(commit)); stream.flush(); os.fsync(stream.fileno())
    sync_directory(pending)
    pending.rename(final); sync_directory(epochs)
    if fault:
        fault('epoch_rename_complete_before_canonical_publication')
    return dict(path=str(final / 'commit.json'), sha256=sha(final / 'commit.json'), **commit)


def scan_committed_epochs(root, registration_sha256, registration_core_sha256, *,
        producer_source_sha256, subject, role, updates_per_epoch, total_epochs, full_epoch_rows):
    root = Path(root); epochs = root / 'epochs'
    if not epochs.exists():
        return [], []
    require(epochs.is_dir() and not epochs.is_symlink(), 'Private epoch transaction directory required')
    committed = []; pending = []
    for path in sorted(epochs.iterdir()):
        require(path.is_dir() and not path.is_symlink(), 'Unexpected object in epoch transaction namespace')
        if '.pending.' in path.name:
            require(path.name.startswith('epoch-') and path.name[6:8].isdigit()
                and 1 <= int(path.name[6:8]) <= total_epochs and path.name[8:].startswith('.pending.'), 'Unrecognized pending transaction namespace')
            pending.append(str(path)); continue
        require(path.name == 'epoch-' + str(len(committed) + 1).zfill(2), 'Committed epochs must be complete and consecutive')
        number = len(committed) + 1
        require(number <= total_epochs and {p.name for p in path.iterdir()} == {'resume.pt', 'epoch.json', 'commit.json'}, 'Exact epoch transaction payloads required')
        commit = json.loads((path / 'commit.json').read_text(encoding='utf-8'))
        row = json.loads((path / 'epoch.json').read_text(encoding='utf-8'))
        require(commit['schema'] == 'BrainHIVE_official_encoder_epoch_transaction_v1' and
            commit['epoch'] == row['epoch'] == number and commit['updates'] == row['updates'] == number * updates_per_epoch,
            'Epoch/log/checkpoint transaction differs')
        require(commit['registration_sha256'] == registration_sha256 and
            commit['registration_core_sha256'] == registration_core_sha256 and
            commit['producer_source_sha256'] == producer_source_sha256, 'Epoch belongs to a different registration or producer')
        require(row['subject'] == subject and row['role'] == role and row['expected_updates'] == total_epochs * updates_per_epoch,
            'Wrong epoch producer identity/budget')
        require(row['full_epoch_rows'] == full_epoch_rows and math.isfinite(row['mean_train_InfoNCE'])
            and math.isfinite(row['grad_norm_last']), 'Incomplete/nonfinite actual epoch log')
        require(sha(path / 'resume.pt') == commit['resume_sha256'] and
            sha(path / 'epoch.json') == commit['epoch_record_sha256'], 'Changed epoch state or real log record')
        committed.append(dict(directory=str(path), commit=commit, row=row,
            commit_ref=dict(path=str(path / 'commit.json'), sha256=sha(path / 'commit.json'))))
    return committed, pending


def publish_final(torch, root, model, logit_scale, *, registration_sha256,
        registration_core_sha256, producer_source_sha256, committed, total_epochs, total_updates, subject, role):
    """Publish final once, or verify an existing exact final without replacement."""
    root = Path(root)
    require(len(committed) == total_epochs and committed[-1]['commit']['updates'] == total_updates,
        'All complete epochs precede final publication')
    rows = [item['row'] for item in committed]
    log = b''.join(json.dumps(row, sort_keys=True, allow_nan=False).encode() + b'\n' for row in rows)
    log_path = root / 'epochs.jsonl'
    if log_path.exists():
        require(log_path.read_bytes() == log, 'Existing canonical epoch log differs; preserve and stop')
    else:
        write_once(log_path, log)
    payload = dict(model=cpu_tree(torch, model.state_dict()), logit_scale=cpu_tree(torch, logit_scale),
        seed=1729, epochs=total_epochs, updates=total_updates, registration_sha256=registration_sha256,
        registration_core_sha256=registration_core_sha256, producer_source_sha256=producer_source_sha256)
    target = root / 'brain_final.pt'
    if target.exists():
        require(not target.is_symlink(), 'Do not adopt a final alias')
        actual = torch.load(target, weights_only=True, map_location='cpu')
        require(set(actual) == set(payload) and all(actual[key] == payload[key] for key in
            ('seed', 'epochs', 'updates', 'registration_sha256', 'registration_core_sha256', 'producer_source_sha256')),
            'Existing final provenance differs')
        require(set(actual['model']) == set(payload['model']) and all(torch.equal(actual['model'][key], payload['model'][key])
            for key in payload['model']) and torch.equal(actual['logit_scale'], payload['logit_scale']),
            'Existing final differs from last complete epoch')
    else:
        temporary = root / ('brain_final.pt.pending.' + str(os.getpid()))
        with temporary.open('xb') as stream:
            torch.save(payload, stream); stream.flush(); os.fsync(stream.fileno())
        os.link(temporary, target); sync_directory(root)
    # Compatibility resume is a no-replace hardlink to the final complete
    # optimizer state. Earlier exact-resume payloads remain in every epoch dir.
    resume = root / 'resume.pt'; latest = Path(committed[-1]['directory']) / 'resume.pt'
    if resume.exists():
        require(sha(resume) == committed[-1]['commit']['resume_sha256'], 'Canonical resume is stale; committed epochs remain authoritative')
    else:
        os.link(latest, resume); sync_directory(root)
    require(all(torch.isfinite(value).all().item() for value in payload['model'].values()), 'Nonfinite final model')
    return dict(subject=subject, role=role, model_path=str(target), model_sha256=sha(target),
        registration_sha256=registration_sha256, registration_core_sha256=registration_core_sha256,
        producer_source_sha256=producer_source_sha256,
        final_epoch_commit=committed[-1]['commit_ref'], epochs=total_epochs, updates=total_updates,
        canonical_resume_role='immutable compatibility alias to the final epoch; resume authority is committed epoch transactions',
        parameter_updates_from_uncommitted_epochs_certified=0)
