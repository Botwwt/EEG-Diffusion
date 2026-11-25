"""Train a personal token-photo encoder with concept-held-out EEG predictions from the other nine participants."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from filelock import FileLock
import numpy as np
import torch

from paper_middle_teacher import completed_fit, losses, registry
from paper_object_tokens import BATCH, EPOCHS, SEED, cross_photo_targets, epoch_batches, group_index
from paper_render_cache import PinnedFiles, atomic_file, key, require, save_json, sha

MODES = ('token-peer-own', 'token-peer-sibling')
SUBJECTS = list(range(1, 11))
CACHE = 'paper_peer_oof_training.pt'


def array_hash(x):
    return hashlib.sha256(np.ascontiguousarray(x)).hexdigest()


def concept_partition(labels):
    objects, inverse = np.unique(labels, return_inverse=True)
    require(len(objects) >= 3, 'Three concept folds require at least three objects')
    folds = np.empty(len(objects), dtype=np.int64)
    folds[np.random.default_rng(SEED).permutation(len(objects))] = np.arange(len(objects)) % 3
    return folds[inverse]


def sibling_permutation(rows, labels):
    """One fixed random cycle per training object; returned indices address the ordered training rows."""
    require(len(rows) == len(labels) == len(np.unique(rows)), 'Expected unique, labeled training photos')
    rng = np.random.default_rng(SEED + 1)
    permutation = np.empty(len(rows), dtype=np.int64)
    for obj in np.unique(labels):
        local = np.flatnonzero(labels == obj)
        require(len(local) >= 2, 'A sibling positive requires at least two training photos per object')
        cycle = rng.permutation(local[np.argsort(rows[local])])
        permutation[cycle] = np.roll(cycle, -1)
    return permutation


def binned_eeg(trials):
    require(trials.ndim == 4 and trials.shape[1:] == (4, 17, 250), 'Expected four real EEG presentations per photo')
    return trials.float().mean(1).reshape(len(trials), 17, 50, 5).mean(-1).flatten(1)


def selected_eeg(path, rows, device, binned):
    raw = np.load(path, mmap_mode='r', allow_pickle=False)
    require(raw.shape == (16540, 4, 17, 250) and raw.dtype.kind == 'f', 'Unexpected participant training EEG array')
    blocks = []
    for indices in np.array_split(rows, math.ceil(len(rows) / 128)):
        trials = torch.from_numpy(raw[indices]).to(device=device, dtype=torch.float32)
        blocks.append(binned_eeg(trials) if binned else trials.mean(1))
    eeg = torch.cat(blocks)
    require(torch.isfinite(eeg).all().item(), 'Nonfinite registered EEG')
    return eeg


def selected_visual(prep, rows, device):
    raw = np.load(prep / 'F_train.npy', mmap_mode='r', allow_pickle=False)
    require(raw.shape == (16540, 1024) and raw.dtype.kind == 'f', 'Unexpected registered visual F array')
    target = torch.from_numpy(raw[rows]).to(device=device, dtype=torch.float32)
    require(torch.isfinite(target).all().item(), 'Nonfinite registered visual target')
    return target


@torch.no_grad()
def oof_predict(eeg, visual, folds):
    require(eeg.ndim == visual.ndim == 2 and len(eeg) == len(visual) == len(folds)
            and np.array_equal(np.unique(folds), [0, 1, 2]), 'Expected three complete OOF partitions')
    x, y = eeg.double(), visual.double()
    fold = torch.as_tensor(folds, device=x.device)
    prediction = torch.empty_like(y, dtype=torch.float32)
    stats = []
    for held in range(3):
        train, test = fold != held, fold == held
        xt, yt = x[train], y[train]
        mean_x, mean_y = xt.mean(0), yt.mean(0)
        scale = xt.std(0, unbiased=False).clamp_min(1e-6)
        xt = (xt - mean_x) / scale
        cov = xt.T @ xt / len(xt)
        penalty = cov.diagonal().mean().item()
        require(math.isfinite(penalty) and penalty > 0., 'Degenerate training covariance')
        cov.diagonal().add_(penalty)
        weights = torch.linalg.solve(cov, xt.T @ (yt - mean_y) / len(xt))
        prediction[test] = (((x[test] - mean_x) / scale) @ weights + mean_y).float()
        stats.append({'fold': held, 'train_rows': len(xt), 'predicted_rows': int(test.sum()), 'lambda': penalty})
    require(torch.isfinite(prediction).all().item(), 'Nonfinite out-of-fold prediction')
    return prediction, stats


def registrations(prep, pins):
    src = dict(code=pins.pin(__file__), registry_code=pins.pin(Path(__file__).with_name('paper_middle_teacher.py')),
               fold_code=pins.pin(Path(__file__).with_name('common.py')), split=pins.pin(prep / 'paper_split.json'),
               photo_ids=pins.pin(prep / 'train_ids.json'), eeg_ids=pins.pin(prep / 'eeg_train_ids.json'),
               concept=pins.pin(prep / 'train_concept.npy'), F=pins.pin(prep / 'F_train.npy'), eeg={}, eeg_receipts={})
    rows, labels, members = registry(prep)
    folds = concept_partition(labels)
    permutation = sibling_permutation(rows, labels)
    src.update(ordered_rows=array_hash(rows.astype('<i8')), ordered_photos=key(members),
               fold_assignments=array_hash(folds.astype('<i8')), sibling_photo_rows=array_hash(rows[permutation].astype('<i8')))
    for subject in SUBJECTS:
        sid = f'sub-{subject:02d}'
        path = prep / f'{sid}_train.npy'
        src['eeg'][sid] = pins.pin(path)
        meta = path.with_suffix('.json')
        src['eeg_receipts'][sid] = pins.pin(meta)
        rec = json.loads(meta.read_bytes())
        require(rec['output_sha256'] == src['eeg'][sid] and rec.get('split', 'train') == 'train'
                and int(str(rec.get('subject', subject)).removeprefix('sub-')) == subject,
                f'{sid} training EEG differs from its receipt')
    require(len(set(src['eeg'].values())) == len(SUBJECTS), 'Expected ten different participant EEG arrays')
    pins.verify()
    return rows, labels, folds, permutation, src


def read_cache(path, rows, labels, folds, permutation, sources):
    x = torch.load(path, map_location='cpu', weights_only=True)
    require(x['schema'] == 1 and x['sources'] == sources and x['subject_ids'] == SUBJECTS,
            'OOF cache has different source files, code or participants')
    for name, values in [('rows', rows), ('labels', labels), ('folds', folds), ('sibling_rows', rows[permutation])]:
        require(x[name].dtype == torch.int64 and np.array_equal(x[name].numpy(), values), f'OOF cache differs on {name}')
    p = x['predictions']
    require(p.dtype == torch.float16 and p.shape == (10, len(rows), 1024), 'Expected ten float16 OOF F matrices')
    for sid, matrix in zip(SUBJECTS, p, strict=True):
        require(array_hash(matrix.numpy()) == x['prediction_sha256'][str(sid)] and torch.isfinite(matrix).all().item(),
                f'Invalid cached prediction for participant {sid}')
    return x


def prepare(prep, cache_path, device):
    pins = PinnedFiles()
    rows, labels, folds, permutation, sources = registrations(prep, pins)
    with FileLock(str(cache_path) + '.lock'):
        pins.verify()
        if cache_path.exists():
            read_cache(cache_path, rows, labels, folds, permutation, sources)
            print('Existing matched OOF cache', sha(cache_path), flush=True)
            return
        start = time.perf_counter()
        visual = selected_visual(prep, rows, device)
        prediction = torch.empty(len(SUBJECTS), len(rows), 1024, dtype=torch.float16, device='cpu')
        stats, hashes = {}, {}
        for i, subject in enumerate(SUBJECTS):
            eeg = selected_eeg(prep / f'sub-{subject:02d}_train.npy', rows, device, binned=True)
            values, stats[str(subject)] = oof_predict(eeg, visual, folds)
            prediction[i].copy_(values)
            require(torch.isfinite(prediction[i]).all().item(), 'OOF values overflowed float16')
            hashes[str(subject)] = array_hash(prediction[i].numpy())
            print(f'Saved OOF predictions in memory for participant {subject:02d}/10', flush=True)
        meta = dict(schema=1, sources=sources, rows=torch.from_numpy(rows.astype('int64')), subject_ids=SUBJECTS,
                    labels=torch.from_numpy(labels.astype('int64')), folds=torch.from_numpy(folds.copy()),
                    sibling_rows=torch.from_numpy(rows[permutation].astype('int64')), predictions=prediction,
                    prediction_sha256=hashes, fold_statistics=stats, seed=SEED, sibling_seed=SEED + 1,
                    method='3-fold by concept, mean of 4 repeats, nonoverlapping 5-sample time bins across all 250 samples, 17 channels; train-only population z-score and intercept; lambda = mean covariance diagonal; fp64 solve',
                    elapsed_seconds=time.perf_counter() - start)
        pins.verify()
        atomic_file(cache_path, lambda f: torch.save(meta, f))
        print('Published shared OOF cache', cache_path, sha(cache_path), flush=True)


def other_people(predictions, subject_ids, subject, device):
    require(len(predictions) == len(subject_ids) == len(set(subject_ids)) and subject in subject_ids and len(subject_ids) > 1,
            'Expected unique teacher IDs and the student participant')
    peers = [sid for sid in subject_ids if sid != subject]
    mean = torch.zeros(predictions.shape[1:], device=device, dtype=torch.float32)
    for index, sid in enumerate(subject_ids):
        if sid != subject:
            mean.add_(predictions[index].to(device=device, dtype=torch.float32))
    return mean.div_(len(peers)), peers


def peer_losses(pred, target, center, variance, student, peer, peer_center):
    return losses(pred, target, center, variance, student - peer_center, peer - peer_center)


def fit(subject, modes, prep, cache_path):
    from common import HIVE, PRIOR, RUNS
    from paper_instance_residual import make_encoder
    if str(HIVE) not in sys.path:
        sys.path.insert(0, str(HIVE))
    from main.models_adapter import FusionEncoderModel, IPProjectionModel

    require(RUNS.name == f'personal-{subject:02d}', 'Expected the independent fit directory of this participant')
    require(len(modes) == len(set(modes)) and modes and set(modes) <= set(MODES), 'Choose distinct peer training arms')
    pins = PinnedFiles()
    rows, labels, folds, permutation, sources = registrations(prep, pins)
    cached_sha = pins.pin(cache_path)
    native = RUNS / f'enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt'
    dest = RUNS / f'enc/object_tokens/sub-{subject:02d}'
    dest.mkdir(parents=True, exist_ok=True)
    proof = dict(native=pins.pin(native), code=sources['code'], seed=SEED, split=sources['split'], concept=sources['concept'],
                 eeg=sources['eeg'][f'sub-{subject:02d}'], F=sources['F'], photo=sources['ordered_photos'],
                 adapter=pins.pin(PRIOR / 'fusion_encoder/diffusion_pytorch_model.safetensors'),
                 projection=pins.pin(PRIOR / 'proj/diffusion_pytorch_model.safetensors'),
                 peer_cache_sha256=cached_sha, peer_sources_sha256=key(sources), permutation=sources['sibling_photo_rows'],
                 token_control_code=pins.pin(Path(__file__).with_name('paper_object_tokens.py')),
                 loss_code=sources['registry_code'], encoder_code=pins.pin(HIVE / 'main/models_brain.py'),
                 encoder_layers=pins.pin(HIVE / 'main/layers.py'))
    control_path = dest / 'token-photo.json'
    if control_path.is_file():
        proof['photo_control_receipt'] = pins.pin(control_path)
        control = json.loads(control_path.read_bytes())
        require((control['mode'], control['subject'], control['seed'], control['steps'], control['photos'], control['concepts'])
                == ('token-photo', subject, SEED, 360, 13230, 1323)
                and all(control['sources'][k] == proof[k] for k in ('native', 'split', 'concept', 'eeg', 'F', 'adapter', 'projection'))
                and control['sources']['code'] == proof['token_control_code'], 'Personal token-photo baseline used different inputs')
    with FileLock(str(dest / '.peer-teacher.lock')):
        pins.verify()
        pending = [mode for mode in modes if not completed_fit(dest / f'{mode}.pt', proof, mode, subject)]
        if not pending:
            return
        cache = read_cache(cache_path, rows, labels, folds, permutation, sources)
        device = torch.device('cuda')
        teacher, peers = other_people(cache['predictions'], cache['subject_ids'], subject, device)
        del cache
        peer_center = teacher.mean(0, keepdim=True)
        center_sha = array_hash(peer_center.cpu().numpy())
        teacher_sha = array_hash(teacher.cpu().numpy())
        inverse, ordered = group_index(torch.as_tensor(labels, device=device))
        eeg = selected_eeg(prep / f'sub-{subject:02d}_train.npy', rows, device, binned=False)
        visual = selected_visual(prep, rows, device)
        adapter = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder='fusion_encoder').adapter.to(device).float().eval().requires_grad_(False)
        proj = IPProjectionModel.from_pretrained(str(PRIOR), subfolder='proj').to(device).float().eval().requires_grad_(False)
        with torch.no_grad():
            tokens = torch.cat([proj(adapter(v)) for v in visual.split(128)])
            require(tokens.shape == (len(rows), 4, 2048) and torch.isfinite(tokens).all().item(), 'Unexpected token-photo visual targets')
            center = tokens.mean(0)
            variance = (tokens - center).square().mean((0, 2)).clamp_min(1e-6)
            target = cross_photo_targets(tokens, inverse, ordered, torch.ones(4, device=device))
        steps = EPOCHS * ordered.shape[1] * math.ceil(len(ordered) / BATCH)
        require(steps == 360, 'Matched token-photo schedule requires 360 updates')
        state = torch.load(native, map_location='cpu', weights_only=True)
        for mode in pending:
            pair = np.arange(len(rows)) if mode == MODES[0] else permutation
            teacher_indices = torch.as_tensor(pair, device=device)
            torch.manual_seed(SEED)
            torch.cuda.manual_seed_all(SEED)
            enc = make_encoder(device)
            enc.load_state_dict(state, strict=True)
            enc.train()
            opt = torch.optim.AdamW(enc.parameters(), lr=8e-5, betas=(.9, .995), weight_decay=.05)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=5e-6)
            g = torch.Generator(device=device).manual_seed(SEED)
            log, epoch_orders = [], []
            updates = 0
            for epoch in range(EPOCHS):
                totals = np.zeros(2)
                order = []
                for ix in epoch_batches(ordered, g):
                    feature = enc.backbone(eeg[ix]).float()
                    token, auxiliary = peer_losses(proj(adapter(feature)), target[ix], center, variance,
                                                   feature, teacher[teacher_indices[ix]], peer_center)
                    opt.zero_grad(set_to_none=True)
                    (token + auxiliary).backward()
                    torch.nn.utils.clip_grad_norm_(enc.parameters(), 1., error_if_nonfinite=True)
                    opt.step()
                    sched.step()
                    totals += np.array([float(token.detach()), float(auxiliary.detach())]) * len(ix)
                    order.append(ix.detach())
                    updates += 1
                epoch_orders.append(array_hash(rows[torch.cat(order).cpu().numpy()].astype('<i8')))
                log.append(dict(zip(('token', 'peer'), (totals / len(rows)).tolist())))
                print(mode, subject, epoch + 1, log[-1], flush=True)
            require(updates == steps, 'Incomplete matched training schedule')
            pins.verify()
            file = dest / f'{mode}.pt'
            atomic_file(file, lambda f: torch.save({k: v.detach().cpu() for k, v in enc.state_dict().items()}, f))
            rec = dict(mode=mode, subject=subject, sources=proof, checkpoint_sha256=sha(file), seed=SEED, epochs=EPOCHS,
                       steps=updates, photos=len(rows), concepts=len(ordered), batch_size=BATCH, epoch_order_sha256=epoch_orders,
                       peer_cache_sha256=cached_sha, peer_subject_ids=peers, peer_mean_sha256=teacher_sha,
                       teacher_center_sha256=center_sha, teacher_center='Mean of this student\'s nine-peer OOF average over every registered train photo; subtracted from student and teacher in both arms',
                       peer_positive='own photograph' if mode == MODES[0] else 'fixed other photograph of same training object',
                       paired_photo_rows_sha256=array_hash(rows[pair].astype('<i8')), same_photo_rows=int(np.sum(rows[pair] == rows)),
                       same_concept_rows=int(np.sum(labels[pair] == labels)), distinct_teacher_rows=len(np.unique(pair)),
                       sibling_seed=SEED + 1, sibling_fixed_points=int(np.sum(permutation == np.arange(len(rows)))),
                       loss=log, weight=1., temperature=.07, token_gains=[1.] * 4, frozen_generator=True,
                       training_supervision='Original token-photo loss plus symmetric centered student-F / other-nine OOF-F contrast',
                       auxiliary_parameters=0, inference='personal BrainMLP -> frozen native adapter -> frozen token projection')
            save_json(file.with_suffix('.json'), rec)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--subject', type=int, choices=SUBJECTS)
    parser.add_argument('--modes', nargs='+', choices=MODES, default=MODES)
    parser.add_argument('--cache', type=Path, help='Training-only shared OOF cache; defaults to PREP/paper_peer_oof_training.pt')
    args = parser.parse_args()
    from common import PREP
    path = args.cache or PREP / CACHE
    if args.prepare:
        path.parent.mkdir(parents=True, exist_ok=True)
        prepare(PREP, path, torch.device('cuda'))
    else:
        fit(args.subject, args.modes, PREP, path)
