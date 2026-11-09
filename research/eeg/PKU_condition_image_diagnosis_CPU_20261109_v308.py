"""Read-only complete-panel diagnosis; no fitting, model selection or GPU use."""
import csv
import gzip
import hashlib
import json
import os
import socket
import time
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path('/dfs/share/maleiLab/wwt/eeg')
OUT = R / 'runs/EEG_condition_image_diagnosis_v308'
OUT.mkdir(exist_ok=False)
started = time.monotonic()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    tmp = path.with_name(path.name + '.pending')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


try:
    import numpy as np
    import torch

    torch.set_num_threads(4)
    root = R / 'runs/EEG_anchor_scores_v288'
    completed = json.loads((root / 'analysis_completion.json').read_text())
    assert completed['status'] == 'COMPLETE_ALL830_20ARMS_ANCHORED_EXTENSION_DINO_PAPER7_EFFNET_SEED_PENDING'
    evidence = []

    def verified(path, expected):
        actual = sha(path)
        assert actual == expected, str(path)
        evidence.append(dict(path=str(path.relative_to(R)), sha256=actual))

    verified(root / 'all830_all20arms_all14metrics.npz', completed['all_values_sha256'])
    verified(root / 'all_paired_summaries.json', completed['summary_sha256'])
    gallery = json.loads((root / 'gallery.json').read_text())['rows']
    assert len(gallery) == 830
    rows = [int(v['array_row']) for v in gallery]
    ids = [v['stimulus_id'] for v in gallery]
    concepts = np.asarray([v['concept_id'] for v in gallery])
    keys = sorted(set(concepts.tolist()))
    assert len(keys) == 83 and all(np.sum(concepts == k) == 10 for k in keys)
    same = concepts[:, None] == concepts[None, :]
    negative_same = same & ~np.eye(830, dtype=bool)
    banks = []
    arms = []
    for folder, n in [('EEG_images_v271', 14), ('EEG_anchor_images_v288', 6)]:
        p = R / 'runs' / folder / 'condition_bank.pt'
        proof = json.loads(p.with_name('conditions_completion.json').read_text())
        verified(p, proof['condition_bank_sha256'])
        bank = torch.load(p, map_location='cpu', weights_only=True)
        assert list(bank['array_rows']) == rows and bank['stimulus_ids'] == ids
        assert bank['condition'].shape == (n, 830, 1024)
        banks.append(bank['condition'].double().numpy())
        arms.extend(bank['arms'])
    condition = np.concatenate(banks, axis=0)
    cube = np.load(root / 'all830_all20arms_all14metrics.npz', allow_pickle=False)
    assert cube['arms'].tolist() == arms and cube['array_rows'].tolist() == rows
    assert cube['concept_ids'].tolist() == concepts.tolist()
    image_metrics = cube['metrics'].tolist()
    image_values = cube['values']
    assert image_values.shape == (20, 14, 830) and np.isfinite(image_values).all()
    visual_path = R / 'runs/HIVE_native_sub03_5096542/deterministic_F_all16540.pt'
    evidence.append(dict(path=str(visual_path.relative_to(R)), sha256=sha(visual_path)))
    target = torch.load(visual_path, map_location='cpu', weights_only=True)[rows].double().numpy()
    assert target.shape == (830, 1024) and np.isfinite(condition).all() and np.isfinite(target).all()
    target_norm = np.linalg.norm(target, axis=1)
    assert np.all(target_norm > 0)
    target_unit = target / target_norm[:, None]
    geometries = []
    geometry_names = ['F_correct_cosine', 'F_mse', 'F_sameconcept_margin', 'F_sameconcept_top1', 'condition_norm', 'condition_common_cosine']
    for c in condition:
        norm = np.linalg.norm(c, axis=1)
        assert np.all(norm > 0)
        scores = (c / norm[:, None]) @ target_unit.T
        correct = np.diag(scores)
        wrong_max = np.where(negative_same, scores, -np.inf).max(axis=1)
        # Report strict wins and ties separately; no hidden argmax tie-break.
        top1 = (correct > wrong_max).astype(float)
        common = c.mean(axis=0)
        common_norm = np.linalg.norm(common)
        assert common_norm > 0
        common_cosine = (c @ common) / (norm * common_norm)
        geometries.append(np.stack([correct, ((c - target) ** 2).mean(axis=1), correct - wrong_max, top1, norm, common_cosine]))
    geometry = np.stack(geometries)
    assert geometry.shape == (20, 6, 830) and np.isfinite(geometry).all()
    with gzip.open(OUT / 'all830_condition_and_image.csv.gz', 'xt', newline='') as f:
        names = ['arm', 'array_row', 'stimulus_id', 'concept_id'] + geometry_names + image_metrics
        writer = csv.DictWriter(f, fieldnames=names)
        writer.writeheader()
        for ai, arm in enumerate(arms):
            for qi in range(830):
                row = dict(arm=arm, array_row=rows[qi], stimulus_id=ids[qi], concept_id=str(concepts[qi]))
                row.update({n: float(geometry[ai, gi, qi]) for gi, n in enumerate(geometry_names)})
                row.update({n: float(image_values[ai, mi, qi]) for mi, n in enumerate(image_metrics)})
                writer.writerow(row)

    def correlation(x, y):
        x = x - x.mean()
        y = y - y.mean()
        denominator = np.linalg.norm(x) * np.linalg.norm(y)
        return float(x @ y / denominator) if denominator > 1e-20 else None

    def concept_center(x):
        y = x.copy()
        for key in keys:
            y[concepts == key] -= y[concepts == key].mean()
        return y

    comparisons = []
    for ai, arm in enumerate(arms):
        for reference in ['parent', 'matched_ordinary']:
            if arm == reference:
                continue
            ri = arms.index(reference)
            dc = geometry[ai, 0] - geometry[ri, 0]
            dm = geometry[ai, 2] - geometry[ri, 2]
            dd = image_values[ai, image_metrics.index('DINO2AFC')] - image_values[ri, image_metrics.index('DINO2AFC')]
            ds = image_values[ai, image_metrics.index('SSIM425')] - image_values[ri, image_metrics.index('SSIM425')]
            comparisons.append(dict(
                arm=arm, reference=reference,
                mean_F_correct_cosine_change=float(dc.mean()),
                mean_F_sameconcept_margin_change=float(dm.mean()),
                mean_DINO2AFC_change=float(dd.mean()), mean_SSIM_change=float(ds.mean()),
                Pearson_Fcosine_change_vs_DINO_change=correlation(dc, dd),
                within_concept_Pearson_Fcosine_change_vs_DINO_change=correlation(concept_center(dc), concept_center(dd)),
                within_concept_Pearson_Fmargin_change_vs_DINO_change=correlation(concept_center(dm), concept_center(dd)),
                counts=dict(Fcosine_up_DINO_up=int(((dc > 0) & (dd > 0)).sum()),
                            Fcosine_up_DINO_down=int(((dc > 0) & (dd < 0)).sum()),
                            Fcosine_down_DINO_up=int(((dc < 0) & (dd > 0)).sum()),
                            Fcosine_down_DINO_down=int(((dc < 0) & (dd < 0)).sum()),
                            any_exact_tie=int(((dc == 0) | (dd == 0)).sum()))))
    training = []
    cases = [('EEG_candidates_v267', c) for c in ['A_positive', 'A_pair', 'A_reverse', 'B_residual', 'B_raw', 'B_shuffle']]
    cases += [('EEG_anchor_v286', c) for c in ['C_positive', 'C_pair', 'C_reverse']]
    for folder, case in cases:
        path = R / 'runs' / folder / case
        assert (path / 'completion.json').exists()
        history = {}
        logs = sorted(path.glob('attempt_*.jsonl'))
        assert logs
        for log in logs:
            evidence.append(dict(path=str(log.relative_to(R)), sha256=sha(log)))
            for line in log.read_text().splitlines():
                event = json.loads(line)
                assert event['case'] == case and event['update'] not in history
                history[event['update']] = event
        assert sorted(history) == list(range(1, 828))
        events = [history[i] for i in range(1, 828)]
        assert sum(v['n'] for v in events) == 13230
        assert all(v['offset'] == 16 * (v['update'] - 1) for v in events)
        block_summaries = []
        available = [k for k in ['loss', 'positive_energy', 'negative_energy', 'token_error', 'visual_replay', 'contrastive', 'grad_norm'] if k in events[0]]
        for lo in range(0, 827, 128):
            block = events[lo:lo + 128]
            weight = sum(v['n'] for v in block)
            block_summaries.append(dict(first_update=lo + 1, last_update=lo + len(block), query_exposures=weight,
                **{k: sum(v[k] * v['n'] for v in block) / weight for k in available}))
        training.append(dict(case=case, updates=827, query_exposures=13230, complete_T_passes=1,
            online_blocks=block_summaries,
            heldout_learning_curve_available=False, convergence_established=False,
            note='Changing noise/t/query composition; online block trends cannot prove convergence, overfitting or a sole failure cause.'))
    result = dict(status='COMPLETE_ALL830_20ARMS_CONDITION_IMAGE_AND_ALL9_TRAINING_LOG_DIAGNOSIS',
        job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic() - started,
        scope=dict(subjects=1, repeat='R4', concepts=83, queries=830, arms=20, complete_training_logs=9),
        geometry_means={arm: {n: float(geometry[ai, gi].mean()) for gi, n in enumerate(geometry_names)} for ai, arm in enumerate(arms)},
        comparisons=comparisons, training=training, input_evidence=evidence,
        per_query_sha256=sha(OUT / 'all830_condition_and_image.csv.gz'),
        claims=dict(second_innovation_confirmed=False, causal_bottleneck_identified=False, converged_fit=False,
                    cross_architecture=False, SOTA=False, whole_idea_rejected=False),
        interpretation='Posthoc descriptive localization in native F coordinates. F geometry is not an independent evaluator or information measure. Mean-direction effects and correlations are not causal. Complete one-pass protocol is not complete method qualification. No fitting, no threshold tuning, no selected examples or automatic next training.',
        new_training_updates=0, new_images=0, GPU_used=False, source_sha256=sha(__file__))
    write(OUT / 'completion.json', result)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < 6 << 20
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'], output_bytes=sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()))))
except BaseException as error:
    import traceback
    write(OUT / 'failure.json', dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error=repr(error), traceback=traceback.format_exc()))
    raise
