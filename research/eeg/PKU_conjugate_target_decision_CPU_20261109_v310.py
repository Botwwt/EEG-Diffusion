"""Full-T nested concept diagnostic: geometry benefit versus a generic output head.

No D/V/official data, new images, GPU reservation, or claim of image improvement.
Every T query is evaluated once in a held-out readout concept fold, using its
original native encoder role that never fitted that concept.
"""
import csv
import gzip
import hashlib
import json
import os
import socket
import shutil
import time
import traceback
from pathlib import Path

assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname().lower()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
R = Path('/dfs/share/maleiLab/wwt/eeg')
OUT = R / 'runs/EEG_conjugate_target_decision_v310'
OUT.mkdir(exist_ok=False)
start = time.monotonic()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(name, value):
    data = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    p = OUT / name
    tmp = p.with_name(p.name + '.pending')
    with tmp.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(p)


try:
    import numpy as np
    import torch
    from PKU_conjugate_coordinates_20261109_v310 import ConjugateChart, OrdinaryCapacityControl

    torch.set_num_threads(4)
    torch.manual_seed(1729)
    spec_path = R / 'code/EEG_conjugate_target_decision_v310.json'
    spec = json.loads(spec_path.read_text())
    assert spec['meta_steps'] == 64 and spec['outer_folds_per_encoder_role'] == 3
    assert shutil.disk_usage(OUT).free > 1 << 30
    # Algebra/gradients checked on a small dimension; this is an engineering check.
    probe = ConjugateChart(dimension=8, hidden=4, blocks=2).double()
    with torch.no_grad():
        for layer in probe.layers:
            layer.last.weight.normal_(0, .05)
            layer.last.bias.normal_(0, .05)
    point = torch.randn(8, dtype=torch.float64, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(probe, point)
    inverse_error = float((probe.inverse(probe(point)) - point).abs().max())
    norm_error = float((probe(point).square().sum() - point.square().sum()).abs())
    determinant = float(torch.linalg.det(jacobian))
    assert inverse_error < 1e-11 and norm_error < 1e-11 and abs(determinant - 1) < 1e-10
    algebra = dict(inverse_max_abs=inverse_error, squared_norm_error=norm_error, Jacobian_determinant=determinant)
    parent = R / 'runs/HIVE_native_sub03_5096542'
    bank_path = parent / 'T_true_OOF_conditions.pt'
    bank = torch.load(bank_path, map_location='cpu', weights_only=True)
    split_path = R / 'data/recovery_20261108_v236/current_split_manifest.json'
    split = json.loads(split_path.read_text())
    rows = bank['array_rows'].tolist()
    assert rows == split['image_rows']['T'] and len(rows) == 13230
    x_all = bank['condition'].float()
    source_role = bank['folds'].long()
    assert set(source_role.tolist()) == {0, 1} and x_all.shape == (13230, 1024)
    visual_path = parent / 'deterministic_F_all16540.pt'
    y_all = torch.load(visual_path, map_location='cpu', weights_only=True)[rows].float()
    assert y_all.shape == x_all.shape
    with (R / 'data/full_T_recovery_20261108_v238/image_manifest.csv').open() as f:
        lookup = {int(v['array_row']): v for v in csv.DictReader(f) if v['dataset_split'] == 'train'}
    concepts = [lookup[r]['concept_id'] for r in rows]
    assert len(set(concepts)) == 1323
    concepts_array = np.asarray(concepts)
    assignment = torch.full((13230,), -1, dtype=torch.long)
    for role in [0, 1]:
        keys = sorted({concepts[i] for i in range(13230) if source_role[i] == role}, key=lambda k: hashlib.sha256(('1729|' + k).encode()).hexdigest())
        for i, key in enumerate(keys):
            selected = [j for j, c in enumerate(concepts) if c == key]
            assert len(selected) == 10 and all(int(source_role[j]) == role for j in selected)
            assignment[selected] = i % 3
    assert (assignment >= 0).all()
    cases = ['identity_ridge', 'orthogonal_equivalent', 'conjugate_chart', 'generic_output_head']
    values = torch.zeros(4, 13230, 4, dtype=torch.float64)
    written = torch.zeros(13230, dtype=torch.bool)
    target_unit = torch.nn.functional.normalize(y_all.double(), dim=-1)
    fit_receipts = []

    def features(x, mu, scale):
        x = (x - mu) / scale
        return torch.cat([x, torch.ones(len(x), 1)], dim=-1)

    def ridge_operator(x):
        n = len(x)
        penalty = torch.eye(x.shape[1]) * (.01 * n)
        penalty[-1, -1] = 0
        return torch.linalg.solve(x.T @ x + penalty, x.T)

    for role in [0, 1]:
        for fold in range(3):
            assert time.monotonic() - start < 1080, 'Finite diagnostic wall limit; retain prefix'
            train = torch.where((source_role == role) & (assignment != fold))[0]
            test = torch.where((source_role == role) & (assignment == fold))[0]
            train_keys = sorted({concepts[i] for i in train.tolist()}, key=lambda k: hashlib.sha256(('inner|1729|' + k).encode()).hexdigest())
            a_keys = set(train_keys[::2])
            inner_a = torch.tensor([i for i in train.tolist() if concepts[i] in a_keys])
            inner_b = torch.tensor([i for i in train.tolist() if concepts[i] not in a_keys])
            assert set(concepts[i] for i in test.tolist()).isdisjoint(train_keys)
            assert len(train) + len(test) == int((source_role == role).sum())
            # All normalization and meta supervision stay inside the outer fit concepts.
            mu_x = x_all[train].mean(0)
            scale_x = x_all[train].std(0).clamp_min(.01)
            mu_y = y_all[train].mean(0)
            scale_y = (y_all[train] - mu_y).square().mean().sqrt().clamp_min(.01)
            xa, xb, xt, xv = [features(x_all[index], mu_x, scale_x) for index in [inner_a, inner_b, train, test]]
            ya, yb, yt = [(y_all[index] - mu_y) / scale_y for index in [inner_a, inner_b, train]]
            operator_a = ridge_operator(xa)
            operator_t = ridge_operator(xt)
            transfer = xb @ operator_a
            base_b = transfer @ ya
            base_v = xv @ (operator_t @ yt)
            torch.manual_seed(1729)
            chart = ConjugateChart()
            head = OrdinaryCapacityControl()
            assert sum(v.numel() for v in chart.parameters()) == 197888
            assert sum(v.numel() for v in head.parameters()) == 197728
            assert torch.equal(chart(ya[:10]), ya[:10]) and torch.equal(head(base_b[:10]), base_b[:10])
            optim_chart = torch.optim.AdamW(chart.parameters(), lr=.001, weight_decay=.01)
            optim_head = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.01)
            history = []
            for step in range(64):
                assert time.monotonic() - start < 1080, 'Finite diagnostic wall limit; retain prefix'
                optim_chart.zero_grad(set_to_none=True)
                predicted_b = chart.inverse(transfer @ chart(ya))
                loss_chart = (predicted_b - yb).square().mean()
                loss_chart.backward()
                torch.nn.utils.clip_grad_norm_(chart.parameters(), 1., error_if_nonfinite=True)
                optim_chart.step()
                optim_head.zero_grad(set_to_none=True)
                loss_head = (head(base_b) - yb).square().mean()
                loss_head.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                optim_head.step()
                event = dict(role=role, fold=fold, step=step + 1, conjugate_original_space_meta_MSE=float(loss_chart.detach()), generic_original_space_meta_MSE=float(loss_head.detach()))
                history.append(event)
                if (step + 1) % 16 == 0:
                    write('progress.json', dict(**event, seconds=time.monotonic() - start, image_effect=False))
                    print(json.dumps(event), flush=True)
            with torch.no_grad():
                cv = chart.inverse(xv @ (operator_t @ chart(yt)))
                hv = head(base_v)
                # An orthogonal coordinate change cancels exactly for isotropic ridge.
                permutation = torch.arange(1024).roll(127)
                inverse_permutation = torch.argsort(permutation)
                ov = (xv @ (operator_t @ yt[:, permutation]))[:, inverse_permutation]
                assert torch.allclose(ov, base_v, atol=2e-5, rtol=2e-5)
                teacher_inverse = chart.inverse(chart(yt))
                reconstruction_error = float((teacher_inverse - yt).abs().max())
                normalized_error = float((chart(yt).norm(dim=-1) - yt.norm(dim=-1)).abs().max())
                assert reconstruction_error < 1e-4 and normalized_error < 1e-4
                predictions = [mu_y + scale_y * p for p in [base_v, ov, cv, hv]]
                for ai, predicted in enumerate(predictions):
                    mse = ((predicted.double() - y_all[test].double()) ** 2).mean(-1)
                    unit = torch.nn.functional.normalize(predicted.double(), dim=-1)
                    cosine = (unit * target_unit[test]).sum(-1)
                    scores = unit @ target_unit.T
                    correct = scores[torch.arange(len(test)), test]
                    same = torch.tensor(concepts_array[test.numpy(), None] == concepts_array[None, :])
                    same[torch.arange(len(test)), test] = False
                    wrong_max = scores.masked_fill(~same, -float('inf')).max(-1).values
                    margin = correct - wrong_max
                    top1 = (margin > 0).double()
                    values[ai, test] = torch.stack([mse, cosine, margin, top1], dim=-1)
                assert not written[test].any()
                written[test] = True
            model_path = OUT / ('chart_role' + str(role) + '_fold' + str(fold) + '.pt')
            torch.save(dict(chart=chart.state_dict(), head=head.state_dict(), mu_x=mu_x, scale_x=scale_x, mu_y=mu_y, scale_y=scale_y, train_rows=[rows[i] for i in train.tolist()], test_rows=[rows[i] for i in test.tolist()]), model_path)
            receipt = dict(role=role, fold=fold, fit_queries=len(train), test_queries=len(test), inner_fit_queries=len(inner_a), inner_meta_queries=len(inner_b), meta_steps=64, history=history, full_teacher_inverse_error=reconstruction_error, norm_error=normalized_error, model_sha256=sha(model_path), seconds=time.monotonic() - start)
            fit_receipts.append(receipt)
            write('fit_role' + str(role) + '_fold' + str(fold) + '.json', receipt)
    assert written.all() and len(fit_receipts) == 6 and torch.isfinite(values).all()
    metric_names = ['original_F_MSE', 'original_F_cosine', 'same_concept_margin', 'same_concept_strict_Top1']
    with gzip.open(OUT / 'all13230_4controls.csv.gz', 'xt', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['case', 'array_row', 'concept_id', 'source_encoder_role', 'outer_fold'] + metric_names)
        writer.writeheader()
        for ai, case in enumerate(cases):
            for i in range(13230):
                writer.writerow(dict(case=case, array_row=rows[i], concept_id=concepts[i], source_encoder_role=int(source_role[i]), outer_fold=int(assignment[i]), **{name:float(values[ai, i, mi]) for mi, name in enumerate(metric_names)}))
    comparisons = []
    rng = np.random.default_rng(1729)
    keys = sorted(set(concepts))
    samples = rng.integers(0, len(keys), size=(1000, len(keys)))
    for left, right in [(2, 0), (2, 3), (3, 0), (1, 0)]:
        for mi, metric in enumerate(metric_names):
            difference = (values[left, :, mi] - values[right, :, mi]).numpy()
            per = np.asarray([difference[concepts_array == k].mean() for k in keys])
            ci = np.quantile(per[samples].mean(1), [.025, .975])
            comparisons.append(dict(left=cases[left], right=cases[right], metric=metric, mean=float(difference.mean()), concept95=[float(v) for v in ci]))
    completion = dict(status='COMPLETE_ALL13230_SIX_CONCEPT_ISOLATED_FITS_FOUR_READOUT_CONTROLS_NO_IMAGES', job_id=os.environ['SLURM_JOB_ID'], seconds=time.monotonic() - start, queries=13230, concepts=1323, subjects=1, repeat='R4', source_EEG_encoder_frozen=True,
        cases=cases, means={case:{name:float(values[ai, :, mi].mean()) for mi, name in enumerate(metric_names)} for ai, case in enumerate(cases)}, comparisons=comparisons,
        algebra=algebra, fits=6, added_parameters=dict(conjugate=197888, generic_head=197728), independent_unexposed_H=False,
        claims=dict(image_improvement=False, method_family_rejected=False, novelty_established=False, CCFA=False),
        interpretation='A fixed-budget diagnostic of learnable visual coordinates against a generic nonlinear head. Meta updates use outer-fit concepts only. Complete row coverage does not establish optimizer convergence. Original F scores are not image quality or evidence of extra EEG information. Exact conjugacy preserves visual teacher function algebraically, not all empirical feature distributions.',
        input_sha256=dict(OOF=sha(bank_path), F=sha(visual_path), split=sha(split_path), policy=sha(spec_path), core=sha(R / 'code/PKU_conjugate_coordinates_20261109_v310.py')),
        table_sha256=sha(OUT / 'all13230_4controls.csv.gz'), source_sha256=sha(__file__), new_GPU_jobs=0, new_images=0)
    write('completion.json', completion)
    assert sum(p.stat().st_size for p in OUT.iterdir() if p.is_file()) < 64 << 20
    print(json.dumps(dict(status=completion['status'], means=completion['means'], seconds=completion['seconds'])), flush=True)
except BaseException as error:
    write('failure.json', dict(status='FAILED_PRESERVED_NO_AUTOMATIC_RETRY', error=repr(error), traceback=traceback.format_exc(), seconds=time.monotonic() - start))
    raise
