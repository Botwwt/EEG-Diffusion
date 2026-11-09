"""Read-only D30 posthoc dynamics/parameter/function diagnostics; never fit.

metadata mode needs only copied fit completion, epoch journal and OOF summary.
checkpoints mode additionally reads the real nine final checkpoints, original
role checkpoints, T-only physical trials/targets and the frozen native OOF bank.
Outputs are new small JSON files; no old source, checkpoint or pin is modified.
All13230 T stimuli are replayed for each role, with FIT/held reported separately.
The four old/new-core by identity/front functions and their exact interaction
are descriptive diagnostics and never admit new images.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parent
ARMS = ('native_continuation', 'serial_reversible', 'parallel_matched')
ROLES = ('full', 'oof0', 'oof1')
torch = None


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def checked(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], path
    return path


def write(path, value):
    raw = (json.dumps(value, indent=2, allow_nan=False) + '\n').encode()
    assert len(raw) < 16 << 20, 'Small diagnostic outputs only'
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())


def journal(completion_path):
    path = completion_path.parent / 'training_log.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 225
    result = []
    for role in ROLES:
        curves = {}
        for arm in ARMS:
            selected = sorted((row for row in rows if row['role'] == role and row['arm'] == arm), key=lambda row: row['epoch'])
            assert [row['epoch'] for row in selected] == list(range(1, 26))
            loss = np.asarray([row['mean_InfoNCE'] for row in selected], np.float64)
            gradient = np.asarray([row['grad_norm_last'] for row in selected], np.float64)
            assert np.isfinite(loss).all() and np.isfinite(gradient).all()
            curves[arm] = selected
            result.append(dict(role=role, arm=arm, loss_by_epoch=loss.tolist(),
                epoch1=float(loss[0]), epoch25=float(loss[-1]), minimum_loss_epoch=int(loss.argmin() + 1),
                nonincreasing_transitions=int((np.diff(loss) <= 0).sum()),
                last5_linear_slope=float(np.polyfit(np.arange(5), loss[-5:], 1)[0]),
                last_minibatch_preclip_gradient_norm_range=[float(gradient.min()), float(gradient.max())],
                total_gradient_not_front_core_split=True))
        for epoch in range(25):
            assert all(curves[arm][epoch]['order_sha256'] == curves[ARMS[0]][epoch]['order_sha256'] and
                curves[arm][epoch]['CUDA_dropout_stream_sha256'] == curves[ARMS[0]][epoch]['CUDA_dropout_stream_sha256']
                for arm in ARMS)
    return dict(source=ref(path), curves=result, all225_epochs_present=True,
        all_role_order_dropout_streams_match=True, front_core_separate_gradients_logged=False,
        logit_scale_trajectory_logged=False, per_epoch_held_curve_logged=False,
        inference='All nine training losses decrease through epoch25; this does not locate onset of held deterioration or attribute the loss drop to front/core/temperature')


def metadata(completion_path, summary_path):
    done = read(completion_path); summary = read(summary_path)
    assert done['status'] == 'D30_ALL9_MATCHED_ADDITIONAL25_EPOCH_T_FITS_AND_TRUE_OOF_COMPLETE'
    assert summary['status'] == 'D30_COMPLETE_OOF_NATIVE_AND_FROZEN_ORDINARY_CPU_READOUT'
    assert summary['completion']['sha256'] == sha(completion_path)
    assert summary['training_log']['sha256'] == sha(completion_path.parent / 'training_log.jsonl')
    return dict(status='D30_EXISTING_METADATA_DYNAMICS_READOUT_COMPLETE',
        completion=ref(completion_path), OOF_summary=ref(summary_path), training=journal(completion_path),
        axes=summary['axes'], original25epoch_frozen_references=summary['original25epoch_frozen_references'],
        final50epoch_metrics=summary['metrics'], paired=summary['paired'],
        final_vs_original=summary['versus_same_ordinary_state_original25epoch_references'],
        output_mean_energy_fraction=summary['condition_overall_mean_energy_fraction'],
        reported_front_facts=summary['trained_serial_front_synthetic_inverse_checks'],
        what_is_supported=['Final OOF condition metrics are worse than both original25epoch and matched native continuation for this recipe',
            'Serial and parallel have close metric means; that is not a measurement of pointwise output closeness',
            'All reported serial fc2 entries changed from zero, but weight nonzero counts do not measure real EEG displacement',
            'Training FIT contrastive objective and final held-concept metrics disagree; generic generalization/objective mismatch is supported as a description'],
        what_is_not_supported=['No per-epoch held curve: no identified late-overfitting epoch or effective intermediate checkpoint',
            'No current real-input front displacement/core bypass data: neither near-identity nor large shared transformation is proved',
            'Total gradient and learned logit-scale prevent attributing training-loss improvements to the front alone',
            'Original25epochs used stochastic visual-teacher dropout; new25epochs use fixed cached F and restarted AdamW/cosine, so the continuation comparison changes training regime',
            'D30 native mean-energy fraction about0.00031 does not resemble D08 output mean dominance about0.915',
            'No image effect or causal neural recovery is established'],
        routing_identity='For one stage, serial-minus-parallel right output with the SAME weights is g(x1+f(x2))-g(x1); near zero heads its leading interaction is Jg*f. Small serial/parallel differences need not imply the common front is near identity.',
        actual_checkpoint_weights_read=False, actual_raw_EEG_values_read=False,
        model_forwards=0, GPU_initialized=False, training_updates=0, new_images=0)


def values_stats(value):
    value = value.detach().double()
    norms = value.norm(dim=-1)
    return dict(RMS=float(value.square().mean().sqrt()), global_L2=float(value.norm()),
        row_L2_median=float(norms.median()), row_L2_95=float(torch.quantile(norms, .95)),
        maximum_abs=float(value.abs().max()))


def displacement(value, base):
    delta = value.double() - base.double()
    denominator = base.double().norm().clamp_min(1e-30)
    result = values_stats(delta)
    result['relative_global_L2'] = float(delta.norm() / denominator)
    result['cosine_mean'] = float(torch.nn.functional.cosine_similarity(value.double(), base.double()).mean())
    return result


def delta_pair(left, right, base):
    a, b = left.double() - base.double(), right.double() - base.double()
    shared, difference = (a + b) / 2, (a - b) / 2
    ea, eb, ec = float(a.square().sum()), float(b.square().sum()), float((a * b).sum())
    energy_shared, energy_specific = float(shared.square().sum()), float(difference.square().sum())
    return dict(left_change_L2=math.sqrt(ea), right_change_L2=math.sqrt(eb),
        change_cosine=ec / math.sqrt(ea * eb) if ea > 0 and eb > 0 else None,
        common_average_energy=energy_shared, difference_half_energy=energy_specific,
        difference_fraction_of_change_energy=energy_specific / (energy_shared + energy_specific)
            if energy_shared + energy_specific > 0 else None,
        meaning='Algebraic common-average/difference decomposition, not an identified shared causal mechanism')


class DifferenceMoments:
    def __init__(self):
        self.energy = self.base_energy = self.cosine = 0.
        self.maximum = 0.; self.rows = self.coordinates = 0; self.norms = []

    def add(self, value, base):
        value, base = value.double(), base.double(); delta = value - base
        self.energy += float(delta.square().sum()); self.base_energy += float(base.square().sum())
        self.cosine += float(torch.nn.functional.cosine_similarity(value, base, dim=-1).sum())
        self.maximum = max(self.maximum, float(delta.abs().max()))
        self.norms.append(delta.norm(dim=-1).numpy()); self.rows += len(value); self.coordinates += delta.numel()

    def result(self):
        norms = np.concatenate(self.norms)
        return dict(rows=self.rows, RMS=math.sqrt(self.energy / self.coordinates),
            global_L2=math.sqrt(self.energy), relative_global_L2=math.sqrt(self.energy / self.base_energy)
                if self.base_energy else None,
            row_delta_L2_median=float(np.median(norms)), row_delta_L2_95=float(np.quantile(norms, .95)),
            maximum_abs=self.maximum, cosine_mean=self.cosine / self.rows)


class SharedChanges:
    def __init__(self):
        self.a = self.b = self.dot = self.shared = self.specific = 0.; self.rows = 0

    def add(self, left, right, base):
        a, b = left.double() - base.double(), right.double() - base.double()
        self.a += float(a.square().sum()); self.b += float(b.square().sum()); self.dot += float((a * b).sum())
        self.shared += float(((a + b) / 2).square().sum()); self.specific += float(((a - b) / 2).square().sum())
        self.rows += len(base)

    def result(self):
        return dict(rows=self.rows, serial_change_L2=math.sqrt(self.a), parallel_change_L2=math.sqrt(self.b),
            change_cosine=self.dot / math.sqrt(self.a * self.b) if self.a and self.b else None,
            common_average_energy=self.shared, difference_half_energy=self.specific,
            difference_fraction_of_change_energy=self.specific / (self.shared + self.specific)
                if self.shared + self.specific else None,
            algebraic_decomposition_not_causal_mechanism=True)


def weight_stats(current, initial):
    answer = {}
    groups = dict(all=list(current), core_projection=[key for key in current if key.startswith('core.proj_in.')],
        core_residual=[key for key in current if key.startswith('core.block.') and '.norm.' not in key],
        output_affine=[key for key in current if key.startswith('core.') and '.norm.' in key],
        front_fc1=[key for key in current if key.startswith('front.') and '.fc1.' in key],
        front_fc2=[key for key in current if key.startswith('front.') and '.fc2.' in key])
    for group, keys in groups.items():
        if not keys:
            continue
        energy = delta_energy = 0.; changed = count = 0; maximum = 0.
        for key in keys:
            a, b = current[key].reshape(-1), initial[key].reshape(-1)
            for lo in range(0, a.numel(), 1 << 20):
                x, y = a[lo:lo + (1 << 20)].double(), b[lo:lo + (1 << 20)].double()
                delta = x - y; energy += float(y.square().sum()); delta_energy += float(delta.square().sum())
                changed += int((delta != 0).sum()); count += delta.numel(); maximum = max(maximum, float(delta.abs().max()))
        answer[group] = dict(tensors=len(keys), parameters=count, changed_values=changed,
            initial_L2=math.sqrt(energy), change_L2=math.sqrt(delta_energy),
            relative_L2=math.sqrt(delta_energy / energy) if energy else None, max_abs_change=maximum)
    return answer


def front_route(front, values, serial):
    left, right = front.split(values)
    for stage in front.stages:
        next_left = left + stage.f(right)
        right = right + stage.g(next_left if serial else left)
        left = next_left
    return front.merge(left, right)


def quality(prediction, target, index, rows, logit_scale, old_scale, tick):
    assert prediction.shape == target.shape == (len(rows), 1024)
    a = torch.nn.functional.normalize(prediction, dim=-1)
    b = torch.nn.functional.normalize(target, dim=-1)
    groups = {}
    for local, row in enumerate(rows):
        groups.setdefault(index[row]['concept_id'], []).append(local)
    hits = afc = 0.
    for positions in groups.values():
        assert len(positions) == 10
        scores = a[positions] @ b[positions].T
        correct = scores.diagonal()
        hits += float((scores.argmax(1) == torch.arange(10)).sum())
        afc += float(((scores < correct[:, None]).sum(1) +
            .5 * ((scores == correct[:, None]).sum(1) - 1)).sum() / 9)
    losses = [0., 0.]
    for lo in range(0, len(rows), 1024):
        tick(); similarity = a[lo:lo + 1024] @ b[lo:lo + 1024].T
        labels = torch.arange(len(similarity))
        for i, scale in enumerate((logit_scale, old_scale)):
            logits = similarity * scale.exp()
            losses[i] += float(.5 * (torch.nn.functional.cross_entropy(logits, labels) +
                torch.nn.functional.cross_entropy(logits.T, labels))) * len(labels)
    return dict(queries=len(rows), concepts=len(groups), F_MSE=float((prediction - target).double().square().mean()),
        F_cosine=float((a * b).sum(-1).double().mean()), same_concept10_Top1=hits / len(rows),
        same_concept10_2AFC=afc / len(rows), fixed_order_eval_InfoNCE_own_scale=losses[0] / len(rows),
        fixed_order_eval_InfoNCE_common_old_scale=losses[1] / len(rows), eval_contrastive_batch=1024,
        dropout_off=True, not_comparable_to_logged_training_minibatch_loss=True)


def checkpoint_diagnostics(completion_path, summary_path, freeze_path, tick, output):
    global torch
    import torch as actual_torch
    torch = actual_torch; torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    done = read(completion_path); spec = read(checked(done['spec']))
    checked(done['source']); checked(done['model_source'])
    assert sha(ROOT / 'brainhive_reversible_eeg_20261028_v1.py') == done['model_source']['sha256']
    import brainhive_reversible_eeg_20261028_v1 as V
    core_class = V.author_brain_class(checked(spec['author_models_source']), checked(spec['author_layers_source']))
    physical = read(checked(spec['physical_R4_input_contract']))
    index = [json.loads(line) for line in checked(spec.get('T_index', physical['files']['index.jsonl'])).read_text().splitlines()]
    trials = np.load(checked(spec.get('T_trials', physical['files']['trials.npy'])), mmap_mode='r', allow_pickle=False)
    targets = np.load(checked(spec['T_targets']), mmap_mode='r', allow_pickle=False)
    bank = np.load(checked(done['OOF_T_native']), mmap_mode='r', allow_pickle=False)
    assert trials.shape == (13230, 4, 17, 250) and trials.dtype == np.float32
    assert targets.shape == (13230, 1024) and targets.dtype == np.float32
    assert bank.shape == (13230, 3, 1024) and bank.dtype == np.float32
    assert len(index) == 13230 and all(row['partition'] == 'T' for row in index)
    freeze = read(freeze_path)
    assert freeze['checkpoint']['sha256'] == 'a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03'
    assert freeze['selected_strength'] == 1.
    checked(freeze['source'])
    assert sha(ROOT / 'brainhive_bounded_component_v1.py') == freeze['source']['sha256']
    from brainhive_bounded_component_v1 import BoundedConditionAdapter
    ordinary_payload = torch.load(checked(freeze['checkpoint']), weights_only=False, map_location='cpu')
    ordinary = BoundedConditionAdapter(predict_risk=False).eval().requires_grad_(False)
    ordinary.load_state_dict(ordinary_payload['model'], strict=True)
    assert ordinary_payload['calibration']['selected_strength'] == 1.
    role_rows = read(checked(done['role_rows']))
    records = {(item['role'], item['arm']): item for item in done['checkpoints']}
    assert len(records) == 9
    result = dict(status='D30_ALL13230_T_REAL_CHECKPOINT_AND_FACTORIAL_FUNCTION_DIAGNOSTICS',
        completion=ref(completion_path), models=[], buckets=[], source=ref(__file__), ordinary_freeze=ref(freeze_path),
        diagnostic_scope='All13230 T stimuli for every role; original role FIT/held buckets remain distinct',
        read_only=True, CPU_FP32_eval=True, no_GPU_cache_bitwise_equality_claim=True,
        no_checkpoint_or_hyperparameter_selection=True, no_large_weight_or_prediction_export=True,
        runtime=dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
            torch_source=str(Path(torch.__file__).resolve()), numpy=np.__version__,
            CPU_threads=torch.get_num_threads(), CUDA_visible_devices=os.environ['CUDA_VISIBLE_DEVICES']))
    parity_maximum = parity_relative = 0.; parity_rows = 0
    for role in ROLES:
        tick(); original_path = checked(spec['roles'][role]['checkpoint'])
        expected_held = [] if role == 'full' else [i for i, row in enumerate(index) if row['fold'] == int(role[-1])]
        expected_fit = list(range(13230)) if role == 'full' else [i for i, row in enumerate(index) if row['fold'] != int(role[-1])]
        assert role_rows[role]['fit_compact_rows'] == expected_fit and role_rows[role]['held_compact_rows'] == expected_held
        old = torch.load(original_path, weights_only=True, map_location='cpu', mmap=True)
        old_scale = old['logit_scale'].detach().float()
        original_core = core_class(4250, 1024, num_layers=1, dropout=.1).eval().requires_grad_(False)
        original_core.load_state_dict(old['model'], strict=True)
        models, states, scales = {}, {}, {}
        for arm in ARMS:
            tick(); cp = checked(records[role, arm]['checkpoint'])
            saved = torch.load(cp, weights_only=True, map_location='cpu', mmap=True)
            assert saved['role'] == role and saved['arm'] == arm and saved['epoch'] == 25
            assert saved['initial_checkpoint'] == spec['roles'][role]['checkpoint'] and saved['spec'] == done['spec']
            state = saved['model']; initial_model = V.from_core_state(core_class, old['model'], arm).eval()
            changes = weight_stats(state, initial_model.state_dict())
            models[arm] = V.from_architecture_state(core_class, state, arm).eval().requires_grad_(False)
            states[arm] = state; scales[arm] = saved['logit_scale'].float()
            result['models'].append(dict(role=role, arm=arm, checkpoint=ref(cp), original_checkpoint=ref(original_path),
                changes=changes, original_logit_scale=float(old_scale), final_logit_scale=float(scales[arm]),
                original_exp_scale=float(old_scale.exp()), final_exp_scale=float(scales[arm].exp()),
                logit_scale_is_not_an_inference_encoder_parameter=True))
            del initial_model, saved
        result.setdefault('parameter_serial_parallel_comparisons', []).append(dict(role=role,
            current_serial_versus_parallel=weight_stats(states[ARMS[1]], states[ARMS[2]]),
            limitation='Weight changes alone do not determine real-input function displacement'))
        all_role_coverage = np.zeros(13230, np.uint8)
        for bucket in ('fit', 'held'):
            rows = role_rows[role][bucket + '_compact_rows']
            if not rows:
                continue
            assert len(set(rows)) == len(rows) and not all_role_coverage[rows].any()
            all_role_coverage[rows] += 1
            target = torch.from_numpy(np.array(targets[rows], copy=True))
            n = len(rows); endpoints = ('old_core_identity', 'new_core_identity', 'old_core_front', 'new_core_front')
            role_outputs = torch.empty((n, 3, 1024), dtype=torch.float32)
            x_source_digest = hashlib.sha256()
            front_shared, output_shared = SharedChanges(), SharedChanges()
            # Full real-input common-change decomposition; no synthetic probe.
            with torch.inference_mode():
                for lo in range(0, n, 256):
                    tick(); selected = rows[lo:lo + 256]
                    x = torch.from_numpy(np.array(trials[selected], copy=True).mean(axis=1, dtype=np.float32)).reshape(len(selected), 4250)
                    assert torch.isfinite(x).all(); x_source_digest.update(x.numpy().tobytes())
                    front_shared.add(models[ARMS[1]].front(x), models[ARMS[2]].front(x), x)
            for ai, arm in enumerate(ARMS):
                model = models[arm]
                arrays = {key: torch.empty((n, 1024), dtype=torch.float32) for key in endpoints}
                ordinary_arrays = {key: torch.empty_like(value) for key, value in arrays.items()}
                moments = {key: DifferenceMoments() for key in ('front', 'first_projection_front',
                    'changed_core_identity_vs_original', 'front_effect_final_core', 'front_effect_original_core',
                    'exact_core_front_interaction', 'opposite_routing_front', 'opposite_routing_output')}
                old_norm = original_core.block.layers[0].norm
                new_norm = model.core.block.layers[0].norm
                parity_abs = parity_rel = 0.
                with torch.inference_mode():
                    for lo in range(0, n, 256):
                        tick(); selected = rows[lo:lo + 256]; count = len(selected)
                        x = torch.from_numpy(np.array(trials[selected], copy=True).mean(axis=1, dtype=np.float32)).reshape(count, 4250)
                        transformed = model.front(x)
                        c00, c10 = original_core(x), model.core(x)
                        c01, c11 = original_core(transformed), model.core(transformed)
                        values = dict(zip(endpoints, (c00, c10, c01, c11)))
                        for key, value in values.items():
                            arrays[key][lo:lo + count] = value
                            norm = old_norm if key.startswith('old_') else new_norm
                            changed = ordinary(value, norm.weight[None].expand_as(value),
                                norm.bias[None].expand_as(value), gate_mode='fixed')[0]
                            ordinary_arrays[key][lo:lo + count] = changed
                        role_outputs[lo:lo + count, ai] = c11
                        moments['front'].add(transformed, x)
                        moments['first_projection_front'].add(model.core.proj_in(transformed), model.core.proj_in(x))
                        moments['changed_core_identity_vs_original'].add(c10, c00)
                        moments['front_effect_final_core'].add(c11, c10)
                        moments['front_effect_original_core'].add(c01, c00)
                        interaction = c11 - c10 - c01 + c00
                        moments['exact_core_front_interaction'].add(c00 + interaction, c00)
                        if arm != ARMS[0]:
                            opposite = front_route(model.front, x, serial=arm == ARMS[2])
                            moments['opposite_routing_front'].add(opposite, transformed)
                            moments['opposite_routing_output'].add(model.core(opposite), c11)
                        if bucket == 'held':
                            expected = torch.from_numpy(np.array(bank[selected, ai], copy=True))
                            parity_abs = max(parity_abs, float((c11 - expected).abs().max()))
                            parity_rel = max(parity_rel, float(((c11 - expected).norm(dim=-1) /
                                expected.norm(dim=-1).clamp_min(1e-12)).max()))
                            assert torch.allclose(c11, expected, atol=3e-4, rtol=3e-5), 'CPU final held forward/saved GPU OOF mismatch'
                statistics = {key: value.result() for key, value in moments.items() if value.rows}
                item = dict(role=role, arm=arm, bucket=bucket, queries=n,
                    concepts=len({index[row]['concept_id'] for row in rows}), all_rows_used=True,
                    compact_rows_sha256=hashlib.sha256(np.asarray(rows, np.int64).tobytes()).hexdigest(),
                    physical_mean4_input_sha256=x_source_digest.hexdigest(), changes=statistics,
                    exact_interaction_formula='c(g(x),phi)-c(x,phi)-c(g(x),phi0)+c(x,phi0)',
                    original_core_affine_applied_to_old_core_endpoints=True,
                    new_core_affine_applied_to_new_core_endpoints=True,
                    conditional_interventions_are_not_retrained_causal_ablations=True,
                    native_endpoint_metrics={key:quality(value, target, index, rows,
                        old_scale if key.startswith('old_') else scales[arm], old_scale, tick)
                        for key,value in arrays.items()},
                    frozen_ordinary_endpoint_metrics={key:quality(value, target, index, rows,
                        old_scale if key.startswith('old_') else scales[arm], old_scale, tick)
                        for key,value in ordinary_arrays.items()},
                    CPU_GPU_saved_OOF_held_parity=dict(performed=bucket=='held',
                        maximum_abs=parity_abs if bucket=='held' else None,
                        maximum_row_relative_L2=parity_rel if bucket=='held' else None,
                        absolute_tolerance=3e-4,relative_tolerance=3e-5,not_bitwise=True))
                result['buckets'].append(item)
                write(output / (role + '_' + arm + '_' + bucket + '.json'), item)
                if bucket=='held':
                    parity_rows += n; parity_maximum=max(parity_maximum,parity_abs); parity_relative=max(parity_relative,parity_rel)
                del arrays, ordinary_arrays, moments
            output_shared.add(role_outputs[:,1], role_outputs[:,2], role_outputs[:,0])
            result.setdefault('real_input_shared_changes', []).append(dict(role=role,bucket=bucket,
                front=front_shared.result(),output=output_shared.result()))
            del target, role_outputs
        assert np.all(all_role_coverage==1), 'Each role must replay every one of13230T rows exactly once across FIT/held'
        del models, states, old, original_core
    full_shared = SharedChanges()
    for lo in range(0,13230,256):
        tick(); values=torch.from_numpy(np.array(bank[lo:lo+256],copy=True))
        full_shared.add(values[:,1],values[:,2],values[:,0])
    result['full13230_OOF_pointwise_changes']=dict(bank=ref(done['OOF_T_native']['path']),**full_shared.result())
    result['all_final_held_forward_parity']=dict(arm_query_comparisons=parity_rows,
        expected=13230*3,maximum_abs=parity_maximum,maximum_row_relative_L2=parity_relative,
        performed_on_all_final_OOF_predictions=True)
    assert parity_rows==13230*3 and not torch.cuda.is_initialized()
    result.update(actual_final_checkpoints_read=9,all_T_rows_per_role=13230,
        actual_T_trial_mean_values_read=True,T_only=True,held_values_used_for_fit_or_selection=False,
        whole_ordinary_chain_independent_OOF=False,old_ordinary_previously_trained_on_all_T=True,
        GPU_initialized=False,training_updates=0,new_images=0,scientific_admission_written=False,
        limitation='Final four-function analysis cannot recover unlogged per-epoch held dynamics or isolate the cause of original-to-continuation training-regime drift')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True)
    parser.add_argument('--OOF-summary', dest='summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ordinary-freeze', type=Path, default=ROOT / 'configs/ordinary_frozen_base_20261028_v1.json')
    parser.add_argument('--mode', choices=('metadata', 'checkpoints'), default='metadata')
    parser.add_argument('--max-seconds', type=int, default=600)
    args = parser.parse_args(); assert 60 <= args.max_seconds <= 600
    assert not args.output.exists(), 'New diagnostic output required; retain prior results'
    args.output.mkdir(parents=True)
    started = time.monotonic()
    def peak_rss():
        path = Path('/proc/self/status')
        if not path.exists():
            return None
        return int(next(line.split()[1] for line in path.read_text().splitlines()
            if line.startswith('VmHWM:'))) * 1024
    initial_rss = peak_rss()
    timer_enabled = hasattr(signal, 'setitimer')
    def timed_out(*_):
        raise TimeoutError('Finite whole CPU diagnostic deadline')
    if timer_enabled:
        signal.signal(signal.SIGALRM, timed_out)
        signal.setitimer(signal.ITIMER_REAL, args.max_seconds)
    def tick():
        assert time.monotonic() - started < args.max_seconds, 'Finite CPU diagnostic limit'
        current_rss = peak_rss()
        if current_rss is not None:
            assert current_rss - initial_rss < 10 << 30, 'Additional10GiB CPU diagnostic limit'
        if torch is not None:
            assert not torch.cuda.is_initialized()
    try:
        meta = metadata(args.completion, args.summary)
        meta.update(source=ref(__file__), UTC=datetime.datetime.now(datetime.timezone.utc).isoformat())
        write(args.output / 'metadata_dynamics.json', meta)
        if args.mode == 'checkpoints':
            details = checkpoint_diagnostics(args.completion, args.summary, args.ordinary_freeze, tick, args.output)
            details['seconds'] = time.monotonic() - started
            write(args.output / 'checkpoint_function_diagnostics.json', details)
        tick()
        write(args.output / 'completion.json', dict(status='D30_READ_ONLY_POSTHOC_DIAGNOSTIC_COMPLETED',
            mode=args.mode, source=ref(__file__), metadata=ref(args.output / 'metadata_dynamics.json'),
            checkpoint_diagnostics=ref(args.output / 'checkpoint_function_diagnostics.json')
                if args.mode == 'checkpoints' else None,
            actual_checkpoint_weights_read=args.mode == 'checkpoints', actual_GPU_jobs=0,
            no_original_sources_or_checkpoints_modified=True, seconds=time.monotonic() - started,
            peak_extra_CPU_bytes=peak_rss() - initial_rss if initial_rss is not None else None,
            memory_check_is_software_not_OS_quota=True))
    except BaseException as exc:
        write(args.output / 'failure.json', dict(status='D30_READ_ONLY_DIAGNOSTIC_FAILURE_PREFIX_PRESERVED',
            error=repr(exc), traceback=traceback.format_exc(), seconds=time.monotonic() - started,
            actual_GPU_jobs=0, no_training_updates=True))
        raise
    finally:
        if timer_enabled:
            signal.setitimer(signal.ITIMER_REAL, 0)


if __name__ == '__main__':
    main()
