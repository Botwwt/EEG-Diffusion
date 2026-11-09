"""D09 fixed calibration math and paired DINO readout, without GPU imports."""
import csv
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import score_BrainHIVE_rapid_cached_DINO_CPU_20261027_v2 as S

SCHEMA = 'BrainHIVE_actual_gain_calibration_20261027_v1'
ARMS = ('parent', 'ordinary', 'risk')
SEED = 1729


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_bytes())
def checked(value):
    path = Path(value['path']); assert sha(path) == value['sha256']; return path
def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.pending'); assert not path.exists() and not temporary.exists()
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.rename(path)
def save_array(path, value):
    with Path(path).open('xb') as stream:
        np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())


def choose_queries(split, metadata, partition, count):
    key = {'Dfit': 'd_fit_concepts', 'Dcheck': 'd_check_concepts'}[partition]
    concepts = set(split[key]); assert not concepts.intersection(split['train_concepts'])
    ranked = sorted(concepts, key=lambda c: (hashlib.sha256(f'{SCHEMA}|1729|{partition}|{c}'.encode()).hexdigest(), c))[:count]
    assert len(ranked) == count
    result = []
    for concept in ranked:
        members = sorted([r for r, item in metadata.items() if item['concept_id'] == concept])
        assert len(members) == 10
        for row in members:
            item = metadata[row]; stimulus = item['stimulus_id']; query = f'sub-01/{stimulus}/R4'
            seed = int(hashlib.sha256(f"1729|sub-01|{stimulus}|R4|{split['version']}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
            result.append(dict(query_id=query, subject_id='sub-01', concept_id=concept, stimulus_id=stimulus,
                               array_row=row, partition=partition, image_seed=seed))
    assert len({q['query_id'] for q in result}) == 10 * count
    return result


def fit_router(features, cosine_values):
    features = np.asarray(features, np.float64); values = np.asarray(cosine_values, np.float64)
    assert features.shape == (200, 28) and values.shape == (200, 3)
    assert np.isfinite(features).all() and np.isfinite(values).all()
    center = features.mean(0); response = values[:, 1:] - values[:, :1]; intercept = response.mean(0)
    x = features - center; y = response - intercept
    matrix = x.T @ x / len(x) + np.eye(28)  # fixed ridge=1, unpenalized intercept
    rhs = x.T @ y / len(x); weights = np.linalg.solve(matrix, rhs)
    assert np.linalg.norm(matrix @ weights - rhs) / max(np.linalg.norm(rhs), 1e-12) < 1e-8
    fitted = (x @ weights + intercept); residual = response - fitted
    return dict(feature_center=center.tolist(), weights=weights.tolist(), intercept=intercept.tolist(), ridge=1.,
        coefficients=58, arms=list(ARMS), calibration_rows=200, calibration_label='actual generated DINO target cosine minus parent',
        fit_response_means=intercept.tolist(), fit_response_std=response.std(0).tolist(),
        fit_residual_RMS=np.sqrt(np.square(residual).mean(0)).tolist(),
        fixed_global_arm=int(values.mean(0).argmax()), fixed_global_selection_metric='Dfit target cosine mean',
        fitting_uses_Dcheck_or_V=False)


def predict_router(features, model):
    features = np.asarray(features, np.float64); assert features.ndim == 2 and features.shape[1] == 28
    values = (features - np.asarray(model['feature_center'])) @ np.asarray(model['weights']) + np.asarray(model['intercept'])
    scores = np.concatenate([np.zeros((len(values), 1)), values], axis=1)
    assert np.isfinite(scores).all()
    return scores.argmax(1), scores  # fixed order parent, ordinary, risk breaks ties


def derangement(count):
    order = np.random.default_rng(SEED).permutation(count)
    result = np.empty(count, np.int64); result[order] = np.roll(order, 1)
    assert sorted(result.tolist()) == list(range(count)) and np.all(result != np.arange(count))
    return result


def score_features(generated, target, queries, gallery, choices, fixed_global, stage, shuffled_choices=None):
    assert stage in (80, 400) and generated.shape == (stage, 3, 768) and target.shape == (400, 768)
    assert len(queries) == stage and len(gallery) == 400 and choices.shape == (stage,)
    assert set(choices).issubset({0, 1, 2}) and int(fixed_global) in (0, 1, 2)
    self_check = S.metric_arrays(target, target, gallery, gallery)
    assert np.all(self_check['DINO2AFC'] == 1) and np.all(self_check['gallery_top1'] == 1)
    metrics = {arm: S.metric_arrays(generated[:, i], target, queries, gallery) for i, arm in enumerate(ARMS)}
    picked = generated[np.arange(stage), choices]
    policy = S.metric_arrays(picked, target, queries, gallery)
    contrasts = {arm: S.compare(policy, metrics[arm], queries) for arm in ARMS}
    primary = contrasts[ARMS[fixed_global]]
    action = S.decision(primary, str(stage))
    if stage == 400 and action['action'] == 'PROMISING_FOR_FULL_EVALUATION':
        all_pass = all(S.decision(value, '400')['action'] == 'PROMISING_FOR_FULL_EVALUATION' for value in contrasts.values())
        if not all_pass: action = action | dict(action='STOP_INSTANCE', reason='NOT_PROMISING_AGAINST_EVERY_FROZEN_ARM')
    means = {arm: {k: float(v.mean()) for k, v in value.items()} for arm, value in metrics.items()}
    means['gain_policy'] = {k: float(v.mean()) for k, v in policy.items()}
    output = dict(status='COMPLETE_D09_PAIRED_DEVELOPMENT_READOUT', stage=stage, queries=stage, concepts=stage // 10,
        gallery_targets=400, fixed_global_arm=ARMS[fixed_global], means=means, policy_minus_each_arm=contrasts, decision=action,
        policy_choice_counts={arm: int((choices == i).sum()) for i, arm in enumerate(ARMS)},
        full_registered_stage_denominator=True, hidden_exclusions=0, choices_locked_before_Dcheck_generation=True,
        query_target_used_for_choice=False, independent_confirmation=False, SOTA_established=False)
    if shuffled_choices is not None:
        assert shuffled_choices.shape == (stage,) and set(shuffled_choices).issubset({0, 1, 2})
        shuffled = S.metric_arrays(generated[np.arange(stage), shuffled_choices], target, queries, gallery)
        means['shuffled_label_policy'] = {k: float(v.mean()) for k, v in shuffled.items()}
        output['policy_minus_shuffled_label_policy'] = S.compare(policy, shuffled, queries)
        output['shuffled_policy_choice_counts'] = {arm: int((shuffled_choices == i).sum()) for i, arm in enumerate(ARMS)}
        output['shuffled_control_additional_GPU_images'] = 0
    return output


def write_query_metrics(path, generated, target, queries, gallery, choices, shuffled_choices=None):
    metrics = {arm: S.metric_arrays(generated[:, i], target, queries, gallery) for i, arm in enumerate(ARMS)}
    metrics['gain_policy'] = S.metric_arrays(generated[np.arange(len(queries)), choices], target, queries, gallery)
    if shuffled_choices is not None:
        metrics['shuffled_label_policy'] = S.metric_arrays(generated[np.arange(len(queries)), shuffled_choices], target, queries, gallery)
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        fields = ['query_id', 'subject_id', 'concept_id', 'stimulus_id', 'array_row', 'image_seed', 'arm', *S.METRICS]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for i, query in enumerate(queries):
            for arm, values in metrics.items():
                writer.writerow({k: query[k] for k in fields[:6]} | dict(arm=arm) | {k: float(v[i]) for k, v in values.items()})


def self_test():
    rng = np.random.default_rng(1729); x = rng.normal(size=(200, 28)); base = rng.normal(size=200) * .01
    labels = np.stack([base, base + .01 * x[:, 0], base - .01 * x[:, 0]], axis=1)
    model = fit_router(x, labels); choices, values = predict_router(x, model)
    assert model['coefficients'] == 58 and np.mean(choices == np.where(x[:, 0] > 0, 1, 2)) > .9
    tie = fit_router(x, np.zeros((200, 3))); assert np.all(predict_router(x, tie)[0] == 0)
    # Shared frozen metric functions, including ties, stay exactly the v2 implementation.
    eye = np.eye(400, 768, dtype=np.float32)
    gallery = [dict(stimulus_id=str(i), concept_id=str(i // 10)) for i in range(400)]
    values = np.stack([eye[:80]] * 3, axis=1)
    result = score_features(values, eye, gallery[:80], gallery, np.zeros(80, np.int64), 0, 80)
    assert result['means']['gain_policy']['DINO2AFC'] == 1
    assert result['policy_minus_each_arm']['parent']['DINO2AFC']['difference'] == 0
    assert result['decision']['action'] == 'EXPAND400'
    perm = derangement(200); assert np.all(perm != np.arange(200)) and np.allclose(labels.mean(0), labels[perm].mean(0))
    print(json.dumps(dict(status='D09_CALIBRATION_AND_POLICY_CPU_CHECKS_PASSED', checks=4,
        checks_scope=['fixed_ridge_known_signal_and_shuffled_marginals', 'parent_tie_break', 'all400_gallery_self_pairing', 'identical_branch_zero_difference'],
        GPU_used=False)))


if __name__ == '__main__': self_test()
