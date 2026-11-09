"""Retrospective diagnosis using all seven CLOSED historical V image scores.

No router is fitted, no V permission is changed, and an oracle is not a method.
Same-query arm choices below deliberately use unavailable image outcomes only
to bound heterogeneity; their values cannot be reported as EEG reconstruction.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np


def sha(path):
    with Path(path).open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()
def ref(path): return dict(path=str(Path(path).resolve()), sha256=sha(path))
def read(path): return json.loads(Path(path).read_bytes())
def records(path):
    with Path(path).open(encoding='utf-8', newline='') as stream: return list(csv.DictReader(stream))
def verify(item):
    path = Path(item['path']); assert sha(path) == item['sha256']; return path


def correlation(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    assert np.isfinite(a).all() and np.isfinite(b).all()
    a = a - a.mean(); b = b - b.mean(); norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-12 else None


def portfolio(tables):
    names = sorted(tables); keys = sorted(tables[names[0]])
    base_rows = [tables[names[0]][key] for key in keys]
    for arm in names:
        assert sorted(tables[arm]) == keys
        for key, row in zip(keys, base_rows):
            for field in ('subject', 'stimulus_id', 'concept_id', 'image_seed', 'score_target_path', 'EEG_source_stimulus_id'):
                assert row[field] == tables[arm][key][field], (arm, key, field)
    metrics = {metric: np.array([[float(tables[arm][key][metric]) for arm in names] for key in keys])
               for metric in ('DINO2AFC', 'DINO_cosine', 'SEED', 'ObjectF1')}
    assert all(np.isfinite(value).all() for value in metrics.values())
    dino = metrics['DINO2AFC']; best = int(dino.mean(0).argmax()); oracle = dino.argmax(1)
    baseline = dino[:, best]; selected = dino[np.arange(len(keys)), oracle]
    grouped = {}
    for field in ('subject', 'concept_id'):
        groups = {}
        for i, row in enumerate(base_rows): groups.setdefault(row[field], []).append(i)
        gains = []; choices = {}
        for label, ids in groups.items():
            choice = int(dino[ids].mean(0).argmax()); choices[label] = names[choice]
            gains.append((dino[ids, choice] - baseline[ids]).mean())
        grouped[field] = dict(oracle_gain_over_best_constant=float(np.mean(gains)),
            groups=len(groups), choices=choices, unavailable_outcome_selection=True)
    return dict(arms=names, queries=len(keys), gallery_targets=1660,
        constant_means={arm: {metric: float(value[:, i].mean()) for metric, value in metrics.items()} for i, arm in enumerate(names)},
        posthoc_best_constant_arm=names[best], posthoc_best_constant_DINO=float(baseline.mean()),
        uniform_random_arm_expected_DINO=float(dino.mean()),
        unavailable_perquery_DINO_oracle_mean=float(selected.mean()),
        unavailable_perquery_DINO_oracle_gain=float((selected - baseline).mean()),
        oracle_pick_counts={arm: int((oracle == i).sum()) for i, arm in enumerate(names)},
        oracle_has_at_least_001_gain_fraction=float((selected - baseline >= .01).mean()),
        other_metrics_under_same_DINO_oracle={metric: dict(mean=float(value[np.arange(len(keys)), oracle].mean()),
            difference_from_best_constant=float((value[np.arange(len(keys)), oracle] - value[:, best]).mean()))
            for metric, value in metrics.items() if metric != 'DINO2AFC'},
        group_oracles=grouped, router_fitted=False, viable_inference_policy_established=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('discovery', 'risk-inference-manifest', 'output'): parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    discovery = read(args.discovery); assert discovery['status'] == 'HISTORICAL_FULLV_READ_ONLY_DISCOVERY_COMPLETE'
    all_tables = {}; families = {}; source_files = []
    for family in discovery['families']:
        assert family['full_family_available'] and family['gallery_images'] == 1660 and family['queries_per_arm'] == 16600
        tables = {}
        for arm, spec in family['scores'].items():
            path = verify(spec['binding']); table = records(path); source_files.append(ref(path))
            assert len(table) == 16600 and len({row['query_id'] for row in table}) == 16600
            tables[arm] = {row['query_id']: row for row in table}
            all_tables[family['name'] + '__' + arm] = tables[arm]
        families[family['name']] = portfolio(tables)
    # A cross-family portfolio is allowed for this diagnostic only if ALL
    # observed query/seed/target fields match. Otherwise retain within-family.
    try: combined = portfolio(all_tables)
    except AssertionError as error: combined = dict(status='CROSS_FAMILY_IDENTITY_NOT_MATCHED', reason=repr(error))
    manifest = [json.loads(line) for line in args.risk_inference_manifest.read_text().splitlines() if line]
    assert len(manifest) == 16600 and len({row['query_id'] for row in manifest}) == 16600
    risk_rows = {row['query_id']: row for row in manifest}
    bounded_name = next(name for name in families if 'bounded_three' in name)
    prefix = bounded_name + '__'; risk = all_tables[prefix + 'risk']; ordinary = all_tables[prefix + 'ordinary']; parent = all_tables[prefix + 'parent']
    assert set(risk_rows) == set(risk) == set(ordinary) == set(parent)
    by_subject = {}
    for key, row in risk.items():
        original = risk_rows[key]
        for field in ('subject', 'stimulus_id', 'image_seed', 'generated_path'): assert row[field] == str(original[field])
        assert original['target_or_visual_features_read_for_condition'] is False
        by_subject.setdefault(row['subject'], []).append(key)
    correlations = {}
    for feature in ('predicted_risk', 'component_gate', 'component_correction_RMS'):
        correlations[feature] = {}
        for label, left, right in (('risk_minus_parent', risk, parent), ('ordinary_minus_parent', ordinary, parent), ('risk_minus_ordinary', risk, ordinary)):
            per_subject = {}
            centered_x = []; centered_y = []
            for subject, keys in sorted(by_subject.items()):
                x = np.array([float(risk_rows[k][feature]) for k in keys])
                y = np.array([float(left[k]['DINO2AFC']) - float(right[k]['DINO2AFC']) for k in keys])
                per_subject[subject] = correlation(x, y); centered_x.extend(x - x.mean()); centered_y.extend(y - y.mean())
            correlations[feature][label] = dict(per_subject_Pearson=per_subject,
                subject_centered_pooled_Pearson=correlation(centered_x, centered_y),
                descriptive_only=True, conditional_causal_effect_identified=False)
    for binding in source_files: verify(binding)
    result = dict(status='COMPLETE_HISTORICAL_EFFECT_HETEROGENEITY_DIAGNOSTIC', source=ref(__file__), discovery=ref(args.discovery),
        score_tables=source_files, risk_inference_manifest=ref(args.risk_inference_manifest), families=families,
        combined_portfolio=combined, original_risk_proxy_vs_actual_increment=correlations,
        no_new_method_fit=True, no_new_GPU_calls=True, V_fit_permission_changed=False,
        inference_router_selected=False, unavailable_oracles_are_not_generated_method_results=True,
        independent_confirmation=False, SOTA_established=False, seconds=time.monotonic() - started)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=result['status'], seconds=result['seconds'],
        families={name: {key: value[key] for key in ('posthoc_best_constant_arm', 'posthoc_best_constant_DINO', 'unavailable_perquery_DINO_oracle_gain')} for name, value in families.items()},
        combined={key: combined[key] for key in ('posthoc_best_constant_arm', 'posthoc_best_constant_DINO', 'unavailable_perquery_DINO_oracle_gain') if key in combined},
        no_new_GPU_calls=True))); return 0


if __name__ == '__main__': raise SystemExit(main())
