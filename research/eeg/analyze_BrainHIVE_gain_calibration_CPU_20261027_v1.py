"""D09 complete calibration replay, blind-choice timing and held utility checks.

This is a post-result readout, never a refit or a source of new selections.
No torch import, generator, visual model, or GPU execution occurs here.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import brainhive_gain_calibration_20261027_v1 as U


def pearson(a, b):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    a = a - a.mean(); b = b - b.mean(); norm = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / norm) if norm > 1e-12 else None


def prediction_check(predictions, actual, fit_constant):
    error = np.square(predictions - actual).mean()
    constant_error = np.square(actual - fit_constant).mean()
    return dict(rows=len(actual), actual_mean=float(actual.mean()), actual_std=float(actual.std()),
        prediction_mean=float(predictions.mean()), prediction_std=float(predictions.std()),
        correlation_Pearson=pearson(predictions, actual), RMS_error=float(np.sqrt(error)),
        frozen_Dfit_constant_RMS_error=float(np.sqrt(constant_error)),
        R2_against_frozen_Dfit_constant=float(1 - error / constant_error) if constant_error > 1e-20 else None,
        predicted_positive_fraction=float((predictions > 0).mean()), actual_positive_fraction=float((actual > 0).mean()),
        sign_agreement_fraction=float(((predictions > 0) == (actual > 0)).mean()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--completion', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); started = time.monotonic()
    assert not args.output.exists(); args.output.mkdir(parents=True)
    done = U.read(args.completion)
    assert done['status'] in ('COMPLETE_D09_STAGE80', 'COMPLETE_D09_STAGE400', 'D09_ANALYTICAL_CONSTANT_ACTION_STOP')
    lock_path = U.checked(done['policy_lock']); lock = U.read(lock_path)
    model_path = U.checked(lock['model']); model = U.read(model_path)
    shuffled_model = U.read(U.checked(lock['shuffled_model']))
    source_dir = model_path.parent
    features = np.load(source_dir / 'features_Dfit.npy', allow_pickle=False)
    generated_fit = np.load(source_dir / 'Dfit_generated_features.npy', allow_pickle=False)
    target_fit = np.load(source_dir / 'Dfit_target_features.npy', allow_pickle=False)
    assert features.shape == (200, 28) and generated_fit.shape == (200, 3, 768) and target_fit.shape == (200, 768)
    labels = (generated_fit / np.linalg.norm(generated_fit, axis=2, keepdims=True) *
              (target_fit / np.linalg.norm(target_fit, axis=1, keepdims=True))[:, None]).sum(2).astype(np.float64)
    replay = U.fit_router(features, labels)
    max_weight_error = float(np.max(np.abs(np.asarray(replay['weights']) - model['weights'])))
    assert max_weight_error < 1e-10 and replay['fixed_global_arm'] == model['fixed_global_arm']
    assert np.allclose(replay['intercept'], model['intercept'], atol=1e-12, rtol=0)
    permutation = np.asarray(lock['Dfit_label_derangement']); assert np.array_equal(permutation, U.derangement(200))
    shuffled_replay = U.fit_router(features, labels[permutation])
    assert np.allclose(shuffled_replay['weights'], shuffled_model['weights'], atol=1e-10, rtol=0)
    check_features = np.load(U.checked(lock['features']), allow_pickle=False); assert check_features.shape == (400, 28)
    choices, predicted = U.predict_router(check_features, model)
    shuffled_choices, shuffled_predicted = U.predict_router(check_features, shuffled_model)
    assert choices.tolist() == lock['choices'] and shuffled_choices.tolist() == lock['shuffled_choices']
    assert np.allclose(predicted, np.asarray(lock['predicted_utilities']), atol=1e-12, rtol=0)
    summary = dict(status='COMPLETE_D09_POSTRESULT_CPU_ANALYSIS', source=U.ref(__file__), completion=U.ref(args.completion),
        policy_lock=U.ref(lock_path), coefficient_replay_max_absolute_error=max_weight_error,
        exact_same_Dfit_coefficient_replay_only=True, model_or_choices_modified=False,
        Dfit_response_means=model['fit_response_means'], Dfit_response_std=model['fit_response_std'],
        Dfit_training_residual_RMS=model['fit_residual_RMS'],
        Dfit_training_R2=[float(1 - (r / s) ** 2) if s > 1e-12 else None for r, s in zip(model['fit_residual_RMS'], model['fit_response_std'])],
        fixed_global_arm=U.ARMS[model['fixed_global_arm']], all400_choice_counts={arm: int((choices == i).sum()) for i, arm in enumerate(U.ARMS)},
        all400_shuffled_choice_counts={arm: int((shuffled_choices == i).sum()) for i, arm in enumerate(U.ARMS)},
        full400_policy_maximum_possible_DINO_gain_over_each_fixed_arm={arm: float((choices != i).mean()) for i, arm in enumerate(U.ARMS)},
        theoretical_bounds_are_not_achieved_image_gains=True, GPU_used=False, V_fit_permission_changed=False,
        independent_confirmation=False, SOTA_established=False)
    if done['status'] == 'D09_ANALYTICAL_CONSTANT_ACTION_STOP':
        assert len(set(choices.tolist())) == 1
        assert not (source_dir / 'Dcheck_inference_manifest.jsonl').exists()
        summary.update(check_images=0, check_metrics=None, analytical_identity_stop_verified=True,
            reason='All400 policy choices equal one fixed arm; strict practical gain against that same arm is impossible')
    else:
        stage = int(done['stage']); assert stage in (80, 400)
        inputs = U.read(U.checked(done['inputs'])); panel = U.read(U.checked(inputs['panel']))
        check_generated = np.load(U.checked(done['check_generated_features']), allow_pickle=False)
        target_check = np.load(U.checked(done['check_target_features']), allow_pickle=False)
        recomputed = U.score_features(check_generated, target_check, panel['Dcheck'][:stage], panel['Dcheck'],
            choices[:stage], model['fixed_global_arm'], stage, shuffled_choices[:stage])
        assert recomputed == done['readout']
        records = [json.loads(line) for line in U.checked(done['check_manifest']).read_text().splitlines() if line]
        assert len(records) == 3 * stage
        earliest_image = min(Path(row['generated_path']).stat().st_mtime_ns for row in records)
        assert lock_path.stat().st_mtime_ns <= earliest_image, 'Choice lock must predate every check image'
        lock_time = lock_path.stat().st_mtime_ns / 1e9; image_time = earliest_image / 1e9
        utility = np.stack([U.S.metric_arrays(check_generated[:, i], target_check, panel['Dcheck'][:stage], panel['Dcheck'])['target_cosine']
                            for i in range(3)], axis=1)
        actual_delta = utility[:, 1:] - utility[:, :1]
        prediction_stats = {}
        for i, arm in enumerate(U.ARMS[1:]):
            prediction_stats[arm] = prediction_check(predicted[:stage, i + 1], actual_delta[:, i], model['intercept'][i])
            prediction_stats[arm]['shuffled_label_prediction'] = prediction_check(shuffled_predicted[:stage, i + 1], actual_delta[:, i], shuffled_model['intercept'][i])
        chosen = utility[np.arange(stage), choices[:stage]]
        fixed = utility[:, model['fixed_global_arm']]
        empirical_max = utility.max(1)
        summary.update(check_queries=stage, check_concepts=stage // 10, descriptive_prediction_stats=prediction_stats,
            real_check_outcomes_used_only_after_choice_lock=True, policy_lock_epoch=lock_time, earliest_check_image_epoch=image_time,
            seconds_lock_precedes_first_check_image=image_time - lock_time,
            target_cosine_policy_gain_vs_preselected_global=float((chosen - fixed).mean()),
            unavailable_target_cosine_perquery_oracle_gain_vs_preselected_global=float((empirical_max - fixed).mean()),
            unavailable_oracle_is_not_an_inference_result=True, readout=done['readout'])
        with (args.output / 'held_utility_predictions.csv').open('x', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['query_id', 'concept_id', 'chosen_arm', 'ordinary_predicted_delta', 'ordinary_actual_delta', 'risk_predicted_delta', 'risk_actual_delta'])
            writer.writeheader()
            for i, query in enumerate(panel['Dcheck'][:stage]):
                writer.writerow(dict(query_id=query['query_id'], concept_id=query['concept_id'], chosen_arm=U.ARMS[choices[i]],
                    ordinary_predicted_delta=float(predicted[i, 1]), ordinary_actual_delta=float(actual_delta[i, 0]),
                    risk_predicted_delta=float(predicted[i, 2]), risk_actual_delta=float(actual_delta[i, 1])))
    summary['seconds'] = time.monotonic() - started
    U.atomic(args.output / 'summary.json', summary)
    print(json.dumps(dict(status=summary['status'], seconds=summary['seconds'], choice_counts=summary['all400_choice_counts'],
        fixed_global_arm=summary['fixed_global_arm'], check_queries=summary.get('check_queries', 0),
        GPU_used=False, model_or_choices_modified=False))); return 0


if __name__ == '__main__': raise SystemExit(main())
