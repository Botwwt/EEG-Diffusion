"""Actual metadata contracts and synthetic future-record checks; no tensors."""
import argparse
import copy
import json
from pathlib import Path
import sys
import unittest

import brainhive_official_encoder_job_contract_CPU_v1 as C

DEFAULT_PLAN_DIRECTORY = C.P.ROOT / 'audit/official_fullfit_metadata_plan_20261022_v1'
PLAN_DIRECTORY = DEFAULT_PLAN_DIRECTORY


def checker_main(argv=None):
    """Consume the explicit plan path; preserve unittest's own selectors/flags."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--plan-directory', type=Path, default=DEFAULT_PLAN_DIRECTORY)
    args, remaining = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    global PLAN_DIRECTORY
    PLAN_DIRECTORY = args.plan_directory
    return unittest.main(argv=[sys.argv[0], *remaining], verbosity=2)


def fitted_metadata(job):
    # Explicit synthetic producer fixtures. No checkpoint/cache exists or is
    # claimed verified by the contract validator's metadata-only verdict.
    complete = dict(subject=job['subject'], role=job['role'], training_seed=1729, epochs=25, updates=225,
        unique_train_stimuli=8270, training_exposures=206750, parameters=6452224,
        checkpoint_selection='fixed final epoch', trainable_logit_parameters=1,
        registration_core_sha256=job['registration_core_sha256'], registration_sha256='1' * 64,
        producer_source_sha256='2' * 64, model_path=job['outputs']['final_checkpoint'], model_sha256='3' * 64)
    epochs = [dict(subject=job['subject'], role=job['role'], epoch=i, updates=i * 9,
        expected_updates=225, full_epoch_rows=8270) for i in range(1, 26)]
    return complete, epochs


def exported_metadata(job, fit):
    return dict(subject=job['subject'], role=job['role'], registration_core_sha256=job['registration_core_sha256'],
        checkpoint_sha256=fit['model_sha256'], checkpoint_path=fit['model_path'],
        cache_path=job['outputs']['condition_cache'], cache_sha256='4' * 64,
        EEG_input_path=job['registration']['EEG_input']['path'], EEG_input_sha256=job['registration']['EEG_input']['sha256'],
        physical_repeat_count=4, model_training_mode=False, gradients_enabled=False, all_model_parameters_frozen=True,
        array_rows=job['expected_export_rows'], condition_shape=[8270, 1024], condition_dtype='float32',
        normalization='raw LayerNorm output', additional_L2_normalization=False, all_condition_values_finite=True,
        affine_source_checkpoint_sha256=fit['model_sha256'], affine_keys=C.AFFINE_KEYS, affine_shape=[1024],
        official_test_inputs_loaded=False, target_image_or_concept_used_for_export=False,
        official_retrieval_computed=False, model_fitted_during_export=False)


class EncoderJobContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = C.read_plan_bundle(PLAN_DIRECTORY)
        cls.job = C.job_for_task(cls.bundle, 'sub-01/official_oof0')
        cls.published = json.loads((C.P.ROOT /
            'audit/BrainHIVE_native_proof_bundle_20261020_v4/native_work/runs/published_source_dropout_v2/registration.json').read_text())
        cls.parent_proof = json.loads((C.P.ROOT / 'audit/BrainHIVE_all40_parent_training_CPU_verification_v1.json').read_text())

    def test_real_plan_to_twenty_exact_jobs_and_unique_outputs(self):
        jobs = [C.job_for_task(self.bundle, task['task_id']) for task in self.bundle['tasks']['tasks']]
        self.assertEqual(len(jobs), 20)
        roots = [j['output_root'] for j in jobs]
        self.assertEqual(len(set(roots)), 20)
        self.assertEqual(len({path for job in jobs for path in job['outputs'].values()}), 140)
        for job in jobs:
            result = C.validate_job(self.bundle, job)
            self.assertEqual(result['status'], C.STATUS)
            self.assertFalse(result['fitting_admitted'])
            self.assertFalse(result['payload_verified'])
            self.assertEqual(result['actual_parameter_updates'], 0)

    def test_fixed_native_teacher_optimizer_logit_seed_and_export_scope(self):
        reg = self.job['registration']
        self.assertEqual(reg['frozen_visual_training_dropout'], .1)
        self.assertFalse(reg['deterministic_F_cache_used_as_training_target'])
        self.assertEqual(reg['learned_logit_scale_initial_value'], 2.6592)
        self.assertEqual(reg['AdamW_betas'], [.9, .995])
        self.assertEqual((reg['training_seed'], reg['shuffle_generator_seed']), (1729, 1730))
        self.assertEqual((len(reg['training_array_rows']), len(self.job['expected_export_rows'])), (8270, 8270))
        self.assertTrue(set(reg['training_concepts']).isdisjoint(reg['excluded_oof_concepts']))
        self.assertFalse(reg['automatic_retrieval_diagnostics'])

    def test_rehashed_hyperparameter_teacher_or_old_T_scope_changes_rejected(self):
        for key, value in [('learning_rate', .01), ('frozen_visual_training_dropout', 0.),
                ('deterministic_F_cache_used_as_training_target', True), ('training_seed', 2025),
                ('training_array_rows', list(range(13230))), ('updates_per_subject', 175),
                ('automatic_retrieval_diagnostics', True)]:
            bad = copy.deepcopy(self.job); bad['registration'][key] = value
            bad['registration_core_sha256'] = C.core_sha(bad['registration'])
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_job(self.bundle, bad)

    def test_wrong_subject_or_cross_fold_job_rejected(self):
        for key, value in [('subject', 'sub-02'), ('role', 'official_oof1')]:
            bad = copy.deepcopy(self.job); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_job(self.bundle, bad)
        with self.assertRaises(ValueError):
            C.job_for_task(self.bundle, 'sub-01/oof0')

    def test_output_alias_and_old_development_namespace_rejected(self):
        with self.assertRaises(ValueError):
            C.job_for_task(self.bundle, self.job['task_id'], '/root/brainhive_reproduction_20261020_v1/runs')
        bad = copy.deepcopy(self.job)
        bad['outputs']['final_checkpoint'] = bad['outputs']['condition_cache']
        with self.assertRaises(ValueError):
            C.validate_job(self.bundle, bad)
        bad = copy.deepcopy(self.job); bad['output_root'] += '/../escape'
        with self.assertRaises(ValueError):
            C.validate_job(self.bundle, bad)

    def test_contract_cannot_issue_method_lock_fit_command_or_executor(self):
        for key, value in [('method_locked', True), ('fitting_admitted', True), ('automatic_launch_enabled', True),
                ('executable_commands', ['python train.py']), ('future_executor_source_sha256', 'a' * 64)]:
            bad = copy.deepcopy(self.job); bad[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_job(self.bundle, bad)
        self.assertNotIn('torch', sys.modules)

    def test_resume_fields_and_all_real_fit_preconditions_are_required(self):
        for key in ('required_resume_fields', 'required_before_any_fit'):
            bad = copy.deepcopy(self.job); bad[key] = []
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_job(self.bundle, bad)
        bad = copy.deepcopy(self.job); bad['command'] = ['python', 'train.py']
        with self.assertRaises(ValueError):
            C.validate_job(self.bundle, bad)

    def test_complete_future_fit_metadata_is_still_not_payload_or_fit_proof(self):
        complete, epochs = fitted_metadata(self.job)
        result = C.validate_fit_metadata(self.job, complete, epochs)
        self.assertEqual(result['status'], C.PAYLOAD_PENDING)
        self.assertFalse(result['checkpoint_payload_verified'])
        self.assertFalse(result['actual_fit_certified_here'])
        self.assertFalse(result['fitting_admitted'])

    def test_partial_epochs_wrong_budget_or_wrong_checkpoint_rejected(self):
        complete, epochs = fitted_metadata(self.job)
        with self.assertRaises(ValueError):
            C.validate_fit_metadata(self.job, complete, epochs[:-1])
        for key, value in [('updates', 175), ('unique_train_stimuli', 6610), ('model_sha256', None),
                ('model_path', '/old/brain_final.pt'), ('training_seed', 999)]:
            bad = dict(complete, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_fit_metadata(self.job, bad, epochs)
        bad_epochs = copy.deepcopy(epochs); bad_epochs[4]['full_epoch_rows'] = 8192
        with self.assertRaises(ValueError):
            C.validate_fit_metadata(self.job, complete, bad_epochs)

    def test_full_future_export_is_metadata_only_with_correct_own_affine(self):
        fit, epochs = fitted_metadata(self.job); record = exported_metadata(self.job, fit)
        result = C.validate_export_metadata(self.job, fit, epochs, record)
        self.assertEqual(result['status'], C.PAYLOAD_PENDING)
        self.assertFalse(result['condition_tensor_payload_verified'])
        self.assertEqual(result['actual_new_OOF_condition_tensors_certified'], 0)

    def test_incomplete_wrong_fold_duplicate_cache_rows_and_parent_affine_rejected(self):
        fit, epochs = fitted_metadata(self.job)
        for key, value in [('array_rows', self.job['expected_export_rows'][:-1]),
                ('array_rows', self.job['expected_export_rows'][:-1] + [self.job['expected_export_rows'][0]]),
                ('array_rows', self.job['registration']['training_array_rows']),
                ('condition_shape', [6610, 1024]), ('normalization', 'L2 unit vector'),
                ('all_condition_values_finite', False), ('affine_source_checkpoint_sha256', '5' * 64),
                ('checkpoint_sha256', '6' * 64), ('official_test_inputs_loaded', True),
                ('official_retrieval_computed', True), ('target_image_or_concept_used_for_export', True),
                ('EEG_input_sha256', '8' * 64), ('physical_repeat_count', 80), ('model_training_mode', True),
                ('gradients_enabled', True), ('all_model_parameters_frozen', False)]:
            record = exported_metadata(self.job, fit); record[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_export_metadata(self.job, fit, epochs, record)

    def test_export_cannot_bypass_incomplete_or_different_fitted_parent(self):
        fit, epochs = fitted_metadata(self.job); record = exported_metadata(self.job, fit)
        for key, value in [('updates', 175), ('role', 'published'), ('registration_core_sha256', '7' * 64),
                ('model_sha256', None), ('model_path', '/wrong/brain_final.pt')]:
            bad = dict(fit, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.validate_export_metadata(self.job, bad, epochs, record)
        with self.assertRaises(ValueError):
            C.validate_export_metadata(self.job, fit, epochs[:-1], record)

    def test_actual_published_metadata_is_only_eligibility_for_ten_models(self):
        result = C.published_metadata_eligibility(self.published, self.parent_proof)
        self.assertEqual(result['candidate_models'], 10)
        self.assertEqual(result['candidate_subjects'], list(C.P.SUBJECTS))
        self.assertFalse(result['may_replace_new_official_OOF'])
        self.assertFalse(result['all_current_checkpoint_payloads_verified'])
        self.assertFalse(result['model_reuse_admitted'])
        self.assertFalse(result['benchmark_readout_admitted'])

    def test_published_partial_T_or_older_teacher_is_not_eligible(self):
        for key, value in [('training_array_rows', list(range(13230))), ('updates_per_subject', 325),
                ('frozen_visual_training_dropout', 0.), ('deterministic_F_cache_used_as_training_target', True)]:
            reg = copy.deepcopy(self.published); reg[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                C.published_metadata_eligibility(reg, self.parent_proof)
        proof = copy.deepcopy(self.parent_proof)
        proof['models'] = [r for r in proof['models'] if r['subject'] != 'sub-10']
        with self.assertRaises(ValueError):
            C.published_metadata_eligibility(self.published, proof)


if __name__ == '__main__':
    checker_main()
