"""Check consequential fixed-readout branches and full-denominator rejection."""
import copy
import unittest

import read_BrainHIVE_decisive_T_analysis_CPU_20261025_v1 as R


class ReadoutChecks(unittest.TestCase):
    def test_unknown_is_not_truthy_pass(self):
        value = R.choose('UNKNOWN', {})
        self.assertEqual(value['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')
        self.assertIs(value['automatic_new_fit'], False)

    def test_optimization_anomaly_has_fixed_priority(self):
        value = R.choose('SUPPORTED', {}, {}, {'optimization_stability': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'RESOLVE_REGISTERED_OPTIMIZATION_ANOMALY_BEFORE_NEW_MECHANISMS')

    def test_fixed_conditional_probe_priority(self):
        value = R.choose('SUPPORTED', {'trajectory_transfer': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'TEST_A_MINIMAL_EEG_INTERMEDIATE_READOUT_HYPOTHESIS_NEXT')

    def test_pre_norm_stats_explain_joint_gain(self):
        value = R.choose('SUPPORTED', {}, {'preLN_statistics': 'SUPPORTED',
                                          'beyond_preLN_statistics': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'TEST_A_MINIMAL_PRE_LAYERNORM_STATISTICS_BYPASS_NEXT')

    def test_uninformative_teacher_blocks_functional_interpretation(self):
        value = R.choose('NOT_SUPPORTED', {'teacher_discrimination': 'NOT_SUPPORTED',
                                          'trajectory_transfer': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'VALIDATE_TEACHER_PROBE_BEFORE_ANY_NEW_FUNCTIONAL_FIT')

    def test_mismatch_is_a_hypothesis(self):
        value = R.choose('UNKNOWN', {'trajectory_transfer': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'INVESTIGATE_TRAINING_STATE_VERSUS_SAMPLING_STATE_MISMATCH')
        self.assertIs(value['causal_mechanism_confirmed'], False)

    def test_conflict_requires_saturation_together(self):
        a = R.choose('UNKNOWN', {'gradient_conflict': 'SUPPORTED'})
        b = R.choose('UNKNOWN', {'gradient_conflict': 'SUPPORTED', 'radius_saturation': 'SUPPORTED'})
        self.assertEqual(a['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')
        self.assertEqual(b['priority_hypothesis'], 'TEST_OBJECTIVE_COMPETITION_WITH_A_FIXED_TRUST_REGION')

    def test_proxy_requires_all_controls(self):
        full = {key: 'SUPPORTED' for key in
                ('PN_over_O', 'PN_over_P', 'shuffled_response_control', 'trajectory_transfer')}
        self.assertEqual(R.choose('NOT_SUPPORTED', full)['priority_hypothesis'],
                         'VALIDATE_WHETHER_FUNCTIONAL_PROXY_TRANSFERS_TO_IMAGE_SEMANTICS')
        full['shuffled_response_control'] = 'UNKNOWN'
        self.assertEqual(R.choose('NOT_SUPPORTED', full)['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')

    def test_partial_coverage_and_updates_rejected(self):
        valid = {'status': R.EXPECTED['function'], 'coverage': {
            'expected_queries': 80, 'expected_timesteps_per_domain': 4,
            'expected_domains': ['DDPM_real_T', 'native_parent_trajectory'],
            'observed_query_timestep_domain_cells': 640, 'expected_native_IP_layers': 70,
            'missing_cells': 0}, 'no_optimizer_created': True, 'parameter_updates': 0,
            'protected_scores_read': False, 'generated_images': 0}
        R.validate_summary('function', valid, R.EXPECTED['function'])
        for mutation in ('missing', 'update', 'scores', 'short'):
            bad = copy.deepcopy(valid)
            if mutation == 'missing': bad['coverage']['missing_cells'] = 1
            if mutation == 'short': bad['coverage']['observed_query_timestep_domain_cells'] = 639
            if mutation == 'update': bad['parameter_updates'] = 1
            if mutation == 'scores': bad['protected_scores_read'] = True
            with self.assertRaises(AssertionError):
                R.validate_summary('function', bad, R.EXPECTED['function'])

    def test_unknown_strings_cannot_be_cast_to_true(self):
        for invalid in (True, 'PASS', 'true', 'UNREGISTERED'):
            with self.assertRaises(ValueError): R.result(invalid)


if __name__ == '__main__':
    unittest.main(verbosity=2)
