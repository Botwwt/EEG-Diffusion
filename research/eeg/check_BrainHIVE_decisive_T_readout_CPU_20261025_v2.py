"""Check consequential fixed-readout branches and full-denominator rejection."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import read_BrainHIVE_decisive_T_analysis_CPU_20261025_v2 as R


class ReadoutChecks(unittest.TestCase):
    def test_unknown_is_not_truthy_pass(self):
        value = R.choose('UNKNOWN', {})
        self.assertEqual(value['priority_hypothesis'], 'FUNCTIONAL_TEACHER_EVIDENCE_UNRESOLVED')
        self.assertIs(value['automatic_new_fit'], False)

    def test_optimization_anomaly_has_fixed_priority(self):
        value = R.choose('SUPPORTED', {}, {}, {'optimization_stability': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'RESOLVE_REGISTERED_OPTIMIZATION_ANOMALY_BEFORE_NEW_MECHANISMS')

    def test_fixed_conditional_probe_priority(self):
        value = R.choose('SUPPORTED', {'trajectory_transfer': 'NOT_SUPPORTED'},
                         {'beyond_preLN_statistics': 'SUPPORTED'})
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
        value = R.choose('UNKNOWN', {'teacher_discrimination': 'SUPPORTED', 'trajectory_transfer': 'NOT_SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'INVESTIGATE_TRAINING_STATE_VERSUS_SAMPLING_STATE_MISMATCH')
        self.assertIs(value['causal_mechanism_confirmed'], False)

    def test_conflict_requires_saturation_together(self):
        a = R.choose('UNKNOWN', {'teacher_discrimination': 'SUPPORTED', 'gradient_conflict': 'SUPPORTED'})
        b = R.choose('UNKNOWN', {'teacher_discrimination': 'SUPPORTED', 'gradient_conflict': 'SUPPORTED', 'radius_saturation': 'SUPPORTED'})
        self.assertEqual(a['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')
        self.assertEqual(b['priority_hypothesis'], 'TEST_OBJECTIVE_COMPETITION_WITH_A_FIXED_TRUST_REGION')

    def test_proxy_requires_all_controls(self):
        full = {key: 'SUPPORTED' for key in
                ('teacher_discrimination', 'paired_EEG_signal', 'PN_over_O', 'PN_over_P', 'shuffled_response_control', 'trajectory_transfer')}
        self.assertEqual(R.choose('NOT_SUPPORTED', full)['priority_hypothesis'],
                         'VALIDATE_WHETHER_FUNCTIONAL_PROXY_TRANSFERS_TO_IMAGE_SEMANTICS')
        full['shuffled_response_control'] = 'UNKNOWN'
        self.assertEqual(R.choose('NOT_SUPPORTED', full)['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')

    def test_teacher_unknown_cannot_pass_proxy_or_conflict_gates(self):
        for teacher in ('UNKNOWN', 'NOT_SUPPORTED'):
            values = {key: 'SUPPORTED' for key in ('paired_EEG_signal', 'PN_over_O', 'PN_over_P',
                'shuffled_response_control', 'trajectory_transfer', 'gradient_conflict', 'radius_saturation')}
            values['teacher_discrimination'] = teacher
            self.assertIn(R.choose('NOT_SUPPORTED', values)['priority_hypothesis'],
                ('FUNCTIONAL_TEACHER_EVIDENCE_UNRESOLVED', 'VALIDATE_TEACHER_PROBE_BEFORE_ANY_NEW_FUNCTIONAL_FIT'))

    def test_no_paired_gain_cannot_be_proxy_pass(self):
        values = {key: 'SUPPORTED' for key in ('teacher_discrimination', 'PN_over_O', 'PN_over_P',
                                              'shuffled_response_control', 'trajectory_transfer')}
        values['paired_EEG_signal'] = 'NOT_SUPPORTED'
        self.assertEqual(R.choose('NOT_SUPPORTED', values)['priority_hypothesis'],
                         'INVESTIGATE_COMMON_RESPONSE_SHIFT_BEFORE_MORE_FUNCTIONAL_OBJECTIVES')
        values['paired_EEG_signal'] = 'UNKNOWN'
        self.assertEqual(R.choose('NOT_SUPPORTED', values)['priority_hypothesis'], 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED')

    def test_incomplete_layer_localization_is_not_h0_recommendation(self):
        self.assertEqual(R.choose('SUPPORTED', {})['priority_hypothesis'], 'LAYER_READOUT_GAIN_NOT_LOCALIZED_BY_CONTROLS')

    def test_cpu_support_does_not_depend_on_teacher(self):
        value = R.choose('SUPPORTED', {'teacher_discrimination': 'UNKNOWN'},
                         {'beyond_preLN_statistics': 'SUPPORTED'})
        self.assertEqual(value['priority_hypothesis'], 'TEST_A_MINIMAL_EEG_INTERMEDIATE_READOUT_HYPOTHESIS_NEXT')

    def test_old_additive_control_cannot_be_read_as_complete_statistics(self):
        payload = {'status': R.EXPECTED['layers'], 'schema':'BrainHIVE_decisive_EEG_layers_T_CPU_v2',
            'queries_per_lambda':132300, 'no_V':True, 'no_D':True, 'no_test':True,
            'GPU_used':False, 'new_EEG_encoder_fit':False, 'branches':{
                'beyond_preLN_statistics':{'required_contrasts':['c_h0_minus_c_prestats']},
                'preLN_statistics':{'required_contrasts':['c_prestats_minus_c_cstats']}}}
        with self.assertRaises(AssertionError): R.validate_summary('layers', payload, R.EXPECTED['layers'])

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

    def test_real_CLI_interoperates_with_v2_summary_contract(self):
        # Synthetic closure metadata exercises the actual CLI and file SHA chain.
        # It is not an assertion that any real 132300/640-row analysis has run.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {
                'logs': {'status': R.EXPECTED['logs'], 'no_optimizer_created': True,
                    'parameter_updates': 0, 'protected_scores_read': False, 'generated_images': 0,
                    'diagnostic_decisions': {'optimization_stability': {'result': 'SUPPORTED'}}},
                'layers': {'status': R.EXPECTED['layers'], 'schema': 'BrainHIVE_decisive_EEG_layers_T_CPU_v2',
                    'queries_per_lambda': 132300, 'no_V': True, 'no_D': True, 'no_test': True,
                    'GPU_used': False, 'new_EEG_encoder_fit': False, 'main_decision': {'status': 'SUPPORTED'},
                    'branches': {'beyond_preLN_statistics': {'status': 'SUPPORTED', 'required_contrasts': ['c_h0_minus_c_hpre']},
                        'preLN_statistics': {'status': 'NOT_SUPPORTED', 'required_contrasts': ['c_hpre_minus_c_c']}}},
                'function': {'status': R.EXPECTED['function'], 'coverage': {
                    'expected_queries': 80, 'expected_timesteps_per_domain': 4,
                    'expected_domains': ['DDPM_real_T', 'native_parent_trajectory'],
                    'observed_query_timestep_domain_cells': 640, 'expected_native_IP_layers': 70,
                    'missing_cells': 0}, 'no_optimizer_created': True, 'parameter_updates': 0,
                    'protected_scores_read': False, 'generated_images': 0,
                    'diagnostic_decisions': {'teacher_discrimination': {'result': 'UNKNOWN'}}}}
            summaries = {}
            for kind, payload in payloads.items():
                path = root / (kind + '.json')
                path.write_text(json.dumps(payload), encoding='utf-8')
                summaries[kind] = {'path': str(path), 'status': R.EXPECTED[kind]}
            registration = root / 'registration.json'
            registration.write_text(json.dumps({'schema': R.SCHEMA, 'analysis_source_sha256': R.sha(R.__file__),
                'automatic_new_fit': False, 'output_directory': str(root / 'out'), 'summaries': summaries}), encoding='utf-8')
            process = subprocess.run([sys.executable, '-B', R.__file__, '--registration', str(registration)],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8', timeout=10)
            self.assertEqual(process.returncode, 0, process.stderr)
            output = json.loads((root / 'out/direction_summary.json').read_text(encoding='utf-8'))
            self.assertEqual(output['decision']['priority_hypothesis'], 'TEST_A_MINIMAL_EEG_INTERMEDIATE_READOUT_HYPOTHESIS_NEXT')
            self.assertEqual(output['inputs']['layers']['sha256'], R.sha(root / 'layers.json'))
            self.assertIs(output['automatic_new_fit'], False)


if __name__ == '__main__':
    unittest.main(verbosity=2)
