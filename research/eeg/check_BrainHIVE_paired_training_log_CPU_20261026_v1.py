"""CPU schema/regression checks for the read-only paired log analyzer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import analyze_BrainHIVE_paired_training_log_CPU_20261026_v1 as A


def fixture(arm='O'):
    rows = []
    pass_means = [2., 1.5, 1.]
    for update in A.EVENT_UPDATES:
        epoch = (update - 1) // A.UPDATES_PER_PASS
        within = (update - 1) % A.UPDATES_PER_PASS + 1
        local_seen = min(within * 16, 132300)
        seen = epoch * 132300 + local_seen
        total = 132300 * sum(pass_means[:epoch]) + local_seen * pass_means[epoch]
        aux = {'O': 0., 'A': 4.25, 'D': .25, 'U': .8}[arm]
        rows.append(dict(status='PAIRED_EFFECT_FULL_T_TRAINING', arm=arm,
            updates=update, expected_updates=24807, cursor=seen,
            seen_primary_queries=seen, seen_paired_queries=seen,
            expected_primary_exposures=396900, loss_mean=total / seen,
            last_loss=1. + .1 * .5 + .1 * aux, epsilon_loss=1., token_loss=.5,
            auxiliary=aux, terms=dict(A=4.25, D=.25, U=.8),
            diagnostics=dict(valid_fraction=.95, shuffled_valid_fraction=.95,
                teacher_separation=2., shuffled_teacher_separation=3.,
                common_component=4., contrast_component=.25, student_difference_energy=1.),
            gradient_norm=.9, seconds=update * 2., UNet_forwards=2))
    return rows


class LogAuditTests(unittest.TestCase):
    def test_complete_schedule_preserves_snapshot_denominator_and_exact_total_loss(self):
        result = A.analyze_rows(fixture(), 'O')
        self.assertEqual(result['registered_event_count'], 390)
        self.assertEqual(result['all_recorded_events']['observed_batch_exposures'], 6228)
        self.assertAlmostEqual(result['snapshot_exposure_fraction'], 6228 / 396900)
        self.assertEqual(result['first_fixed32_events']['event_count'], 32)
        self.assertEqual(result['last_fixed32_events']['event_count'], 32)
        self.assertEqual(result['exact_all_update_total_objective_mean'], 1.5)
        self.assertEqual([result['per_pass'][str(i)]['all_update_total_objective_mean_from_cumulative_sum']
                          for i in (1, 2, 3)], [2., 1.5, 1.])

    def test_all_four_active_auxiliary_mappings(self):
        for arm in A.ARMS:
            A.validate_rows(fixture(arm), arm)

    def test_missing_duplicate_or_reordered_events_fail(self):
        rows = fixture()
        cases = [rows[:-1], [rows[0]] + rows[2:], [rows[1], rows[0]] + rows[2:]]
        for case in cases:
            with self.assertRaises(AssertionError):
                A.validate_rows(case, 'O')

    def test_pass_end_partial_batch_is_not_sixteen(self):
        rows = fixture()
        row = next(row for row in rows if row['updates'] == 8269)
        row['cursor'] = row['seen_primary_queries'] = row['seen_paired_queries'] = 8269 * 16
        with self.assertRaises(AssertionError):
            A.validate_rows(rows, 'O')

    def test_invalid_mask_decomposition_active_aux_and_loss_fail(self):
        for field in ('mask', 'decomposition', 'aux', 'last_loss'):
            rows = fixture()
            if field == 'mask': rows[0]['diagnostics']['shuffled_valid_fraction'] = .9
            if field == 'decomposition': rows[0]['terms']['A'] = 4.5
            if field == 'aux': rows[0]['auxiliary'] = .25
            if field == 'last_loss': rows[0]['last_loss'] += .1
            with self.assertRaises(AssertionError):
                A.validate_rows(rows, 'O')

    def test_nonfinite_boolean_and_negative_values_fail(self):
        for value in (float('nan'), float('inf'), True, -.1):
            rows = fixture(); rows[0]['gradient_norm'] = value
            with self.assertRaises(AssertionError):
                A.validate_rows(rows, 'O')

    def test_recorded_energy_ratio_is_ratio_of_means_not_claim_of_information(self):
        metrics = A.event_metrics(fixture()[0])
        self.assertEqual(metrics['derived.ratio_of_student_to_teacher_layer_equal_batch_mean_energy'], .5)
        self.assertAlmostEqual(metrics['derived.common_share_of_absolute_objective'], 4. / 4.25)

    def test_undefined_zero_denominator_is_preserved(self):
        row = fixture()[0]
        row['terms']['A'] = row['terms']['D'] = 0.
        row['diagnostics']['common_component'] = row['diagnostics']['contrast_component'] = 0.
        row['diagnostics']['teacher_separation'] = 0.
        metrics = A.event_metrics(row)
        self.assertIsNone(metrics['derived.common_share_of_absolute_objective'])
        self.assertIsNone(metrics['derived.ratio_of_student_to_teacher_layer_equal_batch_mean_energy'])
        self.assertEqual(A.describe([None, None]), dict(defined_events=0))

    def test_cross_arm_teacher_schedule_records_difference(self):
        rows = fixture('O'); other = fixture('D')
        pair = {'O': (rows, {}), 'D': (other, {})}
        self.assertTrue(A.compare_logged_teacher_schedules(pair)['O_vs_D']['exactly_equal'])
        other[1]['diagnostics']['teacher_separation'] += .01
        result = A.compare_logged_teacher_schedules(pair)['O_vs_D']
        self.assertFalse(result['exactly_equal'])
        self.assertAlmostEqual(result['maximum_absolute_difference']['teacher_separation'], .01)

    def test_closed_input_sha_binding_and_calibration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = ('\n'.join(json.dumps(row) for row in fixture()) + '\n').encode()
            provenance = dict(arm='O', head_parameters=525568, primary_query_exposures=396900,
                paired_query_exposures=396900, training_seed=1729, input_role='OOF_T', study_core_sha256='a' * 64)
            source = json.dumps(provenance).encode()
            calibration = json.dumps(dict(strength=1.)).encode()
            done = dict(status='PAIRED_EFFECT_FIXED_FINAL_FULL_T_COMPLETE', arm='O', updates=24807,
                passes=3, primary_exposures=396900, paired_exposures=396900, parameters=525568,
                study_core_sha256='a' * 64, provenance=dict(sha256=A.sha(source)),
                calibration=dict(sha256=A.sha(calibration)), checkpoint=dict(path='/fixture/not/read', sha256='b' * 64))
            for name, raw in [('training.jsonl', payload), ('input_provenance.json', source),
                              ('calibration.json', calibration), ('completion.json', json.dumps(done).encode())]:
                (root / name).write_bytes(raw)
            _, result = A.load_closed('O', root)
            self.assertTrue(result['checkpoint_bytes_not_loaded_or_reverified'])
            (root / 'calibration.json').write_bytes(b'{}')
            with self.assertRaises(AssertionError):
                A.load_closed('O', root)


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(LogAuditTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    print(json.dumps(dict(status='PAIRED_LOG_ANALYZER_CPU_REGRESSIONS_PASSED', checks=result.testsRun,
        GPU_used=False, source_training_modified=False, real_study_data_modified=False)))
