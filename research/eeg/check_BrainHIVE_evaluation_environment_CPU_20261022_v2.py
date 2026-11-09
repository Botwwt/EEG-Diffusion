"""CPU identity/rejection checks; imports neither torch nor an evaluator model."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest
import zipfile

import probe_BrainHIVE_evaluation_environment_GPU_20261022_v2 as probe
import restore_BrainHIVE_evaluation_runtime_CPU_20261022_v1 as recovery

ROOT = Path(__file__).resolve().parent


class EnvironmentContract(unittest.TestCase):
    def setUp(self):
        self.protocol = {'image_rows': {'T': [9024], 'D': [7], 'V': [8]}}
        self.registration = {'rows': [{'array_row': 9024, 'stimulus_id': 'moose_05s.jpg'}]}
        self.row = dict(array_row='9024', dataset_split='train', stimulus_id='moose_05s.jpg',
            image_member='training_images/01019_moose/moose_05s.jpg')
        self.job = dict(name='BrainHIVE_evaluation_environment_probe_20261022_v2_attempt1',
            status='RUNNING', pid=123, wrapper_pid=122, max_seconds=900)
        self.expected = [dict(path=f'/dev/shm/evaluator/{i}', bytes=i+1, sha256=f'{i:064x}') for i in range(25)]

    def test_exact_registered_T_join(self):
        path, row = probe.selected_T(self.protocol, self.registration, [self.row], Path('/fixture/images'))
        self.assertEqual(path.as_posix(), '/fixture/images/01019_moose/moose_05s.jpg')
        self.assertIs(row, self.row)

    def test_non_T_and_cross_split_rejected(self):
        for replacement in [dict(T=[], D=[9024], V=[]), dict(T=[9024], D=[], V=[9024])]:
            with self.subTest(replacement=replacement), self.assertRaises(AssertionError):
                probe.selected_T({'image_rows': replacement}, self.registration, [self.row])

    def test_row_missing_duplicate_or_wrong_split_rejected(self):
        for rows in [[], [self.row, self.row], [self.row | dict(dataset_split='test')]]:
            with self.subTest(rows=rows), self.assertRaises(AssertionError):
                probe.selected_T(self.protocol, self.registration, rows)

    def test_changed_query_or_stimulus_rejected(self):
        for registration in [{'rows': [{'array_row': 9025, 'stimulus_id': 'moose_05s.jpg'}]},
            {'rows': [{'array_row': 9024, 'stimulus_id': 'other.jpg'}]}]:
            with self.subTest(registration=registration), self.assertRaises(AssertionError):
                probe.selected_T(self.protocol, registration, [self.row])
        with self.assertRaises(AssertionError):
            probe.selected_T(self.protocol, self.registration, [self.row | dict(stimulus_id='other.jpg')])

    def test_original_member_escape_or_substitution_rejected(self):
        for member in ['/tmp/moose_05s.jpg', 'training_images/../moose_05s.jpg',
                       'training_images/01019_moose/other.jpg']:
            with self.subTest(member=member), self.assertRaises(AssertionError):
                probe.selected_T(self.protocol, self.registration, [self.row | dict(image_member=member)])

    def test_full_asset_identity_accepted(self):
        probe.require_asset_records(self.expected, list(reversed(self.expected)))

    def test_partial_duplicate_or_changed_asset_rejected(self):
        changed = copy.deepcopy(self.expected); changed[0]['sha256'] = 'f' * 64
        wrong_size = copy.deepcopy(self.expected); wrong_size[0]['bytes'] += 1
        for records in [self.expected[:-1], [self.expected[0]] * 25, changed, wrong_size]:
            with self.subTest(records=records[:1]), self.assertRaises(AssertionError):
                probe.require_asset_records(self.expected, records)

    def test_exact_live_finite_wrapper_identity(self):
        probe.require_job_record(self.job, 1, 123, 122)

    def test_stale_wrong_parent_or_wrong_budget_wrapper_rejected(self):
        for change in [dict(status='COMPLETED'), dict(pid=12), dict(wrapper_pid=12),
                       dict(max_seconds=901), dict(name='unrelated')]:
            with self.subTest(change=change), self.assertRaises(AssertionError):
                probe.require_job_record(self.job | change, 1, 123, 122)

    def test_retained_twenty_config_git_blob_identities(self):
        manifests = json.loads((ROOT / 'audit/seed_hf_selected_assets.json').read_text())
        small, large = recovery.inspect_small(manifests, ROOT / 'seed_small_configs.zip')
        self.assertEqual((len(small), len(large)), (20, 3))
        self.assertEqual(sum(a['bytes'] for a in large), 2108180044)

    def test_corrupt_retained_config_is_rejected(self):
        manifests = json.loads((ROOT / 'audit/seed_hf_selected_assets.json').read_text())
        storage = io.BytesIO()
        with zipfile.ZipFile(ROOT / 'seed_small_configs.zip') as source, zipfile.ZipFile(storage, 'w') as target:
            for index, name in enumerate(source.namelist()):
                value = source.read(name)
                if index == 0:
                    value = bytes([value[0] ^ 1]) + value[1:]
                target.writestr(name, value)
        storage.seek(0)
        with self.assertRaises(AssertionError):
            recovery.inspect_small(manifests, storage)

    def test_present_pinned_scientific_sources_unchanged(self):
        # The 16,540-row CSV is server-only. Verify every available fixed source.
        checked = 0
        for name, digest in probe.PINS.items():
            path = ROOT / name
            if path.is_file():
                self.assertEqual(probe.sha(path), digest, name)
                checked += 1
        self.assertGreaterEqual(checked, 6)
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__':
    unittest.main(verbosity=2)
