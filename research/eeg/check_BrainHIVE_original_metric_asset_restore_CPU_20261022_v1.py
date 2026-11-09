"""Small CPU-only identity/rejection checks; no network, assets, images or GPU."""
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

import restore_BrainHIVE_original_metric_assets_CPU_20261022_v1 as R


class Response(io.BytesIO):
    def __init__(self, payload, *, size=None, status=200, url='https://download.pytorch.org/models/test.pth'):
        super().__init__(payload)
        self.status, self.url = status, url
        self.headers = {'Content-Length': str(len(payload) if size is None else size)}


class Opener:
    def __init__(self, response=None, error=None):
        self.response, self.error, self.calls = response, error, []

    def open(self, request, timeout):
        self.calls.append((request, timeout))
        if self.error:
            raise self.error
        return self.response


class RestoreChecks(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.target = self.root / 'asset.pth'
        self.partial = self.root / 'asset.attempt1.partial'
        self.payload = b'pinned-small-weight-fixture'
        self.asset = dict(bytes=len(self.payload), sha256=hashlib.sha256(self.payload).hexdigest(),
            url='https://download.pytorch.org/models/test.pth')

    def download(self, opener, **kwargs):
        R.download_once(self.asset, self.target, self.partial, time.monotonic() + 20, _opener=opener, **kwargs)

    def test_exact_download_promotes_after_full_sha(self):
        opener = Opener(Response(self.payload))
        self.download(opener)
        self.assertEqual(len(opener.calls), 1)
        self.assertEqual(self.target.read_bytes(), self.payload)
        self.assertFalse(self.partial.exists())

    def test_short_transfer_preserves_partial_without_promotion(self):
        opener = Opener(Response(self.payload[:-1], size=len(self.payload)))
        with self.assertRaises(AssertionError):
            self.download(opener)
        self.assertEqual(self.partial.read_bytes(), self.payload[:-1])
        self.assertFalse(self.target.exists())
        self.assertEqual(len(opener.calls), 1)

    def test_wrong_sha_preserves_exact_bad_payload(self):
        opener = Opener(Response(b'X' * len(self.payload)))
        with self.assertRaises(AssertionError):
            self.download(opener)
        self.assertEqual(self.partial.read_bytes(), b'X' * len(self.payload))
        self.assertFalse(self.target.exists())
        self.assertEqual(len(opener.calls), 1)

    def test_overlong_transfer_never_promoted(self):
        opener = Opener(Response(self.payload + b'X', size=len(self.payload)))
        with self.assertRaises(AssertionError):
            self.download(opener)
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.target.exists())

    def test_header_size_mismatch_stops_before_payload(self):
        opener = Opener(Response(self.payload, size=len(self.payload) + 1))
        with self.assertRaises(AssertionError):
            self.download(opener)
        self.assertEqual(self.partial.stat().st_size, 0)
        self.assertFalse(self.target.exists())

    def test_non200_and_unapproved_redirect_rejected(self):
        for response in [Response(self.payload, status=206), Response(self.payload, url='https://example.org/a')]:
            with self.subTest(url=response.url, status=response.status):
                partial = self.root / ('part' + str(response.status))
                with self.assertRaises(AssertionError):
                    R.download_once(self.asset, self.target, partial, time.monotonic() + 20, _opener=Opener(response))
                self.assertFalse(self.target.exists())

    def test_network_error_makes_one_attempt_and_preserves_partial(self):
        opener = Opener(error=TimeoutError('fixture timeout'))
        with self.assertRaises(TimeoutError):
            self.download(opener)
        self.assertEqual(len(opener.calls), 1)
        self.assertTrue(self.partial.exists())
        self.assertFalse(self.target.exists())

    def test_existing_target_and_partial_are_never_overwritten(self):
        for path in [self.target, self.partial]:
            path.write_bytes(b'prior-evidence')
            opener = Opener(Response(self.payload))
            with self.assertRaises(AssertionError):
                self.download(opener)
            self.assertEqual(path.read_bytes(), b'prior-evidence')
            self.assertEqual(opener.calls, [])
            path.unlink()  # Test fixtures only, never production inputs.

    def test_expired_deadline_refuses_before_request(self):
        opener = Opener(Response(self.payload))
        with self.assertRaises(AssertionError):
            R.download_once(self.asset, self.target, self.partial, time.monotonic() - 1, _opener=opener)
        self.assertEqual(opener.calls, [])
        self.assertFalse(self.partial.exists())

    def test_old_proof_requires_exact_five_unique_original_identities(self):
        proof = dict(status='ALL5_AUTHOR_METRIC_ASSETS_STRICT_LOAD_VERIFIED_CPU',
            assets=[dict(name=name, **asset) for name, asset in R.ASSETS.items()])
        R.validate_old_receipt(proof)
        bad = copy.deepcopy(proof); bad['assets'][-1]['sha256'] = 'f' * 64
        with self.assertRaises(AssertionError):
            R.validate_old_receipt(bad)
        bad = copy.deepcopy(proof); bad['assets'][-1] = copy.deepcopy(bad['assets'][0])
        with self.assertRaises(AssertionError):
            R.validate_old_receipt(bad)

    def test_durable_path_rejects_parent_escape(self):
        self.assertEqual(R.persistent_path(self.target, self.root), self.target)
        for path in [self.root, self.root.parent / 'outside.pth', self.root / '..' / 'escape.pth']:
            with self.subTest(path=path), self.assertRaises(AssertionError):
                R.persistent_path(path, self.root)

    def test_only_two_original_assets_can_start_download(self):
        self.assertEqual({name for name, row in R.ASSETS.items() if 'url' in row}, {'AlexNet', 'CLIP_L14'})
        self.assertEqual(sum(R.ASSETS[n]['bytes'] for n in ['AlexNet', 'CLIP_L14']), 1177177045)
        self.assertEqual(R.MIN_MEMORY, 12 << 30)
        self.assertEqual(R.MAX_SECONDS, 1800)

    def test_durable_hash_fadvise_success_and_bad_sha(self):
        self.target.write_bytes(self.payload)
        real_path = Path
        def fixture_path(value):
            if str(value) == '/dev/shm':
                return SimpleNamespace(stat=lambda: SimpleNamespace(st_dev=-1))
            return real_path(value)
        with patch.object(R, 'Path', side_effect=fixture_path), \
                patch.object(R.os, 'posix_fadvise', create=True) as advise, \
                patch.object(R.os, 'POSIX_FADV_DONTNEED', 4, create=True):
            value = R.hash_and_advise(self.target, self.asset)
            self.assertTrue(value['clean_pagecache_advised'])
            self.assertEqual(advise.call_count, 1)
            with self.assertRaises(AssertionError):
                R.hash_and_advise(self.target, dict(self.asset, sha256='f' * 64))
            # Even rejection after a full-byte read releases its clean pages.
            self.assertEqual(advise.call_count, 2)

    def test_headroom_does_not_lower_active_fit_guard(self):
        for value in [R.MIN_MEMORY, R.MIN_MEMORY - 1]:
            with self.subTest(value=value), patch.object(R.C, 'resource_headroom', return_value=dict(effective_headroom_bytes=value)):
                if value == R.MIN_MEMORY:
                    R.headroom()
                else:
                    with self.assertRaises(AssertionError):
                        R.headroom()

    def test_gallery_or_formal_image_rows_refuse_before_formal_remaining(self):
        manifest = self.root / 'manifest.jsonl'; manifest.write_text('{"generated_path":"/never/read/image.webp"}\n')
        plan = self.root / 'audit/functional_full_lifecycle_20261022_v1/four_arm_lifecycle_plan.json'
        plan.parent.mkdir(parents=True)
        plan.write_text(json.dumps(dict(arms=[dict(inference_manifest=str(manifest))])))
        with patch.object(R, 'ROOT', self.root), patch.object(R.C, 'formal_remaining') as remaining:
            with self.assertRaisesRegex(AssertionError, 'no result image bytes'):
                R.lifecycle_space(0)
            remaining.assert_not_called()

    def test_effnet_requires_unique_exact_original80_record(self):
        asset = R.ASSETS['EffNet']; durable = self.root / 'effnet.pth'
        row = dict(original_path=asset['path'], durable_path=str(durable), bytes=asset['bytes'], sha256=asset['sha256'])
        value = dict(status='EXACT_ORIGINAL_IDENTITY_COVERAGE_RAM_AND_DURABLE_RUNTIME_CLOSURE_VERIFIED',
            core_original_records=55, evaluator_original_records=25, stages=[dict(files=[row])])
        with patch.object(R, 'SHARED_DURABLE', self.root):
            self.assertEqual(R.effnet_durable(value), durable)
            value['stages'][0]['files'].append(copy.deepcopy(row))
            with self.assertRaises(AssertionError):
                R.effnet_durable(value)

    def test_real_local_original_metadata_and_source_pins(self):
        old = json.loads((R.ROOT / 'audit/BrainHIVE_native_metric_proofs_20261020_v9/original_metric_assets_CPU_v3.json').read_text())
        R.validate_old_receipt(old)
        for name, digest in R.SOURCE_PINS.items():
            self.assertEqual(R.C.sha(R.ROOT / name), digest, name)
        for local, (_, digest) in R.WHEEL_PINS.items():
            self.assertEqual(R.C.sha(R.AUDIT / local), digest, local)


if __name__ == '__main__':
    unittest.main(verbosity=2)
