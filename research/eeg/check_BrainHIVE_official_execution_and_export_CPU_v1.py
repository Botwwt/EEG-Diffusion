"""Real task METADATA plus tiny injected CPU tensors, never official execution."""
import argparse
import copy
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.dont_write_bytecode = True
import brainhive_official_execution_contract_CPU_v1 as G
import export_BrainHIVE_official_OOF_conditions_v1 as X

ROOT = Path(__file__).resolve().parent


def reject(call, text):
    try: call()
    except (ValueError, KeyError, RuntimeError) as error:
        assert text in str(error), str(error)
    else: raise AssertionError('Invalid state accepted: ' + text)


def check(plan_directory=None):
    started = time.monotonic(); checks = []
    assert 'torch' not in sys.modules and 'brainhive_native_models_v1' not in sys.modules
    with tempfile.TemporaryDirectory(prefix='HIVE_official_export_CPU_') as temporary:
        directory = Path(temporary); missing = directory / 'NO_REVIEW_EXISTS.json'
        def forbidden(*args, **kwargs): raise AssertionError('Plan/payload accessed before admission')
        with patch.object(G.E, 'read_plan_bundle', side_effect=forbidden):
            reject(lambda: G.preflight(directory / 'not_read_plan', 'sub-01/official_oof0', missing,
                action='export', entry_name=X.__file__.split(os.sep)[-1]), 'review does not exist')
            reject(lambda: G.preflight(directory / 'not_read_plan', 'sub-01/official_oof0', missing,
                action='fit', entry_name='train_BrainHIVE_official_encoder_v1.py'), 'review does not exist')
            fake = directory / 'NOT_AN_ADMISSION.json'
            fake.write_text(json.dumps(dict(schema='BrainHIVE_official_fullfit_metadata_plan_v1', method_locked=True, fitting_admitted=True)))
            reject(lambda: G.preflight(directory, 'sub-01/official_oof0', fake, action='export', entry_name=Path(X.__file__).name), 'metadata plan is not admission')
        reject(lambda: G.verify_payload_files(dict(_execution_token=True)), 'real successful CPU preflight')
        reject(lambda: X.execute(dict(_execution_token=True)), 'real successful CPU preflight')
        assert 'torch' not in sys.modules
        checks.append('missing_review_or_handset_plan_Booleans_reject_before_plan_Torch_payload_or_output_and_no_JSON_context_token')
        # A fresh interpreter refuses even if importing Torch/native would fail.
        blocker = directory / 'sitecustomize.py'
        blocker.write_text("import sys\nclass Block:\n def find_spec(self,name,path=None,target=None):\n  if name=='torch' or name.startswith('torch.') or name=='brainhive_native_models_v1':\n   raise AssertionError('FORBIDDEN_MODEL_IMPORT_BEFORE_GATE')\nsys.meta_path.insert(0,Block())\n")
        env = dict(os.environ, PYTHONPATH=str(directory), CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1')
        outputs = []
        for entry in ('export_BrainHIVE_official_OOF_conditions_v1.py', 'train_BrainHIVE_official_encoder_v1.py'):
            result = subprocess.run([sys.executable, '-B', str(ROOT / entry), '--plan-directory', str(directory / 'not_read_plan'),
                '--task-id', 'sub-01/official_oof0', '--execution-review', str(missing)], env=env,
                capture_output=True, text=True, timeout=30)
            assert result.returncode != 0 and 'review does not exist' in result.stderr and 'FORBIDDEN_MODEL_IMPORT' not in result.stderr
            outputs.append(dict(entry=entry, exit_code=result.returncode, official_missing_review_rejected=True))
        assert not (directory / 'not_read_plan').exists()
        checks.append('both_real_CLIs_missing_review_refused_in_fresh_model_import_blocked_interpreters_no_training_output')
        # A sealed forecast must aggregate each actual filesystem exactly once.
        # Exercise real stat/device identities, narrow-inject only available bytes.
        volume_a = directory / 'volume_a'; volume_b = directory / 'volume_b'
        volume_a.mkdir(); volume_b.mkdir()
        reservation_a = volume_a / 'retained_reservation.json'; reservation_b = volume_b / 'retained_reservation.json'
        reservation_a.write_text('SYNTHETIC_CPU_ONLY_NOT_A_RESOURCE_REVIEW\n')
        reservation_b.write_text('SYNTHETIC_CPU_ONLY_NOT_A_RESOURCE_REVIEW\n')
        forecast = [dict(root=str(p), reservation=G.ref(r), remaining_required_bytes=75) for p, r in
            ((volume_a, reservation_a), (volume_b, reservation_b))]
        assert volume_a.stat().st_dev == volume_b.stat().st_dev
        usage_type = type(G.shutil.disk_usage(directory))
        with patch.object(G.shutil, 'disk_usage', return_value=usage_type(total=100, used=0, free=100)):
            assert G.capacity_forecast(forecast[:1])[0]['required'] == 75
            reject(lambda: G.capacity_forecast(forecast), 'one aggregated whole-lifecycle forecast')
            aggregated = [dict(forecast[0], remaining_required_bytes=150)]
            reject(lambda: G.capacity_forecast(aggregated), 'no longer fits')
        assert 'brainhive_functional_training_contract_v2.py' in G.CORE_SOURCES
        checks.append('actual_same_device_forecasts_cannot_spend_free_bytes_twice_and_resource_headroom_dependency_is_mandatory_source_pin')
        # Existing twenty-task bundle is actual pinned metadata, not fitted models.
        plan_directory = Path(plan_directory) if plan_directory else ROOT / 'audit/official_fullfit_metadata_plan_20261022_v1'
        bundle = G.E.read_plan_bundle(plan_directory)
        jobs = [G.E.job_for_task(bundle, task['task_id']) for task in bundle['tasks']['tasks']]
        assert len(jobs) == len({j['output_root'] for j in jobs}) == 20
        for subject in G.E.P.SUBJECTS:
            own = [j for j in jobs if j['subject'] == subject]
            assert len(own) == 2 and all(len(j['expected_export_rows']) == 8270 for j in own)
            assert set(own[0]['expected_export_rows']).isdisjoint(own[1]['expected_export_rows'])
            assert set(own[0]['expected_export_rows']) | set(own[1]['expected_export_rows']) == set(range(16540))
        checks.append('actual_pinned_twenty_tasks_all10_two8270_held_sets_cover_full16540_without_old_T_cache_reuse_metadata_only')
        import torch
        torch.set_num_threads(2)
        # Follow the exporter's actual scanner integration, not a fake scanner
        # result: default official counts bind the real current engine signature.
        import brainhive_official_encoder_training_v1 as T
        import check_BrainHIVE_official_encoder_training_CPU_v1 as K
        transactions = directory / 'TINY_NATIVE_3ROWS_2EPOCHS_NOT_OFFICIAL'; transactions.mkdir()
        raw_registration = dict(status='SYNTHETIC_CPU_ONLY_NOT_AN_OFFICIAL_FIT', rows=3, epochs=2, updates=2)
        registration_path = transactions / 'registration.json'
        T.write_once(registration_path, T.json_bytes(raw_registration))
        registration = (T.sha(registration_path), G.E.core_sha(raw_registration), G.sha(__file__))
        transaction_job = dict(output_root=str(transactions), registration_core_sha256=registration[1],
            subject='SYNTHETIC_CPU_ONLY', role='SYNTHETIC_NATIVE_8D_NO_OFFICIAL_FIT')
        transaction_fit = dict(registration_sha256=registration[0], producer_source_sha256=registration[2])
        actual_scanner = T.scan_committed_epochs
        with patch.object(T, 'scan_committed_epochs', wraps=actual_scanner) as observed:
            assert X.scan_training_epochs(transaction_job, transaction_fit) == ([], [])
            inspect.signature(actual_scanner).bind(*observed.call_args.args, **observed.call_args.kwargs)
            assert observed.call_args.kwargs['full_epoch_rows'] == 8270
            assert observed.call_args.kwargs['updates_per_epoch'] == 9 and observed.call_args.kwargs['total_epochs'] == 25
        tiny_training = K.fixture(torch, samples=3, batch=3, epochs=2)
        updates = 0
        for epoch in (1, 2):
            row, updates = K.one_epoch(torch, tiny_training, epoch, updates)
            K.save_epoch(torch, transactions, tiny_training, row, updates, registration)
        scan_fixture = dict(updates_per_epoch=1, total_epochs=2, full_epoch_rows=3)
        committed, pending = X.scan_training_epochs(transaction_job, transaction_fit, _fixture=scan_fixture)
        assert len(committed) == 2 and not pending and updates == 2
        assert [item['row']['full_epoch_rows'] for item in committed] == [3, 3]
        assert committed[-1]['commit']['resume_sha256'] == T.sha(Path(committed[-1]['directory']) / 'resume.pt')
        state = torch.load(Path(committed[-1]['directory']) / 'resume.pt', weights_only=True, map_location='cpu')
        assert state['epoch'] == 2 and state['updates'] == 2 and state['producer_source_sha256'] == registration[2]
        reject(lambda: X.scan_training_epochs(transaction_job, transaction_fit,
            _fixture=dict(scan_fixture, full_epoch_rows=4)), 'Incomplete/nonfinite actual epoch log')
        reject(lambda: X.scan_training_epochs(transaction_job, dict(transaction_fit, producer_source_sha256='0' * 64),
            _fixture=scan_fixture), 'different registration or producer')
        reject(lambda: X.scan_training_epochs(transaction_job, transaction_fit,
            _fixture=dict(scan_fixture, total_epochs=3)), 'Wrong epoch producer identity/budget')
        checkpoint_path = Path(committed[-1]['directory']) / 'resume.pt'
        with checkpoint_path.open('ab') as stream: stream.write(b'INTENTIONAL_TINY_CORRUPTION')
        reject(lambda: X.scan_training_epochs(transaction_job, transaction_fit, _fixture=scan_fixture), 'Changed epoch state or real log')
        checks.append('export_actual_engine_signature_official_25_9_8270_and_two_real_tiny_native_InfoNCE_epoch_snapshot_publish_scan_transactions_with_row_source_budget_payload_negatives')
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.proj = torch.nn.Linear(4, 4)
                self.block = torch.nn.Module(); self.block.layers = torch.nn.ModuleList([torch.nn.Module()])
                self.block.layers[0].norm = torch.nn.LayerNorm(4)
                with torch.no_grad():
                    self.block.layers[0].norm.weight.copy_(torch.tensor([1., 2., 3., 4.]))
                    self.block.layers[0].norm.bias.copy_(torch.tensor([.2, .3, .4, .5]))
            def forward(self, value): return self.block.layers[0].norm(self.proj(value.flatten(1)))
        torch.manual_seed(1729); model = Tiny()
        eeg = torch.arange(8 * 3 * 2, dtype=torch.float32).reshape(8, 3, 2) / 10
        fixture = dict(EEG_shape=[8, 3, 2], selected_indices=[0, 2], width=4, rows=3)
        job = dict(task_id='SYNTHETIC/sub-01/official_oof1', subject='sub-01', role='official_oof1', expected_export_rows=[7, 1, 5],
            registration_core_sha256='1' * 64, registration=dict(EEG_input=dict(sha256='2' * 64)))
        fit = dict(model_sha256='3' * 64, registration_sha256='4' * 64, producer_source_sha256='5' * 64)
        tensors, mode = X.infer_held_conditions(model, eeg, job, device='cpu', batch_size=2, _fixture=fixture)
        with torch.inference_mode(): expected = model(eeg[job['expected_export_rows']][:, [0, 2], :])
        # Different CPU GEMM batch shapes may differ in the last float32 bits;
        # compare to an independently batched full reference with fixed limits.
        torch.testing.assert_close(tensors['condition'], expected, rtol=1e-6, atol=2e-6)
        assert tensors['array_rows'].tolist() == [7, 1, 5]
        assert not torch.allclose(tensors['condition'].norm(dim=1), torch.ones(3))
        assert mode['model_training_mode'] is mode['gradients_enabled'] is False and mode['all_model_parameters_frozen'] is True
        assert mode['actual_full8270_export'] is False
        cache = X.make_cache(tensors, job, fit, '6' * 64)
        verified = X.validate_cache_tensors(cache, model.state_dict(), job, fit, '6' * 64, _fixture=fixture)
        assert verified['actual_full8270_verified'] is False and verified['shape'] == [3, 4]
        checks.append('tiny_actual_Torch_global_held_order_raw_float32_eval_inference_no_L2_and_own_finalLN_affine')
        saved = directory / 'TINY_NOT_OFFICIAL_conditions.pt'; torch.save(cache, saved)
        loaded = torch.load(saved, weights_only=True, map_location='cpu')
        assert X.validate_cache_tensors(loaded, model.state_dict(), job, fit, '6' * 64, _fixture=fixture) == verified
        checks.append('tiny_actual_saved_tensor_roundtrip_full_values_rows_and_own_affine_hashes_no_official8270_claim')
        for change, expected_error in (('rows', 'array_rows/order'), ('affine', 'own final-LN'), ('model', 'model/source chain'),
                ('exporter', 'model/source chain'), ('normalization', 'model/source chain'), ('dtype', 'shape/dtype'), ('NaN', 'Nonfinite')):
            wrong = copy.deepcopy(cache)
            if change == 'rows': wrong['array_rows'] = wrong['array_rows'].flip(0)
            elif change == 'affine': wrong['affine_weight'][0] += 1
            elif change == 'model': wrong['model_sha256'] = '0' * 64
            elif change == 'exporter': wrong['exporter_source_sha256'] = '0' * 64
            elif change == 'normalization': wrong['additional_L2_normalization'] = True
            elif change == 'dtype': wrong['condition'] = wrong['condition'].double()
            else: wrong['condition'][0, 0] = float('nan')
            reject(lambda wrong=wrong: X.validate_cache_tensors(wrong, model.state_dict(), job, fit, '6' * 64, _fixture=fixture), expected_error)
        for role in ('oof0', 'full', 'published'):
            wrong_job = dict(job, role=role)
            reject(lambda wrong_job=wrong_job: X.infer_held_conditions(model, eeg, wrong_job, _fixture=fixture), 'new official OOF')
        wrong_job = dict(job, expected_export_rows=[0, 1, 2, 3, 4, 5, 6, 7])
        reject(lambda: X.infer_held_conditions(model, eeg, wrong_job, _fixture=fixture), 'ordered held global')
        checks.append('wrong_cache_row_order_affine_checkpoint_exporter_normalization_dtype_NaN_and_old_T_full_roles_refused')
        state = dict(model=copy.deepcopy(model.state_dict()), logit_scale=torch.tensor(2.65), seed=1729, epochs=25, updates=225,
            registration_sha256=fit['registration_sha256'], registration_core_sha256=job['registration_core_sha256'],
            producer_source_sha256=fit['producer_source_sha256'])
        X.validate_checkpoint(state, job, fit)
        for key in ('updates', 'epochs', 'registration_core_sha256', 'producer_source_sha256'):
            wrong = copy.deepcopy(state); wrong[key] = 1 if key in ('updates', 'epochs') else '0' * 64
            reject(lambda wrong=wrong: X.validate_checkpoint(wrong, job, fit), 'chain differs')
        bad_state = copy.deepcopy(state['model']); bad_state['unexpected'] = torch.ones(1)
        reject(lambda: model.load_state_dict(bad_state, strict=True), 'Unexpected key')
        checks.append('actual_Torch_strict_state_load_and_checkpoint_fixed_epoch_update_registration_source_chain_negative_cases')
        visuals = dict(a=torch.ones(8, 2), b=torch.ones(8, 1, 2))
        G.validate_training_tensors(None, eeg, visuals, dict(a=2, b=2), _fixture=fixture)
        reject(lambda: G.validate_training_tensors(None, eeg.double(), _fixture=fixture), 'shape/dtype')
        reject(lambda: G.validate_training_tensors(None, eeg, dict(a=torch.ones(8, 2)), dict(a=2, b=2), _fixture=fixture), 'embedding keys')
        reject(lambda: G.validate_training_tensors(None, eeg, dict(a=torch.ones(7, 2)), dict(a=2), _fixture=fixture), 'source shape/dtype')
        checks.append('shared_payload_shape_dtype_full_row_visual_key_dimension_finite_checks_with_tiny_injection_only')
        assert not torch.cuda.is_initialized()
        tiny_bytes = saved.stat().st_size
    seconds = time.monotonic() - started; assert seconds < 180
    return dict(status='PASS_REAL_TASK_METADATA_AND_TINY_CPU_TENSORS_ONLY_NO_OFFICIAL_ADMISSION', checks=checks, seconds=seconds,
        source_sha256={Path(module.__file__).name: G.sha(module.__file__) for module in (G, X, T, K)} | {Path(__file__).name: G.sha(__file__)},
        original_encoder_contract_sha256=G.sha(G.E.__file__), actual_plan_preparation=bundle['preparation'],
        real_task_metadata_count=20, actual_plan_directory=str(plan_directory), planned_OOF_queries=165400, actual_new_fits_or_official_condition_caches=0,
        actual_tiny_native_training_transactions=dict(rows=3, epochs=2, parameter_updates=2, width=8,
            actual_publish_epoch_and_scanner=True, production_scanner_default_full_epoch_rows=8270,
            official_fit_or_full8270_commit_chain_verified=False),
        tiny_fixture=dict(EEG_shape=[8, 3, 2], global_held_array_rows=[7, 1, 5], condition_shape=[3, 4], saved_bytes=tiny_bytes,
            actual_full8270_export=False, parameter_updates=0, actual_own_affine_and_tensor_roundtrip=True),
        actual_entry_missing_review_refusals=outputs, method_lock_or_admission_created=False,
        actual_EEG_visual_model_payloads_read=False, actual_GPU_used=False, official_test_or_new_V_scores_read=False,
        limits=['Tensor/model values are small synthetic injections; no real8270 cache or real16540 EEG was loaded.',
            'Public preflight has NO positive fixture bypass and no successful official review exists.',
            'Source/method/calibration/equal baseline/whole resource review and the real exclusive wrapper remain mandatory before execution.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--receipt', type=Path)
    parser.add_argument('--plan-directory', type=Path, help='Existing pinned twenty-task METADATA bundle, no actual model/input payloads')
    args = parser.parse_args()
    if args.receipt: assert not args.receipt.exists()
    report = check(args.plan_directory); raw = (json.dumps(report, indent=2, allow_nan=False) + '\n').encode(); assert len(raw) < 64 << 10
    if args.receipt:
        with args.receipt.open('xb') as stream: stream.write(raw)
    print(raw.decode(), end='')


if __name__ == '__main__': main()
