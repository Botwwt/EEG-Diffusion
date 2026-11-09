"""Actual tiny native CPU math/negative permissions; never a GPU profile."""
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
import profile_BrainHIVE_official_encoder_GPU_v1 as P

ROOT = Path(__file__).resolve().parent


def reject(call, message):
    try: call()
    except (ValueError, KeyError, RuntimeError) as error: assert message in str(error), str(error)
    else: raise AssertionError('Invalid state accepted: ' + message)


def check():
    started = time.monotonic(); checks = []
    assert 'torch' not in sys.modules and 'brainhive_native_models_v1' not in sys.modules
    with tempfile.TemporaryDirectory(prefix='HIVE_ENCODER_PROFILE_CPU_ONLY_') as temporary:
        directory = Path(temporary); missing = directory / 'NO_ENGINEERING_PERMISSION.json'
        def forbidden(*_args, **_kwargs): raise AssertionError('Payload/output accessed before engineering permission')
        with patch.object(P, 'run_profile', side_effect=forbidden), patch.object(P.G, 'read', side_effect=forbidden):
            reject(lambda: P.main(['--permission', str(missing)]), 'permission does not exist')
        assert 'torch' not in sys.modules
        reject(lambda: P.run_profile(dict(_token=True)), 'actual engineering permission')
        fake = directory / 'NOT_AN_ENGINEERING_PERMISSION.json'
        fake.write_text(json.dumps(dict(schema='BrainHIVE_official_fullfit_metadata_plan_v1', status='PLANNED', admitted=True)))
        reject(lambda: P.preflight(fake), 'not an engineering permission')
        assert 'torch' not in sys.modules
        checks.append('real_public_main_missing_permission_and_fake_plan_or_JSON_token_reject_before_models_payload_or_output')
        blocker = directory / 'sitecustomize.py'
        blocker.write_text("import sys\nclass Block:\n def find_spec(self,name,path=None,target=None):\n  if name=='torch' or name.startswith('torch.') or name=='brainhive_native_models_v1':\n   raise AssertionError('FORBIDDEN_MODEL_IMPORT_BEFORE_ENGINEERING_PERMISSION')\nsys.meta_path.insert(0,Block())\n")
        env = dict(os.environ, PYTHONPATH=str(directory), PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')
        cli = subprocess.run([sys.executable, '-B', str(ROOT / 'profile_BrainHIVE_official_encoder_GPU_v1.py'),
            '--permission', str(missing)], env=env, capture_output=True, text=True, timeout=30)
        assert cli.returncode != 0 and 'permission does not exist' in cli.stderr and 'FORBIDDEN_MODEL_IMPORT' not in cli.stderr
        checks.append('actual_fresh_CLI_import_blocker_proves_absent_engineering_permission_before_Torch_native')
        protocol = dict(image_rows=dict(T=[7, 1, 5, 2], D=[0, 3], V=[4, 6]))
        rows = P.training_rows(protocol, 3, _fixture=True); assert rows == [7, 1, 5] and rows != list(range(3))
        reject(lambda: P.training_rows(protocol), 'scope is fixed')
        wrong = copy.deepcopy(protocol); wrong['image_rows']['T'][0] = 0
        reject(lambda: P.training_rows(wrong, 3, _fixture=True), 'identity overlap')
        wrong = copy.deepcopy(protocol); wrong['image_rows']['T'][0] = 1
        reject(lambda: P.training_rows(wrong, 3, _fixture=True), 'identity overlap')
        actual_split = P.G.read(ROOT / 'configs/split_manifest.json')
        actual_T_rows = P.training_rows(actual_split)
        assert len(actual_T_rows) == 8270 and set(actual_T_rows) <= set(actual_split['image_rows']['T'])
        checks.append('actual_T_metadata_fixed_first8270_global_ids_and_tiny_nonprefix_T_order_no_D_V_or_duplicate_rows')
        import torch
        import check_BrainHIVE_official_encoder_training_CPU_v1 as K
        import export_BrainHIVE_official_OOF_conditions_v1 as X
        torch.set_num_threads(2)
        generator = torch.Generator(device='cpu').manual_seed(1729)
        raw_eeg = torch.full((8, 3, 3), float('nan'), dtype=torch.float32)
        raw_visuals = {name:torch.full((8, width), float('nan')) for name,width in {'a':4,'b':5,'c':6}.items()}
        raw_eeg[rows] = torch.randn(3, 3, 3, generator=generator)
        for value in raw_visuals.values(): value[rows] = torch.randn(3, value.shape[-1], generator=generator)
        fixture = dict(EEG_shape=[8,3,3], selected_indices=[0,2], rows=3, width=8)
        eeg, visuals, selected = P.select_T_tensors(torch, raw_eeg, raw_visuals, rows, {'a':4,'b':5,'c':6}, _fixture=fixture)
        assert torch.equal(eeg, raw_eeg[rows][:,[0,2],:]) and all(torch.equal(visuals[k],raw_visuals[k][rows]) for k in visuals)
        assert selected['D_or_V_numeric_values_inspected'] is False and torch.isnan(raw_eeg[0]).all()
        wrong = raw_eeg.clone(); wrong[rows[0],0,0] = float('nan')
        reject(lambda: P.select_T_tensors(torch, wrong, raw_visuals, rows, {'a':4,'b':5,'c':6}, _fixture=fixture), 'Nonfinite/degenerate')
        reject(lambda: P.select_T_tensors(torch, raw_eeg.double(), raw_visuals, rows, {'a':4,'b':5,'c':6}, _fixture=fixture), 'metadata shape/dtype')
        reject(lambda: P.select_T_tensors(torch, raw_eeg, raw_visuals, rows, {'a':4,'b':5}, _fixture=fixture), 'visual keys')
        checks.append('actual_tiny_T_selection_visual_EEG_row_alignment_and_nonfinite_D_V_sentinels_not_numerically_read')
        value = K.fixture(torch, samples=3, batch=3, epochs=225)
        initial = P.state_sha(value['model'].state_dict()); teacher_sha = P.state_sha(value['teacher'].state_dict())
        assert all(not p.requires_grad for p in value['teacher'].parameters()) and value['teacher'].training
        assert all(module.p == .1 for module in value['teacher'].modules() if isinstance(module, torch.nn.Dropout))
        row, updates = P.T.run_epoch(torch, value['model'], value['teacher'], value['logit'], value['optimizer'], value['scheduler'],
            value['shuffle'], eeg, visuals, epoch=1, updates=0, total_updates=225, batch_size=3,
            subject='SYNTHETIC_CPU_ONLY', role='TINY_PROFILE_NOT_OFFICIAL', device='cpu', deadline=time.monotonic()+30)
        assert updates == 1 and row['full_epoch_rows'] == 3 and row['grad_norm_last'] > 0 and row['expected_updates'] == 225
        assert P.state_sha(value['model'].state_dict()) != initial and P.state_sha(value['teacher'].state_dict()) == teacher_sha
        assert all(p.grad is None for p in value['teacher'].parameters())
        assert math.isclose(value['optimizer'].param_groups[0]['lr'], 5e-4*.5*(1+math.cos(math.pi/225)), abs_tol=1e-16)
        checks.append('actual_native_8dim_InfoNCE_backward_update_teacher_frozen_stochastic_and_official225_scheduler_horizon_not9')
        snapshot = P.T.snapshot(torch, value['model'], value['logit'], value['optimizer'], value['scheduler'], value['shuffle'],
            1, 1, '1'*64, '2'*64, P.T.sha(__file__), 'cpu')
        transaction_root = directory / 'TINY_ENGINEERING_EPOCH_ONLY'; transaction_root.mkdir()
        commit = P.T.publish_epoch(torch, transaction_root, snapshot, row)
        checkpoint_path = Path(commit['path']).parent / 'resume.pt'
        loaded = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
        assert P.state_sha(loaded['model']) == P.state_sha(value['model'].state_dict())
        assert checkpoint_path.stat().st_size > 0 and not (transaction_root / 'brain_final.pt').exists()
        reject(lambda: P.T.publish_epoch(torch, transaction_root, snapshot, row), 'Never overwrite')
        checks.append('actual_single_tiny_engineering_epoch_snapshot_transaction_reload_no_final_model_and_no_overwrite')
        model = K.native_classes(torch)['BrainMLP'](6,8,num_layers=1,dropout=.1)
        model.load_state_dict(loaded['model'], strict=True)
        conditions, observed = P.export_T_conditions(torch, model, raw_eeg, rows, loaded['model'],
            device='cpu', deadline=time.monotonic()+30, _fixture=fixture)
        job = dict(role='official_oof0', expected_export_rows=rows)
        # The frozen official helper verifies every input value, while this
        # engineering loop must not numerically inspect real D/V rows. Supply
        # finite synthetic unused rows ONLY for this tiny mathematical oracle.
        reference_eeg = torch.nan_to_num(raw_eeg)
        frozen, frozen_mode = X.infer_held_conditions(model, reference_eeg, job, device='cpu', batch_size=256, _fixture=fixture)
        torch.testing.assert_close(conditions, frozen['condition'], rtol=0, atol=0)
        with torch.inference_mode(): direct = model(raw_eeg[rows][:,[0,2],:])
        torch.testing.assert_close(conditions, direct, rtol=0, atol=0)
        assert observed['actual_full8270_GPU_engineering_export'] is False and frozen_mode['actual_full8270_export'] is False
        assert observed['official_OOF_export'] is False and not torch.allclose(conditions.norm(dim=1), torch.ones(3))
        checks.append('new_engineering_export_real_native_math_bitexact_with_frozen_exporter_tiny_path_and_direct_raw_LN_own_affine_no_official_claim')
        for change, text in [('order','global row order'),('affine','own checkpoint affine'),('L2','raw-LN'),('dtype','tensor/global row'),('NaN','tensor/global row')]:
            report = copy.deepcopy(observed); tensor = conditions.clone()
            if change == 'order': report['array_rows'] = report['array_rows'][::-1]
            elif change == 'affine': report['affine_weight_sha256'] = '0'*64
            elif change == 'L2': report['additional_L2_normalization'] = True
            elif change == 'dtype': tensor = tensor.double()
            else: tensor[0,0] = float('nan')
            reject(lambda: P.verify_export_observation(torch,tensor,report,rows,loaded['model'],_fixture=fixture),text)
        assert not torch.cuda.is_initialized()
        checks.append('wrong_export_row_affine_L2_mode_dtype_or_nonfinite_observations_refused')
        # Exercise the same actual publisher used after the full GPU path,
        # without fabricating a positive engineering permission/token.
        receipt_root = directory / 'TINY_RECEIPT_CAP_TEST'; receipt_root.mkdir()
        raw_receipt = b'{"status":"TINY_FIXTURE_NOT_A_GPU_PROFILE"}\n'
        reject(lambda: P.write_engineering_receipt(receipt_root, raw_receipt, 1), 'before publication')
        assert list(receipt_root.iterdir()) == []
        existing = receipt_root / 'existing.bin'; existing.write_bytes(b'x')
        os.link(existing, receipt_root / 'existing_same_inode.bin')
        used = P.directory_allocated(receipt_root)
        assert used == (existing.stat().st_blocks*512 if hasattr(existing.stat(),'st_blocks') else existing.stat().st_size)
        reject(lambda: P.write_engineering_receipt(receipt_root, raw_receipt, used+1), 'before publication')
        assert not (receipt_root / 'engineering_profile.json').exists()
        P.write_engineering_receipt(receipt_root, raw_receipt, used+(1<<20))
        assert (receipt_root / 'engineering_profile.json').read_bytes() == raw_receipt
        reject(lambda: P.write_engineering_receipt(receipt_root, raw_receipt, used+(1<<20)), 'Never overwrite')
        checks.append('actual_receipt_publisher_refuses_tiny_or_existing_usage_cap_before_writing_and_counts_pending_hardlinks_once')
        snapshot_bytes = checkpoint_path.stat().st_size
    seconds = time.monotonic()-started; assert seconds < 180
    sources = {Path(module.__file__).name:P.T.sha(module.__file__) for module in (P,P.G,P.T,K,X)}
    sources[Path(__file__).name] = P.T.sha(__file__)
    return dict(status='PASS_TINY_CPU_ENGINEERING_ONLY_NO_PERMISSION_OR_GPU',checks=checks,seconds=seconds,source_sha256=sources,
        actual_GPU_used=False, actual_engineering_permission_created=False, actual_profile_job_launched=False,
        actual_full8270_export=False, actual_training_container_payloads_read=False, official_test_or_new_V_scores_read=False,
        actual_tiny_native_math=dict(rows=3,width=8,parameter_updates=1,scheduler_horizon=225,
            snapshot_bytes=snapshot_bytes,rawLN_export_shape=[3,8],actual_native_class_source_verified=True),
        full_T_metadata_only_rows=len(actual_T_rows), measured_GPU_timing_or_capacity=None,
        limitations=['CPU tiny math and early rejection are actual; no CUDA timing, BF16/VRAM capacity or real8270 export is measured.',
            'No positive public permission bypass, method lock, official fit/cache, image or score is produced.',
            'A future separately permitted exclusive GPU job and scope compatibility review remain required.'])


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--receipt',type=Path); args=parser.parse_args()
    if args.receipt: assert not args.receipt.exists()
    result=check(); raw=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode(); assert len(raw)<64<<10
    if args.receipt:
        with args.receipt.open('xb') as stream: stream.write(raw)
    print(raw.decode(),end='')


if __name__=='__main__': main()
