"""Bind one frozen T32 diagnostic to existing, actually present inputs."""
import hashlib
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    from diagnose_BrainHIVE_native_function_T32_v1 import REQUIRED_SOURCES, check_registered_input_contract
    main_recipe = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
    assert sha(main_recipe) == '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
    previous = json.loads((WORK / 'audit/two_ideas_queue_v1.json').read_text())
    assert previous['status'] == 'WITHDRAWN_BEFORE_NEW_TRAINING_PENDING_MECHANISM_REVIEW'
    assert previous['automatic_launch_enabled'] is False
    inputs = [WORK / 'images/oracle_Tdiag32_v1/inference_inputs.json',
              WORK / 'audit/component_T_latent_bank_v2.json',
              WORK / 'data/component_input_contract_v1.json', ROOT / 'configs/split_manifest.json',
              ROOT / 'audit/image_manifest.csv', WORK / 'audit/native_asset_verification.json']
    assert all(path.is_file() for path in inputs)
    registration = dict(
        diagnostic='native_function_T32_v1', subject='sub-01', queries=32, seed=1729, batch_size=4,
        output_directory=str(WORK / 'audit/native_function_T32_v1'),
        oracle_inference_inputs=str(inputs[0]), latent_receipt=str(inputs[1]),
        source_sha256={name: sha(ROOT / name) for name in REQUIRED_SOURCES},
        input_sha256={str(path): sha(path) for path in inputs},
        queue_source_sha256={name: sha(ROOT / name) for name in
                             ['queue_BrainHIVE_native_function_T32_v1.py', 'run_gpu_job_v2.py']},
        diagnostic_job_seconds=1200, maximum_queue_wait_seconds=72 * 3600,
        component_training_enabled=False, KAN_enabled=False, optimizer_created=False,
        generated_images=0, new_component_candidate_configurations=0,
        existing_main_recipe_sha256=sha(main_recipe),
        previous_two_candidate_registration_preserved=True,
        registration_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        purpose='Locate native stimulus-specific functional response differences before deciding on a trainable modification.',
        Q_reference='One missing-zero-raw-condition pass for all arms at fixed query/t/noise.',
        functional_scope='DDPM-noised real T latents at native four scheduler time indices, not a sampled generation trajectory.',
        arms=['raw_OOF_EEG', 'permuted_different_concept_OOF_EEG', 'correct_F',
              'wrong_different_concept_F', 'missing_zero_raw_condition'],
        image_gain_claimed=False, native_bottleneck_claimed=False,
        no_protected_D_V_test_tensors_or_scores=True,
        follow_up='Review saved measurements; no automatic component training or extra diagnostic search.',
    )
    checked = check_registered_input_contract(registration)
    assert checked['queries'] == 32 and not checked['GPU_executed']
    path = ROOT / 'configs/BrainHIVE_native_function_T32_v1.json'
    assert not path.exists()
    path.write_text(json.dumps(registration, indent=2) + '\n')
    receipt = ROOT / 'audit/BrainHIVE_native_function_T32_registered_input_CPU_v1.json'
    assert not receipt.exists()
    checked.update(registration_path=str(path), registration_sha256=sha(path),
                   no_new_component_training_started=True, existing_main_preserved=True)
    receipt.write_text(json.dumps(checked, indent=2) + '\n')
    print(json.dumps(checked), flush=True)


if __name__ == '__main__':
    main()
