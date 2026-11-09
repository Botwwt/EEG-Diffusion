"""Reusable pure-CPU contracts for future official-full encoder/export entries.

This module validates concrete per-subject jobs and future producer metadata;
it does not implement training, open a model/EEG/condition tensor, issue an
admission or launch a process. Its metadata verdicts never certify payloads.
The current development recipes and their frozen source files are untouched.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath

import prepare_BrainHIVE_official_fullfit_plan_CPU_v1 as P

STATUS = 'OFFICIAL_ENCODER_JOB_METADATA_CONTRACT_VALID_NO_FIT_NO_ADMISSION'
PAYLOAD_PENDING = 'METADATA_VALID_PAYLOAD_AND_SOURCE_AUDIT_PENDING'
OUTPUT_NAMESPACE = PurePosixPath('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/official_fullfit_v1')
AFFINE_KEYS = ['block.layers.0.norm.weight', 'block.layers.0.norm.bias']
SOURCE_COMMIT = 'f06db69fcc4a029439e369c5c3dba6099e57560e'
OUTPUT_FILES = dict(registration='registration.json', final_checkpoint='brain_final.pt', resume_checkpoint='resume.pt',
    epoch_log='epochs.jsonl', fit_completion='training_complete.json', condition_cache='conditions.pt', export_receipt='export_identity.json')
RESUME_FIELDS = ['model', 'logit_scale', 'optimizer', 'scheduler', 'torch_rng', 'cuda_rng',
    'shuffle_rng', 'epoch', 'updates', 'registration_sha256', 'registration_core_sha256']
PRE_FIT_REQUIREMENTS = ['locked method and hyperparameters', 'equal final-fit baseline opportunity',
    'sealed official-full resource/source admission', 'actual EEG/visual source SHA and shape audit',
    'exclusive existing GPU wrapper; one fixed seed; finite job and exact resume contract']
JOB_KEYS = frozenset(('schema', 'status', 'task_id', 'subject', 'role', 'registration', 'registration_core_sha256',
    'output_root', 'outputs', 'expected_export_rows', 'expected_cache_shape', 'expected_export_dtype',
    'condition_normalization', 'expected_affine_keys', 'expected_affine_shape', 'required_resume_fields',
    'required_before_any_fit', 'method_locked', 'fitting_admitted', 'automatic_launch_enabled',
    'executable_commands', 'future_executor_source_sha256', 'source_payload_verification_complete'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def core_sha(value):
    return hashlib.sha256(P.canonical(value)).hexdigest()


def clean_output_root(value):
    root = PurePosixPath(value)
    require(root.is_absolute() and '..' not in root.parts, 'Absolute canonical future output root required')
    require(root == OUTPUT_NAMESPACE or root.is_relative_to(OUTPUT_NAMESPACE), 'Use a separate official-final namespace')
    return root


def read_plan_bundle(directory):
    """Recompute plan/task metadata and all query identities from pinned inputs."""
    directory = Path(directory)
    proof = json.loads((directory / 'preparation.json').read_text(encoding='utf-8'))
    require(proof['status'] == P.STATUS and proof['metadata_plan_complete'] is True, 'Only the unadmitted metadata plan is accepted')
    require(all(proof[k] is False for k in ('method_locked', 'fitting_admitted', 'formal_training_admitted',
        'automatic_launch_enabled', 'official_fullfit_completed')), 'Planning must not imply execution permission')
    require(all(proof[k] == 0 for k in ('actual_GPU_jobs', 'actual_parameter_updates', 'actual_new_OOF_models',
        'actual_new_OOF_condition_tensors', 'score_tables_read', 'tensor_image_weight_payloads_opened')), 'No fit or payload proof may be invented')
    require(proof['source'] == P.ref(Path(P.__file__)), 'Planner source changed since metadata creation')
    require(len(proof['artifacts']) == 4, 'Four exact plan artifacts required')
    artifacts = {}
    expected_names = {'official_fullfit_plan.json', 'official_fullfit_two_folds.json',
        'official_fullfit_encoder_tasks.json', 'official_fullfit_OOF_query_map.jsonl.gz'}
    for item in proof['artifacts']:
        path = Path(item['path'])
        require(path.resolve().parent == directory.resolve() and path.name in expected_names, 'Unexpected artifact path')
        require(P.ref(path) == item and path.name not in artifacts, 'Changed or duplicated plan artifact')
        artifacts[path.name] = path
    require(set(artifacts) == expected_names, 'Missing plan artifacts')
    paths = {name: item['path'] for name, item in proof['input_metadata'].items()}
    rows, bindings, split, refs = P.pinned_inputs(paths)
    require(refs == proof['input_metadata'], 'Original pinned metadata references changed')
    expected_plan, expected_folds, expected_tasks, expected_records = P.build_metadata(rows, bindings, split, refs)
    documents = [json.loads(artifacts[name].read_text(encoding='utf-8')) for name in
        ('official_fullfit_plan.json', 'official_fullfit_two_folds.json', 'official_fullfit_encoder_tasks.json')]
    require(documents == [expected_plan, expected_folds, expected_tasks], 'Plan/fold/task semantic mismatch')
    count = 0; digest = hashlib.sha256()
    with gzip.open(artifacts['official_fullfit_OOF_query_map.jsonl.gz'], 'rb') as stream:
        for expected in expected_records:
            line = stream.readline()
            require(line == P.canonical(expected) + b'\n', 'Wrong subject/row/held encoder in query map')
            digest.update(line); count += 1
        require(stream.read(1) == b'', 'Unexpected extra query mapping rows')
    require(count == 165400 and digest.hexdigest() == proof['complete_plan_coverage']['uncompressed_rows_sha256'], 'Incomplete full OOF identity coverage')
    return dict(plan=documents[0], folds=documents[1], tasks=documents[2], bindings=bindings,
        image_rows=rows, source_split=split, preparation=P.ref(directory / 'preparation.json'))


def registration_for_task(bundle, task):
    return dict(schema='BrainHIVE_official_single_encoder_registration_v1',
        phase='official_full1654_new_concept_OOF', role=task['role'], subject=task['subject'], task_id=task['task_id'],
        training_seed=1729, subjects=[task['subject']], epochs=25, global_batch=1024, updates_per_subject=225,
        training_array_rows=task['training_array_rows'], excluded_oof_rows=task['exported_OOF_array_rows'],
        training_concepts=task['training_concepts'], excluded_oof_concepts=task['held_concepts'],
        official_train_images=16540, official_train_concepts=1654, unique_fit_images=8270,
        complete_final_phase_fold_sha256=bundle['folds']['concept_fold_sha256'],
        image_manifest_sha256=P.PINS['image_manifest'][0], input_binding_sha256=P.PINS['bindings'][0],
        EEG_input=task['train_EEG_metadata'], source_commit=SOURCE_COMMIT,
        source_class='main.models_brain.BrainMLP', input_dimensions=4250, output_dimensions=1024,
        num_layers=1, dropout=.1, model_parameters=6452224, pretrained_brain_weights=False,
        seed_fixed_before_initialization=True, shuffle_generator_seed=1730,
        learning_rate=5e-4, AdamW_betas=[.9, .995], weight_decay=.05,
        exclude_bias_and_LayerNorm_from_decay=True, warmup_steps=0, scheduler='cosine, optimizer before scheduler',
        gradient_clip=1., learned_logit_scale_initial_value=2.6592, learned_logit_scale_trainable=True,
        source_alignment_target='F(image), raw output; symmetric cosine InfoNCE',
        frozen_F_A_P_IP_UNet=True, frozen_visual_training_dropout=.1,
        frozen_visual_training_mode='train as released Trainer; weights frozen, stochastic targets',
        deterministic_F_cache_used_as_training_target=False,
        raw_brain_to_generator_contract='P(A(brain)); no unit-norm replacement',
        checkpoint_rule='fixed final25epochs; no best-of-validation or benchmark selection',
        official_test_image_or_target_access=False, official_test_EEG_access=False,
        automatic_retrieval_diagnostics=False, training_score_selection=False)


def job_for_task(bundle, task_id, output_root=OUTPUT_NAMESPACE):
    """Concrete registration/output schema; no executable command/admission."""
    matches = [task for task in bundle['tasks']['tasks'] if task['task_id'] == task_id]
    require(len(matches) == 1, 'Exactly one complete official OOF task is required')
    task = matches[0]; root = clean_output_root(output_root) / task['role'] / task['subject']
    registration = registration_for_task(bundle, task)
    job = dict(schema='BrainHIVE_official_single_encoder_job_contract_v1', status=STATUS,
        task_id=task_id, subject=task['subject'], role=task['role'], registration=registration,
        registration_core_sha256=core_sha(registration), output_root=str(root),
        outputs={key: str(root / name) for key, name in OUTPUT_FILES.items()},
        expected_export_rows=task['exported_OOF_array_rows'], expected_cache_shape=[8270, 1024],
        expected_export_dtype='float32', condition_normalization='raw LayerNorm output',
        expected_affine_keys=list(AFFINE_KEYS), expected_affine_shape=[1024],
        required_resume_fields=list(RESUME_FIELDS), required_before_any_fit=list(PRE_FIT_REQUIREMENTS),
        method_locked=False, fitting_admitted=False, automatic_launch_enabled=False,
        executable_commands=[], future_executor_source_sha256=None, source_payload_verification_complete=False)
    validate_job(bundle, job)
    return job


def validate_job(bundle, job):
    require(set(job) == JOB_KEYS, 'Exact job schema required; no extra executable or missing preconditions')
    task = next((r for r in bundle['tasks']['tasks'] if r['task_id'] == job['task_id']), None)
    require(task is not None and job['subject'] == task['subject'] and job['role'] == task['role'], 'Wrong subject or fold task')
    expected_task = next((r for r in P.tasks_for(bundle['folds']['folds'], bundle['bindings']) if r['task_id'] == job['task_id']), None)
    require(task == expected_task, 'Task differs from the complete official fold and actual EEG binding')
    reg = job['registration']; expected = task
    require(job['status'] == STATUS and job['schema'] == 'BrainHIVE_official_single_encoder_job_contract_v1', 'Wrong job contract type')
    require(job['registration_core_sha256'] == core_sha(reg), 'Registration modified')
    require(reg == registration_for_task(bundle, task), 'All native objective/source/seed/optimizer/input fields are fixed by this task')
    require(reg['task_id'] == task['task_id'] and reg['subject'] == task['subject'] and reg['role'] == task['role'], 'Registration identity mismatch')
    require(reg['training_array_rows'] == expected['training_array_rows'] and reg['excluded_oof_rows'] == expected['exported_OOF_array_rows'], 'A protected-T or wrong fold scope cannot be reused')
    require(reg['training_concepts'] == expected['training_concepts'] and reg['excluded_oof_concepts'] == expected['held_concepts'], 'Whole-concept OOF exclusion required')
    require(reg['complete_final_phase_fold_sha256'] == bundle['folds']['concept_fold_sha256'], 'Wrong full-concept fold map')
    require(reg['EEG_input'] == task['train_EEG_metadata'], 'Wrong actual EEG input binding')
    require(reg['source_commit'] == SOURCE_COMMIT and reg['source_class'] == 'main.models_brain.BrainMLP', 'Native class/source mismatch')
    require((reg['training_seed'], reg['epochs'], reg['global_batch'], reg['updates_per_subject']) == (1729, 25, 1024, 225), 'Native fixed-budget fit required')
    require(reg['frozen_visual_training_dropout'] == .1 and reg['frozen_visual_training_mode'] ==
        'train as released Trainer; weights frozen, stochastic targets' and reg['frozen_F_A_P_IP_UNet'] is True,
        'Retain original stochastic frozen teacher')
    require(reg['deterministic_F_cache_used_as_training_target'] is False, 'Deterministic cached teacher substitution is a different objective')
    require(all(reg[k] is False for k in ('official_test_image_or_target_access', 'official_test_EEG_access',
        'automatic_retrieval_diagnostics', 'training_score_selection')), 'Encoder fit/export cannot read official targets or scores')
    require(job['expected_export_rows'] == task['exported_OOF_array_rows'] and job['expected_cache_shape'] == [8270, 1024], 'Full held-row cache required')
    root = clean_output_root(job['output_root'])
    require(root.parts[-2:] == (task['role'], task['subject']), 'Output namespace aliases another task')
    require(job['outputs'] == {key: str(root / name) for key, name in OUTPUT_FILES.items()}, 'Dedicated exact per-task output filenames required')
    require(job['expected_export_dtype'] == 'float32' and job['condition_normalization'] == 'raw LayerNorm output'
        and job['expected_affine_keys'] == AFFINE_KEYS and job['expected_affine_shape'] == [1024], 'Raw native output and own-affine cache schema required')
    require(job['required_resume_fields'] == RESUME_FIELDS and job['required_before_any_fit'] == PRE_FIT_REQUIREMENTS,
        'Preserve every exact-resume and real method/source/resource precondition')
    require(all(job[k] is False for k in ('method_locked', 'fitting_admitted', 'automatic_launch_enabled', 'source_payload_verification_complete')), 'Contract validation is not execution admission')
    require(job['executable_commands'] == [] and job['future_executor_source_sha256'] is None, 'Future executor is not implemented or authorized by this contract')
    return dict(status=STATUS, metadata_contract_valid=True, payload_verified=False,
        method_locked=False, fitting_admitted=False, actual_parameter_updates=0)


def validate_fit_metadata(job, completion, epochs):
    """Validate future producer metadata; never substitute for checkpoint load."""
    reg = job['registration']
    require(completion['subject'] == job['subject'] and completion['role'] == job['role'], 'Wrong fitted model identity')
    require(completion['registration_core_sha256'] == job['registration_core_sha256'], 'Wrong training registration')
    require(completion['model_path'] == job['outputs']['final_checkpoint'] and P.digest(completion['model_sha256']), 'Wrong final checkpoint reference')
    require(P.digest(completion['registration_sha256']) and P.digest(completion['producer_source_sha256']), 'Actual source and registration file references required')
    require((completion['training_seed'], completion['epochs'], completion['updates'], completion['unique_train_stimuli'],
        completion['training_exposures'], completion['parameters']) == (1729, 25, 225, 8270, 206750, 6452224), 'Incomplete or wrong fitted budget')
    require(completion['checkpoint_selection'] == 'fixed final epoch' and completion['trainable_logit_parameters'] == 1, 'Native fixed-final checkpoint rule required')
    require(len(epochs) == 25, 'All fixed epochs must be present')
    for number, row in enumerate(epochs, 1):
        require(row['subject'] == job['subject'] and row['role'] == job['role'] and row['epoch'] == number, 'Epoch identity/order mismatch')
        require(row['updates'] == number * 9 and row['expected_updates'] == 225 and row['full_epoch_rows'] == 8270, 'Partial epoch or dropped final batch')
    require(reg['excluded_oof_rows'] == job['expected_export_rows'], 'OOF scope changed after fit')
    return dict(status=PAYLOAD_PENDING, checkpoint_metadata_valid=True, checkpoint_payload_verified=False,
        actual_fit_certified_here=False, fitting_admitted=False, condition_tensor_payload_verified=False)


def validate_export_metadata(job, fit, epochs, record):
    # A correct cache shape alone cannot legitimize a checkpoint trained on a
    # different fold/budget. Carry the complete fixed-epoch producer chain.
    validate_fit_metadata(job, fit, epochs)
    require(record['subject'] == job['subject'] and record['role'] == job['role'], 'Wrong condition export identity')
    require(record['registration_core_sha256'] == job['registration_core_sha256'], 'Export from another training registration')
    require(record['checkpoint_sha256'] == fit['model_sha256'] and record['checkpoint_path'] == fit['model_path'], 'Export/checkpoint SHA chain mismatch')
    require(record['cache_path'] == job['outputs']['condition_cache'] and P.digest(record['cache_sha256']), 'Wrong condition cache reference')
    require(record['EEG_input_path'] == job['registration']['EEG_input']['path']
        and record['EEG_input_sha256'] == job['registration']['EEG_input']['sha256']
        and record['physical_repeat_count'] == 4, 'Export only the bound physical-mean4 training EEG')
    require(record['array_rows'] == job['expected_export_rows'] and len(set(record['array_rows'])) == 8270, 'Must export all and only held-out rows in fixed order')
    require(record['condition_shape'] == [8270, 1024] and record['condition_dtype'] == 'float32', 'Native complete condition tensor required')
    require(record['normalization'] == 'raw LayerNorm output' and record['additional_L2_normalization'] is False, 'Do not replace the native raw condition')
    require(record['all_condition_values_finite'] is True, 'Nonfinite exported condition metadata')
    require(record['model_training_mode'] is False and record['gradients_enabled'] is False
        and record['all_model_parameters_frozen'] is True, 'Export with native eval/inference mode and frozen weights')
    require(record['affine_source_checkpoint_sha256'] == fit['model_sha256'] and record['affine_keys'] == AFFINE_KEYS and record['affine_shape'] == [1024], 'Use each OOF encoder own output affine')
    require(all(record[k] is False for k in ('official_test_inputs_loaded', 'target_image_or_concept_used_for_export',
        'official_retrieval_computed', 'model_fitted_during_export')), 'Export is inference on held training EEG only')
    return dict(status=PAYLOAD_PENDING, condition_export_metadata_valid=True,
        condition_tensor_payload_verified=False, actual_new_OOF_condition_tensors_certified=0,
        fitting_admitted=False, required_next='Actual cache SHA/torch load, all array_rows, finite tensor and own-checkpoint affine verification')


def published_metadata_eligibility(registration, proof):
    """Historical metadata eligibility only; no payload/current source claims."""
    require(registration['role'] == 'published' and registration['subjects'] == list(P.SUBJECTS), 'Only all-ten published full parents can be considered')
    require(registration['training_array_rows'] == list(range(16540)) and registration['excluded_oof_rows'] == [], 'Full official train scope required')
    require((registration['training_seed'], registration['epochs'], registration['global_batch'],
        registration['updates_per_subject']) == (1729, 25, 1024, 425), 'Wrong published fit scope/budget')
    require(registration['source_class'] == 'main.models_brain.BrainMLP' and registration['source_commit'] == SOURCE_COMMIT,
        'Wrong native parent source')
    require(registration['frozen_visual_training_dropout'] == .1 and registration['deterministic_F_cache_used_as_training_target'] is False
        and registration['frozen_visual_training_mode'] == 'train as released Trainer; weights frozen, stochastic targets', 'Wrong original teacher')
    require(proof['status'] == 'ALL40_CHECKPOINTS_LOGS_AND_CONDITION_IDENTITIES_VERIFIED_CPU', 'Missing historical checkpoint verification metadata')
    models = [row for row in proof['models'] if row['role'] == 'published']
    require(len(models) == 10 and {m['subject'] for m in models} == set(P.SUBJECTS), 'Need all ten distinct published models')
    for model in models:
        require((model['epochs'], model['updates'], model['train_rows'], model['condition_rows'], model['parameters']) ==
            (25, 425, 16540, 200, 6452224), 'Wrong published model or R80 export shape')
        require(P.digest(model['checkpoint_sha256']) and P.digest(model['condition_cache_sha256']), 'Historical references must be complete')
    return dict(status='PUBLISHED_PARENT_METADATA_ELIGIBLE_CURRENT_PAYLOAD_AUDIT_PENDING',
        candidate_subjects=sorted(m['subject'] for m in models), candidate_models=10,
        metadata_expected_checkpoint_sha256={m['subject']: m['checkpoint_sha256'] for m in models},
        all_current_checkpoint_payloads_verified=False, all_current_R80_caches_verified=False,
        source_and_registration_checkpoint_linkage_reverified=False,
        may_replace_new_official_OOF=False, model_reuse_admitted=False, benchmark_readout_admitted=False,
        required_next=['actual full-parent source/registration/checkpoint SHA and strict state/affine audit',
            'actual R80 EEG-only cache row/model binding or separate re-export after method lock',
            'equal final-fit opportunities and final calibration policy locked before official readout'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-directory', type=Path, required=True)
    args = parser.parse_args()
    bundle = read_plan_bundle(args.plan_directory)
    jobs = [job_for_task(bundle, task['task_id']) for task in bundle['tasks']['tasks']]
    print(json.dumps(dict(status=STATUS, complete_task_contracts=len(jobs), expected_new_encoder_fits=20,
        expected_new_OOF_queries=165400, actual_fits=0, method_locked=False, fitting_admitted=False,
        automatic_launch_enabled=False, payloads_verified=0, files_written=0)))


if __name__ == '__main__':
    main()
