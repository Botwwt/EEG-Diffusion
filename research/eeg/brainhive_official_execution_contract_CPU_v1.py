"""Read-only CPU gate and payload checks for future official OOF executors.

This module issues NO method lock, review, admission, resource reservation or
job. A JSON flag cannot replace the actual source-bound four-arm closure,
method/calibration/equal-opportunity review, closed profiling jobs, current
capacity and exclusive running wrapper. None of those official reviews exists
at implementation time; normal entry points consequently reject before Torch,
model/payload reads or output writes. No public fixture/bypass CLI is provided.
"""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import brainhive_official_encoder_job_contract_CPU_v1 as E

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ENCODER_CONTRACT_SHA = '310f1246396720e476c7897f2d593cf82ac91ee573495e38d14fb102862304a5'
ENTRIES = {'train': 'train_BrainHIVE_official_encoder_v1.py', 'export': 'export_BrainHIVE_official_OOF_conditions_v1.py'}
CORE_SOURCES = ('brainhive_official_execution_contract_CPU_v1.py', 'brainhive_official_encoder_job_contract_CPU_v1.py',
    'prepare_BrainHIVE_official_fullfit_plan_CPU_v1.py', 'brainhive_official_encoder_training_v1.py',
    'train_BrainHIVE_official_encoder_v1.py', 'export_BrainHIVE_official_OOF_conditions_v1.py',
    'brainhive_native_models_v1.py', 'brainhive_functional_training_contract_v2.py', 'run_gpu_job_v2.py')
AUTHOR_SOURCES = tuple('reproduction_sources/BrainHIVE_f06db69f/' + name for name in
    ('_fetch_receipt.json', 'main/layers.py', 'main/models_brain.py', 'main/models_adapter.py'))
MIN_ROLES = {'parent', 'O', 'P', 'PN', 'S'}
REVIEW_STATUS = 'OFFICIAL_FULLFIT_SOURCE_METHOD_CALIBRATION_BASELINE_AND_RESOURCE_REVIEWED'
_EXECUTION_TOKEN = object()
require = E.require


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    path = Path(path)
    require(path.is_file() and 0 < path.stat().st_size <= 32 << 20, 'Missing/oversized CPU review metadata: ' + str(path))
    return json.loads(path.read_bytes())


def ref(path):
    return dict(path=str(Path(path)), sha256=sha(path))


def bound(value):
    require(set(value) == {'path', 'sha256'} and E.P.digest(value['sha256']), 'Concrete SHA-bound metadata reference required')
    path = Path(value['path'])
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and sha(path) == value['sha256'], 'Missing/changed/symlink reviewed evidence: ' + str(path))
    return read(path)


def live(pid):
    if type(pid) is not int or pid <= 0: return False
    try:
        parts = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(') ', 1)[1].split()
        return parts[0] not in ('Z', 'X')
    except FileNotFoundError: return False


def validate_development(method):
    """All four real closure files precede any final official fit permission."""
    development = method['development']
    queue = bound(development['queue'])
    require(queue['status'] in ('ALL_FOUR_FULL_FIXED_LIFECYCLES_AND_PREREGISTERED_COMPARISON_COMPLETE',
        'ALL_FOUR_FULL_LIFECYCLES_COMPLETE_AFTER_DOCUMENTED_SINGLE_O_RESTART')
        and not live(queue['pid']) and queue['current_stage'] is None, 'Current four-arm development must actually finish first')
    require(queue['comparison'] == development['comparison'] and queue['plan'] == development['plan'], 'Method lock refers to another development lifecycle')
    plan = bound(development['plan'])
    completed = queue['completed']
    require(len(completed) == 4 and {r['arm'] for r in completed} == {'O', 'P', 'PN', 'S'}, 'All four actual development arms required')
    for item in completed:
        final, generated, evaluation = [bound(item[key]) for key in ('final', 'generated', 'evaluation')]
        require(final['arm'] == generated['arm'] == evaluation['arm'] == item['arm'], 'Cross-arm development closure reuse')
        require(final['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
            and final['updates'] == 24807 and final['seen_queries'] == 396900, 'Development head is not full fixed-final fit')
        require(generated['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL16600_IMAGES_AND_TARGET_JOIN_COMPLETE'
            and generated['queries'] == 16600, 'Development images are incomplete')
        require(evaluation['status'] == 'FUNCTIONAL_ALIGNMENT_V2_FULL16600_EVALUATION_CLOSED'
            and evaluation['queries'] == 16600 and evaluation['actual_full_V16600_evaluation_completed'] is True
            and evaluation['full_identity_target_seed_noise_chain_verified'] is True
            and evaluation['complete_low_level_mechanical_coverage'] is True, 'Development image evaluation is incomplete')
        require(evaluation['recipe_core_sha256'] == final['recipe_core_sha256'] == generated['recipe_core_sha256'], 'Development recipe chain differs')
        require(evaluation['actual_invocations'] and all(r['exit_code'] == 0 for r in evaluation['actual_invocations']), 'No actual successful evaluator invocation')
        for key in ('manifest', 'artifact_summary', 'artifact_per_query', 'score_summary', 'score_per_query', 'low_summary', 'low_per_query'):
            evidence = evaluation[key]
            require(Path(evidence['path']).is_file() and sha(evidence['path']) == evidence['sha256'], 'Closed actual evaluation file changed')
    comparison = bound(development['comparison'])
    require(comparison['status'] == 'ALL_FOUR_FULL16600_V_PAIRED_COMPARISONS_COMPLETE'
        and comparison['actual_full_V16600_comparison_completed'] is True and comparison['queries_per_arm'] == 16600
        and comparison['arms'] == ['O', 'P', 'PN', 'S'] and comparison['subjects'] == 10
        and comparison['repeat_protocol'] == 'R4', 'No full four-arm paired development comparison')
    require(plan['schema'] == 'BrainHIVE_functional_alignment_v2_four_arm_storage_plan', 'Wrong source development plan')
    return development


def validate_method_and_admission(review, bundle):
    method = bound(review['method_lock']); admission = bound(review['admission'])
    require(method['schema'] == 'BrainHIVE_official_fullfit_method_lock_v1'
        and method['status'] == 'METHOD_AND_FINAL_CALIBRATION_LOCKED_AFTER_COMPLETE_DEVELOPMENT', 'No actual sealed official method lock')
    require(admission['schema'] == 'BrainHIVE_official_fullfit_execution_admission_v1'
        and admission['status'] == 'SOURCE_BOUND_WHOLE_OFFICIAL_LIFECYCLE_EXECUTION_REVIEWED', 'No actual whole official execution admission')
    require(admission['method_lock'] == review['method_lock'] and admission['plan_preparation'] == bundle['preparation'], 'Method/admission/official plan chain differs')
    require(admission['official_test_inputs_or_scores_allowed_during_OOF_fit_export'] is False, 'Official test access is forbidden during OOF execution')
    require(method['training_seed'] == 1729 and method['official200_previously_exposed'] is True
        and method['independent_H'] is False and method['official_result_access_used_for_lock'] is False, 'Official method lock must preserve exposed-benchmark limits')
    development = validate_development(method)
    roles = method['final_roles']
    require(len(roles) == len(set(roles)) and MIN_ROLES <= set(roles), 'All fixed strong baseline/final mechanism roles need equal final-fit opportunity')
    opportunities = method['equal_baseline_opportunities']
    require(set(opportunities) == set(roles), 'Missing equal-opportunity role')
    for role, opportunity in opportunities.items():
        require(opportunity['official_train_images'] == 16540 and opportunity['training_seed'] == 1729, 'Unequal official data/seed permission')
        if role != 'parent':
            require(opportunity['OOF_queries'] == 165400 and opportunity['passes'] == 3 and opportunity['updates'] == 31014,
                'All final component roles require full official fit opportunity')
    cases = method['benchmark_cases']
    case_ids = [(case['role'], case['intervention']) for case in cases]
    selected = method['final_candidate_role']
    require(selected in roles and len(case_ids) == len(set(case_ids))
        and {(r, 'correct') for r in roles} | {(selected, 'permuted'), (selected, 'missing_raw_condition')} <= set(case_ids),
        'Whole benchmark must retain every final baseline and both EEG controls')
    require(all(c['role'] in roles and c['intervention'] in ('correct', 'permuted', 'missing_raw_condition')
        and c['repeat_protocol'] == 'R80' and c['queries'] == 2000 and c['images_per_query'] == 1 for c in cases),
        'Official benchmark scope/count differs from locked one-image R80 protocol')
    calibration = bound(method['calibration_registration'])
    require(calibration['schema'] == 'BrainHIVE_official_final_calibration_registration_v1'
        and calibration['status'] == 'LOCKED_BEFORE_ANY_NEW_OFFICIAL_READOUT', 'Final calibration rule is not locked')
    require(all(calibration[k] is False for k in ('old_Dfit_is_independent', 'old_Dcheck_is_independent', 'old_V_is_independent', 'official_test_targets_or_scores_used')),
        'Old D/V become training data after fullfit; official test cannot calibrate')
    require(set(calibration['role_rules']) == set(roles), 'Every baseline/final role needs a fixed calibration rule')
    permission_classes = set()
    for rule in calibration['role_rules'].values():
        require(rule['policy_family'] in ('locked_development_hyperparameter', 'registered_train_only_nested')
            and rule['official_benchmark_access'] is False, 'Unknown or test-dependent final calibration policy')
        rule_document = bound(rule['rule_document'])
        require(rule_document['policy_family'] == rule['policy_family'] and rule_document['official_benchmark_access'] is False,
            'Calibration source/rule chain differs')
        permission_classes.add(rule['permission_class'])
    require(len(permission_classes) == 1, 'Unequal final calibration permissions')
    # A producer execution receipt is required, not only a hand-set lock flag.
    producer = bound(method['producer_receipt'])
    method_core = E.core_sha({key: value for key, value in method.items() if key != 'producer_receipt'})
    require(producer['status'] == 'ACTUAL_COMPLETE_DEVELOPMENT_METHOD_LOCK_CPU_EXECUTION_RECORDED'
        and producer['exit_code'] == 0 and producer['development'] == development
        and producer['method_lock_core_sha256'] == method_core, 'No actual source-bound method-lock producer execution')
    source = producer['source']
    require(set(source) == {'path', 'sha256'} and E.P.digest(source['sha256'])
        and source['path'] in producer['argv'] and sha(source['path']) == source['sha256'],
        'Method lock producer actual source differs')
    return method, admission


def capacity_forecast(volumes):
    """Require one complete aggregate forecast per actual filesystem.

    Two directory roots on one disk must not each spend the same free bytes.
    The future sealed review must combine their whole remaining scope first.
    """
    require(volumes, 'Explicit complete remaining persistent forecast required')
    free = []; devices = set()
    for volume in volumes:
        root = Path(volume['root']); reservation = volume['reservation']
        require(root.is_absolute() and root.is_dir() and type(volume['remaining_required_bytes']) is int
            and volume['remaining_required_bytes'] > 0, 'Invalid actual resource forecast volume')
        require(Path(reservation['path']).is_file() and sha(reservation['path']) == reservation['sha256'], 'Physical retained reservation changed')
        device = root.stat().st_dev
        require(device == Path(reservation['path']).stat().st_dev, 'Reservation is on another volume')
        require(device not in devices, 'A physical filesystem requires one aggregated whole-lifecycle forecast; duplicate free bytes cannot be spent twice')
        devices.add(device)
        available = shutil.disk_usage(root).free
        require(available >= volume['remaining_required_bytes'], 'Complete official lifecycle no longer fits; no payload or job writes')
        free.append(dict(root=str(root), device=device, free=available, required=volume['remaining_required_bytes']))
    return free


def resource_review(review, method):
    resource = bound(review['resource_review'])
    require(resource['schema'] == 'BrainHIVE_official_fullfit_resource_review_v1'
        and resource['status'] == 'COMPLETE_OFFICIAL_FIT_EXPORT_COMPONENT_IMAGE_EVALUATION_SCOPE_FUNDED', 'No reviewed whole official lifecycle resources')
    require(resource['method_lock'] == review['method_lock'] and resource['OOF_encoder_models'] == 20
        and resource['OOF_export_queries'] == 165400 and set(resource['final_roles']) == set(method['final_roles']), 'Resource review omits full official scope')
    require(resource['benchmark_cases'] == method['benchmark_cases']
        and resource['benchmark_images'] == 2000 * len(method['benchmark_cases']), 'Whole benchmark/control images and evaluation costs were omitted')
    require(resource['profile_jobs'], 'Actual measured profiling evidence required')
    purposes = set()
    for evidence in resource['profile_jobs']:
        require(evidence['purpose'] in ('train', 'export'), 'Profile evidence must cover encoder fit/export')
        purposes.add(evidence['purpose'])
        job = bound(evidence['job']); source = evidence['source']
        require(job['status'] == 'COMPLETED' and job['exit_code'] == 0 and job['charged_seconds'] > 0
            and not live(job.get('pid')), 'Profiling evidence must be a real completed finite GPU job')
        require(source['path'] in job['command'] and E.P.digest(source['sha256']) and sha(source['path']) == source['sha256'], 'Actual profiling source/job chain differs')
        compatibility = bound(evidence['scope_compatibility_review'])
        require(compatibility['status'] == 'ACTUAL_PROFILE_SCOPE_AND_SOURCE_COMPATIBILITY_REVIEWED'
            and compatibility['profile_job'] == evidence['job'] and compatibility['profile_source'] == source
            and compatibility['target_source_sha256'] == review['source_sha256']
            and compatibility['model_class'] == 'main.models_brain.BrainMLP'
            and compatibility['parameters'] == 6452224, 'Unrelated profiling cannot fund the official encoder/export scope')
    require(purposes == {'train', 'export'}, 'Both actual fit and actual full-export costs need reviewed evidence')
    free = capacity_forecast(resource['capacity_forecast'])
    from brainhive_functional_training_contract_v2 import resource_headroom
    memory = resource_headroom()
    require(resource['minimum_effective_CPU_headroom_bytes'] > 0
        and memory['effective_headroom_bytes'] >= resource['minimum_effective_CPU_headroom_bytes'], 'Fresh reviewed CPU headroom unavailable')
    return resource, dict(volumes=free, memory=memory)


def exclusive_wrapper(review, task, action, entry_name, resume):
    require(os.name == 'posix' and sys.executable == review['runtime']['python_executable'], 'Use the actual reviewed existing Linux Python runtime')
    parent = Path('/proc') / str(os.getppid())
    argv = [v.decode() for v in (parent / 'cmdline').read_bytes().split(b'\0') if v]
    require(str(ROOT / 'run_gpu_job_v2.py') in argv and '--name' in argv and '--' in argv, 'Exclusive existing GPU wrapper required')
    job_name = argv[argv.index('--name') + 1]
    permitted = task['train_job_names'] if action == 'train' else [task['export_job_name']]
    require(job_name in permitted and len(set(permitted)) == len(permitted) and len(permitted) <= 2, 'Unregistered job/retry is forbidden')
    require((resume and action == 'train' and job_name == permitted[-1] and len(permitted) == 2)
        or (not resume and job_name == permitted[0]), 'Only a predeclared exact-resume slot is allowed')
    job_path = ROOT / 'audit' / (job_name + '_job.json'); job = read(job_path)
    require(job['status'] == 'RUNNING' and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid(), 'Actual wrapper/job/process identity differs')
    require(job['command'] == argv[argv.index('--') + 1:] and str(ROOT / entry_name) in job['command'], 'Actual executor command differs')
    tail = job['command'][job['command'].index(str(ROOT / entry_name)) + 1:]
    expected = ['--plan-directory', task['plan_directory'], '--task-id', task['task_id'], '--execution-review', review['_review_path']]
    if resume: expected.append('--resume')
    require(tail == expected, 'Executor CLI differs from the exact reviewed task')
    cap = task['training_seconds' if action == 'train' else 'export_seconds']
    require(math.isfinite(cap) and 0 < cap <= 86400 and job['max_seconds'] == cap, 'Actual finite wrapper bound differs')
    authorization = ref(ROOT / 'configs/resource_authorization_20261020_v3.json')
    require(job['authorization_sha256'] == authorization['sha256'] and review['resource_authorization'] == authorization, 'Existing user resource authority changed')
    lock = ROOT / 'audit/GPU_JOB.lock'; details = lock.stat()
    held = any((fd.stat().st_dev, fd.stat().st_ino) == (details.st_dev, details.st_ino) for fd in (parent / 'fd').iterdir() if fd.exists())
    require(held, 'Trusted wrapper must retain the real GPU lock descriptor')
    import fcntl
    with lock.open('r') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise ValueError('GPU lock is not exclusively held by the actual wrapper')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    require({int(v.strip()) for v in apps.stdout.splitlines() if v.strip()} <= {os.getpid()}, 'Another GPU process is active')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    gpu_rows = [row.split(',') for row in gpu.stdout.splitlines() if row.strip()]
    require(len(gpu_rows) == 1 and gpu_rows[0][0].strip() == review['runtime']['GPU_uuid'], 'Only the reviewed existing single GPU may execute')
    minimum_GPU = task['resource_limits']['minimum_free_GPU_bytes']
    require(type(minimum_GPU) is int and minimum_GPU > 0 and int(gpu_rows[0][1].strip()) * (1 << 20) >= minimum_GPU, 'Reviewed actual free GPU memory unavailable')
    started = datetime.datetime.fromisoformat(job['start_utc'].replace('Z', '+00:00')).timestamp()
    remaining = cap - (time.time() - started)
    require(remaining > 0, 'Registered task timeout already expired')
    return dict(job_reference=ref(job_path), job_name=job_name, remaining_seconds=remaining, wrapper_pid=os.getppid())


def preflight(plan_directory, task_id, execution_review, *, action, entry_name, resume=False):
    """First executor operation. Normal missing-review path does not read payloads."""
    action = 'train' if action == 'fit' else action
    require(action in ENTRIES and entry_name == ENTRIES[action], 'Wrong official execution action/source')
    filename = Path(execution_review)
    require(filename.is_absolute() and filename.is_file(), 'Official method/resource execution review does not exist; fit/export is NOT admitted')
    review = read(filename)
    require(review['schema'] == 'BrainHIVE_official_fullfit_execution_review_v1' and review['status'] == REVIEW_STATUS,
        'No actual sealed official execution review; a metadata plan is not admission')
    require(sha(E.__file__) == ENCODER_CONTRACT_SHA, 'Original reviewed encoder metadata contract changed')
    require(set(CORE_SOURCES) | set(AUTHOR_SOURCES) <= set(review['source_sha256']), 'Pin every actual new executor/native/author dependency')
    for name, expected in review['source_sha256'].items():
        p = PurePosixPath(name)
        require(not p.is_absolute() and '..' not in p.parts and E.P.digest(expected) and sha(ROOT / name) == expected, 'Reviewed executable source changed: ' + name)
    require(review['source_sha256'][Path(__file__).name] == sha(__file__), 'This gate was not reviewed with the executors')
    bundle = E.read_plan_bundle(plan_directory)
    require(review['plan_preparation'] == bundle['preparation'], 'Wrong actual twenty-task official bundle')
    job = E.job_for_task(bundle, task_id); E.validate_job(bundle, job)
    require(len(review['tasks']) == 20 and set(review['tasks']) == {t['task_id'] for t in bundle['tasks']['tasks']}, 'All twenty complete actual jobs must be reviewed together')
    task = review['tasks'][task_id]
    require(task['task_id'] == task_id and task['registration_core_sha256'] == job['registration_core_sha256']
        and task['output_root'] == job['output_root'] and task['plan_directory'] == str(Path(plan_directory)), 'Task/registration/output namespace differs from review')
    require(type(task['maximum_output_bytes']) is int and task['maximum_output_bytes'] > 0
        and isinstance(task['resource_limits'], dict) and task['resource_limits'], 'Concrete whole-task output/resource limits required')
    resume_commit = task['resume_commit_ref']
    if action == 'train':
        require((resume and resume_commit is not None) or (not resume and resume_commit is None), 'Fresh/resume permission needs explicit committed-epoch identity')
        if resume:
            commit = bound(resume_commit)
            require(commit['registration_core_sha256'] == job['registration_core_sha256']
                and commit['schema'] == 'BrainHIVE_official_encoder_epoch_transaction_v1', 'Resume commit belongs to another task')
    output = Path(job['output_root']); namespace = Path(str(E.OUTPUT_NAMESPACE))
    require(output.resolve().is_relative_to(namespace.resolve()) and output.resolve() != namespace.resolve(), 'Output path aliases a protected/current namespace')
    require(all(not p.is_symlink() for p in [output, *output.parents] if p.exists()), 'No symlink output namespace or ancestor')
    method, admission = validate_method_and_admission(review, bundle)
    resource, capacity = resource_review(review, method)
    review['_review_path'] = str(filename)
    wrapper = exclusive_wrapper(review, task, action, entry_name, resume)
    payload = review['payload_contract']
    eeg = job['registration']['EEG_input']
    require(payload['subjects'][job['subject']]['EEG'] == {'path': eeg['path'], 'sha256': eeg['sha256']}, 'Wrong subject physical-mean4 EEG payload')
    require(payload['train_visual_embeddings']['path'] == str(WORK / 'data/train_visual_embeddings.pt')
        and payload['fusion_config']['path'] == str(WORK / 'assets/prior_H14/fusion_encoder/config.json'), 'Only actual training visual source/native F config allowed')
    require(payload['runtime']['python_executable'] == sys.executable and payload['runtime'] == review['runtime'], 'Actual pinned runtime contract differs')
    require(review['runtime']['python_version'] == sys.version.split()[0]
        and review['runtime']['package_source_sha256'], 'Concrete reviewed runtime/package sources required')
    for name, expected in review['runtime']['package_source_sha256'].items():
        require(Path(name).is_absolute() and E.P.digest(expected) and sha(name) == expected, 'Pinned Torch/runtime package source changed')
    return dict(_execution_token=_EXECUTION_TOKEN, bundle=bundle, job=job, plan_directory=str(Path(plan_directory)), task_id=task_id,
        action=action, resume=resume, payload_contract=payload, payload_contract_sha256=E.core_sha(payload), deadline_seconds=wrapper['remaining_seconds'],
        deadline_monotonic=time.monotonic() + wrapper['remaining_seconds'], source_sha256=review['source_sha256'],
        method_lock_ref=review['method_lock'], admission_ref=review['admission'], execution_review_ref=ref(filename),
        wrapper_job_ref=wrapper['job_reference'], wrapper_evidence=wrapper, resource_capacity=capacity,
        resource_review_ref=review['resource_review'], resource_limits=task['resource_limits'],
        maximum_output_bytes=task['maximum_output_bytes'], resume_commit_ref=resume_commit, review=review)


def require_context(context):
    require(context.get('_execution_token') is _EXECUTION_TOKEN, 'Only a real successful CPU preflight context can open official payloads/write outputs')
    require(time.monotonic() < context['deadline_monotonic'], 'Finite official task deadline expired')
    require(ref(context['execution_review_ref']['path']) == context['execution_review_ref'], 'Execution review changed during job')


def check_output_budget(context, additional_bytes=0):
    require_context(context)
    root = Path(context['job']['output_root']); used = 0; seen = set()
    if root.exists():
        for file in root.rglob('*'):
            require(not file.is_symlink(), 'No output alias may bypass the reviewed complete task cap')
            if file.is_file():
                details = file.stat(); identity = details.st_dev, details.st_ino
                if identity not in seen:
                    used += details.st_blocks * 512 if hasattr(details, 'st_blocks') else details.st_size
                    seen.add(identity)
    require(type(additional_bytes) is int and additional_bytes >= 0
        and used + additional_bytes <= context['maximum_output_bytes'], 'Reviewed complete task output cap is insufficient; retain and stop')
    return used


def verify_payload_files(context, include_visual=True):
    """Streaming actual hashes, after real review/wrapper gate; never test payloads."""
    require_context(context)
    contract = context['payload_contract']; job = context['job']
    eeg = contract['subjects'][job['subject']]['EEG']
    items = [eeg]
    if include_visual:
        items += [contract['train_visual_embeddings'], contract['fusion_config']]
        require(set(contract['native_prior_assets']) == {str(WORK / 'assets/prior_H14' / module / name)
            for module in ('fusion_encoder', 'proj') for name in ('config.json', 'diffusion_pytorch_model.safetensors')}, 'Pin actual full F/A/P config and weight source')
        items += [dict(path=p, sha256=s) for p, s in contract['native_prior_assets'].items()]
    unique = {}
    for item in items:
        require(item['path'] not in unique or unique[item['path']] == item, 'Conflicting native source SHA declarations')
        unique[item['path']] = item
    items = list(unique.values())
    for item in items:
        require(set(item) == {'path', 'sha256'} and E.P.digest(item['sha256']) and Path(item['path']).is_file()
            and sha(item['path']) == item['sha256'], 'Actual official input payload SHA differs: ' + str(item.get('path')))
        require_context(context)
    proj_meta = read(contract['fusion_config']['path'])['proj_meta'] if include_visual else None
    return dict(EEG_path=eeg['path'], EEG_sha256=eeg['sha256'],
        visual_path=contract['train_visual_embeddings']['path'] if include_visual else None,
        visual_sha256=contract['train_visual_embeddings']['sha256'] if include_visual else None,
        proj_meta=proj_meta, files_verified=items, official_test_or_retrieval_payloads_opened=False)


def validate_training_tensors(context, eeg, visuals=None, proj_meta=None, *, _fixture=None):
    """Actual tensor checks; import Torch only when an executor/test calls this.

    Private tiny fixture changes dimensions only; it never supplies admission.
    """
    if _fixture is None: require_context(context)
    import torch
    expected = (16540, 63, 250) if _fixture is None else tuple(_fixture['EEG_shape'])
    require(_fixture is not None or context.get('_execution_token') is _EXECUTION_TOKEN, 'Tensor load requires successful preflight')
    if _fixture is None:
        require(torch.__version__ == context['payload_contract']['runtime']['torch_version'], 'Actual Torch runtime differs from sealed review')
    require(isinstance(eeg, torch.Tensor) and tuple(eeg.shape) == expected and eeg.dtype == torch.float32, 'Full bound physical-mean4 EEG tensor shape/dtype differs')
    for chunk in eeg.split(256): require(bool(torch.isfinite(chunk).all()), 'Nonfinite actual EEG input')
    if visuals is not None:
        require(isinstance(visuals, dict) and set(visuals) == set(proj_meta), 'Actual visual embedding keys differ from native F.proj_meta')
        for key, dim in proj_meta.items():
            value = visuals[key]
            require(isinstance(value, torch.Tensor) and value.dtype == torch.float32 and value.shape[0] == expected[0]
                and math.prod(value.shape[1:]) == dim, 'Actual full training visual source shape/dtype differs: ' + key)
            for chunk in value.split(256): require(bool(torch.isfinite(chunk).all()), 'Nonfinite training visual source')
    return dict(status='CPU_TINY_TENSOR_FIXTURE_ONLY' if _fixture else 'ACTUAL_OFFICIAL_TRAIN_INPUT_TENSORS_VERIFIED',
        EEG_shape=list(eeg.shape), visual_keys=sorted(visuals) if visuals is not None else [],
        official_test_inputs_opened=False, actual_full16540_verified=_fixture is None)
