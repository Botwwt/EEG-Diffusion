"""Actual full-official OOF component inputs, behind a separate real stage gate.

Import is stdlib only. Missing official review refuses before plan/payload
reads, Torch or output creation. The latent stage shares the gate, but never
opens an OOF/checkpoint/teacher payload. This module issues no review or job.
The CPU checker uses private tiny tensor injections, never execution tokens.
"""
import argparse
import datetime
import gzip
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import time

sys.dont_write_bytecode = True
import brainhive_official_execution_contract_CPU_v1 as G
import export_BrainHIVE_official_OOF_conditions_v1 as X

E, require = G.E, G.require
ROOT = Path(__file__).resolve().parent
STATUS = 'SOURCE_METHOD_WHOLE_RESOURCES_AND_INPUT_STAGE_EXECUTION_REVIEWED'
_TOKEN = object()
PURPOSES = {'component_inputs', 'latents'}


def file_ref(path):
    """Hash real regular files, checking inode/size/mtime and releasing clean cache."""
    path = Path(path)
    require(path.is_absolute() and not path.is_symlink() and path.is_file(), 'Real absolute input file required: ' + str(path))
    before = path.stat()
    with path.open('rb') as stream:
        actual = os.fstat(stream.fileno())
        require((actual.st_dev, actual.st_ino) == (before.st_dev, before.st_ino), 'Input identity changed before hashing')
        try:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            after = os.fstat(stream.fileno())
            require((actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'Input changed while hashing')
        finally:
            if hasattr(os, 'posix_fadvise'):
                os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    current = path.stat()
    require((actual.st_dev, actual.st_ino, actual.st_size, actual.st_mtime_ns) ==
        (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns), 'Input replaced while hashing')
    return dict(path=str(path), sha256=digest)


def bound_file(reference):
    require(set(reference) == {'path', 'sha256'} and E.P.digest(reference['sha256'])
        and file_ref(reference['path']) == reference, 'Bound actual payload/source SHA differs')
    return Path(reference['path'])


def require_context(context, purpose=None):
    require(isinstance(context, dict) and context.get('_stage_token') is _TOKEN,
        'Only an actual successful official input-stage preflight can open payloads')
    require(purpose is None or context['purpose'] == purpose, 'Wrong official input-stage purpose')
    require(time.monotonic() < context['deadline_monotonic'], 'Finite official input-stage deadline expired')
    require(G.ref(context['stage_review_ref']['path']) == context['stage_review_ref'], 'Stage review changed during execution')
    require(G.ref(context['encoder_execution_review_ref']['path']) == context['encoder_execution_review_ref'], 'Encoder review changed during execution')


def check_memory(context):
    require_context(context)
    from brainhive_functional_training_contract_v2 import resource_headroom
    value = resource_headroom()
    require(value['effective_headroom_bytes'] >= context['resource_limits']['minimum_effective_CPU_headroom_bytes'],
        'Fresh reviewed input-stage CPU memory headroom unavailable')
    return value


def check_sources(context):
    require_context(context)
    for name, digest in context['source_sha256'].items():
        require(G.sha(ROOT / name) == digest, 'Reviewed input-stage source changed: ' + name)


def _wrapper(stage, review, review_path, plan_directory, entry_name):
    """Own finite stage CLI/PID/lock check; encoder G CLI is intentionally unused."""
    require(os.name == 'posix' and sys.executable == review['runtime']['python_executable'], 'Reviewed existing Linux runtime required')
    parent = Path('/proc') / str(os.getppid())
    argv = [v.decode() for v in (parent / 'cmdline').read_bytes().split(b'\0') if v]
    require(str(ROOT / 'run_gpu_job_v2.py') in argv and '--name' in argv and '--' in argv, 'Existing exclusive GPU wrapper required')
    name = argv[argv.index('--name') + 1]
    require(name == stage['wrapper_job_name'] and Path(name).name == name, 'Stage wrapper name differs')
    path = ROOT / 'audit' / (name + '_job.json'); job = G.read(path)
    expected = [sys.executable, str(ROOT / entry_name), '--plan-directory', str(plan_directory), '--execution-review', str(review_path)]
    require(job['command'] == argv[argv.index('--') + 1:] == expected and job['status'] == 'RUNNING'
        and job['pid'] == os.getpid() and job['wrapper_pid'] == os.getppid(), 'Actual stage wrapper/CLI/process identity differs')
    cap = stage['maximum_seconds']
    require(type(cap) in (int, float) and math.isfinite(cap) and 0 < cap <= 86400
        and job['max_seconds'] == cap, 'Finite reviewed stage wrapper cap required')
    require(job['authorization_sha256'] == review['resource_authorization']['sha256']
        and G.ref(ROOT / 'configs/resource_authorization_20261020_v3.json') == review['resource_authorization'], 'Existing user resource authority differs')
    lock = ROOT / 'audit/GPU_JOB.lock'; details = lock.stat()
    require(any((fd.stat().st_dev, fd.stat().st_ino) == (details.st_dev, details.st_ino)
        for fd in (parent / 'fd').iterdir() if fd.exists()), 'Wrapper must retain actual GPU lock descriptor')
    import fcntl
    with lock.open('r') as stream:
        try: fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError: pass
        else:
            fcntl.flock(stream, fcntl.LOCK_UN)
            raise ValueError('Actual GPU lock is not exclusively held')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    require({int(v.strip()) for v in apps.stdout.splitlines() if v.strip()} <= {os.getpid()}, 'Another GPU application is active')
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.free', '--format=csv,noheader,nounits'],
        check=True, capture_output=True, text=True, timeout=10)
    values = [line.split(',') for line in gpu.stdout.splitlines() if line.strip()]
    minimum = stage['resource_limits']['minimum_free_GPU_bytes']
    require(type(minimum) is int and minimum > 0 and len(values) == 1
        and values[0][0].strip() == review['runtime']['GPU_uuid']
        and int(values[0][1].strip()) * (1 << 20) >= minimum, 'Actual reviewed GPU UUID/free memory differs')
    started = datetime.datetime.fromisoformat(job['start_utc'].replace('Z', '+00:00')).timestamp()
    remaining = cap - (time.time() - started)
    require(remaining > 0, 'Stage wrapper deadline already expired')
    return dict(job=G.ref(path), job_name=name, remaining_seconds=remaining, pid=os.getpid(), wrapper_pid=os.getppid())


def preflight(plan_directory, review_path, *, purpose, entry_name):
    """No official review exists now; normal use rejects before even plan reads.

    A later consumer calls this in its actual exclusive wrapper using exactly
    --plan-directory/--execution-review. No public Boolean/fixture bypass exists.
    """
    filename = Path(review_path)
    require(filename.is_absolute() and filename.is_file(), 'Official input-stage execution review does not exist; no payload access or stage execution admitted')
    review = G.read(filename)
    require(review['schema'] == 'BrainHIVE_official_training_inputs_execution_review_v1' and review['status'] == STATUS,
        'No sealed source/method/resource input-stage execution review; planning flags are not admission')
    require(purpose in PURPOSES and Path(entry_name).name == entry_name, 'Exact known input-stage entry required')
    encoder_review = G.bound(review['encoder_execution_review'])
    require(encoder_review['schema'] == 'BrainHIVE_official_fullfit_execution_review_v1'
        and encoder_review['status'] == G.REVIEW_STATUS, 'Actual original encoder execution review required')
    require(G.sha(E.__file__) == G.ENCODER_CONTRACT_SHA, 'Frozen original encoder contract changed')
    required_sources = set(G.CORE_SOURCES) | set(G.AUTHOR_SOURCES) | {Path(__file__).name, entry_name}
    require(required_sources <= set(review['source_sha256'])
        and all(review['source_sha256'].get(k) == v for k, v in encoder_review['source_sha256'].items()), 'All actual new and original dependencies must be source bound')
    for name, digest in review['source_sha256'].items():
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and '..' not in relative.parts and E.P.digest(digest)
            and G.sha(ROOT / name) == digest, 'Actual stage dependency source changed: ' + name)
    bundle = E.read_plan_bundle(plan_directory)
    require(review['plan_preparation'] == encoder_review['plan_preparation'] == bundle['preparation'], 'Wrong complete official plan')
    method, admission = G.validate_method_and_admission(encoder_review, bundle)
    resource, capacity = G.resource_review(encoder_review, method)
    require(review['runtime'] == encoder_review['runtime'] and review['resource_authorization'] == encoder_review['resource_authorization']
        and sys.version.split()[0] == review['runtime']['python_version'], 'Reviewed input-stage runtime/user authority differs')
    for path, digest in review['runtime']['package_source_sha256'].items():
        require(Path(path).is_absolute() and E.P.digest(digest) and G.sha(path) == digest, 'Reviewed runtime package source changed')
    stage = review['stages'][purpose]
    require(stage['entry_name'] == entry_name, 'Purpose/actual consumer source differs')
    output = Path(stage['output_root']); namespace = Path(str(E.OUTPUT_NAMESPACE))
    require(output.is_absolute() and output.resolve().is_relative_to(namespace.resolve()) and output.resolve() != namespace.resolve()
        and all(not p.is_symlink() for p in [output, *output.parents] if p.exists()), 'Private canonical official stage output required')
    for encoder_task in bundle['tasks']['tasks']:
        encoder_root = Path(E.job_for_task(bundle, encoder_task['task_id'])['output_root']).resolve()
        require(not output.resolve().is_relative_to(encoder_root) and not encoder_root.is_relative_to(output.resolve()),
            'Input stage output must not overlap any official encoder task namespace')
    require(type(stage['maximum_output_bytes']) is int and stage['maximum_output_bytes'] > 0, 'Complete stage output cap required')
    threshold = stage['resource_limits']['minimum_effective_CPU_headroom_bytes']
    require(type(threshold) is int and threshold >= 8 << 30, 'The original 8GiB CPU floor cannot be lowered')
    from brainhive_functional_training_contract_v2 import resource_headroom
    memory = resource_headroom()
    require(memory['effective_headroom_bytes'] >= threshold, 'Fresh whole-stage CPU memory headroom unavailable')
    profile = stage['profile']; profile_job = G.bound(profile['job']); profile_scope = G.bound(profile['scope_review'])
    require(profile_job['status'] == 'COMPLETED' and profile_job['exit_code'] == 0 and profile_job['charged_seconds'] > 0
        and not G.live(profile_job.get('pid')) and profile['source']['path'] in profile_job['command']
        and file_ref(profile['source']['path']) == profile['source'], 'Real completed finite source-bound stage profile required')
    require(profile_scope['status'] == 'ACTUAL_PROFILE_SCOPE_AND_SOURCE_COMPATIBILITY_REVIEWED'
        and profile_scope['profile_job'] == profile['job'] and profile_scope['profile_source'] == profile['source']
        and profile_scope['purpose'] == purpose and profile_scope['plan_preparation'] == bundle['preparation']
        and profile_scope['target_source_sha256'] == review['source_sha256'], 'Actual profile scope must be reviewed for this complete future stage')
    wrapper = _wrapper(stage, review, filename, Path(plan_directory), entry_name)
    return dict(_stage_token=_TOKEN, purpose=purpose, bundle=bundle, plan_directory=str(Path(plan_directory)),
        source_sha256=review['source_sha256'], deadline_monotonic=time.monotonic() + wrapper['remaining_seconds'],
        deadline_seconds=wrapper['remaining_seconds'], encoder_execution_review_ref=review['encoder_execution_review'],
        stage_review_ref=G.ref(filename), method_lock_ref=encoder_review['method_lock'], admission_ref=encoder_review['admission'],
        resource_review_ref=encoder_review['resource_review'], stage_output_root=str(output), maximum_output_bytes=stage['maximum_output_bytes'],
        resource_limits=stage['resource_limits'], stage_spec=stage, resource_capacity=capacity, fresh_memory=memory,
        wrapper_evidence=wrapper, runtime=review['runtime'], encoder_review=encoder_review, review=review)


def metadata_layout(bundle):
    """All real165400 planned query identities, without opening tensor payloads."""
    tasks = bundle['tasks']['tasks']; subjects = list(E.P.SUBJECTS)
    require(len(tasks) == 20 and len({t['task_id'] for t in tasks}) == 20, 'Exactly20 distinct full-official OOF tasks required')
    jobs = [E.job_for_task(bundle, task['task_id']) for task in tasks]
    train = [r for r in bundle['image_rows'] if r['dataset_split'] == 'train']
    require(len(train) == 16540 and len({r['concept_id'] for r in train}) == 1654, 'Complete1654concept/16540image teacher identity required')
    by_row = {int(r['array_row']): r for r in train}
    require(set(by_row) == set(range(16540)), 'Complete official global train array rows required')
    task_index = {job['task_id']: index for index, job in enumerate(jobs)}
    source_by_query = [None] * 165400
    for index, job in enumerate(jobs):
        require(job['role'] in ('official_oof0', 'official_oof1'), 'Old protectedT/full/published caches cannot supply official OOF')
        train_rows = set(job['registration']['training_array_rows']); held = job['expected_export_rows']
        require(len(train_rows) == len(held) == 8270 and train_rows.isdisjoint(held)
            and train_rows | set(held) == set(by_row), 'Incomplete/mixed official concept fold')
        require({by_row[r]['concept_id'] for r in train_rows}.isdisjoint(by_row[r]['concept_id'] for r in held), 'OOF model fitted a held concept')
        subject_index = subjects.index(job['subject'])
        for cache_row, array_row in enumerate(held):
            query = subject_index * 16540 + array_row
            require(source_by_query[query] is None, 'Duplicate OOF query assignment')
            source_by_query[query] = (index, cache_row)
    proof = G.read(bundle['preparation']['path'])
    artifact = next(r for r in proof['artifacts'] if Path(r['path']).name == 'official_fullfit_OOF_query_map.jsonl.gz')
    require(file_ref(artifact['path']) == {k: artifact[k] for k in ('path', 'sha256')}, 'Complete query-map artifact changed')
    digest = hashlib.sha256(); count = 0
    with gzip.open(artifact['path'], 'rb') as stream:
        for line in stream:
            row = json.loads(line); q = count; source = source_by_query[q] if q < 165400 else None
            require(source is not None, 'Extra/missing query identity')
            index, cache_row = source; job = jobs[index]; actual = by_row[q % 16540]
            require(row['query_index'] == q and row['subject'] == subjects[q // 16540]
                and row['array_row'] == q % 16540 and row['dataset_split'] == 'official_train'
                and row['stimulus_id'] == actual['stimulus_id'] and row['concept_id'] == actual['concept_id']
                and row['encoder_task_id'] == job['task_id'] and row['held_fold'] == int(job['role'][-1])
                and row['future_OOF_cache_row'] == cache_row and row['EEG_repeat_protocol'] == 'R4'
                and row['physical_repeat_count'] == 4 and row['official_benchmark_query'] is False, 'Wrong full query subject/stimulus/fold/cache-row join')
            digest.update(line); count += 1
    require(count == 165400 and all(v is not None for v in source_by_query)
        and digest.hexdigest() == proof['complete_plan_coverage']['uncompressed_rows_sha256'], 'Missing or repeated full-official OOF query')
    identity = [dict(array_row=r, stimulus_id=by_row[r]['stimulus_id'], concept_id=by_row[r]['concept_id']) for r in range(16540)]
    return dict(jobs=jobs, subjects=subjects, source_by_query=source_by_query, task_index=task_index,
        train_identity=identity, train_identity_sha256=E.core_sha(identity), query_map={k: artifact[k] for k in ('path', 'sha256')},
        queries=count, images=16540, concepts=1654, metadata_only=True)


def _closed_wrapper(reference, *, entry_name, task_id, execution_review, expected_name=None):
    job = G.bound(reference); command = job['command']; entry = str(ROOT / entry_name)
    review = G.bound(execution_review); task = review['tasks'][task_id]
    expected = [review['runtime']['python_executable'], entry, '--plan-directory', task['plan_directory'],
        '--task-id', task_id, '--execution-review', execution_review['path']]
    if entry_name == 'train_BrainHIVE_official_encoder_v1.py' and len(task['train_job_names']) == 2 and expected_name == task['train_job_names'][1]:
        expected.append('--resume')
    require(job['status'] == 'COMPLETED' and job['exit_code'] == 0 and job['charged_seconds'] > 0
        and type(job['pid']) is int and job['pid'] > 0 and type(job['wrapper_pid']) is int and job['wrapper_pid'] > 0
        and not G.live(job['pid']) and not G.live(job['wrapper_pid']) and command == expected
        and job['authorization_sha256'] == review['resource_authorization']['sha256'],
        'Actual independent successful fit/export wrapper required for this exact task/review')
    if expected_name is not None:
        require(Path(reference['path']).name == expected_name + '_job.json', 'Actual fit wrapper reference differs')
    return job


def fit_export_chain(context, job):
    """Read-only equivalent of X.load_fit_chain's guarded chain, plus export closure.

    G's private encoder token cannot be issued to a component consumer. We reuse
    its bound-file/method/resource validators and X's actual epoch scanner and
    checkpoint/cache validators; no token is forged and no gate is monkeypatched.
    """
    require_context(context, 'component_inputs'); paths = job['outputs']; base = context['encoder_review']
    registration, fit, record = [G.read(paths[k]) for k in ('registration', 'fit_completion', 'export_receipt')]
    require(fit['status'] == 'OFFICIAL_SINGLE_ENCODER_ALL8270_FIXED25_EPOCHS_COMPLETE'
        and fit['actual_GPU'] is True and fit['official_test_inputs_loaded'] is False
        and fit['official_scores_read'] is False, 'Actual complete GPU encoder fit with no official test/score access required')
    require(registration == job['registration'] and E.core_sha(registration) == job['registration_core_sha256'], 'Actual full-official fitted registration differs')
    require(file_ref(paths['registration'])['sha256'] == fit['registration_sha256']
        and file_ref(paths['final_checkpoint'])['sha256'] == fit['model_sha256'], 'Actual fitted registration/checkpoint SHA differs')
    fit_review = G.bound(fit['execution_review'])
    require(fit_review['status'] == G.REVIEW_STATUS and fit_review['source_sha256'] == base['source_sha256']
        and fit_review['plan_preparation'] == base['plan_preparation']
        and E.core_sha(fit_review['payload_contract']) == E.core_sha(base['payload_contract'])
        and fit['producer_source_sha256'] == base['source_sha256']['train_BrainHIVE_official_encoder_v1.py'], 'Actual fitted producer/review/input chain differs')
    name = fit['actual_wrapper_job_name']
    require(Path(name).name == name and name in fit_review['tasks'][job['task_id']]['train_job_names'], 'Actual fit wrapper was not registered')
    fit_job_ref = G.ref(ROOT / 'audit' / (name + '_job.json'))
    fit_job = _closed_wrapper(fit_job_ref, entry_name='train_BrainHIVE_official_encoder_v1.py', task_id=job['task_id'],
        execution_review=fit['execution_review'], expected_name=name)
    require(fit_job['pid'] == fit['actual_training_pid'] and fit_job['wrapper_pid'] == fit['actual_wrapper_pid'], 'Actual fit process identities differ')
    provenance = G.bound(fit['input_provenance']); payload_digest = E.core_sha(base['payload_contract'])
    require(fit['input_provenance']['path'] == str(Path(job['output_root']) / 'input_provenance.json')
        and provenance['schema'] == 'BrainHIVE_official_encoder_input_provenance_v1'
        and provenance['status'] == 'ACTUAL_SOURCE_BOUND_TRAIN_INPUT_PAYLOADS_VERIFIED'
        and fit['payload_contract_sha256'] == provenance['payload_contract_sha256'] == payload_digest
        and provenance['source_sha256'] == base['source_sha256'] and provenance['task_id'] == job['task_id']
        and provenance['subject'] == job['subject'] and provenance['role'] == job['role']
        and provenance['registration_sha256'] == fit['registration_sha256']
        and provenance['registration_core_sha256'] == job['registration_core_sha256'], 'Actual fitted input provenance differs')
    epochs = [json.loads(line) for line in Path(paths['epoch_log']).read_text().splitlines()]
    E.validate_fit_metadata(job, fit, epochs)
    committed, pending = X.scan_training_epochs(job, fit)
    require(len(committed) == 25 and [r['row'] for r in committed] == epochs
        and fit['final_epoch_commit'] == committed[-1]['commit_ref'], 'Complete25epoch immutable payload chain required')
    E.validate_export_metadata(job, fit, epochs, record)
    require(record['status'] == 'OFFICIAL_OOF_ALL8270_ACTUAL_CONDITIONS_AND_OWN_AFFINE_EXPORTED'
        and record['actual_full8270_export'] is True and record['exporter_source_sha256'] == base['source_sha256'][Path(X.__file__).name]
        and record['training_producer_source_sha256'] == fit['producer_source_sha256']
        and record['payload_contract_sha256'] == payload_digest and record['runtime'] == base['runtime'], 'Actual complete official export/source chain differs')
    require(record['fit_chain']['final_epoch_commit'] == committed[-1]['commit_ref']
        and record['fit_chain']['input_provenance'] == fit['input_provenance']
        and record['fit_chain']['actual_fit_wrapper_job'] == fit_job_ref
        and record['fit_chain']['all25_committed_epoch_payload_SHAs_verified'] is True,
        'Export producer did not bind this exact closed fit/25epoch/provenance chain')
    export_review = G.bound(record['execution_review'])
    require(export_review['status'] == G.REVIEW_STATUS and export_review['source_sha256'] == base['source_sha256']
        and E.core_sha(export_review['payload_contract']) == payload_digest, 'Actual export review source/input chain differs')
    export_name = export_review['tasks'][job['task_id']]['export_job_name']
    export_ref = G.ref(ROOT / 'audit' / (export_name + '_job.json'))
    export_job = _closed_wrapper(export_ref, entry_name=Path(X.__file__).name, task_id=job['task_id'],
        execution_review=record['execution_review'], expected_name=export_name)
    require(record['wrapper_job']['path'] == export_ref['path'], 'Actual export wrapper path differs from producer receipt')
    require(file_ref(paths['condition_cache'])['sha256'] == record['cache_sha256'], 'Actual full condition cache file SHA differs')
    require_context(context, 'component_inputs'); check_memory(context)
    return dict(job=job, fit=fit, record=record, fit_wrapper=fit_job_ref, export_wrapper=export_ref,
        fit_pid=fit_job['pid'], export_pid=export_job['pid'], final_epoch_commit=committed[-1]['commit_ref'],
        retained_pending_transactions=pending)


def _assemble(layout, sources, target, *, _fixture=None):
    """Actual raw tensor assembly. Private dimensions are for CPU tests only."""
    import torch
    images, width, subject_count = (16540, 1024, 10) if _fixture is None else (_fixture['images'], _fixture['width'], _fixture['subjects'])
    jobs = layout['jobs']; mapping = layout['source_by_query']; queries = images * subject_count
    require(len(jobs) == 2 * subject_count and len(mapping) == queries, 'Complete task/query assembly coverage required')
    require(isinstance(target, torch.Tensor) and target.device.type == 'cpu' and target.dtype == torch.float32
        and tuple(target.shape) == (images, width) and bool(torch.isfinite(target).all()), 'Complete finite raw float32 teacher bank required')
    result = torch.empty((queries, width), dtype=torch.float32); affine_ids = torch.full((queries,), -1, dtype=torch.long)
    coverage = torch.zeros(queries, dtype=torch.int16); weights = []; biases = []; source_refs = []
    seen = set()
    for source in sources:
        job, cache, state, fit, exporter_sha = [source[k] for k in ('job', 'cache', 'model', 'fit', 'exporter_sha256')]
        require(job['role'] in ('official_oof0', 'official_oof1'), 'OldT/full/published condition caches cannot enter full-official assembly')
        index = layout['task_index'].get(job['task_id'])
        require(index is not None and job == jobs[index] and index not in seen, 'Wrong/mixed/duplicate task in OOF tensor join')
        seen.add(index)
        fixture = None if _fixture is None else dict(width=width, rows=len(job['expected_export_rows']))
        verified = X.validate_cache_tensors(cache, state, job, fit, exporter_sha, _fixture=fixture)
        held = job['expected_export_rows']; subject = layout['subjects'].index(job['subject'])
        destinations = [subject * images + row for row in held]
        require(all(mapping[q] == (index, row) for row, q in enumerate(destinations)), 'Wrong planned query/cache-row identity join')
        ids = torch.tensor(destinations, dtype=torch.long)
        require(bool((coverage[ids] == 0).all()), 'Duplicate tensor query assignment')
        result.index_copy_(0, ids, cache['condition']); coverage[ids] += 1; affine_ids[ids] = index
        weights.append((index, cache['affine_weight'].clone())); biases.append((index, cache['affine_bias'].clone()))
        source_refs.append(dict(task_id=job['task_id'], verified=verified, chain=source.get('chain')))
    require(len(seen) == len(jobs) and bool((coverage == 1).all()) and bool((affine_ids >= 0).all())
        and bool(torch.isfinite(result).all()), 'Missing/repeated/nonfinite complete OOF tensor coverage')
    weights = torch.stack([value for _, value in sorted(weights)]); biases = torch.stack([value for _, value in sorted(biases)])
    require(bool(torch.isfinite(weights).all()) and bool(torch.isfinite(biases).all())
        and bool((weights.abs() > 1e-6).all()), 'Own-affine nonfinite/zero/near-zero scale cannot be safely undone')
    return dict(condition=result, affine_ids=affine_ids, weight=weights, bias=biases, target=target.clone(),
        subjects=layout['subjects'], array_rows=list(range(images)), task_ids=[j['task_id'] for j in jobs],
        query_array_rows=torch.arange(images, dtype=torch.long).repeat(subject_count),
        query_subject_ids=torch.arange(subject_count, dtype=torch.long).repeat_interleave(images),
        query_cache_rows=torch.tensor([r[1] for r in mapping], dtype=torch.long), sources=source_refs,
        normalization='raw LayerNorm output; no additional L2 normalization', actual_full165400_verified=_fixture is None,
        fixture_only=_fixture is not None)


def load_training(context):
    """CPU-only full loader for a future admitted component trainer; no outputs.

    Returns condition[165400,1024], affine_ids[165400], weight/bias[20,1024],
    target[16540,1024] and compact exact global row/subject/cache-row mappings.
    The consumer moves its own batches/model to GPU after this function.
    """
    require_context(context, 'component_inputs'); check_sources(context); check_memory(context)
    layout = metadata_layout(context['bundle']); teacher = context['stage_spec']['component_teacher']
    require(teacher['train_identity_sha256'] == layout['train_identity_sha256']
        and teacher['image_manifest_sha256'] == E.P.PINS['image_manifest'][0], 'Teacher bank full-training identity mapping differs')
    target_path = bound_file(teacher['cache']); contract = G.bound(teacher['contract'])
    require(contract['shape'] == [16540, 1024] and contract['stored_dtype'] == 'float32'
        and contract['target_sha256'] == teacher['cache']['sha256']
        and contract['source'] == 'Frozen H14 FusionEncoderModel.vision_model before its residual adapter'
        and contract['additional_L2_normalization_at_inference'] is False, 'Full native raw F teacher cache contract differs')
    provenance = G.bound(teacher['provenance'])
    require(provenance['status'] == 'ACTUAL_FULL16540_TRAIN_F_TARGET_SOURCE_IDENTITY_AND_SHA_VERIFIED'
        and provenance['cache'] == teacher['cache'] and provenance['contract'] == teacher['contract']
        and provenance['train_identity_sha256'] == layout['train_identity_sha256']
        and provenance['payload_contract_sha256'] == E.core_sha(context['encoder_review']['payload_contract'])
        and provenance['official_test_inputs_or_scores_read'] is False, 'Actual full-training teacher source provenance required')
    producer = G.bound(provenance['producer_execution'])
    producer_path = Path(provenance['producer_source']['path'])
    require(producer_path.is_relative_to(ROOT) and context['source_sha256'].get(producer_path.relative_to(ROOT).as_posix())
        == provenance['producer_source']['sha256'], 'Actual teacher verification producer must be included in reviewed stage sources')
    require(producer['exit_code'] == 0 and producer['status'] == 'COMPLETED' and producer['actual_full16540_target_verification'] is True
        and producer['target_provenance_core_sha256'] == E.core_sha({k: v for k, v in provenance.items() if k != 'producer_execution'})
        and provenance['producer_source']['path'] in producer['command']
        and file_ref(provenance['producer_source']['path']) == provenance['producer_source'], 'No actual source-bound full teacher verification producer execution')
    chains = [fit_export_chain(context, job) for job in layout['jobs']]
    require(len({c['fit_wrapper']['path'] for c in chains}) == len({c['export_wrapper']['path'] for c in chains}) == 20,
        'All20 tasks require independent actual successful fit/export wrappers')
    require_context(context, 'component_inputs'); check_sources(context); check_memory(context)
    import torch
    require(torch.__version__ == context['encoder_review']['runtime']['torch_version'], 'Reviewed actual Torch runtime differs')
    target = torch.load(target_path, weights_only=True, map_location='cpu', mmap=True)
    def tensors():
        for chain in chains:
            require_context(context, 'component_inputs'); check_memory(context); job = chain['job']; fit = chain['fit']
            state = torch.load(job['outputs']['final_checkpoint'], weights_only=True, map_location='cpu', mmap=True)
            X.validate_checkpoint(state, job, fit)
            cache = torch.load(job['outputs']['condition_cache'], weights_only=True, map_location='cpu', mmap=True)
            verified = X.validate_cache_tensors(cache, state['model'], job, fit, chain['record']['exporter_source_sha256'])
            require(all(verified[k] == chain['record'][k] for k in
                ('condition_tensor_sha256', 'affine_weight_sha256', 'affine_bias_sha256')), 'Actual all-row tensor SHA differs from complete export receipt')
            yield dict(job=job, cache=cache, model=state['model'], fit=fit, exporter_sha256=chain['record']['exporter_source_sha256'],
                chain={k: chain[k] for k in ('fit_wrapper', 'export_wrapper', 'final_epoch_commit')})
    result = _assemble(layout, tensors(), target)
    require_context(context, 'component_inputs'); check_sources(context); check_memory(context)
    result.update(query_map=layout['query_map'], train_identity_sha256=layout['train_identity_sha256'],
        input_stage_review=context['stage_review_ref'], method_lock=context['method_lock_ref'], admission=context['admission_ref'],
        teacher_source=teacher, actual_EEG_payloads_loaded=False, actual_model_forward=False,
        official_test_inputs_or_scores_read=False, new_files_written=0)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-directory', type=Path, required=True)
    parser.add_argument('--execution-review', type=Path, required=True)
    args = parser.parse_args(argv)
    context = preflight(args.plan_directory, args.execution_review, purpose='component_inputs', entry_name=Path(__file__).name)
    data = load_training(context)
    print(json.dumps(dict(status='ALL165400_ACTUAL_RAW_OOF_AND20_OWN_AFFINES_AND16540_TRAIN_TARGETS_VERIFIED',
        condition_shape=list(data['condition'].shape), new_files_written=0)), flush=True)


if __name__ == '__main__': main()
