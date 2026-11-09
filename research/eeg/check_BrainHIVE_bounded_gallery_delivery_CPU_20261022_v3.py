"""Tiny metadata delivery fixtures; no real gallery scores or image payloads."""
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import shlex
import shutil
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import quote

import verify_BrainHIVE_bounded_gallery_delivery_CPU_20261022_v3 as entry


def raw(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(value)


def save(path, value):
    raw(path, (json.dumps(value, indent=2) + '\n').encode())


def csv_write(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(root):
    work = root / 'work'; gallery = work / 'gallery/delivery'; old = work / 'gallery/bounded_component_V_v1'
    image_root = root / 'private_images'; source_recipe = root / 'recipe.json'; storage_path = root / 'storage.json'
    storage = dict(allowed_private_roots=[str(image_root)])
    save(storage_path, storage)
    recipe = dict(run_names={a: 'protected_R4_' + a + '_bounded_v1' for a in entry.ARMS},
        storage_lifecycle_receipt=str(storage_path), storage_receipt_sha256=sha(storage_path))
    save(source_recipe, recipe)
    save(work / 'audit/main_storage_continuation_20261021_v1.json', dict(
        status='ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED', completed=list(entry.ARMS), failures={}, pid=99999999))
    save(work / 'comparison/bounded_v1/summary.json', dict(status='ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE'))
    save(old / 'coverage.json', dict(rows=2, generated_images=6, pages=1))
    raw(old / 'page_000.html', b'<html>Tiny old saved page</html>')
    targets, origins, rows, traces = [], [], [], []
    sources, scores_by_arm, manifests = {}, {}, {}
    for arm in entry.ARMS:
        scores_by_arm[arm], manifests[arm] = [], []
    for index, stimulus in enumerate(('q0.jpg', 'q1 #.jpg')):
        qid = 'sub-01/' + stimulus + '/R4'; target = work / 'gallery/V' / stimulus
        raw(target, b'Placeholder target, never opened by verifier')
        target_alias = old / 'assets/targets' / stimulus
        target_alias.parent.mkdir(parents=True, exist_ok=True); target_alias.hardlink_to(target)
        targets.append(dict(stimulus_id=stimulus, concept_id='concept',
            image_sha256=sha(target), image_path=str(target)))
        origins.append(dict(dataset_split='train', array_row=str(100 + index), stimulus_id=stimulus, concept_id='concept',
            local_concept_id='concept', image_zip=str(root / 'not_opened.zip'), image_member=stimulus,
            image_bytes='123', image_sha256=sha(target), alignment='author_release_order_plus_metadata_and_zip_receipt',
            independent_event_label_replay='False'))
        target_url = quote(Path(os.path.relpath(target_alias, gallery)).as_posix(), safe='/')
        row = dict(id=qid, subject='sub-01', concept='concept', target=target_url, targetFull=target_url,
            pageLocator='../bounded_component_V_v1/page_000.html', cells=[])
        trace = dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
            image_seed=1000 + index, target_image_zip=str(root / 'not_opened.zip'), target_image_member=stimulus,
            target_image_sha256=sha(target), score_target_path=str(target), page_locator=row['pageLocator'], position_in_old_page=index + 1)
        for number, arm in enumerate(entry.ARMS):
            run = recipe['run_names'][arm]
            path = image_root / run / 'sub-01' / (Path(stimulus).stem + '.webp')
            raw(path, ('Never decoded placeholder ' + arm + stimulus).encode())
            alias = old / 'assets/volume0' / path.relative_to(image_root)
            alias.parent.mkdir(parents=True, exist_ok=True); alias.hardlink_to(path)
            url = quote(Path(os.path.relpath(alias, gallery)).as_posix(), safe='/')
            value = .2 + .1 * number + .01 * index
            row['cells'].append(dict(label=arm.title(), full=url, thumb=url, dino=value, seed=value, object=value, blank=False))
            values = {k: (0 if k == 'DINO_Top1' else 1 if k == 'DINO_Top5' else value) for k in entry.METRICS}
            scored = dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept',
                EEG_source_stimulus_id=stimulus, target_image_path=str(target), generated_path=str(path), score_target_path=str(target),
                image_seed=1000 + index, **values)
            scores_by_arm[arm].append(scored)
            manifests[arm].append(dict(query_id=qid, subject='sub-01', stimulus_id=stimulus, concept_id='concept', array_row=100 + index,
                image_seed=1000 + index, target_image_sha256=sha(target), score_target_path=str(target), generated_path=str(path),
                generated_sha256=sha(path), repeat_protocol='R4', intervention='correct', EEG_source_stimulus_id=stimulus))
            trace.update({arm + '_generated_path': str(path), arm + '_generated_sha256': sha(path), arm + '_score_target_path': str(target)})
        rows.append(row); traces.append(trace)
    def jsonl(path, values):
        raw(path, ''.join(json.dumps(v) + '\n' for v in values).encode())
    jsonl(work / 'gallery/V/manifest.jsonl', targets)
    jsonl(work / 'data/generation_inputs/V_targets.jsonl', origins)
    means = []
    for arm in entry.ARMS:
        run = recipe['run_names'][arm]
        manifest = work / 'images' / run / 'manifest.jsonl'; scores = work / 'evaluation' / run / 'scores/per_query.csv'
        jsonl(manifest, manifests[arm]); csv_write(scores, scores_by_arm[arm])
        sources[arm] = dict(manifest_path=str(manifest), manifest_sha256=sha(manifest), score_csv_path=str(scores), score_csv_sha256=sha(scores))
        for trace in traces:
            trace.update({arm + '_' + k: v for k, v in sources[arm].items()})
        means.append(dict(arm=arm, subject='sub-01', **{k: sum(float(r[k]) for r in scores_by_arm[arm]) / 2 for k in entry.METRICS}))
        save(work / 'evaluation' / run / 'low_level/summary.json', dict(status='COMPLETED', expected_images=2, actual_present=2,
            verified_and_decoded=2, missing_count=0, corrupt_or_mismatched_count=0, duplicate_hash_groups=0,
            duplicate_images_beyond_first=0, manifest_sha256=sha(manifest), gallery_manifest_sha256=sha(work / 'gallery/V/manifest.jsonl'),
            source_sha256='9d7c4a4e0304490b7934dbdaa32271556572227507ba166ab5a9bbda01fe8b7f'))
    raw(gallery / 'data.js', ('window.GALLERY=' + json.dumps(dict(labels=[a.title() for a in entry.ARMS], blinded=False, rows=rows)) + ';\n').encode())
    raw(gallery / 'index.html', b'<html>Tiny static UI; no browser QA</html>'); raw(gallery / 'README.txt', b'Tiny fixture only')
    csv_write(gallery / 'query_provenance.csv', traces); csv_write(gallery / 'per_subject_metric_means.csv', means)
    for reference in ('parent', 'ordinary'):
        cases = []
        for row, trace in zip(rows, traces):
            cases.append({k: trace[k] for k in ('query_id', 'subject', 'stimulus_id', 'concept_id', 'page_locator')} |
                {metric + '_delta': row['cells'][2][field] - row['cells'][entry.ARMS.index(reference)][field]
                 for metric, field in (('DINO2AFC', 'dino'), ('SEED', 'seed'), ('ObjectF1', 'object'))} |
                dict(risk_generated_path=trace['risk_generated_path'], reference_generated_path=trace[reference + '_generated_path']))
        for direction, sign in (('improvement_descending', -1), ('regression_ascending', 1)):
            csv_write(gallery / f'risk_minus_{reference}_{direction}_all16600.csv', sorted(cases, key=lambda r: (sign * r['DINO2AFC_delta'], r['query_id'])))
    save(gallery / 'completion.json', dict(status='COMPLETED_FULL_BOUNDED_R4_GALLERY_INDEX', queries=2, generated_images=6,
        subject_metric_rows=3, old_pages_preserved=1, browser_UI_QA_performed=False, new_training_or_scores=False,
        primary_metrics_changed=False, copied_decoded_or_linked_images=False,
        file_sha256={name: sha(gallery / name) for name in entry.OUTPUT_NAMES},
        source_and_input_sha256={str(entry.ROOT / 'build_BrainHIVE_bounded_gallery_index_CPU_v1.py'): entry.BUILDER_SHA,
            str(entry.ROOT / 'build_offline_gallery.py'): entry.UI_SHA, str(old / 'page_000.html'): sha(old / 'page_000.html')}, sources=sources))
    return gallery, dict(work=work, count=2, subjects=1, concepts=1, per_concept=2, recipe=source_recipe)


def verify(item):
    return entry.verify_gallery(item[0], entry.Metadata(time.monotonic() + 30), _fixture=item[1])


def reseal(gallery, name):
    path = gallery / 'completion.json'; done = json.loads(path.read_text()); done['file_sha256'][name] = sha(gallery / name); save(path, done)


def rejects(call, text):
    try:
        call()
    except (ValueError, KeyError, OSError) as error:
        assert text in str(error), str(error)
    else:
        raise AssertionError('Invalid metadata accepted: ' + text)


def runtime_fixture(root):
    """Saved actual scalar-state/three-process schema, with OS resources injected.

    No runtime_snapshot mock: its source29, JSON/state, argv, scalar source,
    plan hash, no-new-V and formal-budget-call checks all execute normally.
    """
    repo = root / 'repo'; repo.mkdir(parents=True)
    original = entry.ROOT
    observation_path = original / 'audit/new_instance_restore_20261022_v2/formal_recovery_observation_20261022T223926Z.json'
    observed = json.loads(observation_path.read_bytes())
    running = [j for j in observed['formal_jobs'] if j['record']['status'] == 'RUNNING']
    assert len(running) == 1
    running = running[0]
    source = original / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json'
    raw(repo / source.relative_to(original), source.read_bytes())
    source_hashes = json.loads(source.read_bytes())['source_sha256']
    assert len(source_hashes) == 29 and sha(source) == entry.SOURCE29_SHA
    for name in source_hashes:
        shutil.copyfile(original / name, repo / name)
    recovery_name = 'queue_BrainHIVE_functional_recovery_20261022_v1.py'
    shutil.copyfile(original / recovery_name, repo / recovery_name)
    # Use the actual metadata's Linux prefix, not a hardcoded local mock schema.
    old_repo = observed['source_manifest']['path'].rsplit('/configs/', 1)[0]
    replacements = ((old_repo, str(repo)),
        ('/root/brainhive_reproduction_20261020_v1', str(root / 'work')),
        ('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1', str(root / 'data')))
    def relocate(value):
        if isinstance(value, dict): return {k: relocate(v) for k, v in value.items()}
        if isinstance(value, list): return [relocate(v) for v in value]
        if isinstance(value, str):
            for before, after in replacements:
                if value.startswith(before + '/') or value == before:
                    value = str(Path(after) / value[len(before):].lstrip('/'))
                    break
        return value
    state = relocate(copy.deepcopy(observed['recovery']['observed']))
    job = relocate(copy.deepcopy(running['record']))
    assert type(state['source_sha256']) is str and type(state['recovery_controller_source_sha256']) is str
    assert state['source_sha256'] != state['recovery_controller_source_sha256']
    plan_name = 'audit/functional_full_lifecycle_20261022_v1/four_arm_lifecycle_plan.json'
    plan = relocate(json.loads((original / plan_name).read_bytes()))
    plan_path = repo / plan_name; save(plan_path, plan)
    state['plan'] = dict(path=str(plan_path), sha256=sha(plan_path))
    state_path, job_path = repo / 'state.json', repo / 'job.json'
    save(state_path, state); save(job_path, job)
    specs = ((observed['recovery']['process'], 1), (running['wrapper'], state['pid']),
        (running['process'], job['wrapper_pid']))
    processes = {p['pid']: dict(pid=p['pid'], state=p['state'], startticks=100000 + p['pid'],
        parent_pid=parent, argv=relocate(shlex.split(p['cmdline']))) for p, parent in specs}
    return dict(repo=repo, state=state, job=job, state_path=state_path, job_path=job_path, plan=plan,
        plan_path=plan_path, processes=processes, observation=dict(path=str(observation_path), sha256=sha(observation_path)))


def check_runtime(root):
    import brainhive_functional_training_contract_v2 as C
    import inspect
    # This binds with one argument against the actual frozen helper signature.
    inspect.signature(C.formal_remaining).bind({})
    case = runtime_fixture(root)
    baseline_state, baseline_job = copy.deepcopy(case['state']), copy.deepcopy(case['job'])
    baseline_processes = copy.deepcopy(case['processes'])
    budget_calls = []
    memory = dict(effective_headroom_bytes=entry.MIN_MEMORY)
    remaining = [dict(root=str(root), required=1024, free=entry.MAX_RECEIPT_BYTES + 2048, metadata_reserve_only=True)]
    def budget_backend(plan):
        assert plan == case['plan']
        budget_calls.append(copy.deepcopy(plan))
        return copy.deepcopy(remaining)
    def snapshot():
        save(case['state_path'], case['state']); save(case['job_path'], case['job'])
        return entry.runtime_snapshot(case['state_path'], case['job_path'])
    def reset():
        case['state'], case['job'] = copy.deepcopy(baseline_state), copy.deepcopy(baseline_job)
        case['processes'] = copy.deepcopy(baseline_processes)
    with ExitStack() as stack:
        stack.enter_context(patch.object(entry, 'ROOT', case['repo']))
        stack.enter_context(patch.object(entry, 'process', side_effect=lambda pid: copy.deepcopy(case['processes'][pid])))
        stack.enter_context(patch.object(C, 'resource_headroom', side_effect=lambda: dict(memory)))
        stack.enter_context(patch.object(C, 'formal_remaining', side_effect=budget_backend))
        before = snapshot(); after = snapshot(); entry.same_runtime(before, after)
        assert before['source29_verified'] == 29 and len(budget_calls) == 2
        assert before['original_runner_source']['sha256'] == baseline_state['source_sha256']
        assert before['recovery_controller_source']['sha256'] == baseline_state['recovery_controller_source_sha256']
        for wrong in ({'wrong': '0' * 64}, ['0' * 64], '0' * 64, baseline_state['recovery_controller_source_sha256']):
            reset(); case['state']['source_sha256'] = wrong
            rejects(snapshot, 'Runner scalar source SHA')
        for wrong in ({}, [], '0' * 64, baseline_state['source_sha256']):
            reset(); case['state']['recovery_controller_source_sha256'] = wrong
            rejects(snapshot, 'Recovery controller actual source')
        for mode, expected in (('child', 'controller/wrapper/trainer'), ('parent', 'controller/wrapper/trainer'),
                ('stage', 'stage/job'), ('jobargv', 'training argv'), ('wrapperargv', 'wrapper name/training argv'),
                ('controllerargv', 'Recovery controller actual source'), ('plan', 'lifecycle plan changed')):
            reset()
            if mode == 'child': case['state']['child_pid'] += 1
            elif mode == 'parent': case['processes'][case['job']['pid']]['parent_pid'] += 1
            elif mode == 'stage': case['state']['current_stage'] = 'incorrect'
            elif mode == 'jobargv': case['job']['command'].append('--wrong')
            elif mode == 'wrapperargv': case['processes'][case['job']['wrapper_pid']]['argv'][-1] = 'wrong'
            elif mode == 'controllerargv':
                argv = case['processes'][case['state']['pid']]['argv']
                argv[3] = str(case['repo'] / 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py')
            elif mode == 'plan': case['state']['plan']['sha256'] = '0' * 64
            rejects(snapshot, expected)
        reset(); memory['effective_headroom_bytes'] = entry.MIN_MEMORY - 1
        rejects(snapshot, '12GiB CPU headroom'); memory['effective_headroom_bytes'] = entry.MIN_MEMORY
        # New V must be refused before the resource backend could inspect it.
        reset(); forbidden = Path(case['plan']['arms'][0]['inference_manifest']); raw(forbidden, b'not read')
        count_before = len(budget_calls)
        rejects(snapshot, 'New V generation has begun')
        assert len(budget_calls) == count_before
        forbidden.unlink()  # Only this newly-created temporary fixture file.
        remaining[0]['free'] = 0
        rejects(snapshot, 'complete functional lifecycle is not funded')
        remaining[0]['free'] = entry.MAX_RECEIPT_BYTES + 2048
        reset(); snapshot()
    return dict(actual_saved_state_and_job_schema=case['observation'],
        tested_function='runtime_snapshot', original_runner_scalar_SHA_and_distinct_recovery_scalar_SHA=True,
        source29_real_bytes_copied_and_checked=True, three_process_argv_and_parent_relations_checked=True,
        actual_formal_remaining_single_plan_signature_bound=True,
        injected_dependencies=['Linux /proc evidence', 'CPU memory snapshot', 'formal_remaining resource backend'],
        actual_server_processes_or_fresh_budget_checked=False,
        refused=['wrong_scalar_type_or_SHA_or_swapped_roles', 'child_parent_stage_argv_mismatch', 'plan_hash_drift',
                 'below_12GiB_CPU_headroom', 'nonempty_new_V_manifest_before_budget_backend', 'unfunded_lifecycle'])


def saved_schema_evidence(item):
    """Compare every tiny join input to the five saved actual old schemas.

    Values are not used to choose examples or decisions; only keys/types and
    the explicit gallery-vs-input ownership of array_row are checked.
    """
    sample_path = entry.ROOT / 'audit/gallery_index_actual_20261022_v1/actual_old_schema_samples.json'
    samples = json.loads(sample_path.read_bytes())
    work = item[1]['work']; gallery = item[0]
    def sample(suffix):
        found = [value for path, value in samples.items() if path.endswith(suffix)]
        assert len(found) == 1
        return found[0]
    def first_jsonl(path): return json.loads(path.read_text().splitlines()[0])
    def first_csv(path):
        with path.open(newline='', encoding='utf-8') as stream: return next(csv.DictReader(stream))
    inputs = {
        '/gallery/V/manifest.jsonl': first_jsonl(work / 'gallery/V/manifest.jsonl'),
        '/data/generation_inputs/V_targets.jsonl': first_jsonl(work / 'data/generation_inputs/V_targets.jsonl'),
        '/images/protected_R4_parent_bounded_v1/manifest.jsonl': first_jsonl(work / 'images/protected_R4_parent_bounded_v1/manifest.jsonl'),
        '/evaluation/protected_R4_parent_bounded_v1/scores/per_query.csv': first_csv(work / 'evaluation/protected_R4_parent_bounded_v1/scores/per_query.csv'),
        '/gallery/bounded_component_V_index_20261022_v1/query_provenance.csv': first_csv(gallery / 'query_provenance.csv')}
    for suffix, tiny in inputs.items():
        actual = sample(suffix)
        assert set(tiny) <= set(actual), ('Invented fixture field absent from actual schema', suffix, set(tiny) - set(actual))
        assert all(type(value) is type(actual[key]) for key, value in tiny.items()), ('Fixture field type differs', suffix)
    target = sample('/gallery/V/manifest.jsonl')
    assert set(target) == {'stimulus_id', 'image_path', 'image_sha256', 'concept_id'}
    assert 'array_row' not in target and 'array_row' not in inputs['/gallery/V/manifest.jsonl']
    origin = sample('/data/generation_inputs/V_targets.jsonl')
    assert type(origin['array_row']) is str and set(origin) == set(inputs['/data/generation_inputs/V_targets.jsonl'])
    return dict(path=str(sample_path), sha256=sha(sample_path), actual_schema_samples=5,
        all_tiny_join_field_names_and_types_match_actual_samples=True,
        target_gallery_exactly_four_fields_without_array_row=True, array_row_source='V_targets.jsonl',
        metric_values_used_for_selection_or_new_decisions=False)


def check():
    checks = []
    with tempfile.TemporaryDirectory(prefix='hive_delivery_CPU_') as temporary:
        base = Path(temporary)
        runtime = check_runtime(base / 'runtime')
        checks.append('actual_scalar_state_job_schema_full_runtime_snapshot_branch_with_narrow_OS_resource_injection')
        item = fixture(base / 'valid'); report = verify(item)
        saved_schema = saved_schema_evidence(item)
        assert report['status'] == 'CPU_TINY_FIXTURE_ONLY' and report['actual_full16600_delivery_verified'] is False
        assert report['rows'] == 2 and report['generated_files'] == 6 and report['subject_mean_rows'] == 3
        assert report['browser_UI_QA_performed'] is report['images_decoded'] is report['original_image_payloads_rehashed'] is False
        checks.append('real_builder_output_schema_sources_URLs_cases_means_lowlevel_tiny_delivery')
        checks.append('five_actual_old_join_schemas_gallery_four_fields_no_array_row_origin_row_string')
        for key, wrong in (('array_row', '999'), ('concept_id', 'wrong')):
            item = fixture(base / ('origin_' + key))
            path = item[1]['work'] / 'data/generation_inputs/V_targets.jsonl'
            origins = [json.loads(line) for line in path.read_text().splitlines()]
            origins[0][key] = wrong
            raw(path, ''.join(json.dumps(v) + '\n' for v in origins).encode())
            rejects(lambda item=item: verify(item), 'Original target array/concept/ZIP/member differs')
        checks.append('actual_origin_array_and_concept_mismatch_refused_without_gallery_array_field')
        item = fixture(base / 'hash'); raw(item[0] / 'index.html', b'changed')
        rejects(lambda: verify(item), 'SHA mismatch'); checks.append('output_SHA_drift_refused')
        item = fixture(base / 'new_V_scope')
        completion_path = item[0] / 'completion.json'; value = json.loads(completion_path.read_text())
        forbidden = item[1]['work'] / 'images/protected_R4_functional_alignment_v2_O/scores/per_query.csv'
        value['source_and_input_sha256'][str(forbidden)] = '0' * 64
        save(completion_path, value)
        rejects(lambda: verify(item), 'New functional score metadata')
        assert not forbidden.exists()
        checks.append('actual_new_V_run_prefix_refused_before_even_missing_file_open')
        for mode in ('sort', 'labels', 'path', 'seed', 'delta', 'low'):
            item = fixture(base / mode); gallery, options = item
            if mode in ('sort', 'labels', 'path'):
                data = json.loads((gallery / 'data.js').read_text()[len('window.GALLERY='):-2])
                if mode == 'sort': data['rows'].reverse()
                elif mode == 'labels': data['labels'][0] = 'Other'
                else: data['rows'][0]['cells'][0]['full'] = data['rows'][0]['cells'][0]['thumb'] = '../wrong.webp'
                raw(gallery / 'data.js', ('window.GALLERY=' + json.dumps(data) + ';\n').encode()); reseal(gallery, 'data.js')
                expected = {'sort': 'stable order', 'labels': 'columns', 'path': 'samefile'}[mode]
            elif mode in ('seed', 'delta'):
                name = 'query_provenance.csv' if mode == 'seed' else 'risk_minus_parent_improvement_descending_all16600.csv'
                with (gallery / name).open(newline='') as stream: rows = list(csv.DictReader(stream))
                rows[0]['image_seed' if mode == 'seed' else 'DINO2AFC_delta'] = '999'
                csv_write(gallery / name, rows); reseal(gallery, name)
                expected = 'identity differs' if mode == 'seed' else 'Case delta differs'
            else:
                low = options['work'] / 'evaluation/protected_R4_risk_bounded_v1/low_level/summary.json'
                value = json.loads(low.read_text()); value['missing_count'] = 1; save(low, value); expected = 'coverage/source'
            rejects(lambda item=item: verify(item), expected)
        checks.append('stable_sort_labels_path_seed_case_delta_and_incomplete_lowlevel_refused')
        identity = dict(controller=dict(pid=1, startticks=2, argv=['controller']), training=dict(pid=3, startticks=4, argv=['train']),
            wrapper=dict(pid=5, startticks=6, argv=['wrapper']), plan=dict(path='fixed', sha256='x'))
        entry.same_runtime(identity, copy.deepcopy(identity))
        altered = copy.deepcopy(identity); altered['training']['startticks'] += 1
        rejects(lambda: entry.same_runtime(identity, altered), 'process identity changed')
        checks.append('PID_reuse_or_runtime_identity_change_refused')
        rejects(lambda: entry.Metadata(time.monotonic() - 1).raw(item[0] / 'completion.json'), 'time bound')
        rejects(lambda: entry.Metadata(time.monotonic() + 30).raw(base / 'forbidden.webp'), 'Payload hashing')
        checks.append('finite_deadline_and_image_payload_hash_read_refused')
        assert 'torch' not in sys.modules and 'PIL' not in sys.modules
    return dict(status='PASS_TINY_DELIVERY_AND_ACTUAL_RUNTIME_SCHEMA_NO_ACTUAL_SERVER_OR_GALLERY_READ', checks=checks,
        runtime_snapshot_fixture=runtime,
        saved_actual_join_schema_fixture=saved_schema,
        source_sha256={Path(entry.__file__).name: sha(entry.__file__), Path(__file__).name: sha(__file__)},
        actual_full16600_delivery_verified=False, actual_processes_or_formal_remaining_checked=False,
        real_V_scores_read=False, GPU_used=False, images_decoded_or_real_payloads_read=False, browser_UI_QA_performed=False)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--receipt', type=Path); args = parser.parse_args()
    if args.receipt: assert not args.receipt.exists()
    result = check()
    if args.receipt: raw(args.receipt, (json.dumps(result, indent=2) + '\n').encode())
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
