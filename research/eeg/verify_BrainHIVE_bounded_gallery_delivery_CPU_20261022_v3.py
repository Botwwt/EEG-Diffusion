"""Verify the saved old three-arm gallery while the separate O fit continues.

Metadata and samefile/stat only: no image bytes, decoder, tensors, GPU calls,
new functional-V score files, refitting or scientific decision computation.
"""
import argparse
import collections
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import quote, unquote, urlsplit

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')
ARMS = ('parent', 'ordinary', 'risk')
METRICS = ('DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5', 'EffNet', 'CapSim', 'ObjectF1', 'SEED')
MAX_SECONDS, MAX_RECEIPT_BYTES = 900, 16 << 20
MAX_FILE_BYTES, MAX_READ_BYTES = 128 << 20, 2 << 30
MIN_MEMORY = 12 << 30
BUILDER_SHA = '7c3d7ce34a8a2ac1f811c23d75a6eff3f165be174be5d6611a42d44ac870128b'
UI_SHA = 'f046276975824e02c0b592199ebd27fa31babe895097f72166f71c5a96ca1523'
SOURCE29_SHA = 'f695c233742ca5dab414d298e7dc968a5da28e1b22e70f57005b75198831f025'
OLD_RECIPE_SHA = '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
SCHEMA = 'BrainHIVE_bounded_gallery_delivery_CPU_20261022_v3'
OUTPUT_NAMES = {'data.js', 'index.html', 'README.txt', 'query_provenance.csv', 'per_subject_metric_means.csv'} | {
    f'risk_minus_{arm}_{direction}_all16600.csv' for arm in ('parent', 'ordinary')
    for direction in ('improvement_descending', 'regression_ascending')}


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Metadata:
    def __init__(self, deadline):
        self.deadline, self.bytes_read, self.pins = deadline, 0, {}
    def tick(self):
        require(time.monotonic() < self.deadline, 'Finite metadata verification time bound reached')
    def raw(self, path, expected=None):
        self.tick()
        path = Path(path)
        require(path.suffix in ('.json', '.jsonl', '.csv', '.js', '.html', '.txt', '.py', '.log'), 'Payload hashing is forbidden: ' + str(path))
        require(path.is_file() and 0 < path.stat().st_size <= MAX_FILE_BYTES, 'Missing/oversized metadata: ' + str(path))
        raw = path.read_bytes()
        self.bytes_read += len(raw)
        require(self.bytes_read <= MAX_READ_BYTES, 'Finite metadata input byte budget reached')
        actual = hashlib.sha256(raw).hexdigest()
        require(expected is None or actual == expected, 'Metadata SHA mismatch: ' + str(path))
        require(str(path) not in self.pins or self.pins[str(path)] == actual, 'Metadata changed during verification')
        self.pins[str(path)] = actual
        return raw
    def json(self, path, expected=None):
        return json.loads(self.raw(path, expected))
    def csv(self, path, expected=None):
        return csv.DictReader(io.StringIO(self.raw(path, expected).decode('utf-8'), newline=''))
    def jsonl(self, path, expected=None):
        return [json.loads(line) for line in self.raw(path, expected).decode('utf-8').splitlines()]
    def unchanged(self):
        for path, expected in list(self.pins.items()):
            self.raw(path, expected)


def process(pid):
    require(os.name == 'posix', 'Actual Linux /proc evidence is required')
    root = Path('/proc') / str(pid)
    stat = (root / 'stat').read_text().rsplit(') ', 1)[1].split()
    argv = [x.decode() for x in (root / 'cmdline').read_bytes().split(b'\0') if x]
    require(stat[0] not in ('Z', 'X') and argv, 'Required process is inactive')
    return dict(pid=int(pid), state=stat[0], startticks=int(stat[19]), parent_pid=int(stat[1]), argv=argv)


def runtime_snapshot(state_path, job_path):
    """Dynamic training snapshots are observations, never immutable score pins."""
    source_path = ROOT / 'configs/BrainHIVE_new_instance_continuation_sources_20261022_v2.json'
    source_raw = source_path.read_bytes()
    require(hashlib.sha256(source_raw).hexdigest() == SOURCE29_SHA, 'Bound SOURCE29 manifest changed')
    manifest = json.loads(source_raw)
    require(len(manifest['source_sha256']) == 29, 'Expected exactly29 frozen sources')
    for name, expected in manifest['source_sha256'].items():
        require(Path(name).name == name and hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected,
                'Frozen source29 changed: ' + name)
    import brainhive_functional_training_contract_v2 as C  # Pure stdlib contract, no Torch.
    memory = C.resource_headroom()
    require(memory['effective_headroom_bytes'] >= MIN_MEMORY, 'Gallery verification requires the assigned12GiB CPU headroom')
    state_raw, job_raw = (Path(p).read_bytes() for p in (state_path, job_path))
    state, job = json.loads(state_raw), json.loads(job_raw)
    controller, worker, wrapper = process(state['pid']), process(job['pid']), process(job['wrapper_pid'])
    require(job['status'] == 'RUNNING', 'Current O job is not RUNNING')
    require(state['child_pid'] == wrapper['pid'] and worker['parent_pid'] == wrapper['pid']
            and wrapper['parent_pid'] == controller['pid'], 'Runtime controller/wrapper/trainer identity differs')
    require(job['name'] == state['current_stage'], 'Runtime stage/job name differs')
    recipe_path = ROOT / 'configs/functional_full_lifecycle_20261022_v1/O.json'
    argv = worker['argv']
    require(str(ROOT / 'train_BrainHIVE_functional_alignment_v2.py') in argv
            and '--recipe' in argv and Path(argv[argv.index('--recipe') + 1]) == recipe_path,
            'Actual live process is not the bound O training recipe')
    require(str(ROOT / 'run_gpu_job_v2.py') in wrapper['argv'], 'Actual live wrapper is not the frozen GPU job wrapper')
    require(job['command'] == worker['argv'], 'Actual training argv differs from the running job receipt')
    wrapper_argv = wrapper['argv']
    require('--name' in wrapper_argv and wrapper_argv[wrapper_argv.index('--name') + 1] == job['name']
            and '--' in wrapper_argv and wrapper_argv[wrapper_argv.index('--') + 1:] == job['command'],
            'Actual wrapper name/training argv differs from the running job receipt')
    # Q.Runner.__init__ writes the lexical original Runner SHA as one scalar;
    # RecoveryRunner records its distinct current entry source separately.
    runner = ROOT / 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py'
    require(type(state['source_sha256']) is str and state['source_sha256'] == manifest['source_sha256'][runner.name]
            == hashlib.sha256(runner.read_bytes()).hexdigest(), 'Live original Runner scalar source SHA differs')
    recovery = ROOT / 'queue_BrainHIVE_functional_recovery_20261022_v1.py'
    scripts = [Path(a) for a in controller['argv'] if a.endswith('.py') and Path(a).parent == ROOT]
    require(scripts == [recovery] and type(state['recovery_controller_source_sha256']) is str
            and hashlib.sha256(recovery.read_bytes()).hexdigest() == state['recovery_controller_source_sha256'],
            'Recovery controller actual source differs from state')
    binding = state['plan']
    raw = Path(binding['path']).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == binding['sha256'], 'Live sealed lifecycle plan changed')
    plan = json.loads(raw)
    # Current O fit precedes any new V inference. Never let the helper enter an
    # image-prefix payload audit or open new functional V metric files here.
    for entry in plan['arms']:
        path = Path(entry['inference_manifest'])
        require(not path.exists() or path.stat().st_size == 0, 'New V generation has begun; this training-window verifier stops')
    remaining = C.formal_remaining(plan)
    require(all(row['free'] >= row['required'] for row in remaining), 'Current complete functional lifecycle is not funded')
    return dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), controller=controller, training=worker,
        wrapper=wrapper, current_stage=state.get('current_stage'), source_manifest=dict(path=str(source_path), sha256=SOURCE29_SHA),
        state_reference=dict(path=str(state_path), sha256=hashlib.sha256(state_raw).hexdigest()),
        job_reference=dict(path=str(job_path), sha256=hashlib.sha256(job_raw).hexdigest()),
        source29_verified=29, original_runner_source=dict(path=str(runner), sha256=state['source_sha256']),
        recovery_controller_source=dict(path=str(recovery), sha256=state['recovery_controller_source_sha256']),
        plan=binding, memory=memory, formal_remaining_storage=remaining,
        current_new_V_images_or_scores_read=False)


def same_runtime(before, after):
    for key in ('controller', 'training', 'wrapper'):
        require((before[key]['pid'], before[key]['startticks'], before[key]['argv']) ==
                (after[key]['pid'], after[key]['startticks'], after[key]['argv']), 'O/runtime process identity changed during verification')
    require(before['plan'] == after['plan'], 'Live plan changed during verification')


def asset(url, expected_path, gallery, old, roots, *, target=False):
    parsed = urlsplit(url)
    require(not parsed.scheme and not parsed.netloc and not parsed.query and not parsed.fragment, 'External/ambiguous gallery URL')
    source = Path(expected_path)
    if target:
        expected_asset = old / 'assets/targets' / source.name
    else:
        matches = [(i, source.relative_to(root)) for i, root in enumerate(roots) if source.is_relative_to(root)]
        require(len(matches) == 1, 'Generated path is outside one registered old private root')
        volume, relative = matches[0]
        expected_asset = old / 'assets' / ('volume' + str(volume)) / relative
    expected_url = quote(Path(os.path.relpath(expected_asset, gallery)).as_posix(), safe='/')
    require(url == expected_url and (gallery / unquote(url)).is_file() and source.is_file()
            and os.path.samefile(gallery / unquote(url), source), 'Gallery URL/provenance samefile identity differs')
    details = source.stat()
    return details.st_dev, details.st_ino


def verify_gallery(gallery, metadata, *, _fixture=None):
    """Production is full16600; tiny count is internal software testing only."""
    gallery = Path(gallery)
    fixture = _fixture is not None
    work = WORK if not fixture else _fixture['work']
    count = 16600 if not fixture else _fixture['count']
    subjects_n, concepts_n, per_concept = (10, 166, 10) if not fixture else (_fixture['subjects'], _fixture['concepts'], _fixture['per_concept'])
    require(gallery.parent == work / 'gallery' and gallery.is_dir(), 'Gallery must be an existing direct private WORK/gallery child')
    completed = metadata.json(gallery / 'completion.json')
    require(completed['status'] == 'COMPLETED_FULL_BOUNDED_R4_GALLERY_INDEX' and completed['queries'] == count
            and completed['generated_images'] == 3 * count and completed['subject_metric_rows'] == 3 * subjects_n
            and completed['old_pages_preserved'] == math.ceil(count / 100), 'Index completion denominator differs')
    require(completed['browser_UI_QA_performed'] is False and completed['new_training_or_scores'] is False
            and completed['primary_metrics_changed'] is False and completed['copied_decoded_or_linked_images'] is False,
            'Index scope changed')
    require(set(completed['file_sha256']) == OUTPUT_NAMES, 'Missing/extra delivery outputs')
    require({p.name for p in gallery.iterdir()} == OUTPUT_NAMES | {'completion.json'}, 'Unregistered gallery output')
    require(sum(p.stat().st_size for p in gallery.iterdir()) <= 128 << 20, 'Original gallery128MiB output cap exceeded')
    for name, expected in completed['file_sha256'].items():
        metadata.raw(gallery / name, expected)
    # These are builder-recorded old metadata/source paths, never image payloads.
    for path, expected in completed['source_and_input_sha256'].items():
        require(not any('functional_alignment_v2' in part for part in Path(path).parts), 'New functional score metadata is outside this verifier')
        metadata.raw(path, expected)
    for name, expected in (('build_BrainHIVE_bounded_gallery_index_CPU_v1.py', BUILDER_SHA), ('build_offline_gallery.py', UI_SHA)):
        require(completed['source_and_input_sha256'].get(str(ROOT / name)) == expected, 'Builder/UI source binding differs')
    recipe_path = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json' if not fixture else _fixture['recipe']
    recipe = metadata.json(recipe_path, OLD_RECIPE_SHA if not fixture else None)
    storage = metadata.json(recipe['storage_lifecycle_receipt'], recipe['storage_receipt_sha256'])
    roots = [Path(p) for p in storage['allowed_private_roots']]
    require(set(recipe['run_names']) == set(ARMS), 'Wrong old arm names')
    old = work / 'gallery/bounded_component_V_v1'
    comparison = metadata.json(work / 'comparison/bounded_v1/summary.json')
    require(comparison['status'] == 'ALL_THREE_FULL16600_QUERY_IMAGE_ARMS_PAIRED_COMPARISON_COMPLETE', 'Old three-arm comparison not closed')
    main = metadata.json(work / 'audit/main_storage_continuation_20261021_v1.json')
    require(main['status'] == 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
            and set(main['completed']) == set(ARMS) and not main['failures'], 'Old three-arm lifecycle not complete')
    if not fixture:
        require(main['recipe_sha256'] == main['recipe_sha256_at_exit'] == OLD_RECIPE_SHA, 'Old terminal recipe identity differs')
        try:
            process(main['pid'])
        except (FileNotFoundError, ProcessLookupError, ValueError):
            pass
        else:
            raise ValueError('Old main PID is still active; do not verify a moving gallery')
    coverage = metadata.json(old / 'coverage.json')
    require(coverage['rows'] == count and coverage['generated_images'] == 3 * count
            and coverage['pages'] == math.ceil(count / 100), 'Old paged gallery coverage differs')
    for i in range(coverage['pages']):
        path = old / f'page_{i:03d}.html'
        require(str(path) in completed['source_and_input_sha256'], 'Old page omitted from bound delivery')
    target_path = work / 'gallery/V/manifest.jsonl'
    targets = {r['stimulus_id']: r for r in metadata.jsonl(target_path)}
    target_metadata = {r['stimulus_id']: r for r in metadata.jsonl(work / 'data/generation_inputs/V_targets.jsonl')}
    require(set(targets) == set(target_metadata), 'Target ZIP/gallery identity sets differ')
    data_raw = metadata.raw(gallery / 'data.js').decode('utf-8')
    require(data_raw.startswith('window.GALLERY=') and data_raw.endswith(';\n'), 'data.js must be static JSON assignment')
    data = json.loads(data_raw[len('window.GALLERY='):-2])
    require(data['labels'] == [a.title() for a in ARMS] and data['blinded'] is False and len(data['rows']) == count, 'Wrong gallery columns/row count')
    rows = data['rows']
    ids = [row['id'] for row in rows]
    require(ids == sorted(set(ids)) and len(ids) == count, 'Gallery IDs are not complete unique stable order')
    by_id = dict(zip(ids, rows))
    subjects = sorted({r['subject'] for r in rows})
    require(len(subjects) == subjects_n and len({str(r['concept']) for r in rows}) == concepts_n, 'Gallery subject/concept coverage differs')
    require(set(collections.Counter((r['subject'], str(r['concept'])) for r in rows).values()) == {per_concept}, 'Gallery subject/concept cells incomplete')
    traces, all_paths, inodes = {}, set(), set()
    for index, trace in enumerate(metadata.csv(gallery / 'query_provenance.csv')):
        metadata.tick()
        require(index < count and trace['query_id'] == ids[index], 'Provenance query order/coverage differs')
        row, qid = by_id[trace['query_id']], trace['query_id']
        require(qid == trace['subject'] + '/' + trace['stimulus_id'] + '/R4' and row['subject'] == trace['subject']
                and str(row['concept']) == trace['concept_id'], 'Provenance stimulus/subject/concept identity differs')
        target = targets[trace['stimulus_id']]
        target_origin = target_metadata[trace['stimulus_id']]
        # The saved gallery manifest has stimulus/path/SHA/concept only. EEG
        # array identity belongs to V_targets, then to all three image manifests.
        require(target['image_sha256'] == trace['target_image_sha256']
                and str(target['concept_id']) == trace['concept_id'] and target['image_path'] == trace['score_target_path'], 'Target concept/path/SHA differs')
        require(target_origin['image_zip'] == trace['target_image_zip'] and target_origin['image_member'] == trace['target_image_member']
                and target_origin['image_sha256'] == target['image_sha256'] and str(target_origin['array_row']) == trace['array_row']
                and str(target_origin['concept_id']) == trace['concept_id'], 'Original target array/concept/ZIP/member differs')
        page = f'../bounded_component_V_v1/page_{index // 100:03d}.html'
        require(row['pageLocator'] == trace['page_locator'] == page and int(trace['position_in_old_page']) == index % 100 + 1,
                'Old page locator differs')
        require(row['target'] == row['targetFull'], 'Target thumbnail silently differs')
        asset(row['target'], trace['score_target_path'], gallery, old, roots, target=True)
        require(len(row['cells']) == 3 and [c['label'] for c in row['cells']] == data['labels'], 'Wrong per-query3 cells')
        for arm, cell in zip(ARMS, row['cells']):
            path = trace[arm + '_generated_path']
            require(path not in all_paths and trace[arm + '_score_target_path'] == trace['score_target_path'], 'Generated path reused or scoring target differs')
            require(cell['full'] == cell['thumb'] and type(cell['blank']) is bool
                    and all(type(cell[k]) in (int, float) and math.isfinite(cell[k]) for k in ('dino', 'seed', 'object')), 'Bad cell values/thumbnail')
            inode = asset(cell['full'], path, gallery, old, roots)
            require(inode not in inodes, 'Same underlying generated file reused across gallery')
            all_paths.add(path); inodes.add(inode)
            source = completed['sources'][arm]
            for key, value in source.items():
                require(trace[arm + '_' + key] == value, 'Provenance source path/SHA differs')
        traces[qid] = trace
    require(len(traces) == count, 'Incomplete provenance denominator')
    # Full manifests and saved score rows are OLD CLOSED arms only. No feature
    # tensors, generated image payloads or new functional-V scores are opened.
    coverage_reports, numeric_means = {}, {}
    for arm, cell_index in zip(ARMS, range(3)):
        run = recipe['run_names'][arm]
        require(run == 'protected_R4_' + arm + '_bounded_v1', 'Unexpected old run')
        manifest = work / 'images' / run / 'manifest.jsonl'
        evaluation = work / 'evaluation' / run
        source = completed['sources'][arm]
        require(source['manifest_path'] == str(manifest) and source['score_csv_path'] == str(evaluation / 'scores/per_query.csv'), 'Noncanonical closed source')
        manifest_rows = metadata.jsonl(manifest, source['manifest_sha256'])
        require(len(manifest_rows) == count and {r['query_id'] for r in manifest_rows} == set(ids), 'Incomplete source manifest')
        for item in manifest_rows:
            trace = traces[item['query_id']]
            for key in ('subject', 'stimulus_id', 'array_row', 'image_seed', 'target_image_sha256', 'score_target_path'):
                require(str(item[key]) == trace[key], 'Manifest/provenance identity differs: ' + key)
            require(item['generated_path'] == trace[arm + '_generated_path'] and item['generated_sha256'] == trace[arm + '_generated_sha256'], 'Manifest generated path/SHA differs')
            require(item['repeat_protocol'] == 'R4' and item['intervention'] == 'correct'
                    and item['EEG_source_stimulus_id'] == trace['stimulus_id'], 'Old EEG condition protocol differs')
        sums, seen = collections.defaultdict(list), set()
        for scored in metadata.csv(evaluation / 'scores/per_query.csv', source['score_csv_sha256']):
            qid = scored['query_id']; require(qid in by_id and qid not in seen, 'Duplicate/unknown saved score ID')
            seen.add(qid); trace = traces[qid]; cell = by_id[qid]['cells'][cell_index]
            for key in ('subject', 'stimulus_id', 'concept_id', 'image_seed', 'score_target_path'):
                require(scored[key] == trace[key], 'Saved score query identity differs: ' + key)
            require(scored['EEG_source_stimulus_id'] == trace['stimulus_id'] and scored['target_image_path'] == trace['score_target_path'], 'Saved EEG/target identity differs')
            require(scored['generated_path'] == trace[arm + '_generated_path'] and scored['score_target_path'] == trace['score_target_path']
                    and scored['image_seed'] == trace['image_seed'], 'Saved score path/seed differs')
            for field, metric in (('dino', 'DINO2AFC'), ('seed', 'SEED'), ('object', 'ObjectF1')):
                require(float(scored[metric]) == cell[field], 'Gallery cell differs from saved score')
            for metric in METRICS:
                value = float(scored[metric]); require(math.isfinite(value), 'Nonfinite old score')
                sums[(scored['subject'], metric)].append(value)
        require(seen == set(ids), 'Saved score denominator differs')
        for subject in subjects:
            for metric in METRICS:
                values = sums[(subject, metric)]
                require(len(values) == count // subjects_n, 'Incomplete subject saved scores')
                numeric_means[(arm, subject, metric)] = math.fsum(values) / len(values)
        low = metadata.json(evaluation / 'low_level/summary.json')
        metadata.raw(ROOT / 'score_low_level.py', '9d7c4a4e0304490b7934dbdaa32271556572227507ba166ab5a9bbda01fe8b7f')
        require(low['status'] == 'COMPLETED' and all(low[k] == count for k in ('expected_images', 'actual_present', 'verified_and_decoded'))
                and low['missing_count'] == low['corrupt_or_mismatched_count'] == 0
                and low['manifest_sha256'] == source['manifest_sha256']
                and low['gallery_manifest_sha256'] == metadata.pins[str(target_path)]
                and low['source_sha256'] == '9d7c4a4e0304490b7934dbdaa32271556572227507ba166ab5a9bbda01fe8b7f', 'Closed low-level coverage/source differs')
        fields = ('status', 'expected_images', 'actual_present', 'verified_and_decoded', 'missing_count',
                  'corrupt_or_mismatched_count', 'duplicate_hash_groups', 'duplicate_images_beyond_first', 'manifest_sha256', 'gallery_manifest_sha256')
        require(all(type(low[k]) is int and low[k] >= 0 for k in fields if k.endswith('_count') or k.startswith('duplicate_')), 'Invalid coverage count')
        coverage_reports[arm] = {k: low[k] for k in fields}
    means_seen = set()
    for row in metadata.csv(gallery / 'per_subject_metric_means.csv'):
        key = row['arm'], row['subject']
        require(key not in means_seen and key[0] in ARMS and key[1] in subjects, 'Duplicate/unknown subject mean')
        means_seen.add(key)
        for metric in METRICS:
            require(abs(float(row[metric]) - numeric_means[(*key, metric)]) <= 2e-6, 'Subject mean differs from full saved scores')
    require(len(means_seen) == 3 * subjects_n, 'Incomplete30 subject means')
    case_counts = {}
    for reference in ('parent', 'ordinary'):
        for direction, sign in (('improvement_descending', -1), ('regression_ascending', 1)):
            name = f'risk_minus_{reference}_{direction}_all16600.csv'
            seen, previous = set(), None
            for item in metadata.csv(gallery / name):
                qid = item['query_id']; require(qid in by_id and qid not in seen, 'Duplicate/unknown case query')
                seen.add(qid); row, trace = by_id[qid], traces[qid]
                for key in ('subject', 'stimulus_id', 'concept_id', 'page_locator'):
                    require(item[key] == trace[key], 'Case identity/page differs')
                require(item['risk_generated_path'] == trace['risk_generated_path']
                        and item['reference_generated_path'] == trace[reference + '_generated_path'], 'Case generated paths differ')
                for metric, field in (('DINO2AFC', 'dino'), ('SEED', 'seed'), ('ObjectF1', 'object')):
                    expected = row['cells'][2][field] - row['cells'][ARMS.index(reference)][field]
                    require(float(item[metric + '_delta']) == expected, 'Case delta differs from gallery cells')
                ordering = sign * float(item['DINO2AFC_delta']), qid
                require(previous is None or previous <= ordering, 'Case fixed delta/ID ordering differs')
                previous = ordering
            require(len(seen) == count, 'Incomplete case full denominator')
            case_counts[name] = len(seen)
    return dict(status='CPU_TINY_FIXTURE_ONLY' if fixture else 'FULL_OLD_THREE_ARM_GALLERY_DELIVERY_VERIFIED',
        actual_full16600_delivery_verified=not fixture, rows=count, generated_files=3 * count, subject_mean_rows=len(means_seen),
        old_pages_verified=coverage['pages'], case_rows=case_counts, coverage_statistics=coverage_reports,
        output=dict(path=str(gallery), completion_sha256=metadata.pins[str(gallery / 'completion.json')],
                    total_existing_output_bytes=sum(p.stat().st_size for p in gallery.iterdir())),
        metadata_samefile_links_verified=True, original_image_payloads_rehashed=False, images_decoded=False,
        browser_UI_QA_performed=False, R4_permuted_or_missing_controls_present=False,
        server_original_asset_dependency=True, portable_single_HTML_package=False, perceptual_quality_established=False,
        human_quality_ratings_performed=False, new_functional_V_scores_read=False, new_scores_computed=False,
        limits=['Metadata/file identity and existing closed coverage only; browser rendering remains unverified.',
            'Existing low-level duplicate counts are per arm; no fresh global decoded-pixel duplicate computation.',
            'DINO/ObjectF1/blank filters are metric proxies; no manual error labels or perceptual NI evidence.'])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', type=Path, required=True)
    parser.add_argument('--launch', type=Path, required=True)
    parser.add_argument('--run-log', type=Path, required=True)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--expected-source-sha256', required=True, help='Reviewed verifier SHA, pinned by the caller before launch.')
    parser.add_argument('--runtime-state', type=Path, default=ROOT / 'audit/functional_full_lifecycle_20261022_v1/formal_recovery_queue.json')
    parser.add_argument('--job', type=Path, default=ROOT / 'audit/BrainHIVE_functional_v2_O_expanded_lifecycle_20261022_v1_exact_resume1_job.json')
    args = parser.parse_args()
    started = time.monotonic()
    own_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    require(own_sha == args.expected_source_sha256, 'Reviewed verifier source SHA differs')
    metadata = Metadata(started + MAX_SECONDS)
    require(not args.receipt.exists() and args.receipt.parent.is_dir(), 'Fresh receipt with existing private parent required')
    os.environ.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    if os.name == 'posix':
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (MAX_SECONDS, MAX_SECONDS))
    before = runtime_snapshot(args.runtime_state, args.job)
    launch = metadata.json(args.launch)
    try:
        process(launch['timeout_pid'])
    except (FileNotFoundError, ProcessLookupError, ValueError):
        pass
    else:
        raise ValueError('Old gallery builder timeout PID is still active; wait for its terminal state')
    require(launch['source_sha256'] == BUILDER_SHA and launch['output'] == str(args.gallery)
            and launch['current_GPU_training_process_verified_live'] == before['training']['pid'], 'Actual launch/source/training evidence differs')
    command = launch['command']; script = str(ROOT / 'build_BrainHIVE_bounded_gallery_index_CPU_v1.py')
    require(script in command, 'Launch did not invoke the actual old builder')
    tail = command[command.index(script) + 1:]
    require(len(tail) == 4 and dict(zip(tail[::2], tail[1::2])) == {
        '--recipe': str(ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'), '--output': str(args.gallery)}
        and launch['maximum_output_bytes'] == 128 << 20 and 0 < launch['maximum_seconds'] <= MAX_SECONDS,
        'Builder launch CLI or finite limits differ')
    require(launch['actual_memory_before']['effective_headroom_bytes'] >= MIN_MEMORY
            and all(row['free'] >= row['required'] for row in launch['formal_remaining_storage_before']), 'Launch was not resource-admitted')
    result = verify_gallery(args.gallery, metadata)
    log = metadata.json(args.run_log)
    require(log['status'] == 'COMPLETED_FULL_BOUNDED_R4_GALLERY_INDEX' and log['output'] == str(args.gallery)
            and log['encoded_bytes'] == result['output']['total_existing_output_bytes'], 'Builder completion log/encoded bytes differ')
    result['builder_run_log'] = dict(path=str(args.run_log), sha256=metadata.pins[str(args.run_log)], timeout_PID_verified_inactive=True)
    migration = ROOT / 'audit/new_instance_restore_20261022_v2/migrated_results_verified.json'
    metadata.raw(migration)
    result['existing_migration_payload_verification'] = dict(path=str(migration), sha256=metadata.pins[str(migration)],
        reused_historical_evidence_not_new_image_payload_verification=True)
    metadata.unchanged()
    after = runtime_snapshot(args.runtime_state, args.job)
    same_runtime(before, after)
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == own_sha, 'Verifier source changed during execution')
    result.update(schema=SCHEMA, UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), source_sha256=own_sha,
        runtime_before=before, runtime_after=after, immutable_metadata_sha256=metadata.pins, GPU_used=False,
        job_stop_restart_or_mutation_performed=False, source29_or_recipe_modified=False,
        budget=dict(maximum_seconds=MAX_SECONDS, seconds=time.monotonic() - started, maximum_receipt_bytes=MAX_RECEIPT_BYTES,
                    maximum_metadata_bytes_read=MAX_READ_BYTES, metadata_bytes_read=metadata.bytes_read))
    encoded = (json.dumps(result, indent=2, allow_nan=False) + '\n').encode()
    require(len(encoded) <= MAX_RECEIPT_BYTES, 'Finite receipt cap reached')
    device = args.receipt.parent.stat().st_dev
    matching = [r for r in after['formal_remaining_storage'] if Path(r['root']).stat().st_dev == device]
    require(len(matching) == 1 and matching[0]['free'] - matching[0]['required'] >= MAX_RECEIPT_BYTES, 'Receipt cap would consume retained formal budget')
    metadata.tick()
    with args.receipt.open('xb') as stream:
        stream.write(encoded)
        stream.flush(); os.fsync(stream.fileno())
    print(json.dumps(dict(status=result['status'], rows=result['rows'], generated_files=result['generated_files'],
        receipt=str(args.receipt), receipt_sha256=hashlib.sha256(encoded).hexdigest(), browser_UI_QA_performed=False)), flush=True)


if __name__ == '__main__':
    main()
