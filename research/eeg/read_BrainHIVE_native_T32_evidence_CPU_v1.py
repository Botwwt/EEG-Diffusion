"""Read a completed, frozen T32 diagnostic; never issue admission or load tensors.

All scientific labels are descriptive directions in a missing-reference-Q DDPM
probe. They are not image benefit, parent-Q validation, or practical significance.
The separate contract test uses synthetic files, never an actual diagnostic.
"""
import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
ARMS = ('raw_OOF_EEG', 'permuted_different_concept_OOF_EEG', 'correct_F',
        'wrong_different_concept_F', 'missing_zero_raw_condition')
METRICS = ('d_positive', 'd_negative', 'teacher_separation', 'normalized_margin',
           'J', 'rank_loss', 'near_zero_teacher_fraction')
MAIN_SUCCESS = 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
DIAG_SUCCESS = 'FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING'
COMPLETION = 'COMPLETE_FROZEN_NATIVE_T32_FUNCTIONAL_DIAGNOSTIC_NO_IMAGE_QUALITY_CLAIM'
Q_REFERENCE = ('Exactly one missing_zero_raw_condition full-UNet pass per query/time; '
               'its detached native Q reused for every response arm. Fixed before data, no response-specific Q.')
MISSING = 'Exactly zeros_like(raw_OOF_EEG) at input to frozen A; not BrainMLP(zero EEG) and not zero IP tokens'
RULES = {
    'aggregation': 'Equal mean of all four registered native times within each query, then equal mean of all32 queries. No selected times/layers/queries.',
    'direction_boundary': 0,
    'support': 'All four EEG paired contrasts and correct-versus-wrong-F full-UNet epsilon contrast strictly positive; no near-zero teacher entries.',
    'not_supported': 'All four EEG paired contrasts strictly negative while correct-versus-wrong-F epsilon contrast strictly positive; no near-zero teacher entries.',
    'otherwise': 'UNKNOWN; includes conflicts, exact ties, non-supportive teacher epsilon comparison, or any near-zero teacher entry.',
    'four_EEG_contrasts': ['EEG_J_minus_permuted_EEG_J', 'EEG_J_minus_missing_J',
                          'permuted_d_positive_minus_EEG_d_positive', 'missing_d_positive_minus_EEG_d_positive'],
    'teacher_contrast': 'wrong_F_epsilon_MSE_minus_correct_F',
    'teacher_epsilon': 1e-6,
    'numerical_consistency_only': {'relative_tolerance': 1e-4, 'absolute_tolerance': 3e-5,
        'source': 'Existing diagnostic numerical consistency checks at lines176-178; never an effect-size threshold.'},
    'no_p_values_or_effect_size_cutoffs': True,
    'teacher_self_J_or_zero_positive_MSE_not_utility_evidence': True,
    'practical_and_statistical_importance': 'UNKNOWN_NOT_PREREGISTERED',
    'automatic_admission': False,
}


class EvidenceRefused(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise EvidenceRefused(message)


def boolean(value, expected, label):
    require(type(value) is bool and value is expected, 'Expected literal boolean: ' + label)


def number(value):
    require(type(value) in (int, float) and math.isfinite(value), 'Nonfinite/non-numeric metric')
    return float(value)


def integer(value, label):
    require(type(value) is int, 'Expected integer: ' + label)
    return value


def same(a, b, label):
    require(math.isclose(number(a), number(b), rel_tol=1e-4, abs_tol=3e-5), 'Numerical inconsistency: ' + label)


def mean(values):
    require(bool(values), 'Empty aggregation')
    return math.fsum(number(x) for x in values) / len(values)


def json_value(payload):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(payload, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(EvidenceRefused('Nonfinite JSON constant')))


class Files:
    """Bounded metadata reads, exposed to synthetic tests for early-gate checks."""
    def __init__(self):
        self.reads = []

    def bytes(self, path, maximum=16 << 20):
        path = Path(path)
        require(path.is_file() and path.stat().st_size <= maximum, 'Missing/oversized metadata: ' + str(path))
        before = path.stat()
        self.reads.append(str(path))
        payload = path.read_bytes()
        after = path.stat()
        require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), 'Metadata changed during read')
        require(len(payload) <= maximum, 'Metadata grew during read')
        return payload

    def json(self, path):
        return json_value(self.bytes(path))

    def bound(self, item):
        payload = self.bytes(item['path'])
        require(hashlib.sha256(payload).hexdigest() == item['sha256'], 'SHA mismatch: ' + item['path'])
        return json_value(payload)


def process_live(pid):
    if pid is None:
        return False
    require(type(pid) is int and pid > 0, 'Invalid actual PID')
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminal_gate(policy, files, live):
    # Do not move registration, completion, summary or raw reads above this gate.
    snapshots = {}
    for key, status in [('main_state', MAIN_SUCCESS), ('diagnostic_queue_state', DIAG_SUCCESS),
                        ('diagnostic_job_state', 'COMPLETED')]:
        payload = files.bytes(policy[key])
        state = json_value(payload)
        require(state.get('status') == status, 'Not fully successful terminal: ' + key)
        require(type(state.get('pid')) is int and state['pid'] > 0, 'Missing/invalid actual PID: ' + key)
        if key == 'diagnostic_job_state':
            require(type(state.get('wrapper_pid')) is int and state['wrapper_pid'] > 0, 'Missing wrapper PID')
        for field in ('pid', 'wrapper_pid'):
            require(not live(state.get(field)), 'Actual PID still live: ' + key + '/' + field)
        if key == 'diagnostic_job_state':
            require(type(state.get('exit_code')) is int and state['exit_code'] == 0, 'Diagnostic wrapper did not exit0')
        snapshots[key] = (hashlib.sha256(payload).hexdigest(), state)
    out = Path(policy['diagnostic_output_directory'])
    for name in ('completion.json', 'provenance.json', 'fixed_queries.json', 'per_query_timestep.jsonl',
                 'per_layer_timestep.json', 'summary.json'):
        require((out / name).is_file(), 'Completed diagnostic lacks artifact: ' + name)
    return snapshots


def validate_policy(policy, synthetic):
    require(policy['schema'] == 'BrainHIVE_native_T32_evidence_policy_v1', 'Wrong policy schema')
    require(policy['rules'] == RULES, 'Frozen interpretation policy changed')
    boolean(policy['automatic_admission'], False, 'policy automatic_admission')
    require(policy['expected'] == dict(queries=32, timesteps=4, rows=128, native_IP_layers=70,
        full_UNet_calls=160, full_UNet_samples=640, conditional_forwards_per_query_time=5), 'Fixed denominator changed')
    expected_mode = 'SYNTHETIC_FIXTURE_ONLY' if synthetic else 'ACTUAL_COMPLETED_DIAGNOSTIC_READER_ONLY'
    require(policy['execution_mode'] == expected_mode, 'Wrong execution mode')
    if not synthetic:
        require(sys.platform.startswith('linux'), 'Actual server PID checks require execution on the server, not local PID inference')
        require(Path(policy['repository_root']).resolve() == ROOT, 'Wrong actual repository')
    payload = Path(__file__).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == policy['reader_source_sha256'], 'Reader source differs from locked policy')


def metadata(policy, files, snapshots):
    reg = files.bound(policy['diagnostic_registration'])
    require(reg['diagnostic'] == 'native_function_T32_v2' and reg['queries'] == 32 and reg['subject'] == 'sub-01', 'Wrong frozen diagnostic')
    require(reg['output_directory'] == policy['diagnostic_output_directory'], 'Wrong output directory')
    for field in ('component_training_enabled', 'KAN_enabled', 'optimizer_created'):
        boolean(reg[field], False, 'diagnostic registration ' + field)
    require(type(reg['generated_images']) is int and reg['generated_images'] == 0, 'Unexpected registered images')
    for name, sha in (reg['source_sha256'] | reg['queue_source_sha256']).items():
        require(Path(name).name == name, 'Unsafe source filename')
        require(hashlib.sha256(files.bytes(Path(policy['repository_root']) / name)).hexdigest() == sha, 'Actual source SHA changed: ' + name)
    require(reg['input_sha256'] == policy['registered_metadata_input_sha256'], 'Registered metadata set changed')
    inputs = {}
    for path, sha in reg['input_sha256'].items():
        payload = files.bytes(path)
        require(hashlib.sha256(payload).hexdigest() == sha, 'Actual registered metadata SHA changed: ' + path)
        inputs[path] = payload
    manifest = files.bound(policy['fixed_queries_metadata'])
    boolean(manifest['execution']['admitted'], False, 'fixed input manifest admitted')
    require(manifest['queries'] == 32 and len(manifest['rows']) == 32, 'Fixed input count')
    protocol = json_value(inputs[policy['split_manifest_path']])
    require(len(protocol['image_rows']['T']) == 13230, 'Wrong full-T partition')
    allowed = protocol['image_rows']['T']
    require(len(set(allowed)) == 13230 and set(allowed).isdisjoint(protocol['image_rows']['D'] + protocol['image_rows']['V']), 'T/D/V row overlap')
    csv_rows = {}
    for row in csv.DictReader(io.StringIO(inputs[policy['image_manifest_path']].decode('utf-8'))):
        if row['dataset_split'] == 'train':
            index = int(row['array_row'])
            require(index not in csv_rows, 'Duplicate image-manifest array row')
            csv_rows[index] = row
    oracle = json_value(inputs[reg['oracle_inference_inputs']])
    require(len(oracle) == 32, 'Oracle identity count')
    expected = []
    for index, (entry, old) in enumerate(zip(manifest['rows'], oracle)):
        require(entry['query_index'] == index and entry['oracle_query_index'] == index, 'Query order drift')
        require(entry['subject'] == 'sub-01' and entry['query_id'] == 'sub-01/' + entry['stimulus_id'] + '/R4', 'Query identity drift')
        require((entry['array_row'], entry['stimulus_id']) == (old['array_row'], old['stimulus_id']), 'Oracle identity mismatch')
        row = csv_rows[entry['array_row']]
        require(row['stimulus_id'] == entry['stimulus_id'] and row['concept_id'] == entry['concept_id'], 'Actual image-manifest identity mismatch')
        require(allowed[entry['T_position']] == entry['array_row'] and entry['concept_id'] in protocol['train_concepts']['T'], 'Fixed identity outside T')
        require(entry['source_role'] == 'oof' + str(protocol['oof_fold_by_concept'][entry['concept_id']]), 'OOF role mismatch')
        expected.append(entry)
    require(len({entry['array_row'] for entry in expected}) == 32, 'Duplicate fixed query')
    contract = json_value(inputs[policy['condition_contract_path']])
    require(contract['status'] == 'ALL10_COMPLETE_REAL_OOF_AND_SEPARATE_D_INPUTS_BOUND_NO_FIT', 'Wrong condition contract')
    subject = next(s for s in contract['subjects'] if s['subject'] == 'sub-01')
    for entry in expected:
        source = next(s for s in subject['sources'] if s['role'] == entry['source_role'])
        for a, b in [('condition_cache_path', 'condition_cache_path'), ('condition_cache_sha256', 'condition_cache_sha256'),
                     ('parent_checkpoint_path', 'checkpoint_path'), ('parent_checkpoint_sha256', 'checkpoint_sha256')]:
            require(entry[a] == source[b], 'Own OOF/cache/affine checkpoint binding changed')
        require(source['actual_native_output_affine_keys'] == ['block.layers.0.norm.weight', 'block.layers.0.norm.bias'], 'Own affine-key contract changed')
        matches = [r for r in source['groups']['component_train'] if r['array_row'] == entry['array_row']]
        require(len(matches) == 1 and matches[0]['cache_row'] == entry['cache_row'], 'Actual contract cache row mismatch')
    return reg, expected, csv_rows, allowed, contract


def fixed_old_negative(allowed, csv_rows):
    """CPU transcription of the SHA-pinned old concept_derangement, not a fit."""
    groups = {}
    for position, row in enumerate(allowed):
        groups.setdefault(str(csv_rows[row]['concept_id']), []).append(position)
    rng = random.Random(1729)
    keys = sorted(groups)
    rng.shuffle(keys)
    ordered = []
    for key in keys:
        group = groups[key][:]
        rng.shuffle(group)
        ordered.extend(group)
    shift = max(map(len, groups.values()))
    require(shift * 2 <= len(allowed), 'Impossible fixed concept negative')
    result = [-1] * len(allowed)
    for rank, position in enumerate(ordered):
        result[position] = ordered[(rank + shift) % len(ordered)]
    require(sorted(result) == list(range(len(allowed))), 'Negative is not a bijection')
    require(all(csv_rows[allowed[i]]['concept_id'] != csv_rows[allowed[j]]['concept_id']
                for i, j in enumerate(result)), 'Same-concept negative')
    return result


def load_artifacts(policy, files, snapshots):
    out = Path(policy['diagnostic_output_directory'])
    completion_bytes = files.bytes(out / 'completion.json')
    completion = json_value(completion_bytes)
    queue = snapshots['diagnostic_queue_state'][1]
    require(queue['completion_path'] == str(out / 'completion.json'), 'Queue completion path mismatch')
    require(queue['completion_sha256'] == hashlib.sha256(completion_bytes).hexdigest(), 'Queue completion SHA mismatch')
    require(completion['status'] == COMPLETION, 'Wrong completion status')
    require(completion['registration_sha256'] == policy['diagnostic_registration']['sha256'], 'Completion registration SHA mismatch')
    require(completion['actual_full_UNet_forwards'] == dict(calls=160, samples=640), 'Actual diagnostic forward count')
    for key, value in [('real_T_queries', 32), ('rows', 128), ('native_IP_layers', 70), ('generated_images', 0), ('fitted_parameters', 0)]:
        require(type(completion[key]) is int and completion[key] == value, 'Completion denominator: ' + key)
    for key, value in [('engineering_checks_passed', True), ('protected_scores_read', False),
                       ('component_effectiveness_established', False), ('native_defect_mechanism_established', False)]:
        boolean(completion[key], value, 'completion ' + key)
    artifacts = {}
    for key, name in [('provenance', 'provenance.json'), ('fixed_queries', 'fixed_queries.json'),
                      ('per_query_timestep', 'per_query_timestep.jsonl'), ('per_layer_timestep', 'per_layer_timestep.json'),
                      ('summary', 'summary.json')]:
        payload = files.bytes(out / name)
        require(hashlib.sha256(payload).hexdigest() == completion[key + '_sha256'], 'Artifact SHA mismatch: ' + name)
        artifacts[key] = [json_value(line) for line in payload.splitlines() if line] if name.endswith('.jsonl') else json_value(payload)
    return completion, artifacts


def validate_provenance(reg, provenance, contract, policy, negative, fixed_queries, allowed):
    require(provenance['registration_sha256'] == policy['diagnostic_registration']['sha256'], 'Provenance registration mismatch')
    require(provenance['source_sha256'] == reg['source_sha256'] and provenance['input_sha256'] == reg['input_sha256'], 'Provenance source/input bindings')
    require(provenance['input_contract_sha256'] == reg['input_sha256'][policy['condition_contract_path']], 'Provenance contract mismatch')
    require(provenance['oracle_metadata_sha256'] == reg['input_sha256'][reg['oracle_inference_inputs']], 'Provenance oracle mismatch')
    require(provenance['training_target_cache_path'] == contract['target_cache_path'] and provenance['training_target_cache_sha256'] == contract['target_cache_sha256'], 'Target-cache binding changed')
    require(provenance['Q_reference'] == Q_REFERENCE and provenance['missing_definition'] == MISSING, 'Actual reference/missing definition changed')
    require(provenance['diagnostic_noise_scheduler'] == 'DDPMScheduler', 'Wrong diagnostic scheduler')
    negative_sha = hashlib.sha256(struct.pack('<' + str(len(negative)) + 'q', *negative)).hexdigest()
    require(provenance['negative_mapping']['negative_position_sha256'] == negative_sha, 'Actual old negative mapping changed')
    required_rows = {q[key] for q in fixed_queries for key in ('array_row', 'wrong_array_row')}
    subject = next(s for s in contract['subjects'] if s['subject'] == 'sub-01')
    coverage = {row: 0 for row in required_rows}
    source_roles = set()
    for source in provenance['actual_sub01_OOF_sources']:
        require(source['role'] in ('oof0', 'oof1') and source['role'] not in source_roles, 'Invalid/repeated actual OOF role')
        source_roles.add(source['role'])
        actual = next(s for s in subject['sources'] if s['role'] == source['role'])
        for field in ('checkpoint_path', 'checkpoint_sha256', 'condition_cache_path', 'condition_cache_sha256'):
            require(source[field] == actual[field], 'Actual diagnostic OOF provenance drift: ' + field)
        selected = [r for r in actual['groups']['component_train'] if r['array_row'] in required_rows]
        require(source['selected_mapping'] == selected, 'Actual diagnostic selected cache mapping drift')
        for row in selected:
            coverage[row['array_row']] += 1
    require(set(coverage.values()) == {1}, 'Actual diagnostic OOF coverage is not exactly one per needed row')
    mapping = provenance['negative_mapping']
    require(mapping['seed'] == 1729, 'Old mapping seed drift')
    boolean(mapping['different_concept_verified'], True, 'negative mapping different-concept')
    boolean(mapping['stimulus_or_array_index_used_as_position'], False, 'negative mapping positions')
    require(mapping['mapping_array_rows_sha256'] == hashlib.sha256(struct.pack('<' + str(len(allowed)) + 'q', *allowed)).hexdigest(), 'Negative array-row order changed')
    require(mapping['image_manifest_sha256'] == reg['input_sha256'][policy['image_manifest_path']] and
            mapping['split_manifest_sha256'] == reg['input_sha256'][policy['split_manifest_path']], 'Negative metadata provenance mismatch')
    for key, expected in [('all_conditions_share_same_real_latent_noise_and_timestep', True), ('optimizer_created', False),
                          ('all_native_weights_frozen', True), ('protected_D_V_or_test_tensors_or_scores_loaded', False)]:
        boolean(provenance[key], expected, 'provenance ' + key)
    require(provenance['generated_images'] == provenance['parameter_updates'] == 0, 'Unexpected fitting/images')


def validate_rows(artifacts, completion, expected, csv_rows, allowed, negative):
    times = completion['timesteps']
    require(len(times) == len(set(times)) == 4 and all(type(t) is int and 0 <= t < 1000 for t in times), 'Wrong native times')
    require(artifacts['provenance']['native_scheduler_timesteps'] == times, 'Native time provenance mismatch')
    fixed = artifacts['fixed_queries']
    require(len(fixed) == 32, 'Saved fixed query count')
    for a, b in zip(fixed, expected):
        for key in ('array_row', 'stimulus_id', 'oracle_query_index', 'T_position', 'concept_id'):
            require(a[key] == b[key], 'Actual fixed identity/order mismatch: ' + key)
        wrong = csv_rows[a['wrong_array_row']]
        require(a['wrong_T_position'] == negative[a['T_position']], 'Frozen old negative assignment mismatch')
        require(allowed[a['wrong_T_position']] == a['wrong_array_row'] and wrong['concept_id'] != a['concept_id'], 'Wrong negative identity or same-concept negative')
        require((wrong['concept_id'], wrong['stimulus_id']) == (a['wrong_concept_id'], a['wrong_stimulus_id']), 'Actual negative identity mismatch')
    rows = artifacts['per_query_timestep']
    require(len(rows) == 128, 'Missing/extra query-time records')
    keyed = {}
    for row in rows:
        index = integer(row['oracle_query_index'], 'oracle_query_index')
        require(0 <= index < 32, 'Out-of-range query index')
        query = fixed[index]
        for key in query:
            require(row[key] == query[key], 'Raw-row fixed identity mismatch: ' + key)
        require(row['subject'] == 'sub-01' and row['repeat_protocol'] == 'R4', 'Wrong subject/repeats')
        time_index = integer(row['native_time_index'], 'native_time_index')
        require(0 <= time_index < 4 and row['timestep'] == times[time_index], 'Raw native time/index mismatch')
        key = (index, time_index)
        require(key not in keyed, 'Duplicate query-time record')
        keyed[key] = row
        seed_key = '1729|native_function_T32_v1|sub-01|' + row['stimulus_id'] + '|R4|DDPM|' + str(row['timestep'])
        seed = int(hashlib.sha256(seed_key.encode()).hexdigest()[:16], 16) % (2**63 - 1)
        require(type(row['noise_seed']) is int and row['noise_seed'] == seed, 'Actual diagnostic noise seed mismatch')
        for field in ('noise_sha256', 'noised_real_T_latent_sha256'):
            require(isinstance(row[field], str) and len(row[field]) == 64 and all(c in '0123456789abcdef' for c in row[field]), 'Missing noise/latent hash')
        require(row['conditional_full_UNet_forwards'] == 5 and row['captured_IP_layers'] == 70 and row['Q_reference_arm'] == ARMS[4], 'Raw native call/reference coverage')
        boolean(row['no_generated_image'], True, 'raw no_generated_image')
        require(set(row['epsilon_MSE']) == set(row['native_IP_response_by_arm']) == set(ARMS), 'Wrong condition arms')
        for arm in ARMS:
            require(number(row['epsilon_MSE'][arm]) >= 0, 'Negative epsilon MSE')
            metrics = row['native_IP_response_by_arm'][arm]
            require(set(metrics) == set(METRICS), 'Wrong response metrics')
            for metric in METRICS:
                number(metrics[metric])
            require(all(metrics[x] >= 0 for x in ('d_positive', 'd_negative', 'teacher_separation', 'rank_loss')), 'Negative response distance/loss')
            require(0 <= metrics['near_zero_teacher_fraction'] <= 1, 'Invalid near-zero fraction')
            same(metrics['J'], -metrics['normalized_margin'], 'J sign')
        e = row['epsilon_MSE']; f = row['native_IP_response_by_arm']
        contrasts = dict(EEG_epsilon_MSE_minus_correct_F=e[ARMS[0]] - e[ARMS[2]],
            permuted_EEG_epsilon_MSE_minus_EEG=e[ARMS[1]] - e[ARMS[0]],
            wrong_F_epsilon_MSE_minus_correct_F=e[ARMS[3]] - e[ARMS[2]],
            missing_epsilon_MSE_minus_EEG=e[ARMS[4]] - e[ARMS[0]],
            EEG_J_minus_permuted_EEG_J=f[ARMS[0]]['J'] - f[ARMS[1]]['J'],
            EEG_J_minus_missing_J=f[ARMS[0]]['J'] - f[ARMS[4]]['J'])
        for field, value in contrasts.items():
            same(row[field], value, field)
        for arm in ARMS:
            for metric in ('teacher_separation', 'near_zero_teacher_fraction'):
                same(f[arm][metric], f[ARMS[2]][metric], 'Shared teachers: ' + metric)
        # These are algebra/implementation checks, never useful-teacher evidence.
        same(f[ARMS[2]]['d_positive'], 0, 'Teacher self-positive MSE identity')
        same(f[ARMS[3]]['d_negative'], 0, 'Wrong teacher self-negative MSE identity')
        same(f[ARMS[2]]['d_negative'], f[ARMS[2]]['teacher_separation'], 'Teacher distance identity')
        same(f[ARMS[3]]['d_positive'], f[ARMS[2]]['teacher_separation'], 'Wrong teacher distance identity')
        same(f[ARMS[2]]['J'], -f[ARMS[3]]['J'], 'Teacher self-J antisymmetry')
        require(-3e-5 <= f[ARMS[2]]['J'] <= 1 + 3e-5, 'Teacher self-J range')
    require(set(keyed) == {(q, t) for q in range(32) for t in range(4)}, 'Incomplete cross product')
    return times, keyed


def validate_aggregates(artifacts, times, keyed):
    summary = artifacts['summary']
    require(summary['status'] == 'FROZEN_T_ONLY_FUNCTIONAL_MEASUREMENTS_SAVED', 'Wrong summary status')
    boolean(summary['teacher_advantage_required_for_engineering_success'], False, 'teacher advantage engineering flag')
    boolean(summary['final_test_inference_performed'], False, 'final test inference flag')
    by_time = summary['by_timestep']
    require(len(by_time) == 4 and [s['timestep'] for s in by_time] == times, 'Summary time coverage/order')
    for ti, item in enumerate(by_time):
        require(type(item['queries']) is int and item['queries'] == 32, 'Summary count')
        rows = [keyed[q, ti] for q in range(32)]
        for arm in ARMS:
            same(item['mean_epsilon_MSE'][arm], mean([r['epsilon_MSE'][arm] for r in rows]), 'Summary epsilon')
            for metric in METRICS:
                same(item['mean_native_IP_response_by_arm'][arm][metric],
                     mean([r['native_IP_response_by_arm'][arm][metric] for r in rows]), 'Summary response')
        for field in ('EEG_epsilon_MSE_minus_correct_F', 'permuted_EEG_epsilon_MSE_minus_EEG',
                      'wrong_F_epsilon_MSE_minus_correct_F', 'missing_epsilon_MSE_minus_EEG',
                      'EEG_J_minus_permuted_EEG_J', 'EEG_J_minus_missing_J'):
            same(item['mean_' + field], mean([r[field] for r in rows]), 'Summary paired metric')
        same(item['fraction_correct_F_lower_epsilon_MSE_than_EEG'], mean([float(r['EEG_epsilon_MSE_minus_correct_F'] > 0) for r in rows]), 'Summary fraction')
        same(item['fraction_EEG_IP_response_closer_to_correct_than_wrong'], mean([float(r['native_IP_response_by_arm'][ARMS[0]]['J'] > 0) for r in rows]), 'Summary J fraction')
    layer_rows = artifacts['per_layer_timestep']
    require(len(layer_rows) == 70 * 4 * 5, 'Missing/extra layer-time-arm rows')
    groups = {}
    seen = set()
    for layer in layer_rows:
        key = (layer['layer'], layer['timestep'], layer['arm'])
        require(key not in seen and layer['timestep'] in times and layer['arm'] in ARMS, 'Duplicate/invalid layer-time-arm')
        seen.add(key)
        require(layer['queries'] == 32 and type(layer['query_positions']) is int and 1 <= layer['query_positions'] <= 16, 'Layer sampling/count')
        for metric in METRICS:
            number(layer[metric])
        groups.setdefault((layer['timestep'], layer['arm']), []).append(layer)
    reference = None
    for ti, timestep in enumerate(times):
        for arm in ARMS:
            group = groups[timestep, arm]
            names = {v['layer'] for v in group}
            require(len(names) == 70 and (reference is None or names == reference), 'Actual native-layer coverage changed')
            reference = names
            for metric in METRICS:
                same(mean([v[metric] for v in group]), mean([keyed[q, ti]['native_IP_response_by_arm'][arm][metric] for q in range(32)]), 'Layer/query marginal consistency')


def interpret(times, keyed):
    details = []
    for q in range(32):
        for ti in range(4):
            r = keyed[q, ti]; f = r['native_IP_response_by_arm']
            details.append(dict(query_index=q, stimulus_id=r['stimulus_id'], array_row=r['array_row'],
                timestep=times[ti], native_time_index=ti,
                EEG_J_minus_permuted_EEG_J=r['EEG_J_minus_permuted_EEG_J'],
                EEG_J_minus_missing_J=r['EEG_J_minus_missing_J'],
                permuted_d_positive_minus_EEG_d_positive=f[ARMS[1]]['d_positive'] - f[ARMS[0]]['d_positive'],
                missing_d_positive_minus_EEG_d_positive=f[ARMS[4]]['d_positive'] - f[ARMS[0]]['d_positive'],
                wrong_F_epsilon_MSE_minus_correct_F=r['wrong_F_epsilon_MSE_minus_correct_F'],
                teacher_separation=f[ARMS[2]]['teacher_separation'],
                near_zero_teacher_fraction=f[ARMS[2]]['near_zero_teacher_fraction'],
                epsilon_MSE=r['epsilon_MSE'], native_IP_response_by_arm=f))
    fields = RULES['four_EEG_contrasts'] + [RULES['teacher_contrast']]
    per_query = [dict(query_index=q, **{k: mean([d[k] for d in details if d['query_index'] == q]) for k in fields}) for q in range(32)]
    aggregate = {k: mean([r[k] for r in per_query]) for k in fields}
    by_time = [dict(timestep=t, **{k: mean([d[k] for d in details if d['timestep'] == t]) for k in fields}) for t in times]
    near_zero = any(d['near_zero_teacher_fraction'] > 0 for d in details)
    eeg = [aggregate[k] for k in RULES['four_EEG_contrasts']]
    teacher = aggregate[RULES['teacher_contrast']]
    if near_zero:
        status, reason = 'UNKNOWN', 'Near-zero teacher entries; exact v2 masked joint metrics cannot be recovered from the saved marginals.'
    elif all(v > 0 for v in eeg) and teacher > 0:
        status, reason = 'DIRECTIONAL_SUPPORT_THIS_PROBE_ONLY', 'All frozen full-denominator directional contrasts are positive; no importance, efficacy or admission claim.'
    elif all(v < 0 for v in eeg) and teacher > 0:
        status, reason = 'NOT_SUPPORTED_THIS_PROBE_ONLY', 'EEG contrasts consistently reverse the predicted direction in this probe; no global model refutation.'
    else:
        status, reason = 'UNKNOWN', 'Conflicting directions, exact ties, or no positive full-UNet correct-teacher comparison.'
    return dict(status=status, reason=reason, aggregation=RULES['aggregation'],
        aggregate=aggregate, per_query_four_time_means=per_query, all_four_timestep_details=by_time,
        all128_query_timestep_details=details,
        teacher_valid_fraction_across_all_query_time_layer_entries=1 - mean([d['near_zero_teacher_fraction'] for d in details]),
        functional_v2_masked_correspondence=('UNKNOWN_NEAR_ZERO_WITHOUT_QUERY_BY_LAYER_DATA' if near_zero else
            'ALL_TEACHER_ENTRIES_VALID_IN_THIS_MISSING_Q_PROBE_ONLY_NOT_PARENT_Q_VALIDATION'),
        teacher_self_J_and_zero_positive_MSE='ALGEBRAIC_IDENTITIES_NOT_UTILITY_EVIDENCE',
        statistical_or_practical_importance='UNKNOWN_NOT_PREREGISTERED',
        images_benefit_established=False, parent_Q_v2_training_validated=False, automatic_admission=False)


def read_evidence(policy, *, files=None, live=process_live, synthetic=False):
    files = files or Files()
    validate_policy(policy, synthetic)
    snapshots = terminal_gate(policy, files, live)
    reg, expected, csv_rows, allowed, contract = metadata(policy, files, snapshots)
    completion, artifacts = load_artifacts(policy, files, snapshots)
    negative = fixed_old_negative(allowed, csv_rows)
    validate_provenance(reg, artifacts['provenance'], contract, policy, negative, artifacts['fixed_queries'], allowed)
    times, keyed = validate_rows(artifacts, completion, expected, csv_rows, allowed, negative)
    validate_aggregates(artifacts, times, keyed)
    evidence = interpret(times, keyed)
    for key, (digest, _) in snapshots.items():
        payload = files.bytes(policy[key])
        require(hashlib.sha256(payload).hexdigest() == digest, 'Terminal state changed during read')
        state = json_value(payload)
        for field in ('pid', 'wrapper_pid'):
            require(not live(state.get(field)), 'Actual PID revived during read')
    return dict(schema='BrainHIVE_native_T32_descriptive_evidence_CPU_v1',
        scope='SYNTHETIC_CONTRACT_FIXTURE_ONLY' if synthetic else 'ACTUAL_COMPLETED_FIXED_T32_PROBE_ONLY',
        numerical_integrity='PASS_COMPLETE32_BY4_NATIVE70_SHA_AND_IDENTITY', evidence=evidence,
        automatic_admission=False, contract_consumable_review=False, fitting_authorized=False,
        GPU_used=False, torch_imported=False, generated_images=0,
        reader_source_sha256=policy['reader_source_sha256'],
        diagnostic_registration_sha256=policy['diagnostic_registration']['sha256'],
        actual_completion_sha256=snapshots['diagnostic_queue_state'][1]['completion_sha256'],
        limits=['Previously exposed sub-01 T32; not independent performance confirmation.',
                'True image latent plus independently DDPM-noised times; not a four-step Euler trajectory or image quality.',
                'Fixed missing-raw-reference Q differs from functional-v2 parent Q; negative construction also differs.',
                'No query-by-layer joint output; no exact masked-v2 response reconstruction when any teacher entry is near zero.',
                'No EEG/noise/latent/weight tensors are reloaded; fresh metadata and SHA-bound diagnostic artifacts are checked, not tensor re-execution.',
                'Natural-zero directions are descriptive, with no registered practical effect threshold or significance test.',
                'No review/admission schema or EEG_pairing_signal_present/teacher_response_nondegenerate boolean is issued.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        require(not args.output.exists(), 'Preserve earlier evidence: output already exists')
        policy_bytes = args.policy.read_bytes()
        policy = json_value(policy_bytes)
        result = read_evidence(policy)
        require(args.policy.read_bytes() == policy_bytes, 'Policy changed during evidence read')
        result['policy_sha256'] = hashlib.sha256(policy_bytes).hexdigest()
        result['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
        require(len(payload.encode()) <= 2 << 20, 'Evidence output exceeds fixed2MiB metadata cap')
        require(args.output.parent.is_dir(), 'Output parent must already exist')
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(payload)
        print(json.dumps(dict(status='DESCRIPTIVE_EVIDENCE_SAVED_NOT_ADMISSION',
            interpretation=result['evidence']['status'], automatic_admission=False, output=str(args.output))))
    except (EvidenceRefused, KeyError, ValueError, TypeError, OSError, StopIteration) as error:
        print(json.dumps(dict(status='REFUSED_NO_DIAGNOSTIC_REVIEW_OR_ADMISSION',
            error_type=type(error).__name__, error=str(error)[:1000], automatic_admission=False)))
        return 2
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
