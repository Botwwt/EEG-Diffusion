"""Importable CPU identity/join contracts for FUTURE official R80 producers.

No fit, admission, method lock, model import, image decode or metric calculation.
Metadata consistency is not tensor/file authenticity. Future producers must
audit actual cache/model/image payloads and invoke observation helpers on the
actual tensor bytes at the documented hooks. Existing native rows lack the
first-UNet observation and cannot be upgraded by hashing their seed.
"""
import collections
import copy
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import random
import re
import struct

SUBJECTS = tuple('sub-' + str(i).zfill(2) for i in range(1, 11))
INTERVENTIONS = ('correct', 'permuted', 'missing_raw_condition')
PROTOCOL_SCHEMA = 'BrainHIVE_official_R80_identity_protocol_v1'
ARM_SCHEMA = 'BrainHIVE_official_R80_identity_arm_v1'
NORMALIZATION = 'raw LayerNorm output'
PERMUTATION_POLICY = 'native_Sattolo_Random1729_on_registered_array_order'
ZERO_RAW_SHA = hashlib.sha256(bytes(1024 * 4)).hexdigest()
ZERO_A_FP16_SHA = hashlib.sha256(bytes(1024 * 2)).hexdigest()
PAPER7 = {'eval_pixcorr': 'higher', 'eval_ssim': 'higher', 'eval_alex2': 'higher',
    'eval_alex5': 'higher', 'eval_inception': 'higher', 'eval_clip': 'higher', 'eval_swav': 'lower'}
PAPER_EXTRA = {'eval_effnet': 'lower'}
COMMON = dict.fromkeys(('DINO2AFC', 'DINO_cosine', 'DINO_Top1', 'DINO_Top5',
    'EffNet', 'CapSim', 'ObjectF1', 'SEED'), 'higher')
TARGET_KEYS = {'concept_id', 'target_image_sha256', 'score_target_path',
    'target_image_zip', 'target_image_member', 'target_mapping_attached_after_inference'}
INFERENCE_KEYS = set(('query_id global_query_index subject stimulus_id array_row repeat_protocol intervention method run_name '
    'image_seed EEG_source_stimulus_id EEG_source_array_row EEG_R80_source_sha256 condition_cache_row condition_cache_sha256 '
    'parent_checkpoint_sha256 parent_registration_sha256 parent_affine_sha256 condition_export_sha256 '
    'component_checkpoint_sha256 calibration_sha256 component_applied component_bypass_reason condition_sha256 '
    'condition_dtype condition_shape condition_normalization actual_A_input_fp16_sha256 actual_A_input_capture '
    'initial_UNet_input_sha256 initial_UNet_input_shape initial_UNet_input_dtype initial_UNet_input_capture '
    'generator_receipt_sha256 producer_source_sha256 protocol_sha256 arm_core_sha256 '
    'generated_path generated_sha256 UNet_NFE IP_conditioned_attention_calls VAE_decoder_calls '
    'lossless_pixel_roundtrip target_or_visual_features_read_for_condition oracle_diagnostic_only').split())
MAX_METADATA_BYTES = 32 << 20


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def core_sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def path(value):
    require(isinstance(value, str), 'Canonical absolute private path required')
    p = PurePosixPath(value)
    require(p.is_absolute() and '..' not in p.parts and str(p) == value, 'Canonical absolute private path required')
    return p


def reference(value):
    require(isinstance(value, dict) and set(value) == {'path', 'sha256'} and digest(value['sha256']), 'Explicit path/SHA reference required')
    path(value['path'])
    return value


def verdict(**fields):
    return dict(status='OFFICIAL_R80_METADATA_IDENTITY_CONTRACT_VALID', metadata_only=True,
        payloads_verified=False, actual_hook_execution_certified=False, execution_admitted=False,
        method_lock_issued=False, images_generated_here=0, metrics_calculated_here=0, **fields)


def observe_tensor_bytes(raw, dtype, shape, capture):
    """Future hooks pass ACTUAL contiguous little-endian tensor bytes.

    This helper hashes the supplied bytes, not a seed or a tensor description;
    it cannot independently establish that its caller really ran the hook.
    """
    require(dtype in ('float16', 'float32') and isinstance(raw, (bytes, bytearray, memoryview)), 'Actual float tensor bytes required')
    require(shape in ([1, 1024], [1, 4, 64, 64]) and all(type(n) is int for n in shape), 'Unexpected native observation shape')
    require(capture in ('native_A_forward_pre_hook', 'first_native_UNet_forward_pre_hook'), 'Observation must name the actual hook')
    require((capture == 'native_A_forward_pre_hook') == (shape == [1, 1024]), 'Hook/shape mismatch')
    require(capture != 'native_A_forward_pre_hook' or dtype == 'float16', 'Actual A input is native float16')
    raw = bytes(raw); width = 2 if dtype == 'float16' else 4
    require(len(raw) == math.prod(shape) * width, 'Observed tensor byte count differs')
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<e' if width == 2 else '<f', raw)), 'Nonfinite observed tensor')
    return dict(sha256=hashlib.sha256(raw).hexdigest(), dtype=dtype, shape=list(shape), capture=capture)


def observe_condition_bytes(raw):
    require(isinstance(raw, (bytes, bytearray, memoryview)) and len(raw) == 4096, 'Actual float32 1024-condition bytes required')
    raw = bytes(raw)
    require(all(math.isfinite(v[0]) for v in struct.iter_unpack('<f', raw)), 'Nonfinite raw condition')
    return hashlib.sha256(raw).hexdigest()


def validate_protocol(protocol):
    require(set(protocol) == {'schema', 'training_seed', 'subjects', 'repeat_protocol', 'physical_repeat_indices',
        'permutation_policy', 'images_per_query', 'best_of_N', 'official200_previously_exposed', 'independent_H',
        'split_version', 'inputs', 'generator', 'generator_receipt', 'identity_contract_sha256'}, 'Blind protocol contains unregistered/target fields')
    require(protocol['schema'] == PROTOCOL_SCHEMA and protocol['training_seed'] == 1729, 'Wrong official R80 protocol/seed')
    require(protocol['subjects'] == list(SUBJECTS) and protocol['repeat_protocol'] == 'R80'
        and protocol['physical_repeat_indices'] == list(range(80)), 'Complete real R80 ten-subject protocol required')
    require(protocol['permutation_policy'] == PERMUTATION_POLICY, 'Unregistered EEG permutation policy')
    require(protocol['images_per_query'] == 1 and protocol['best_of_N'] is False, 'Exactly one predetermined image/query required')
    require(protocol['official200_previously_exposed'] is True and protocol['independent_H'] is False, 'Exposed benchmark must not become independent H')
    require(protocol['split_version'] == 'eeg_full10_concept_protocol_v1', 'Wrong audited source split version')
    queries = protocol['inputs']
    require(len(queries) == 200 and all(set(q) == {'stimulus_id', 'array_row'} for q in queries), 'Blind inference input must contain only 200 stimulus/array identities')
    require([q['array_row'] for q in queries] == list(range(200)) and all(type(q['array_row']) is int for q in queries), 'Complete official array order required')
    require(len({q['stimulus_id'] for q in queries}) == 200 and len({PurePosixPath(q['stimulus_id']).stem for q in queries}) == 200
        and all(q['stimulus_id'] == PurePosixPath(q['stimulus_id']).name and q['stimulus_id'] for q in queries), 'Unique canonical stimulus basenames required')
    require(protocol['generator'] == dict(model='SDXL-Turbo', resolution=512, steps=4, CFG=0, IP_scale=1,
        scheduler='EulerAncestralDiscreteScheduler', timestep_spacing='trailing'), 'Native matched generator settings required')
    reference(protocol['generator_receipt'])
    require(protocol['identity_contract_sha256'] == hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'Reviewed identity contract source changed')
    return verdict(queries=2000, query_inputs=200, protocol_sha256=core_sha(protocol))


def permutation(protocol):
    """The existing native Sattolo algorithm; no labels/targets are accessed."""
    order = list(range(200)); rng = random.Random(1729)
    for i in range(199, 0, -1):
        j = rng.randrange(i); order[i], order[j] = order[j], order[i]
    require(sorted(order) == list(range(200)) and all(i != j for i, j in enumerate(order)), 'EEG permutation is not a derangement')
    return order


def validate_arm(protocol, arm):
    validate_protocol(protocol)
    require(set(arm) == {'schema', 'protocol_sha256', 'intervention', 'run_name', 'candidate', 'image_root',
        'producer_source_sha256', 'component', 'subject_bindings'}, 'Arm contains unregistered inference fields')
    require(arm['schema'] == ARM_SCHEMA and arm['protocol_sha256'] == core_sha(protocol), 'Arm/protocol binding differs')
    require(arm['intervention'] in INTERVENTIONS, 'Missing raw c is distinct from historical missing zero-EEG')
    require(arm['run_name'] == PurePosixPath(arm['run_name']).name and arm['run_name']
        and isinstance(arm['candidate'], str) and arm['candidate'], 'Explicit candidate/run name required')
    path(arm['image_root'])
    require(digest(arm['producer_source_sha256']), 'Future producer actual source SHA must be bound')
    component = arm['component']
    require(set(component) == {'kind', 'checkpoint', 'calibration'}, 'Explicit component/checkpoint/calibration binding required')
    require(component['kind'] in ('native_parent', 'bounded_condition_adapter'), 'Explicit native/learned component kind required')
    if component['kind'] == 'native_parent':
        require(component['checkpoint'] is None and component['calibration'] is None, 'Native parent must not fabricate component/calibration weights')
    else:
        reference(component['checkpoint']); reference(component['calibration'])
    require(set(arm['subject_bindings']) == set(SUBJECTS), 'All ten real subject bindings required')
    for subject, binding in arm['subject_bindings'].items():
        require(set(binding) == {'EEG_R80_source', 'parent_checkpoint', 'parent_registration', 'parent_affine', 'condition_cache', 'condition_export',
            'condition_cache_model_sha256',
            'condition_shape', 'condition_dtype', 'normalization', 'additional_L2_normalization', 'rows'}, 'Subject binding contains unregistered fields')
        for key in ('EEG_R80_source', 'parent_checkpoint', 'parent_registration', 'parent_affine', 'condition_cache', 'condition_export'):
            reference(binding[key])
        require(binding['condition_cache_model_sha256'] == binding['parent_checkpoint']['sha256'], 'Condition export names a different parent model')
        require(binding['condition_shape'] == [200, 1024] and binding['condition_dtype'] == 'float32'
            and binding['normalization'] == NORMALIZATION and binding['additional_L2_normalization'] is False,
            'Native complete raw R80 condition cache required')
        rows = binding['rows']
        require(len(rows) == 200 and len({r['cache_row'] for r in rows}) == 200, 'Complete unique condition-cache rows required')
        for query, row in zip(protocol['inputs'], rows):
            require(set(row) == {'stimulus_id', 'array_row', 'cache_row', 'condition_sha256'}
                and type(row['array_row']) is int
                and (row['stimulus_id'], row['array_row']) == (query['stimulus_id'], query['array_row']), 'Cache row/stimulus/array identity differs')
            require(type(row['cache_row']) is int and 0 <= row['cache_row'] < 200 and digest(row['condition_sha256']), 'Invalid actual cache-row hash/index')
    return verdict(queries=2000, arm_core_sha256=core_sha(arm))


def expected_identity(protocol, arm, subject, query_index, *, _cached=None):
    """The caller validates the complete arm once before its producer loop."""
    require(subject in SUBJECTS and type(query_index) is int and 0 <= query_index < 200, 'Unknown subject/query')
    query = protocol['inputs'][query_index]; binding = arm['subject_bindings'][subject]
    missing = arm['intervention'] == 'missing_raw_condition'
    psha, asha, perm = _cached or (core_sha(protocol), core_sha(arm), permutation(protocol))
    source_index = perm[query_index] if arm['intervention'] == 'permuted' else query_index
    source = protocol['inputs'][source_index]; cache = binding['rows'][source_index]
    component = arm['component']; applied = component['kind'] != 'native_parent' and not missing
    image_seed = int(hashlib.sha256(('1729|' + subject + '|' + query['stimulus_id'] + '|R80|' + protocol['split_version']).encode()).hexdigest()[:16], 16) % (2**63 - 1)
    return dict(query_id=subject + '/' + query['stimulus_id'] + '/R80', global_query_index=SUBJECTS.index(subject) * 200 + query_index,
        subject=subject, stimulus_id=query['stimulus_id'], array_row=query['array_row'], repeat_protocol='R80',
        intervention=arm['intervention'], method='BrainHIVE_' + arm['candidate'], run_name=arm['run_name'], image_seed=image_seed,
        EEG_source_stimulus_id=None if missing else source['stimulus_id'], EEG_source_array_row=None if missing else source['array_row'],
        EEG_R80_source_sha256=binding['EEG_R80_source']['sha256'],
        condition_cache_row=None if missing else cache['cache_row'], condition_cache_sha256=binding['condition_cache']['sha256'],
        parent_checkpoint_sha256=binding['parent_checkpoint']['sha256'], parent_registration_sha256=binding['parent_registration']['sha256'],
        parent_affine_sha256=binding['parent_affine']['sha256'], condition_export_sha256=binding['condition_export']['sha256'],
        component_checkpoint_sha256=None if component['checkpoint'] is None else component['checkpoint']['sha256'],
        calibration_sha256=None if component['calibration'] is None else component['calibration']['sha256'], component_applied=applied,
        component_bypass_reason='missing_raw_condition_zero_at_A' if missing else ('no_component_native_parent' if not applied else None),
        condition_sha256=ZERO_RAW_SHA if missing else cache['condition_sha256'], condition_dtype='float32', condition_shape=[1, 1024],
        condition_normalization=NORMALIZATION, actual_A_input_capture='native_A_forward_pre_hook',
        generator_receipt_sha256=protocol['generator_receipt']['sha256'], producer_source_sha256=arm['producer_source_sha256'],
        protocol_sha256=psha, arm_core_sha256=asha,
        generated_path=str(path(arm['image_root']) / arm['run_name'] / subject / (PurePosixPath(query['stimulus_id']).stem + '.webp')),
        UNet_NFE=4, IP_conditioned_attention_calls=280, VAE_decoder_calls=1, lossless_pixel_roundtrip=True,
        target_or_visual_features_read_for_condition=False, oracle_diagnostic_only=False)


def validate_inference(protocol, arm, rows):
    validate_arm(protocol, arm)
    require(len(rows) == 2000, 'Complete 10x200 inference denominator required before target access')
    generated = set()
    cached = core_sha(protocol), core_sha(arm), permutation(protocol)
    for index, row in enumerate(rows):
        require(set(row) == INFERENCE_KEYS, 'Blind inference has missing evidence or forbidden target/score fields')
        expected = expected_identity(protocol, arm, SUBJECTS[index // 200], index % 200, _cached=cached)
        require(all(type(row[k]) is type(v) and row[k] == v for k, v in expected.items()), 'Inference subject/query/cache/parent/component/calibration/protocol binding differs')
        require(digest(row['generated_sha256']) and digest(row['actual_A_input_fp16_sha256']), 'Actual saved image/A input hashes required')
        require(digest(row['initial_UNet_input_sha256']) and row['initial_UNet_input_shape'] == [1, 4, 64, 64]
            and row['initial_UNet_input_dtype'] in ('float16', 'float32')
            and row['initial_UNet_input_capture'] == 'first_native_UNet_forward_pre_hook', 'Actual first-UNet observation metadata required; seed hashes are not a substitute')
        if arm['intervention'] == 'missing_raw_condition':
            require(row['actual_A_input_fp16_sha256'] == ZERO_A_FP16_SHA and row['component_applied'] is False,
                'Missing raw condition must BYPASS component and be exactly zero at native A, not encoder(zero EEG)')
        require(row['generated_path'] not in generated, 'Generated file path reused within arm')
        generated.add(row['generated_path'])
    return verdict(queries=2000, inference_rows_canonical_sha256=core_sha(rows),
        recorded_UNet_NFE=8000, recorded_VAE_decoder_calls=2000, per_condition_costs_are_separate=True,
        actual_noise_claim='Recorded first UNet input only; not all ancestral random draws; hook execution requires pinned producer audit')


def read_inference(protocol, arm, manifest_reference):
    """Only completed, saved blind JSONL is accepted by the target join."""
    # Real host-native absolute path here also permits honest Windows CPU
    # fixtures; model/cache/output bindings above remain canonical POSIX paths.
    require(set(manifest_reference) == {'path', 'sha256'} and digest(manifest_reference['sha256']), 'Saved inference path/SHA required')
    filename = Path(manifest_reference['path'])
    require(filename.is_absolute(), 'Saved inference must have an absolute path')
    require(filename.suffix == '.jsonl' and filename.is_file() and 0 < filename.stat().st_size <= MAX_METADATA_BYTES, 'Finite saved blind inference JSONL required')
    raw = filename.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == manifest_reference['sha256'], 'Saved blind inference manifest SHA differs')
    rows = [json.loads(line) for line in raw.splitlines()]
    validate_inference(protocol, arm, rows)
    return rows


def _target_map(protocol, targets, gallery):
    require(len(targets) == len(gallery) == 200, 'Complete official 200-target denominator required')
    origins = {r['stimulus_id']: r for r in targets}; real = {r['stimulus_id']: r for r in gallery}
    require(len(origins) == len(real) == 200 and set(origins) == set(real) == {q['stimulus_id'] for q in protocol['inputs']}, 'Target/gallery stimulus identities differ')
    require(len({r['concept_id'] for r in targets}) == 200, 'Official benchmark must have 200 distinct concepts')
    for query in protocol['inputs']:
        origin, image = origins[query['stimulus_id']], real[query['stimulus_id']]
        require(origin['dataset_split'] == 'test' and str(origin['array_row']) == str(query['array_row']), 'Target array/stimulus official split differs')
        require(origin['concept_id'] == image['concept_id'] and origin['image_sha256'] == image['image_sha256']
            and digest(image['image_sha256']), 'Gallery target concept/SHA differs')
        path(image['image_path']); path(origin['image_zip'])
        member = PurePosixPath(origin['image_member'])
        require(not member.is_absolute() and '..' not in member.parts and member.name == query['stimulus_id'], 'Target archive member/stimulus differs')
    return origins, real


def _joined_rows(protocol, arm, rows, targets, gallery):
    origins, real = _target_map(protocol, targets, gallery)
    result = []
    for row in rows:
        target, image = origins[row['stimulus_id']], real[row['stimulus_id']]
        if arm['intervention'] == 'permuted':
            require(origins[row['EEG_source_stimulus_id']]['concept_id'] != target['concept_id'], 'Permuted EEG must be from a different concept of the SAME subject')
        result.append(dict(row, concept_id=target['concept_id'], target_image_sha256=target['image_sha256'],
            score_target_path=image['image_path'], target_image_zip=target['image_zip'], target_image_member=target['image_member'],
            target_mapping_attached_after_inference=True))
    return result


def join_targets(protocol, arm, inference_manifest_reference, targets, gallery):
    """Call only AFTER closing all2000 blind inference rows; never in sampling.

    No target iterator is examined until the saved inference file is complete,
    SHA-bound and validated. Gallery schema deliberately has no array_row.
    """
    rows = read_inference(protocol, arm, inference_manifest_reference)
    joined = _joined_rows(protocol, arm, rows, targets, gallery)
    return joined, verdict(queries=2000, inference_manifest=copy.deepcopy(inference_manifest_reference),
        joined_rows_canonical_sha256=core_sha(joined), target_join_after_full_inference=True)


def validate_joined(protocol, arm, rows, targets, gallery):
    require(len(rows) == 2000 and all(set(r) == INFERENCE_KEYS | TARGET_KEYS for r in rows), 'Complete joined identity schema required')
    blind = [{k: r[k] for k in INFERENCE_KEYS} for r in rows]
    validate_inference(protocol, arm, blind)
    require(rows == _joined_rows(protocol, arm, blind, targets, gallery), 'Post-inference target join identity differs')
    return verdict(queries=2000, joined_rows_canonical_sha256=core_sha(rows))


def validate_pair(protocol, reference_arm, reference_rows, candidate_arm, candidate_rows, targets, gallery, kind):
    """Return identities only. No metric computation, selection or thresholds."""
    require(kind in ('matched_methods', 'EEG_intervention'), 'Explicit comparison type required')
    validate_joined(protocol, reference_arm, reference_rows, targets, gallery)
    validate_joined(protocol, candidate_arm, candidate_rows, targets, gallery)
    require(reference_arm['subject_bindings'] == candidate_arm['subject_bindings'], 'Paired methods must share exact parent/cache/affine lineage')
    if kind == 'matched_methods':
        require(reference_arm['intervention'] == candidate_arm['intervention'], 'Main methods cannot be compared across EEG interventions')
    else:
        require(reference_arm['candidate'] == candidate_arm['candidate'] and reference_arm['component'] == candidate_arm['component']
            and reference_arm['intervention'] != candidate_arm['intervention'], 'EEG intervention must keep the same learned method/calibration')
    same_fields = ('query_id', 'global_query_index', 'subject', 'stimulus_id', 'array_row', 'repeat_protocol', 'image_seed',
        'concept_id', 'target_image_sha256', 'score_target_path', 'target_image_zip', 'target_image_member',
        'initial_UNet_input_sha256', 'initial_UNet_input_dtype', 'initial_UNet_input_shape', 'initial_UNet_input_capture',
        'generator_receipt_sha256', 'UNet_NFE', 'IP_conditioned_attention_calls', 'VAE_decoder_calls')
    all_paths = [r['generated_path'] for r in reference_rows + candidate_rows]
    require(len(set(all_paths)) == 4000, 'Cross-arm generated path reuse is forbidden; no pixel-hash uniqueness claim')
    for left, right in zip(reference_rows, candidate_rows):
        require(all(left[k] == right[k] for k in same_fields), 'Paired target/query seed/ACTUAL first-UNet observation differs')
        if kind == 'matched_methods':
            require(all(left[k] == right[k] for k in ('EEG_source_stimulus_id', 'EEG_source_array_row', 'condition_cache_row', 'condition_sha256')),
                'Matched methods must receive exactly the same source condition')
    return verdict(queries=2000, comparison_kind=kind, metadata_pairing_complete=True,
        filesystem_samefile_and_image_payload_verification_still_required=True, per_condition_costs_must_be_reported_separately=True)


def metric_registry(family):
    require(family in ('paper', 'common'), 'Explicit separate paper/common metric family required')
    return dict(primary=dict(PAPER7 if family == 'paper' else COMMON),
        extra_code_metric=dict(PAPER_EXTRA) if family == 'paper' else {},
        direction_applies_to='raw metric values; no gain threshold or winner selection',
        EffNet_warning='paper eval_effnet is a LOWER correlation distance; common EffNet is a distinct HIGHER similarity')


def validate_score_rows(protocol, arm, joined, targets, gallery, scores, family):
    """Future scorer passes its full saved-row identities; metrics stay separate.

    Undefined values remain in the full denominator. No averaging/bootstrap or
    success decision occurs here. Actual evaluator invocation/file closure is
    the future producer's separate obligation.
    """
    validate_joined(protocol, arm, joined, targets, gallery)
    registry = metric_registry(family); required = registry['primary']; extra = registry['extra_code_metric']
    require(len(scores) == 2000, 'Complete score denominator required')
    lookup = {r['query_id']: r for r in scores}
    require(len(lookup) == 2000 and set(lookup) == {r['query_id'] for r in joined}, 'Score query coverage differs')
    included_extra = set(extra) & set(scores[0])
    undefined = collections.Counter()
    for row in joined:
        score = lookup[row['query_id']]
        require(set(extra) & set(score) == included_extra, 'Extra author-code metric coverage differs between rows')
        require(set(required) <= set(score) and not (set(score) & (set(COMMON) if family == 'paper' else set(PAPER7) | set(PAPER_EXTRA))),
            'Paper/common metric columns cannot be mixed or substituted')
        for key in ('subject', 'stimulus_id', 'concept_id', 'generated_path', 'score_target_path', 'EEG_source_stimulus_id'):
            require(score[key] in (None, '') if row[key] is None else score[key] == row[key], 'Score identity differs: ' + key)
        require(str(score['image_seed']) == str(row['image_seed']) and score['target_image_path'] == row['score_target_path'], 'Score target/seed differs')
        if family == 'paper':
            require(score['generated_sha256'] == row['generated_sha256'] and score['target_sha256'] == row['target_image_sha256'], 'Original evaluator target/generated SHA differs')
        for key in set(required) | included_extra:
            value = score[key]
            if value is None or value == '': undefined[key] += 1
            else: require(math.isfinite(float(value)), 'Nonfinite metric must be represented explicitly as undefined')
    return verdict(queries=2000, metric_family=family, directions=registry,
        defined_query_counts={k: 2000 - undefined[k] for k in set(required) | set(extra) if k in scores[0]},
        full_denominator_preserved=True, evaluator_invocation_and_payload_authenticity_not_certified=True)
