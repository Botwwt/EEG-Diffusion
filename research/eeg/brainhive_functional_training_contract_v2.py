"""Independent v2 recipe/mapping/admission contracts; no Torch or GPU access.

A reviewed receipt is SHA-bound engineering evidence, NOT a digital signature
or a request for external approval. No such admission is issued by this module.
"""
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import time

ROOT = Path(__file__).resolve().parent
REPO = PurePosixPath('/root/atm_diffusion_20261015/research/eeg_sota_autoresearch_20261018_v1')
WORK = PurePosixPath('/root/brainhive_reproduction_20261020_v1')
DATA = PurePosixPath('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')
GENERATOR_ASSET_ROOT = PurePosixPath('/dev/shm/eeg_diffusion_next_20261018_v1')
ARMS = ('O', 'P', 'PN', 'S')
CONFIG_FILES = ('model_index.json', 'unet/config.json', 'scheduler/scheduler_config.json',
    'vae/config.json', 'text_encoder/config.json', 'text_encoder_2/config.json',
    'tokenizer/merges.txt', 'tokenizer/special_tokens_map.json', 'tokenizer/tokenizer_config.json', 'tokenizer/vocab.json',
    'tokenizer_2/merges.txt', 'tokenizer_2/special_tokens_map.json', 'tokenizer_2/tokenizer_config.json', 'tokenizer_2/vocab.json')
FIXED_INPUTS = tuple(str(path) for path in (
    WORK / 'data/component_input_contract_v1.json', WORK / 'data/fused_train_targets.pt',
    WORK / 'data/generation_inputs/V.json', WORK / 'data/generation_inputs/V_targets.jsonl',
    REPO / 'configs/split_manifest.json', REPO / 'audit/image_manifest.csv',
    REPO / 'configs/resource_authorization_20261020_v3.json',
    REPO / 'configs/ATM_public_decoder_registration_20261020_v1.json',
    WORK / 'audit/native_asset_verification.json',
    REPO / 'reproduction_sources/BrainHIVE_f06db69f/_fetch_receipt.json',
    *(REPO / 'reproduction_sources/BrainHIVE_f06db69f/main' / name for name in ('layers.py', 'models_brain.py', 'models_adapter.py')),
    *(WORK / 'assets/prior_H14' / module / name for module in ('fusion_encoder', 'proj', 'attn_adapter')
      for name in ('config.json', 'diffusion_pytorch_model.safetensors')),
    GENERATOR_ASSET_ROOT / 'sdxl_turbo_fp16.safetensors',
    *(GENERATOR_ASSET_ROOT / 'sdxl_config' / name for name in CONFIG_FILES)))
MAIN = WORK / 'audit/main_storage_continuation_20261021_v1.json'
DIAG = WORK / 'audit/native_function_T32_queue_v2.json'
MAIN_TERMINAL = {
    'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED',
    'FINITE_MAIN_STORAGE_CONTINUATION_PARTIAL_WITH_EXPLICIT_FAILURES',
    'STORAGE_FORECAST_STOPPED_PRESERVED_NO_CASCADING_JOBS',
    'FAILED_MAIN_STORAGE_CONTINUATION_PRESERVED_NEEDS_DIAGNOSIS',
}
DIAG_TERMINAL = {
    'FROZEN_FUNCTIONAL_DIAGNOSTIC_SAVED_NO_FORMAL_COMPONENT_TRAINING',
    'DIAGNOSTIC_FAILED_WITH_PRESERVED_LOG_NO_AUTOMATIC_RETRY',
    'STOPPED_WITH_EXPLICIT_DIAGNOSTIC_FAILURE',
}
SOURCES = (
    'brainhive_native_models_v1.py', 'brainhive_bounded_component_v1.py',
    'brainhive_component_data_v1.py', 'brainhive_component_generator_v1.py',
    'train_BrainHIVE_bounded_component_v1.py', 'brainhive_response_rank_v1.py',
    'brainhive_functional_alignment_v2.py', 'brainhive_functional_training_contract_v2.py',
    'train_BrainHIVE_functional_alignment_v2.py', 'run_gpu_job_v2.py',
)


_HASH_CACHE = {}


def sha(path):
    """First read hashes all bytes; reuse only while actual file identity is unchanged."""
    resolved = Path(path).resolve(strict=True)
    details = resolved.stat()
    identity = (details.st_dev, details.st_ino, details.st_size, details.st_mtime_ns, details.st_ctime_ns)
    key = str(resolved)
    if key not in _HASH_CACHE or _HASH_CACHE[key][0] != identity:
        with resolved.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        after = resolved.stat()
        assert identity == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'File changed during SHA audit'
        _HASH_CACHE[key] = (identity, digest)
    return _HASH_CACHE[key][1]


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def core_sha(recipe):
    return hashlib.sha256(canonical({key: value for key, value in recipe.items()
        if key not in {'admission_receipt', 'admission_sha256'}})).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def bound_read(item):
    assert PurePosixPath(item['path']).is_absolute() and sha(item['path']) == item['sha256']
    return read(item['path'])


def validate_recipe(recipe, *, check_files=True):
    arm = recipe['arm']
    assert arm in ARMS
    fixed = dict(schema='BrainHIVE_functional_alignment_v2_recipe',
        candidate='functional_alignment_v2_' + arm, training_seed=1729, subjects=10,
        unique_T_images=13230, unique_OOF_training_queries=132300, passes=3,
        total_training_query_exposures=396900, expected_updates=24807,
        batch_size=16, learning_rate=.0001, weight_decay=.01, gradient_clip=1.,
        trainable_parameters=525568, predict_risk=False, gate_policy='fixed_one',
        maximum_standardized_direction_radius=.25, timestep_policy='uniform_integer_0_999',
        reference_Q_policy='frozen_unmodified_parent_same_query_t_noise_detached',
        training_UNet_forwards_per_update=2, token_angular_weight=.1,
        generator_asset_root=str(GENERATOR_ASSET_ROOT),
        functional_auxiliary_weight=.1, functional_epsilon=1e-6,
        maximum_query_positions=16, expected_native_IP_layers=70,
        minimum_nondegenerate_teacher_fraction=.5,
        engineering_profile_updates=2, engineering_profile_batch_size=16,
        training_job_seconds=86400, maximum_cumulative_training_seconds_per_arm=172800,
        checkpoint_selection='fixed_final_three_passes',
        strength_policy='Dfit_only_global_strength_calibration',
        output_directory=str(WORK / 'components/functional_alignment_v2' / arm),
        profile_output=str(WORK / 'audit/functional_alignment_v2' / arm / 'actual_GPU_profile.json'),
        generator=dict(resolution=512, steps=4, CFG=0, IP_scale=1, images_per_query=1))
    for key, expected in fixed.items():
        assert recipe[key] == expected, (key, recipe.get(key), expected)
    assert 90 < recipe['training_job_seconds'] <= 86400
    assert 90 < recipe['profile_job_seconds'] <= 3600
    assert recipe['training_job_seconds'] <= recipe['maximum_cumulative_training_seconds_per_arm'] <= 172800
    assert isinstance(recipe['formal_job_names'], list) and 1 <= len(recipe['formal_job_names']) <= 2
    names = recipe['formal_job_names'] + [recipe['profile_job_name']]
    assert len(names) == len(set(names)) and all(isinstance(name, str) and Path(name).name == name for name in names)
    assert set(SOURCES) <= set(recipe['source_sha256'])
    assert all(Path(name).name == name for name in recipe['source_sha256'])
    assert PurePosixPath(recipe['latent_receipt']).is_relative_to(WORK / 'audit')
    mandatory = set(FIXED_INPUTS) | {recipe['latent_receipt']}
    assert mandatory <= set(recipe['input_sha256'])
    assert all(PurePosixPath(path).is_absolute() for path in recipe['input_sha256'])
    assert recipe['input_sha256'][str(WORK / 'data/component_input_contract_v1.json')] == recipe['input_contract_sha256']
    if check_files:
        for filename, expected in recipe['source_sha256'].items():
            assert sha(ROOT / filename) == expected, filename
        for filename, expected in recipe['input_sha256'].items():
            assert sha(filename) == expected, filename
        registration = read(str(REPO / 'configs/ATM_public_decoder_registration_20261020_v1.json'))
        assert registration['generator_asset_root'] == recipe['generator_asset_root']
        selected = [item for item in registration['generator_assets'] if 'sdxl_turbo_fp16' in item['path']]
        assert len(selected) == 1
        assert selected[0]['path'] == str(GENERATOR_ASSET_ROOT / 'sdxl_turbo_fp16.safetensors')
        assert selected[0]['sha256'] == recipe['input_sha256'][selected[0]['path']]
        actual_config_files = {str(path) for path in Path(str(GENERATOR_ASSET_ROOT / 'sdxl_config')).rglob('*') if path.is_file()}
        assert {str(GENERATOR_ASSET_ROOT / 'sdxl_config' / name) for name in CONFIG_FILES} <= actual_config_files
        assert actual_config_files <= set(recipe['input_sha256']), 'Pin every actual native config/tokenizer file, not an assumed ZIP inventory'
        latent = read(recipe['latent_receipt'])
        assert latent['path'] in recipe['input_sha256'] and recipe['input_sha256'][latent['path']] == latent['sha256']
    return core_sha(recipe)


def teacher_mappings(array_rows, concept_by_row, *, seed=1729):
    """Concept-level fixed bijections; array values are NEVER tensor positions.

    The actual allowed T has ten stimuli per concept. Reject an incompatible
    input instead of padding, dropping, or silently treating a concept as a row.
    """
    assert seed == 1729 and len(array_rows) == len(set(array_rows)) > 0
    assert set(array_rows) <= set(concept_by_row)
    groups = {}
    for position, row in enumerate(array_rows):
        groups.setdefault(str(concept_by_row[row]), []).append(position)
    assert len(groups) >= 4, 'Four distinct concepts are required by the fixed 1/2/3 shifts'
    assert len({len(group) for group in groups.values()}) == 1, 'Unbalanced concept input requires reviewed mapping'
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    for key in keys:
        groups[key].sort(key=lambda position: array_rows[position])
    result = {}
    for name, shift in [('negative', 1), ('rank_positive_shuffled', 2), ('rank_negative_shuffled', 3)]:
        mapping = [-1] * len(array_rows)
        for k, concept in enumerate(keys):
            source, target = groups[concept], groups[keys[(k + shift) % len(keys)]]
            for original, teacher in zip(source, target):
                mapping[original] = teacher
        assert sorted(mapping) == list(range(len(array_rows)))
        assert all(concept_by_row[array_rows[i]] != concept_by_row[array_rows[j]] for i, j in enumerate(mapping))
        result[name] = mapping
    assert all(concept_by_row[array_rows[a]] != concept_by_row[array_rows[b]]
               for a, b in zip(result['rank_positive_shuffled'], result['rank_negative_shuffled']))
    result['receipt'] = dict(seed=seed, concept_count=len(keys), stimuli_per_concept=len(groups[keys[0]]),
        arrays_are_positions_not_array_rows=True, all_three_mappings_are_bijections=True,
        both_shuffled_rank_teachers_have_different_query_concepts=True,
        concept_level_fixed_shift_policy=dict(negative=1, rank_positive_shuffled=2, rank_negative_shuffled=3),
        array_rows_sha256=hashlib.sha256(canonical(array_rows)).hexdigest(),
        mapping_sha256={key: hashlib.sha256(canonical(value)).hexdigest() for key, value in result.items()})
    return result


def process_live(pid):
    try:
        text = (Path('/proc') / str(int(pid)) / 'stat').read_text()
        return text[text.rfind(')') + 2:].split()[0] not in ('Z', 'X')
    except (OSError, ValueError, TypeError):
        return False


def resource_headroom():
    status = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key, value = line.split(':', 1)
        if key == 'MemAvailable':
            status['host_available_bytes'] = int(value.split()[0]) * 1024
    assert 'host_available_bytes' in status
    remaining = [status['host_available_bytes']]
    bounds = []
    # Traverse the visible cgroup and its ancestors rather than trusting host RAM.
    group = Path('/sys/fs/cgroup')
    memberships = [line.split(':', 2)[2] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    locations = [group]
    if memberships:
        candidate = group / memberships[0].lstrip('/')
        while candidate.is_relative_to(group):
            locations.append(candidate)
            if candidate == group:
                break
            candidate = candidate.parent
    for location in dict.fromkeys(locations):
        limit = location / 'memory.max'
        current = location / 'memory.current'
        if limit.exists() and current.exists() and limit.read_text().strip() != 'max':
            cap, used = int(limit.read_text()), int(current.read_text())
            remaining.append(max(0, cap - used))
            bounds.append(dict(path=str(location), limit=cap, current=used))
    status.update(cgroup_bounds=bounds, effective_headroom_bytes=min(remaining))
    return status


def private_path(value):
    p = Path(value)
    assert p.is_absolute() and not p.is_symlink()
    resolved = p.resolve()
    assert any(resolved.is_relative_to(Path(str(root))) and resolved != Path(str(root)) for root in (WORK, DATA))
    return p


def directory_bytes(path):
    p = Path(path)
    return sum(item.stat().st_size for item in p.rglob('*') if item.is_file() and not item.is_symlink()) if p.exists() else 0


def run_name(arm):
    assert arm in ARMS
    return 'protected_R4_functional_alignment_v2_' + arm


def query_identity(subject, query, version):
    identity = f"{subject}/{query['stimulus_id']}/R4"
    noise_seed = int(hashlib.sha256(f"1729|{subject}|{query['stimulus_id']}|R4|{version}".encode()).hexdigest()[:16], 16) % (2**63 - 1)
    return identity, noise_seed


def validate_group_identity(plan, recipes):
    """Pure metadata checks: no arm may borrow another arm's host or outputs."""
    assert set(recipes) == set(ARMS)
    baseline = recipes['O']
    for arm, recipe in recipes.items():
        assert recipe['arm'] == arm and recipe['candidate'] == 'functional_alignment_v2_' + arm
        for key in ('source_sha256', 'input_sha256', 'input_contract_sha256', 'latent_receipt',
                    'generator_asset_root', 'generator', 'reference_Q_policy', 'timestep_policy', 'strength_policy'):
            assert recipe[key] == baseline[key], ('Four-arm common frozen host/data pin differs', arm, key)
    assert {entry['arm'] for entry in plan['arms']} == set(ARMS) and len(plan['arms']) == 4
    roots = []
    for entry in plan['arms']:
        arm = entry['arm']
        expected_run = run_name(arm)
        assert entry['candidate'] == 'functional_alignment_v2_' + arm and entry['run_name'] == expected_run
        assert entry['recipe_core_sha256'] == core_sha(recipes[arm])
        assert entry['training_root'] == str(WORK / 'components/functional_alignment_v2' / arm)
        assert entry['metadata_root'] == str(WORK / 'images' / expected_run)
        assert entry['inference_manifest'] == str(WORK / 'images' / expected_run / 'inference_manifest.jsonl')
        assert entry['feature_root'] == str(WORK / 'evaluation' / expected_run / 'features')
        for segment in entry['image_segments']:
            root = PurePosixPath(segment['root'])
            assert root.is_absolute() and '..' not in root.parts
            for other_arm, other_root in roots:
                assert arm == other_arm or not (root == other_root or root.is_relative_to(other_root) or other_root.is_relative_to(root)), 'Cross-arm image roots alias'
            roots.append((arm, root))
    return True


def verified_manifest_prefix(entry, inputs, protocol_version):
    """Only exact own-arm query/file/SHA evidence can reduce future budget."""
    manifest = private_path(entry['inference_manifest'])
    records = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    expected = [(subject, query) for subject in inputs['subjects'] for query in inputs['inputs']]
    assert len(expected) == 16600 and len(records) <= len(expected)
    if records:
        done = read(Path(entry['training_root']) / 'completion.json')
        assert done['status'] == 'FUNCTIONAL_ALIGNMENT_V2_ALL10_REAL_OOF_FIXED_FINAL_COMPONENT_COMPLETE'
        assert done['candidate'] == entry['candidate'] and done['arm'] == entry['arm']
        assert done['recipe_core_sha256'] == entry['recipe_core_sha256']
        assert sha(done['checkpoint_path']) == done['checkpoint_sha256']
    paths = set()
    for index, row in enumerate(records):
        subject, query = expected[index]
        identity, noise_seed = query_identity(subject, query, protocol_version)
        assert row['global_query_index'] == index and row['query_id'] == identity
        assert row['subject'] == subject and row['stimulus_id'] == row['EEG_source_stimulus_id'] == query['stimulus_id']
        assert row['array_row'] == query['array_row'] and row['image_seed'] == noise_seed
        assert row['candidate'] == entry['candidate'] and row['arm'] == entry['arm']
        assert row['recipe_core_sha256'] == entry['recipe_core_sha256'] and row['run_name'] == entry['run_name']
        assert row['component_checkpoint_sha256'] == done['checkpoint_sha256']
        segment = next(item for item in entry['image_segments'] if item['start'] <= index < item['stop'])
        expected_path = Path(segment['root']) / entry['run_name'] / subject / (Path(query['stimulus_id']).stem + '.webp')
        assert row['generated_path'] == str(expected_path) and str(expected_path) not in paths
        private_path(expected_path)
        assert expected_path.is_file() and sha(expected_path) == row['generated_sha256']
        paths.add(str(expected_path))
    return records


def formal_remaining(plan, recipes=None):
    assert plan['schema'] == 'BrainHIVE_functional_alignment_v2_four_arm_storage_plan'
    assert plan['all_four_train_generate_score_lifecycles_implemented'] is True
    recipes = recipes or {entry['arm']: read(entry['path']) for entry in plan['recipes']}
    validate_group_identity(plan, recipes)
    inputs = read(str(WORK / 'data/generation_inputs/V.json'))
    assert inputs['expected_queries'] == 16600 and len(inputs['subjects']) == 10 and len(inputs['inputs']) == 1660
    assert not inputs['target_image_or_class_fields_present'] and inputs['repeat_protocol'] == 'R4'
    assert all(set(row) == {'stimulus_id', 'array_row'} for row in inputs['inputs'])
    protocol_version = read(str(REPO / 'configs/split_manifest.json'))['version']
    volumes = {}
    for volume in plan['volumes']:
        root = private_path(volume['root'])
        assert root.is_dir()
        device = root.stat().st_dev
        assert device not in volumes, 'Describe each actual filesystem once'
        reserve = private_path(volume['reservation_path'])
        assert reserve.is_file() and reserve.stat().st_dev == device
        assert stat.S_IMODE(reserve.stat().st_mode) == 0o600
        assert reserve.stat().st_blocks * 512 >= volume['minimum_reservation_allocated_bytes'] >= 16 << 20
        assert volume['minimum_buffer_bytes'] >= 256 << 20
        volumes[device] = dict(root=str(root), required=volume['minimum_buffer_bytes'],
            free=shutil.disk_usage(root).free, metadata_reserve_only=True)
    for entry in plan['arms']:
        assert entry['formal_queries'] == 16600 and entry['bytes_per_future_image'] >= 330000
        assert entry['training_budget_bytes'] >= 16 << 20
        assert entry['feature_budget_bytes'] > 0 and entry['metadata_budget_bytes'] > 0
        training_root = private_path(entry['training_root'])
        assert str(training_root) == str(WORK / 'components/functional_alignment_v2' / entry['arm'])
        for name, expected in entry['lifecycle_source_sha256'].items():
            assert Path(name).name == name and sha(ROOT / name) == expected
        assert len(entry['lifecycle_source_sha256']) >= 2
        records = verified_manifest_prefix(entry, inputs, protocol_version)
        done = {row['global_query_index'] for row in records}
        assert len(done) == len(records) and done <= set(range(16600))
        segments = entry['image_segments']
        assert segments[0]['start'] == 0 and segments[-1]['stop'] == 16600
        assert all(a['stop'] == b['start'] for a, b in zip(segments, segments[1:]))
        for segment in segments:
            image_root = private_path(segment['root'])
            assert image_root.is_dir() and segment['start'] < segment['stop']
            for other in plan['arms']:
                if other['arm'] == entry['arm']:
                    continue
                for other_segment in other['image_segments']:
                    other_root = private_path(other_segment['root']).resolve()
                    assert not (image_root.resolve() == other_root or image_root.resolve().is_relative_to(other_root) or other_root.is_relative_to(image_root.resolve())), 'Resolved cross-arm image roots alias'
            device = image_root.stat().st_dev
            assert device in volumes
            missing = sum(i not in done for i in range(segment['start'], segment['stop']))
            volumes[device]['required'] += missing * entry['bytes_per_future_image']
        paths = [('training_root', 'training_budget_bytes'), ('feature_root', 'feature_budget_bytes'),
                 ('metadata_root', 'metadata_budget_bytes')]
        for path_key, budget_key in paths:
            p = private_path(entry[path_key])
            ancestor = p
            while not ancestor.exists():
                ancestor = ancestor.parent
            device = ancestor.stat().st_dev
            assert device in volumes
            volumes[device]['required'] += max(0, entry[budget_key] - directory_bytes(p))
    for volume in volumes.values():
        assert volume['free'] >= volume['required'], ('Unfunded complete four-arm lifecycle', volume)
    return list(volumes.values())


def verify_admission(recipe, purpose):
    """Fresh CPU preconditions. Never issues a receipt or starts a GPU operation."""
    assert purpose in ('profile', 'formal')
    core = validate_recipe(recipe)
    path = Path(recipe['admission_receipt'])
    assert path.is_absolute() and sha(path) == recipe['admission_sha256']
    receipt = read(path)
    assert receipt['schema'] == 'BrainHIVE_functional_alignment_v2_reviewed_admission'
    expected_status = 'ENGINEERING_PROFILE_ONLY_ADMITTED' if purpose == 'profile' else 'FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED'
    assert receipt['status'] == expected_status and receipt['purpose'] == purpose
    assert receipt['issuer'] == 'verified_internal_engineering_audit'
    assert receipt['candidate'] == recipe['candidate'] and receipt['recipe_core_sha256'] == core
    assert receipt['source_sha256'] == recipe['source_sha256']
    assert receipt['sha_bound_review_not_cryptographic_signature'] is True
    assert receipt['issued_epoch'] <= time.time() < receipt['expires_epoch']
    assert 0 < receipt['expires_epoch'] - receipt['issued_epoch'] <= 7 * 86400
    prerequisites = receipt['terminal_prerequisites']
    assert {item['path'] for item in prerequisites} == {str(MAIN), str(DIAG)} and len(prerequisites) == 2
    for item in prerequisites:
        state = bound_read(item)
        terminal = MAIN_TERMINAL if item['path'] == str(MAIN) else DIAG_TERMINAL
        assert state['status'] in terminal and not process_live(state.get('pid'))
        if purpose == 'formal' and item['path'] == str(MAIN):
            assert state['status'] == 'ALL_THREE_COMPLETE_REAL_OOF_TRAINING_IMAGE_AND_QUERY_COMPARISONS_SAVED'
    diagnostic = bound_read(receipt['functional_prerequisite_review'])
    assert diagnostic['status'] == 'FROZEN_NATIVE_FIXED_Q_FUNCTIONAL_PREREQUISITE_REVIEWED'
    assert diagnostic['actual_GPU_diagnostic'] and diagnostic['native_fixed_Q_reused']
    assert diagnostic['reference_condition'] == 'missing_raw_condition'
    assert diagnostic['teacher_response_nondegenerate'] and diagnostic['EEG_pairing_signal_present']
    assert diagnostic['engineering_prerequisite_is_not_image_gain'] is True
    actual_summary = bound_read(diagnostic['actual_diagnostic_summary'])
    actual_provenance = bound_read(diagnostic['actual_diagnostic_provenance'])
    actual_completion = bound_read(diagnostic['actual_diagnostic_completion'])
    assert actual_summary['status'] == 'FROZEN_T_ONLY_FUNCTIONAL_MEASUREMENTS_SAVED'
    assert len(actual_summary['by_timestep']) == 4 and all(row['queries'] == 32 for row in actual_summary['by_timestep'])
    assert actual_completion['status'] == 'COMPLETE_FROZEN_NATIVE_T32_FUNCTIONAL_DIAGNOSTIC_NO_IMAGE_QUALITY_CLAIM'
    assert actual_completion['summary_sha256'] == diagnostic['actual_diagnostic_summary']['sha256']
    assert actual_completion['provenance_sha256'] == diagnostic['actual_diagnostic_provenance']['sha256']
    assert actual_completion['generated_images'] == actual_completion['fitted_parameters'] == 0
    assert actual_completion['native_IP_layers'] == 70 and actual_completion['real_T_queries'] == 32
    assert actual_completion['engineering_checks_passed'] and not actual_completion['protected_scores_read']
    assert actual_provenance['Q_reference'] == (
        'Exactly one missing_zero_raw_condition full-UNet pass per query/time; its detached native Q reused for every response arm. Fixed before data, no response-specific Q.')
    assert actual_provenance['missing_definition'] == (
        'Exactly zeros_like(raw_OOF_EEG) at input to frozen A; not BrainMLP(zero EEG) and not zero IP tokens')
    assert actual_provenance['all_conditions_share_same_real_latent_noise_and_timestep']
    assert actual_provenance['generated_images'] == actual_provenance['parameter_updates'] == 0
    assert not actual_provenance['optimizer_created']
    ledger = bound_read(receipt['ledger_review'])
    assert ledger['status'] == 'INHERITED_LIMITS_RECONCILED_FOUR_ARM_PLAN_RECORDED'
    assert ledger['candidate_limit'] == 12 and ledger['structural_revision_limit'] == 2
    assert ledger['candidate_limit_not_reset'] is True and 1 <= ledger['structural_revision_index'] <= 2
    assert {entry['arm'] for entry in ledger['arms']} == set(ARMS) and len(ledger['arms']) == 4
    assert all(type(entry['counts_as_new_candidate']) is bool for entry in ledger['arms'])
    added = sum(entry['counts_as_new_candidate'] for entry in ledger['arms'])
    assert 0 <= ledger['used_before'] <= ledger['used_after'] == ledger['used_before'] + added <= 12
    snapshot = ledger['ledger_snapshot']
    assert sha(snapshot['path']) == snapshot['sha256']
    assert (ROOT / 'experiment_ledger.jsonl').read_bytes().startswith(Path(snapshot['path']).read_bytes())
    memory = resource_headroom()
    limits = receipt['resource_limits']
    assert limits['minimum_cgroup_headroom_bytes'] >= 2 << 30
    assert memory['effective_headroom_bytes'] >= limits['minimum_cgroup_headroom_bytes']
    assert limits['minimum_free_GPU_bytes'] > 0
    metadata = limits['profile_metadata_budget_bytes']
    assert 65536 <= metadata <= 16 << 20
    assert shutil.disk_usage(str(WORK)).free >= metadata + limits['minimum_retained_disk_buffer_bytes']
    assert limits['minimum_retained_disk_buffer_bytes'] >= 64 << 20
    result = dict(recipe_core_sha256=core, admission_sha256=sha(path), memory=memory)
    if purpose == 'formal':
        plan = bound_read(receipt['four_arm_lifecycle_plan'])
        assert {entry['arm'] for entry in plan['recipes']} == set(ARMS) and len(plan['recipes']) == 4
        siblings = {}
        for item in plan['recipes']:
            sibling = read(item['path'])
            assert sibling['arm'] == item['arm'] and validate_recipe(sibling) == item['recipe_core_sha256']
            if item['arm'] == recipe['arm']:
                assert item['recipe_core_sha256'] == core
            siblings[item['arm']] = sibling
        assert {item['arm'] for item in receipt['actual_GPU_profiles']} == set(ARMS)
        for item in receipt['actual_GPU_profiles']:
            profile = bound_read(item)
            expected = next(entry['recipe_core_sha256'] for entry in plan['recipes'] if entry['arm'] == item['arm'])
            assert profile['status'] == 'ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE'
            assert profile['recipe_core_sha256'] == expected and profile['actual_GPU'] is True
            assert profile['arm'] == item['arm'] and profile['updates'] == 2 and profile['batch_size'] == 16
            assert profile['UNet_forwards_per_update'] == [2, 2] and not profile['formal_training']
            assert profile['generated_images'] == 0 and profile['engineering_checks_passed']
            assert profile['actual_parent_Q_engineering_verified'] is True
            assert profile['provenance']['reference_Q_policy'] == 'frozen_unmodified_parent_same_query_t_noise_detached'
            assert profile['linear_training_time_forecast_seconds'] <= sibling_training_seconds(plan, item['arm'])
        result['remaining_storage'] = formal_remaining(plan, siblings)
    return result


def sibling_training_seconds(plan, arm):
    item = next(entry for entry in plan['recipes'] if entry['arm'] == arm)
    return read(item['path'])['training_job_seconds']
