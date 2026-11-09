"""Prepare complete four-arm recipes, then admit only real reviewed evidence.

No Torch import, GPU work or changes to the frozen v2 sources. ``prepare``
adds the generation contract BEFORE new two-update profiles. ``admit`` never
relabels an old profile: exact new core hashes and actual runtime proof are
required. Existing profile files are copied and SHA-verified before removal
from the trainer's fixed output path; their recipes/logs remain unchanged.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import brainhive_functional_training_contract_v2 as C

ROOT = Path(__file__).resolve().parent
WORK, DATA = Path(str(C.WORK)), Path(str(C.DATA))
TAG = '20261022_v1'
BASE = ROOT / ('audit/functional_full_lifecycle_' + TAG)
CONFIG = ROOT / ('configs/functional_full_lifecycle_' + TAG)
OLD = ROOT / 'configs/functional_profiles_20261022_v1'
GALLERY = WORK / 'gallery/V/manifest.jsonl'
GALLERY_FEATURES = WORK / 'evaluation/bounded_V_gallery_features_v1'
CONTROLLER = 'queue_BrainHIVE_functional_lifecycle_20261022_v1.py'
SOURCES = (CONTROLLER, Path(__file__).name,
    'train_BrainHIVE_functional_alignment_v2.py', 'generate_BrainHIVE_functional_alignment_V_v2.py',
    'generate_BrainHIVE_functional_postfit_T32_v1.py', 'check_BrainHIVE_functional_postfit_T32_CPU_v1.py',
    'audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py', 'audit_BrainHIVE_spatial_image_integrity_CPU_v1.py',
    'run_BrainHIVE_functional_evaluation_CPU_v1.py', 'compare_BrainHIVE_functional_V_CPU_v2.py',
    'extract_image_features.py', 'score_full_manifest.py', 'score_low_level.py',
    'image_evaluation.py', 'locked_metrics.py', 'semantic_evaluator.py', 'run_gpu_job_v2.py')
EVALUATION_SOURCES = ('extract_image_features.py', 'image_evaluation.py', 'locked_metrics.py', 'semantic_evaluator.py')
GENERATION_KEYS = ('generation_job_names', 'generation_job_seconds',
    'maximum_cumulative_generation_seconds', 'generation_gallery_manifest')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def ref(path):
    return dict(path=str(path), sha256=C.sha(path))


def load_ref(item):
    assert C.sha(item['path']) == item['sha256'], item['path']
    return read(item['path'])


def assert_host():
    assert os.name == 'posix' and ROOT == Path(str(C.REPO)), 'Run only in original private Linux repository'


def gpu_free():
    options = dict(check=True, capture_output=True, text=True, timeout=15)
    active = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], **options)
    assert not active.stdout.strip(), 'An actual GPU process is active'
    free = subprocess.run(['nvidia-smi', '--query-gpu=memory.free', '--format=csv,noheader,nounits'], **options)
    values = [int(line) * (1 << 20) for line in free.stdout.splitlines() if line.strip()]
    assert len(values) == 1 and values[0] >= 16 << 30
    return values[0]


def new_recipe(original, arm, sources, gallery_sha):
    value = copy.deepcopy(original)
    assert value['arm'] == arm
    for name in ('admission_receipt', 'admission_sha256'):
        value.pop(name, None)
    value['source_sha256'] = dict(sorted((value['source_sha256'] | sources).items()))
    value['input_sha256'][GALLERY.as_posix()] = gallery_sha
    prefix = 'BrainHIVE_functional_v2_' + arm + '_expanded_lifecycle_' + TAG
    value.update(profile_job_name=prefix + '_profile', formal_job_names=[prefix + '_fit', prefix + '_exact_resume1'],
        generation_job_names=[prefix + '_images', prefix + '_images_exact_resume1'],
        generation_job_seconds=43200, maximum_cumulative_generation_seconds=86400,
        generation_gallery_manifest=GALLERY.as_posix())
    C.validate_recipe(value, check_files=False)
    assert all(key in value for key in GENERATION_KEYS)
    assert C.core_sha(value) != C.core_sha(original), 'Old profiles cannot certify a different complete recipe'
    return value


def reserve(root, name):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    assert not path.exists()
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.posix_fallocate(descriptor, 0, 16 << 20)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    assert path.stat().st_blocks * 512 >= 16 << 20
    return path


def archive_profile(recipe, arm):
    """Preserve actual proof before releasing the trainer's immutable path."""
    source = Path(recipe['profile_output'])
    if not source.exists():
        return dict(arm=arm, prior_profile_present=False)
    previous = read(source)
    assert previous['status'] == 'ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE'
    assert previous['arm'] == arm and previous['recipe_core_sha256'] == C.core_sha(recipe)
    job = ROOT / 'audit' / (recipe['profile_job_name'] + '_job.json')
    status = read(job)
    assert status['status'] == 'COMPLETED' and not C.process_live(status.get('pid'))
    target = BASE / 'preserved_previous_profiles' / arm
    target.mkdir(parents=True, exist_ok=False)
    destination = target / source.name
    digest = C.sha(source)
    shutil.copy2(source, destination)
    assert C.sha(destination) == C.sha(source) == digest
    write(target / 'preservation.json', dict(original_profile=ref(source), retained_profile=ref(destination),
        old_recipe=ref(OLD / (arm + '.json')), old_job=ref(job),
        reason='New GPU and complete generation recipe require a new two-update engineering profile; no score selection.',
        old_evidence_not_relabelled=True))
    source.unlink()  # One exact copied metadata file; no model, image or result removal.
    return dict(arm=arm, prior_profile_present=True, original_path=str(source), preserved_profile=ref(destination))


def storage_plan(recipes, sources):
    image_base = DATA / ('functional_alignment_v2_images_' + TAG)
    root_binding = WORK / ('audit/functional_lifecycle_root_volume_' + TAG)
    image_base.mkdir(parents=True, exist_ok=True)
    root_binding.mkdir(parents=True, exist_ok=True)
    volume_roots = [root_binding, image_base]
    # Contract describes every actual filesystem exactly once.
    unique = {}
    for root in volume_roots:
        unique.setdefault(root.stat().st_dev, root)
    volumes = []
    for root in unique.values():
        reservation = reserve(root, 'retained_16MiB_reservation.bin')
        volumes.append(dict(root=str(root), reservation_path=str(reservation),
            minimum_reservation_allocated_bytes=16 << 20, minimum_buffer_bytes=512 << 20))
    entries = []
    for arm in C.ARMS:
        run = C.run_name(arm)
        metadata = WORK / 'images' / run
        image_root = image_base / arm
        image_root.mkdir(parents=True, exist_ok=False)
        entries.append(dict(arm=arm, candidate=recipes[arm]['candidate'], run_name=run,
            recipe_core_sha256=C.core_sha(recipes[arm]), formal_queries=16600,
            training_root=recipes[arm]['output_directory'], metadata_root=str(metadata),
            inference_manifest=str(metadata / 'inference_manifest.jsonl'),
            feature_root=str(WORK / 'evaluation' / run / 'features'),
            score_root=str(metadata / 'scores'), low_level_root=str(metadata / 'low_level'),
            artifact_audit_root=str(metadata / 'artifact_CPU_v1'),
            training_budget_bytes=64_000_000, feature_budget_bytes=320_000_000,
            metadata_budget_bytes=256 << 20, bytes_per_future_image=350000,
            lifecycle_source_sha256=sources, image_segments=[dict(start=0, stop=16600, root=str(image_root))]))
    return dict(schema='BrainHIVE_functional_alignment_v2_four_arm_storage_plan',
        all_four_train_generate_score_lifecycles_implemented=True,
        recipes=[dict(arm=a, path=str(CONFIG / (a + '.json')), recipe_core_sha256=C.core_sha(recipes[a])) for a in C.ARMS],
        volumes=volumes, arms=entries,
        evaluation=dict(schema='BrainHIVE_functional_evaluation_CPU_v1_sealed_plan', maximum_seconds=14400,
            maximum_output_bytes=64 << 20, minimum_CPU_headroom_bytes=2 << 30,
            gallery_features_root=str(GALLERY_FEATURES), lifecycle_source_sha256=sources),
        comparison=dict(schema='BrainHIVE_functional_V_comparison_CPU_v2_sealed_plan', maximum_seconds=1800,
            maximum_output_bytes=64 << 20, lifecycle_source_sha256=sources,
            gallery_features_root=str(GALLERY_FEATURES), gallery_features_reuse_only=True,
            paired_statistics_plan=ref(ROOT / 'configs/paired_statistics_plan.json'),
            output_root=str(WORK / 'images' / C.run_name('PN') / 'comparison_CPU_v2')),
        execution=dict(controller_source=CONTROLLER, fixed_arm_order=list(C.ARMS),
            postfit_T32_before_full_V=True, full_V_all_four_before_comparison=True,
            automatic_metric_selection=False, maximum_controller_seconds=6 * 86400,
            maximum_feature_seconds_per_arm=14400, maximum_feature_attempts_per_arm=1,
            postfit_seconds_per_arm=1200, postfit_bytes_per_arm=64 << 20,
            postfit_total_bytes=256 << 20, additional_postfit_persistent_buffer_bytes=64 << 20,
            controller_metadata_cap_bytes=64 << 20,
            original_main_and_diagnostic_must_be_terminal=True,
            formal_admission_not_implied_by_plan=True))


def prepare():
    assert_host()
    assert not BASE.exists() and not CONFIG.exists(), 'Preserve previous preparation attempt'
    gpu_free()
    old_recipes = {a: read(OLD / (a + '.json')) for a in C.ARMS}
    originals = {a: ref(OLD / (a + '.json')) for a in C.ARMS}
    for recipe in old_recipes.values():
        C.validate_recipe(recipe)
    source_pins = {name: C.sha(ROOT / name) for name in SOURCES}
    recipes = {a: new_recipe(old_recipes[a], a, source_pins, C.sha(GALLERY)) for a in C.ARMS}
    for recipe in recipes.values():
        C.validate_recipe(recipe)
    BASE.mkdir(); CONFIG.mkdir()
    archives = [archive_profile(old_recipes[a], a) for a in C.ARMS]
    plan = storage_plan(recipes, source_pins)
    plan_path = BASE / 'four_arm_lifecycle_plan.json'
    write(plan_path, plan)
    # Use already quantified diagnostic/ledger evidence, with new receipt dates
    # and the new exact recipe core. UNKNOWN remains UNKNOWN.
    refs = []
    for arm in C.ARMS:
        old_admission = read(old_recipes[arm]['admission_receipt'])
        for key in ('functional_prerequisite_review', 'ledger_review'):
            load_ref(old_admission[key])
        receipt = copy.deepcopy(old_admission)
        now = time.time()
        receipt.update(status='ENGINEERING_PROFILE_ONLY_ADMITTED', purpose='profile',
            candidate=recipes[arm]['candidate'], recipe_core_sha256=C.core_sha(recipes[arm]),
            source_sha256=recipes[arm]['source_sha256'], issued_epoch=now, expires_epoch=now + 7 * 86400,
            terminal_prerequisites=[ref(Path(str(p))) for p in (C.MAIN, C.DIAG)],
            formal_fit_or_full_V_admitted=False, new_complete_recipe_requires_new_actual_profile=True)
        receipt_path = BASE / (arm + '_profile_admission.json')
        write(receipt_path, receipt)
        recipes[arm].update(admission_receipt=str(receipt_path), admission_sha256=C.sha(receipt_path))
        C.verify_admission(recipes[arm], 'profile')
        path = CONFIG / (arm + '.json')
        write(path, recipes[arm])
        refs.append(dict(arm=arm, **ref(path), recipe_core_sha256=C.core_sha(recipes[arm])))
    funded = C.formal_remaining(plan, recipes)
    assert {a: ref(OLD / (a + '.json')) for a in C.ARMS} == originals
    write(BASE / 'preparation.json', dict(status='COMPLETE_LIFECYCLE_RECIPES_READY_FOR_NEW_GPU_ENGINEERING_PROFILES',
        recipes=refs, previous_recipes=originals, previous_profiles_preserved=archives,
        plan=ref(plan_path), remaining_storage=funded, source_sha256=source_pins,
        formal_training_admitted=False, formal_training_started=False, new_images=0,
        all_four_actual_new_core_profiles_complete=False,
        known_pending=['Four actual profiles with complete recipe cores on new GPU',
            'Full frozen evaluation runtime readiness and gallery-feature validation',
            'Separate formal admission after actual source/input/resource checks']))
    return dict(status='PREPARED_PROFILE_ONLY_NO_FORMAL_FIT', directory=str(BASE), remaining_storage=funded)


def evaluation_ready(path, plan):
    value = read(path)
    assert value['status'] == 'FROZEN_EVALUATION_ASSETS_READY'
    assert value['metric_selection_or_tuning'] is False
    for name in EVALUATION_SOURCES:
        assert value['source_sha256'][name] == plan['evaluation']['lifecycle_source_sha256'][name] == C.sha(ROOT / name)
    assert value['assets'] and len({r['path'] for r in value['assets']}) == len(value['assets'])
    for asset in value['assets']:
        assert C.sha(asset['path']) == asset['sha256']
    probe = load_ref(value['actual_GPU_probe'])
    assert probe['status'] == 'FROZEN_FOUR_METRIC_MODELS_RUNTIME_PROBE_COMPLETE'
    assert probe['actual_GPU'] is True and probe['stages'] == ['DINO', 'EffNet', 'CapSim', 'ObjectF1']
    assert probe['metric_selection_or_tuning'] is False
    probe_source = ROOT / 'probe_BrainHIVE_evaluation_environment_GPU_20261022_v2.py'
    assert C.sha(probe_source) == probe['source_sha256']
    job = read(probe['wrapper_receipt'])
    assert job['status'] == 'COMPLETED' and job['exit_code'] == 0
    assert not C.process_live(job.get('pid')) and str(probe_source) in job['command']
    assert job['pid'] == probe['pid']
    assert all(probe[key] == 0 for key in ('trained_parameters', 'new_generated_images',
        'paired_metrics_computed', 'score_tables_read', 'feature_tensors_saved'))
    assert probe['optimizer_created'] is False and probe['training_seed'] == 1729
    assert [row['stage'] for row in probe['stage_records']] == probe['stages']
    assert all(row['original_route_completed'] is True and row['optimizer_created'] is False
        and row['parameter_gradients_absent'] is True and row['frozen_parameters'] > 0 for row in probe['stage_records'])
    for input_path, digest in probe['input_sha256'].items():
        assert C.sha(input_path) == digest
    import run_BrainHIVE_functional_evaluation_CPU_v1 as evaluation
    gallery = [json.loads(line) for line in GALLERY.read_text().splitlines()]
    assert len(gallery) == len({r['stimulus_id'] for r in gallery}) == 1660
    pins = {str(GALLERY): C.sha(GALLERY)}
    evaluation.feature_inputs(GALLERY_FEATURES, GALLERY, [r['stimulus_id'] for r in gallery],
        True, pins, ROOT, time.monotonic() + 300)
    return dict(readiness=ref(path), gallery_inputs_sha256=pins)


def admit(evaluation_path):
    assert_host()
    preparation = read(BASE / 'preparation.json')
    assert preparation['status'] == 'COMPLETE_LIFECYCLE_RECIPES_READY_FOR_NEW_GPU_ENGINEERING_PROFILES'
    plan = load_ref(preparation['plan'])
    recipes = {a: read(CONFIG / (a + '.json')) for a in C.ARMS}
    for arm, recipe in recipes.items():
        assert C.validate_recipe(recipe) == next(i['recipe_core_sha256'] for i in preparation['recipes'] if i['arm'] == arm)
    profiles = []
    for arm, recipe in recipes.items():
        profile = read(recipe['profile_output'])
        assert profile['status'] == 'ACTUAL_PARENT_Q_TWO_FORWARD_ENGINEERING_PROFILE_COMPLETE'
        assert profile['arm'] == arm and profile['recipe_core_sha256'] == C.core_sha(recipe)
        assert profile['actual_GPU'] is True and profile['updates'] == 2 and profile['batch_size'] == 16
        assert profile['UNet_forwards_per_update'] == [2, 2] and profile['engineering_checks_passed'] is True
        assert profile['actual_parent_Q_engineering_verified'] is True and profile['formal_training'] is False
        assert profile['generated_images'] == profile['saved_trained_checkpoints'] == 0
        assert profile['linear_training_time_forecast_seconds'] <= recipe['training_job_seconds']
        assert profile['source_sha256'] == recipe['source_sha256']
        job = ROOT / 'audit' / (recipe['profile_job_name'] + '_job.json')
        assert read(job)['status'] == 'COMPLETED' and not C.process_live(read(job).get('pid'))
        profiles.append(dict(arm=arm, **ref(recipe['profile_output'])))
    readiness = evaluation_ready(evaluation_path, plan)
    funded = C.formal_remaining(plan, recipes)
    gpu_free()
    assert C.resource_headroom()['effective_headroom_bytes'] >= 8 << 30
    final_refs = []
    # The plan binds recipe CORES, not entire files: adding admission references
    # cannot create a recipe -> receipt -> plan -> recipe-file SHA cycle.
    for arm, recipe in recipes.items():
        receipt = read(recipe['admission_receipt'])
        assert receipt['purpose'] == 'profile'
        now = time.time()
        receipt.update(status='FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMITTED', purpose='formal',
            issued_epoch=now, expires_epoch=now + 7 * 86400, actual_GPU_profiles=profiles,
            four_arm_lifecycle_plan=preparation['plan'],
            terminal_prerequisites=[ref(Path(str(p))) for p in (C.MAIN, C.DIAG)],
            scope=['training', 'actual_images', 'per_query_evaluation'],
            frozen_evaluation_readiness=readiness,
            formal_fit_or_full_V_admitted=True, image_gain_or_mechanism_established=False)
        path = BASE / (arm + '_formal_admission.json')
        write(path, receipt)
        recipe.update(admission_receipt=str(path), admission_sha256=C.sha(path))
        temporary = CONFIG / (arm + '.formal.pending')
        write(temporary, recipe)
        final_refs.append(dict(arm=arm, pending=temporary, final=CONFIG / (arm + '.json')))
    # Preserve the previous profile-only recipe bytes before changing the two
    # fields deliberately excluded from core_sha. Core never changes here.
    profile_archive = BASE / 'profile_only_recipe_bytes'
    profile_archive.mkdir(exist_ok=False)
    for item in final_refs:
        shutil.copy2(item['final'], profile_archive / item['final'].name)
        assert C.sha(profile_archive / item['final'].name) == C.sha(item['final'])
        item['pending'].replace(item['final'])
    for recipe in recipes.values():
        C.verify_admission(recipe, 'formal')
    write(BASE / 'formal_admission_closure.json', dict(status='FOUR_ARM_FULL_LIFECYCLE_FORMAL_ADMISSION_ACTUALLY_VERIFIED',
        plan=preparation['plan'], recipes=[dict(arm=a, **ref(CONFIG / (a + '.json')),
            recipe_core_sha256=C.core_sha(recipes[a])) for a in C.ARMS], actual_profiles=profiles,
        evaluation=readiness, remaining_storage=funded, formal_training_started=False,
        generated_images=0, scientific_gain_established=False))
    return dict(status='FORMAL_ADMISSION_VERIFIED_NO_TRAINING_STARTED', remaining_storage=funded)


def postfit(arm):
    """Late-bind only the current real final head; never invent future heads."""
    assert_host()
    import generate_BrainHIVE_functional_postfit_T32_v1 as T
    recipes = {a: read(CONFIG / (a + '.json')) for a in C.ARMS}
    recipe = recipes[arm]
    C.verify_admission(recipe, 'formal')
    folder = Path(recipe['output_directory'])
    done, calibration, provenance = [read(folder / n) for n in ('completion.json', 'calibration.json', 'input_provenance.json')]
    T.final_metadata(recipe, done, calibration, provenance)
    query = ROOT / 'configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json'
    assert C.sha(query) == T.QUERY_SHA
    doc = read(query)
    inputs = dict(recipe['input_sha256'])
    inputs[str(query)] = T.QUERY_SHA
    oracle = doc['source_selection']['oracle_inference_inputs_path']
    inputs[oracle] = doc['source_selection']['oracle_inference_inputs_sha256']
    for row in doc['rows']:
        for name in ('condition_cache', 'parent_checkpoint', 'parent_registration'):
            inputs[row[name + '_path']] = row[name + '_sha256']
    for name in ('completion.json', 'calibration.json', 'input_provenance.json'):
        inputs[str(folder / name)] = C.sha(folder / name)
    inputs[done['checkpoint_path']] = done['checkpoint_sha256']
    for path, digest in inputs.items():
        assert C.sha(path) == digest
    plan = dict(schema=T.SCHEMA, status=T.READY, arm=arm, candidate=recipe['candidate'],
        training_seed=1729, subject='sub-01', queries=32, generator=T.GENERATOR,
        maximum_seconds=T.SECONDS, maximum_arm_output_bytes=T.ARM_BYTES,
        maximum_four_arm_output_bytes=T.TOTAL_BYTES, output_directory=str(T.BASE / arm),
        query_manifest=ref(query), formal_recipes=[dict(arm=a, **ref(CONFIG / (a + '.json'))) for a in C.ARMS],
        source_sha256={name: C.sha(ROOT / name) for name in sorted(T.SOURCES)}, input_sha256=inputs,
        job_name='BrainHIVE_functional_v2_' + arm + '_expanded_lifecycle_' + TAG + '_postfit_T32',
        engineering_only=True, semantic_evaluation_enabled=False, automatic_retry=False)
    now = time.time()
    admission = dict(schema='BrainHIVE_functional_postfit_T32_v1_reviewed_admission',
        status='CURRENT_ARM_FINAL_POSTFIT_T32_ENGINEERING_ADMITTED', arm=arm,
        plan_core_sha256=T.plan_core(plan), reviewed_internal_engineering_facts=True,
        image_gain_claimed=False, issued_epoch=now, expires_epoch=now + 86400,
        terminal_state_sha256=T.terminal_check(), maximum_seconds=T.SECONDS,
        maximum_arm_output_bytes=T.ARM_BYTES, maximum_four_arm_output_bytes=T.TOTAL_BYTES,
        final_completion_sha256=C.sha(folder / 'completion.json'),
        resource_limits=dict(minimum_CPU_headroom_bytes=8 << 30, minimum_free_GPU_bytes=16 << 30))
    path = BASE / (arm + '_postfit_admission.json')
    write(path, admission)
    plan['postfit_admission'] = ref(path)
    output = CONFIG / (arm + '_postfit_T32.json')
    write(output, plan)
    T.prepare(plan)
    return dict(status='CURRENT_REAL_FINAL_POSTFIT_WRAPPER_VERIFIED_NO_IMAGES', path=str(output))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['prepare', 'admit', 'postfit'])
    parser.add_argument('--evaluation-readiness', type=Path)
    parser.add_argument('--arm', choices=C.ARMS)
    args = parser.parse_args()
    if args.mode == 'prepare':
        result = prepare()
    elif args.mode == 'admit':
        assert args.evaluation_readiness is not None
        result = admit(args.evaluation_readiness)
    else:
        assert args.arm is not None
        result = postfit(args.arm)
    print(json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    main()
