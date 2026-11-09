"""Fixed single-subject T80 native image screen for the trial-set pilot.

Prepare the panel before reading fit results.  Two held folds each contribute
four hash-selected concepts, with ALL ten stimuli per concept.  Predictions
must be complete held predictions in raw F coordinates; no oracle norm scale
or target-conditioned generation is implemented.  This is a development
screen, not a full-V result or an independent confirmation experiment.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import signal
import time
import zipfile


ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_trialset_fastscreen_native_T80_20261027_v1'
ARMS = ('mean', 'each_mean')
PREDICTION_COORDINATES = 'raw Linear+learned LayerNorm output in fitted F-target coordinates; no L2 renormalization or query-target scaling'
MAXIMUM_SECONDS = 1800
OUTPUT_CAP = 512 << 20


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref(path):
    return dict(path=str(Path(path).resolve()), sha256=sha(path))


def read(path):
    return json.loads(Path(path).read_bytes())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + '.pending')
    assert not path.exists() and not pending.exists(), 'Never overwrite an existing artifact'
    with pending.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    pending.rename(path)


def identity(row):
    # target_row indexes the pilot's compact target array; it is NOT a THINGS
    # source row.  Require the original source row explicitly.
    source = row.get('source_row', row.get('array_row'))
    assert source is not None, 'Original source_row/array_row is required'
    if 'array_row' in row:
        assert int(row['array_row']) == int(source)
    subject = row.get('subject_id', row.get('subject'))
    assert subject is not None
    fold = int(row['fold'])
    assert fold in (0, 1)
    return dict(row=int(row['row']), source_row=int(source), array_row=int(source),
                subject_id=str(subject), concept_id=str(row['concept_id']),
                stimulus_id=str(row['stimulus_id']), fold=fold)


def metadata(path):
    with Path(path).open(encoding='utf-8', newline='') as stream:
        result = {int(row['array_row']): row for row in csv.DictReader(stream)
                  if row['dataset_split'] == 'train'}
    assert len(result) == 16540
    return result


def choose_panel(index, split, image_metadata, seed=1729):
    clean = [identity(row) for row in index]
    assert len(clean) == 13230 and len({r['row'] for r in clean}) == 13230
    assert len({r['subject_id'] for r in clean}) == 1, 'This screen is single-subject'
    assert {r['array_row'] for r in clean} == set(split['image_rows']['T'])
    concepts = {}
    for row in clean:
        target = image_metadata[row['array_row']]
        assert target['concept_id'] == row['concept_id'] and target['stimulus_id'] == row['stimulus_id']
        concepts.setdefault(row['concept_id'], []).append(row)
    assert len(concepts) == 1323
    for group in concepts.values():
        assert len(group) == 10 and len({r['array_row'] for r in group}) == 10
        assert len({r['fold'] for r in group}) == 1, 'Concept crosses held folds'
    selected = []
    selected_concepts = {}
    for fold in (0, 1):
        eligible = [name for name, group in concepts.items() if group[0]['fold'] == fold]
        assert len(eligible) >= 4
        ranked = sorted(eligible, key=lambda name: (hashlib.sha256(
            f'{SCHEMA}|panel|{seed}|fold{fold}|{name}'.encode()).hexdigest(), name))[:4]
        selected_concepts[str(fold)] = ranked
        for name in ranked:
            for item in sorted(concepts[name], key=lambda row: row['array_row']):
                query = dict(item)
                label = f"{SCHEMA}|{seed}|{query['subject_id']}|{query['stimulus_id']}"
                query['query_id'] = hashlib.sha256(label.encode()).hexdigest()
                query['image_seed'] = int(hashlib.sha256((label + '|noise').encode()).hexdigest()[:15], 16)
                selected.append(query)
    assert len(selected) == 80 and len({r['query_id'] for r in selected}) == 80
    return dict(selected_concepts=selected_concepts, queries=selected)


def prepare_panel(args):
    assert args.index and args.split and args.image_manifest and args.panel
    if args.predictions_root:
        assert not (args.predictions_root / 'summary.json').exists(), 'Seal panel before fit result export'
        assert not any(args.predictions_root.glob('*/predictions.npy'))
    split = read(args.split)
    assert sha(args.image_manifest) == split['image_manifest_sha256']
    selected = choose_panel(rows(args.index), split, metadata(args.image_manifest))
    document = dict(schema=SCHEMA, status='FIXED_BLIND_SINGLE_SUBJECT_T80_PANEL',
                    UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    selection_seed=1729, folds=2, concepts_per_fold=4, stimuli_per_concept=10,
                    subjects=1, expected_queries=80, expected_images=160, arms=list(ARMS),
                    index=ref(args.index), split=ref(args.split), image_manifest=ref(args.image_manifest),
                    results_or_predictions_read=False, full_concept_denominators=True,
                    development_screen_only=True, **selected)
    atomic(args.panel, document)
    print(json.dumps(dict(status=document['status'], panel=ref(args.panel), queries=80, images=160)), flush=True)


def validate_inputs(args):
    assert args.panel and args.predictions_root and args.fit_completion and args.image_manifest
    panel = read(args.panel)
    assert panel['schema'] == SCHEMA and panel['status'] == 'FIXED_BLIND_SINGLE_SUBJECT_T80_PANEL'
    assert panel['arms'] == list(ARMS) and panel['expected_queries'] == 80 and panel['expected_images'] == 160
    for key in ('index', 'split', 'image_manifest'):
        assert sha(panel[key]['path']) == panel[key]['sha256']
    assert sha(args.image_manifest) == panel['image_manifest']['sha256']
    split = read(panel['split']['path'])
    selected = choose_panel(rows(panel['index']['path']), split, metadata(args.image_manifest))
    assert panel['queries'] == selected['queries'] and panel['selected_concepts'] == selected['selected_concepts']
    image_metadata = metadata(args.image_manifest)
    containers = {}
    for query in panel['queries']:
        item = image_metadata[query['array_row']]
        containers.setdefault(item['image_zip'], []).append(item)
    for path, members in containers.items():
        assert Path(path).is_file(), 'Original target container missing before GPU work'
        with zipfile.ZipFile(path) as archive:
            for item in members:
                assert archive.getinfo(item['image_member']).file_size == int(item['image_bytes'])
    complete = read(args.fit_completion)
    assert complete['status'] == 'COMPLETE_REPRESENTATION_PILOT', 'Require successful complete held predictions'
    prediction_index = Path(complete['prediction_index'])
    assert prediction_index.resolve() == (args.predictions_root / 'prediction_index.jsonl').resolve()
    assert sha(prediction_index) == complete['prediction_index_sha256']
    predicted_rows = [identity(row) for row in rows(prediction_index)]
    original = sorted([identity(row) for row in rows(panel['index']['path'])], key=lambda row: row['row'])
    assert predicted_rows == original, 'Prediction/index order or source identities changed'
    positions = {row['row']: position for position, row in enumerate(predicted_rows)}
    assert len(positions) == 13230
    predictions = {}
    import numpy as np
    for arm in ARMS:
        receipt = complete['arms'][arm]
        path = args.predictions_root / arm / 'predictions.npy'
        assert path.resolve() == Path(receipt['predictions_path']).resolve()
        assert sha(path) == receipt['predictions_sha256']
        assert receipt['predictions_shape'] == [13230, 1024]
        assert receipt['oof_coverage_min'] == receipt['oof_coverage_max'] == 1
        assert receipt['prediction_coordinates'] == PREDICTION_COORDINATES, 'Require raw F coordinates; no unit-sphere predictions'
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        assert array.shape == (13230, 1024) and array.dtype == np.float32 and np.isfinite(array).all()
        predictions[arm] = array
    return panel, complete, predictions, positions


def observed_sample(pipe, prompts, tokens, image_seed, torch):
    from generate_BrainHIVE_paired_effect_20261025_v1 import sample_native
    observed = []
    original = pipe.prepare_latents
    def prepare(*args, **kwargs):
        value = original(*args, **kwargs)
        assert value.shape == (1, 4, 64, 64) and torch.isfinite(value).all()
        observed.append(hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        return value
    pipe.prepare_latents = prepare
    try:
        image, proof = sample_native(pipe, prompts, tokens, image_seed, torch)
    finally:
        pipe.prepare_latents = original
    assert len(observed) == 1
    proof['initial_noise_latents_sha256'] = observed[0]
    proof['noise_observation'] = 'actual native prepare_latents return, scheduler-scaled; no RNG replay proxy'
    return image, proof


def attach_targets(image_metadata, generated, output, np, Image):
    # Read target image pixels only AFTER all 160 images have been generated.
    targets = {}
    for row in generated:
        source = row['array_row']
        if source in targets:
            continue
        item = image_metadata[source]
        with zipfile.ZipFile(item['image_zip']) as archive:
            payload = archive.read(item['image_member'])
        assert hashlib.sha256(payload).hexdigest() == item['image_sha256']
        path = output / 'targets' / f'{source:05d}_{Path(item["stimulus_id"]).name}'
        assert not path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        with Image.open(path) as image:
            pixels = np.asarray(image.convert('RGB'))
            assert pixels.ndim == 3 and pixels.shape[2] == 3
        targets[source] = dict(target_image_path=str(path), score_target_path=str(path),
            target_image_sha256=item['image_sha256'], target_source_zip=item['image_zip'],
            target_source_member=item['image_member'])
    assert len(targets) == 80
    joined = [row | targets[row['array_row']] | dict(target_mapping_attached_after_inference=True)
              for row in generated]
    path = output / 'manifest.jsonl'
    with path.open('x', encoding='utf-8') as stream:
        stream.write(''.join(json.dumps(row, allow_nan=False) + '\n' for row in joined))
        stream.flush()
        os.fsync(stream.fileno())
    return path


def generate(args):
    invoked = time.monotonic()
    assert args.native_registration and args.output_dir and args.gpu_job_name
    panel, complete, predictions, positions = validate_inputs(args)
    if args.preflight_only:
        print(json.dumps(dict(status='TRIALSET_T80_ALL_HELD_INPUTS_CPU_PREFLIGHT_PASSED',
                              queries=80, arms=list(ARMS), starts_GPU=False)), flush=True)
        return
    assert not args.output_dir.exists(), 'Single finite attempt; preserve earlier artifacts'
    import brainhive_paired_study_contract_20261025_v1 as C
    assert not args.output_dir.resolve().is_relative_to(C.DATA.resolve()), 'Keep new outputs outside the frozen paired study'
    config = C.read(args.native_registration)
    C.validate(config)
    C.setup_runtime(config)
    import torch
    import numpy as np
    from PIL import Image
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from brainhive_component_generator_v1 import load_native
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_paired_supervision_20261026_v3 import safe_bytes_used
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.output_dir.mkdir(parents=True)
    document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=MAXIMUM_SECONDS,
        output_directory=str(args.output_dir), maximum_output_bytes=OUTPUT_CAP,
        maximum_extra_CPU_bytes=10 << 30, maximum_GPU_bytes=20 << 30,
        minimum_CPU_headroom_bytes=10 << 30, resource_authorization=config['resource_authorization'])
    D.torch = torch
    if args.paused_host_proof:
        from run_gpu_job_BrainHIVE_paused_host_20261027_v1 import verify_wrapper
        wrapper = verify_wrapper(document, args.paused_host_proof)
    else:
        wrapper = D.verify_wrapper(document)
    budget = D.Budget(document)
    interrupted = [False]
    signal.signal(signal.SIGTERM, lambda signum, frame: interrupted.__setitem__(0, True))
    inference = args.output_dir / 'inference_manifest.jsonl'
    generated = []
    try:
        C.resources(config, training_start=True)  # Retain D's resume reserve.
        binding = dict(schema=SCHEMA, source=ref(__file__), panel=ref(args.panel),
            fit_completion=ref(args.fit_completion), image_manifest=ref(args.image_manifest),
            native_registration=ref(args.native_registration), runtime_manifest=config['runtime_manifest'],
            arms=list(ARMS), queries=80, expected_images=160,
            no_target_norm_scaling=True, no_prediction_L2_normalization=True,
            normal_inference_target_access=False, development_screen_only=True)
        atomic(args.output_dir / 'input_provenance.json', binding)
        loader_dir = args.output_dir / 'loader'
        loader_dir.mkdir()
        audit = ConvertedWeightAudit(torch, loader_dir)
        audit.install()
        try:
            pipe, adapter, projector, prompts, host = load_native(training=False)
        finally:
            audit.restore()
        loader_proof = audit.complete()
        assert all(float(processor.scale) == 1. for processor in pipe.unet.attn_processors.values()
                   if type(processor).__name__ == 'IPAttnProcessor2_0')
        budget.check()
        noises = {}
        with inference.open('x', encoding='utf-8', buffering=1) as stream:
            for query_index, query in enumerate(panel['queries']):
                for arm in ARMS:
                    assert not interrupted[0] and time.monotonic() - invoked < MAXIMUM_SECONDS - 120, 'Finite native screen stop'
                    budget.check()
                    assert safe_bytes_used(args.output_dir) < OUTPUT_CAP - (16 << 20)
                    prediction_row = positions[query['row']]
                    raw = np.array(predictions[arm][prediction_row:prediction_row + 1], copy=True)
                    raw_sha = hashlib.sha256(raw.tobytes()).hexdigest()
                    with torch.inference_mode():
                        condition = torch.from_numpy(raw).cuda()
                        tokens = projector(adapter(condition.to(torch.float16)))
                        assert tokens.shape == (1, 4, 2048) and torch.isfinite(tokens).all()
                        image, observation = observed_sample(pipe, prompts, tokens, query['image_seed'], torch)
                    noise = (observation['initial_noise_latents_sha256'], observation['initial_UNet_input_sha256'])
                    if query['query_id'] in noises:
                        assert noises[query['query_id']] == noise, 'Arms used different actual initial noise'
                    else:
                        noises[query['query_id']] = noise
                    path = args.output_dir / arm / 'pixels' / f'{query_index:03d}_{query["array_row"]:05d}.webp'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    pending = path.with_name(path.name + '.pending')
                    assert not path.exists() and not pending.exists()
                    image.save(pending, format='WEBP', lossless=True, quality=90, method=4)
                    assert pending.stat().st_size <= 1 << 20
                    pixels = np.asarray(image)
                    with Image.open(pending) as decoded:
                        assert np.array_equal(pixels, np.asarray(decoded.convert('RGB')))
                    with pending.open('rb') as handle:
                        os.fsync(handle.fileno())
                    pending.rename(path)
                    row = query | observation | dict(arm=arm, global_query_index=query_index,
                        prediction_row=prediction_row, condition_sha256=raw_sha,
                        condition_coordinates=complete['arms'][arm]['prediction_coordinates'],
                        prediction_file_sha256=complete['arms'][arm]['predictions_sha256'],
                        condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                        tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        generated_path=str(path), generated_sha256=sha(path),
                        generated_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        method='BrainHIVE_trialset_fastscreen_' + arm, intervention='correct',
                        repeat_protocol='R4', split='T_HELD_CONCEPT_SCREEN',
                        EEG_source_stimulus_id=query['stimulus_id'], normal_inference_target_access=False,
                        oracle_normalization=False, one_image_per_query=True)
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    os.fsync(stream.fileno())
                    generated.append(row)
                if (query_index + 1) % 10 == 0:
                    print(json.dumps(dict(stage='NATIVE_T80_GENERATION', completed_queries=query_index + 1,
                        expected_queries=80, generated_images=len(generated), expected_images=160,
                        seconds=time.monotonic() - invoked)), flush=True)
        assert len(generated) == 160 and len(noises) == 80
        joined = attach_targets(metadata(args.image_manifest), generated, args.output_dir, np, Image)
        budget.check()
        assert safe_bytes_used(args.output_dir) <= OUTPUT_CAP
        # Reject concurrent replacement of predictions/results while generating.
        for arm in ARMS:
            assert sha(args.predictions_root / arm / 'predictions.npy') == complete['arms'][arm]['predictions_sha256']
        assert sha(args.fit_completion) == binding['fit_completion']['sha256']
        for key in ('panel', 'image_manifest', 'native_registration', 'source'):
            assert sha(binding[key]['path']) == binding[key]['sha256'], 'A bound generation input changed'
        C.resources(config, training_start=True)
        atomic(args.output_dir / 'completion.json', dict(status='COMPLETE_FIXED_T80_TWO_ARM_NATIVE_IMAGES',
            schema=SCHEMA, queries=80, arms=list(ARMS), images=160, manifest=ref(joined),
            inference_manifest=ref(inference), panel=ref(args.panel), input_provenance=ref(args.output_dir / 'input_provenance.json'),
            per_arm_images={arm: 80 for arm in ARMS}, host=host, loader_proof=loader_proof, wrapper=wrapper,
            native_NFE_per_image=4, resolution=[512, 512], CFG=0, IP_scale=1,
            no_best_of_N=True, normal_inference_target_access=False, target_pixels_loaded_after_generation=True,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes,
            seconds=time.monotonic() - invoked, development_screen_only=True, SOTA_established=False))
        print('COMPLETE_FIXED_T80_TWO_ARM_NATIVE_IMAGES', flush=True)
    except BaseException as error:
        failure = args.output_dir / 'failure.json'
        if not failure.exists():
            atomic(failure, dict(status='FAILED_FIXED_DENOMINATOR_PRESERVED_NO_AUTOMATIC_RETRY',
                error=repr(error), observed_images=len(generated), expected_images=160,
                expected_queries=80, seconds=time.monotonic() - invoked, development_screen_only=True))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-panel-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    for name in ('index', 'split', 'image-manifest', 'panel', 'predictions-root',
                 'fit-completion', 'native-registration', 'output-dir', 'paused-host-proof'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--gpu-job-name')
    args = parser.parse_args()
    if args.prepare_panel_only:
        prepare_panel(args)
    else:
        generate(args)


if __name__ == '__main__':
    main()
