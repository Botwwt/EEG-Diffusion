"""Generic native-HIVE image screen of complete, provenance-bound OOF predictions.

The reused T80 panel is already exposed development data.  It is neither a new
blind selection nor independent confirmation.  Every explicit arm generates all
80 images with the same observed query noise, four native UNet evaluations and
no target-dependent scale.  Old frozen trial-set programs are not modified.
"""
import argparse
import hashlib
import re
from pathlib import Path
import signal
import time
import zipfile

import generate_BrainHIVE_trialset_fastscreen_20261027_v1 as G

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_generic_native_prediction_screen_20261027_v1'
ARMS = ('native_parent', 'posterior_mean', 'component_mean')
F_COORDINATES = 'raw F-target coordinates by inverse outer-train centered PCA; no L2 renormalization or query-target scaling'
NATIVE_COORDINATES = 'raw native BrainMLP OOF output; original affine retained; no query target access or scaling'
MAX_SECONDS = 1800
OUTPUT_CAP = 768 << 20
sha, ref, read, rows, atomic = G.sha, G.ref, G.read, G.rows, G.atomic


def validate_arms(arms):
    assert 2 <= len(arms) <= 6 and len(set(arms)) == len(arms)
    assert all(re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,63}', arm) for arm in arms)
    return tuple(arms)


def validate_inputs(args):
    import numpy as np
    arms = validate_arms(args.arms)
    panel = read(args.panel)
    assert panel['status'] == 'FIXED_BLIND_SINGLE_SUBJECT_T80_PANEL'
    assert panel['expected_queries'] == 80 and len(panel['queries']) == 80
    # Preserve the existing selection and seeds exactly; its later exposure is
    # declared in every new provenance/completion record.
    for key in ('index', 'split', 'image_manifest'):
        assert sha(panel[key]['path']) == panel[key]['sha256']
    assert sha(args.image_manifest) == panel['image_manifest']['sha256']
    metadata = G.metadata(args.image_manifest)
    selected = G.choose_panel(rows(panel['index']['path']), read(panel['split']['path']), metadata)
    assert selected['queries'] == panel['queries'] and selected['selected_concepts'] == panel['selected_concepts']
    containers = {}
    for query in panel['queries']:
        item = metadata[query['array_row']]
        containers.setdefault(item['image_zip'], []).append(item)
    for path, members in containers.items():
        with zipfile.ZipFile(path) as archive:
            for item in members:
                assert archive.getinfo(item['image_member']).file_size == int(item['image_bytes'])
    complete = read(args.fit_completion)
    assert complete['status'] == 'COMPLETE_EEG_CONDITIONAL_MODEL_SCREEN'
    index_path = Path(complete['prediction_index'])
    if getattr(args, 'predictions_root', None):
        assert index_path.resolve() == (args.predictions_root / 'prediction_index.jsonl').resolve()
    assert sha(index_path) == complete['prediction_index_sha256']
    index = [G.identity(row) for row in rows(index_path)]
    original = sorted([G.identity(row) for row in rows(panel['index']['path'])], key=lambda row: row['row'])
    assert index == original and len(index) == 13230
    positions = {row['row']: position for position, row in enumerate(index)}
    assert len(positions) == 13230
    receipts = {}
    if 'native_parent' in arms:
        assert args.native_parent_binding, 'Native baseline has separate provenance; do not rewrite fit summary'
        native = read(args.native_parent_binding)
        assert native['status'] == 'COMPLETE_NATIVE_PARENT_OOF_PREDICTIONS'
        assert native['prediction_index_sha256'] == complete['prediction_index_sha256']
        assert native['prediction_coordinates'] == NATIVE_COORDINATES
        receipts['native_parent'] = native
    predictions = {}
    for arm in arms:
        receipt = receipts.get(arm, complete['arms'].get(arm))
        assert receipt is not None, f'Missing complete prediction receipt for {arm}'
        expected_coordinates = NATIVE_COORDINATES if arm == 'native_parent' else F_COORDINATES
        assert receipt['prediction_coordinates'] == expected_coordinates
        assert receipt['predictions_shape'] == [13230, 1024]
        assert receipt['oof_coverage_min'] == receipt['oof_coverage_max'] == 1
        path = Path(receipt['predictions_path'])
        assert sha(path) == receipt['predictions_sha256']
        array = np.load(path, mmap_mode='r', allow_pickle=False)
        assert array.dtype == np.float32 and array.shape == (13230, 1024) and np.isfinite(array).all()
        receipts[arm] = receipt
        predictions[arm] = array
    return panel, complete, predictions, positions, receipts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('panel', 'fit-completion', 'native-parent-binding', 'image-manifest', 'native-registration', 'output-dir'):
        parser.add_argument('--' + name, type=Path, required=name not in ('native-parent-binding', 'native-registration', 'output-dir'))
    parser.add_argument('--arms', nargs='+', default=list(ARMS))
    parser.add_argument('--predictions-root', type=Path, help='Optional extra binding of prediction_index parent')
    parser.add_argument('--gpu-job-name')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--max-seconds', type=int, default=1800)
    args = parser.parse_args(argv)
    assert 0 < args.max_seconds <= MAX_SECONDS
    invoked = time.monotonic()
    arms = validate_arms(args.arms)
    expected_images = 80 * len(arms)
    panel, complete, predictions, positions, receipts = validate_inputs(args)
    if args.preflight_only:
        print(G.json.dumps(dict(status='GENERIC_OOF_NATIVE_INPUTS_CPU_PREFLIGHT_PASSED',
                               queries=80, arms=list(arms), images=expected_images, starts_GPU=False,
                               panel_already_exposed_development=True)), flush=True)
        return 0
    assert args.native_registration and args.output_dir and args.gpu_job_name
    assert not args.output_dir.exists(), 'Preserve earlier attempts; no adaptive replay'
    import brainhive_paired_study_contract_20261025_v1 as C
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
    document = dict(gpu_job_name=args.gpu_job_name, maximum_seconds=args.max_seconds,
                    maximum_output_bytes=OUTPUT_CAP, maximum_extra_CPU_bytes=10 << 30,
                    maximum_GPU_bytes=20 << 30, minimum_CPU_headroom_bytes=10 << 30,
                    resource_authorization=config['resource_authorization'])
    D.torch = torch
    wrapper = D.verify_wrapper(document)
    budget = D.Budget(document)
    stopped = [False]
    signal.signal(signal.SIGTERM, lambda signum, frame: stopped.__setitem__(0, True))
    generated, noises = [], {}
    inference = args.output_dir / 'inference_manifest.jsonl'
    try:
        C.resources(config, training_start=True)
        binding = dict(schema=SCHEMA, source=ref(__file__), helper_source=ref(G.__file__),
                       panel=ref(args.panel), fit_completion=ref(args.fit_completion),
                       native_parent_binding=ref(args.native_parent_binding) if args.native_parent_binding else None,
                       image_manifest=ref(args.image_manifest), native_registration=ref(args.native_registration),
                       arms=list(arms), queries=80, expected_images=expected_images,
                       predictions={arm: dict(path=receipts[arm]['predictions_path'], sha256=receipts[arm]['predictions_sha256'],
                                               coordinates=receipts[arm]['prediction_coordinates']) for arm in arms},
                       no_target_norm_scaling=True, no_prediction_L2_normalization=True,
                       normal_inference_target_access=False, panel_already_exposed_development=True,
                       independent_confirmation=False, development_screen_only=True)
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
        with inference.open('x', encoding='utf-8', buffering=1) as stream:
            for query_index, query in enumerate(panel['queries']):
                for arm in arms:
                    assert not stopped[0] and time.monotonic() - invoked < args.max_seconds - 120
                    budget.check()
                    assert safe_bytes_used(args.output_dir) < OUTPUT_CAP - (16 << 20)
                    prediction_row = positions[query['row']]
                    raw = np.array(predictions[arm][prediction_row:prediction_row + 1], copy=True)
                    with torch.inference_mode():
                        condition = torch.from_numpy(raw).cuda()
                        tokens = projector(adapter(condition.to(torch.float16)))
                        assert tokens.shape == (1, 4, 2048) and torch.isfinite(tokens).all()
                        image, observation = G.observed_sample(pipe, prompts, tokens, query['image_seed'], torch)
                    noise = (observation['initial_noise_latents_sha256'], observation['initial_UNet_input_sha256'])
                    if query['query_id'] in noises:
                        assert noises[query['query_id']] == noise, 'All arms must share actual initial query noise'
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
                    with pending.open('rb') as stream_sync:
                        G.os.fsync(stream_sync.fileno())
                    pending.rename(path)
                    row = query | observation | dict(arm=arm, global_query_index=query_index,
                        prediction_row=prediction_row, condition_sha256=hashlib.sha256(raw.tobytes()).hexdigest(),
                        condition_coordinates=receipts[arm]['prediction_coordinates'],
                        prediction_file_sha256=receipts[arm]['predictions_sha256'],
                        condition_FP16_sha256=hashlib.sha256(condition.to(torch.float16).cpu().contiguous().numpy().tobytes()).hexdigest(),
                        tokens_sha256=hashlib.sha256(tokens.cpu().contiguous().numpy().tobytes()).hexdigest(),
                        generated_path=str(path), generated_sha256=sha(path),
                        generated_pixel_sha256=hashlib.sha256(pixels.tobytes()).hexdigest(),
                        method='BrainHIVE_prediction_fastscreen_' + arm, intervention='correct',
                        repeat_protocol='R4', split='T_HELD_CONCEPT_EXPOSED_DEVELOPMENT_SCREEN',
                        EEG_source_stimulus_id=query['stimulus_id'], normal_inference_target_access=False,
                        oracle_normalization=False, one_image_per_query=True,
                        panel_already_exposed_development=True)
                    stream.write(G.json.dumps(row, allow_nan=False) + '\n')
                    stream.flush()
                    G.os.fsync(stream.fileno())
                    generated.append(row)
                if (query_index + 1) % 10 == 0:
                    print(G.json.dumps(dict(stage='GENERIC_NATIVE_PREDICTION_SCREEN', queries=query_index + 1,
                        expected_queries=80, images=len(generated), expected_images=expected_images,
                        seconds=time.monotonic() - invoked)), flush=True)
        assert len(generated) == expected_images and len(noises) == 80
        joined = G.attach_targets(G.metadata(args.image_manifest), generated, args.output_dir, np, Image)
        budget.check()
        assert safe_bytes_used(args.output_dir) <= OUTPUT_CAP
        for arm in arms:
            assert sha(receipts[arm]['predictions_path']) == receipts[arm]['predictions_sha256']
        assert sha(args.fit_completion) == binding['fit_completion']['sha256']
        if args.native_parent_binding:
            assert sha(args.native_parent_binding) == binding['native_parent_binding']['sha256']
        C.resources(config, training_start=True)
        atomic(args.output_dir / 'completion.json', dict(status='COMPLETED_GENERIC_NATIVE_PREDICTION_SCREEN',
            schema=SCHEMA, queries=80, arms=list(arms), images=expected_images, manifest=ref(joined),
            inference_manifest=ref(inference), panel=ref(args.panel), input_provenance=ref(args.output_dir / 'input_provenance.json'),
            per_arm_images={arm: 80 for arm in arms}, host=host, loader_proof=loader_proof, wrapper=wrapper,
            native_NFE_per_image=4, resolution=[512, 512], CFG=0, IP_scale=1,
            no_best_of_N=True, normal_inference_target_access=False, target_pixels_loaded_after_generation=True,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes, peak_GPU_bytes=budget.peak_GPU_bytes,
            seconds=time.monotonic() - invoked, panel_already_exposed_development=True,
            independent_confirmation=False, development_screen_only=True, SOTA_established=False))
        return 0
    except BaseException as error:
        atomic(args.output_dir / 'failure.json', dict(status='FAILED_GENERIC_SCREEN_FIXED_DENOMINATOR_NO_AUTOMATIC_RETRY',
            error=repr(error), observed_images=len(generated), expected_images=expected_images,
            expected_queries=80, arms=list(arms), seconds=time.monotonic() - invoked,
            panel_already_exposed_development=True, no_effect_claim=True))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
