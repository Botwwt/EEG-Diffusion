"""Finite T-only visual-teacher intervention; no EEG inference or model fitting.

Reuses all 80 earlier T query seeds (8 stimuli x 10 seed slots), generates each
of correct F / wrong F / missing once, and measures fixed DINO cosine margins.
This follow-up cannot change the earlier diagnostic's UNKNOWN decision.
"""
import argparse
import csv
import gc
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parent
SCHEMA = 'BrainHIVE_T_visual_teacher_image_intervention_20261025_v1'
CONDITIONS = ('correct_F', 'wrong_F', 'missing')


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def bound(item):
    assert sha(item['path']) == item['sha256'], item['path']
    return read(item['path'])


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_rows(rows, protocol, metadata):
    assert len(rows) == 80
    assert len({r['array_row'] for r in rows}) == len({r['concept_id'] for r in rows}) == 8
    assert {r['subject'] for r in rows} == {'sub-%02d' % i for i in range(1, 11)}
    assert len({r['trajectory_seed'] for r in rows}) == 80
    for index, row in enumerate(rows):
        negative = rows[row['negative_query_index']]
        actual = metadata[row['array_row']]
        assert row['array_row'] in protocol['image_rows']['T']
        assert row['stimulus_id'] == actual['stimulus_id'] and row['concept_id'] == actual['concept_id']
        assert actual['dataset_split'] == 'train'
        assert row['negative_array_row'] == negative['array_row']
        assert row['source_role'] == negative['source_role'] and row['subject'] == negative['subject']
        assert row['concept_id'] != negative['concept_id']
        assert type(row['trajectory_seed']) is int and 0 <= row['trajectory_seed'] < 2**63
    return rows


def prepare(path):
    config = read(path)
    assert config['schema'] == SCHEMA and config['conditions'] == list(CONDITIONS)
    assert config['automatic_new_fit'] is False and config['images_per_condition'] == 80
    assert config['maximum_seconds'] == 1200 and config['maximum_extra_CPU_bytes'] == 10 << 30
    assert config['maximum_GPU_bytes'] == 20 << 30 and config['maximum_output_bytes'] == 512 << 20
    assert config['source_sha256'][Path(__file__).name] == sha(__file__)
    for name, digest in config['source_sha256'].items():
        assert Path(name).name == name and sha(ROOT / name) == digest
    assert bound(config['closed_recovery'])['status'] == 'UNCHANGED_FROZEN_FUNCTION_AND_FIXED_DIRECTION_READOUT_COMPLETE'
    summary = bound(config['function_summary'])
    assert summary['status'] == 'COMPLETE_FROZEN_T_ONLY_FUNCTIONAL_DIAGNOSTIC'
    assert summary['coverage']['missing_cells'] == 0
    assert bound(config['fixed_direction'])['decision']['priority_hypothesis'] == 'FUNCTIONAL_TEACHER_EVIDENCE_UNRESOLVED'
    protocol = bound(config['split_manifest'])
    assert sha(config['image_manifest']['path']) == config['image_manifest']['sha256'] == protocol['image_manifest_sha256']
    with Path(config['image_manifest']['path']).open(encoding='utf-8', newline='') as stream:
        metadata = {int(r['array_row']): r for r in csv.DictReader(stream) if r['dataset_split'] == 'train'}
    rows = verify_rows(bound(config['queries'])['rows'], protocol, metadata)
    contract = bound(config['input_contract'])
    assert sha(contract['target_cache_path']) == contract['target_cache_sha256']
    assert sha(config['DINO_weight']['path']) == config['DINO_weight']['sha256']
    # Validate every actual JPEG before admitting a GPU job, including zip members.
    for index in sorted({r['array_row'] for r in rows}):
        entry = metadata[index]
        with zipfile.ZipFile(entry['image_zip']) as archive:
            value = archive.read(entry['image_member'])
        assert len(value) == int(entry['image_bytes']) and hashlib.sha256(value).hexdigest() == entry['image_sha256']
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import verify_config
    recovery, runtime, original = verify_config(Path(config['loader_recovery_registration']['path']))
    assert sha(config['loader_recovery_registration']['path']) == config['loader_recovery_registration']['sha256']
    original = dict(original, maximum_seconds=1200, output_directory=config['output_directory'],
                    gpu_job_name=config['gpu_job_name'], maximum_output_bytes=512 << 20)
    assert shutil.disk_usage(Path(config['output_directory']).parent).free >= 6 << 30
    return config, rows, metadata, contract, runtime, original


def cosine_margin_rows(generated, targets, rows):
    """Pure scalar algebra after fixed feature extraction; no fitted thresholds."""
    import torch
    assert generated.shape[0] == 240 and targets.shape[0] == 8
    assert torch.isfinite(generated).all() and torch.isfinite(targets).all()
    ids = sorted({r['array_row'] for r in rows})
    positions = {value: index for index, value in enumerate(ids)}
    results = []
    for query_index, query in enumerate(rows):
        for condition_index, condition in enumerate(CONDITIONS):
            feature = generated[query_index * 3 + condition_index]
            positive = float(feature @ targets[positions[query['array_row']]])
            negative = float(feature @ targets[positions[query['negative_array_row']]])
            results.append(dict(query_index=query_index, subject_seed_slot=query['subject'],
                array_row=query['array_row'], negative_array_row=query['negative_array_row'],
                condition=condition, target_cosine=positive, negative_cosine=negative,
                paired_cosine_margin=positive-negative))
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    config_sha = sha(args.registration)
    config, rows, metadata, contract, runtime, original = prepare(args.registration)
    if args.preflight_only:
        print('T_ONLY_TEACHER_IMAGE_CPU_PREFLIGHT_PASSED_80_SEEDS_8_EXACT_JPEGS_NO_GPU')
        return
    assert 'diffusers' not in sys.modules and 'transformers' not in sys.modules
    sys.path.insert(0, runtime['path'])
    import brainhive_native_models_v1
    import accelerate
    import torch
    from PIL import Image
    import numpy as np
    import diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2 as D
    from run_BrainHIVE_frozen_function_low_memory_20261025_v3 import ConvertedWeightAudit
    from brainhive_component_generator_v1 import load_native
    assert accelerate.__version__ == '0.34.2'
    D.torch = torch
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    wrapper = D.verify_wrapper(original)
    output = Path(config['output_directory'])
    output.mkdir(exist_ok=False)
    (output/'images').mkdir(); (output/'targets').mkdir(); (output/'loader').mkdir()
    budget = D.Budget(original)
    records = []
    try:
        write(output/'registration.json', config)
        write(output/'queries.json', dict(rows=rows))
        targets = []
        for index in sorted({r['array_row'] for r in rows}):
            entry = metadata[index]
            with zipfile.ZipFile(entry['image_zip']) as archive:
                value = archive.read(entry['image_member'])
            assert hashlib.sha256(value).hexdigest() == entry['image_sha256']
            path = output/'targets'/('%05d.jpg' % index)
            path.write_bytes(value); targets.append(path)
        audit = ConvertedWeightAudit(torch, output/'loader')
        audit.install()
        try:
            pipe, adapter, projector, prompts, host = load_native(training=False)
        finally:
            audit.restore()
        proof = audit.complete()
        cache = torch.load(contract['target_cache_path'], map_location='cpu', mmap=True, weights_only=True)
        features = cache[[r['array_row'] for r in rows]].clone().cuda()
        del cache
        permutation = torch.tensor([r['negative_query_index'] for r in rows], device='cuda')
        with torch.no_grad():
            tokens = {key: D.encode_tokens(adapter, projector, value) for key, value in
                dict(correct_F=features, wrong_F=features[permutation], missing=torch.zeros_like(features)).items()}
        del features
        budget.check()
        generated = []
        for qi, query in enumerate(rows):
            initial_hashes = []
            for condition in CONDITIONS:
                observed = []
                def capture(module, positional, keyword):
                    sample = positional[0] if positional else keyword['sample']
                    if not observed:
                        observed.append(D.tensor_digest(sample))
                handle = pipe.unet.register_forward_pre_hook(capture, with_kwargs=True)
                try:
                    with torch.no_grad():
                        image = pipe(height=512, width=512,
                            generator=torch.Generator(device='cuda').manual_seed(query['trajectory_seed']),
                            prompt_embeds=prompts[0], pooled_prompt_embeds=prompts[1],
                            cross_attention_kwargs={'ip_hidden_states':tokens[condition][qi:qi+1]},
                            num_inference_steps=4, guidance_scale=0., output_type='pil').images[0].convert('RGB')
                finally:
                    handle.remove()
                assert len(observed) == 1
                initial_hashes.append(observed[0])
                assert image.size == (512,512)
                buffer = io.BytesIO(); image.save(buffer,format='WEBP',lossless=True,quality=90,method=4)
                path = output/'images'/('%02d_%s.webp' % (qi,condition));path.write_bytes(buffer.getvalue())
                with Image.open(path) as decoded:
                    assert np.array_equal(np.asarray(image), np.asarray(decoded.convert('RGB')))
                records.append(dict(query_index=qi, array_row=query['array_row'], stimulus_id=query['stimulus_id'],
                    condition=condition, seed=query['trajectory_seed'], initial_UNet_input_sha256=observed[0],
                    tokens_sha256=D.tensor_digest(tokens[condition][qi:qi+1]), path=str(path), sha256=sha(path)))
                generated.append(path);budget.check()
            assert len(set(initial_hashes)) == 1, 'All three interventions require identical initial noise'
            if (qi+1)%8 == 0: print(json.dumps(dict(stage='T_TEACHER_IMAGES',queries=qi+1,expected=80)),flush=True)
        assert len(records)==240
        assert len({(p.stat().st_dev,p.stat().st_ino) for p in generated})==240
        write(output/'image_manifest.json',records)
        del pipe,adapter,projector,prompts,tokens
        gc.collect();torch.cuda.empty_cache()
        from image_evaluation import dino_model, dino_features
        model=dino_model()
        target_features=dino_features(targets,model,batch_size=8,real_images=True)
        generated_features=dino_features(generated,model,batch_size=16,real_images=False)
        torch.save(dict(targets=target_features,generated=generated_features),output/'dino_features.pt')
        scores=cosine_margin_rows(generated_features,target_features,rows)
        write(output/'paired_DINO_scores.json',scores)
        import statistics
        metrics=('target_cosine','paired_cosine_margin')
        means={c:{m:statistics.fmean(r[m] for r in scores if r['condition']==c) for m in metrics} for c in CONDITIONS}
        deltas={c:{m:means['correct_F'][m]-means[c][m] for m in metrics} for c in ('wrong_F','missing')}
        per_stimulus={str(i):{c:{m:statistics.fmean(r[m] for r in scores if r['array_row']==i and r['condition']==c)
            for m in metrics} for c in CONDITIONS} for i in sorted({r['array_row'] for r in rows})}
        budget.check();assert sha(args.registration)==config_sha
        for n,d in config['source_sha256'].items():assert sha(ROOT/n)==d
        summary=dict(status='COMPLETE_T_ONLY_VISUAL_TEACHER_IMAGE_INTERVENTION',registration_sha256=config_sha,
            images=240,query_seeds=80,unique_T_stimuli=8,subject_labels_are_seed_slots_not_EEG_inputs=True,
            conditions=list(CONDITIONS),means=means,correct_F_minus_control=deltas,per_stimulus=per_stimulus,
            DINO_weight_sha256=config['DINO_weight']['sha256'],metric='Frozen DINOv3 cosine and positive-minus-registered-negative cosine; no retrieval accuracy claim',
            original_function_summary=config['function_summary'],original_direction_unchanged=True,
            interpretation='Exploratory fixed T-only follow-up to the known-noise UNKNOWN; not a new scientific admission or independent confirmation',
            no_optimizer_created=True,parameter_updates=0,EEG_or_final_head_used=False,normal_inference_permission=False,
            V_or_official_test_inputs_read=False,automatic_new_fit=False,host=host,wrapper=wrapper,
            loader_proof=proof,elapsed_seconds=time.monotonic()-budget.started,
            peak_extra_CPU_bytes=budget.peak_extra_CPU_bytes,peak_GPU_bytes=budget.peak_GPU_bytes,
            image_manifest_sha256=sha(output/'image_manifest.json'),scores_sha256=sha(output/'paired_DINO_scores.json'))
        write(output/'summary.json',summary)
        write(output/'completion.json',dict(status=summary['status'],summary_sha256=sha(output/'summary.json')))
        print(summary['status'],flush=True)
    except BaseException as error:
        write(output/'failure.json',dict(status='FAILED_T_ONLY_TEACHER_IMAGE_INTERVENTION_PRESERVED',
            error=repr(error),completed_images=len(records),automatic_retry=False,parameter_updates=0))
        raise


if __name__=='__main__':main()
