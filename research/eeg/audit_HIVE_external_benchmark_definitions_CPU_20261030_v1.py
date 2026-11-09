"""Read-only source review; neither reproduction nor a cross-paper ranking.

The downloaded author code is parsed as text/AST, never imported or executed.
Published numbers retain their subject, backbone and sampling uncertainties.
"""
import ast
import datetime
import hashlib
import json
from pathlib import Path

L = Path(__file__).resolve().parent
OUT = L / 'audit/HIVE_external_reconstruction_benchmark_scope_20261030_v1'


def read(path):
    return json.loads(Path(path).read_bytes())


def ref(path):
    path = Path(path)
    data = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def main():
    manifest = read(OUT / 'actual_all_author_text_source_manifest.json')
    root = Path(manifest['root'])
    assert manifest['commit'] == 'cf3fc5bc6e80106f767f902de7f5a2bf78972f60'
    texts = {}
    for entry in manifest['files']:
        data = (root / entry['path']).read_bytes()
        assert len(data) == entry['bytes'] and hashlib.sha256(data).hexdigest() == entry['sha256']
        assert hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest() == entry['git_blob_sha1']
        texts[entry['path']] = data.decode('utf8')
    assert len(texts) == 34
    data_tree = ast.parse(texts['src/cogcappro/data/eeg.py'])
    returns = [n for n in ast.walk(data_tree) if isinstance(n, ast.Return)
               and isinstance(n.value, ast.Tuple)]
    shared_test = [n.lineno for n in returns if [getattr(x, 'id', None) for x in n.value.elts]
                   == ['train_loader', 'test_loader', 'test_loader']]
    assert shared_test == [51]
    batch_tree = ast.parse(texts['src/cogcappro/generate_image/batch_generate.py'])
    generation_steps = [(node.lineno, k.value.value)
                        for node in ast.walk(batch_tree) if isinstance(node, ast.Call)
                        for k in node.keywords if k.arg == 'num_inference_steps' and isinstance(k.value, ast.Constant)]
    assert generation_steps == [(163, 15)]
    simple_tree = ast.parse(texts['src/cogcappro/align/diffusion_pipe.py'])
    simple = next(node for node in simple_tree.body if isinstance(node, ast.ClassDef) and node.name == 'SimpleAlignPipe')
    best_tests = [n.lineno for n in ast.walk(simple) if isinstance(n, ast.Compare)
                  and isinstance(n.left, ast.Name) and n.left.id == 'avg_cos'
                  and len(n.ops) == 1 and isinstance(n.ops[0], ast.Gt)
                  and isinstance(n.comparators[0], ast.Name) and n.comparators[0].id == 'best_cos']
    assert best_tests == [456]
    main_text = texts['src/cogcappro/align/main.py']
    assert 'diffusion_clip_embeddings_test.pt' in main_text
    assert 'source_loader=val_loader' in main_text and 'val_dataloader=diffusion_val_loader' in main_text
    assert 'default="diffusion"' in main_text
    hive = L / 'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py'
    hive_text = hive.read_text(encoding='utf8')
    hive_tree = ast.parse(hive_text)
    clip_models = [(n.lineno, n.args[0].value) for n in ast.walk(hive_tree)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                   and isinstance(n.func.value, ast.Name) and n.func.value.id == 'clip'
                   and n.func.attr == 'load' and n.args and isinstance(n.args[0], ast.Constant)]
    assert clip_models == [(179, 'ViT-L/14')]
    assert 'r = np.corrcoef(reals, preds)' in hive_text and 'success = r < congruents' in hive_text
    execution_path = L / 'audit/HIVE_ordinary_R80_complete_20261029_v1/metadata/full_execution.json'
    execution = read(execution_path)
    def file_refs(value):
        if isinstance(value, dict):
            if all(key in value for key in ('path', 'bytes', 'sha256')):
                yield value
            for item in value.values():
                yield from file_refs(item)
        elif isinstance(value, list):
            for item in value:
                yield from file_refs(item)
    runtime_pin = next(item for item in file_refs(execution)
                       if item['path'].endswith('/brainhive_original_metrics_runtime_v1.py'))
    runtime = L / 'brainhive_original_metrics_runtime_v1.py'
    actual_runtime = ref(runtime)
    assert actual_runtime['sha256'] == runtime_pin['sha256'] and actual_runtime['bytes'] == runtime_pin['bytes']
    runtime_text = runtime.read_text(encoding='utf8')
    assert 'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py' in runtime_text
    assert "name=='ViT-L/14'" in runtime_text
    symbols = ('two_way_identification', 'structural_similarity', 'eval_clip', 'corrcoef')
    symbol_hits = {symbol:[name for name, text in texts.items() if symbol in text] for symbol in symbols}
    means_path = L / 'audit/HIVE_ordinary_R80_complete_20261029_v1/metadata/author_completion.json'
    prior = read(means_path)
    assert prior['queries'] == 2000 and prior['subjects'] == 10
    result = dict(
        status='COMPLETE_PRIMARY_SCOPE_AND_STATIC_SOURCE_REVIEW_NOT_BASELINE_REPRODUCTION',
        UTC=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_manifest=ref(OUT / 'actual_all_author_text_source_manifest.json'),
        verified_source_files=len(texts), verified_source_bytes=sum(e['bytes'] for e in manifest['files']),
        repository_commit=manifest['commit'],
        own_original_evaluator=ref(hive), own_clip_backbone='OpenAI ViT-L/14',
        actual_closed_execution_source_manifest=ref(execution_path),
        actual_closed_execution_runtime_pin=runtime_pin,
        local_runtime_matches_actual_closed_execution=actual_runtime,
        own_clip_score='feature Pearson correlation, generated query against real200 gallery, strict less-than two-way identification',
        own_closed_R80_means_read_only=prior['means'], own_means_source=ref(means_path),
        own_means_not_recomputed_again=True,
        published_CogCapPro=dict(url='https://arxiv.org/html/2603.12722v1',
            table='III', metrics=dict(PixCorr=.163, SSIM=.398, Inception=.779, CLIP=.830, SwAV=.553),
            dataset_scope='THINGS EEG dataset described as ten subjects, R4 training/R80 test',
            row_exact_subject_seed_generation_aggregation='not independently established from TableIII',
            general_implementation_repetitions=5,
            reconstruction_CLIP_evaluator_backbone='not resolved by this paper-method and pinned code review; do not infer from training backbone',
            original_reconstruction_scores_not_reproduced=True),
        published_single_subject_adaptation=dict(url='https://arxiv.org/html/2605.23996v1',
            table='2', subjects=1, test_queries_per_seed=200, reported_seeds=10,
            seed_type='paper calls them independent seeds; not our new training runs',
            denoising_steps=5, CFG=0,
            metric_means=dict(CLIP_L14_two_way_ID=.870, CLIP_H14_two_way_ID=.903, SSIM=.409,
                              PixCorr=.166, Alex2=.818, Alex5=.913, Inception=.831,
                              EffNet_correlation_distance=.794, SwAV_correlation_distance=.489),
            same_subject_as_our_sub01='not independently established',
            not_ten_subject_reconstruction_benchmark=True, not_reproduced=True),
        current_author_source_facts=dict(
            shared_intrasubject_validation_and_test_loader_line=shared_test[0],
            simple_alignment_best_validation_cosine_selection_line=best_tests[0],
            validation_image_embeddings_file='diffusion_clip_embeddings_test.pt',
            default_alignment_branch='diffusion; paper describes STH-Align/simple, explicit configuration needed',
            batch_generation_explicit_steps=generation_steps[0][1],
            no_import_or_runtime_dataset_execution=True,
            no_reconstruction_evaluator_symbol_hits=symbol_hits,
            source_path_test_based_selection='static path requires separate T-internal validation before eligible fitting; no claim about authors actual historical paper runs'),
        conclusions=[
            'Do not subtract CogCapPro CLIP.830 from our L14 score until the exact evaluator is established.',
            'Do not substitute the adaptation H14.903 for L14.870 or for a ten-subject average.',
            'Reported SSIM/semantic strength motivates eligible baselines, not a verified causal gap or SOTA ranking.',
            'Pinned author defaults are not automatically the reported STH-Align and equal-four-step baseline.',
            'The existing five-arm study and its primary/conditional gates remain unchanged.'],
        author_raw_query_scores_available=False, cross_paper_paired_CI_computed=False,
        new_models=0, new_images=0, new_GPU_jobs=0, new_candidate_results_read=False,
        source_only_no_weights_downloaded=True, baseline_fit_or_generation_admitted=False,
        old_ordinary_unchanged=True, NEW_reserved=True, goal_complete=False, SOTA=False)
    path = OUT / 'actual_metric_scope_and_static_source_audit_v2.json'
    assert not path.exists()
    path.write_text(json.dumps(result, indent=2, ensure_ascii=True) + '\n', encoding='utf8')
    print(json.dumps(dict(status=result['status'], verified_sources=len(texts),
                         own_CLIP='L14 correlation two-way ID',
                         author_default_generation_steps=15,
                         static_validation_test_alias_detected=True,
                         baseline_reproduced=False, new_GPU_jobs=0)))


if __name__ == '__main__':
    main()
