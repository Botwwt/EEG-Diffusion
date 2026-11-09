"""Compare pinned author evaluation input paths without importing either model."""
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/benchmark_primary_scope_review_20261101_v1/actual_HIVE_ENIGMA_static_metric_scope_v2.json'


def ref(path):
    data = path.read_bytes()
    return dict(path=str(path), bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


def assignment(tree, name):
    found = [node for node in ast.walk(tree) if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    assert len(found) == 1, name
    return found[0].value


def function(tree, name):
    found = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == name]
    assert len(found) == 1, name
    return found[0]


def resize_argument(node):
    found = [call for call in ast.walk(node) if isinstance(call, ast.Call)
             and isinstance(call.func, ast.Attribute) and call.func.attr == 'Resize']
    assert len(found) == 1
    return ast.literal_eval(found[0].args[0])


def references(value):
    if isinstance(value, dict):
        if 'path' in value and 'sha256' in value:
            yield value
        for item in value.values():
            yield from references(item)
    elif isinstance(value, list):
        for item in value:
            yield from references(item)


def main():
    hive = ROOT / 'reproduction_sources/BrainHIVE_f06db69f/main/utils_eval.py'
    enigma = ROOT / 'reproduction_sources/ENIGMA_460af4b6/evaluate_recons.py'
    reconstruction = enigma.parent / 'recon_inference.py'
    runtime = ROOT / 'brainhive_original_metrics_runtime_v1.py'
    enigma_manifest_path = enigma.parent / 'source_manifest.json'
    enigma_manifest = json.loads(enigma_manifest_path.read_text(encoding='utf-8'))
    assert enigma_manifest['commit'] == '460af4b6a0ca6eee11aeaf35c5431e8e3d55c11a'
    source_records = {item['path']: item for item in enigma_manifest['files']}
    for path in (enigma, reconstruction):
        expected = source_records[path.name]
        assert ref(path)['bytes'] == expected['bytes'] and ref(path)['sha256'] == expected['sha256']
    registration_path = ROOT / 'configs/BrainHIVE_original_image_metrics_registration_v1.json'
    registration = json.loads(registration_path.read_text(encoding='utf-8'))
    for name, digest in registration['sources'].items():
        assert ref(ROOT / name.replace('\\', '/'))['sha256'] == digest

    old_scope_path = ROOT / 'audit/HIVE_external_reconstruction_benchmark_scope_20261030_v1/actual_metric_scope_and_static_source_audit_v2.json'
    old_scope = json.loads(old_scope_path.read_text(encoding='utf-8'))
    execution_path = ROOT / 'audit/HIVE_ordinary_R80_complete_20261029_v1/metadata/full_execution.json'
    assert ref(execution_path) == old_scope['actual_closed_execution_source_manifest']
    actual_execution = json.loads(execution_path.read_text(encoding='utf-8'))
    pins = [value for value in references(actual_execution)
            if value['path'].replace('\\', '/').endswith('/brainhive_original_metrics_runtime_v1.py')]
    assert pins and all(value['sha256'] == ref(runtime)['sha256'] and value['bytes'] == ref(runtime)['bytes'] for value in pins)

    h = ast.parse(hive.read_text(encoding='utf-8'))
    e = ast.parse(enigma.read_text(encoding='utf-8'))
    r = ast.parse(reconstruction.read_text(encoding='utf-8'))
    runtime_tree = ast.parse(runtime.read_text(encoding='utf-8'))
    model_loads = []
    for tree in (function(h, 'clip_'), e):
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute) and node.func.attr == 'load'
                 and isinstance(node.func.value, ast.Name) and node.func.value.id == 'clip']
        assert len(calls) == 1
        model_loads.append(ast.literal_eval(calls[0].args[0]))
    assert model_loads == ['ViT-L/14', 'ViT-L/14']
    shape_assertions = [ast.unparse(node) for node in ast.walk(function(runtime_tree, 'evaluate'))
                        if isinstance(node, ast.Assert) and 'images.shape' in ast.unparse(node)]
    assert len(shape_assertions) == 1 and 'images.shape[0] == 200' in shape_assertions[0]
    assert 'images.shape[1:] == (3, 512, 512)' in shape_assertions[0]
    hp = assignment(function(h, 'clip_'), 'preprocess')
    ep = assignment(e, 'preprocess_clip')
    same_clip_transform = ast.dump(hp, include_attributes=False) == ast.dump(ep, include_attributes=False)
    assert same_clip_transform and resize_argument(hp) == 224
    low = {name: dict(HIVE=resize_argument(assignment(function(h, name), 'preprocess')),
                      ENIGMA=resize_argument(assignment(e, {'pixcorr': 'preprocess_pixcorr', 'ssim': 'preprocess_ssim'}[name])))
           for name in ('pixcorr', 'ssim')}
    assert all(value == dict(HIVE=425, ENIGMA=512) for value in low.values())
    ground_truth_resize = resize_argument(assignment(e, 'ground_truth_images'))
    assert ground_truth_resize == (224, 224)
    recon_resize = [node for node in ast.walk(r) if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute) and node.func.attr == 'resize'
                    and ast.literal_eval(node.args[0]) == (224, 224)]
    assert len(recon_resize) == 1
    jpeg_calls = [node for node in ast.walk(r) if isinstance(node, ast.Call)
                  and isinstance(node.func, ast.Attribute) and node.func.attr == 'save'
                  and any(k.arg == 'quality' and isinstance(k.value, ast.Constant) and k.value.value == 75 for k in node.keywords)]
    assert len(jpeg_calls) == 2

    result = dict(status='PASS_PINNED_SOURCE_SCOPE_AND_CLOSED_HIVE_RUNTIME_BINDING_ONLY',
        UTC=datetime.now(timezone.utc).isoformat(),
        sources=[ref(path) for path in (hive, enigma, reconstruction, runtime, registration_path, enigma_manifest_path, old_scope_path, execution_path)],
        audit_source=ref(Path(__file__).resolve()),
        clip_transform_AST_equal=same_clip_transform, clip_transform=ast.unparse(hp),
        source_clip_model_names=model_loads, low_level_resize=low,
        ENIGMA_target_pre_resize=ground_truth_resize,
        ENIGMA_generated_JPEG_quality=75, ENIGMA_generated_post_decode_resize=(224, 224),
        HIVE_runtime_input_shape_assertion=shape_assertions[0],
        HIVE_runtime_exact_closed_execution_pin_verified=True,
        ENIGMA_complete_execution_or_paper_version_verified=False,
        old_strict_count_minus_one_finding_not_new=True,
        paper_score_adjusted=False, new_model_forwards=0, research_images_read=0,
        active_candidate_scores_read=0, new_GPU_jobs=0,
        conclusion='Same named CLIP model and same final transform do not establish same pixels, repeat statistic, or counting convention. Low-level resizing also differs. Keep current sealed evaluator and do not subtract published tables as a paired comparison.')
    assert not OUT.exists()
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(status=result['status'], output=str(OUT), low_level_resize=low), ensure_ascii=False))


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    main()
