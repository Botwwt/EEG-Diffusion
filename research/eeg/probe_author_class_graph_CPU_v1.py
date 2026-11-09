"""Execute unchanged author class bodies on CPU without generator dependencies.

This limited check cannot certify the author's full environment or pipeline.
The separate full-module import failure is retained in its original receipt.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--extra-python-path')
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not a.output.exists()
    if a.extra_python_path:
        sys.path.insert(0, a.extra_python_path)
    import torch
    import numpy as np
    from einops.layers.torch import Rearrange
    torch.set_num_threads(2)
    torch.manual_seed(1729)
    start = time.monotonic()
    source = a.root/'ENIGMA_460af4b6/source/models.py'
    manifest = json.loads((source.parent.parent/'source_manifest.json').read_text())
    item = next(x for x in manifest['files'] if x['path'] == 'source/models.py')
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == item['sha256']
    names = {'Spatio_Temporal_CNN', 'ResidualAdd', 'MLP_Projector', 'ENIGMA'}
    nodes = [x for x in ast.parse(source.read_text()).body if isinstance(x, ast.ClassDef) and x.name in names]
    assert {x.name for x in nodes} == names
    ns = dict(torch=torch, nn=torch.nn, np=np, Rearrange=Rearrange, Tensor=torch.Tensor)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), ns)
    rows = []
    for count in [1, 10]:
        subjects = [f'sub-{i:02d}' for i in range(1, count+1)]
        model = ns['ENIGMA'](63, 250, subjects).eval()
        with torch.inference_mode():
            out = model(torch.randn(count, 63, 250), subjects)
        assert list(out.shape) == [count, 1024] and torch.isfinite(out).all()
        rows.append(dict(subjects=count, parameters=sum(p.numel() for p in model.parameters()),
                         input_shape=[count, 63, 250], output_shape=list(out.shape), finite=True))
    assert not torch.cuda.is_initialized()
    receipt = dict(status='PASSED_CPU_UNCHANGED_AUTHOR_CLASS_GRAPH_ONLY',
                   author_commit=manifest['commit'], source_sha256=digest,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   executed_classes=sorted(names), author_class_bodies_modified=False,
                   full_module_import_passed=False, prior_failed_receipt='author_model_CPU_preflight_20261020_v1.json',
                   full_training_or_generation_pipeline_verified=False, weights='random initialization',
                   input='synthetic; not EEG quality evidence', trained=False, images_generated=0,
                   cuda_initialized=False, rows=rows, seconds=time.monotonic()-start)
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
