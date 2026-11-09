"""A mathematical diagnostic of the pinned author's actual 2AFC function."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not a.output.exists()
    import numpy as np
    import torch
    torch.set_num_threads(2)
    start = time.monotonic()
    path = a.source_root/'evaluate_recons.py'
    m = json.loads((a.source_root/'source_manifest.json').read_text())
    source_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    assert source_sha == next(x['sha256'] for x in m['files'] if x['path'] == path.name)
    node = next(x for x in ast.walk(ast.parse(path.read_text())) if isinstance(x, ast.FunctionDef) and x.name == 'two_way_identification')
    ns = dict(np=np, torch=torch, device=torch.device('cpu'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    original = ns['two_way_identification']
    torch.manual_seed(1729)
    real = torch.randn(200, 512)
    tests = []
    for name, pred in [('correct_self_pairing', real), ('fixed_circular_shuffle', real.roll(1, 0))]:
        actual = original(pred, real, lambda x: x, lambda x: x, return_avg=True)
        corr = np.corrcoef(real.numpy().T, pred.numpy().T, rowvar=False)[:200, 200:]
        congruent = np.diag(corr)
        rates = [np.count_nonzero(corr[:, i] < congruent[i]) / 199 for i in range(200)]
        independent = float(np.mean(rates))
        assert abs(actual - (independent - 1/199)) < 1e-12
        tests.append(dict(name=name, author_function=float(actual), independent_strict_comparison=independent,
                          offset=float(actual-independent)))
    assert tests[0]['independent_strict_comparison'] == 1.0
    receipt = dict(status='CONFIRMED_AUTHOR_2AFC_EXTRA_SUBTRACTION', author_commit=m['commit'],
                   source_sha256=source_sha, function='evaluate_recons.py:two_way_identification',
                   source_function_body_modified=False, fixture='200 synthetic feature vectors; identity extractor on CPU',
                   reason='Strict less-than already excludes the diagonal; author code subtracts one again.',
                   bias_per_query=-1/199, tests=tests, historical_scores_modified=False,
                   images_generated=0, trained=False, seconds=time.monotonic()-start,
                   script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    a.output.write_text(json.dumps(receipt, indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
