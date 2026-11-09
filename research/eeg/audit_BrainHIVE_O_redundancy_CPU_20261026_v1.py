"""Read-only comparison of two completed O runs; never opens V score files."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


OLD = Path('/root/brainhive_reproduction_20261020_v1/components/functional_alignment_v2/O')
NEW = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/paired_effect_20261025_v1/arms/O/train')
ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'audit/redundancy_breakout_20261026_v1/O_terminal_state_comparison.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def compare(a, b):
    if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
        same_layout = a.shape == b.shape and a.dtype == b.dtype
        result = dict(equal=bool(same_layout and torch.equal(a, b)), elements=a.numel())
        if same_layout and a.is_floating_point():
            difference = (a.double() - b.double()).abs()
            result.update(max_absolute_difference=float(difference.max()) if a.numel() else 0.,
                          finite=bool(torch.isfinite(a).all() and torch.isfinite(b).all()))
        return result
    if isinstance(a, dict) and isinstance(b, dict):
        children = {str(key): compare(a[key], b[key]) for key in a.keys() & b.keys()}
        missing = dict(only_old=[str(key) for key in a.keys() - b.keys()],
                       only_new=[str(key) for key in b.keys() - a.keys()])
        return dict(equal=not any(missing.values()) and all(row['equal'] for row in children.values()),
                    **missing, children=children)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        children = [compare(x, y) for x, y in zip(a, b)]
        return dict(equal=len(a) == len(b) and all(row['equal'] for row in children), children=children)
    return dict(equal=type(a) is type(b) and a == b, old=a, new=b)


def main():
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    assert not OUT.exists(), 'Preserve the original observation'
    paths = [root / name for root in (OLD, NEW) for name in
             ('component_final.pt', 'resume.pt', 'input_provenance.json', 'training.jsonl')]
    identities = {str(path): dict(bytes=path.stat().st_size, sha256=sha(path)) for path in paths}
    finals = [torch.load(root / 'component_final.pt', map_location='cpu', weights_only=False)
              for root in (OLD, NEW)]
    resumes = [torch.load(root / 'resume.pt', map_location='cpu', weights_only=False)
               for root in (OLD, NEW)]
    provenance = [json.loads((root / 'input_provenance.json').read_text()) for root in (OLD, NEW)]
    shared = ('model', 'optimizer', 'cursor', 'updates', 'seen', 'coverage', 'noise_rng', 'torch_rng', 'cuda_rng')
    state = {key: compare(resumes[0][key], resumes[1][key]) for key in shared}
    final = {key: compare(finals[0][key], finals[1][key]) for key in
             ('model', 'seed', 'updates', 'seen_queries', 'passes', 'calibration')}
    provenance_checks = {key: compare(provenance[0][key], provenance[1][key]) for key in
                         ('orders_sha256', 'input_contract_sha256')}
    logs = [[json.loads(line) for line in (root / 'training.jsonl').read_text().splitlines()]
            for root in (OLD, NEW)]
    log_checks = {}
    for key in ('updates', 'epsilon_loss', 'token_loss', 'last_loss', 'gradient_norm'):
        left, right = [[row[key] for row in log] for log in logs]
        log_checks[key] = dict(events=[len(left), len(right)], exactly_equal=left == right,
                               max_absolute_difference=max((abs(x - y) for x, y in zip(left, right)), default=None))
    report = dict(schema='BrainHIVE_completed_O_redundancy_CPU_v1', UTC=datetime.now(timezone.utc).isoformat(),
                  old_root=str(OLD), new_root=str(NEW), input_files=identities,
                  final=final, terminal_training_state=state, shared_provenance=provenance_checks,
                  logged_current_batch_values=log_checks,
                  parameters=sum(value.numel() for value in finals[0]['model'].values()),
                  all_terminal_states_exactly_equal=all(row['equal'] for row in state.values()),
                  final_heads_exactly_equal=final['model']['equal'],
                  limits=['Final-state equality is not a proof of every intermediate update.',
                          'Non-equality does not establish different scientific hypotheses.',
                          'No V images, features, scores, ranking, GPU, or training were accessed.'],
                  CUDA_initialized=torch.cuda.is_initialized())
    assert all(sha(path) == identities[str(path)]['sha256'] for path in paths)
    assert report['parameters'] == 525568 and not torch.cuda.is_initialized()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(dict(output=str(OUT), final_heads_exactly_equal=report['final_heads_exactly_equal'],
                          terminal_states_exactly_equal=report['all_terminal_states_exactly_equal'],
                          final_max_difference=max(row.get('max_absolute_difference', 0.)
                                                   for row in final['model']['children'].values()),
                          CUDA_initialized=report['CUDA_initialized'])))


if __name__ == '__main__':
    main()
