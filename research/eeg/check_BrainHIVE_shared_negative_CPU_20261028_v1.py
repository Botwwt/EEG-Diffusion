"""Independent autograd check of the full renormalized fractional-mask effect."""
import argparse
import json
from pathlib import Path
import numpy as np
import brainhive_shared_negative_20261028_v1 as D


def run():
    import torch
    torch.set_num_threads(2)
    rng = np.random.default_rng(1729)
    c = rng.normal(size=(7, 9))
    y = rng.normal(size=(7, 9))
    w = np.zeros((7, 7))
    w[np.arange(7), (np.arange(7) + 1) % 7] = np.linspace(.05, .5, 7)
    metrics, proof = D.batch_gradients(c, y, w, 3.7)
    tc = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    ty = torch.nn.functional.normalize(torch.tensor(y, dtype=torch.float64), dim=-1)
    logits = 3.7 * torch.nn.functional.normalize(tc, dim=-1) @ ty.T
    loss = (-logits.diag() + logits.logsumexp(1)).sum()
    gradient = torch.autograd.grad(loss, tc, retain_graph=True)[0].detach().numpy()
    masked = (-logits.diag() + (logits + torch.log1p(-torch.tensor(w))).logsumexp(1)).sum()
    masked_gradient = torch.autograd.grad(masked, tc)[0].detach().numpy()
    assert np.max(np.abs(gradient - proof['original_gradient'])) < 1e-12
    assert np.max(np.abs(masked_gradient - proof['masked_gradient'])) < 1e-12
    checks = ['ordinary_InfoNCE_gradient_matches_independent_torch_autograd',
              'fractional_mask_including_denominator_matches_independent_autograd']
    _, zero = D.batch_gradients(c, y, np.zeros_like(w), 3.7)
    assert np.array_equal(zero['original_gradient'], zero['masked_gradient'])
    checks.append('zero_graph_exact_original_gradient_fallback')
    size = 30
    source = np.repeat(np.arange(size), 2)
    target = np.column_stack(((np.arange(size) + 1) % size, (np.arange(size) + 5) % size)).reshape(-1)
    weights = np.linspace(.01, .4, len(source))
    original_source, original_target, original_weights = source.copy(), target.copy(), weights.copy()
    changed, note = D.rewire(source, target, weights, size, 1729)
    assert note['changed_destination_edges'] > .5 * len(source)
    assert np.array_equal(source, original_source) and np.array_equal(target, original_target)
    assert np.array_equal(weights, original_weights)
    assert np.array_equal(np.bincount(target, minlength=size), np.bincount(changed, minlength=size))
    checks.append('rewiring_preserves_both_degrees_weights_and_excludes_self_duplicates')
    single, _ = D.rewire(np.array([0]), np.array([1]), np.array([.2]), 3, 0)
    assert np.array_equal(single, [1])
    checks.append('unrewirable_single_edge_is_preserved_and_not_fabricated')
    target = np.array([4, 5, 2, 3, 8, 9, 10, 11, 0, 1, 2, 3])
    source, dest, weights, notes = D.stimulus_graph(target, np.ones(12, bool), np.ones(12))
    found = {(int(a), int(b)): float(z) for a, b, z in zip(source, dest, weights)}
    assert found[(0, 1)] == .5 and not any(a == b for a, b in found)
    checks.append('physical_repeat_graph_collapses_duplicates_and_removes_self_stimulus')
    sm, sp = D.symmetric_batch_gradients(c, y, w, 3.7)
    tc = torch.tensor(c, dtype=torch.float64, requires_grad=True)
    logits = 3.7 * torch.nn.functional.normalize(tc, dim=-1) @ ty.T
    row_loss = (-logits.diag() + logits.logsumexp(1)).sum()
    col_loss = (-logits.diag() + logits.logsumexp(0)).sum()
    sym_gradient = torch.autograd.grad((row_loss + col_loss) / 2, tc, retain_graph=True)[0].detach().numpy()
    masked_logits = logits + torch.log1p(-torch.tensor(w))
    masked_loss = ((-logits.diag() + masked_logits.logsumexp(1)).sum()
                   + (-logits.diag() + masked_logits.logsumexp(0)).sum()) / 2
    masked_sym_gradient = torch.autograd.grad(masked_loss, tc)[0].detach().numpy()
    assert np.max(np.abs(sym_gradient - sp['original_gradients'][:, 2])) < 1e-12
    assert np.max(np.abs(masked_sym_gradient - sp['masked_gradients'][:, 2])) < 1e-12
    assert np.max(np.abs(sm[:, 0] - metrics)) < 1e-12
    checks.append('complete_symmetric_two_direction_gradient_matches_independent_autograd')
    checks.append('column_renormalization_uses_each_visual_query_denominator')
    assert not torch.cuda.is_initialized()
    return dict(status='SHARED_NEGATIVE_GRADIENT_AND_GRAPH_CPU_CHECKS_PASSED', tests=len(checks), checks=checks,
                original_gradient_error=float(np.max(np.abs(gradient - proof['original_gradient']))),
                masked_gradient_error=float(np.max(np.abs(masked_gradient - proof['masked_gradient']))),
                actual_EEG_read=False, GPU_initialized=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run()
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))
