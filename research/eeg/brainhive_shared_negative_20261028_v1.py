"""Frozen-condition InfoNCE diagnostics; no modified training objective is fitted."""
import numpy as np

METRICS = ('weighted_negative_probability', 'fraction_of_all_negative_probability',
           'shared_target_repulsion_ratio', 'fractional_mask_update_target_alignment_ratio',
           'fractional_mask_gradient_change_norm_ratio')


def stimulus_graph(target, reciprocal, held_cosine, repeats=4):
    n = len(target)
    assert n % repeats == 0 and target.shape == reciprocal.shape == held_cosine.shape
    assert np.all((target >= 0) & (target < n)) and np.isfinite(held_cosine).all()
    assert np.max(np.abs(held_cosine)) <= 1 + 1e-10
    source = np.arange(n) // repeats
    destination = target // repeats
    weight = np.clip(held_cosine, 0, 1) ** 2 * reciprocal / repeats
    keep = (source != destination) & (weight > 0)
    size = n // repeats
    keys, inverse = np.unique(source[keep] * size + destination[keep], return_inverse=True)
    w = np.zeros(len(keys), np.float64)
    np.add.at(w, inverse, weight[keep])
    src, dst = keys // size, keys % size
    assert np.max(np.bincount(src, weights=w, minlength=size), initial=0) <= 1 + 1e-12
    return src, dst, w, dict(physical_candidate_nodes=n, physical_same_stimulus_edges=int((source == destination).sum()),
                             weighted_stimulus_edges=len(src), weight_is_recording_overlap_proxy_not_covariance=True)


def rewire(src, dst, weights, size, seed):
    """Keep every source edge's weight and preserve both integer degree vectors."""
    src = np.asarray(src, np.int64)
    dst = np.asarray(dst, np.int64)
    result = dst.copy()
    m = len(src)
    adjacency = [set() for _ in range(size)]
    for a, b in zip(src, result):
        assert a != b and b not in adjacency[a]
        adjacency[a].add(int(b))
    random = np.random.default_rng(seed)
    accepted = attempts = 0
    target_swaps = 10 * m if m > 1 else 0
    while accepted < target_swaps and attempts < 100 * max(m, 1):
        attempts += 1
        a, b = random.integers(0, m, size=2)
        sa, sb = src[a], src[b]
        da, db = result[a], result[b]
        if a == b or sa == sb or da == db or sa == db or sb == da:
            continue
        if db in adjacency[sa] or da in adjacency[sb]:
            continue
        adjacency[sa].remove(int(da)); adjacency[sb].remove(int(db))
        adjacency[sa].add(int(db)); adjacency[sb].add(int(da))
        result[a], result[b] = db, da
        accepted += 1
    assert np.array_equal(np.bincount(dst, minlength=size), np.bincount(result, minlength=size))
    assert not np.any(src == result)
    assert len(set(zip(src.tolist(), result.tolist()))) == m
    return result, dict(edges=m, accepted_double_edge_swaps=accepted, attempted_double_edge_swaps=attempts,
                        changed_destination_edges=int((dst != result).sum()),
                        changed_destination_weight=float(np.asarray(weights)[dst != result].sum()),
                        total_edge_weight=float(np.asarray(weights).sum()),
                        same_row_weight_multiset_and_in_out_degrees=True,
                        adjacency_operator_eigenvalues_preserved=False)


def batch_gradients(condition, targets, weights, temperature):
    """Return exact gradients w.r.t. input conditions, not encoder parameters."""
    c, y, w = (np.asarray(x, np.float64) for x in (condition, targets, weights))
    assert c.ndim == y.ndim == 2 and c.shape == y.shape
    n = len(c)
    assert w.shape == (n, n) and np.all(w >= 0) and np.all(w <= 1)
    assert np.all(np.diag(w) == 0) and np.max(w.sum(1), initial=0) <= 1 + 1e-12
    radius = np.linalg.norm(c, axis=1, keepdims=True)
    assert np.all(radius > 1e-20) and np.all(np.linalg.norm(y, axis=1) > 1e-20)
    unit = c / radius
    f = y / np.linalg.norm(y, axis=1, keepdims=True)
    logits = float(temperature) * unit @ f.T
    logits -= logits.max(1, keepdims=True)
    probability = np.exp(logits)
    probability /= probability.sum(1, keepdims=True)
    diagonal = np.diag(probability)
    mean_f = probability @ f
    shared = (probability * w) @ f
    h = (probability * w).sum(1)
    assert np.all(h < 1 - 1e-10)
    projection = lambda x: x - unit * (unit * x).sum(1, keepdims=True)
    positive = projection(f)
    all_projection = projection(mean_f)
    shared_projection = projection(shared)
    target_energy = np.square(positive).sum(1)
    original = all_projection - positive
    # g_original-g_masked: this is the change in a negative-gradient step.
    removal = (shared_projection - h[:, None] * all_projection) / (1 - h[:, None])
    interference = (shared_projection * positive).sum(1) / np.maximum(target_energy, 1e-30)
    alignment = (removal * positive).sum(1) / np.maximum(target_energy, 1e-30)
    norm_fraction = np.linalg.norm(removal, axis=1) / np.maximum(np.linalg.norm(original, axis=1), 1e-30)
    metrics = np.column_stack((h, h / np.maximum(1 - diagonal, 1e-30), interference, alignment, norm_fraction))
    factor = float(temperature) / radius
    return metrics, dict(original_gradient=original * factor, shared_gradient=shared_projection * factor,
                         masked_gradient=(original - removal) * factor,
                         correct_probability=diagonal, original_loss=-np.log(np.maximum(diagonal, 1e-300)))


def symmetric_batch_gradients(condition, targets, weights, temperature):
    """Both original InfoNCE directions and their complete mask renormalization."""
    c, y, w = (np.asarray(x, np.float64) for x in (condition, targets, weights))
    n = len(c)
    assert c.shape == y.shape and w.shape == (n, n)
    assert np.all(w >= 0) and np.all(w <= 1) and not np.diag(w).any()
    radius = np.linalg.norm(c, axis=1, keepdims=True)
    unit = c / radius
    f = y / np.linalg.norm(y, axis=1, keepdims=True)
    logits = float(temperature) * unit @ f.T
    row = np.exp(logits - logits.max(1, keepdims=True)); row /= row.sum(1, keepdims=True)
    col = np.exp(logits - logits.max(0, keepdims=True)); col /= col.sum(0, keepdims=True)
    hr, hc = (row * w).sum(1), (col * w).sum(0)
    assert np.all(hr < 1 - 1e-10) and np.all(hc < 1 - 1e-10)
    masked_row = row * (1 - w) / (1 - hr[:, None])
    masked_col = col * (1 - w) / (1 - hc[None, :])
    means = [row @ f, col @ f]
    masked_means = [masked_row @ f, masked_col @ f]
    shared_means = [(row * w) @ f, (col * w) @ f]
    means.append((means[0] + means[1]) / 2)
    masked_means.append((masked_means[0] + masked_means[1]) / 2)
    shared_means.append((shared_means[0] + shared_means[1]) / 2)
    projection = lambda x: x - unit * (unit * x).sum(1, keepdims=True)
    positive = projection(f)
    target_energy = np.square(positive).sum(1)
    row_fraction = hr / np.maximum(1 - np.diag(row), 1e-30)
    col_fraction = hc / np.maximum(1 - np.diag(col), 1e-30)
    masses = [hr, hc, (hr + hc) / 2]
    fractions = [row_fraction, col_fraction, (row_fraction + col_fraction) / 2]
    metrics = np.empty((n, 3, len(METRICS)), np.float64)
    gradients, masked_gradients = [], []
    for objective in range(3):
        original = projection(means[objective]) - positive
        masked = projection(masked_means[objective]) - positive
        shared = projection(shared_means[objective])
        removal = original - masked
        metrics[:, objective] = np.column_stack((masses[objective], fractions[objective],
            (shared * positive).sum(1) / np.maximum(target_energy, 1e-30),
            (removal * positive).sum(1) / np.maximum(target_energy, 1e-30),
            np.linalg.norm(removal, axis=1) / np.maximum(np.linalg.norm(original, axis=1), 1e-30)))
        gradients.append(float(temperature) * original / radius)
        masked_gradients.append(float(temperature) * masked / radius)
    return metrics, dict(original_gradients=np.stack(gradients, axis=1),
                         masked_gradients=np.stack(masked_gradients, axis=1),
                         correct_probability=np.column_stack((np.diag(row), np.diag(col))),
                         original_loss=-np.log(np.maximum(np.column_stack((np.diag(row), np.diag(col))), 1e-300)))
