"""Finite EEG-only overlap identification, without learning event or image labels."""
import numpy as np


def coordinates(seed=1729, channels=17, selection_samples=64, dimensions=32):
    ids = np.random.default_rng(seed).choice(channels * selection_samples, dimensions, replace=False)
    return ids // selection_samples, ids % selection_samples


def fingerprints(trials, lag, guard=20, selection_samples=64, dimensions=32):
    """Selection uses only the first64 interior temporal differences."""
    assert trials.ndim == 3 and trials.shape[1:] == (17, 250)
    c, t = coordinates(dimensions=dimensions)
    n = len(trials)
    left = np.empty((n, dimensions), np.float32)
    right = np.empty_like(left)
    for lo in range(0, n, 512):
        x = np.asarray(trials[lo:lo + 512], np.float64)
        assert np.isfinite(x).all()
        for output, offset in ((left, lag), (right, 0)):
            z = x[:, c, offset + guard + t + 1] - x[:, c, offset + guard + t]
            norm = np.linalg.norm(z, axis=1, keepdims=True)
            assert np.all(norm > 1e-20), 'Zero-energy EEG fingerprint is not identifiable'
            output[lo:lo + len(x)] = (z / norm).astype(np.float32)
    return left, right


def search(left, right, block=256, check=lambda: None):
    """Exact all-pair top2 in registered fingerprint coordinates, both directions."""
    assert left.shape == right.shape and len(left) >= 3
    n = len(left)
    target = np.empty((n, 2), np.int32)
    score = np.empty((n, 2), np.float32)
    inverse = np.full(n, -1, np.int32)
    inverse_score = np.full(n, -np.inf, np.float32)
    for lo in range(0, n, block):
        hi = min(n, lo + block)
        matrix = left[lo:hi] @ right.T
        matrix[np.arange(hi - lo), np.arange(lo, hi)] = -np.inf
        reverse = matrix.argmax(axis=0)
        reverse_score = matrix[reverse, np.arange(n)]
        improve = reverse_score > inverse_score
        inverse_score[improve] = reverse_score[improve]
        inverse[improve] = reverse[improve] + lo
        best = matrix.argmax(axis=1)
        target[lo:hi, 0] = best
        score[lo:hi, 0] = matrix[np.arange(hi - lo), best]
        matrix[np.arange(hi - lo), best] = -np.inf
        second = matrix.argmax(axis=1)
        target[lo:hi, 1] = second
        score[lo:hi, 1] = matrix[np.arange(hi - lo), second]
        check()
    return target, score, inverse, inverse_score


def verify(trials, lag, targets, guard=20, selection_samples=64, verification_samples=64):
    """Disjoint later64 differences, plus demeaned corresponding65 raw samples."""
    n = len(trials)
    result = np.empty((n, 3), np.float64)
    start = guard + selection_samples
    count = verification_samples
    for lo in range(0, n, 256):
        hi = min(n, lo + 256)
        left = np.asarray(trials[lo:hi, :, lag + start:lag + start + count + 1], np.float64)
        right = np.asarray(trials[targets[lo:hi], :, start:start + count + 1], np.float64)
        a = np.diff(left, axis=-1).reshape(hi - lo, -1)
        b = np.diff(right, axis=-1).reshape(hi - lo, -1)
        an = np.linalg.norm(a, axis=1)
        bn = np.linalg.norm(b, axis=1)
        relative = np.linalg.norm(a - b, axis=1) / np.maximum(np.sqrt((an * an + bn * bn) / 2), 1e-30)
        cosine = (a * b).sum(1) / np.maximum(an * bn, 1e-30)
        a = (left - left.mean(-1, keepdims=True)).reshape(hi - lo, -1)
        b = (right - right.mean(-1, keepdims=True)).reshape(hi - lo, -1)
        raw_relative = np.linalg.norm(a - b, axis=1) / np.maximum(np.sqrt(((a * a).sum(1) + (b * b).sum(1)) / 2), 1e-30)
        result[lo:hi] = np.stack((relative, cosine, raw_relative), axis=-1)
    assert np.isfinite(result).all()
    return result


def accepted(targets, scores, inverse, verified):
    reciprocal = inverse[targets[:, 0]] == np.arange(len(targets))
    keep = (reciprocal & (scores[:, 0] >= .99999)
            & (scores[:, 0] - scores[:, 1] >= .0001)
            & (verified[:, 0] <= .0001) & (verified[:, 2] <= .0001))
    return keep, reciprocal


def graph(nodes, target, keep):
    edges = {int(i): int(target[i]) for i in np.flatnonzero(keep)}
    assert len(set(edges.values())) == len(edges), 'Reciprocal top1 must have unique destinations'
    roots = sorted(set(edges) - set(edges.values()))
    lengths = []
    visited = set()
    for root in roots:
        cursor = root
        length = 1
        while cursor in edges:
            assert cursor not in visited
            visited.add(cursor)
            length += 1
            cursor = edges[cursor]
        lengths.append(length)
    cycles = sorted(set(edges) - visited)
    return dict(accepted_edges=len(edges), nontrivial_path_components=len(lengths),
                maximum_path_nodes=max(lengths, default=0),
                path_node_counts=np.bincount(lengths).tolist(),
                cyclic_source_nodes=len(cycles), isolated_nodes=nodes - len(set(edges) | set(edges.values())))
