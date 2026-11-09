"""Finite classical single-term debiasing, not an admitted EEG method.

Scope: frozen full-sequence class likelihoods, global latent class (rho=1),
finite K, EEG softmax logits. No learned emissions, HMM transition gradient,
native VAR execution, optimizer, EEG data, or speedup claim is included.

The single-term telescoping identity is established prior art, e.g.
Shi and Cornish, AISTATS 2021, Eq. (5):
https://proceedings.mlr.press/v130/shi21d.html
Our finite endpoint avoids an infinite-series convergence claim. It does not
ensure useful variance or preserve unbiasedness after nonlinear clipping.
"""
from dataclasses import dataclass
import numpy as np


def logsumexp(values):
    values = np.asarray(values, dtype=np.float64)
    assert values.ndim == 1 and values.size and np.isfinite(values).all()
    maximum = float(values.max())
    return maximum + float(np.log(np.exp(values - maximum).sum()))


@dataclass(frozen=True)
class Hierarchy:
    classes: int
    head: tuple
    tail_sizes: tuple
    level_probabilities: tuple

    def validate(self):
        assert isinstance(self.classes, int) and self.classes >= 2
        assert isinstance(self.head, tuple) and isinstance(self.tail_sizes, tuple)
        assert len(set(self.head)) == len(self.head)
        assert all(isinstance(c, int) and 0 <= c < self.classes for c in self.head)
        tail = self.classes - len(self.head)
        assert len(self.tail_sizes) >= 2 and tail >= 2
        assert all(isinstance(n, int) and n > 0 for n in self.tail_sizes)
        assert all(a < b for a, b in zip(self.tail_sizes, self.tail_sizes[1:]))
        assert self.tail_sizes[-1] == tail, 'Exact full-class endpoint is required.'
        p = np.asarray(self.level_probabilities, dtype=np.float64)
        assert p.shape == (len(self.tail_sizes)-1,)
        assert np.isfinite(p).all() and (p > 0).all(), 'Every correction level needs positive probability.'
        assert abs(float(p.sum())-1.) < 1e-13
        return self

    @property
    def tail(self):
        head = set(self.head)
        return tuple(c for c in range(self.classes) if c not in head)

    @property
    def expected_queried_classes(self):
        self.validate()
        return float(np.dot(self.level_probabilities,
                            np.asarray(self.tail_sizes[1:])+len(self.head)))


def query_hierarchy(logits, hierarchy, permutation, level, query_log_likelihood):
    """Query only the chosen fine prefix; reuse its coarse and base prefixes.

    The caller supplies a fixed image-only head and a uniform permutation of
    the tail, independent of level and current EEG logits. Fresh permutation
    and level per optimizer update are required for conditional unbiasedness.
    Storing deterministic emissions is compatible with this requirement;
    repeatedly optimizing one frozen random bank is a different objective.

    callback(ids) must return the COMPLETE sequence log likelihood for these
    classes. A partial-token likelihood is a different target. This routine
    cannot establish that a future native callback is sparse or economical.
    """
    hierarchy.validate()
    h = np.asarray(logits, dtype=np.float64)
    assert h.shape == (hierarchy.classes,) and np.isfinite(h).all()
    order = tuple(permutation)
    assert len(order) == len(hierarchy.tail) and set(order) == set(hierarchy.tail)
    assert isinstance(level, int) and 1 <= level < len(hierarchy.tail_sizes)
    fine_size = hierarchy.tail_sizes[level]
    queried = tuple(hierarchy.head) + order[:fine_size]
    # No all-class likelihood or all-class likelihood maximum is read here.
    ell = np.asarray(query_log_likelihood(queried), dtype=np.float64)
    assert ell.shape == (len(queried),) and np.isfinite(ell).all()
    logpi = h - logsumexp(h)
    pi = np.exp(logpi)
    head_size = len(hierarchy.head)
    tail_total = len(hierarchy.tail)

    def prefix(size):
        ids = np.asarray(queried[:head_size+size], dtype=np.int64)
        logweight = logpi[ids]+ell[:head_size+size]
        logweight = logweight.copy()
        logweight[head_size:] += np.log(tail_total/size)
        logz = logsumexp(logweight)
        posterior = np.zeros(hierarchy.classes, dtype=np.float64)
        posterior[ids] = np.exp(logweight-logz)
        return dict(negative_log_evidence=-logz,
                    negative_log_evidence_logit_gradient=pi-posterior)

    base = prefix(hierarchy.tail_sizes[0])
    coarse = prefix(hierarchy.tail_sizes[level-1])
    fine = prefix(fine_size)
    probability = hierarchy.level_probabilities[level-1]
    gradient = (base['negative_log_evidence_logit_gradient'] +
                (fine['negative_log_evidence_logit_gradient']-
                 coarse['negative_log_evidence_logit_gradient'])/probability)
    loss = (base['negative_log_evidence'] +
            (fine['negative_log_evidence']-coarse['negative_log_evidence'])/probability)
    return dict(negative_log_evidence=loss,
                negative_log_evidence_logit_gradient=gradient,
                fine=fine, base=base,
                queried_classes=queried, queried_class_count=len(queried),
                level=level, probability=probability,
                is_lower_bound_per_realization=False,
                theoretical_scope='FINITE_GLOBAL_CLASS_FIXED_LOGITS_UNCLIPPED_GRADIENT_ONLY')


def dense_reference(logits, log_likelihood):
    h = np.asarray(logits, dtype=np.float64)
    ell = np.asarray(log_likelihood, dtype=np.float64)
    assert h.ndim == 1 and h.shape == ell.shape and len(h) >= 2
    assert np.isfinite(h).all() and np.isfinite(ell).all()
    pi = np.exp(h-logsumexp(h))
    joint = h-logsumexp(h)+ell
    logz = logsumexp(joint)
    posterior = np.exp(joint-logz)
    return dict(negative_log_evidence=-logz,
                negative_log_evidence_logit_gradient=pi-posterior)


def sample(logits, hierarchy, query_log_likelihood, rng):
    hierarchy.validate()
    assert isinstance(rng, np.random.Generator)
    permutation = tuple(int(c) for c in rng.permutation(hierarchy.tail))
    level = int(rng.choice(np.arange(1, len(hierarchy.tail_sizes)),
                           p=hierarchy.level_probabilities))
    return query_hierarchy(logits, hierarchy, permutation, level,
                           query_log_likelihood)
