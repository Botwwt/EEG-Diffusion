"""Does the mean EEG condition move with the number of averaged repeats, relative to the EEG-image gap?

python shift.py <tag> <sub> [seed ...]   (CPU only; uses the mean vectors saved by train.py)

mu(k)  = mean over images of c/||c|| for inputs that average k repeats
g      = mean(F/||F||) - mu_oof(4): the EEG-to-image offset that a mean-shift calibration removes (train data only)
"""
import sys

import numpy as np

from common import PREP, RUNS

tag, sub = sys.argv[1], int(sys.argv[2])
seeds = [int(s) for s in sys.argv[3:]] or [2025, 2026, 2027]


def unit(v):
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


mu_F = unit(np.load(PREP / "F_train.npy")).mean(0)
mu_Ft = unit(np.load(PREP / "F_test.npy")).mean(0)
for seed in seeds:
    d = RUNS / "enc" / tag / f"sub-{sub:02d}"
    mu_oof = np.mean([np.load(d / f"fold{k}_seed{seed}" / "mu_oof_by_k.npy") for k in range(5)], 0)   # [4, 1024], k = 1..4
    mu_te = np.load(d / f"fold-1_seed{seed}" / "mu_test_by_k.npy")                                     # k = 1,2,4,8,16,40,80
    ks_te = [1, 2, 4, 8, 16, 40, 80]
    g = mu_F - mu_oof[3]
    print(f"== {tag} sub-{sub:02d} seed {seed}")
    print(f"  mean resultant length: F {np.linalg.norm(mu_F):.3f} | EEG cond (held-out concepts, 4 repeats) {np.linalg.norm(mu_oof[3]):.3f} "
          f"| cos(mu_F, mu_c) {cos(mu_F, mu_oof[3]):.3f} | gap ||g|| {np.linalg.norm(g):.3f}")
    print("  held-out train concepts, shift of mu(k) from mu(4), as a fraction of ||g|| (and its cosine with g):")
    for k in (1, 2, 3):
        s = mu_oof[k - 1] - mu_oof[3]
        print(f"     k={k}: {np.linalg.norm(s) / np.linalg.norm(g):.3f} (cos {cos(s, g):+.2f})")
    # per-dimension least squares mu(k) = a + b/k on k = 1..4, extrapolated to 80 repeats
    X = np.stack([np.ones(4), 1 / np.arange(1, 5)], 1)
    coef, *_ = np.linalg.lstsq(X, mu_oof, rcond=None)
    pred80 = coef[0] + coef[1] / 80
    s_pred = pred80 - mu_oof[3]
    # leave-k=3-out check of the 1/k model
    Xl = X[[0, 1, 3]]
    cl, *_ = np.linalg.lstsq(Xl, mu_oof[[0, 1, 3]], rcond=None)
    err3 = np.linalg.norm(cl[0] + cl[1] / 3 - mu_oof[2]) / np.linalg.norm(mu_oof[2] - mu_oof[3])
    print(f"  1/k model: held-out error at k=3 relative to the k=3 vs k=4 distance {err3:.2f}")
    print(f"  extrapolated shift 4 -> 80 repeats: {np.linalg.norm(s_pred) / np.linalg.norm(g):.3f} of ||g|| (cos with g {cos(s_pred, g):+.2f})")
    print("  official test set (full model), measured shift of mu(k) from mu(4), fraction of ||g||:")
    for i, k in enumerate(ks_te):
        if k != 4:
            s = mu_te[i] - mu_te[2]
            print(f"     k={k:<2}: {np.linalg.norm(s) / np.linalg.norm(g):.3f} (cos with g {cos(s, g):+.2f}"
                  + (f", cos with extrapolation {cos(s, s_pred):+.2f})" if k == 80 else ")"))
    g80 = mu_Ft - mu_te[-1]
    g4 = mu_Ft - mu_te[2]
    print(f"  test-set gap to images: at 4 repeats {np.linalg.norm(g4):.3f}, at 80 repeats {np.linalg.norm(g80):.3f}; "
          f"cos(train-estimated g, test gap at 80) {cos(g, g80):+.3f}; "
          f"test gap at 80 left after subtracting g: {np.linalg.norm(g80 - g):.3f}")
