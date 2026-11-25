"""Transfer canonical EEG neighbors into native H/14 directions using generic training photographs."""

import argparse
import hashlib
from pathlib import Path

import numpy as np
from calibrate_official_repeat_angles import expmap, tangent
from fit_official_repeat_objects import whitening
from fit_within_photo_embedding import images
from official_object_readout import unit
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from run_prospective_readout import frozen_npz
from scipy.linalg import svd

CODE = Path(__file__).resolve().parent
PUB = CODE.parent / "results/object_fidelity"
DESIGN = PUB / "canonical_neighbor_design.json"
LOCK = "3b05a596410064177ee5b3122b095ba250ab5cf43260e35cf41c67fe2153efd1"
GRID = tuple((k, w) for k in (8, 32, 128) for w in (0.25, 0.5, 0.75))
METRICS = ("top1", "top5", "mrr", "within10", "within_pair", "target_cosine")


def sources(a):
    o = a.data
    root = o / "official-repeat-object"
    source = o / "official-object-readout/sub-03"
    p = dict(
        split=root / "design.json",
        original_selection=root / "selection.json",
        original_model=CODE / "fit_official_repeat_objects.py",
        native_geometry=CODE / "calibrate_official_repeat_angles.py",
        expanded=o / "photo-structure/expanded/features/sub-04.json",
        extra=o / "double-calibration/features/sub-04.json",
        generic_images=source / "assets/text_train.pt",
        row_order=source / "atm_train_rows.csv",
    )
    check(
        sha(DESIGN) == LOCK
        and {k: sha(v) for k, v in p.items()} == read(DESIGN)["historical_source"],
        "Changed canonical protocol or archive",
    )
    lock = read(p["split"])
    held = np.array(lock["held_generic_classes"])
    tr = np.setdiff1d(lock["extended_classes"], held)
    check(
        len(tr) == 884
        and len(held) == 128
        and not set(tr) & set(held)
        and read(p["original_selection"])["choice"] == "joint_32",
        "Historical split or rank changed",
    )
    return p, source, tr, held


def physical(p, ids):
    a, b = read(p["expanded"]), read(p["extra"])
    old = set(a["training_concepts"])
    tensors = []
    for c in ids:
        use = c in old
        f = (p["expanded"] if use else p["extra"]).parent / "sub-04" / f"{int(c):04d}.npz"
        with np.load(f, allow_pickle=False) as z:
            xx = z["features"]
            raw = (a["raw_block_sha256"] if use else b["raw_sha256"])[str(c)]
            expected = str(z["feature_sha"]) if use else b["feature_sha256"][str(c)]
            check(
                int(z["start"]) == c * 10
                and str(z["raw_sha"]) == raw
                and xx.shape == (10, 1003)
                and hashlib.sha256(xx.tobytes()).hexdigest() == expected,
                "Training EEG block differs",
            )
            tensors.append(xx)
    return np.stack(tensors).astype(np.float64)


def photo_bank(source, ids):
    b = images(source)
    return unit(b[(np.asarray(ids)[:, None] * 10 + np.arange(10)).ravel()])


def model(x, y):
    # Keep the historical native ridge and canonical whitening arithmetic, including original float32 image mean.
    x = x.reshape(-1, 1003)
    xm = x.mean(0)
    sd = x.std(0).clip(1e-4)
    x = (x - xm) / sd
    ym = y.mean(0)
    dy = y - ym
    wx, wy = whitening(x, 0.5), whitening(dy, 0.25)
    cross = x.T @ dy / len(x)
    u, val, vh = svd(wx @ cross @ wy, full_matrices=False, check_finite=False)
    mag = np.sqrt(val[:32].clip(0))
    py = (wy @ vh[:32].T) * mag
    return dict(
        xm=xm,
        sd=sd,
        ym=ym,
        wx=wx,
        cross=cross,
        px=(wx @ u[:, :32]) * mag,
        py=py,
        gallery=unit(dy @ py),
        axis=ym.astype(float) / np.linalg.norm(ym.astype(float)),
    )


def native(m, q):
    q = (q.reshape(-1, 1003) - m["xm"]) / m["sd"]
    ridge = unit(m["ym"] + q @ m["wx"] @ m["wx"] @ m["cross"])
    return ridge, unit(q @ m["px"])


def nhood(m, query, real, ks, perm=None):
    out = {k: [] for k in ks}
    for first in range(0, len(query), 192):
        s = query[first : first + 192] @ m["gallery"].T
        # Stable full sort supplies deterministic ordering when neighbors tie.
        ix = np.argsort(-s, axis=1, kind="stable")[:, : max(ks)]
        scores = np.take_along_axis(s, ix, axis=1)
        for k in ks:
            sub = scores[:, :k] / 0.08
            weights = np.exp(sub - sub[:, [0]])
            weights /= weights.sum(1, keepdims=True)
            rows = ix[:, :k] if perm is None else perm[ix[:, :k]]
            center = np.einsum("nk,nkd->nd", weights, real[rows])
            out[k].append(tangent(center, m["axis"]))
    return {k: np.vstack(v) for k, v in out.items()}


def direction(v, backup):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return np.where(norm > 1e-10, v / norm.clip(1e-10), backup)


def angle_set(t, other, w, axis, gain, cap):
    radius = np.linalg.norm(t, axis=1, keepdims=True)
    d = t / radius.clip(1e-12)
    v = direction((1 - w) * d + w * direction(other, d), d)
    return expmap(v * radius * gain, axis, cap)


def metrics(pred, y, within=True):
    score = np.asarray(pred, dtype=np.float32) @ y.astype(np.float32).T
    row = np.arange(len(y))
    v = score[row, row]
    rank = (
        1
        + (score > v[:, None]).sum(1)
        + ((score == v[:, None]) & (row[None, :] < row[:, None])).sum(1)
    )
    out = dict(
        top1=(rank == 1).astype(float),
        top5=(rank <= 5).astype(float),
        mrr=1 / rank,
        target_cosine=np.einsum("ij,ij->i", pred.astype(float), y.astype(float)),
    )
    if within:
        group = row[:, None] // 10 * 10 + np.arange(10)
        s = np.take_along_axis(score, group, axis=1)
        truth = s[row, row % 10]
        out["within10"] = (truth == s.max(1)) / (s == s.max(1, keepdims=True)).sum(1)
        out["within_pair"] = (
            (truth[:, None] > s).sum(1) + 0.5 * (truth[:, None] == s).sum(1) - 0.5
        ) / 9
    return out


def summary(rows):
    return {
        k: (float(v.sum()) if k in ("top1", "top5", "within10") else float(v.mean()))
        for k, v in rows.items()
    }


def fit(a):
    p, source, tr, _ = sources(a)
    x = physical(p, tr)
    y = photo_bank(source, tr).reshape(884, 10, 1024)
    folds = np.array_split(np.random.default_rng(260571).permutation(884), 4)
    caches = []
    true_angles = []
    native_angles = []
    for val in folds:
        fi = np.setdiff1d(np.arange(len(tr)), val)
        yf = y[fi].reshape(-1, 1024)
        yv = y[val].reshape(-1, 1024)
        m = model(x[fi], yf)
        raw, query = native(m, x[val])
        axis = m["axis"]
        t = tangent(raw, axis)
        near = nhood(m, query, yf, (8, 32, 128))
        caches.append((yv, axis, t, near))
        true_angles.extend(np.linalg.norm(tangent(yv, axis), axis=1))
        native_angles.extend(np.linalg.norm(t, axis=1))
    gain = float(np.median(true_angles) / np.median(native_angles))
    cap = float(np.quantile(true_angles, 0.95))
    all_metrics = {k: [] for k in ((0, 0), *GRID)}
    for yv, axis, t, near in caches:
        for k, w in all_metrics:
            pred = angle_set(t, near[k] if k else t, w, axis, gain, cap)
            all_metrics[(k, w)].append(metrics(pred, yv))
    stats = {
        key: summary({k: np.concatenate([z[k] for z in vv]) for k in METRICS})
        for key, vv in all_metrics.items()
    }
    base = stats[(0, 0)]
    eligible = [
        key
        for key in GRID
        if stats[key]["within10"] >= base["within10"]
        and stats[key]["target_cosine"] >= base["target_cosine"]
    ]
    selected = max(
        eligible or GRID,
        key=lambda pair: (
            stats[pair]["mrr"],
            stats[pair]["within10"],
            stats[pair]["target_cosine"],
            -pair[0],
            -pair[1],
        ),
    )
    receipt = dict(
        design_sha256=LOCK,
        source_sha256=sha(Path(__file__)),
        training_classes=tr.tolist(),
        folds=[tr[v].tolist() for v in folds],
        gain=gain,
        cap=cap,
        k=selected[0],
        mix=selected[1],
        eligible_candidates=len(eligible),
        oof={f"k{k}_w{w}": v for (k, w), v in stats.items()},
    )
    freeze(a.root / "selection.json", receipt)
    print("OOF", selected, "eligible", len(eligible), "angle", gain, cap, flush=True)


def calculate(m, q, y, sel, donor, perm):
    raw, query = native(m, q)
    near = nhood(m, query, y, (1, sel["k"]))
    shuffled = nhood(m, query, y, (sel["k"],), perm=perm)
    loc = near[sel["k"]]
    t = tangent(raw, m["axis"])

    def fuse(other):
        return angle_set(t, other, sel["mix"], m["axis"], sel["gain"], sel["cap"])

    global_angle = angle_set(t, t, 0, m["axis"], sel["gain"], sel["cap"])
    arms = dict(
        native=raw.astype(np.float32),
        global_angle=global_angle,
        canonical_neighbor=fuse(loc),
        other_local=fuse(loc[donor]),
        single_neighbor=fuse(near[1]),
        shuffled_bank=fuse(shuffled[sel["k"]]),
        train_mean=np.broadcast_to(m["axis"].astype(np.float32), raw.shape).copy(),
        mean_fused=fuse(np.zeros_like(t)),
    )
    check(
        np.allclose(arms["global_angle"], arms["mean_fused"], atol=1e-7, rtol=0),
        "Mean prior must be neutral",
    )
    for key in (
        "canonical_neighbor",
        "other_local",
        "single_neighbor",
        "shuffled_bank",
        "mean_fused",
    ):
        check(
            np.max(
                abs(arms[key].astype(float) @ m["axis"] - global_angle.astype(float) @ m["axis"])
            )
            < 2e-6,
            "Recipient angles differ",
        )
    return arms, query


def predict(a):
    p, source, tr, held = sources(a)
    sel = read(a.root / "selection.json")
    check(
        sel["design_sha256"] == LOCK
        and sel["source_sha256"] == sha(Path(__file__))
        and sel["training_classes"] == tr.tolist(),
        "Wrong fit",
    )
    x = physical(p, tr)
    y = photo_bank(source, tr)
    m = model(x, y)
    q = physical(p, held)
    rows = (held[:, None] * 10 + np.arange(10)).ravel()
    # Evaluation-only counterfactual swaps two photos in the same ten-image object; no image data required.
    donor = np.arange(len(rows)).reshape(-1, 10)[:, np.roll(np.arange(10), 1)].ravel()
    perm = np.random.default_rng(260572).permutation(len(y))
    arms, cq = calculate(m, q, y, sel, donor, perm)
    f = a.root / "generic.npz"
    frozen_npz(
        f,
        arms
        | dict(
            rows=rows,
            donor=donor,
            canonical_query=cq,
            py=m["py"],
            mean=m["ym"],
            train_classes=tr,
            selection_sha256=sha(a.root / "selection.json"),
        ),
    )
    freeze(
        a.root / "prediction.json",
        dict(
            design_sha256=LOCK,
            selection_sha256=sha(a.root / "selection.json"),
            packet_sha256=sha(f),
            n=1280,
        ),
    )
    print("Generic directions frozen", sha(f), flush=True)


def evaluate(a):
    _, source, tr, held = sources(a)
    path = a.root / "generic.npz"
    rec = read(a.root / "prediction.json")
    selection = a.root / "selection.json"
    sel = read(selection)
    check(
        rec["packet_sha256"] == sha(path) and rec["selection_sha256"] == sha(selection),
        "Predictions changed",
    )
    y = photo_bank(source, held)
    with np.load(path, allow_pickle=False) as z:
        check(
            np.array_equal(z["rows"], (held[:, None] * 10 + np.arange(10)).ravel())
            and np.array_equal(z["train_classes"], tr),
            "Gallery identity changed",
        )
        data = {
            k: metrics(z[k], y)
            for k in (
                "native",
                "global_angle",
                "canonical_neighbor",
                "other_local",
                "single_neighbor",
                "shuffled_bank",
                "train_mean",
                "mean_fused",
            )
        }
        # Native and canonical retrieval are separate feature spaces.
        cy = unit((y - z["mean"]) @ z["py"])
        can = metrics(z["canonical_query"], cy)["top1"].sum()
    historical = read(a.data / "official-repeat-object/native_generic_validation.json")
    check(
        abs(data["native"]["target_cosine"].mean() - historical["actual_photo_cosine"]) < 2e-6
        and data["native"]["within10"].sum()
        == historical["actual_photo_top1_within_10_per_held_generic_class"],
        "Native generic reference failed",
    )
    draws = np.random.default_rng(260573).integers(128, size=(10000, 128))

    def contrast(v, b):
        d = (v - b).reshape(128, 10).mean(1)
        return dict(
            delta=float(d.mean()), ci95=np.quantile(d[draws].mean(1), [0.025, 0.975]).tolist()
        )

    comparisons = {
        arm: {k: contrast(data["canonical_neighbor"][k], data[arm][k]) for k in METRICS}
        for arm in ("global_angle", "other_local", "single_neighbor", "shuffled_bank")
    }
    g = comparisons["global_angle"]
    gate = (
        g["top1"]["delta"] > 0
        and g["mrr"]["ci95"][0] > 0
        and g["target_cosine"]["delta"] >= 0
        and g["within10"]["delta"] >= 0
        and comparisons["other_local"]["within_pair"]["delta"] > 0
    )
    result = dict(
        design_sha256=LOCK,
        source_sha256=sha(Path(__file__)),
        selection_sha256=sha(selection),
        packet_sha256=sha(path),
        subject="sub-04",
        generic_repetitions_per_photo=4,
        n_train_objects=len(tr),
        n_held_objects=len(held),
        train_gallery_photos=len(tr) * 10,
        evaluation_photos=len(y),
        oof=dict(
            k=sel["k"],
            mix=sel["mix"],
            temperature=0.08,
            gain=sel["gain"],
            cap_radians=sel["cap"],
            eligible_settings=sel["eligible_candidates"],
        ),
        generic_canonical_space_top1=int(can),
        metrics={k: summary(v) for k, v in data.items()},
        candidate_minus=comparisons,
        advance=bool(gate),
    )
    freeze(a.release, result)
    print(result["metrics"], "advance", gate, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=("fit", "predict", "evaluate"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--root", type=Path, default=Path("outputs/canonical-neighbor"))
    p.add_argument("--release", type=Path, default=PUB / "canonical_neighbor.json")
    a = p.parse_args()
    dict(fit=fit, predict=predict, evaluate=evaluate)[a.stage](a)
