"""Evaluate personal12 foreground edits from reopened pixels, registered object clusters and controls."""

import argparse
import hashlib
import inspect
import io
import json
import zipfile
from pathlib import Path

import numpy as np
from measure_foreground_geometry import MODEL_SHA
from PIL import Image, ImageDraw
from refine_registered_color import metrics, thumbnail
from render_official_words import crop512
from score_reconstruction_jpegs import registered
from score_within_photo_dino import FILES, credit, encoder, reference, sha

DESIGN = (
    Path(__file__).resolve().parents[1]
    / "results/object_fidelity/personal12_pose_evaluation_design.json"
)
LOCK = "a2f0e54309908f404d0b9c1953fa5b087d779a57a2749d61393705595c1e4d5e"
ARMS = ("neutral", "prior", "own", "wrong")
PEOPLE = ("sub-04", "sub-05", "sub-01")
SEED = 260556


def check(test, msg):
    if not test:
        raise ValueError(msg)


def read(path):
    return json.loads(path.read_bytes())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((json.dumps(value, indent=2, allow_nan=False) + "\n").encode())


def source_root(root, sub):
    if sub == "sub-04":
        return root / "trial-object-encoder/personal12/render_inputs"
    return root / "trial-object-replication" / sub / "render_inputs"


def unpack(raw, digest, require_generated=False):
    check(hashlib.sha256(raw).hexdigest() == digest, "Image bytes disagree with the receipt")
    with Image.open(io.BytesIO(raw)) as image:
        if require_generated:
            check(image.format == "JPEG" and image.size == (512, 512), "Unexpected render format")
        return crop512(image.convert("RGB"))


def ort_session(args):
    import onnxruntime as ort

    check(sha(args.mask_model) == MODEL_SHA, "Unexpected U2NetP weights")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_mem_pattern = False
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(args.mask_model), opts, providers=[args.provider])
    check(session.get_providers()[0] == args.provider, "Segmentation provider unavailable")
    return session


def moments(image, session):
    pix = np.array(image.resize((320, 320), Image.Resampling.LANCZOS), dtype=np.float32)
    pix = (pix / max(pix.max(), 1) - np.array([0.485, 0.456, 0.406], np.float32)) / np.array(
        [0.229, 0.224, 0.225], np.float32
    )
    tensor = np.ascontiguousarray(pix.transpose(2, 0, 1)[None])
    mask = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: tensor})[0][
        0, 0
    ].astype(float)
    mask -= mask.min()
    check(mask.max() > 1e-8 and np.isfinite(mask).all(), "Degenerate saliency")
    mask /= mask.sum()
    xy = (np.arange(320) + 0.5) / 320
    cx, cy = mask.sum(0) @ xy, mask.sum(1) @ xy
    dx, dy = xy - cx, xy - cy
    vx, vy, cov = mask.sum(0) @ (dx * dx), mask.sum(1) @ (dy * dy), dy @ mask @ dx
    trace = vx + vy
    check(trace > 0, "No foreground extent")
    return np.array([cx, cy, 0.5 * np.log(trace), (vx - vy) / trace, 2 * cov / trace])


def fingerprints(values):
    h = hashlib.sha256()
    for arr in values:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def extract(args, sub, arm, images, hashes, encode, session):
    target = args.out / "features" / sub / (arm + ".npz")
    # Cache derives exclusively from decoded image bytes and the two frozen extractors.
    identity = [
        MODEL_SHA,
        FILES["model.safetensors"],
        FILES["preprocessor_config.json"],
        args.provider,
        args.device,
        str(args.batch_size),
        hashlib.sha256(
            (
                inspect.getsource(moments)
                + inspect.getsource(encoder)
                + inspect.getsource(thumbnail)
                + inspect.getsource(crop512)
            ).encode()
        ).hexdigest(),
        *hashes,
    ]
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            arrays = [z[k] for k in ("pose", "dino", "rgb")]
            if z["identity"].tolist() == identity and fingerprints(arrays) == str(z["digest"]):
                return arrays
    pose = np.stack([moments(im, session) for im in images])
    dino = encode(images)
    rgb = np.stack([thumbnail(im) for im in images])
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        target,
        pose=pose,
        dino=dino,
        rgb=rgb,
        identity=identity,
        digest=fingerprints([pose, dino, rgb]),
    )
    print(sub, arm, "segmented and encoded", len(images), "images", flush=True)
    return pose, dino, rgb


def setup(args, sub, spec):
    refs = args.data / "object-head-confirmation/evaluation/references" / sub
    reg = registered(refs, "personal12")
    sr = source_root(args.data, sub)
    folder = sr / "renders/personal12" / sub
    fp = folder / "settings.json"
    src = read(fp)
    check(
        reg["sha"] == spec["registration_sha256"][sub]
        and sha(fp) == spec["source_settings_sha256"][sub],
        "Changed reference or source",
    )
    rows, folds = reg["rows"], reg["folds"]
    words = np.array(src["predicted_words"])
    support = reg["calibration"][folds, [reg["vocab"].tolist().index(w) for w in words]]
    pos = {r: i for i, r in enumerate(rows)}
    donor = np.array([pos[r] for r in src["donor_rows"]])
    check(
        np.array_equal(rows, src["rows"])
        and np.array_equal(folds, src["folds"])
        and np.array_equal(support, src["support_rows"])
        and np.array_equal(src["seeds"], rows + 923761)
        and np.array_equal(folds, folds[donor])
        and np.array_equal(words, words[donor]),
        "Registration, fold, seed or donor differs",
    )
    return reg, folder, src, donor


def image_list(folder, arm, hashes):
    check(len(hashes) == 120, "Expected 120 image hashes")
    return [
        unpack((folder / "images" / arm / f"{i:03}.jpg").read_bytes(), d, True)
        for i, d in enumerate(hashes)
    ]


def reference_list(args, table, bank, sub, reg):
    fp = (
        args.data
        / "object-head-confirmation/evaluation/pixels"
        / sub
        / "original_cls/personal12.npz"
    )
    with np.load(fp, allow_pickle=False) as z:
        sig = z["jpeg_sha256"].tolist()
        check(
            str(z["registration_sha"]) == reg["sha"]
            and np.array_equal(z["rows"], reg["rows"])
            and str(z["model_sha"]) == FILES["model.safetensors"],
            "Prior evaluation original cache changed",
        )
    images = []
    for row, label, d in zip(reg["rows"], reg["labels"], sig, strict=True):
        rec = table[row]
        check(
            rec["unique_id"] == label and int(rec["train_row"]) == row,
            "Original row/label mismatch",
        )
        member = f"training_images/{rec['official_folder']}/{rec['stimulus_stem']}.jpg"
        images.append(unpack(bank.read(member), d))
    return images, sig


def edited_settings(args, sub, src):
    folder = args.renders / "personal12" / sub
    fp = folder / "settings.json"
    release = read(args.renders / "blind_images.json")
    cfg = read(fp)
    check(
        sha(fp) == release["cases"][sub]
        and release["generated_jpegs"] == 1440
        and cfg["image_design_sha256"] == release["image_design_sha256"]
        and cfg["packet_sha256"] == sha(args.renders.parent / "personal12" / sub / "pose.npz")
        and cfg["pose_receipt_sha256"]
        == sha(args.renders.parent / "personal12" / sub / "receipt.json")
        and cfg["source_settings_sha256"]
        == sha(source_root(args.data, sub) / "renders/personal12" / sub / "settings.json")
        and cfg["source_packet_sha256"] == src["packet_sha256"]
        and cfg["source_arm"] == "own"
        and cfg["source_jpeg_sha256"] == src["jpeg_sha256"]["own"],
        "Different generated source or unsealed pose edits",
    )
    for k in ("rows", "folds", "predicted_words", "donor_rows", "support_rows", "seeds"):
        check(cfg[k] == src[k], f"Changed source {k}")
    check(
        cfg["subject"] == sub
        and cfg["panel"] == "personal12"
        and set(cfg["jpeg_sha256"]) == set(ARMS),
        "Missing pose edit arm or wrong subject",
    )
    return folder, cfg


def per_query(reg, words, donor, ref, source, edits):
    rows, truth, folds = reg["rows"], reg["labels"], reg["folds"]
    n = len(rows)
    xpose, xref, xrgb = ref
    same_word = truth == words
    group = dict(
        all=np.ones(n, bool),
        correct_word=same_word,
        nonself_same_predicted_word_fold=rows != rows[donor],
        correct_word_nonself_same_true_object=same_word
        & (rows != rows[donor])
        & (truth == truth[donor]),
    )
    target_length = np.linalg.norm(xpose[:, 3:], axis=1)
    group["axis_defined_all_arms"] = (target_length >= 0.2) & np.all(
        [np.linalg.norm(edits[a][0][:, 3:], axis=1) >= 0.2 for a in ARMS], axis=0
    )
    clouds = {}
    pos = {r: i for i, r in enumerate(rows)}
    for f in range(5):
        ids = np.asarray([[pos[r] for r in grp] for grp in reg["calibration"][f]])
        centers = xref[ids].mean(1)
        clouds[f] = centers / np.linalg.norm(centers, axis=1, keepdims=True).clip(1e-12)

    def object_decision(v):
        res = np.empty(n, int)
        for f, proto in clouds.items():
            active = folds == f
            res[active] = np.argmax(v[active] @ proto.T, axis=1)
        return reg["vocab"][res]

    fixed = object_decision(source[1])
    pairs = np.array(
        [np.flatnonzero((folds == f) & (truth == w)) for f in range(5) for w in reg["vocab"]]
    )
    pairs = pairs[np.all(same_word[pairs], axis=1)]
    i, j = pairs.T
    photo = {}
    joint = {}
    for a in ARMS:
        pose, rep, rgb = edits[a]
        c = rep.astype(float) @ xref.astype(float).T
        pred = object_decision(rep)
        scores = {}
        scores["axis_vector_squared_error"] = ((pose[:, 3:] - xpose[:, 3:]) ** 2).sum(1)
        scores["center_distance"] = np.linalg.norm(pose[:, :2] - xpose[:, :2], axis=1)
        scores["absolute_log_radius_error"] = np.abs(pose[:, 2] - xpose[:, 2])
        denom = np.linalg.norm(pose[:, 3:], axis=1) * target_length
        ang = np.sum(pose[:, 3:] * xpose[:, 3:], axis=1) / denom.clip(1e-12)
        scores["axis_degrees"] = np.degrees(np.arccos(ang.clip(-1, 1))) / 2
        measures = np.array([metrics(x, y) for x, y in zip(rgb, xrgb, strict=True)])
        scores["mean_rgb_abs_error"], scores["rgb_rmse32"], scores["luma_pearson32"] = measures.T
        scores["dino_intra10"] = np.array(
            [
                float(v[k] == v[truth == truth[k]].max())
                / np.sum(v[truth == truth[k]] == v[truth == truth[k]].max())
                for k, v in enumerate(c)
            ]
        )
        scores["dino_object12_viewed"] = pred == truth
        scores["dino_object12_predicted_word"] = pred == words
        scores["dino_preserve_source_top1"] = pred == fixed
        scores["dino_source_cosine"] = (source[1].astype(float) * rep).sum(1)
        m1, m2 = c[i, i] - c[i, j], c[j, j] - c[j, i]
        joint[a] = dict(
            dino_strict_assignment=np.array([credit(m) for m in (m1 + m2) / 2]),
            dino_independent_two=np.array(
                [(credit(u) + credit(v)) / 2 for u, v in zip(m1, m2, strict=True)]
            ),
        )
        photo[a] = scores

    def bundle(sets):
        return {k: np.stack([sets[a][k] for a in ARMS], axis=1).astype(float) for k in sets["own"]}

    packed = {}
    photo = bundle(photo)
    for name, mask in group.items():
        scores = {
            k: v[mask]
            for k, v in photo.items()
            if (k == "axis_degrees") == (name == "axis_defined_all_arms")
        }
        packed[name] = (truth[mask], scores)
    packed["strict_pairs_both_correct"] = (truth[i], bundle(joint))
    return packed, dict(
        target_near_round=int((target_length < 0.2).sum()),
        eligible_angles=int(group["axis_defined_all_arms"].sum()),
        strict_pair_rows=rows[pairs].tolist(),
        independent_two_trials_per_pair=2,
        self_donors=int((donor == np.arange(n)).sum()),
        correct_words=int(same_word.sum()),
    )


def table(labels, metrics, universe):
    # Single multiplicity per object, including all people who saw its images.
    universe = np.asarray(sorted(universe))
    weight = np.random.default_rng(SEED).multinomial(
        len(universe), np.ones(len(universe)) / len(universe), size=10000
    )
    n = np.array([sum(labels == c) for c in universe])
    den = weight @ n
    valid = den > 0
    result = dict(n=len(labels), objects=int(sum(n > 0)), metrics={})
    for key, val in metrics.items():
        if not len(val):
            result["metrics"][key] = None
            continue
        by = np.stack([val[labels == c].sum(0) for c in universe])
        boot = (weight[valid] @ by) / den[valid, None]
        means = val.mean(0)
        cell = dict(means=dict(zip(ARMS, means.tolist(), strict=True)), paired={})
        if key in ("full1854", "closed12", "closed12_predicted_word"):
            cell["credit"] = dict(zip(ARMS, val.sum(0).tolist(), strict=True))
        if key.startswith(
            ("dino_intra", "dino_object", "dino_preserve", "dino_strict", "dino_independent")
        ):
            cell["credit"] = dict(zip(ARMS, val.sum(0).tolist(), strict=True))
        for left, right in [
            ("own", "prior"),
            ("own", "wrong"),
            ("own", "neutral"),
            ("prior", "neutral"),
        ]:
            u, v = ARMS.index(left), ARMS.index(right)
            cell["paired"][left + "_minus_" + right] = dict(
                delta=float(means[u] - means[v]),
                ci95=np.quantile(boot[:, u] - boot[:, v], [0.025, 0.975]).tolist()
                if sum(n > 0) > 1
                else None,
            )
        result["metrics"][key] = cell
    return result


def pool(data, subs, universe):
    report = {}
    for name in data[subs[0]]:
        labels = np.concatenate([data[s][name][0] for s in subs])
        merged = {
            k: np.concatenate([data[s][name][1][k] for s in subs]) for k in data[subs[0]][name][1]
        }
        report[name] = table(labels, merged, universe)
    return report


def gallery(folder, reg, words, ordered, pics):
    tile = 120
    cols = ["original", "source", *ARMS]
    out = Image.new("RGB", (2 * (6 * tile + 16), 6 * (tile + 34)), "white")
    draw = ImageDraw.Draw(out)
    for k, (w, row) in enumerate(sorted(ordered.items())):
        i = np.flatnonzero(reg["rows"] == row).item()
        check(row == reg["rows"][reg["labels"] == w].min(), "Changed preregistered gallery row")
        x = (k % 2) * (len(cols) * tile + 16)
        y = (k // 2) * (tile + 34)
        draw.text((x, y), f"{w} | EEG: {words[i]} | row {row}", fill="black")
        for h, c in enumerate(cols):
            out.paste(
                pics[c][i].resize((tile, tile), Image.Resampling.LANCZOS), (x + h * tile, y + 14)
            )
            draw.text((x + h * tile, y + tile + 14), c, fill="black")
    folder.parent.mkdir(exist_ok=True, parents=True)
    out.save(folder, "JPEG", quality=94)
    return sha(folder)


def run(a):
    check(sha(DESIGN) == LOCK, "Evaluation design changed")
    design = read(DESIGN)
    if a.stage == "clip":
        return score_clip(a, design)
    sess = ort_session(a)
    encode = encoder(a)
    table_rows = reference(a)
    groups = {}
    meta = {}
    classes = set()
    with zipfile.ZipFile(a.originals) as bank:
        for sub in PEOPLE:
            reg, srcfolder, src, donor = setup(a, sub, design)
            words = np.array(src["predicted_words"])
            pictures = {}
            pictures["original"], hashes = reference_list(a, table_rows, bank, sub, reg)
            ref = extract(a, sub, "original", pictures["original"], hashes, encode, sess)
            srcsha = src["jpeg_sha256"]["own"]
            pictures["source"] = image_list(srcfolder, "own", srcsha)
            source = extract(a, sub, "source", pictures["source"], srcsha, encode, sess)
            if a.stage == "prepare":
                continue
            folder, cfg = edited_settings(a, sub, src)
            edits = {}
            for arm in ARMS:
                pics = image_list(folder, arm, cfg["jpeg_sha256"][arm])
                pictures[arm] = pics
                edits[arm] = extract(a, sub, arm, pics, cfg["jpeg_sha256"][arm], encode, sess)
            groups[sub], meta[sub] = per_query(reg, words, donor, ref, source, edits)
            classes.update(reg["vocab"])
            gh = gallery(
                a.out / "private_gallery" / (sub + ".jpg"),
                reg,
                words,
                design["gallery_rows"][sub],
                pictures,
            )
            rowfile = a.out / "rows" / (sub + ".npz")
            vals = {}
            for group, (labels, ms) in groups[sub].items():
                vals[group + "__objects"] = labels
                vals.update({group + "__" + k: v for k, v in ms.items()})
            rowfile.parent.mkdir(exist_ok=True, parents=True)
            np.savez(rowfile, rows=reg["rows"], **vals)
            meta[sub].update(
                source_settings_sha256=sha(srcfolder / "settings.json"),
                edited_settings_sha256=sha(folder / "settings.json"),
                original_jpeg_digest=fingerprints([np.asarray(hashes, dtype="S64")]),
                row_metrics_sha256=sha(rowfile),
                private_gallery_sha256=gh,
            )
    if a.stage == "prepare":
        return
    result = dict(
        design_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        mask_model=MODEL_SHA,
        dino_model=FILES["model.safetensors"],
        dino_preprocess=FILES["preprocessor_config.json"],
        arm_order=ARMS,
        source_by_subject=meta,
        pooled=pool(groups, PEOPLE, classes),
        by_subject={s: pool(groups, [s], set(groups[s]["all"][0])) for s in PEOPLE},
    )
    result["primary_gate"] = all(
        result["pooled"][g]["metrics"]["axis_vector_squared_error"]["paired"][k]["ci95"][1] < 0
        for g, k in [
            ("all", "own_minus_prior"),
            ("correct_word_nonself_same_true_object", "own_minus_wrong"),
        ]
    )
    save(a.out / "summary.json", result)
    if a.release:
        save(a.release, result)
    print("primary gate", result["primary_gate"], flush=True)


def score_clip(a, spec):
    """Optional offline naming stage. Requires only frozen edited JPEGs, references and model assets."""
    from eval_instance_starts import load_visual_scorer
    from score_official_objects import load_dictionary, load_tokenizer, text_bank, topk_credit, unit
    from score_reconstruction_jpegs import CLIP_SHA, raw_features

    model, wm = load_visual_scorer(a.archive.resolve())
    check(
        wm == CLIP_SHA
        and sha(a.dictionary) == "1aebc7a2c35edcfcfffc63934754440c90534ccc46df0537db41dc6197450132",
        "Changed L/14 or THINGS label dictionary",
    )
    ids, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    text = text_bank(model, tok, common, a.batch_size)
    groups = {}
    meta = {}
    universe = set()
    seal = read(a.renders / "blind_images.json")
    for s in PEOPLE:
        reg = registered(a.references / s, "personal12")
        folder = a.renders / "personal12" / s
        fp = folder / "settings.json"
        cfg = read(fp)
        words = np.asarray(cfg["predicted_words"])
        wi = [reg["vocab"].tolist().index(v) for v in words]
        check(
            sha(fp) == seal["cases"][s]
            and seal["generated_jpegs"] == 1440
            and cfg["image_design_sha256"] == seal["image_design_sha256"]
            and reg["sha"] == spec["registration_sha256"][s]
            and cfg["source_settings_sha256"] == spec["source_settings_sha256"][s]
            and np.array_equal(cfg["rows"], reg["rows"])
            and np.array_equal(cfg["folds"], reg["folds"])
            and np.array_equal(cfg["support_rows"], reg["calibration"][reg["folds"], wi])
            and set(cfg["jpeg_sha256"]) == set(ARMS),
            "Unknown render source, fold or menu",
        )
        by = {}
        idx = np.arange(120)
        mapped = {v: i for i, v in enumerate(ids)}
        truth = np.array([mapped[w] for w in reg["labels"]])
        predicted = np.array([mapped[w] for w in words])
        subset = np.unique(truth)
        for arm in ARMS:
            pics = image_list(folder, arm, cfg["jpeg_sha256"][arm])
            v = raw_features(model, pics.__getitem__, len(pics), a.batch_size)
            logits = (unit(v) @ text.T)[:, inverse]
            full = topk_credit(logits, 1)
            closed = topk_credit(logits[:, subset], 1)
            by[arm] = dict(
                full1854=full[idx, truth],
                closed12=closed[idx, np.searchsorted(subset, truth)],
                closed12_predicted_word=closed[idx, np.searchsorted(subset, predicted)],
            )
        groups[s] = {
            "all": (
                reg["labels"],
                {k: np.stack([by[arm][k] for arm in ARMS], axis=1) for k in by["own"]},
            )
        }
        meta[s] = dict(
            settings_sha256=sha(fp),
            row_hits={k: vs.tolist() for k, vs in groups[s]["all"][1].items()},
        )
        universe.update(reg["vocab"])
    report = dict(
        design_sha256=LOCK,
        driver_sha256=sha(Path(__file__)),
        arms=ARMS,
        model_sha256=wm,
        dictionary_sha256=sha(a.dictionary),
        pooled=pool(groups, PEOPLE, universe),
        by_subject={s: pool(groups, [s], set(groups[s]["all"][0])) for s in PEOPLE},
    )
    save(a.out / "clip_rows.json", meta)
    save(a.out / "clip_names.json", report)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["prepare", "score", "clip"])
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--renders", type=Path, default=Path("outputs/personal-pose/renders"))
    p.add_argument("--out", type=Path, default=Path("outputs/personal-pose/evaluation"))
    p.add_argument(
        "--references",
        type=Path,
        default=Path("outputs/object-head-confirmation/evaluation/references"),
    )
    p.add_argument(
        "--originals", type=Path, default=Path("outputs/photo-structure/training_images.zip")
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
    )
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--mask-model", type=Path, default=Path("outputs/photo-structure/u2netp.onnx"))
    p.add_argument("--provider", default="DmlExecutionProvider")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--archive", type=Path)
    p.add_argument("--dictionary", type=Path)
    p.add_argument("--release", type=Path)
    a = p.parse_args()
    if a.stage == "clip" and not (a.archive and a.dictionary):
        p.error("clip needs --archive and --dictionary")
    run(a)


if __name__ == "__main__":
    main()
