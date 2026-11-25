"""Evaluate fixed background switches using both actual image outcomes for every sub-04 query."""

import argparse
import csv
import hashlib
import io
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from measure_background_attribute import PUBLIC, attributes, check, digest, mask_input
from measure_foreground_geometry import MODEL_SHA, ZIP_SHA
from PIL import Image, ImageDraw
from refine_registered_color import thumbnail
from register_fresh_photo_panel import freeze, read, sha
from render_background_prompt import FLAGS, LOCK, PANELS, SUB, controls
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA
from run_prospective_readout import frozen_npz
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, encoder

METRICS = ("background_mse", "full1854", "closed12", "dino_ten", "rgb_rmse32")
ARMS = (*FLAGS, "texture_only", "background_oracle")
CONTROLS = ("original", "prior", "wrong")


class BackgroundMasks:
    def __init__(self, model, provider):
        import onnxruntime as ort

        check(digest(model) == MODEL_SHA, "Wrong saliency weights")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        opts.inter_op_num_threads = 1
        opts.enable_mem_pattern = False
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(model), sess_options=opts, providers=[provider])
        check(self.session.get_providers()[0] == provider, "Wrong saliency backend")
        self.inp = self.session.get_inputs()[0].name
        self.pred = self.session.get_outputs()[0].name
        self.cache = {}

    def get(self, image):
        key = hashlib.sha256(image.tobytes()).hexdigest()
        if key not in self.cache:
            raw = self.session.run([self.pred], {self.inp: mask_input(image)})[0][0, 0]
            pixels = np.asarray(image.resize((320, 320), Image.Resampling.LANCZOS))
            label, mom, _ = attributes(pixels, raw)
            self.cache[key] = (
                label,
                np.asarray(mom),
                hashlib.sha256(np.ascontiguousarray(raw).tobytes()).hexdigest(),
            )
        return self.cache[key]


def dictionary_hash(file, words, suffix):
    check(sha(file) == DICTIONARY, "Different original text dictionary")
    with file.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        cols = reader.fieldnames
        records = list(reader)
    result = io.StringIO(newline="")
    writer = csv.DictWriter(result, fieldnames=cols, lineterminator="\n")
    writer.writeheader()
    for r in records:
        if r["unique_id"] in words:
            r["word"] += suffix
        writer.writerow(r)
    return hashlib.sha256(result.getvalue().encode("utf8")).hexdigest()


def naming(logits, names, ref):
    check(
        logits.shape == (120, 1854)
        and len(names) == len(set(names)) == 1854
        and np.isfinite(logits).all(),
        "Malformed CLIP vocabulary or logits",
    )
    select = np.array([names.index(w) for w in ref["vocab"]])
    q = np.arange(120)
    target = ref["truth"]
    return np.column_stack(
        (topk_credit(logits, 1)[q, select[target]], topk_credit(logits[:, select], 1)[q, target])
    )


def dino_ten(v, r, truth):
    result = np.empty(len(v))
    for g in np.unique(truth):
        ix = np.flatnonzero(truth == g)
        check(len(ix) == 10, "Incomplete ten-photo gallery")
        result[ix] = np.diag(topk_credit(v[ix] @ r[ix].T, 1))
    return result


def gallery(path, panel, images, ref, words, flags, private):
    rows = ref["rows"]
    truth = ref["truth"]
    samples = [
        (np.flatnonzero(truth == j)[np.argmin(rows[truth == j])], "class first") for j in range(12)
    ]
    for state, group in ((True, "first ON"), (False, "first OFF")):
        subset = np.flatnonzero(flags == state)
        if len(subset):
            samples.append((subset[np.argmin(rows[subset])], f"{group} ({len(subset)}/120)"))
    labels = (["reference"] if private else []) + ["native", "textured option", "matched"]
    w, h = 128, 128
    cell = w * len(labels) + 12
    sheet = Image.new("RGB", (2 * cell, ((len(samples) + 1) // 2) * (h + 56) + 30), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (5, 5),
        f"{panel}, sub-04 | first per object, then first switch states in source row order",
        fill="black",
    )
    for n, (i, note) in enumerate(samples):
        x = n % 2 * cell + 3
        y = 30 + n // 2 * (h + 56)
        draw.text((x, y), f"{rows[i]} {ref['labels'][i]} | EEG: {words[i]} | {note}", fill="black")
        group = ([images[0][i]] if private else []) + [
            images[1][i],
            images[2][i],
            images[2 if flags[i] else 1][i],
        ]
        for j, (lab, img) in enumerate(zip(labels, group, strict=True)):
            sheet.paste(img.resize((w, h), Image.Resampling.LANCZOS), (x + j * w, y + 15))
            draw.text((x + j * w, y + h + 18), lab, fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path, "JPEG", quality=90, subsampling=0)
    return [int(rows[i]) for i, _ in samples]


def case(a, p, refs, enc, masks, archive, manifest, oldscored, worddesign):
    protocol, worker, data, flags = controls(a.root, p)
    rows, folds, words, donors, support, _, radius, packet = data
    ref = registered(a.fresh / "references", p)
    rr = ref["rows"]
    truth = ref["truth"]
    meta = worddesign["panels"][p]
    check(
        ref["sha"] == meta["registration_sha256"]
        and np.array_equal(rr, rows)
        and np.array_equal(folds, ref["folds"])
        and np.array_equal(np.array(meta["object_ids"])[truth], rr // 10),
        "Wrong registered image evaluation",
    )
    ix = np.array([ref["vocab"].tolist().index(w) for w in words])
    check(
        np.array_equal(ref["calibration"][folds, ix], support)
        and all(
            manifest[int(row)]["unique_id"] == label
            for row, label in zip(rows, ref["labels"], strict=True)
        ),
        "Changed supports or original photo identities",
    )
    native = a.fresh / "render_inputs/learned/renders" / p / SUB
    redo = a.root / "native_audit/renders" / p / SUB
    tex = a.root / "texture/renders" / p / SUB
    old = oldscored["panels"][p]
    cfg = [read(folder / "settings.json") for folder in (native, redo, tex)]
    check(
        sha(native / "settings.json")
        == old["settings_sha256"]["learned"]
        == refs["panels"][p]["source_settings_sha256"]
        and packet == refs["panels"][p]["source_h14_sha256"] == old["condition_sha256"]["learned"]
        and sha(a.root / "flags" / (p + ".npz")) == refs["panels"][p]["sha256"],
        "Different historical source or trigger file",
    )
    expected_dict = dictionary_hash(a.dictionary, set(words), protocol["suffix"])
    for j, (c, n) in enumerate(zip(cfg, (120, 8, 120), strict=True)):
        check(
            c["subject"] == SUB
            and c["panel"] == p
            and c["packet_sha256"] == packet
            and c["dictionary_sha256"] == (expected_dict if j == 2 else DICTIONARY)
            and c["adapter_sha256"] == ADAPTER_SHA
            and c["turbo_sha256"] == TURBO
            and c["batch_size"] == 8
            and c["radius"] == radius
            and c["driver_sha256"] == sha(Path(__file__).with_name("render_within_photo_ip.py"))
            and c["design_sha256"] == sha(a.root / "input/design.json")
            and c["radius_sha256"] == sha(a.root / "input/radius.json"),
            "Potential generation source, dictionary, or assets changed",
        )
        for name, expected in (
            ("rows", rows),
            ("folds", folds),
            ("support_rows", support),
            ("predicted_words", words),
            ("donor_rows", donors),
            ("seeds", 923761 + rows),
        ):
            check(np.array_equal(c[name], expected[:n]), "Potential changed " + name)
        check(
            set(c["jpeg_sha256"]) == {"own"} and len(c["jpeg_sha256"]["own"]) == n,
            "Unexpected JPEG list",
        )
    hashes = [c["jpeg_sha256"]["own"] for c in cfg]
    check(
        hashes[0] == worker["baseline_jpeg_sha256"][p]
        and hashes[1] == hashes[0][:8]
        and all(sha(redo / "images/own" / f"{j:03d}.jpg") == hashes[0][j] for j in range(8)),
        "Baseline rerender failed",
    )
    images = [
        [original(archive, manifest, int(row)) for row in rows],
        pictures(native / "images/own", hashes[0]),
        pictures(tex / "images/own", hashes[2]),
    ]
    oldlog = a.fresh / "render_inputs/scores" / p / "clip_logits.npz"
    newlog = tex / "naming.npz"
    check(sha(oldlog) == old["clip_logits_sha256"], "Earlier naming logits changed")
    with (
        np.load(oldlog, allow_pickle=False) as historical,
        np.load(newlog, allow_pickle=False) as textured,
    ):
        check(
            str(historical["learned_settings_sha256"]) == sha(native / "settings.json")
            and str(textured["settings_sha256"]) == sha(tex / "settings.json")
            and str(historical["model_sha256"]) == str(textured["model_sha256"]) == CLIP_SHA
            and str(historical["dictionary_sha256"]) == DICTIONARY
            and np.array_equal(historical["names"], textured["names"])
            and np.array_equal(historical["rows"], rows)
            and np.array_equal(textured["rows"], rows),
            "Independent CLIP features cannot be paired",
        )
        names = historical["names"].tolist()
        naming_legacy = naming(historical["learned"], names, ref)
        naming_texture = naming(textured["scores"], names, ref)
    reference = enc(images[0]).astype(float)
    dino = [dino_ten(enc(ims).astype(float), reference, truth) for ims in images[1:]]
    prior_rows = a.fresh / "evaluation" / p / "per_row.npz"
    check(sha(prior_rows) == old["row_scores_sha256"], "Historical DINO score file changed")
    with np.load(prior_rows, allow_pickle=False) as prev:
        check(
            np.array_equal(prev["rows"], rows)
            and np.array_equal(
                prev["scores"][:, prev["methods"].tolist().index("learned")],
                np.column_stack((naming_legacy, dino[0])),
            ),
            "Historical DINO or independent CLIP row credit failed parity",
        )
    measurement = [[masks.get(image) for image in ims] for ims in images]
    code = np.array([[v[0] for v in measured] for measured in measurement]).T
    original_code = code[:, 0]
    # Recheck the independently remeasured reference labels against the earlier legal support archive where present.
    known = {}
    for ob in np.unique(rr // 10):
        source = a.data / "eeg-background-confirmation/annotations/supports" / f"{ob:04d}.npz"
        if source.exists():
            with np.load(source, allow_pickle=False) as z:
                known.update(zip(z["rows"], z["labels"], strict=True))
    check(
        all(original_code[i] == known[row] for i, row in enumerate(rr) if row in known),
        "Reference annotation parity failed",
    )
    thumb = [np.stack([thumbnail(v) for v in ims]).astype(float) for ims in images]
    scores = np.stack(
        [
            np.column_stack(
                (
                    (code[:, j + 1] - original_code) ** 2,
                    nm,
                    dino[j],
                    np.sqrt(((thumb[j + 1] - thumb[0]) ** 2).mean(axis=(1, 2, 3))),
                )
            )
            for j, nm in enumerate((naming_legacy, naming_texture))
        ],
        axis=1,
    )
    masks_used = np.column_stack(
        [flags[k] for k in FLAGS] + [np.ones(120, bool), scores[:, 1, 0] < scores[:, 0, 0]]
    )
    per = scores[np.arange(120)[:, None], masks_used.astype(int)]
    h = np.array([hashes[0], hashes[2]]).T
    selected = h[np.arange(120)[:, None], masks_used.astype(int)]
    check(
        np.isfinite(per).all()
        and all(
            np.all(
                selected[:, i][masks_used[:, i] == masks_used[:, j]]
                == selected[:, j][masks_used[:, i] == masks_used[:, j]]
            )
            for i in range(len(ARMS))
            for j in range(i)
        ),
        "Different pixels selected for identical action",
    )
    output = a.out / p
    record = output / "per_row.npz"
    frozen_npz(
        record,
        dict(
            rows=rows,
            folds=folds,
            truth=ref["labels"],
            predicted_words=words,
            objects=rr // 10,
            strict=np.isin(rr // 10, meta["primary_object_ids"]),
            donor_rows=donors,
            arms=ARMS,
            metrics=METRICS,
            background_label_order=("reference", "original", "texture"),
            background_labels=code,
            background_measurement=np.array(
                [[v[1] for v in part] for part in measurement]
            ).transpose(1, 0, 2),
            mask_sha256=np.array([[v[2] for v in part] for part in measurement]).T,
            selected_sha256=selected,
            potential_sha256=h,
            texture_selected=masks_used,
            scores=per,
        ),
    )
    indices = gallery(output / "private.jpg", p, images, ref, words, flags["matched"], True)
    g = a.gallery / ("background_prompt_" + p + ".jpg")
    gallery(g, p, images, ref, words, flags["matched"], False)
    return dict(
        row_file=record,
        panel=p,
        settings_sha256=[sha(folder / "settings.json") for folder in (native, redo, tex)],
        flags_sha256=worker["flags"][p],
        texture_naming_sha256=sha(newlog),
        legacy_naming_sha256=sha(oldlog),
        reference_label_replays=sum(row in known for row in rr),
        unchanged_baseline_redraws=8,
        previous_dino_clip_row_replays=120,
        per_row_sha256=sha(record),
        gallery_rows=indices,
        gallery_sha256=sha(g),
    )


def aggregate(cases, seed):
    keys = (
        "scores",
        "rows",
        "objects",
        "strict",
        "donor_rows",
        "background_labels",
        "texture_selected",
        "truth",
        "predicted_words",
    )
    arrays = {k: [] for k in keys}
    for c in cases:
        with np.load(c["row_file"], allow_pickle=False) as z:
            for k in keys:
                arrays[k].append(z[k])
    a = {k: np.concatenate(parts) for k, parts in arrays.items()}
    rows = a["rows"]
    objs = a["objects"]
    panels = np.repeat(PANELS, 120)
    check(
        len(set(rows)) == 480 and len(set(objs)) == 48 and a["strict"].sum() == 350,
        "Wrong image population",
    )
    flags = a["texture_selected"]
    values = a["scores"]
    codes = a["background_labels"]
    target = codes[:, 0]
    idx = {k: i for i, k in enumerate(ARMS)}
    own = idx["matched"]

    def report(mask, pop, comparators):
        n = int(mask.sum())
        result = dict(n=n, objects=int(len(np.unique(objs[mask]))))
        if not n:
            return result
        means = values[mask].mean(axis=0)
        result["means"] = {
            arm: dict(zip(METRICS, means[j].tolist(), strict=True)) for j, arm in enumerate(ARMS)
        }
        result["credits"] = {
            arm: dict(zip(METRICS[1:4], values[mask, j, 1:4].sum(0).tolist(), strict=True))
            for j, arm in enumerate(ARMS)
        }
        # Draw objects once per menu, irrespective of arm. Preserve ten paired photographs together.
        rng = np.random.default_rng(seed)
        den = np.zeros(10000)
        boots = np.zeros((10000, len(comparators), len(METRICS)))
        for panel in PANELS:
            labels = np.unique(objs[pop & (panels == panel)])
            draw = rng.integers(len(labels), size=(10000, len(labels)))
            counts = np.array([(mask & (objs == ob)).sum() for ob in labels])
            den += counts[draw].sum(1)
            for j, ctrl in enumerate(comparators):
                delta = values[:, own] - values[:, idx[ctrl]]
                sums = np.stack([delta[mask & (objs == ob)].sum(0) for ob in labels])
                boots[:, j] += sums[draw].sum(1)
        keep = den > 0
        ci = np.quantile(boots[keep] / den[keep, None, None], [0.025, 0.975], axis=0)
        result["matched_minus"] = {
            ctrl: {
                metric: dict(
                    delta=float(means[own, k] - means[idx[ctrl], k]), ci95=ci[:, j, k].tolist()
                )
                for k, metric in enumerate(METRICS)
            }
            for j, ctrl in enumerate(comparators)
        }
        return result

    populations = {}
    allrows = np.ones(len(rows), bool)
    for name, pop in (("all", allrows), ("historical_strict", a["strict"])):
        m = {}
        m["every_row"] = report(pop, pop, CONTROLS)
        m["clear_reference"] = report(pop & (target != 0.5), pop, CONTROLS)
        m["any_policy_changes"] = report(pop & flags[:, : len(FLAGS)].any(1), pop, CONTROLS)
        for ctrl in CONTROLS:
            m["matched_differs_from_" + ctrl] = report(
                pop & (flags[:, own] != flags[:, idx[ctrl]]), pop, (ctrl,)
            )
        m["nonself_donor_and_matched_differs_from_wrong"] = report(
            pop & (rows != a["donor_rows"]) & (flags[:, own] != flags[:, idx["wrong"]]),
            pop,
            ("wrong",),
        )
        populations[name] = m
    n = populations["all"]["every_row"]
    differences = n["matched_minus"]
    improved = all(differences[c]["background_mse"]["ci95"][1] < 0 for c in CONTROLS)
    naming = n["credits"]["matched"]["full1854"] >= n["credits"]["original"]["full1854"]
    gates = dict(
        background_intervals_all_negative=improved,
        no_lost_naming=naming,
        background_gate=improved and naming,
        dino_improvement_lower_above_zero=differences["original"]["dino_ten"]["ci95"][0] > 0,
    )
    gates["joint_background_and_photo_gate"] = (
        gates["background_gate"] and gates["dino_improvement_lower_above_zero"]
    )
    label_counts = {
        key: {
            name: int(np.sum(codes[:, j] == v))
            for name, v in (("textured", 0), ("uncertain", 0.5), ("clean_light", 1))
        }
        for j, key in enumerate(("reference", "original", "texture_only"))
    }
    outcome = dict(
        triggered_rows={k: int(flags[:, idx[k]].sum()) for k in ARMS},
        matched_fallback_rows=int((~flags[:, own]).sum()),
        reference_labels=label_counts,
        wrong_word_rows=int(np.sum(a["truth"] != a["predicted_words"])),
        nonself_donor_rows=int(np.sum(rows != a["donor_rows"])),
        background_label_effect={},
        menus={},
    )
    for key, selected in (
        ("all_potential_pairs", allrows),
        ("matched_texture_requests", flags[:, own]),
    ):
        delta = values[:, idx["texture_only"], 0] - values[:, idx["original"], 0]
        outcome["background_label_effect"][key] = dict(
            n=int(selected.sum()),
            classified_differently=int(np.sum(selected & (codes[:, 1] != codes[:, 2]))),
            closer_to_reference=int(np.sum(selected & (delta < 0))),
            unchanged_disagreement=int(np.sum(selected & (delta == 0))),
            farther_from_reference=int(np.sum(selected & (delta > 0))),
            native_to_textured_prompt_counts=[
                [
                    int(np.sum(selected & (codes[:, 1] == before) & (codes[:, 2] == after)))
                    for after in (0, 0.5, 1)
                ]
                for before in (0, 0.5, 1)
            ],
        )
    for k in PANELS:
        m = panels == k
        outcome["menus"][k] = dict(
            rows=int(m.sum()),
            texture_requests=int(flags[m, own].sum()),
            reference_clear=int((target[m] != 0.5).sum()),
            mean_background_mse=dict(zip(ARMS, values[m, :, 0].mean(0).tolist(), strict=True)),
        )
    return populations, gates, outcome


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=Path("outputs"))
    ap.add_argument("--root", type=Path, default=Path("outputs/background-prompt/worker"))
    ap.add_argument("--out", type=Path, default=Path("outputs/background-prompt/evaluation"))
    ap.add_argument("--gallery", type=Path, default=PUBLIC.parents[1] / "gallery")
    ap.add_argument("--release", type=Path, default=PUBLIC / "background_prompt_images.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--provider", default="DmlExecutionProvider")
    a = ap.parse_args()
    a.fresh = a.data / "fresh-mix"
    a.model = a.data / "photo-structure/dino/facebook-dinov2-small"
    a.dictionary = a.data / "official-object-readout/sub-03/dictionary.csv"
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    image_lock = PUBLIC / "background_prompt_image_design.json"
    protocol = read(image_lock)
    worker = read(a.root / "worker.json")
    oldscored = read(PUBLIC / "fresh_mix_images.json")
    worddesign = read(PUBLIC / "fresh_mix_design.json")
    refs = read(PUBLIC / "background_attribute_target_switches.json")
    check(
        sha(image_lock) == LOCK == sha(a.root / "protocol.json")
        and worker["policy_sha256"] == sha(PUBLIC / "background_attribute_target_switches.json")
        and worker["confirmation_sha256"] == sha(PUBLIC / "background_attribute_confirmation.json")
        and read(PUBLIC / "background_attribute_confirmation.json")["gate_passed"]
        and sha(a.manifest) == worddesign["sources"]["original_photo_rows"],
        "Publication registry changed",
    )
    path = a.data / "photo-structure/training_images.zip"
    check(digest(path) == ZIP_SHA, "Reference stimulus ZIP hash differs")
    manifest = table(a)
    enc = encoder(a)
    masks = BackgroundMasks(a.data / "photo-structure/u2netp.onnx", a.provider)
    cases = []
    with ZipFile(path) as z:
        for p in PANELS:
            result = case(a, p, refs, enc, masks, z, manifest, oldscored, worddesign)
            cases.append(result)
            print("Verified", p, len(masks.cache), "unique decoded images", flush=True)
    populations, gates, outcome = aggregate(cases, protocol["seed"])
    for c in cases:
        c.pop("row_file")
    result = dict(
        design_sha256=LOCK,
        scorer_sha256=sha(Path(__file__)),
        worker_sha256=sha(a.root / "worker.json"),
        trigger_receipt_sha256=worker["policy_sha256"],
        stimulus_sha256=ZIP_SHA,
        manifest_sha256=sha(a.manifest),
        dino_sha256=FILES["model.safetensors"],
        clip_sha256=CLIP_SHA,
        mask_model_sha256=MODEL_SHA,
        provider=a.provider,
        unique_raw_image_masks=len(masks.cache),
        effects_units="Matched minus comparator. Background disagreement and RGB32 RMSE favor lower; naming and ten-photo credit favor higher. Means/intervals for naming/retrieval use the fractional 0-1 scale; credits are summed counts. Oracle uses original background exclusively for evaluation; ties keep native.",
        panels=cases,
        populations=populations,
        coverage=outcome,
        gates=gates,
    )
    freeze(a.out / "summary.json", result)
    freeze(a.release, result)
    print(
        "Gates",
        gates,
        "all matched minus",
        populations["all"]["every_row"]["matched_minus"],
        flush=True,
    )


if __name__ == "__main__":
    main()
