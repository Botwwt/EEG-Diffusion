"""Compare Fisher and mixed trial+Fisher EEG reconstructions within each registered personal12 menu."""

import argparse
from html import escape
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, save
from fit_support_structure import population, visual
from measure_foreground_geometry import ZIP_SHA
from PIL import Image, ImageDraw
from render_within_photo_ip import ARMS, check, packet
from score_fresh_photo import validate_packet
from score_prospective_readouts import strict
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_within_photo_dino import FILES, read, sha

PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"
DESIGN = PUBLIC / "trial_object_personal12_design.json"
REPLICATION = PUBLIC / "trial_object_personal12_replication_design.json"
SUBJECTS = ("sub-04", "sub-05", "sub-01")
PANEL = "personal12"
NAMES = ("fisher", "trial_plus_fisher")
SEED = 260552


def baseline_registration(root, subject):
    first = read(DESIGN)["inputs"]
    protocol = DESIGN if subject == "sub-04" else REPLICATION
    if subject == "sub-04":
        eeg, registration, baseline = (
            first[k] for k in ("sealed_eeg_receipt", "registration", "baseline_conditions")
        )
    else:
        design = read(protocol)
        check(
            design["participants"] == list(SUBJECTS[1:])
            and design["inputs"]["sub04_protocol"] == sha(DESIGN)
            and design["inputs"]["registration_receipt"] == first["baseline_inputs"]
            and design["inputs"]["renderer"] == first["intervention_renderer"],
            "Wrong replication design",
        )
        info = design["by_subject"][subject]["inputs"]
        eeg, registration, baseline = (
            info[k] for k in ("physical_receipt", "reference", "original_packet")
        )
    reg = registered(root / "evaluation/references" / subject, PANEL)
    receipt = root / "render_packets.json"
    personal = read(root / "references" / f"{PANEL}_{subject}_lock.json")
    official = read(root / "candidate_panel_lock.json")
    check(
        sha(receipt) == first["baseline_inputs"]
        and sha(root / "candidate_panel_lock.json") == first["panel_lock"]
        and sha(root / "eeg" / (subject + ".json")) == eeg
        and sha(root / "eeg" / (subject + ".npz")) == read(receipt)["target_eeg_sha256"][subject]
        and reg["sha"] == registration == personal["registration_sha256"]
        and personal["session_pair"] == official["panels"][subject]["session_pair"],
        "The registered personal12 population or physical R4 EEG changed",
    )
    return reg | dict(
        subject=subject,
        protocol=protocol,
        baseline_signature=baseline,
        session_pair=personal["session_pair"],
    )


def load_input(root, reg, baseline):
    subject = reg["subject"]
    b = root / "render_inputs"
    rows, folds, words, donors, supports, vectors, radius, signature = packet(b, subject, PANEL)
    path = b / "conditions" / subject / PANEL / "conditions_512.npz"
    word_file = b / "evaluations" / subject / PANEL / "word_predictions.json"
    pred = read(word_file)
    with np.load(path, allow_pickle=False) as data:
        check(
            str(data["subject"]) == subject
            and int(data["physical_trials_per_photo"]) == 4
            and str(data["registration_sha256"]) == reg["sha"]
            and str(data["word_prediction_sha256"]) == sha(word_file)
            and np.array_equal(pred["words"], reg["vocab"])
            and np.array_equal(words, reg["vocab"][pred["predicted"]])
            and pred["subject"] == subject
            and pred["panel"] == PANEL
            and np.array_equal(rows, pred["rows"])
            and np.array_equal(folds, pred["folds"])
            and np.array_equal(rows, reg["rows"])
            and np.array_equal(folds, reg["folds"])
            and np.array_equal(supports, reg["calibration"][folds, pred["predicted"]]),
            "Word predictions, EEG, or 8-image support set changed",
        )
        shared = {
            k: data[k]
            for k in (
                "target_eeg_sha256",
                "training_classes",
                "visual_sha256",
                "angular_scale_sha256",
            )
        }
    if baseline:
        check(signature == reg["baseline_signature"], "Wrong Fisher baseline")
    return dict(
        root=root,
        bundle=b,
        signature=signature,
        words=words,
        donors=donors,
        supports=supports,
        vectors=vectors,
        shared=shared,
        radius=radius,
        word_sha=sha(word_file),
        folder=b / "renders" / PANEL / subject,
    )


def same_inputs(left, right):
    same = left["words"] == right["words"]
    check(
        all(np.array_equal(v, right["shared"][k]) for k, v in left["shared"].items())
        and left["radius"] == right["radius"]
        and sha(left["bundle"] / "radius.json") == sha(right["bundle"] / "radius.json")
        and np.array_equal(left["supports"][same], right["supports"][same]),
        "Changing the word head also changed visual training, recipient, radius, or supports",
    )
    result = {}
    for arm in ARMS:
        a, b = left["vectors"][arm], right["vectors"][arm]
        equal = (a == b).all(1)
        if arm != "wrong":
            check(
                np.allclose(a[same], b[same], rtol=0, atol=2e-6),
                "Same word: visual condition differs",
            )
        result[arm] = same & equal
    return same, result


def image_scores(case, reg):
    subject = reg["subject"]
    root, bundle, folder = (case[k] for k in ("root", "bundle", "folder"))
    sf = folder / "settings.json"
    s = read(sf)
    args = SimpleNamespace(
        conditions=bundle / "conditions", panel=PANEL, evaluations=bundle / "evaluations"
    )
    validate_packet(args, subject, reg, s)
    check(
        s["design_sha256"] == sha(bundle / "design.json")
        and s["radius_sha256"] == sha(bundle / "radius.json")
        and s["packet_sha256"] == case["signature"]
        and np.array_equal(s["rows"], reg["rows"])
        and np.array_equal(s["folds"], reg["folds"])
        and np.array_equal(s["predicted_words"], case["words"])
        and np.array_equal(s["donor_rows"], case["donors"])
        and np.array_equal(s["seeds"], 923761 + reg["rows"]),
        "Generated image manifest differs from its inputs",
    )
    for arm in ARMS:
        paths = [folder / "images" / arm / f"{i:03d}.jpg" for i in range(120)]
        check([sha(p) for p in paths] == s["jpeg_sha256"][arm], "Changed generated image")
    clip_file = root / "evaluation/clip" / (subject + ".json")
    dino_file = root / "evaluation/pixels" / subject / PANEL / subject / "per_query.json"
    clip, dino = read(clip_file), read(dino_file)
    cc = clip["cases"][PANEL + "/" + subject]
    check(
        clip["clip_sha256"] == CLIP_SHA
        and clip["design_sha256"] == s["design_sha256"]
        and cc["settings_sha256"] == dino["settings_sha256"] == sha(sf)
        and dino["condition_sha256"] == case["signature"]
        and np.array_equal(cc["rows"], reg["rows"])
        and np.array_equal(cc["truth"], reg["labels"])
        and np.array_equal(cc["predicted_words"], case["words"])
        and np.array_equal(cc["folds"], reg["folds"])
        and np.array_equal(cc["donor_rows"], case["donors"]),
        "CLIP or DINO scorer read a different case",
    )
    shots = dino["images"]
    for field, expected in (
        ("row", reg["rows"]),
        ("fold", reg["folds"]),
        ("object", reg["labels"]),
        ("word", case["words"]),
    ):
        check(np.array_equal([r[field] for r in shots], expected), "DINO order or labels differ")
    summary_file = root / "evaluation/pixels" / subject / "summary.json"
    report = read(summary_file)
    check(
        report["dino"]["sha256"] == FILES
        and report["registration_sha256"] == reg["sha"]
        and report["cases"][subject]["provenance"]["settings_sha256"] == sha(sf),
        "DINO feature weights, photo set, or aggregate receipt changed",
    )
    metrics = {
        k: np.column_stack([cc["per_row"][k][a] for a in ARMS]) for k in ("full1854", "closed12")
    }
    metrics["dino_within10"] = np.array([[r["intra10"][a] for a in ARMS] for r in shots])
    check(
        all(
            x.shape == (120, 3) and np.isfinite(x).all() and np.all((x >= 0) & (x <= 1))
            for x in metrics.values()
        ),
        "Invalid per-image identification credit",
    )
    pairs, dp, cp = strict(dino, cc, reg["labels"], reg["folds"], reg["rows"], case["words"])
    return dict(
        settings=s,
        settings_sha=sha(sf),
        metrics=metrics,
        pairs=pairs,
        dino_pairs=dp,
        clip_pairs=cp,
        clip_meta={k: clip[k] for k in ("scorer_sha256", "manifest_sha256", "dictionary_sha256")},
        orig_sha=dino["original_jpeg_sha256_in_row_order"],
        dino_inputs=report["inputs"],
        scores_sha=dict(clip=sha(clip_file), dino=sha(dino_file), dino_summary=sha(summary_file)),
    )


def measure(x, labels, n_classes):
    """Last axis comprises paired endpoints. Weight eligible photos/pairs equally in each draw."""
    sums = np.zeros((n_classes, x.shape[1]))
    np.add.at(sums, labels, x)
    count = np.bincount(labels, minlength=n_classes)
    draw = np.random.default_rng(SEED).integers(n_classes, size=(10000, n_classes))
    den = count[draw].sum(1)
    sample = sums[draw].sum(1)[den > 0] / den[den > 0, None]
    return x.sum(0), sample


def difference(d, draw):
    return dict(
        net_credit=float(d.sum()),
        delta_rate=float(d.mean()),
        ci95_rate=np.quantile(draw, [0.025, 0.975]).tolist(),
        improved=int(np.sum(d > 0)),
        worsened=int(np.sum(d < 0)),
        unchanged=int(np.sum(d == 0)),
    )


def paired(x, y, labels, n_classes=12):
    if not len(labels):
        return dict(n=0)
    totals, draws = measure(np.column_stack([x, y]), labels, n_classes)
    na = x.shape[1]
    names = ARMS if na == 3 else ("word",)
    out = dict(
        n=len(x),
        credits={
            m: dict(zip(names, totals[j * na : (j + 1) * na].tolist(), strict=True))
            for j, m in enumerate(NAMES)
        },
    )
    out["trial_minus_fisher"] = {
        a: difference(y[:, k] - x[:, k], draws[:, na + k] - draws[:, k])
        for k, a in enumerate(names)
    }
    if na == 3:
        out["own_minus_control"] = {
            m: {
                a: difference(v[:, 0] - v[:, k], draws[:, j * na] - draws[:, j * na + k])
                for k, a in enumerate(names)
                if k
            }
            for j, (m, v) in enumerate(zip(NAMES, (x, y), strict=True))
        }
    return out


def visual_cases(a, reg, cases, word_case):
    subject = reg["subject"]
    spec = read(PUBLIC / "personal12_image_evaluation_design.json")[
        "frozen_gallery_lowest_row_by_object"
    ][subject]
    indices = [int(np.flatnonzero(reg["rows"] == v)[0]) for v in spec.values()]
    selected = set(indices)
    examples = []
    for kind in ("gain", "loss", "both_correct", "both_incorrect"):
        eligible = np.flatnonzero(word_case == kind)
        if len(eligible):
            i = int(eligible[np.argmin(reg["rows"][eligible])])
            examples.append(i)
            if i not in selected:
                indices.append(i)
                selected.add(i)
    args = SimpleNamespace(source=a.visual_source, structure=None)
    table = population(args)
    bank, sign = visual(args)
    check(sha(a.photos) == ZIP_SHA, "Gallery archive changed")
    tile = 134
    sheet = Image.new("RGB", (5 * (tile + 8), 30 + len(indices) * (tile + 34)), "white")
    draw = ImageDraw.Draw(sheet)
    for col, name in enumerate(
        (
            "Viewed photograph",
            "Fisher support lookup",
            "Fisher generation",
            "Mixed support lookup",
            "Mixed generation",
        )
    ):
        draw.text((col * (tile + 8) + 3, 7), name, fill="black")
    records = []
    with ZipFile(a.photos) as z:
        for j, i in enumerate(indices):
            row = int(reg["rows"][i])
            ims = [original(z, table, row)]
            labels = [f"{row} {reg['labels'][i]}"]
            rec = dict(row=row, ordinal=i, transition=str(word_case[i]))
            for name, c in zip(NAMES, cases, strict=True):
                support = c["supports"][i]
                lookup = int(support[np.argmax(bank["h14"][support] @ c["vectors"]["own"][i])])
                check(
                    lookup != row and table[lookup]["unique_id"] == c["words"][i],
                    "Illegal illustrative lookup",
                )
                ims.append(original(z, table, lookup))
                with Image.open(c["folder"] / "images/own" / f"{i:03d}.jpg") as im:
                    ims.append(im.convert("RGB"))
                labels.extend([f"{lookup} {c['words'][i]}", str(c["words"][i])])
                rec[name + "_support"] = lookup
            y = 30 + j * (tile + 34)
            for col, (im, label) in enumerate(zip(ims, labels, strict=True)):
                x = col * (tile + 8) + 3
                sheet.paste(im.resize((tile, tile), Image.Resampling.LANCZOS), (x, y))
                draw.text((x, y + tile + 1), label, fill="black")
            records.append(rec)
    dest = a.trial / "evaluation/paired"
    dest.mkdir(parents=True, exist_ok=True)
    sheet.save(dest / "private_cases.jpg", quality=92)
    complete_gallery(a, reg, cases, table, dest, examples, indices[:12])
    save(
        dest / "private_cases.json",
        dict(
            selection="Lowest registered original row per class plus the first numerical row of each available word transition",
            rows=records,
            lookup="Nearest of eight permitted support-photo H14 unit vectors to each readout's own H14 condition",
            visual_sha256=sign["h14"],
            sheet_sha256=sha(dest / "private_cases.jpg"),
            example_sha256=sha(dest / "private_examples.jpg"),
            example_rows=reg["rows"][examples].tolist(),
            all_object_rows=reg["rows"][indices[:12]].tolist(),
            overview_sha256=sha(dest / "private_all_objects.jpg"),
        ),
    )


def complete_gallery(a, reg, cases, table, dest, examples, overview):
    tile = 168
    panels = []
    with ZipFile(a.photos) as archive:
        for i, row in enumerate(reg["rows"]):
            ims = [original(archive, table, int(row))]
            for c in cases:
                with Image.open(c["folder"] / "images/own" / f"{i:03d}.jpg") as im:
                    ims.append(im.convert("RGB"))
            panels.append([im.resize((tile, tile), Image.Resampling.LANCZOS) for im in ims])

    def sheet(ids, dst):
        im = Image.new("RGB", (3 * (tile + 8), 28 + len(ids) * (tile + 22)), "white")
        draw = ImageDraw.Draw(im)
        for c, name in enumerate(("Viewed photo", "Fisher reconstruction", "Mixed reconstruction")):
            draw.text((c * (tile + 8) + 3, 6), name, fill="black")
        for j, i in enumerate(ids):
            y = 28 + j * (tile + 22)
            titles = [f"{reg['rows'][i]} {reg['labels'][i]}", *[str(c["words"][i]) for c in cases]]
            for c, (view, label) in enumerate(zip(panels[i], titles, strict=True)):
                x = c * (tile + 8) + 3
                im.paste(view, (x, y))
                draw.text((x, y + tile + 1), label.replace("_", " "), fill="black")
        im.save(dst, quality=92)

    sheet(examples, dest / "private_examples.jpg")
    sheet(overview, dest / "private_all_objects.jpg")
    tile_small = 133
    cell_w, cell_h = 2 * tile_small + 14, tile_small + 46
    generated = Image.new("RGB", (3 * cell_w, 4 * cell_h + 27), "white")
    pen = ImageDraw.Draw(generated)
    pen.text(
        (5, 7),
        f"{reg['subject']}   Viewed object above each pair.   Left: Fisher   Right: mixed trial + Fisher",
        fill="black",
    )
    for j, i in enumerate(overview):
        x = (j % 3) * cell_w + 4
        y = (j // 3) * cell_h + 29
        pen.text((x, y), f"{reg['labels'][i].replace('_', ' ')}   #{reg['rows'][i]}", fill="black")
        for c in range(2):
            xx = x + c * (tile_small + 3)
            generated.paste(
                panels[i][c + 1].resize((tile_small, tile_small), Image.Resampling.LANCZOS),
                (xx, y + 16),
            )
            pen.text(
                (xx, y + tile_small + 18), str(cases[c]["words"][i]).replace("_", " "), fill="black"
            )
    generated.save(dest / "generated_comparison.jpg", quality=94)
    if a.publish:
        public_gallery = PUBLIC.parent.parent / "gallery"
        public_gallery.mkdir(exist_ok=True, parents=True)
        (public_gallery / f"personal12_trial_{reg['subject'].replace('-', '')}.jpg").write_bytes(
            (dest / "generated_comparison.jpg").read_bytes()
        )
    pages = dest / "private_all_rows"
    pages.mkdir(exist_ok=True)
    index = []
    for j, word in enumerate(reg["vocab"]):
        ids = np.flatnonzero(reg["labels"] == word)
        ids = ids[np.argsort(reg["rows"][ids])]
        check(len(ids) == 10, "Gallery requires ten photos per true category")
        name = f"category{j + 1:02d}.jpg"
        sheet(ids, pages / name)
        index.append(
            dict(
                word=str(word), image=name, sha256=sha(pages / name), rows=reg["rows"][ids].tolist()
            )
        )
    links = " · ".join(
        f'<a href="#c{j}">{escape(x["word"].replace("_", " "))}</a>' for j, x in enumerate(index)
    )
    cards = "".join(
        f'<section id="c{j}"><h2>{escape(x["word"].replace("_", " "))}</h2><img src="{x["image"]}" alt="Ten viewed photos, Fisher reconstructions and mixed reconstructions"><p><a href="#menu">All categories</a></p></section>'
        for j, x in enumerate(index)
    )
    markup = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
        "<title>Personal12 paired reconstructions</title><style>body{font:16px system-ui;color:#17212b;max-width:62rem;margin:2rem auto;padding:0 1rem}section{margin:2rem 0}img{max-width:100%;height:auto}nav{line-height:1.8}</style>"
        f'<main><h1 id="menu">{escape(reg["subject"])} · personal12</h1><p>All 120 viewed photographs and their two matched-EEG reconstructions. Rows within each true category follow ascending original ID. Each reconstruction label gives its EEG-predicted word.</p><nav>{links}</nav>{cards}</main></html>'
    )
    (pages / "index.html").write_text(markup, encoding="utf8", newline="\n")
    save(
        pages / "manifest.json",
        dict(subject=reg["subject"], registration_sha256=reg["sha"], pages=index),
    )


def run(a):
    subject = a.subject
    reg = baseline_registration(a.baseline, subject)
    left = load_input(a.baseline, reg, baseline=True)
    check(
        str(left["shared"]["target_eeg_sha256"]) == sha(a.baseline / "eeg" / (subject + ".npz")),
        "Physical EEG packet differs",
    )
    if a.stage == "baseline":
        print(
            "Fisher packet verified:",
            left["signature"],
            "words correct",
            int(np.sum(left["words"] == reg["labels"])),
            "/120",
        )
        return
    right = load_input(a.trial, reg, baseline=False)
    receipt_path = a.trial / "receipt.json"
    receipt = read(receipt_path)
    check(
        receipt["protocol_sha256"] == sha(reg["protocol"])
        and receipt["registration_sha256"] == reg["sha"]
        and receipt["mixed_packet_sha256"] == right["signature"]
        and receipt["original_fisher_packet_sha256"] == left["signature"]
        and receipt["render_design_sha256"] == sha(right["bundle"] / "design.json")
        and receipt["word_predictions_sha256"] == right["word_sha"]
        and receipt["raw_results_sha256"] == sha(a.trial / "decisions.npz"),
        "Trial prediction receipt differs from the registered intervention",
    )
    same, equal = same_inputs(left, right)
    correct = [c["words"] == reg["labels"] for c in (left, right)]
    b, t = correct
    word_case = np.select(
        [~b & t, b & ~t, b & t], ["gain", "loss", "both_correct"], default="both_incorrect"
    )
    print(
        "Words:",
        int(b.sum()),
        "->",
        int(t.sum()),
        "unchanged words",
        int(same.sum()),
        "of 120",
        flush=True,
    )
    if a.stage == "inputs":
        print(
            "Same word with bit-identical conditions", {k: int(v.sum()) for k, v in equal.items()}
        )
        return
    cases = [left, right]
    data = [image_scores(c, reg) for c in cases]
    old, new = data
    for k in (
        "dictionary_sha256",
        "turbo_sha256",
        "adapter_sha256",
        "driver_sha256",
        "radius_sha256",
        "radius",
        "seeds",
        "rows",
        "folds",
        "batch_size",
    ):
        check(old["settings"][k] == new["settings"][k], "Different rendering setup: " + k)
    check(
        old["settings"]["driver_sha256"] == read(DESIGN)["inputs"]["intervention_renderer"],
        "Unexpected diffusion implementation",
    )
    check(
        old["clip_meta"] == new["clip_meta"]
        and old["dino_inputs"] == new["dino_inputs"]
        and old["orig_sha"] == new["orig_sha"],
        "Independent scoring source, dictionary or photo bank differs",
    )
    reference_names = a.baseline / "evaluation/clip" / (subject + "-refnames.json")
    reference = read(reference_names)
    check(
        reference["clip_sha256"] == CLIP_SHA
        and reference["registration_sha256"] == reg["sha"]
        and reference["dictionary_sha256"] == old["clip_meta"]["dictionary_sha256"]
        and reference["image_manifest_sha256"] == old["clip_meta"]["manifest_sha256"]
        and reference["targets"] == 120,
        "True-photo naming reference differs from the registered scoring gallery",
    )
    parity = {}
    for k in ARMS:
        jpeg = np.array(old["settings"]["jpeg_sha256"][k]) == np.array(
            new["settings"]["jpeg_sha256"][k]
        )
        parity[k] = dict(
            same_word_rows=int(same.sum()),
            same_word_equal_jpegs=int((same & jpeg).sum()),
            same_word_bitwise_equal_conditions=int(equal[k].sum()),
            same_word_bitwise_equal_conditions_and_jpegs=int((equal[k] & jpeg).sum()),
            max_abs_condition_difference_same_word=float(
                np.max(abs(left["vectors"][k][same] - right["vectors"][k][same]), initial=0)
            ),
            changed_word_rows=int((~same).sum()),
            changed_word_equal_jpegs=int((~same & jpeg).sum()),
        )
    outcomes = {}
    conditional = {}
    private = [
        dict(
            row=int(r),
            fold=int(f),
            truth=str(w),
            fisher_word=str(first),
            mixed_word=str(second),
            word_transition=str(wc),
        )
        for r, f, w, first, second, wc in zip(
            reg["rows"],
            reg["folds"],
            reg["labels"],
            left["words"],
            right["words"],
            word_case,
            strict=True,
        )
    ]
    for key, x in old["metrics"].items():
        y = new["metrics"][key]
        outcomes[key] = paired(x, y, reg["truth"])
        for i, rec in enumerate(private):
            rec[key] = {
                name: dict(zip(ARMS, m[i].tolist(), strict=True))
                for name, m in zip(NAMES, (x, y), strict=True)
            }
        subsets = {
            "same_word": same,
            "changed_word": ~same,
            **{tag: word_case == tag for tag in ("gain", "loss", "both_correct", "both_incorrect")},
        }
        conditional[key] = {}
        for tag, idx in subsets.items():
            conditional[key][tag] = dict(
                n=int(idx.sum()),
                fisher_own_credit=float(x[idx, 0].sum()),
                trial_own_credit=float(y[idx, 0].sum()),
                improved=int(np.sum(y[idx, 0] > x[idx, 0])),
                worsened=int(np.sum(y[idx, 0] < x[idx, 0])),
            )
    common = sorted(set(old["pairs"]) & set(new["pairs"]))
    vocab = reg["vocab"].tolist()
    for k in ("dino_pairs", "clip_pairs"):
        vec = [np.array([u[k][pair] for pair in common]).reshape(-1, 3) for u in data]
        outcomes[k + "_shared_correct_words"] = paired(
            *vec, np.array([vocab.index(old["pairs"][p]) for p in common], int)
        )
    report = dict(
        study=f"{subject}, four physical trials per stimulus, five fixed folds with eight support photos per candidate object. The baseline Fisher score on this panel was known before the mixed readout registration.",
        population=dict(
            participant=subject,
            physical_trials_per_photo=4,
            stimuli=120,
            objects=12,
            stimuli_per_object=10,
            session_pair=reg["session_pair"],
            generated_jpegs=720,
        ),
        provenance=dict(
            trial_design=sha(reg["protocol"]),
            trial_receipt=sha(receipt_path),
            registration=reg["sha"],
            source=sha(Path(__file__)),
            source_photo_sha_in_row_order=old["orig_sha"],
            independent_clip=old["clip_meta"] | {"model_sha256": CLIP_SHA},
            independent_dino_model_sha256=FILES["model.safetensors"],
            readouts={
                m: dict(
                    conditions=c["signature"],
                    settings=u["settings_sha"],
                    words=c["word_sha"],
                    **u["scores_sha"],
                )
                for m, c, u in zip(NAMES, cases, data, strict=True)
            },
        ),
        definitions=dict(
            full1854="OpenAI CLIP-L/14 names the true object against all 1,854 dictionary words.",
            closed12="The same scorer names the true object among the twelve panel words.",
            dino_within10="DINOv2-S/14 identifies the exact photograph among ten photos of its true category.",
            strict="Only a true-category pair for which both images received the correct word from both heads can contribute to the matched comparison. Ties split credit.",
            intervals="Paired percentile 95% bounds on rates, resampling twelve whole true categories with replacement, 10000 draws, seed 260552. Credits count fractional ties. Subset counts are descriptive.",
        ),
        eeg_word=paired(b[:, None].astype(float), t[:, None].astype(float), reg["truth"]),
        changed_words=int((~same).sum()),
        original_photo_naming=dict(
            n=120,
            full1854=reference["full1854_credit"],
            closed12=reference["closed12_credit"],
            source_sha256=sha(reference_names),
        ),
        word_correctness_groups={
            k: int(np.sum(word_case == k))
            for k in ("gain", "loss", "both_correct", "both_incorrect")
        },
        parity=parity,
        strict_eligibility={
            NAMES[0]: len(old["pairs"]),
            NAMES[1]: len(new["pairs"]),
            "common": len(common),
        },
        images=outcomes,
        matched_images_by_eeg_word=conditional,
    )
    dest = a.trial / "evaluation/paired"
    pairs = [
        dict(
            rows=list(p),
            truth=str(old["pairs"][p]),
            **{
                k: {
                    name: dict(zip(ARMS, u[k][p].tolist(), strict=True))
                    for name, u in zip(NAMES, data, strict=True)
                }
                for k in ("clip_pairs", "dino_pairs")
            },
        )
        for p in common
    ]
    save(dest / "per_row.json", dict(rows=private, pairs=pairs))
    report["provenance"]["private_score_rows_sha256"] = sha(dest / "per_row.json")
    visual_cases(a, reg, cases, word_case)
    report["generated_gallery"] = dict(
        filename=f"personal12_trial_{subject.replace('-', '')}.jpg",
        sha256=sha(dest / "generated_comparison.jpg"),
        selection="Smallest original row per registered true category; left Fisher, right mixed. Both images generated.",
    )
    save(dest / "summary.json", report)
    if a.publish and subject == "sub-04":
        (PUBLIC / "personal12_trial_images.json").write_bytes((dest / "summary.json").read_bytes())
    for key, r in outcomes.items():
        print(key, r["n"], r.get("credits", {}), flush=True)


def pooled_records(records):
    rows = [row for item in records for row in item["rows"]]
    pairs = [p for item in records for p in item["pairs"]]
    objects = sorted({r["truth"] for r in rows})
    lookup = {word: i for i, word in enumerate(objects)}
    classes = np.array([lookup[r["truth"]] for r in rows])
    words = [
        np.array([[float(r[field] == r["truth"])] for r in rows])
        for field in ("fisher_word", "mixed_word")
    ]
    outcomes = {"eeg_word": paired(*words, classes, len(objects))}
    for metric in ("closed12", "full1854", "dino_within10", "dino_pairs", "clip_pairs"):
        source = pairs if metric.endswith("_pairs") else rows
        arrays = [
            np.array([[r[metric][m][arm] for arm in ARMS] for r in source]).reshape(-1, 3)
            for m in NAMES
        ]
        outcomes[metric] = paired(
            *arrays, np.array([lookup[r["truth"]] for r in source], int), len(objects)
        )
    return dict(
        participant_photo_queries=len(rows),
        distinct_photographs=len({r["row"] for r in rows}),
        distinct_objects=len(objects),
        common_strict_pairs=len(pairs),
        changed_words=sum(r["fisher_word"] != r["mixed_word"] for r in rows),
        outcomes=outcomes,
    )


def pool(a):
    records = []
    individual = {}
    for subject in SUBJECTS:
        root = (
            (a.trial or Path("outputs/trial-object-encoder/personal12"))
            if subject == "sub-04"
            else a.replications / subject
        )
        reg = baseline_registration(a.baseline, subject)
        dest = root / "evaluation/paired"
        summary = read(dest / "summary.json")
        provenance = summary["provenance"]
        local = read(dest / "per_row.json")
        check(
            provenance["private_score_rows_sha256"] == sha(dest / "per_row.json")
            and provenance["trial_design"] == sha(reg["protocol"])
            and summary["population"]["participant"] == subject
            and provenance["trial_receipt"] == sha(root / "receipt.json")
            and provenance["source"] == sha(Path(__file__))
            and [r["row"] for r in local["rows"]] == reg["rows"].tolist()
            and [r["truth"] for r in local["rows"]] == reg["labels"].tolist(),
            "Run the personal image comparison on the registered case before pooling",
        )
        records.append(local)
        individual[subject] = dict(
            phase="initial" if subject == "sub-04" else "replication",
            EEG=summary["eeg_word"],
            reference_names=summary["original_photo_naming"],
            parity=summary["parity"],
            changed_words=summary["changed_words"],
            source_sha256=sha(dest / "summary.json"),
            generated_gallery=summary["generated_gallery"],
            metrics={
                k: dict(
                    n=v["n"],
                    credits=v.get("credits"),
                    trial_minus_fisher_own=v.get("trial_minus_fisher", {}).get("own"),
                )
                for k, v in summary["images"].items()
            },
        )
    report = dict(
        design_sha256=sha(REPLICATION),
        initial_design_sha256=sha(DESIGN),
        scorer_sha256=sha(Path(__file__)),
        methods="Each decision uses four physical EEG trials from one person and eight off-fold calibration photos per candidate. Confidence intervals resample the union of true object identities synchronously across the specified fixed people. Each eligible participant-photo or common strict pair receives equal weight. Rate intervals omit draws with no eligible pairs.",
        participants=list(SUBJECTS),
        first_participant="sub-04",
        prespecified_replication_subjects=list(SUBJECTS[1:]),
        individual=individual,
        replication_only=pooled_records(records[1:]),
        all_observed=pooled_records(records),
    )
    path = a.replications / "paired_images.json"
    save(path, report)
    if a.publish:
        (PUBLIC / "personal12_trial_replication_images.json").write_bytes(path.read_bytes())
    print("Replication and all-observed aggregates:", path, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("baseline", "inputs", "score", "pool"))
    parser.add_argument("--subject", choices=SUBJECTS, default="sub-04")
    parser.add_argument(
        "--replications", type=Path, default=Path("outputs/trial-object-replication")
    )
    parser.add_argument("--baseline", type=Path, default=Path("outputs/object-head-confirmation"))
    parser.add_argument(
        "--trial", type=Path, help="One input root, with render_inputs and receipt.json"
    )
    parser.add_argument(
        "--visual-source", type=Path, default=Path("outputs/official-object-readout/sub-03")
    )
    parser.add_argument(
        "--photos", type=Path, default=Path("outputs/photo-structure/training_images.zip")
    )
    parser.add_argument(
        "--publish", action="store_true", help="Copy the aggregate summary into research results"
    )
    a = parser.parse_args()
    if a.stage == "pool":
        pool(a)
    else:
        a.trial = a.trial or (
            Path("outputs/trial-object-encoder/personal12")
            if a.subject == "sub-04"
            else a.replications / a.subject
        )
        run(a)
