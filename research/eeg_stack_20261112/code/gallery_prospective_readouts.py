"""Freeze one row per object and compare both models' own-EEG JPEGs on those rows."""

import argparse
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from register_fresh_photo_panel import freeze
from render_official_words import crop512
from score_fresh_photo import validate_packet
from score_reconstruction_jpegs import registered
from score_within_photo_dino import ROWS_SHA, ZIP_SHA, read, reference, sha

PUBLIC = Path("research/eeg_stack_20261112")
PANEL = "prospective"
METHODS = ("expanded_r4", "full_head_512")
LABELS = {METHODS[0]: "Posterior EEG", METHODS[1]: "Full-scalp EEG"}


def make_plan(reg, lock):
    picks = []
    for label in sorted(reg["vocab"].tolist()):
        ix = np.flatnonzero(reg["labels"] == label)
        i = int(ix[np.argmin(reg["rows"][ix])])
        picks.append(
            dict(object=label, row=int(reg["rows"][i]), ordinal=i, fold=int(reg["folds"][i]))
        )
    return dict(
        registration_sha256=reg["sha"],
        panel_lock_sha256=sha(lock),
        rule="Take the lowest original row for each registered object, in alphabetical object order, for every participant. Display the own-EEG arm of each frozen readout.",
        participants=read(lock)["participants"],
        methods=list(METHODS),
        shown=picks,
    )


def generated(root, method, subject, reg, samples, provenance):
    folder = root / "renders" / PANEL / subject
    settings = read(folder / "settings.json")
    if (
        settings["design_sha256"] != provenance["render_designs_sha256"][method]
        or settings["design_sha256"] != sha(root / "design.json")
        or settings["packet_sha256"]
        != provenance["cases"][subject]["methods"][method]["conditions_sha256"]
        or not np.array_equal(settings["rows"], reg["rows"])
        or not np.array_equal(settings["folds"], reg["folds"])
        or not np.array_equal(settings["seeds"], 923761 + reg["rows"])
    ):
        raise ValueError(
            "Render design, EEG packet, row, fold or seed differs from the frozen inputs"
        )
    validate_packet(
        SimpleNamespace(
            conditions=root / "conditions", evaluations=root / "evaluations", panel=PANEL
        ),
        subject,
        reg,
        settings,
    )
    word_file = root / "evaluations" / subject / PANEL / "word_predictions.json"
    if not np.array_equal(settings["predicted_words"], reg["vocab"][read(word_file)["predicted"]]):
        raise ValueError("Rendered word labels differ from frozen readout")
    images, keys = [], []
    for record in samples:
        i = record["ordinal"]
        raw = (folder / "images/own" / f"{i:03d}.jpg").read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != settings["jpeg_sha256"]["own"][i]:
            raise ValueError("Changed generated JPEG")
        with Image.open(io.BytesIO(raw)) as im:
            if im.format != "JPEG" or im.size != (512, 512):
                raise ValueError("Unexpected rendered image format")
            images.append((im.convert("RGB"), settings["predicted_words"][i]))
        keys.append(digest)
    return images, dict(settings_sha256=sha(folder / "settings.json"), own_jpeg_sha256=keys)


def private_originals(args, samples):
    table = reference(args)
    images, keys = [], []
    with ZipFile(args.originals) as archive:
        for record in samples:
            info = table[record["row"]]
            if info["unique_id"] != record["object"]:
                raise ValueError("Registered object differs from the original row manifest")
            raw = archive.read(
                f"training_images/{info['official_folder']}/{info['stimulus_stem']}.jpg"
            )
            with Image.open(io.BytesIO(raw)) as im:
                images.append(crop512(im.convert("RGB")))
            keys.append(hashlib.sha256(raw).hexdigest())
    return images, keys


def sheet(path, subject, samples, branches, originals=None):
    tile, gap, margin, columns = 171, 6, 18, 3
    per_object = 3 if originals is not None else 2
    card_width = (tile + gap) * per_object + 12
    card_height = tile + 72
    width = 2 * margin + columns * card_width
    out = Image.new("RGB", (width, 62 + 4 * card_height + 10), "white")
    draw = ImageDraw.Draw(out)
    small, body, title = (ImageFont.load_default(size=s) for s in (13, 16, 23))
    draw.text((margin, 10), f"Fourth object panel   /   {subject}", fill="#152534", font=title)
    draw.text(
        (margin, 40),
        "Lowest original row in each of 12 objects. Both images use the query's own EEG and the same generation seed.",
        fill="#3c5565",
        font=small,
    )
    for j, record in enumerate(samples):
        x = margin + (j % columns) * card_width
        y = 65 + (j // columns) * card_height
        draw.text(
            (x, y),
            f"{record['object'].replace('_', ' ')}   /   row {record['row']}",
            fill="#152534",
            font=body,
        )
        photos = [(LABELS[m], *branches[m][j]) for m in METHODS]
        if originals is not None:
            photos.insert(0, ("Viewed photo", originals[j], record["object"]))
        for col, (role, image, word) in enumerate(photos):
            xx = x + col * (tile + gap)
            out.paste(image.resize((tile, tile), Image.Resampling.LANCZOS), (xx, y + 23))
            draw.text((xx, y + tile + 26), role, fill="#3c5565", font=small)
            draw.text((xx, y + tile + 44), word.replace("_", " "), fill="#152534", font=body)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.save(path, "JPEG", quality=94, optimize=True, subsampling=0)
    return dict(file=path.name, sha256=sha(path), size=list(out.size))


def run(a):
    reg = registered(a.references, PANEL)
    if read(a.panel_lock)["registration_sha256"] != reg["sha"]:
        raise ValueError("Changed independent panel registration")
    plan = make_plan(reg, a.panel_lock)
    freeze(a.plan, plan)
    if a.stage == "plan":
        return
    if not a.private_dir.resolve().is_relative_to(Path("outputs").resolve()):
        raise ValueError("Local original-photo sheets must remain under outputs/")
    provenance = read(a.input_receipt)
    if provenance["registration_sha256"] != reg["sha"] or provenance["lock_sha256"] != sha(
        a.panel_lock
    ):
        raise ValueError("Wrong pair of frozen render inputs")
    # Verify both independent generated branches, across all three participants, before opening references.
    payloads, cases = {}, {}
    for sub in plan["participants"]:
        payloads[sub], cases[sub] = {}, {}
        for name, root in ((METHODS[0], a.posterior_root), (METHODS[1], a.full_head_root)):
            payloads[sub][name], cases[sub][name] = generated(
                root, name, sub, reg, plan["shown"], provenance
            )
    sources, original_hashes = private_originals(a, plan["shown"])
    for sub in plan["participants"]:
        stem = f"prospective_{sub.replace('-', '')}"
        cases[sub]["public"] = sheet(
            a.public_dir / (stem + ".jpg"), sub, plan["shown"], payloads[sub]
        )
        cases[sub]["private"] = sheet(
            a.private_dir / (stem + "_originals.jpg"), sub, plan["shown"], payloads[sub], sources
        )
    freeze(
        a.receipt,
        dict(
            plan_sha256=sha(a.plan),
            render_inputs_sha256=sha(a.input_receipt),
            source_zip_sha256=ZIP_SHA,
            row_manifest_sha256=ROWS_SHA,
            selected_original_sha256=original_hashes,
            cases=cases,
        ),
    )
    freeze(
        a.public_receipt,
        dict(
            plan_sha256=sha(a.plan),
            render_inputs_sha256=sha(a.input_receipt),
            original_comparison_receipt_sha256=sha(a.receipt),
            cases={
                sub: {name: value for name, value in record.items() if name != "private"}
                for sub, record in cases.items()
            },
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", choices=("plan", "build"), default="plan")
    work = Path("outputs/prospective-photo")
    results = PUBLIC / "results/object_fidelity"
    for k, val in dict(
        references=work / "references",
        panel_lock=results / "prospective_panel_design.json",
        plan=results / "prospective_gallery_plan.json",
        input_receipt=results / "prospective_render_inputs.json",
        posterior_root=work / "render_inputs" / METHODS[0],
        full_head_root=work / "render_inputs" / METHODS[1],
        private_dir=work / "gallery",
        public_dir=PUBLIC / "gallery",
        originals=Path("outputs/photo-structure/training_images.zip"),
        manifest=Path("outputs/official-object-readout/sub-03/atm_train_rows.csv"),
        receipt=work / "gallery/receipt.json",
        public_receipt=results / "prospective_gallery.json",
    ).items():
        p.add_argument("--" + k.replace("_", "-"), type=Path, default=val)
    run(p.parse_args())
