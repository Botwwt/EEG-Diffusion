"""Score verified paired fresh-mix JPEGs against true objects and original photographs."""

import argparse
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from fit_spatial_photo import original, table
from PIL import Image, ImageDraw
from register_fresh_photo_panel import freeze, read, sha
from render_fresh_mix_images import LOCK, METHODS, PANELS, condition_pair, folder, verify_images
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check
from run_prospective_readout import frozen_npz
from score_official_objects import topk_credit
from score_reconstruction_jpegs import CLIP_SHA, registered
from score_support_view import pictures
from score_within_photo_dino import FILES, ZIP_SHA, encoder

BASE = Path(__file__).resolve().parents[1]
PUBLIC = BASE / "results/object_fidelity"
METRICS = ("full1854", "closed12", "dino_ten")


def gallery(path, panel, images, reg, words, private):
    arms = ("original", *METHODS) if private else METHODS
    size = 126
    blockw = len(arms) * size + 14
    blockh = size + 60
    sheet = Image.new("RGB", (blockw * 2, blockh * 6 + 26), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text(
        (5, 4), f"{panel} | lowest original row in each registered object | sub-04", fill="black"
    )
    for k in range(12):
        ix = np.flatnonzero(reg["truth"] == k)
        q = ix[np.argmin(reg["rows"][ix])]
        x = blockw * (k % 2) + 4
        y = blockh * (k // 2) + 24
        draw.text((x, y), f"True: {reg['labels'][q]}   Row {reg['rows'][q]}", fill="black")
        for j, key in enumerate(arms):
            offset = x + size * j
            pic = images[key][q].resize((size, size), Image.Resampling.LANCZOS)
            sheet.paste(pic, (offset, y + 16))
            draw.text((offset, y + size + 19), key, fill="black")
            if key in METHODS:
                draw.text((offset, y + size + 34), str(words[key][q]), fill="black")
    path.parent.mkdir(exist_ok=True, parents=True)
    sheet.save(path, "JPEG", quality=90, subsampling=0)


def evaluate_case(a, panel, enc, archive, manifest, word_protocol):
    ref = registered(a.study / "references", panel)
    rows, truth, folds = ref["rows"], ref["truth"], ref["folds"]
    obj = rows // 10
    meta = word_protocol["panels"][panel]
    check(
        ref["sha"] == meta["registration_sha256"]
        and np.array_equal(np.array(meta["object_ids"])[truth], obj)
        and np.array_equal(ref["vocab"], meta["words"])
        and all(manifest[r]["unique_id"] == w for r, w in zip(rows, ref["labels"], strict=True)),
        "Original photo identity or menu differs",
    )
    data = condition_pair(a.root, panel)
    decisions = a.study / "decisions" / (panel + ".npz")
    expected = read(PUBLIC / "fresh_mix.json")
    check(sha(decisions) == expected["decision_sha256"][panel], "EEG word decisions changed")
    with np.load(decisions, allow_pickle=False) as frozen:
        check(
            np.array_equal(frozen["rows"], rows)
            and np.array_equal(frozen["folds"], folds)
            and np.array_equal(frozen["words"], ref["vocab"])
            and all(np.array_equal(frozen["words"][frozen[m]], data[m][2]) for m in METHODS),
            "Generated conditions disagree with the scored old or new word classifier",
        )
    settings, replay = verify_images(a.root, panel, data)
    words = {}
    images = {}
    namescores = a.root / "scores" / panel / "clip_logits.npz"
    per = np.empty((120, len(METHODS), len(METRICS)), dtype=float)
    with np.load(namescores, allow_pickle=False) as z:
        names = z["names"].tolist()
        check(
            str(z["model_sha256"]) == CLIP_SHA
            and str(z["dictionary_sha256"]) == DICTIONARY
            and np.array_equal(z["rows"], rows)
            and len(names) == len(set(names)) == 1854,
            "Incorrect naming rows, candidates or weights",
        )
        select = np.array([names.index(w) for w in ref["vocab"]])
        for j, m in enumerate(METHODS):
            cfg = settings[m]
            check(
                cfg["turbo_sha256"] == TURBO
                and cfg["adapter_sha256"] == ADAPTER_SHA
                and cfg["dictionary_sha256"] == DICTIONARY
                and str(z[m + "_settings_sha256"])
                == sha(folder(a.root, m, panel) / "settings.json")
                and np.array_equal(cfg["rows"], rows)
                and np.array_equal(cfg["folds"], folds),
                "Naming/condition receipt mismatch",
            )
            words[m] = np.array(cfg["predicted_words"])
            ix = np.array([ref["vocab"].tolist().index(w) for w in words[m]])
            check(
                np.array_equal(cfg["support_rows"], ref["calibration"][folds, ix]),
                "Wrong eight-support identities",
            )
            logits = z[m]
            check(
                logits.shape == (120, 1854) and np.isfinite(logits).all(),
                "Incomplete naming logits",
            )
            per[:, j, 0] = topk_credit(logits, 1)[np.arange(120), select[truth]]
            per[:, j, 1] = topk_credit(logits[:, select], 1)[np.arange(120), truth]
            images[m] = pictures(folder(a.root, m, panel) / "images/own", cfg["jpeg_sha256"]["own"])
    images["original"] = [original(archive, manifest, int(row)) for row in rows]
    r = enc(images["original"]).astype(float)
    for j, m in enumerate(METHODS):
        y = enc(images[m]).astype(float)
        sims = y @ r.T
        for k in range(12):
            idx = np.flatnonzero(truth == k)
            check(len(idx) == 10, "True-object image gallery incomplete")
            per[idx, j, 2] = topk_credit(sims[np.ix_(idx, idx)], 1).diagonal()
    check(np.isfinite(per).all(), "Nonfinite endpoint")
    same = words["legacy"] == words["learned"]
    check(
        np.array_equal(per[same, 0], per[same, 1]),
        "Identical generated JPEG receives different endpoint credit",
    )
    cache = a.study / "evaluation" / panel
    frozen_npz(
        cache / "per_row.npz",
        dict(
            rows=rows,
            object_ids=obj,
            metrics=np.array(METRICS),
            methods=np.array(METHODS),
            scores=per,
            words=np.stack(list(words.values())),
            strict=np.isin(obj, meta["primary_object_ids"]),
        ),
    )
    gallery(cache / "private.jpg", panel, images, ref, words, True)
    gallery(a.gallery / f"fresh_mix_{panel}.jpg", panel, images, ref, words, False)
    receipt = dict(
        registration_sha256=ref["sha"],
        clip_logits_sha256=sha(namescores),
        settings_sha256={m: sha(folder(a.root, m, panel) / "settings.json") for m in METHODS},
        condition_sha256={m: data[m][-1] for m in METHODS},
        row_scores_sha256=sha(cache / "per_row.npz"),
        fixed_generated_gallery_sha256=sha(a.gallery / f"fresh_mix_{panel}.jpg"),
        replay=replay,
    )
    return per, obj, receipt


def paired(data, selected, seed):
    # Each selected original object occurs in exactly one menu and contributes ten paired photos.
    rng = np.random.default_rng(seed)
    boot = np.zeros((10000, len(METRICS)))
    grouped = {}
    for panel in PANELS:
        values, objects, _ = data[panel]
        ids = np.asarray(selected[panel])
        g = np.stack([values[objects == o] for o in ids])
        check(g.shape == (len(ids), 10, 2, len(METRICS)), "Incomplete object bootstrap group")
        grouped[panel] = g
        delta = (g[:, :, 1] - g[:, :, 0]).sum(1)
        draws = rng.integers(len(ids), size=(10000, len(ids)))
        boot += delta[draws].sum(1)
    total = np.concatenate(list(grouped.values()))
    n = 10 * len(total)
    boot *= 100 / n

    def summary(x, k):
        counts = x[:, :, :, k].sum(axis=(0, 1))
        d = x[:, :, 1, k] - x[:, :, 0, k]
        return dict(
            n=10 * len(x),
            legacy_hits=float(counts[0]),
            learned_hits=float(counts[1]),
            credit_gained=float(d.clip(0).sum()),
            credit_lost=float((-d).clip(0).sum()),
            delta_pp=float(100 * d.mean()),
        )

    metrics = {}
    for k, key in enumerate(METRICS):
        res = summary(total, k)
        res["ci95_pp"] = np.quantile(boot[:, k], [0.025, 0.975]).tolist()
        res["per_menu"] = {p: summary(g, k) for p, g in grouped.items()}
        res["leave_one_menu_out_delta_pp"] = {
            p: summary(np.concatenate([g for n, g in grouped.items() if n != p]), k)["delta_pp"]
            for p in grouped
        }
        metrics[key] = res
    return dict(n=n, objects=len(total), metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", type=Path, default=Path("outputs/fresh-mix"))
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument(
        "--model", type=Path, default=Path("outputs/photo-structure/dino/facebook-dinov2-small")
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--gallery", type=Path, default=BASE / "gallery")
    p.add_argument("--out", type=Path, default=PUBLIC / "fresh_mix_images.json")
    a = p.parse_args()
    a.root = a.study / "render_inputs"
    reg = PUBLIC / "fresh_mix_images_design.json"
    design = read(reg)
    wordlock = PUBLIC / "fresh_mix_design.json"
    check(
        sha(reg) == LOCK
        and sha(wordlock) == design["word_design_sha256"]
        and sha(PUBLIC / "fresh_mix.json") == design["word_result_sha256"]
        and list(PANELS) == design["panels"],
        "Registered menu or endpoints changed",
    )
    w = read(wordlock)
    a.manifest = a.data / "official-object-readout/sub-03/atm_train_rows.csv"
    archive = a.data / "photo-structure/training_images.zip"
    check(
        sha(a.manifest) == w["sources"]["original_photo_rows"] and sha(archive) == ZIP_SHA,
        "Original stimulus records or archive changed",
    )
    manifest = table(a)
    enc = encoder(a)
    with ZipFile(archive) as z:
        cases = {p: evaluate_case(a, p, enc, z, manifest, w) for p in PANELS}
    groups = {}
    for name, key, n in (("strict", "primary_object_ids", 350), ("all", "object_ids", 480)):
        ids = {p: w["panels"][p][key] for p in PANELS}
        check(
            len(set(np.concatenate(list(ids.values())))) == n // 10
            and sum(map(len, ids.values())) == n // 10,
            "Overlapping object bootstrap groups",
        )
        groups[name] = paired(cases, ids, design["seed"])
    out = dict(
        protocol_sha256=LOCK,
        word_protocol_sha256=sha(wordlock),
        scorer_sha256=sha(Path(__file__)),
        clip_sha256=CLIP_SHA,
        dino_sha256=FILES["model.safetensors"],
        original_stimulus_archive_sha256=ZIP_SHA,
        participant="sub-04",
        populations=groups,
        panels={p: c[2] for p, c in cases.items()},
        positive_primary_lower_bound=bool(
            groups["strict"]["metrics"]["full1854"]["ci95_pp"][0] > 0
        ),
    )
    freeze(a.out, out)
    for key, v in groups["strict"]["metrics"].items():
        print(key, v["legacy_hits"], v["learned_hits"], v["ci95_pp"], flush=True)


if __name__ == "__main__":
    main()
