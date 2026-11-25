"""Score the two generated sets using the frozen independent 1,854-word CLIP-L/14 vocabulary."""

import argparse
from pathlib import Path

import numpy as np
from eval_instance_starts import load_visual_scorer
from register_fresh_photo_panel import sha
from render_fresh_mix_images import METHODS, PANELS, condition_pair, folder, verify_images
from render_personal_query import DICTIONARY, TURBO
from render_within_photo_ip import ADAPTER_SHA, check
from run_prospective_readout import frozen_npz
from score_official_objects import load_dictionary, load_tokenizer, text_bank, unit
from score_reconstruction_jpegs import CLIP_SHA, raw_features
from score_support_view import pictures


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--panel", choices=PANELS, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    a = p.parse_args()
    settings, identity = verify_images(a.root, a.panel, condition_pair(a.root, a.panel))
    model, digest = load_visual_scorer(a.archive.resolve())
    check(
        digest == CLIP_SHA and sha(a.dictionary) == DICTIONARY,
        "Changed independent naming model or vocabulary",
    )
    names, common, inverse, _, _ = load_dictionary(a.dictionary)
    tok = load_tokenizer(a.archive / "assets/native_generator_v243/turbo/tokenizer")
    text = text_bank(model, tok, common, 16)
    arrays = {}
    for method in METHODS:
        base = folder(a.root, method, a.panel)
        cfg = settings[method]
        check(
            cfg["turbo_sha256"] == TURBO
            and cfg["adapter_sha256"] == ADAPTER_SHA
            and cfg["dictionary_sha256"] == DICTIONARY,
            "Unexpected diffusion assets",
        )
        images = pictures(base / "images/own", cfg["jpeg_sha256"]["own"])
        vector = unit(raw_features(model, images.__getitem__, 120, 16))
        arrays[method] = (vector @ text.T)[:, inverse]
        arrays[method + "_settings_sha256"] = sha(base / "settings.json")
    check(
        len(names) == 1854
        and all(arrays[m].shape == (120, 1854) and np.isfinite(arrays[m]).all() for m in METHODS),
        "Naming logits incomplete",
    )
    frozen_npz(
        a.root / "scores" / a.panel / "clip_logits.npz",
        dict(
            rows=np.asarray(settings["legacy"]["rows"]),
            names=np.asarray(names),
            model_sha256=digest,
            dictionary_sha256=DICTIONARY,
            **arrays,
        ),
    )
    print(a.panel, identity, "complete naming rows", flush=True)


if __name__ == "__main__":
    main()
