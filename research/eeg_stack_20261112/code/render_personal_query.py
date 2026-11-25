"""Render one inference packet as a 512 px image with the archived personal EEG pipeline."""

import argparse
import csv
import tempfile
from pathlib import Path

import numpy as np
from render_fixed_word_guidance import embedding, load_pipeline
from render_within_photo_ip import ADAPTER_SHA, check, read, save, sha

TURBO = "e869ac7d6942cb327d68d5ed83a40447aadf20e0c3358d98b2cc9e270db0da26"
DICTIONARY = "1aebc7a2c35edcfcfffc63934754440c90534ccc46df0537db41dc6197450132"


def render(a):
    import torch

    with np.load(a.input, allow_pickle=False) as data:
        word = str(data["word"])
        direction = data["own"]
        radius = float(data["radius"])
        check(
            direction.shape == (1024,)
            and np.isclose(np.linalg.norm(direction), 1.0, atol=1e-5)
            and np.isfinite(radius)
            and radius > 1
            and word in data["candidate_words"].tolist(),
            "Invalid predicted word, H/14 direction or native magnitude",
        )
    weight = a.archive / "assets/native_generator_v243/turbo/sd_xl_turbo_1.0_fp16.safetensors"
    check(
        sha(weight) == TURBO and sha(a.adapter) == ADAPTER_SHA and sha(a.dictionary) == DICTIONARY,
        "Unexpected generator, adapter or word dictionary",
    )
    check(0 <= a.seed < 2**63, "Use a nonnegative 63-bit seed")
    with a.dictionary.open(encoding="utf-8-sig", newline="") as f:
        names = {r["unique_id"]: r["word"] for r in csv.DictReader(f)}
    check(word in names, "Predicted word missing from the calibrated vocabulary")
    lock = dict(
        prediction_sha256=sha(a.input),
        word=word,
        seed=a.seed,
        radius=radius,
        adapter_sha256=ADAPTER_SHA,
        turbo_sha256=TURBO,
        dictionary_sha256=DICTIONARY,
        renderer_sha256=sha(Path(__file__)),
    )
    if a.out.exists():
        saved = read(a.out / "settings.json")
        check(
            saved == {**lock, "jpeg_sha256": sha(a.out / "reconstruction.jpg")},
            "Existing reconstruction differs from the requested prediction or seed",
        )
        return
    pipe = load_pipeline(a)
    with torch.inference_mode():
        prompt, _, pooled, _ = pipe.encode_prompt(
            "A photo of " + names[word] + ".", device="cuda", do_classifier_free_guidance=False
        )
        pipe.text_encoder = pipe.text_encoder_2 = None
        torch.cuda.empty_cache()
        result = pipe(
            prompt_embeds=prompt,
            pooled_prompt_embeds=pooled,
            ip_adapter_image_embeds=[embedding(direction[None], [0], radius)],
            generator=torch.Generator(device="cuda").manual_seed(a.seed),
            width=512,
            height=512,
            num_inference_steps=4,
            guidance_scale=0.0,
            num_images_per_prompt=1,
            output_type="pil",
        ).images[0]
    check(result.size == (512, 512), "Unexpected reconstruction shape")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=a.out.parent) as t:
        stage = Path(t) / "render"
        stage.mkdir()
        image = stage / "reconstruction.jpg"
        result.convert("RGB").save(image, "JPEG", quality=95, subsampling=0)
        save(stage / "settings.json", {**lock, "jpeg_sha256": sha(image)})
        stage.rename(a.out)
    print(word, a.out / "reconstruction.jpg", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("input", "archive", "dictionary", "adapter", "out"):
        p.add_argument("--" + key, type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    render(p.parse_args())
