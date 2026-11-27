"""Score the frozen readouts and two-object assignments on held photographs."""
import json
import os
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn.functional import normalize

from common import PREP
from paper_background_pairs import border_descriptor
from paper_object_foreground import closest_other_context, root
from paper_object_foreground_render import language_scorer, targets_for_budget
from paper_render_cache import PinnedFiles, save_json
from paper_rgb_object import read_photo
from score_official_objects import topk_credit


@torch.inference_mode()
def run():
    archive = Path(os.environ["ARCHIVE"])
    pins = PinnedFiles()
    text = language_scorer(archive, pins)
    panels = {"dev-r4": (1, 3), "official-r4": (1, 2, 3, 4), "official-r80": (1, 2, 3, 4)}
    member_names = json.loads((PREP / "train_ids.json").read_bytes())["paths"]
    images = archive / "assets/things_images_v254/training_images.zip"
    image_proof = pins.pin(images)
    metrics, inputs, paired = {}, {}, {}
    for budget, people in panels.items():
        metrics[budget], inputs[budget] = {}, {}
        for s in people:
            path = root() / "images" / budget / f"sub-{s:02d}/semantic-readout.pt"
            input_digest = pins.pin(path)
            data = torch.load(path, map_location="cpu", weights_only=True)
            rows = data["proof"]["row"]
            labels, proof = targets_for_budget(budget, rows, text, pins)
            actual = np.array([text.reverse[one] for one in labels])
            sample = np.arange(len(rows))
            if budget == "dev-r4" and s == people[0]:
                with zipfile.ZipFile(images) as photos:
                    edges = np.stack([border_descriptor(read_photo(photos, member_names[i], 64)) for i in rows])
                context = torch.from_numpy(edges)
                donors = closest_other_context(context, torch.from_numpy(actual), use_context=True).numpy()
                paired["closest_other_object"] = donors.tolist()
            inputs[budget][s] = {"readout": input_digest, "reference": proof}
            variants = {}
            for arm, pool in data["pool"].items():
                logits = (normalize(pool.cuda()) @ text.anchors.T).cpu().numpy()[:, text.inverse]
                rank = 1 + (logits > logits[sample, actual][:, None]).sum(1)
                stat = {f"top{k}": float(topk_credit(logits, k)[sample, actual].sum()) for k in (1, 5)}
                stat["median_rank"] = float(np.median(rank))
                if budget == "dev-r4":
                    match = logits[sample, actual] + logits[donors, actual[donors]]
                    swap = logits[sample, actual[donors]] + logits[donors, actual]
                    stat["background_matched_assignment"] = float(((match > swap) + .5 * (match == swap)).sum())
                variants[arm] = stat
            metrics[budget][s] = variants
    pins.verify()
    save_json(root() / "lexical-pair-diagnostics.json", dict(
        results=metrics, inputs=inputs, vocabulary=text.source, images=image_proof, **paired,
        source=pins.pin(Path(__file__))))
    for budget, results in metrics.items():
        print(budget, {arm: {m: sum(p[arm][m] for p in results.values()) for m in values
                             if m != "median_rank"} for arm, values in next(iter(results.values())).items()}, flush=True)


if __name__ == "__main__":
    run()
