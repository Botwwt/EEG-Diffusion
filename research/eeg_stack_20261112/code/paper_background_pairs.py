"""Create training-only color and texture descriptors from the edges of photographs."""
import os
from pathlib import Path
import zipfile

import numpy as np
import torch


def border_descriptor(rgb):
    height, width, colors = rgb.shape
    if colors != 3 or height % 4 or width % 4:
        raise ValueError("Expected an RGB photo with four equal cell divisions")
    cells = rgb.reshape(4, height // 4, 4, width // 4, 3).transpose(0, 2, 1, 3, 4)
    border = np.ones((4, 4), dtype=bool)
    border[1:3, 1:3] = False
    tiles = cells[border].reshape(12, -1, 3).astype(np.float32) / 255
    return np.concatenate((tiles.mean(1), tiles.std(1)), axis=1).ravel()


def photo_pairs(members, context, seed, *, distant, alternatives=16):
    """Draw independent pairings for each object and choose by training border distance."""
    count, n_photos = members.shape
    if n_photos % 2 or alternatives < 1:
        raise ValueError("Photo pairings need an even number of photos and at least one alternative")
    order = np.random.default_rng(seed).random((count, alternatives, n_photos)).argsort(axis=-1)
    choices = np.broadcast_to(members[:, None, :], order.shape)
    pairs = np.take_along_axis(choices, order, axis=2).reshape(count, alternatives, n_photos // 2, 2)
    if not distant:
        return pairs[:, 0]
    distance = ((context[pairs[..., 0]] - context[pairs[..., 1]]) ** 2).mean(axis=-1)
    return pairs[np.arange(count), distance.sum(axis=-1).argmax(axis=1)]


def prepare():
    from common import PREP
    from paper_render_cache import PinnedFiles, atomic_file, save_json
    from paper_rgb_object import read_photo, registered_photos

    root = Path(os.environ["EEG_CROSSFIT_ROOT"])
    source = Path(os.environ["ARCHIVE"]) / "assets/things_images_v254/training_images.zip"
    rows, labels, members = registered_photos(PREP)
    pins = PinnedFiles()
    proof = dict(archive=pins.pin(source), split=pins.pin(PREP / "paper_split.json"),
                 code=pins.pin(Path(__file__)), photo_index=pins.pin(PREP / "train_ids.json"),
                 photo_labels=pins.pin(PREP / "train_concept.npy"))
    values = np.empty((len(rows), 72), dtype=np.float32)
    with zipfile.ZipFile(source) as archive:
        for i, row in enumerate(rows):
            values[i] = border_descriptor(read_photo(archive, members[int(row)], 64))
            if (i + 1) % 4000 == 0:
                print("Photograph border descriptors", i + 1, flush=True)
    values = (values - values.mean(0)) / values.std(0).clip(.03)
    group = np.argsort(labels, kind="stable").reshape(1323, 10)
    gap = {}
    for name, distant in (("uniform", False), ("background", True)):
        pair = photo_pairs(group, values, 20261201, distant=distant)
        gap[name] = float(((values[pair[..., 0]] - values[pair[..., 1]]) ** 2).mean())
    pack = dict(rows=torch.from_numpy(rows), border=torch.from_numpy(values), proof=proof)
    pins.verify()
    root.mkdir(parents=True, exist_ok=True)
    dest = root / "training-borders.pt"
    atomic_file(dest, lambda f: torch.save(pack, f))
    save_json(dest.with_suffix(".json"), dict(proof, checkpoint_sha256=pins.pin(dest),
                                             photos=len(rows), objects=len(group), squared_pair_gap=gap))
    print("Photograph border descriptors ready", len(rows), gap, flush=True)


if __name__ == "__main__":
    prepare()
