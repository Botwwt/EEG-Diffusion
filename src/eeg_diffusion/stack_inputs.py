"""Read-only inspection of prepared inputs for the archived O13 replay."""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PREPARED = Path("data/prep")
_FILES = (
    "code/Brain-HIVE/main/layers.py",
    "code/Brain-HIVE/main/models_adapter.py",
    "data/prep/test_ids.json",
    "data/prep/emb_train.pt",
    "data/prep/emb_test.pt",
    "data/prep/F_sem_train.pt",
    "data/prep/unet_kv_metric.pt",
    "data/prep/lat512_train.npy",
    "data/prep/lat512_train_ok.npy",
    "models/sdxl-turbo/model_index.json",
)
_DIRECTORIES = (
    "data/things-eeg/Image_set/test_images",
    "models/Fusion-Prior-H14_B32_VAE/fusion_encoder",
    "models/Fusion-Prior-H14_B32_VAE/attn_adapter",
    "models/Fusion-Prior-H14_B32_VAE/proj",
)


def check_o13_inputs(rad_root: Path) -> dict:
    """Check expected paths and array shapes without loading weights or EEG values."""
    root = rad_root.resolve()
    arrays = {
        _PREPARED / "F_train.npy": (16540, 1024),
        _PREPARED / "F_test.npy": (200, 1024),
        _PREPARED / "train_concept.npy": (16540,),
    }
    for index in range(1, 11):
        subject = f"sub-{index:02d}"
        arrays[_PREPARED / f"{subject}_train.npy"] = (16540, 4, 17, 250)
        arrays[_PREPARED / f"{subject}_test.npy"] = (200, 80, 17, 250)
        for mode, condition in (
            ("v2", "cone_o10_bifurc_kvtta_w1.00"),
            ("0tta", "cone_o10_bifurc_0tta_w1.00"),
        ):
            arrays[Path("runs/cond") / condition / f"{subject}.npy"] = (200, 1024)
            arrays[Path("runs/low") / subject / f"lat512_o10_bifurc_{mode}_n0.80.npy"] = (
                200,
                4,
                64,
                64,
            )

    missing = [name for name in _FILES if not (root / name).is_file()]
    missing.extend(name for name in _DIRECTORIES if not (root / name).is_dir())
    invalid = {}
    for relative, expected in arrays.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative.as_posix())
            continue
        try:
            array = np.lib.format.open_memmap(path, mode="r")
            actual = array.shape
            del array
        except (OSError, ValueError) as error:
            invalid[relative.as_posix()] = str(error)
            continue
        if actual != expected:
            invalid[relative.as_posix()] = f"Expected shape {expected}, found {actual}"
    return {
        "rad_root": str(root),
        "status": "missing_or_invalid" if missing or invalid else "passed",
        "paths_checked": len(_FILES) + len(_DIRECTORIES) + len(arrays),
        "missing": missing,
        "invalid_arrays": invalid,
        "scope": "O13 path and NumPy shape preflight; model files, data ordering and CUDA untested.",
    }
