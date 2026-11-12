"""Shared paths and helpers for the repeat-aware EEG decoding experiments (THINGS-EEG + Brain-HIVE host)."""
import io
import json
import os
import pickle
import zipfile
from pathlib import Path

import numpy as np

if "RAD_ROOT" not in os.environ:
    raise SystemExit("Set RAD_ROOT to the working directory that holds data/, models/, code/Brain-HIVE and runs/ (see ../README.md).")
ROOT = Path(os.environ["RAD_ROOT"])
RAW = ROOT / "data/things-eeg/Preprocessed_data_250Hz_whiten"
IMG = ROOT / "data/things-eeg/Image_set"
EMB = ROOT / "data/hive_embeddings/things-eeg"
PREP = ROOT / "data/prep"
PRIOR = ROOT / "models/Fusion-Prior-H14_B32_VAE"
TURBO = ROOT / "models/sdxl-turbo"
HIVE = ROOT / "code/Brain-HIVE"
RUNS = ROOT / "runs"

# The 17 occipito-parietal channels of the paper (Sec. 4.1 / App. B).
CH17 = ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2"]
# scripts/build_reconstruction.sh at commit f06db69 lists "Oz" where "POz" belongs (Oz twice, POz dropped).
CH17_TYPO = ["P7", "P5", "P3", "P1", "Pz", "P2", "P4", "P6", "P8", "PO7", "PO3", "Oz", "PO4", "PO8", "O1", "Oz", "O2"]
EMB_KEYS = ["CLIP-ViT-H-14-laion2B-s32B-b79K", "CLIP-ViT-B-32-laion2B-s34B-b79K", "vae"]

# Globals a pickle of plain numpy arrays / python containers may import. Nothing else is allowed, so loading
# a downloaded file cannot run code from it.
_SAFE = {
    ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
    ("numpy.core.multiarray", "scalar"), ("numpy._core.multiarray", "scalar"),
    ("numpy.core.numeric", "_frombuffer"), ("numpy._core.numeric", "_frombuffer"),
    ("numpy", "ndarray"), ("numpy", "dtype"), ("collections", "OrderedDict"),
}


class _RestrictedUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _SAFE:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"blocked global {module}.{name}")

    def persistent_load(self, pid):
        raise pickle.UnpicklingError("persistent ids (torch storages) are not expected in these files")


def safe_load_pt(path):
    """Load a torch-zip .pt whose payload is a pickle of numpy arrays, with a strict allow-list."""
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith("data.pkl")]
        assert len(names) == 1, names
        with z.open(names[0]) as f:
            return _RestrictedUnpickler(io.BufferedReader(f, buffer_size=1 << 24)).load()


def channel_indices(names):
    name2idx = {}
    with open(HIVE / "configs/THINGS_EEG_CHANNELS.jsonl", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                o = json.loads(line)
                name2idx[o["name"]] = int(o["index"])
    return [name2idx[n] for n in names]


def concept_folds(n_concepts, n_folds=5, seed=20261112):
    """Concept-level fold id per training concept (fixed permutation, independent of training seeds)."""
    perm = np.random.default_rng(seed).permutation(n_concepts)
    fold = np.empty(n_concepts, dtype=np.int64)
    fold[perm] = np.arange(n_concepts) % n_folds
    return fold


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False))
    tmp.replace(path)
