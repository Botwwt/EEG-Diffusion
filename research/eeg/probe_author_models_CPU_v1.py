"""CPU-only author-model preflight; this is not training or image reproduction."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extra-python-path", type=Path)
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Run with GPUs hidden"
    assert not args.output.exists(), "Preserve previous receipts"
    if args.extra_python_path:
        sys.path.insert(0, str(args.extra_python_path))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    start = time.monotonic()
    receipt = {
        "status": "CPU_MODEL_PREFLIGHT_ONLY",
        "script_sha256": sha(Path(__file__)),
        "seed": 1729,
        "trained": False,
        "images_generated": 0,
        "weights_downloaded": False,
        "models": [],
        "failures": [],
    }
    try:
        import torch
        torch.set_num_threads(2)
        torch.set_num_interop_threads(1)
        torch.manual_seed(1729)
        assert not torch.cuda.is_initialized()
        receipt["torch_version"] = torch.__version__
        enigma_root = args.root / "ENIGMA_460af4b6"
        manifest = json.loads((enigma_root / "source_manifest.json").read_text())
        assert manifest["commit"] == "460af4b6a0ca6eee11aeaf35c5431e8e3d55c11a"
        for item in manifest["files"]:
            assert sha(enigma_root / item["path"]) == item["sha256"]
        mod = load_module(enigma_root / "source/models.py", "author_enigma_cpu")
        subjects = [f"sub-{i:02d}" for i in range(1, 11)]
        for selected in [subjects[:1], subjects]:
            model = mod.ENIGMA(63, 250, subjects=selected, retrieval_only=False).eval()
            eeg = torch.randn(len(selected), 63, 250)
            with torch.inference_mode():
                output = model(eeg, selected)
            assert output.shape == (len(selected), 1024)
            assert torch.isfinite(output).all()
            receipt["models"].append({
                "class": "ENIGMA",
                "source_commit": manifest["commit"],
                "source_sha256": sha(enigma_root / "source/models.py"),
                "subjects": selected,
                "parameters": sum(p.numel() for p in model.parameters()),
                "input_shape": list(eeg.shape),
                "output_shape": list(output.shape),
                "finite": True,
                "checkpoint": None,
                "interpretation": "Randomly initialized author model on synthetic input; no quality evidence",
            })
        atm_root = args.root / "ATM_main_49cb3a20"
        manifest = json.loads((atm_root / "source_manifest.json").read_text())
        item = next(x for x in manifest["files"] if x["author_path"] == "Generation/diffusion_prior.py")
        path = atm_root / "Generation/diffusion_prior.py"
        assert sha(path) == item["export_sha256"]
        mod = load_module(path, "author_atm_prior_cpu")
        model = mod.DiffusionPriorUNet(cond_dim=1024, dropout=0.1).eval()
        with torch.inference_mode():
            out = model(torch.randn(10, 1024), torch.arange(10), torch.randn(10, 1024))
        assert out.shape == (10, 1024) and torch.isfinite(out).all()
        receipt["models"].append({
            "class": "DiffusionPriorUNet",
            "source_commit": manifest["commit"],
            "source_sha256": sha(path),
            "parameters": sum(p.numel() for p in model.parameters()),
            "input_shapes": {"x": [10, 1024], "t": [10], "c": [10, 1024]},
            "output_shape": list(out.shape),
            "finite": True,
            "checkpoint": None,
            "interpretation": "Random initialization and CPU forward only; no prior training or image generation",
        })
        assert not torch.cuda.is_initialized()
        receipt["cuda_initialized"] = False
        receipt["status"] = "PASSED_CPU_AUTHOR_MODEL_IMPORT_AND_FORWARD_ONLY"
    except Exception:
        receipt["status"] = "FAILED_CPU_PREFLIGHT_NOT_REPRODUCED"
        receipt["failures"].append(traceback.format_exc())
    receipt["seconds"] = time.monotonic() - start
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))
    return 0 if not receipt["failures"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
