"""Check historical CBraMod batching against the current official condition cache."""
from pathlib import Path
import json
import os

import numpy as np
import torch

from common import PREP, ROOT
from paper_cbramod_repeats import PhotoCondition, measured_trials, upstream_model
from paper_condition_probe import load_projection
from paper_render import physical_eeg
from paper_render_cache import PinnedFiles, require, save_json
from paper_token_evidence import project_tokens

@torch.inference_mode()
def main():
    store = Path(os.environ["EEG_CROSSFIT_ROOT"])
    output = {}
    adapter, project = load_projection()
    rows = np.asarray(json.loads((PREP / "paper_split.json").read_bytes())["official_rows"])
    for subject in (6, 9):
        pins = PinnedFiles()
        path = store / "probe" / f"conditions-sub-{subject:02d}.pt"
        data = torch.load(path, map_location="cpu", weights_only=True)
        cdir = ROOT / "cbramod_repeats"
        model_path = cdir / "fit" / f"sub-{subject:02d}" / "joint" / "model.pt"
        require(data["proof"]["sources"]["joint"]["model_sha256"] == pins.pin(model_path),
                "Previous and current prediction do not use the same checkpoint")
        raw, proof = physical_eeg(subject, "test", pins)
        require(proof == data["proof"]["official"], "Different test EEG during the replay")
        weights = torch.load(model_path, map_location="cpu", weights_only=True)["state"]
        args = [weights[k] for k in ("input_center", "input_scale", "photo_center", "photo_scale")]
        model = PhotoCondition(upstream_model(cdir / "upstream", False, pins), "joint", *args)
        model.load_state_dict(weights, strict=True)
        model.cuda().eval().requires_grad_(False)
        trials = measured_trials(raw, rows)
        original = torch.cat([model(batch) for batch in trials.split(32)]).cpu()
        cached = data["models"]["joint"]["query_fused"]
        old_ip = project_tokens(original, adapter, project).cpu().half()
        new_ip = data["models"]["joint"]["query_ip"]
        pins.verify()
        output[subject] = dict(fused_max_abs=float((original - cached).abs().max()),
                               ip_half_max_abs=float((old_ip.float() - new_ip.float()).abs().max()),
                               ip_half_unequal=int((old_ip != new_ip).sum()),
                               ip_unequal_queries=int((old_ip != new_ip).flatten(1).any(1).sum()),
                               n_ip_coordinates=old_ip.numel(), probe_sha256=pins.pin(path),
                               checkpoint_sha256=pins.pin(model_path))
    save_json(store / "probe" / "prior-batches.json", output)
    print(output, flush=True)

if __name__ == '__main__':
    main()
