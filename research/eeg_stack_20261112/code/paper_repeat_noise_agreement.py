"""Check if two physical EEG subsets agree when the open-vocabulary readout is right."""
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import cosine_similarity, normalize

from paper_object_foreground import root, text_targets
from paper_object_foreground_render import language_scorer, split_input, targets_for_budget
from paper_instance_residual import make_encoder
from paper_render_cache import PinnedFiles, require, save_json
from paper_repeat_noise_readout import PHOTO_PENALTY, eeg_projection, readout_path, semantic_weights, standardized
from score_official_objects import topk_credit


@torch.inference_mode()
def run():
    archive = Path(os.environ["ARCHIVE"])
    basis_files = PinnedFiles()
    _, _, _, _, states, basis = text_targets(basis_files)
    teacher = language_scorer(archive, basis_files)
    subjects = (1, 3, 5, 6)
    paired_result = {}
    for subject in subjects:
        files = PinnedFiles()
        personal = Path(os.environ["RAD_ROOT"]) / "fits" / f"personal-{subject:02d}"
        checkpoint = personal / f"enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
        model = make_encoder("cuda").float().eval()
        model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu"))
        native = files.pin(checkpoint)
        source = readout_path(subject, root())
        digest = files.pin(source)
        saved = torch.load(source, weights_only=True, map_location="cpu")
        require(saved["proof"]["native_checkpoint_sha256"] == native and saved["proof"]["semantic_basis_sha256"] == basis,
                "Expected pinned personal weights and semantic target")
        fit = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in saved["state"].items()}
        raw, physics, rows = split_input(subject, "official-r80", files)
        readout_file = root() / "images/official-r80" / f"sub-{subject:02d}/semantic-readout.pt"
        data = torch.load(readout_file, weights_only=True, map_location="cpu")
        require(data["proof"]["row"] == rows and data["proof"]["readout_sha256"] == digest,
                "Held split and full readout correspond to different source EEG")
        w = semantic_weights(fit["noise"], fit["photo"], fit["cross"], 80, PHOTO_PENALTY) @ states["pooled_basis"].cuda()
        vectors = []
        for start in range(0, len(rows), 20):
            trials = torch.from_numpy(np.array(raw[start:start + 20], copy=True)).cuda().float()
            halves = torch.stack((trials[:, ::2].mean(1), trials[:, 1::2].mean(1)), dim=1)
            vectors.append(standardized(eeg_projection(halves, model.backbone.proj_in), fit) @ w)
        a, b = torch.cat(vectors).unbind(dim=1)
        agreement = cosine_similarity(a, b).cpu().numpy()
        wrong = cosine_similarity(a, b.roll(len(rows) // 2, dims=0)).cpu().numpy()
        labels, ref = targets_for_budget("official-r80", rows, teacher, files)
        true = np.array([teacher.reverse[name] for name in labels])
        estimate = data["pool"]
        samples = np.arange(len(rows))
        table = []
        for quartile in np.array_split(np.argsort(agreement, kind="stable"), 4):
            overall = dict(size=len(quartile), mean_agreement=float(agreement[quartile].mean()), arms={})
            for name, value in estimate.items():
                scores = (normalize(value.cuda()) @ teacher.anchors.T).cpu().numpy()[:, teacher.inverse]
                outcome = {f"top{k}": float(topk_credit(scores, k)[samples, true][quartile].sum()) for k in (1, 5)}
                overall["arms"][name] = outcome
            table.append(overall)
        paired_result[subject] = dict(mean_agreement=float(agreement.mean()), mean_different=float(wrong.mean()), table=table,
            provenance=dict(readout=files.pin(readout_file), checkpoint=digest, source_eeg=physics, reference=ref))
        files.verify()
    basis_files.verify()
    save_json(root() / "held-agreement.json", dict(subjects=subjects, results=paired_result, word_scoring=teacher.source))
    for quartile in range(4):
        print(quartile + 1, {arm: {key: sum(s["table"][quartile]["arms"][arm][key] for s in paired_result.values())
                                      for key in ("top1", "top5")} for arm in paired_result[subjects[0]]["table"][quartile]["arms"]}, flush=True)


if __name__ == "__main__":
    run()
