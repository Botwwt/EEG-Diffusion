"""Check the learned photograph matching on two held-out, session-balanced EEG half-views."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from paper_render_cache import PinnedFiles, atomic_file, require, save_json
from paper_repeat_semantics import ARMS, split_trials
from paper_repeat_semantics_eval import read_checkpoint
from paper_token_evidence import development


def matching_scores(left, right, panels):
    q, v = F.normalize(left[panels], dim=-1), F.normalize(right[panels], dim=-1)
    scores = q @ v.transpose(1, 2)
    correct = scores.diagonal(dim1=1, dim2=2)
    delta = correct[:, :, None] + correct[:, None, :] - scores - scores.transpose(1, 2)
    i, j = torch.triu_indices(scores.shape[1], scores.shape[2], offset=1, device=scores.device)
    photo = torch.arange(scores.shape[1], device=scores.device)
    photo_hit = .5 * ((scores.argmax(1) == photo).float() + (scores.argmax(2) == photo).float())
    return delta[:, i, j], photo_hit


@torch.inference_mode()
def run(sessions_root):
    from common import PREP, ROOT
    from paper_instance_residual import make_encoder
    from paper_oof_conditions import registered_rows, study_root
    from paper_render import physical_eeg

    root, files = study_root(), PinnedFiles()
    plan, rows = development(PREP, files)
    train, _, _ = registered_rows(PREP)
    photo_file = PREP / "F_train.npy"
    label_file = PREP / "train_concept.npy"
    proof = dict(code=files.pin(Path(__file__)), training_photo=files.pin(photo_file), labels=files.pin(label_file),
                 split=files.pin(PREP / "paper_split.json"), dev=files.pin(PREP / "paper_development.json"), people={})
    train_f = np.load(photo_file, mmap_mode="r", allow_pickle=False)[train]
    center = torch.from_numpy(train_f).cuda().float().mean(0)
    concepts = np.load(label_file, allow_pickle=False)[rows]
    _, labels, sizes = np.unique(concepts, return_inverse=True, return_counts=True)
    require(len(sizes) == 20 and np.all(sizes == 10), "Expected ten photographs of each held-out object")
    panels = torch.from_numpy(labels.argsort(kind="stable").reshape(20, 10)).cuda()
    comparisons, saved = {}, {}
    for subject in plan["subjects"]:
        raw, receipt = physical_eeg(subject, "train", files)
        sessions_path = sessions_root / f"sub-{subject:02d}_train.npy"
        dig = files.pin(sessions_path)
        raw_definition = json.loads((PREP / f"sub-{subject:02d}_train.json").read_bytes())
        require(dig == raw_definition["session_sha256"], "EEG acquisition session order differs")
        session = np.load(sessions_path, allow_pickle=False)[rows]
        order = np.sort(session, axis=1)
        panel_order = order[panels.cpu().numpy()]
        require(np.all(order[:, [0, 2]] == order[:, [1, 3]]) and np.all(order[:, 0] != order[:, 2])
                and np.all(panel_order == panel_order[:, :1]), "Unmatched physical repetitions")
        session = torch.from_numpy(session.astype(np.int64)).cuda()
        eeg = torch.from_numpy(raw[rows].copy()).cuda().float()
        row = torch.arange(len(eeg), device="cuda")[:, None, None]
        views = []
        for orientation in (0, 1):
            flips = torch.tensor([[0, orientation]], device="cuda").expand(len(eeg), -1)
            views.append(eeg[row, split_trials(session, flips)].mean(2).unbind(1))
        path = ROOT / "fits" / f"personal-{subject:02d}/enc/native/sub-{subject:02d}/fold0_seed2025/brain_model.pt"
        native = files.pin(path)
        proof["people"][str(subject)] = dict(eeg=receipt, native=native, sessions=dig, weights={})
        model = make_encoder(torch.device("cuda")).float().eval()
        scores = {}
        for arm in ("native-paper", *ARMS):
            path_to_load = path
            if arm != "native-paper":
                path_to_load, source = read_checkpoint(root, subject, arm, files, native)
                require(source["training_inputs"]["eeg"] == receipt and source["training_inputs"]["sessions"] == dig
                        and source["training_inputs"]["split"] == proof["split"],
                        "Pair comparison used a different recording or split")
                proof["people"][str(subject)]["weights"][arm] = source["weights"]
            model.load_state_dict(torch.load(path_to_load, map_location="cpu", weights_only=True))
            margins, hits = [], []
            for left, right in views:
                latent = torch.cat([model.backbone(x).float() - center for x in torch.cat((left, right)).split(20)])
                margin, hit = matching_scores(*latent.chunk(2), panels)
                margins.append(margin)
                hits.append(hit)
            margin = torch.stack(margins).mean(0)
            hits = torch.stack(hits).mean(0)
            pair_credit = (margin > 0).float() + .5 * (margin == 0)
            scores[arm] = dict(pairs=int(margin.numel()), pair_wins=pair_credit.sum().item(),
                               mean_margin=margin.mean().item(), photo_queries=int(hits.numel()),
                               within_photo_top1=hits.sum().item())
            saved[f"sub{subject}_{arm}_margin"] = margin.cpu().numpy()
            saved[f"sub{subject}_{arm}_photo_credit"] = hits.cpu().numpy()
        comparisons[str(subject)] = scores
    files.verify()
    output = root / "development-physical-pairs.npz"
    atomic_file(output, lambda f: np.savez_compressed(f, **saved))
    save_json(root / "development-physical-pairs.json", dict(comparisons=comparisons, proof=proof,
                                                             observations_sha256=files.pin(output)))
    print(json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions-root", type=Path, required=True)
    run(parser.parse_args().sessions_root)
