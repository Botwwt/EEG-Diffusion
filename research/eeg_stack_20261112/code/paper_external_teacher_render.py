"""Render the matched external EEG training comparison with one person's real EEG."""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from paper_external_teacher import CANDIDATE, MODES
from paper_render_cache import PinnedFiles, require
from paper_repeat_semantics_eval import decode
from paper_token_evidence import development, project_tokens, render_arms, score_outputs


@torch.inference_mode()
def render(person, panel):
    from common import PREP
    from paper_identity_render import load_generator
    from paper_oof_conditions import study_root
    from paper_render import condition, donor_rows, draw_images, generator_inputs, groups, physical_eeg, query_seed

    root, files = study_root(), PinnedFiles()
    split_path = PREP / "paper_split.json"
    split = files.pin(split_path)
    dev_hash = None
    if panel == "dev":
        plan, rows = development(PREP, files)
        require(person in plan["subjects"], "Expected a registered development participant")
        dev_hash = files.pin(PREP / "paper_development.json")
    elif panel == "fresh":
        plan = json.loads(split_path.read_bytes())
        rows = plan["fresh_rows"]
        require(len(set(rows)) == len(rows) == 500 and not set(rows).intersection(plan["train_rows"]),
                "Fresh panel differs from the training split")
    else:
        rows = json.loads(split_path.read_bytes())["official_rows"]
        require(rows == list(range(200)), "Official stimulus order differs")
    repeats = groups(f"{panel}-r4")
    require(repeats == [[0, 1, 2, 3]], "The two models must see the same four physical records")
    raw, receipt = physical_eeg(person, "test" if panel == "official" else "train", files)
    queries = raw[rows]
    native, base = condition("native-paper", person, queries, repeats, features_only=True, files=files)
    assets = generator_inputs(files)
    pipe = load_generator(torch.float16, False)
    _, adapter, projection, _, _ = pipe
    order = list(range(len(rows)))
    if panel in ("dev", "fresh"):
        label_file = PREP / "train_concept.npy"
        files.pin(label_file)
        wrong = donor_rows("token-reliable-wrong", panel, np.load(label_file, allow_pickle=False)[rows]).tolist()
    else:
        wrong = np.roll(order, len(rows) // 2).tolist()
    eeg = torch.from_numpy(queries[:, repeats[0]].copy()).cuda().float().mean(1)
    inputs = [("native-paper", project_tokens(native, adapter, projection).half().cpu(), dict(eeg_donor=order))]
    traces = {}
    for arm in MODES:
        latent, fit = decode(arm, root, person, eeg, files, base)
        require(fit["training_inputs"]["split"] == split, "Checkpoint used a different stimulus split")
        if panel != "official":
            require(fit["training_inputs"]["eeg"] == receipt, "Model and renderer use different real EEG")
        traces[arm] = fit
        token = project_tokens(latent, adapter, projection).half().cpu()
        inputs.append((arm, token, dict(eeg_donor=order, fitted_checkpoint=fit["weights"])))
        if arm == CANDIDATE:
            inputs.append((arm + "-wrong", token[wrong], dict(eeg_donor=wrong, fitted_checkpoint=fit["weights"])))
    require(len({item["schedule"] for item in traces.values()}) == 1
            and all(item["training_inputs"] == traces[MODES[0]]["training_inputs"] for item in traces.values()),
            "Matched arms use different training inputs or schedules")
    shared = dict(budget=f"{panel}-r4", subject=person, repeats=repeats, split_sha256=split,
                  development_split_sha256=dev_hash, physical_eeg=receipt, native_checkpoint_sha256=base,
                  generator_files_sha256=assets, code_sha256=files.pin(Path(__file__)),
                  render_source_sha256=files.pin(Path(inspect.getfile(draw_images))),
                  training_teacher_sha256=traces[CANDIDATE]["training_inputs"]["teachers"],
                  training_schedule_sha256=traces[CANDIDATE]["schedule"])
    output = render_arms(root / "images" / panel / f"sub-{person:02d}", inputs, rows,
                         [query_seed(panel, i) for i in order], shared, files, pipe)
    del pipe, adapter, projection
    torch.cuda.empty_cache()
    score_outputs(output, panel)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("subject", type=int, choices=range(1, 11))
    cli.add_argument("--panel", choices=("dev", "official", "fresh"), default="dev")
    args = cli.parse_args()
    render(args.subject, args.panel)
