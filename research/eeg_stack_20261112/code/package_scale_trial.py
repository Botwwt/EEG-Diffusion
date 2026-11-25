"""Package an independently gated 1,024-object trial fit for matched personal12 image generation."""

import argparse
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from fit_fresh_photo_panel import fisher
from fit_personal_session_metric import similarities
from fit_scale_trial import material
from fit_trial_object_encoder import TrialObjectEncoder
from fit_within_photo_embedding import H14_SHA, bases, images, predict, training
from package_full_head_512 import directions
from package_trial_object_confirmation import PANEL, SUB, read_trials
from prepare_expanded_personal import extract
from prepare_scale_trial import DESIGN, PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha
from render_within_photo_ip import packet
from run_prospective_readout import frozen_npz
from safetensors.torch import load_file


def package(a):
    # This check precedes any target EEG access.
    _, _, _, train_ids, _ = material(a, 1024)
    folder = a.out / "objects1024_steps1500"
    model_file = folder / "model.safetensors"
    trained = read(folder / "receipt.json")
    check(trained["design_sha256"] == sha(DESIGN)
          and trained["driver_sha256"] == sha(Path(__file__).with_name("fit_scale_trial.py"))
          and trained["training_objects"] == train_ids.tolist() and trained["updates"] == 1500
          and trained["files"]["model.safetensors"] == sha(model_file), "Final fit changed")
    base = a.data / "object-head-confirmation"
    historic = a.data / "trial-object-encoder/personal12"
    legacy = read(PUBLIC / "trial_object_personal12.json")
    regfile = base / "references/personal12_sub-04_ablation.json"
    old = historic / "render_inputs"
    old_packet = old / "conditions" / SUB / PANEL / "conditions_512.npz"
    check(sha(regfile) == legacy["registration_sha256"]
          and sha(old_packet) == legacy["mixed_packet_sha256"]
          and sha(historic / "decisions.npz") == legacy["raw_results_sha256"]
          and sha(old / "design.json") == legacy["render_design_sha256"]
          and sha(old / "radius.json") == legacy["radius_sha256"], "Personal baseline changed")
    reg = read(regfile)
    rows, folds, vocab, cal = (np.array(reg[k]) for k in
                               ("rows", "folds", "words", "calibration_rows_by_fold_and_word"))
    source_rows = np.sort(rows)
    ids = np.unique(source_rows // 10)
    check(len(ids) == 12 and not np.intersect1d(ids, train_ids).size
          and np.array_equal(source_rows, (ids[:, None] * 10 + np.arange(10)).ravel()), "Target leakage")
    eegfile = base / "eeg/sub-04.npz"
    receipt = base / "eeg/sub-04.json"
    check(sha(receipt) == read(PUBLIC / "trial_object_personal12_design.json")["inputs"]["sealed_eeg_receipt"]
          and sha(eegfile) == read(receipt)["packet_sha256"], "Physical target EEG changed")
    with np.load(eegfile, allow_pickle=False) as eeg:
        trials = read_trials(base, a.data, ids, eeg)
        physical = extract(eeg["half_eeg"].mean(0)).astype(float)
    state = load_file(model_file, device=a.device)
    mu, sd = state.pop("input_mean"), state.pop("input_scale")
    model = TrialObjectEncoder().to(a.device)
    model.load_state_dict(state)
    model.eval()
    with torch.inference_mode():
        deep = model(((torch.from_numpy(trials).to(a.device) - mu) / sd).clamp(-12, 12)).cpu().numpy()
    bank = images(a.data / "official-object-readout/sub-03")
    vis_train, x, y = training(SimpleNamespace(expanded=a.data / "photo-structure/expanded"), SUB, 512, bank)
    check(np.isin(vis_train, train_ids).all() and len(set(vis_train)) == 512, "Wrong visual training")
    native = fisher(x, physical)
    scores = np.empty((2, 120, 12))
    for f in range(5):
        ix = np.flatnonzero(folds == f)
        qi = np.searchsorted(source_rows, rows[ix])
        si = np.searchsorted(source_rows, cal[f])
        check(set(cal[f].ravel()) == set(rows[folds != f]) and np.array_equal(source_rows[si], cal[f]),
              "Calibration leaks into query fold")
        scores[:, ix, :] = [similarities(z[qi], z[si]) for z in (native, deep.astype(float))]
    with np.load(historic / "decisions.npz", allow_pickle=False) as orig:
        baseline = orig["mixed"]
        check(np.array_equal(orig["rows"], rows) and np.array_equal(orig["folds"], folds)
              and np.array_equal(orig["words"], vocab) and np.allclose(scores[0], orig["standardized_scores"][0],
                                                                       rtol=0, atol=2e-6),
              "Fixed Fisher branch or decisions changed")
    mixed = scores.mean(0).argmax(1)
    within = a.data / "photo-structure/within_photo/selection.json"
    angle = a.data / "photo-structure/residual_angle/scale.json"
    fitted = bases(x, y, physical)
    residual = predict(fitted, read(within)["selected"]) @ fitted[2].T
    scale = read(angle)
    calculated = directions(rows, folds, vocab, cal, mixed, residual, source_rows, bank, scale)
    baseline_conditions = directions(rows, folds, vocab, cal, baseline, residual, source_rows, bank, scale)
    with np.load(old_packet, allow_pickle=False) as z:
        check(all(np.array_equal(baseline_conditions[k], z[k]) for k in
                  ("rows", "folds", "predicted_words", "support_rows", "donor_rows")),
              "Original condition routing differs")
        err = max(float(np.abs(baseline_conditions[k] - z[k]).max()) for k in ("own", "uniform", "wrong"))
        check(err < 2e-6 and np.allclose(calculated["own"][mixed == baseline], z["own"][mixed == baseline],
                                         rtol=0, atol=2e-6), "Changed H14 direction on unchanged word")
    dest = a.out / "personal12/render_inputs"
    word = dest / "evaluations" / SUB / PANEL / "word_predictions.json"
    freeze(word, dict(subject=SUB, panel=PANEL, physical_trials_per_photo=4,
                      rows=rows.tolist(), folds=folds.tolist(), words=vocab.tolist(),
                      predicted=mixed.tolist(), registration_sha256=sha(regfile), protocol_sha256=sha(DESIGN)))
    out = dest / "conditions" / SUB / PANEL / "conditions_512.npz"
    frozen_npz(out, calculated | dict(subject=SUB, physical_trials_per_photo=4, training_classes=vis_train,
               visual_sha256=H14_SHA, target_eeg_sha256=sha(eegfile), word_prediction_sha256=sha(word),
               registration_sha256=sha(regfile), selection_sha256=sha(within), angular_scale_sha256=sha(angle),
               trial_readout_design_sha256=sha(DESIGN), trial_training_objects=1024,
               trial_weights_sha256=sha(model_file)))
    freeze(dest / "design.json", read(old / "design.json") | dict(protocol_sha256=sha(DESIGN),
           packets_sha256={f"{SUB}/{PANEL}":sha(out)}, registration_sha256={f"{SUB}/{PANEL}":sha(regfile)}))
    shutil.copyfile(old / "radius.json", dest / "radius.json")
    packet(dest, SUB, PANEL)
    decisions = dest.parent / "decisions.npz"
    frozen_npz(decisions, dict(rows=rows, folds=folds, words=vocab, mixed=mixed, baseline=baseline,
                               fisher=scores[0].argmax(1), encoder=scores[1].argmax(1), standardized_scores=scores))
    # Word choices and the generation packet were frozen before scoring against labels.
    truth = np.array(reg["truth"])
    old_hits, new_hits = baseline == truth, mixed == truth
    effect = new_hits.astype(float) - old_hits
    grouped = np.array([effect[truth == j].mean() for j in range(12)])
    draws = np.random.default_rng(260552).integers(12, size=(10000, 12))
    summary = dict(design_sha256=sha(DESIGN), training_receipt_sha256=sha(folder / "receipt.json"),
                   weights_sha256=sha(model_file), registration_sha256=sha(regfile), participant=SUB,
                   n=120, objects=12, presentations=4, encoder_training_objects=1024, visual_training_objects=512,
                   original_mixed_hits=int(old_hits.sum()), new_mixed_hits=int(new_hits.sum()),
                   gains=int((new_hits & ~old_hits).sum()), losses=int((old_hits & ~new_hits).sum()),
                   delta_pp=float(100 * grouped.mean()),
                   ci95_pp=(100 * np.quantile(grouped[draws].mean(1), [.025, .975])).tolist(),
                   original_h14_max_error=err, unchanged_words=int((mixed == baseline).sum()),
                   packet_sha256=sha(out), radius_sha256=sha(dest / "radius.json"),
                   render_design_sha256=sha(dest / "design.json"), word_sha256=sha(word),
                   decisions_sha256=sha(decisions))
    freeze(dest.parent / "receipt.json", summary)
    freeze(PUBLIC / "scale_trial_personal12.json", summary)
    print(summary, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data", type=Path, default=Path("outputs"))
    p.add_argument("--out", type=Path, default=Path("outputs/scale-trial"))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    package(p.parse_args())
