"""Test a direct H/14 visual bridge at the generator's Value projections."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from paper_horizon_readout import VIEWS, registration
from paper_joint_condition_diagnostic import ReplacedImageProjection, RoutedUNet
from paper_raw_semantic_readout import leave_photo_out
from paper_render_cache import PinnedFiles, render_transaction, require, save_json
from paper_token_evidence import project_tokens, score_outputs, token_parts


def blend_relative(native, visual, alpha):
    common, initial = token_parts(native.float())
    _, proposed = token_parts(visual.float())
    return (common + torch.lerp(initial, proposed, alpha)).half()


def match_relative_norm(candidate, reference):
    common, relative = token_parts(candidate.float())
    _, target = token_parts(reference.float())
    norm = relative.flatten(1).norm(dim=-1)
    expected = target.flatten(1).norm(dim=-1)
    require(bool((norm > 1e-8).all()) and bool(torch.isfinite(expected).all()), "Incomplete relative token norms")
    return common + relative * (expected / norm).view(-1, 1, 1)


def native_h14(folder, subject, held, pins):
    path = folder / f"sub-{subject:02d}.h14.npz"
    meta = folder / f"sub-{subject:02d}.json"
    source = json.loads(meta.read_bytes())
    require(pins.pin(path) == source["h14_sha256"] and source["proof"]["views"] == list(VIEWS)
            and source["proof"]["subject"] == subject and source["proof"]["repeats"] == list(range(4)),
            "Frozen H/14 predictions differ from their physical EEG receipt")
    with np.load(path, allow_pickle=False) as payload:
        require(np.array_equal(payload["row"], held) and payload["embedding"].shape == (len(held), 4, 1024), "Frozen H/14 row order changed")
        estimate = torch.from_numpy(payload["embedding"][:200, 0].copy())
    return estimate, {k: source["proof"][k] for k in ("source", "eeg", "registration")}, pins.pin(meta)


def acquisition_h14(folder, subject, rows, registry, physical, pins):
    payload = folder / f"sub-{subject:02d}.npz"
    receipt = payload.with_suffix(".json")
    source = json.loads(receipt.read_bytes())
    require(source["subject"] == subject and source["rows"] == list(rows)
            and source["splits"] == {k: registry[k] for k in ("split", "development", "labels")}
            and source["eeg"] == {"array": physical["array_sha256"], "manifest": physical["receipt_sha256"]}
            and source["repeats"] == [0, 1, 2, 3] and source["predictions"] == pins.pin(payload),
            "Cross-acquisition predictions differ from the same held-out physical queries")
    with np.load(payload, allow_pickle=False) as values:
        require(np.array_equal(values["row"], rows), "Cross-acquisition query order changed")
        result = {arm: torch.from_numpy(values[arm].copy()) for arm in ("eeg1", "eeg2")}
    require(all(x.shape == (len(rows), 1024) and torch.isfinite(x).all() for x in result.values()),
            "Cross-acquisition H/14 conditions are incomplete")
    return result, pins.pin(receipt)


def cross_photo_h14(folder, subject, rows, registry, physical, pins):
    from paper_cross_photo_layout import ARMS, CONTEXT_ARMS

    archive = folder / f"sub-{subject:02d}.npz"
    manifest = archive.with_suffix(".json")
    source = json.loads(manifest.read_bytes())
    contexts = source["eeg_contexts"]
    arms = {1: ARMS, 2: CONTEXT_ARMS}.get(contexts)
    require(source["subject"] == subject and source["rows"] == list(rows) and source["repeats"] == [0, 1, 2, 3]
            and source["splits"] == {k: registry[k] for k in ("split", "development", "labels")}
            and source["recording_source_sha256"] == physical["source_sha256_as_recorded"]
            and source["predictions_sha256"] == pins.pin(archive)
            and arms is not None and set(source["arm_receipts"]) == set(arms),
            "Cross-photo readouts differ from the registered physical inputs")
    with np.load(archive, allow_pickle=False) as values:
        require(np.array_equal(values["row"], rows), "Cross-photo query order changed")
        result = {arm: torch.from_numpy(values[arm].copy()).float() for arm in arms}
    require(all(x.shape == (len(rows), 1024) and torch.isfinite(x).all() for x in result.values()),
            "Cross-photo H/14 conditions are incomplete")
    return result, pins.pin(manifest), arms[-1]


def region_readout_h14(folder, subject, rows, registry, physical, pins):
    archive = folder / f"sub-{subject:02d}.npz"
    manifest = archive.with_suffix(".json")
    source = json.loads(manifest.read_bytes())
    arms = ("whole", "foreground", "background")
    require(source["subject"] == subject and source["registration"] == registry
            and source["rows"] == list(rows) and source["repeats"] == [0, 1, 2, 3]
            and source["source_recording_sha256"] == physical["source_sha256_as_recorded"]
            and source["matching_electrodes"] == 17 and source["predictions_sha256"] == pins.pin(archive)
            and source["scope"] == "eeg-only" and set(source["fit_receipts"]) == set(arms),
            "Region readouts differ from the registered physical queries")
    with np.load(archive, allow_pickle=False) as values:
        require(np.array_equal(values["row"], rows), "Region readout order changed")
        result = {arm: torch.from_numpy(values[arm].copy()).float() for arm in arms}
    require(all(x.shape == (len(rows), 1024) and torch.isfinite(x).all() for x in result.values()),
            "Region readouts are incomplete")
    return result, pins.pin(manifest)


def layout_response_h14(folder, subject, rows, registry, physical, pins,
                        arms=("real-control", "replace-layout", "reverse-layout")):
    archive = folder / f"sub-{subject:02d}.npz"
    manifest = archive.with_suffix(".json")
    source = json.loads(manifest.read_bytes())
    require(source["subject"] == subject and source["rows"] == list(rows)
            and source["repeats"] == [0, 1, 2, 3]
            and source["splits"] == {key: registry[key] for key in ("split", "development", "labels")}
            and source["physical"] == {key: physical[key] for key in ("array_sha256", "receipt_sha256")}
            and source["predictions_sha256"] == pins.pin(archive)
            and set(source["models"]) == set(arms), "Auxiliary heads differ from the same physical queries")
    with np.load(archive, allow_pickle=False) as predictions:
        require(np.array_equal(predictions["row"], rows), "Layout-response predictions changed order")
        result = {name: torch.from_numpy(predictions[name].copy()) for name in arms}
    require(all(v.shape == (len(rows), 1024) and torch.isfinite(v).all() for v in result.values()),
            "Incomplete EEG predictions for the layout-response comparison")
    return result, pins.pin(manifest)


def scalp_conditions(folder, subject, rows, registry, physical, predicted, pins):
    archive = folder / f"sub-{subject:02d}.npz"
    manifest = archive.with_suffix(".json")
    proof = json.loads(manifest.read_bytes())
    arms = ("full", "full-wrong", "reliable-residual", "reliable-residual-wrong")
    native = predicted.float().contiguous()
    require(proof["subject"] == subject and proof["rows"] == rows and proof["repeats"] == list(range(4))
            and proof["predictions_sha256"] == pins.pin(archive)
            and proof["splits"] == {key: registry[key] for key in ("split", "development", "labels")}
            and proof["full_scalp_sha256"] == physical["source_sha256_as_recorded"]
            and proof["posterior_sha256"] == physical["array_sha256"] and proof["blend"] == .25
            and proof["native_h14_sha256"] == sha256(native.numpy().tobytes()).hexdigest(),
            "Full-scalp conditions must use the same single query and verified physical recording")
    with np.load(archive, allow_pickle=False) as data:
        donors = data["wrong"].copy()
        require(data["row"].tolist() == rows and np.array_equal(donors, np.roll(np.arange(len(rows)), 10)),
                "Full-scalp query or wrong EEG order differs")
        directions = {"native": F.normalize(native)}
        directions.update({arm: torch.from_numpy(data[arm].copy()) for arm in arms})
    require(all(value.shape == (len(rows), 1024) and torch.isfinite(value).all() and
                torch.allclose(value.norm(dim=1), torch.ones(len(rows)), atol=1e-5, rtol=0.)
                for value in directions.values()), "Incomplete normalized full-scalp conditions")
    return directions, pins.pin(manifest), donors


@torch.inference_mode()
def run(subject, folder, output, full_replacement=False, acquisition=None, cross_photo=None, region_reference=None,
        region_readout=None, whole_readout_timing=False, spatial_value=False, layout_response=None, categorical=None,
        scalp=None):
    from common import EMB_KEYS, PREP, PRIOR, RUNS
    from main.models_adapter import FusionEncoderModel
    from paper_eeg_attention import image_layers
    from paper_identity_render import load_generator
    from paper_photo_top1 import DEFAULT_LAYOUT, render_inputs
    from paper_render import condition, donor_rows, draw_images, generator_inputs, layout, query_seed, runtime, tensor_fingerprint
    from paper_uniform_baselines import reference_layout

    require(RUNS.name == "shared", "Use the archived shared D2 checkpoint")
    require(not whole_readout_timing or region_readout is not None, "The timing experiment requires region readouts")
    pins = PinnedFiles()
    (train, dev, fresh), labels, registry = registration(PREP, pins)
    raw, rows, _, budget, repeats, _, split, study, physical = render_inputs(subject, "dev", pins)
    require(np.array_equal(rows, dev), "D2 development pool changed")
    predicted, h14_proof, receipt = native_h14(folder, subject, np.concatenate((dev, fresh)), pins)
    require(h14_proof["registration"] == registry and h14_proof["eeg"] == physical, "The H/14 head uses a different physical recording")
    weights = generator_inputs(pins)
    fitted, layout_proof = reference_layout(subject, split, pins)
    starts = layout(fitted, raw, repeats)
    pipe, adapter, projector, prompt, vae = load_generator(torch.float16, False)
    tokens, checkpoint, sources = condition("d2", subject, raw, repeats, adapter, projector, files=pins, return_source=True)
    require(sources == h14_proof["source"], "The instance and H/14 conditions use different encoders")

    reference_files = [PREP / "emb_train.pt"]
    additional = (acquisition, cross_photo, region_readout, layout_response, categorical, scalp)
    if not (full_replacement or spatial_value) and all(path is None for path in (*additional, region_reference)):
        reference_files.append(PREP / "F_train.npy")
    representations = {p.name: pins.pin(p) for p in reference_files}
    embeds = torch.load(reference_files[0], map_location="cpu", weights_only=True)
    h14, *remaining = EMB_KEYS
    style = {key: embeds[key][train].float().mean(0, keepdim=True).cuda() for key in remaining}
    norm = embeds[h14][train].float().norm(dim=1).median().cuda()
    teacher = FusionEncoderModel.from_pretrained(str(PRIOR), subfolder="fusion_encoder").vision_model.cuda().float().eval()

    def compose(h):
        parts = []
        for row in h.float().cuda().split(64):
            other = {key: val.expand(len(row), *val.shape[1:]) for key, val in style.items()}
            parts.append(teacher(**{h14: F.normalize(row) * norm, **other}))
        return project_tokens(torch.cat(parts), adapter, projector).cpu()

    wrong_rows = donor_rows("token-reliable-wrong", "dev", labels[dev])
    model_steps = pipe.scheduler.timesteps.tolist()
    require(model_steps == [999, 749, 499, 249], "Unexpected four-step generator")
    projections, spatial_projections = [], []
    for name, (processor, attr) in image_layers(pipe.unet).items():
        if attr == "to_v_ip":
            wrapped = ReplacedImageProjection(getattr(processor, attr))
            setattr(processor, attr, wrapped)
            projections.append(wrapped)
            if spatial_value:
                from paper_spatial_value import SpatialImageValue

                path = name.removesuffix(".processor.to_v_ip")
                site = pipe.unet.get_submodule(path)
                require(site.processor is processor, "Image Value projection belongs to a different attention site")
                spatial_projections.append(SpatialImageValue(site, wrapped.base))
    require(len(projections) == 70, "Unexpected frozen image Value registry")
    supplemental, bands = {}, {}
    if spatial_value:
        import paper_spatial_value

        visual = compose(predicted)
        correct = blend_relative(tokens, visual, 1.)
        wrong = blend_relative(tokens, visual[wrong_rows], 1.)
        late = model_steps[2:]
        configs = {"d2": (None, [], False, None)}
        for band in ("direct", "coarse", "detail"):
            for donor, source in (("", correct), ("-wrong", wrong)):
                arm = f"h14-late-{band}{donor}-values"
                configs[arm] = (source, late, False, 1.)
                if band != "direct":
                    bands[arm] = band
        supplemental = dict(spatial_operator=pins.pin(Path(paper_spatial_value.__file__)), spatial_cells=[2, 2])
    elif region_reference is not None:
        from paper_region_value_probe import load_views

        views, region_receipt = load_views(region_reference, "dev", dev, registry, pins)
        configs = {f"truth-{view}-values": (blend_relative(tokens, compose(h), 1.), model_steps, True, 1.)
                   for view, h in views.items()}
        representations.update(region_reference=pins.pin(region_reference), region_receipt=region_receipt)
        supplemental = dict(region_reference_receipt=region_receipt)
    elif scalp is not None:
        estimates, extra_receipt, wrong_rows = scalp_conditions(scalp, subject, rows, registry, physical, predicted, pins)
        require(np.all(labels[dev] != labels[dev][wrong_rows]), "Full-scalp wrong EEG includes the current object")
        norm_control = compose(estimates["native"])
        configs = {"d2": (None, [], False, None)}
        for arm, estimate in estimates.items():
            visual = norm_control if arm == "native" else match_relative_norm(compose(estimate), norm_control)
            configs[f"{arm}-scalp-values"] = (blend_relative(tokens, visual, .5), model_steps, False, .5)
        supplemental = dict(scalp_receipt=extra_receipt, scalp_weight=.25, relative_norm_reference="native-h14")
    elif any(path is not None for path in additional):
        native = F.normalize(predicted.float())
        timing = {}
        if acquisition is not None:
            estimates, extra_receipt = acquisition_h14(acquisition, subject, rows, registry, physical, pins)
            auxiliary = {"eeg1": estimates["eeg1"], "eeg2": estimates["eeg2"],
                         "eeg1-wrong": estimates["eeg1"][wrong_rows]}
            suffix = "acquisition"
            supplemental = dict(acquisition_receipt=extra_receipt, acquisition_weight=.25)
        elif cross_photo is not None:
            estimates, extra_receipt, target = cross_photo_h14(cross_photo, subject, rows, registry, physical, pins)
            auxiliary = {**estimates, f"{target}-wrong": estimates[target][wrong_rows]}
            suffix = "cross-photo"
            supplemental = dict(cross_photo_receipt=extra_receipt, cross_photo_weight=.25)
        elif layout_response is not None:
            estimates, extra_receipt = layout_response_h14(layout_response, subject, rows, registry, physical, pins)
            auxiliary = {**estimates, "replace-layout-wrong": estimates["replace-layout"][wrong_rows]}
            timing = dict.fromkeys(auxiliary, model_steps[2:])
            suffix = "layout-response"
            supplemental = dict(layout_response_receipt=extra_receipt, layout_response_weight=.25)
        elif categorical is not None:
            encoded = ("continuous", "posterior-mean", "posterior-mode")
            estimates, extra_receipt = layout_response_h14(categorical, subject, rows, registry, physical, pins, encoded)
            auxiliary = {**estimates, "posterior-mode-wrong": estimates["posterior-mode"][wrong_rows]}
            suffix = "categorical"
            supplemental = dict(categorical_receipt=extra_receipt, categorical_weight=.25, relative_norm_reference="native-h14")
        else:
            estimates, extra_receipt = region_readout_h14(region_readout, subject, rows, registry, physical, pins)
            auxiliary = {**estimates, "foreground-wrong": estimates["foreground"][wrong_rows]}
            suffix = "region-readout"
            supplemental = dict(region_readout_receipt=extra_receipt, region_readout_weight=.25)
            if whole_readout_timing:
                whole = estimates["whole"]
                wrong_whole = whole[wrong_rows]
                auxiliary = {"whole-early": whole, "whole-late": whole, "whole-late-wrong": wrong_whole,
                             "whole-all-wrong": wrong_whole}
                timing = {"whole-early": model_steps[:2], "whole-late": model_steps[2:], "whole-late-wrong": model_steps[2:]}
                supplemental["whole_readout_timing"] = True
        configs = {} if region_readout is not None else {"d2": (None, [], False, None)}
        if categorical is not None:
            norm_control = compose(native)
            configs["native-half-categorical-values"] = (blend_relative(tokens, norm_control, .5), model_steps, False, .5)
        for arm, estimate in auxiliary.items():
            mixed = .75 * native + .25 * F.normalize(estimate.float())
            visual = compose(mixed)
            if categorical is not None:
                visual = match_relative_norm(visual, norm_control)
            values = blend_relative(tokens, visual, .5)
            configs[f"{arm}-{suffix}-values"] = (values, timing.get(arm, model_steps), False, .5)
    else:
        photo_h14 = embeds[h14][dev].float()
        eeg = compose(predicted)
        mix = 1. if full_replacement else .5
        correct = blend_relative(tokens, eeg, mix)
        wrong = blend_relative(tokens, eeg[wrong_rows], mix)
        oracle_photo = blend_relative(tokens, compose(photo_h14), .5 if full_replacement else 1.)
        if full_replacement:
            configs = {
                "truth-h14-half-values": (oracle_photo, model_steps, True, .5),
                "eeg-h14-full-values": (correct, model_steps, False, 1.),
                "eeg-h14-full-wrong-values": (wrong, model_steps, False, 1.),
            }
        else:
            features = np.load(reference_files[1], mmap_mode="r", allow_pickle=False)
            whole = project_tokens(torch.from_numpy(features[dev].copy()), adapter, projector).cpu()
            _, groups = np.unique(labels[dev], return_inverse=True)
            sibling = leave_photo_out(F.normalize(photo_h14).numpy(), groups)
            oracle_full = blend_relative(tokens, whole, 1.)
            oracle_concept = blend_relative(tokens, compose(torch.from_numpy(sibling)), 1.)
            configs = {
                "d2": (None, [], False, None),
                "truth-full-values": (oracle_full, model_steps, True, 1.),
                "truth-h14-values": (oracle_photo, model_steps, True, 1.),
                "sibling-h14-values": (oracle_concept, model_steps, True, 1.),
                "eeg-h14-values": (correct, model_steps, False, .5),
                "eeg-h14-late-values": (correct, model_steps[2:], False, .5),
                "eeg-h14-wrong-values": (wrong, model_steps, False, .5),
            }
    del embeds, teacher, adapter, projector
    strength, keep = DEFAULT_LAYOUT
    seeds = [query_seed("dev", i) for i in range(len(rows))]
    manifest = dict(render_cache_version=2, subject=subject, budget=budget, training="direct-h14-values", row=rows, n=len(rows),
                    repeats=repeats, seeds=seeds, prompt="", draws=1, unet_calls_per_draw=4, split_sha256=split,
                    development_split_sha256=study, physical_eeg=physical, d2_source=sources, checkpoint_sha256=checkpoint,
                    h14_receipt=receipt, training_visual_scale=float(norm), base_token_sha256=tensor_fingerprint(tokens),
                    layout_source=layout_proof, layout_sha256=tensor_fingerprint(starts), generator=weights, runtime=runtime(),
                    code=pins.pin(Path(__file__)), **supplemental)
    paths = []
    for name, (values, steps, oracle, mix) in configs.items():
        path = output / "images/direct-h14-values" / name / budget / f"sub-{subject:02d}"
        proof = dict(manifest, arm=name, reference_pixels_available_to_arm=oracle, mix=mix, value_timesteps=steps,
                     value_tokens_sha256=None if values is None else tensor_fingerprint(values), oracle_sources=representations if oracle else None,
                     auxiliary_donors=wrong_rows.tolist() if "-wrong-" in name else list(range(len(rows))))
        active = projections
        if spatial_value:
            band = bands.get(name)
            proof["spatial_band"] = band
            if band is not None:
                active = spatial_projections
                for projector in active:
                    projector.band = band

        def draw(stage, batches, done):
            for first in batches:
                replacement = None if values is None else values[first:first + 8].cuda()
                unet = RoutedUNet(pipe.unet, active, replacement, steps)
                runner = SimpleNamespace(unet=unet, scheduler=pipe.scheduler)
                draw_images(stage, runner, vae, prompt, tokens, seeds, starts, batch_starts=[first], batch_complete=done,
                            layout_strength=strength, layout_low_keep=keep)

        render_transaction(path, proof, draw, pins.verify)
        paths.append(path)
        print(f"Verified renders for {name} sub-{subject:02d}", flush=True)
    del pipe, vae
    torch.cuda.empty_cache()
    score_outputs(paths, "dev")
    save_json(output / "direct-h14-values.json", dict(input_proof=manifest, arms=list(configs)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", type=int, choices=range(1, 11), required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    experiment = parser.add_mutually_exclusive_group()
    experiment.add_argument("--full-replacement", action="store_true")
    experiment.add_argument("--acquisition", type=Path)
    experiment.add_argument("--cross-photo", type=Path)
    experiment.add_argument("--region-reference", type=Path)
    experiment.add_argument("--region-readout", type=Path)
    experiment.add_argument("--spatial-value", action="store_true")
    experiment.add_argument("--layout-response", type=Path)
    experiment.add_argument("--categorical", type=Path)
    experiment.add_argument("--scalp", type=Path)
    parser.add_argument("--whole-readout-timing", action="store_true")
    args = parser.parse_args()
    run(args.subject, args.features, args.output, args.full_replacement, args.acquisition, args.cross_photo,
        args.region_reference, args.region_readout, args.whole_readout_timing, args.spatial_value, args.layout_response,
        args.categorical, args.scalp)
