"""Independently compare archived PNGs, effective model files and fitted layout chronology with the original receipts."""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import torch


WRAPPERS = {
    "matched":"swap_diff/arm1b_rep_condlyr_st600_s2025.pt",
    "rec":"swap_diff/arm3b_rec_only_st600_s2025.pt",
    "swap":"swap_diff/arm3b_rep_rawswap_condlyr_st600_s2025.pt",
    "d1":"swap_diff/draft_intra4/d1_simonly_cond_st120_s2025.pt",
    "d2":"swap_diff/draft_intra4/d2_compete4_cond_st120_s2025.pt",
    "response":"swap_diff/draft_intra4/response_s20261125.pt",
    "response-positive":"swap_diff/draft_intra4/response_positive_s20261125.pt",
}


@lru_cache(maxsize=None)
def fingerprint(path):
    with path.open("rb") as read:
        return hashlib.file_digest(read, "sha256").hexdigest()


@lru_cache(maxsize=None)
def embedded_encoder(file):
    pack = torch.load(file, map_location="cpu", weights_only=True)
    # The renderer uses embedded state when available, else the exact referenced backbone.
    return None if "state_dict" in pack else Path(pack["backbone_ckpt"]).resolve()


def audit(root, subject):
    person = f"sub-{subject:02d}"
    name = f"{subject:02d}"
    receipts = []
    expected = json.loads((root/"expected-inputs.json").read_bytes())
    train = root / f"data/prep/{person}_train.npy"
    test = root / f"data/prep/{person}_test.npy"
    for eeg in (train, test):
        if fingerprint(eeg) != expected[eeg.with_suffix(".json").name]["output_sha256"]:
            raise ValueError(f"Physical EEG integrity failed for {eeg.name}")
    for path in sorted(root.glob(f"images/*/*/*/{person}/render.json")):
        m = json.loads(path.read_bytes())
        track, arm, budget = path.relative_to(root/"images").parts[:3]
        if m.get("render_cache_version"):  # prospective runs have an in-render versioned source receipt
            continue
        fit = root / "fits" / (track if track == "shared" else f"{track}-{name}")
        source = m.get("semantic_condition", arm)
        source = {"wrong":"d2","d2-no-layout":"d2","native-layout":"native-paper"}.get(source, source)
        wrapper = WRAPPERS.get(source)
        base = None
        if wrapper:
            model = fit / wrapper
            base = embedded_encoder(model)
            if base is not None and not base.is_relative_to(fit.resolve()):
                raise ValueError(f"Nested checkpoint outside declared training scope at {path}")
        elif source in ("native-paper","native-code","instance-native"):
            fold = -1 if track == "native-full" else 0
            model = fit / f"enc/native/{person}/fold{fold}_seed2025/brain_model.pt"
        elif source in ("instance-direct","instance-residual"):
            model = fit / f"enc/instance_residual/{person}/fit.pt"
        else:
            raise ValueError(f"Unrecognized archived source {source}")
        if fingerprint(model) != m["checkpoint_sha256"] or m["split_sha256"] != fingerprint(root/"data/prep/paper_split.json"):
            raise ValueError(f"Original model/split differs at {path}")
        images = [path.parent / f"{i:05d}.png" for i in range(m["n"])]
        actual = [fingerprint(image) for image in images]
        if m["pixels_sha256"] != actual:
            raise ValueError(f"Generated PNG digest differs at {path}")
        ridge = (root / "layouts" / person / "ridge.pt") if m.get("layout_mode", "noise" if source.startswith("native") else "ridge-shell") == "ridge-shell" else None
        oldest = min(im.stat().st_mtime_ns for im in images)
        late = [("checkpoint",model),("nested_backbone",base),("layout",ridge)]
        later = [k for k,item in late if item is not None and item.stat().st_mtime_ns > oldest]
        rec = dict(folder=path.parent.relative_to(root).as_posix(), render_sha256=fingerprint(path),
                   png_count=len(actual), ordered_png_list_sha256=hashlib.sha256("\n".join(actual).encode()).hexdigest(),
                   checkpoint_sha256=fingerprint(model), effective_backbone_sha256=fingerprint(base) if base else None,
                   current_ridge_sha256=fingerprint(ridge) if ridge else None, artifacts_modified_after_first_png=later,
                   physical_eeg_sha256=fingerprint(test if budget.startswith("official") else train))
        receipts.append(rec)
    out = root / f"audit/archived/{person}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".partial")
    tmp.write_text(json.dumps(dict(scope="current file verification and ordering of recorded modification dates",
                         subject=subject, rows=receipts, auditor_sha256=fingerprint(Path(__file__))), indent=1))
    tmp.replace(out)
    print(person, len(receipts), sum(r["png_count"] for r in receipts), "later source files",
          [(r["folder"],r["artifacts_modified_after_first_png"]) for r in receipts if r["artifacts_modified_after_first_png"]], flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--subject", required=True, type=int, choices=range(1,11))
    a = p.parse_args()
    audit(a.root.resolve(), a.subject)
