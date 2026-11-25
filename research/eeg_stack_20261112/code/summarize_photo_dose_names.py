"""Pair independent CLIP image naming across acquisition budgets and diffusion guidance."""

import argparse
from pathlib import Path

import numpy as np
from fit_shared_photo_eeg import read, require, write
from score_reconstruction_jpegs import registered
from score_within_photo_dino import sha
from score_within_photo_ip import CLIP_SHA, estimate

GROUPS = ("single-1", "cohort-3", "cohort-9", "guided-9")
ARMS = ("own", "uniform", "wrong")
PUBLIC = Path(__file__).resolve().parents[1] / "results/object_fidelity"


def summarize(a):
    cases = {k: [] for k in GROUPS}
    receipts = {}
    classes = []
    for panel in ("registered", "replicate"):
        reg = registered(a.references, panel)
        classes.extend([panel + "/" + label for label in reg["labels"]])
        for group in GROUPS:
            guided = group == "guided-9"
            folder = a.root / ("guided_inputs" if guided else "render_inputs")
            person = "cohort-9" if guided else group
            source = a.root / "names" / panel / ("guided.json" if guided else "ordinary.json")
            r = read(source)
            rec = r["cases"][panel + "/" + person]
            settings = folder / "renders" / panel / person / "settings.json"
            cfg = read(settings)
            require(
                r["clip_sha256"] == CLIP_SHA
                and rec["settings_sha256"] == sha(settings)
                and r["design_sha256"] == sha(folder / "design.json")
                and rec["rows"] == reg["rows"].tolist()
                and rec["truth"] == reg["labels"].tolist()
                and rec["predicted_words"] == cfg["predicted_words"]
                and rec["donor_rows"] == cfg["donor_rows"],
                "Mismatched identity or render source",
            )
            if guided:
                require(
                    cfg["uniform_parity_hits"] == 120
                    and cfg["guidance_design_sha256"]
                    == sha(PUBLIC / "fixed_word_guidance_design.json"),
                    "Guidance comparison changed",
                )
            cases[group].append(rec)
            receipts[group + "/" + panel] = dict(source=sha(source), settings=sha(settings))
        require(
            all(
                cases[k][-1]["predicted_words"] == cases[GROUPS[0]][-1]["predicted_words"]
                for k in GROUPS
            ),
            "The conditioning words differ between budgets",
        )
    metrics = {}
    for field in ("full1854", "closed12", "intra10"):
        values = {
            g: {arm: np.concatenate([r["per_row"][field][arm] for r in cases[g]]) for arm in ARMS}
            for g in GROUPS
        }

        def comparison(x, y):
            value = estimate(x - y, classes)
            return dict(delta=value["mean"], ci95=value["ci95"])

        metrics[field] = dict(
            credits={g: {arm: float(v.sum()) for arm, v in d.items()} for g, d in values.items()},
            guided_minus_ordinary_own=comparison(
                values["guided-9"]["own"], values["cohort-9"]["own"]
            ),
            nine_minus_single_own=comparison(values["cohort-9"]["own"], values["single-1"]["own"]),
            guided_own_minus_wrong=comparison(
                values["guided-9"]["own"], values["guided-9"]["wrong"]
            ),
        )
    write(
        a.release,
        dict(
            n=240,
            objects=24,
            fixed_word_correct=sum(
                x == y
                for r in cases["single-1"]
                for x, y in zip(r["truth"], r["predicted_words"], strict=True)
            ),
            clip_sha256=CLIP_SHA,
            receipts=receipts,
            metrics=metrics,
            method="Full and restricted exact name credits. CLIP intra10 uses reference and generated features centered separately by panel and arm. Bootstrap the 24 objects and the fixed viewers together, 10000 draws.",
        ),
    )
    print({k: v["credits"] for k, v in metrics.items()})


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs/fixed-word-photo-dose"))
    p.add_argument("--references", type=Path, default=Path("outputs/object-fidelity"))
    p.add_argument("--release", type=Path, default=PUBLIC / "fixed_word_photo_dose_names.json")
    summarize(p.parse_args())
