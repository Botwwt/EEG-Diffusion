"""Replay the locked generic UNet test from both losses for every photo, and audit the H14 readout."""

import argparse
from pathlib import Path

import numpy as np
from fit_ip_denoiser_gradient import apply_head, digest, read, require, save, validate


def summarize(root, result, out):
    packet, design, receipt = validate(root)
    report, fit, per = [read(result / f) for f in ("report.json", "fit.json", "heldout.json")]
    for key, value in receipt.items():
        require(fit[key] == report[key] == value, "Changed input " + key)
    require(
        fit["head_sha256"] == report["head_sha256"] == digest(result / "head.npz")
        and digest(result / "heldout.json") == report["heldout_sha256"],
        "Changed output",
    )
    require(
        per["classes"] == design["eval_classes"]
        and per["rows"] == design["pair_rows"][128:]
        and per["timesteps"] == design["timesteps"][128:]
        and per["loss_axes"] == ["class", "matched_or_swapped", "photo_A_or_B"],
        "Evaluation alignment differs",
    )
    require(set(per["losses"]) == {"native", "learned", "sham", "oracle"}, "Missing method")
    errors = {a: np.array(v) for a, v in per["losses"].items()}
    require(
        all(
            x.shape == (48, 2, 2) and (x >= 0).all() and np.isfinite(x).all()
            for x in errors.values()
        ),
        "Invalid raw losses",
    )
    idx = np.random.default_rng(260525).integers(48, size=(10000, 48))

    def stat(x):
        return dict(
            mean=float(np.mean(x)), ci95=np.percentile(x[idx].mean(1), [2.5, 97.5]).tolist()
        )

    def verify(x, ref):
        require(
            abs(x["mean"] - ref["mean"]) < 1e-12
            and np.allclose(x["ci95"], ref["ci95"], rtol=0, atol=1e-12),
            "Paired statistic differs",
        )

    contrasts = {a: (x[:, 1] - x[:, 0]).mean(1) for a, x in errors.items()}
    for a, x in contrasts.items():
        verify(stat(x), report["arms"][a]["swapped_minus_matched"])
        verify(stat(errors[a][:, 0].mean(1)), report["arms"][a]["matched_mse"])
        require(int((x > 0).sum()) == report["arms"][a]["positive_classes"], "Paired count differs")
    for a in ("native", "sham"):
        verify(
            stat(contrasts["learned"] - contrasts[a]),
            report["paired_learned_contrast_advantage"][a],
        )
    absolute = stat((errors["native"][:, 0] - errors["learned"][:, 0]).mean(1))
    verify(absolute, report["absolute_mse_baseline_minus_learned"])
    passed = (
        all(stat(contrasts["learned"] - contrasts[a])["ci95"][0] > 0 for a in ("native", "sham"))
        and absolute["mean"] >= 0
        and contrasts["oracle"].mean() > 0
    )
    require(bool(passed) == report["advance_locked_generation_gate"], "Decision differs")

    with np.load(result / "head.npz", allow_pickle=False) as h:
        own = packet["own"][128:]
        prior = packet["prior"][128:]
        conditions = {
            "native": own,
            **{a: apply_head(prior, own, h["basis"], h[a]) for a in ("learned", "sham")},
        }
    actual = packet["oracle_eval_only"].astype(float)
    original = {}
    for a, vec in conditions.items():
        m = np.einsum("nkd,nkd->nk", vec.astype(float), actual - actual[:, ::-1]).mean(1)
        original[a] = dict(
            correct=int((m > 1e-12).sum()),
            ties=int((abs(m) <= 1e-12).sum()),
            pairs=len(m),
            matched_minus_swapped_cosine=stat(m),
        )
    data = dict(
        protocol="Same-object common-noise raw epsilon assay on historical training-image VAE modes; one independent pair per held-out generic concept, one participant.",
        input_receipts=receipt,
        study_subject=report["subject"],
        n_generic_in_historical_T=len(design["historical_generic_classes"]),
        n_adaptation_classes=report["n_training_classes"],
        n_generic_evaluation_classes=48,
        n_evaluation_photos=96,
        evaluator="48 paired object units; 10000 percentile bootstrap replicates, seed 260525. Positive epsilon contrast favors pairing EEG with its own true photo.",
        arms=report["arms"],
        paired_learned_contrast_advantage=report["paired_learned_contrast_advantage"],
        native_minus_learned_matched_epsilon_mse=absolute,
        passed_advance_gate=bool(passed),
        secondary_h14_embedding_match=original,
        secondary_h14_definition="Mean matched cosine minus exchanged cosine for the two original image embeddings, using the fixed per-query condition. H14 is the training representation. This descriptive check has no role in the generation gate.",
        scope="Generic, known matched word, eight available visual supports, real noised image modes. This measures local denoising sensitivity. It supplies no independent generated-image or target-object score.",
        runtime=report["runtime"],
        elapsed_seconds=report["elapsed_seconds"],
        peak_cuda_reserved_bytes=report["peak_cuda_reserved_bytes"],
        result_sha256={
            p: digest(result / p) for p in ("report.json", "heldout.json", "fit.json", "head.npz")
        },
        driver_sha256=report["driver_sha256"],
        summary_driver_sha256=digest(Path(__file__)),
    )
    out.parent.mkdir(exist_ok=True, parents=True)
    save(out, data)
    print("Verified", len(contrasts["native"]), "paired units; advance =", bool(passed), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--result", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    summarize(a.root, a.result, a.out)
