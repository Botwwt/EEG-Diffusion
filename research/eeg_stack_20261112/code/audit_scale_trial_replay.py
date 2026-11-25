"""Verify exact artifact and scientific parity between signature-preserving trial releases."""

import argparse
from pathlib import Path

from prepare_scale_trial import PUBLIC
from prepare_trial_object_inputs import check
from register_fresh_photo_panel import freeze, read, sha

CODE = Path(__file__).resolve().parent
HISTORICAL = "9a87881482547aa8f3d27f305c11a4121ad95575"


def scientific_equal(old, new, exclude):
    check(
        set(old) == set(new) and all(old[k] == new[k] for k in set(old) - set(exclude)),
        "The scientific contents changed",
    )


def audit(a):
    # The small archived records bind the private source receipts and each output payload.
    # When replaying elsewhere, a published equality receipt can supply the old payload
    # hashes without requiring a second full fit from the historical Git checkout.
    scale = a.root / "scale-trial"
    pair = a.root / "trial-top-two"
    studies = {}
    for key, base, name, signature in (
        (
            "scale",
            scale,
            "scale_trial",
            "517bfc9001e86fc84847eddfc97846617929d9dc1126df2f0ba44bd08399c368",
        ),
        (
            "top_two",
            pair,
            "trial_top_two",
            "e34e9680b2ab878d63f8643280054da3604f8f6bafb2341944f9229e012de90a",
        ),
    ):
        released = PUBLIC / (name + ".json")
        check(sha(released) == signature, "Historical release hash differs")
        old = read(released)
        fresh = base / "replay"
        new = read(fresh / "summary.json")
        path = PUBLIC / (name + "_replay.json")
        prior = read(path) if path.exists() else None
        if (base / "summary.json").exists():
            check(sha(base / "summary.json") == signature, "Original private replay root changed")
        evidence = dict(
            historical_git_commit=HISTORICAL,
            historical_result_sha256=signature,
            replay_summary_sha256=sha(fresh / "summary.json"),
            audit_source_sha256=sha(Path(__file__)),
        )
        if key == "scale":
            scientific_equal(old, new, ("cells", "driver_sha256"))
            check(new["driver_sha256"] == sha(CODE / "fit_scale_trial.py"), "Changed replay source")
            cells = {}
            for cell, hs in old["cells"].items():
                n, u = cell.split("_")
                folder = f"objects{n}_steps{u}"
                rec = fresh / folder / "receipt.json"
                rr = read(rec)
                check(
                    new["cells"][cell] == sha(rec) and rr["driver_sha256"] == new["driver_sha256"],
                    "Wrong replay fit",
                )
                hist = base / folder / "receipt.json"
                if hist.exists():
                    check(sha(hist) == hs, "Changed original cell")
                    before = read(hist)
                    scientific_equal(before, rr, ("driver_sha256",))
                    hashes = before["files"]
                else:
                    check(prior is not None, "Original cell or released parity receipt required")
                    hashes = prior["cells"][cell]["identical_file_sha256"]
                check(
                    rr["files"] == hashes
                    and all(sha(fresh / folder / k) == v for k, v in hashes.items()),
                    "Replayed weights/embeddings/per-query bytes differ",
                )
                cells[cell] = dict(
                    original_receipt_sha256=hs,
                    replay_receipt_sha256=sha(rec),
                    identical_file_sha256=hashes,
                )
            evidence.update(
                cells=cells,
                current_fit_source_sha256=new["driver_sha256"],
                exact_checkpoints=4,
                exact_embedding_packets=4,
                exact_row_packets=8,
                exact_mixed_predictions_across_all_cells_and_menus=7680,
                exact_neural_predictions_across_all_cells_and_menus=7680,
            )
        else:
            scientific_equal(
                old, new, ("source_sha256", "rule_sha256", "held_embedding_receipt_sha256")
            )
            check(
                new["source_sha256"] == sha(CODE / "score_trial_top_two.py")
                and sha(fresh / "per_query.npz") == old["per_query_sha256"]
                and sha(fresh / "locked_rule.json") == new["rule_sha256"],
                "Wrong top-two prediction source",
            )
            r = read(fresh / "locked_rule.json")
            check(
                r["extractor_sha256"] == sha(CODE / "fit_trial_top_two.py"),
                "Changed encoder driver",
            )
            if (base / "locked_rule.json").exists():
                check(
                    sha(base / "locked_rule.json") == old["rule_sha256"],
                    "Wrong original top-two fit",
                )
                scientific_equal(
                    read(base / "locked_rule.json"),
                    r,
                    ("source_sha256", "extractor_sha256", "folds"),
                )
            else:
                check(
                    prior is not None and prior["identical_rule"] == r["rule"],
                    "Original fit or published calibration parameters required",
                )
            view_hashes = {}
            for fold in ("fold0", "fold1", "held"):
                full = fresh / fold / "views.npz"
                h = sha(full)
                if (base / fold / "receipt.json").exists():
                    orig = read(base / fold / "receipt.json")
                    check(orig["packet_sha256"] == h, "Different originally fitted views")
                else:
                    check(
                        prior is not None and prior["identical_view_packets"][fold] == h,
                        "Different views from the release",
                    )
                check(
                    read(fresh / fold / "receipt.json")["packet_sha256"] == h,
                    "Different replay views",
                )
                view_hashes[fold] = h
            evidence.update(
                current_score_source_sha256=new["source_sha256"],
                current_fit_source_sha256=r["extractor_sha256"],
                identical_view_packets=view_hashes,
                identical_rule=r["rule"],
                exact_held_queries=960,
                per_query_sha256=old["per_query_sha256"],
            )
        freeze(path, evidence)
        studies[key] = sha(path)
    print(studies)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("outputs"))
    audit(p.parse_args())
