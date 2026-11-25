"""The fixtures are synthetic scorer files for checking aggregation arithmetic."""

import csv
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/code/paper_aggregate.py"


@pytest.fixture
def aggregator():
    assert SCRIPT.is_file(), "Missing benchmark aggregator"
    spec = importlib.util.spec_from_file_location("paper_aggregate", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score_files(root, arm, sub, budget="official-r4", edit=None, track="personal"):
    panel = budget.split("-")[0]
    n = 500 if panel == "fresh" else 200
    rows = np.arange(n)
    data = {
        "row": rows,
        "concept": np.asarray(
            [f"object-{i}" for i in (rows if panel == "official" else rows // 10)]
        ),
    }
    rates = "alex2_2way alex5_2way inception_2way clip_2way clip_image_top1 original_name1854"
    for key in rates.split():
        data[key] = (rows % 4 == 0).astype(float)
    data["generated_name1854"] = 0.5 * (rows % 2 == 0) * (arm == "d2")
    for key in ("ssim", "pixcorr", "effnet_distance", "swav_distance"):
        data[key] = rows / n
    for key in ("generated_top_word", "original_top_word"):
        data[key] = data["concept"].copy()
    if panel != "official":
        data["clip_within10"] = ((rows % 2 == 0) & (arm == "d2")).astype(float)
        data["clip_within_swap"] = data["clip_within10"] / 2
    if panel == "dev":
        data["clip_within10_uncentered"] = data["clip_within10"].copy()
        data["clip_global200_centered"] = np.zeros(n)
    if edit:
        edit(data)
    order = rows[::-1] if arm == "d2" else rows
    folder = root / "images" / track / arm / budget / sub
    folder.mkdir(parents=True, exist_ok=True)
    observations = folder / "observations.npz"
    np.savez_compressed(observations, **{k: v[order] for k, v in data.items()})
    receipt = {
        "panel": panel,
        "n": n,
        "raw_RGB": True,
        "mean": {
            k: float(np.mean(v)) for k, v in data.items() if k != "row" and v.dtype.kind in "biuf"
        },
        "observations_sha256": hashlib.sha256(observations.read_bytes()).hexdigest(),
    }
    (folder / "scores.json").write_text(json.dumps(receipt))
    (folder / "render.json").write_text(
        json.dumps({"training": track, "arm": arm, "budget": budget, "subject": sub})
    )
    return folder


def report(module, root, budget="official-r4", subjects=("sub-01", "sub-02")):
    return module.aggregate(
        root,
        "personal",
        budget,
        ["native-paper", "d2"],
        subjects,
        [("d2", "native-paper")],
        resamples=512,
        seed=31,
    )


def test_joins_shuffled_ids_keeps_fractional_ties_and_reports_missing(aggregator, tmp_path):
    for arm in ("native-paper", "d2"):
        for sub in ("sub-01", "sub-02"):
            score_files(tmp_path, arm, sub)
    # A completed single-arm run must not enter the paired denominator.
    score_files(tmp_path, "d2", "sub-03")
    r = report(aggregator, tmp_path, subjects=("sub-01", "sub-02", "sub-03"))
    pair = r["comparisons"][0]
    assert not r["complete"] and r["coverage"]["native-paper"]["missing"] == ["sub-03"]
    assert r["arms"]["d2"]["scored_images"] == 600
    assert pair["participants"] == ["sub-01", "sub-02"] and pair["scored_image_pairs"] == 400
    assert pair["metrics"]["clip_image_top1"]["crossed_ci95"] == [0, 0]
    name = pair["metrics"]["generated_name1854"]
    assert (name["credit_a"], name["credit_b"], name["difference_a_minus_b"]) == (100, 0, 0.25)


def test_bootstrap_preserves_whole_concepts_and_shared_item_draws(aggregator, tmp_path):
    for arm in ("native-paper", "d2"):
        for sub in ("sub-01", "sub-02"):
            score_files(tmp_path, arm, sub, "fresh-r4")
    r = report(aggregator, tmp_path, "fresh-r4")
    assert r["comparisons"][0]["metrics"]["clip_within10"]["crossed_ci95"] == [0.5, 0.5]
    # Eight people observe identical outcomes for two items. One shared item draw
    # preserves the full [0,1] interval. Independent item draws per person shrink it.
    shared = np.tile([0.0, 1.0], (8, 1))[:, :, None]
    people = np.tile([[0.0], [1.0]], (1, 8))[:, :, None]
    for values in (shared, people):
        low, high = aggregator.bootstrap_ci(values, 1000, 16)
        assert [float(low[0]), float(high[0])] == [0.0, 1.0]


@pytest.mark.parametrize("fault", ["duplicate", "concept", "receipt", "budget", "training"])
def test_damaged_or_mixed_inputs_raise(aggregator, tmp_path, fault):
    score_files(tmp_path, "native-paper", "sub-01")

    def change(data):
        if fault == "duplicate":
            data["row"][1] = data["row"][0]
        if fault == "concept":
            data["concept"][[0, 1]] = data["concept"][[1, 0]]

    folder = score_files(tmp_path, "d2", "sub-01", edit=change)
    if fault in ("budget", "training"):
        p = folder / "render.json"
        d = json.loads(p.read_text())
        d[fault] = "official-r80" if fault == "budget" else "shared"
        p.write_text(json.dumps(d))
    if fault == "receipt":
        p = folder / "scores.json"
        d = json.loads(p.read_text())
        d["observations_sha256"] = "0" * 64
        p.write_text(json.dumps(d))
    with pytest.raises(ValueError):
        report(aggregator, tmp_path, subjects=("sub-01",))


def test_lower_distance_favors_a_and_single_person_interval_is_conditional(aggregator, tmp_path):
    score_files(tmp_path, "native-paper", "sub-01")
    score_files(tmp_path, "d2", "sub-01", edit=lambda obs: obs["effnet_distance"].fill(0.0))
    r = report(aggregator, tmp_path, subjects=("sub-01",))
    pair = r["comparisons"][0]
    metric = pair["metrics"]["effnet_distance"]
    assert metric["difference_a_minus_b"] == pytest.approx(-0.4975)
    assert metric["participant_direction"] == {"better_a": 1, "equal": 0, "better_b": 0}
    assert metric["crossed_ci95"] is None
    assert pair["per_participant"]["sub-01"]["effnet_distance"]["stimulus_ci95"][1] < 0
    assert r["reference"]["scored_photographs"] == 200


def test_unfinished_case_and_wrong_reference_source_are_distinct(aggregator, tmp_path):
    folder = score_files(tmp_path, "d2", "sub-02")
    (folder / "scores.json").unlink()
    r = report(aggregator, tmp_path)
    assert "scores.json" in r["coverage"]["d2"]["missing_files"]["sub-02"]
    assert r["comparisons"][0]["scored_image_pairs"] == 0
    assert r["comparisons"][0]["metrics"] == {}
    score_files(tmp_path, "native-paper", "sub-01")
    score_files(tmp_path, "d2", "sub-01", edit=lambda obs: obs["original_name1854"].fill(0))
    with pytest.raises(ValueError, match="reference"):
        report(aggregator, tmp_path)


def test_external_draw_zero_requires_row_identity_and_labels_training(aggregator, tmp_path):
    subject = "sub-01"
    folder = score_files(tmp_path, "atm", subject, "official-r80")
    release = tmp_path / "external" / "atm-sub01"
    release.parent.mkdir()
    folder.rename(release)
    (release / "render.json").unlink()
    manifest = release / "manifest.csv"
    rows = [
        {
            "eeg_row": i,
            "unique_id": f"object-{i}",
            "subject": subject,
            "draw": 0,
            "reconstruction_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
        }
        for i in reversed(range(200))
    ]

    def write_rows():
        with manifest.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_rows()
    score_files(tmp_path, "d2", subject, "official-r80")
    settings = tmp_path / "external-sources.json"
    settings.write_text(
        json.dumps(
            {
                "arm": "atm",
                "budget": "official-r80",
                "training": "released ATM",
                "participants": {subject: {"folder": "external/atm-sub01"}},
            }
        )
    )
    args = (tmp_path, "personal", "official-r80", ["d2", "atm"], (subject,), [("d2", "atm")])
    r = aggregator.aggregate(*args, resamples=200, external_manifest=settings)
    comparison = r["comparisons"][0]
    assert comparison["scored_image_pairs"] == 200
    assert comparison["training_a"] == "personal" and comparison["training_b"] == "released ATM"
    assert comparison["generation_settings_matched"] is None
    assert (
        r["arms"]["atm"]["sources"][subject]["manifest_sha256"]
        == hashlib.sha256(manifest.read_bytes()).hexdigest()
    )
    standalone = aggregator.aggregate(
        tmp_path, "external", "official-r80", ["atm"], [subject], [], resamples=200
    )
    assert standalone["complete"]
    known = json.loads(settings.read_text())
    known["training_participants_per_fit"] = 10
    settings.write_text(json.dumps(known))
    r = aggregator.aggregate(*args, resamples=200, external_manifest=settings)
    assert r["comparisons"][0]["training_scope_matched"] is False
    rows[0]["unique_id"] = "unrelated"
    write_rows()
    with pytest.raises(ValueError, match="manifest"):
        aggregator.aggregate(*args, resamples=200, external_manifest=settings)


@pytest.mark.parametrize("fault", ["mean", "reference_type", "render_rows", "render_sha256"])
def test_receipts_cannot_substitute_another_score_or_render(aggregator, tmp_path, fault):
    p = score_files(tmp_path, "d2", "sub-01")
    path = p / ("render.json" if fault == "render_rows" else "scores.json")
    meta = json.loads(path.read_text())
    if fault == "mean":
        meta["mean"]["generated_name1854"] = 1
    elif fault == "reference_type":
        meta["raw_RGB"] = False
    elif fault == "render_rows":
        meta["row"] = [0] * 200
    else:
        meta["render_sha256"] = "0" * 64
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        report(aggregator, tmp_path)


def test_current_fresh_endpoints_keep_interleaved_photos_of_a_concept_together(
    aggregator, tmp_path
):
    def interleave(obs, treatment):
        row = obs["row"]
        obs["concept"] = np.asarray([f"object-{i % 50}" for i in row])
        obs["clip_within10"] = ((row // 50) % 2 == 0) * treatment
        obs["clip_global500_centered"] = np.zeros(len(row))
        obs["clip_within10_uncentered"] = np.zeros(len(row))

    for arm in ("d2", "native-paper"):
        for sub in ("sub-01", "sub-02"):
            score_files(
                tmp_path,
                arm,
                sub,
                "fresh-r4",
                lambda obs, arm=arm: interleave(obs, int(arm == "d2")),
            )
    r = report(aggregator, tmp_path, "fresh-r4")
    metric = r["comparisons"][0]["metrics"]["clip_within10"]
    assert metric["crossed_ci95"] == [0.5, 0.5]
    assert "clip_global500_centered" in r["arms"]["d2"]["metrics"]
    assert r["arms"]["d2"]["metrics"]["clip_within_swap"]["credit_denominator"] == 4500
    # An older scorer used uncentered cosine under the same within-ten metric name.
    score_files(tmp_path, "d2", "sub-01", "fresh-r4")
    with pytest.raises(ValueError, match="centering"):
        report(aggregator, tmp_path, "fresh-r4")


def test_cli_reports_cross_training_groups_and_writes_json(tmp_path):
    source = score_files(tmp_path, "d2", "sub-01")
    shared = tmp_path / "images" / "shared" / "d2" / "official-r4" / "sub-01"
    shared.parent.mkdir(parents=True)
    source.rename(shared)
    render = shared / "render.json"
    meta = json.loads(render.read_text())
    meta["training"] = "shared"
    render.write_text(json.dumps(meta))
    score_files(tmp_path, "native-paper", "sub-01")
    out = tmp_path / "report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPT),
            "--root",
            str(tmp_path),
            "--training",
            "personal",
            "--budget",
            "official-r4",
            "--arms",
            "shared:d2",
            "native-paper",
            "--pair",
            "shared:d2",
            "native-paper",
            "--subjects",
            "sub-01",
            "--resamples",
            "128",
            "--output",
            str(out),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    assert out.is_file(), proc.stdout
    r = json.loads(out.read_text())
    assert r["comparisons"][0]["training_scope_matched"] is False
    assert r["arms"]["shared:d2"]["training_participants_per_fit"] == 10
    assert r["arms"]["shared:d2"]["layout_training_participants_per_fit"] == 1
    assert r["arms"]["native-paper"]["layout_training_participants_per_fit"] is None
    assert "sub-01" in proc.stdout and "generated_name1854" in proc.stdout


def test_complete_render_hashes_and_physical_repeat_controls(aggregator, tmp_path):
    folder = score_files(tmp_path, "d2", "sub-01")
    # Only the byte integrity is exercised here. Scores and payloads are fixtures.
    hashes = []
    for i in range(200):
        data = f"fixture {i}".encode()
        (folder / f"{i:05}.png").write_bytes(data)
        hashes.append(hashlib.sha256(data).hexdigest())
    render = folder / "render.json"
    settings = dict(
        training="personal",
        subject=1,
        arm="d2",
        budget="official-r4",
        n=200,
        prompt="",
        draws=1,
        unet_calls_per_draw=4,
        repeats=[[0, 1, 2, 3]],
        row=list(range(200)),
        eeg_donor=list(range(200)),
        seeds=canonical_seeds("official", 200),
        split_sha256="a" * 64,
        checkpoint_sha256="b" * 64,
        pixels_sha256=hashes,
    )
    render.write_text(json.dumps(settings))
    path = folder / "scores.json"
    receipt = json.loads(path.read_text())
    receipt["render_sha256"] = hashlib.sha256(render.read_bytes()).hexdigest()
    receipt["ordered_image_hashes_sha256"] = hashlib.sha256("\n".join(hashes).encode()).hexdigest()
    path.write_text(json.dumps(receipt))
    args = (tmp_path, "personal", "official-r4", ["d2"], ["sub-01"], [])
    r = aggregator.aggregate(*args, resamples=64, verify_pixels=True)
    proof = r["arms"]["d2"]["sources"]["sub-01"]
    assert proof["pixels_locally_verified"] == 200 and proof["render_bound_by_scorer"]
    (folder / "00199.png").write_bytes(b"tampered fixture")
    with pytest.raises(ValueError, match="SHA-256"):
        aggregator.aggregate(*args, resamples=64, verify_pixels=True)
    settings["repeats"] = [[0, 20, 40, 60]]
    render.write_text(json.dumps(settings))
    receipt["render_sha256"] = hashlib.sha256(render.read_bytes()).hexdigest()
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="repeats"):
        aggregator.aggregate(*args, resamples=64)


def test_changed_score_schema_cannot_disguise_new_centering_as_an_older_run(aggregator, tmp_path):
    def current(obs):
        obs["clip_global500_centered"] = obs["clip_within10"]
        obs["clip_within10_uncentered"] = 1.0 - obs["clip_within10"]

    folder = score_files(tmp_path, "d2", "sub-01", "fresh-r4", current)
    receipt = folder / "scores.json"
    info = json.loads(receipt.read_text())
    del info["mean"]["clip_global500_centered"]
    del info["mean"]["clip_within10_uncentered"]
    receipt.write_text(json.dumps(info))
    with pytest.raises(ValueError, match="schema"):
        aggregator.aggregate(tmp_path, "personal", "fresh-r4", ["d2"], ["sub-01"], [], resamples=64)


def canonical_seeds(panel, n):
    return [
        int(hashlib.sha256(f"paper-20261125/{panel}/{i}".encode()).hexdigest()[:16], 16)
        % (2**63 - 1)
        for i in range(n)
    ]


def registered_fixtures(root):
    prep = root / "data/prep"
    prep.mkdir(parents=True)
    for name, size, per_class in (("test", 200, 1), ("train", 16540, 10)):
        ids = [f"{name}-{i}" for i in range(size)]
        paths = [
            f"images/{i // per_class + 1:05d}_object-{i // per_class}/{ids[i]}.jpg"
            for i in range(size)
        ]
        (prep / f"{name}_ids.json").write_text(json.dumps({"ids": ids, "paths": paths}))
    np.save(prep / "train_concept.npy", np.arange(16540) // 10)
    split = {
        "train_rows": list(range(13230)),
        "fresh_rows": list(range(13230, 13730)),
        "fresh_concepts": list(range(1323, 1373)),
        "excluded_possible_historical_concepts": [1400],
        "official_rows": list(range(200)),
    }
    dev = {
        "dev_rows": list(range(13730, 13930)),
        "concepts": list(range(1373, 1393)),
        "seed": 287311,
        "subjects": [1, 3],
        "conditions": ["d2", "response-positive", "response"],
    }
    (prep / "paper_split.json").write_text(json.dumps(split))
    (prep / "paper_development.json").write_text(json.dumps(dev))
    return split, dev


def protocol_case(root, arm, sub, budget, track="personal"):
    prep = root / "data/prep"
    panel = budget.split("-")[0]
    if panel == "dev":
        rows = json.loads((prep / "paper_development.json").read_text())["dev_rows"]
    else:
        rows = list(range(200))

    def set_truth(obs):
        obs["row"] = np.asarray(rows)
        obs["concept"] = np.asarray([f"object-{r // 10 if panel == 'dev' else r}" for r in rows])
        obs["original_top_word"] = obs["concept"].copy()

    folder = score_files(root, arm, sub, budget, set_truth, track)
    path = folder / "render.json"
    m = json.loads(path.read_text())
    m.update(
        row=rows,
        n=len(rows),
        seeds=canonical_seeds(panel, len(rows)),
        eeg_donor=list(range(len(rows))),
        split_sha256=hashlib.sha256((prep / "paper_split.json").read_bytes()).hexdigest(),
        semantic_condition={"native-layout": "native-paper", "d2-no-layout": "d2"}.get(arm, arm),
        layout_mode="noise"
        if arm in ("native-paper", "native-code", "d2-no-layout")
        else "ridge-shell",
        development_split_sha256=hashlib.sha256(
            (prep / "paper_development.json").read_bytes()
        ).hexdigest()
        if panel == "dev"
        else None,
    )
    path.write_text(json.dumps(m))
    return folder


def test_dev_gallery_truth_seeds_and_registered_cohort(aggregator, tmp_path):
    registered_fixtures(tmp_path)
    for arm in ("native-layout", "d2-no-layout", "response", "response-positive"):
        for sub in ("sub-01", "sub-03"):
            protocol_case(tmp_path, arm, sub, "dev-r4")
    r = aggregator.aggregate(
        tmp_path,
        "personal",
        "dev-r4",
        ["native-layout", "d2-no-layout", "response", "response-positive"],
        pairs=[("response", "response-positive")],
        resamples=64,
    )
    assert r["requested_participants"] == ["sub-01", "sub-03"] and r["complete"]
    assert r["panel"] == "dev" and r["stimulus_units_per_participant"] == 20
    assert r["expected_photographs_per_participant"] == 200
    assert "clip_global200_centered" in r["comparisons"][0]["metrics"]
    assert r["arms"]["d2-no-layout"]["layout_training_participants_per_fit"] is None
    assert r["arms"]["native-layout"]["semantic_condition"] == "native-paper"
    assert r["arms"]["response-positive"]["layout_training_participants_per_fit"] == 1
    assert r["arms"]["response"]["metrics"]["clip_within_swap"]["credit_denominator"] == 1800


@pytest.mark.parametrize(
    "fault",
    [
        "seed",
        "truth",
        "dev_hash",
        "sample_leak",
        "concept_leak",
        "global_gallery",
        "component",
        "layout",
        "split_hash",
        "concept_index",
    ],
)
def test_development_integrity_single_arm(aggregator, tmp_path, fault):
    split, dev = registered_fixtures(tmp_path)
    folder = protocol_case(tmp_path, "native-layout", "sub-01", "dev-r4")
    prep = tmp_path / "data/prep"
    meta = json.loads((folder / "render.json").read_text())
    if fault == "seed":
        meta["seeds"] = canonical_seeds("official", 200)
    elif fault == "truth":
        p = prep / "train_ids.json"
        d = json.loads(p.read_text())
        a, b = dev["dev_rows"][0], dev["dev_rows"][10]
        for k in ("ids", "paths"):
            d[k][a], d[k][b] = d[k][b], d[k][a]
        p.write_text(json.dumps(d))
    elif fault == "dev_hash":
        meta["development_split_sha256"] = "0" * 64
    elif fault == "split_hash":
        meta["split_sha256"] = "0" * 64
    elif fault in ("sample_leak", "concept_leak"):
        if fault == "sample_leak":
            split["train_rows"].append(dev["dev_rows"][0])
        else:
            split["excluded_possible_historical_concepts"].append(dev["concepts"][0])
        p = prep / "paper_split.json"
        p.write_text(json.dumps(split))
        meta["split_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    elif fault == "global_gallery":
        path = folder / "scores.json"
        d = json.loads(path.read_text())
        d["mean"]["clip_global500_centered"] = d["mean"].pop("clip_global200_centered")
        path.write_text(json.dumps(d))
    elif fault == "component":
        meta["semantic_condition"] = "d2"
    elif fault == "layout":
        meta["layout_mode"] = "noise"
    else:
        p = prep / "train_concept.npy"
        arr = np.load(p)
        arr[dev["dev_rows"][0]], arr[dev["dev_rows"][10]] = (
            arr[dev["dev_rows"][10]],
            arr[dev["dev_rows"][0]],
        )
        np.save(p, arr)
    (folder / "render.json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        aggregator.aggregate(
            tmp_path, "personal", "dev-r4", ["native-layout"], ["sub-01"], resamples=64
        )


def test_full_native_keeps_training_concepts_separate_even_for_same_person(aggregator, tmp_path):
    registered_fixtures(tmp_path)
    for track in ("native-full", "personal"):
        protocol_case(tmp_path, "native-paper", "sub-01", "official-r80", track)
    first, second = "native-paper", "personal:native-paper"
    r = aggregator.aggregate(
        tmp_path,
        "native-full",
        "official-r80",
        [first, second],
        ["sub-01"],
        [(first, second)],
        resamples=64,
    )
    assert r["arms"][first]["training_participants_per_fit"] == 1
    assert r["arms"][first]["training_concepts"] == 1654
    assert r["arms"][second]["training_concepts"] == 1323
    assert r["comparisons"][0]["training_scope_matched"] is False
    with pytest.raises(ValueError):
        aggregator.aggregate(tmp_path, "native-full", "dev-r4", ["native-paper"], resamples=64)
    with pytest.raises(ValueError):
        aggregator.aggregate(tmp_path, "native-full", "official-r80", ["d2"], resamples=64)


def test_factorial_arms_keep_their_semantic_checkpoint_fixed(aggregator, tmp_path):
    registered_fixtures(tmp_path)
    for arm, digest in (("d2", "b"), ("d2-no-layout", "c")):
        folder = protocol_case(tmp_path, arm, "sub-01", "dev-r4")
        manifest = folder / "render.json"
        meta = json.loads(manifest.read_text())
        meta["checkpoint_sha256"] = digest * 64
        manifest.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="checkpoint"):
        aggregator.aggregate(
            tmp_path, "personal", "dev-r4", ["d2", "d2-no-layout"], ["sub-01"], resamples=64
        )


def test_full_native_cannot_use_restricted_fit_receipt(aggregator, tmp_path):
    registered_fixtures(tmp_path)
    protocol_case(tmp_path, "native-paper", "sub-01", "official-r80", "native-full")
    fit = tmp_path / "fits/native-full-01/enc/native/sub-01/fold-1_seed2025/fit.json"
    fit.parent.mkdir(parents=True)
    fit.write_text(json.dumps({"args": {"sub": 1, "fold": -1, "seed": 2025}, "n_train": 13230}))
    with pytest.raises(ValueError, match="training rows"):
        aggregator.aggregate(
            tmp_path, "native-full", "official-r80", ["native-paper"], ["sub-01"], resamples=64
        )
    fit.write_text(json.dumps({"args": {"sub": 1, "fold": -1, "seed": 2025}, "n_train": 16540}))
    r = aggregator.aggregate(
        tmp_path, "native-full", "official-r80", ["native-paper"], ["sub-01"], resamples=64
    )
    assert (
        r["arms"]["native-paper"]["sources"]["sub-01"]["native_fit_sha256"]
        == hashlib.sha256(fit.read_bytes()).hexdigest()
    )
