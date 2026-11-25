"""Synthetic fixtures for isolated Inception scoring and its immutable receipts."""

import importlib.util
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "research/eeg_stack_20261112/code/paper_inception_delta.py"
)
SPEC = importlib.util.spec_from_file_location("inception_delta", SCRIPT)
delta = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(delta)


@pytest.fixture
def panel(tmp_path):
    n = 200
    rows = np.arange(n)
    concepts = np.array([f"synthetic-{i}" for i in rows])
    folder = tmp_path / "images/personal/d2/official-r4/sub-01"
    folder.mkdir(parents=True)
    hashes = []
    for i in range(n):
        raw = io.BytesIO()
        Image.new("RGB", (8, 8), (i, (i * 3) % 256, 85)).save(raw, format="PNG")
        (folder / f"{i:05d}.png").write_bytes(raw.getvalue())
        hashes.append(delta.digest(raw.getvalue()))
    source = dict(
        paper_split_sha256="1" * 64,
        gallery_sha256=delta.GALLERY["official"],
        reference_index_sha256="2" * 64,
    )
    render = folder / "render.json"
    render.write_text(
        json.dumps(
            dict(
                row=rows.tolist(),
                n=n,
                budget="official-r4",
                pixels_sha256=hashes,
                split_sha256=source["paper_split_sha256"],
            )
        )
    )
    observations = folder / "observations.npz"
    historical = (rows % 2).astype(float)
    np.savez_compressed(observations, row=rows, concept=concepts, inception_2way=historical)
    (folder / "scores.json").write_text(
        json.dumps(
            dict(
                panel="official",
                n=n,
                raw_RGB=True,
                observations_sha256=delta.file_sha(observations),
                render_sha256=delta.file_sha(render),
                ordered_image_hashes_sha256=delta.chain(hashes),
                mean={"inception_2way": historical.mean()},
            )
        )
    )
    features = np.random.default_rng(123).normal(size=(n, 12)).astype(np.float32)
    gallery = delta.Gallery(rows, concepts, [], [], source, features)
    return folder, gallery


def test_strict_pearson_matches_authors_orientation_and_ties():
    rng = np.random.default_rng(53)
    real = rng.normal(size=(13, 27)).astype(np.float32)
    predicted = rng.normal(size=real.shape).astype(np.float32)
    real[1] = real[0]
    wins, ties = delta.strict_counts(real, predicted)
    r = np.corrcoef(real, predicted)[:13, 13:]
    expected = np.array([(r[:, i] < r[i, i]).sum() for i in range(13)])
    wrong_direction = (r < np.diag(r)[:, None]).sum(1)
    np.testing.assert_array_equal(wins, expected)
    assert np.any(wins != wrong_direction)
    assert np.all(ties[:2] == 1)


def test_sidecar_round_trip_and_primary_immutability(panel):
    folder, gallery = panel
    primary_hashes = {
        f: delta.file_sha(folder / f) for f in ("scores.json", "observations.npz", "render.json")
    }
    runtime = dict(weights_sha256=delta.WEIGHTS_SHA256, scorer_sha256="3" * 64)
    delta.score_folder(
        folder, gallery, delta.read_primary(folder), lambda raw: gallery.features, runtime
    )
    saved, data = delta.read_sidecar(folder)
    assert saved["mean"] == {delta.METRIC: 1.0, delta.LEGACY: 0.5}
    assert saved["denominator"] == 199 and np.all(data["wins"] == 199)
    assert saved["source_sha256"] == primary_hashes

    def no_reencode(raw):
        pytest.fail("An already completed score was recomputed")

    paths = [folder / f"{delta.STEM}.{ext}" for ext in ("json", "npz")]
    previous = [p.read_bytes() for p in paths]
    delta.score_folder(folder, gallery, delta.read_primary(folder), no_reencode, runtime)
    assert previous == [p.read_bytes() for p in paths]
    assert primary_hashes == {f: delta.file_sha(folder / f) for f in primary_hashes}
    (folder / "render.json").write_bytes(b"{}")
    with pytest.raises(ValueError, match="Primary source"):
        delta.read_sidecar(folder)


def test_query_rejects_swapped_or_unbound_pixels(panel):
    folder, gallery = panel
    first = folder / "00000.png"
    raw = first.read_bytes()
    first.write_bytes((folder / "00001.png").read_bytes())
    with pytest.raises(ValueError, match="row 0"):
        delta.query(folder, delta.read_primary(folder), gallery)
    first.write_bytes(raw)
    score_path = folder / "scores.json"
    score = json.loads(score_path.read_bytes())
    del score["ordered_image_hashes_sha256"]
    score_path.write_text(json.dumps(score))
    with pytest.raises(ValueError, match="RGB sequence"):
        delta.query(folder, delta.read_primary(folder), gallery)


def test_discovery_shards_remain_stable_when_new_arms_finish(tmp_path):
    all_dirs = [
        "images/personal/d2/dev-r4/sub-01",
        "external/atm-sub02",
        "calibration/dev/original-photo",
        "exploratory/soft-wrong/dev-r4/sub-03",
        "side/repeat_images/r4/scored",
    ]

    def stage(directory):
        folder = tmp_path / directory
        folder.mkdir(parents=True)
        (folder / "scores.json").touch()

    for x in all_dirs:
        stage(x)
    buckets = [delta.select_folders(tmp_path, [], i, 4) for i in range(4)]
    selected = sum(buckets, [])
    assert len(selected) == len(set(selected)) == 4
    stage("images/shared/wrong/fresh-r4/sub-01")
    for i, bucket in enumerate(buckets):
        assert set(bucket) <= set(delta.select_folders(tmp_path, [], i, 4))


def test_numeric_export_preserves_raw_receipts_and_corrected_rows(panel, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import package_paper_scores as package
    import paper_inception_report as report

    folder, gallery = panel
    runtime = dict(weights_sha256=delta.WEIGHTS_SHA256, scorer_sha256="3" * 64)
    delta.score_folder(
        folder, gallery, delta.read_primary(folder), lambda raw: gallery.features, runtime
    )
    downstream = tmp_path / "unpacked"
    package.export(tmp_path, tmp_path / "private.zip")
    package.scores_archive(tmp_path / "private.zip", tmp_path / "numeric.zip")
    package.unpack(tmp_path / "numeric.zip", downstream)
    rel = folder.relative_to(tmp_path)
    for name in (
        "scores.json",
        "observations.npz",
        "render.json",
        f"{delta.STEM}.npz",
        f"{delta.STEM}.json",
    ):
        assert (folder / name).read_bytes() == (downstream / rel / name).read_bytes()
    assert not (downstream / rel / "00000.png").exists()
    manifest = json.loads((downstream / "coverage.json").read_bytes())
    assert len(manifest["rows"]) == len(manifest["inception_official"]) == 1
    primary = {r["folder"]: r for r in manifest["rows"]}
    corrected = {r["folder"]: r for r in manifest["inception_official"]}
    evidence, values = report.read_export(downstream, corrected, primary)
    assert evidence[rel.as_posix()]["n"] == 200
    np.testing.assert_array_equal(values[rel.as_posix()][2], np.ones((200, 1)))
    corrected[rel.as_posix()]["scores_sha256"] = "9" * 64
    with pytest.raises(ValueError, match="committed export"):
        report.read_export(downstream, corrected, primary)


def test_partial_corrected_coverage_pairs_only_same_people(monkeypatch):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import paper_inception_report as report

    people = ["sub-01", "sub-02", "sub-03"]

    def arm(key):
        return dict(
            participants=people,
            sources={p: {"folder": key + p} for p in people},
            metrics={},
            per_participant={p: {} for p in people},
        )

    a, b = arm("a"), arm("b")
    comparison = dict(
        arm_a="a",
        arm_b="b",
        participants=people,
        metrics={},
        per_participant={p: {} for p in people},
    )
    matrix = dict(arms={"a": a, "b": b}, comparisons=[comparison])
    scores = {}
    for folder, fraction in [("asub-01", 1), ("asub-02", 0.2), ("bsub-02", 0.75), ("bsub-03", 0.9)]:
        scores[folder] = (
            np.arange(200),
            np.array([f"stim{i}" for i in range(200)]),
            np.full((200, 1), fraction),
        )
    report.add_matrix(matrix, scores, "official", 15, 123)
    assert a["metrics"][delta.METRIC]["mean"] == pytest.approx(0.6)
    assert b["metrics"][delta.METRIC]["mean"] == pytest.approx(0.825)
    assert comparison["inception_official"]["participants"] == ["sub-02"]
    assert comparison["inception_official"]["scored_image_pairs"] == 200
    stats = comparison["metrics"][delta.METRIC]
    assert stats["difference_a_minus_b"] == pytest.approx(-0.55)
    assert stats["credit_denominator"] == 200 * 199
    assert stats["crossed_ci95"] is None
    assert comparison["per_participant"]["sub-02"][delta.METRIC]["stimulus_ci95"] == pytest.approx(
        [-0.55, -0.55]
    )


def test_supplementary_object_matrices_keep_raw_metrics_and_count_each_folder_once(
    monkeypatch, tmp_path
):
    monkeypatch.syspath_prepend(str(SCRIPT.parent))
    import paper_inception_report as report

    rows = np.arange(200)
    labels = np.array([f"object-{i}" for i in rows])
    folders = {
        "native": "images/personal/native-paper/official-r4/sub-01",
        "token": "images/personal/token/official-r4/sub-01",
    }
    scores = {
        folder: (rows, labels, np.full((200, 1), i)) for i, folder in enumerate(folders.values())
    }
    monkeypatch.setattr(report, "read_export", lambda *_: ({}, scores))

    def matrix(names):
        arms = {
            k: dict(
                participants=["sub-01"],
                sources={"sub-01": {"folder": folders[k]}},
                metrics={"inception_2way": {"mean": 0.5}},
                per_participant={"sub-01": {}},
            )
            for k in names
        }
        return dict(panel="official", arms=arms, comparisons=[])

    main = {"official-r4": matrix(["native"])}
    extra = {
        "development": matrix([]),
        "confirmation": {"official-r4": matrix(["native", "token"])},
        "components": {"official-r4": matrix(["token"])},
    }
    summary = report.attach(tmp_path, {}, {}, main, None, 20, 14, extra)
    assert summary["formal_scored_cases"] == 1
    assert summary["object_study_scored_unique_cases"] == 2
    assert (
        summary["auxiliary_scored_cases"]
        == summary["object_study_additional_cases_outside_formal"]
        == 1
    )
    token = extra["confirmation"]["official-r4"]["arms"]["token"]
    assert token["metrics"]["inception_2way"] == {"mean": 0.5}
    assert token["metrics"][delta.METRIC]["mean"] == 1.0
