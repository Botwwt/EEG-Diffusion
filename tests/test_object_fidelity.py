"""Regression checks for the scoring protocol used by the object reconstruction experiments."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_scorer():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/eeg_stack_20261112/code/score_object_panel.py"
    )
    spec = importlib.util.spec_from_file_location("score_object_panel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_photo_identity_cannot_identify_an_object_through_the_held_out_ordinal():
    scorer = load_scorer()
    rows = np.arange(30)
    classes = rows // 10
    result = scorer.score_object_panel(np.eye(30), rows, classes)
    assert result["hits"]["photo_top1"] == 30
    assert result["hits"]["intra10_top1"] == 30
    assert np.isclose(result["hits"]["object_top1"], 10)


def test_paired_class_scores_are_invariant_to_joint_query_and_gallery_reordering():
    scorer = load_scorer()
    rows = np.arange(30)
    classes = rows // 10
    scores = (classes[:, None] == classes[None, :]).astype(float)
    baseline = scorer.score_object_panel(scores, rows, classes)
    order = np.random.default_rng(918).permutation(len(rows))
    candidate = scorer.score_object_panel(scores[np.ix_(order, order)], rows[order], classes[order])
    assert baseline["hits"]["object_top1"] == 30
    paired = scorer.paired_object_panel(baseline, candidate)
    for contrast in paired.values():
        assert contrast["delta"] == 0
        assert contrast["ci95"] == [0.0, 0.0]


def test_cross_participant_release_excludes_the_person_used_to_fit_the_source_model():
    results = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    controls = json.loads((results / "session_controls.json").read_text())
    source = controls["source_replay_excluded_from_averages"]
    for panel in controls["panels"].values():
        held_out = [v for k, v in panel["per_person_hits"].items() if k != source]
        assert len(held_out) == controls["heldout_people"] == 9
        for i, name in enumerate(panel["method_order"]):
            row = panel["heldout_performance"][name]
            hits = sum(v[i] for v in held_out)
            assert row["hits"] == hits
            assert np.isclose(row["mean"], hits / (len(held_out) * 120))


def test_confirmation_totals_match_the_locked_people_and_paired_render_records():
    results = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    design_bytes = (results / "personal_confirmation_design.json").read_bytes()
    design = json.loads(design_bytes)
    report = json.loads((results / "personal_confirmation.json").read_text())
    assert report["preregistered_protocol_sha256"] == hashlib.sha256(design_bytes).hexdigest()
    assert report["people"] == design["additional_participants"]
    assert set(report["people"]).isdisjoint(design["excluded_from_confirmation_mean"])
    assert report["unique_stimuli"] == 240
    assert report["participant_photo_queries"] == 1680
    for panel in report["panels"].values():
        assert set(panel["people"]) == set(report["people"])
        assert panel["n_queries"] == 7 * 120
        for record in panel["people"].values():
            for arm in ("frozen", "personal"):
                by_object = record[arm]["full1854_credit_by_object"]
                assert len(by_object) == len(panel["words"]) == 12
                assert np.isclose(sum(by_object), record[arm]["generated_full1854"])
        for metric, pair in panel["metrics"].items():
            for arm in ("frozen", "personal"):
                credit = sum(person[arm][metric] for person in panel["people"].values())
                assert credit == pair[arm]["credit"]
                assert np.isclose(pair[arm]["mean"], credit / panel["n_queries"])
            assert np.isclose(
                pair["paired_personal_minus_frozen"]["mean"],
                pair["personal"]["mean"] - pair["frozen"]["mean"],
            )
