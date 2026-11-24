"""Regression checks for the scoring protocol used by the object reconstruction experiments."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def load_scorer():
    path = (
        Path(__file__).resolve().parents[1]
        / "research/eeg_stack_20261112/code/score_object_panel.py"
    )
    spec = importlib.util.spec_from_file_location("score_object_panel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rendering_uses_the_panel_list_in_the_frozen_design():
    path = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/code"
    spec = importlib.util.spec_from_file_location("renderer", path / "render_within_photo_ip.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    assert renderer.panel_names({}) == ["registered", "replicate"]
    assert renderer.panel_names({"panels": ["fresh"]}) == ["fresh"]
    for requested in (["registered"], ["fresh", "fresh"], []):
        with pytest.raises(ValueError):
            renderer.panel_names({"panels": ["fresh"]}, requested)
    with pytest.raises(ValueError):
        renderer.panel_names({"panels": ["../fresh"]})


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


def test_expanded_readout_pairs_use_the_original_budget_and_registered_photos():
    root = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    design = json.loads((root / "expanded_calibration_design.json").read_bytes())
    confirmation = json.loads((root / "expanded_confirmation_design.json").read_bytes())
    data = json.loads((root / "readout_scale.json").read_bytes())
    people = [*design["participants"], confirmation["participant"]]
    assert len(set(people)) == 3
    assert set(data["cases"]) == set(people)
    for sub in people:
        for panel in design["panels"]:
            result = data["cases"][sub][panel]
            baseline = result["personal_r4"]["credit"]
            candidate = result["expanded_r4"]
            for metric, pair in candidate["paired_vs_personal_r4"].items():
                delta = candidate["credit"][metric] - baseline[metric]
                assert np.isclose(delta, pair["credit_change"])
                assert np.isclose(delta / data["photos_per_case"], pair["rate_change"])
                assert np.isclose(np.mean(pair["rate_change_by_object"]), pair["rate_change"])
            assert ("full_head" in result) == (sub in design["participants"])


def test_within_photo_prior_ties_only_count_as_half_a_correct_pair():
    root = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    result = json.loads((root / "within_photo_embedding.json").read_bytes())
    for subject in result["participants"].values():
        panels, combined = subject["panels"].values(), subject["combined"]
        n = sum(panel["eligible_pairs"] for panel in panels)
        assert n == combined["eligible_pairs"]
        for arm, credit in combined["paired_top1_credit"].items():
            assert credit == sum(panel["paired_top1_credit"][arm] for panel in panels)
        assert combined["paired_top1_credit"]["512_uniform"] == n / 2
        assert combined["paired_top1_credit"]["512_paired_swap"] == (
            n - combined["paired_top1_credit"]["512_own"]
        )


def test_angle_renders_use_frozen_conditions_and_report_actual_image_counts():
    root = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    for report, design in (
        ("within_photo_render.json", "within_photo_render_design.json"),
        ("photo_angle_render.json", "photo_angle_render_design.json"),
    ):
        scores = json.loads((root / report).read_bytes())
        source = (root / design).read_bytes()
        spec = json.loads(source)
        assert scores["design_sha256"] == hashlib.sha256(source).hexdigest()
        assert (
            scores["radius_sha256"]
            == hashlib.sha256((root / "within_photo_render_radius.json").read_bytes()).hexdigest()
        )
        assert scores["participant_photo_queries"] == 240 * len(spec["cohort"])
        assert scores["generated_jpegs"] == scores["participant_photo_queries"] * 3
        for key, case in scores["cases"].items():
            panel, subject = key.split("/")
            assert case["condition_packet_sha256"] == spec["packets_sha256"][f"{subject}/{panel}"]


def test_generated_bank_shared_controls_tie_on_pairs_and_keep_calibration_cost_separate():
    root = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    for filename in ("generated_bank_clip.json", "generated_bank_dino.json"):
        scores = json.loads((root / filename).read_bytes())
        assert scores["unique_generated_candidates"] == 2 * 12 * 5 * scores["generator_budget"]
        for person in scores["subjects"].values():
            arms = person["both_panels"]["paired_assignment"]["arms"]
            assert arms["uniform"]["credit"] == arms["fixed0"]["credit"] == arms["own"]["n"] / 2
