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


def test_donor_control_preserves_the_registered_recipients_angle():
    path = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/code"
    spec = importlib.util.spec_from_file_location("renderer", path / "render_within_photo_ip.py")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)
    angle = np.array([0.2, 0.7])
    prior = np.tile([1.0, 0, 0], (2, 1))
    direction = np.array([[0, 1, 0], [0, 0, 1]])
    donor = np.array([1, 0])
    own = np.cos(angle[:, None]) * prior + np.sin(angle[:, None]) * direction
    wrong = np.cos(angle[:, None]) * prior + np.sin(angle[:, None]) * direction[donor]
    vectors = {"own": own, "wrong": wrong, "uniform": prior}
    renderer.check_donor_geometry(vectors, donor, angle)
    with pytest.raises(ValueError):
        renderer.check_donor_geometry(vectors, donor)
    vectors["wrong"] = own[donor]
    renderer.check_donor_geometry(vectors, donor)
    with pytest.raises(ValueError):
        renderer.check_donor_geometry(vectors, donor, angle)


def test_repeat_and_cross_subject_feature_gains_share_the_same_held_out_objects():
    results = (
        Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/results/object_fidelity"
    )
    split = json.loads((results / "shared_photo_eeg_design.json").read_bytes())
    repeat = json.loads((results / "repeat_photo_subspace.json").read_bytes())
    shared = json.loads((results / "shared_photo_eeg.json").read_bytes())
    assert repeat["split_sha256"] == shared["design_sha256"]
    assert set(split["train_classes"]).isdisjoint(split["test_classes"])
    assert repeat["evaluation_objects"] == split["test_classes"]
    for subject in split["subjects"]:
        scores = repeat["class_pair_hits"][subject]
        assert len(scores["repeat"]) == len(split["test_classes"]) == 96
        assert sum(scores["repeat"]) == repeat["correct_pair_assignments"][subject]["repeat"]
        assert (
            sum(scores["posterior"])
            == shared["arms"]["direct"]["per_subject"][subject]["correct_equivalent"]
        )


def test_fourth_panel_scores_and_generated_galleries_use_the_frozen_registration():
    research = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112"
    results = research / "results/object_fidelity"
    panel = json.loads((results / "prospective_panel_design.json").read_bytes())
    evidence = json.loads((results / "prospective_paired_images.json").read_bytes())
    names = json.loads((results / "prospective_reference_names.json").read_bytes())
    gallery = json.loads((results / "prospective_gallery.json").read_bytes())
    assert (
        evidence["inputs"]["registration_sha256"]
        == names["registration_sha256"]
        == panel["registration_sha256"]
    )
    assert names["targets"] == evidence["unique_original_photos"] == 120
    assert evidence["participant_queries"] == 120 * len(panel["participants"]) == 360
    for sub in panel["participants"]:
        entry = gallery["cases"][sub]["public"]
        image = research / "gallery" / entry["file"]
        assert hashlib.sha256(image.read_bytes()).hexdigest() == entry["sha256"]
    for method in evidence["models"]:
        for endpoint in ("clip_closed12", "clip_full1854", "dino_intra10"):
            counts = evidence["pooled"][endpoint]["credits"][method]
            assert set(counts) == {"own", "wrong", "uniform"}
            assert all(0 <= value <= evidence["participant_queries"] for value in counts.values())


def test_session_matched_images_keep_candidate_menus_and_acquisition_budgets_distinct():
    research = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112"
    results = research / "results/object_fidelity"
    report = json.loads((results / "cohort_session_images.json").read_bytes())
    menus = json.loads((results / "cohort_session_menus.json").read_bytes())["panels"]
    for panel, choices in (("stress12", 12), ("new_three4", 4)):
        study = report["panels"][panel]
        assert study["eeg_choices"] == choices
        assert study["registration_sha256"] == menus[panel]["registration_sha256"]
        assert study["n_unique_photos"] == study["original_names"]["targets"] == 120
        assert study["generated_jpegs"] == 2 * 120 * 3
        for mode, observations in (("sub-04", 4), ("cohort-3", 12)):
            source = study["inputs"][mode]
            assert source["physical_trials_per_photo"] == observations
            image = research / "gallery" / source["generated_gallery_filename"]
            assert (
                hashlib.sha256(image.read_bytes()).hexdigest() == source["generated_gallery_sha256"]
            )


def test_fresh_reports_score_the_registered_subjects_photographs_and_controls():
    release = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112"
    results = release / "results/object_fidelity"
    design = json.loads((results / "fresh_panel_design.json").read_bytes())
    eeg = json.loads((results / "fresh_panel_eeg.json").read_bytes())
    dino = json.loads((results / "fresh_photo_dino_pixels.json").read_bytes())
    clip = json.loads((results / "fresh_photo_clip.json").read_bytes())
    people = design["participants"]
    assert set(eeg["participants"]) == set(dino["cases"]) == set(people)
    assert dino["subjects"] == people
    assert (
        design["registration_sha256"] == eeg["registration_sha256"] == dino["registration_sha256"]
    )
    assert clip["generated_jpegs"] == dino["generated_jpegs"] == 3 * len(people) * 120
    assert dino["unique_stimuli"] == clip["unique_originals"] == design["distinct_photos"] == 120
    for subject in people:
        case = dino["cases"][subject]
        photo = clip["cases"]["fresh/" + subject]
        assert photo["reg_sha256"] == eeg["registration_sha256"]
        assert photo["condition_packet_sha256"] == eeg["participants"][subject]["condition_sha256"]
        assert (
            case["metrics"]["dino_strict_pair"]["n"]
            == eeg["participants"][subject]["strict_pairs_both_words_correct"]
        )
        gallery = release / "gallery" / f"photo_angle_{subject.replace('-', '')}_fresh.jpg"
        assert hashlib.sha256(gallery.read_bytes()).hexdigest() == case["gallery"]["sha256"]
    for metric in ("dino_intra10", "dino_strict_pair"):
        pooled = dino["pooled"][metric]
        for arm in ("own", "uniform", "wrong"):
            credit = sum(dino["cases"][s]["metrics"][metric]["arms"][arm]["credit"] for s in people)
            assert credit == pooled["arms"][arm]["credit"]
            assert np.isclose(credit / pooled["n"], pooled["arms"][arm]["mean"])


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
