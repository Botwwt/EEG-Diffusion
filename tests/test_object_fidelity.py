"""Regression checks for the scoring protocol used by the object reconstruction experiments."""

import importlib.util
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
