"""Regression checks for the object-first, participant-paired development decision."""

import importlib
from pathlib import Path

import pytest


@pytest.fixture
def study(monkeypatch):
    code = Path(__file__).resolve().parents[1] / "research/eeg_stack_20261112/code"
    monkeypatch.syspath_prepend(str(code))
    return importlib.import_module("paper_object_report")


def arm(sub01, sub03, within):
    hit_counts = {"sub-01": sub01, "sub-03": sub03}
    return {
        "participants": list(hit_counts),
        "per_participant": {
            p: {"generated_name1854": {"mean": value / 200}} for p, value in hit_counts.items()
        },
        "metrics": {
            "generated_name1854": {"credit": sub01 + sub03},
            "clip_within10": {"credit": within},
        },
    }


def test_selection_preserves_both_participants_and_secondary_endpoint(study):
    matrix = {
        "arms": {
            study.NATIVE: arm(5, 0, 79),
            study.PHOTO: arm(5, 7, 80),
            study.OBJECT: arm(6, 1, 78),
            study.RELIABLE: arm(6, 1, 79),
        }
    }
    people = ["sub-01", "sub-03"]
    result = study.select_candidate(matrix, people)
    assert result["selected"] == study.RELIABLE
    assert not result["candidates"][study.PHOTO]["qualifies"]
    assert not result["candidates"][study.OBJECT]["qualifies"]
    matrix["arms"][study.RELIABLE]["participants"].pop()
    assert study.select_candidate(matrix, people)["status"] == "incomplete"


def test_equal_followup_scores_select_the_registered_control(study):
    matrix = {
        "arms": {
            study.NATIVE: arm(5, 0, 79),
            study.FINAL_AUX: arm(6, 1, 79),
            study.MIDDLE_AUX: arm(6, 1, 79),
        }
    }
    result = study.select_candidate(
        matrix, ["sub-01", "sub-03"], (study.FINAL_AUX, study.MIDDLE_AUX)
    )
    assert result["selected"] == study.FINAL_AUX
