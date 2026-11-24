import json
from pathlib import Path

import pytest

from eeg_diffusion.cli import main

ROOT = Path(__file__).resolve().parents[1]
RESULTS = Path("research/eeg_stack_20261112/results")


def test_r4_cli_recomputes_eight_arms_and_paired_comparisons(capsys):
    assert main(["verify-r4", "--root", str(ROOT)]) == 0
    report = json.loads(capsys.readouterr().out)
    all10 = report["splits"]["all10"]
    assert {arm: result["hits"] for arm, result in all10["arms"].items()} == {
        "Naive Arm1b": 906,
        "M1a + Arm1b": 939,
        "Naive SwapDiff": 943,
        "M1a + SwapDiff": 973,
        "Naive D2": 938,
        "M1a + D2": 982,
        "Naive M2a": 943,
        "M1a + M2a": 938,
    }
    assert all10["queries"] == 5000
    heldout = report["splits"]["heldout6"]
    assert heldout["queries"] == 3000
    assert heldout["arms"]["M1a + D2"]["hits"] == 592
    expected = {
        "all10": [(0.18, -0.5814, 0.9414), (0.86, 0.1088, 1.6112), (0.88, -0.1185, 1.8785)],
        "heldout6": [(0.6, -0.4866, 1.6866), (1, -0.2453, 2.2453), (1.6, 0.1889, 3.0111)],
    }
    for split, values in expected.items():
        paired = report["splits"][split]["paired_vs_selected"]
        for control, (delta, low, high) in zip(
            ("M1a + SwapDiff", "M1a + Arm1b", "Naive D2"), values, strict=True
        ):
            assert paired[control]["difference_pp"] == pytest.approx(delta)
            assert paired[control]["ci95_pp"] == pytest.approx([low, high], abs=1e-4)
    symmetric = heldout["paired_vs_selected"]["M1a + Arm1b"]
    assert symmetric["participants_improved"] == 5
    assert symmetric["p_sign_flip"] == pytest.approx(0.09375)


@pytest.mark.parametrize("corruption", ["participant", "fractional hit", "participant split"])
def test_r4_cli_rejects_invalid_participant_records(tmp_path, capsys, corruption):
    dest = tmp_path / RESULTS
    dest.mkdir(parents=True)
    for pattern in ("exp7_*", "exp8c_*", "exp8d_*", "exp9b_*"):
        source = next((ROOT / RESULTS).glob(pattern + ".json"))
        data = json.loads(source.read_text(encoding="utf-8"))
        if corruption == "participant split" and pattern.startswith("exp7"):
            data["heldout_subs"] = None
        elif corruption != "participant split" and pattern.startswith("exp8c"):
            records = data["m1a_d2_all10"]["s375"]
            if corruption == "participant":
                del records["sub-04"]
            else:
                records["sub-04"]["clip"]["top1"] = 0.123
        (dest / source.name).write_text(json.dumps(data), encoding="utf-8")
    assert main(["verify-r4", "--root", str(tmp_path)]) == 2
    assert corruption in capsys.readouterr().err
