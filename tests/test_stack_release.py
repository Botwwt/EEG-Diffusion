import json
from pathlib import Path

import numpy as np
import pytest

from eeg_diffusion.cli import main
from eeg_diffusion.stack_audit import STACK_LOG, verify_stack_results

ROOT = Path(__file__).resolve().parents[1]


def test_stack_audit_recomputes_paired_counts_and_interval():
    report = verify_stack_results(ROOT)
    assert report["participants"] == 10
    assert report["images_per_participant"] == 800
    assert report["top1_hits"]["O14_noBigG"] == 1477
    assert report["top1_hits"]["O14"] == report["top1_hits"]["O14_sel4"] == 1473
    ablation = next(item for item in report["paired"] if item["treatment"] == "O13")
    assert ablation["difference_pp"] == pytest.approx(0.6875)
    assert ablation["ci95_pp"] == pytest.approx([-0.1036, 1.4786], abs=1e-4)
    assert ablation["participants_improved"] == 6


def test_stack_cli_rejects_an_inconsistent_published_rank_count(tmp_path, capsys):
    log = (ROOT / STACK_LOG).read_text(encoding="utf-8")
    published = tmp_path / STACK_LOG
    published.parent.mkdir(parents=True)
    published.write_text(log.replace("(1465/8000)", "(1466/8000)"), encoding="utf-8")
    assert main(["verify-stack", "--root", str(tmp_path)]) == 2
    assert "rank count differs" in capsys.readouterr().err


def test_o13_preflight_reports_missing_inputs_and_malformed_arrays(tmp_path, capsys):
    prepared = tmp_path / "data" / "prep"
    prepared.mkdir(parents=True)
    np.save(prepared / "sub-01_train.npy", np.zeros((2, 4, 17, 250), dtype=np.float32))
    (prepared / "F_train.npy").write_bytes(b"not an array")
    assert main(["check-o13-inputs", "--rad-root", str(tmp_path)]) == 2
    audit = json.loads(capsys.readouterr().out)
    assert audit["status"] == "missing_or_invalid"
    assert "data/prep/sub-10_test.npy" in audit["missing"]
    assert "data/prep/F_train.npy" in audit["invalid_arrays"]
    assert (
        "Expected shape (16540, 4, 17, 250)"
        in audit["invalid_arrays"]["data/prep/sub-01_train.npy"]
    )
