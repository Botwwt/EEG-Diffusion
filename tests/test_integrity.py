"""Integrity failures must be explicit; they must not disappear under python -O."""

import hashlib
import json

import pytest

from eeg_diffusion.audit import verify_manifest
from eeg_diffusion.cli import main


def test_manifest_rejects_changed_evidence_and_duplicate_paths(tmp_path):
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"value": 1}', encoding="utf-8")
    record = {"path": "evidence.json", "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest()}
    assert verify_manifest(tmp_path, {"files": [record]}) == 1
    with pytest.raises(ValueError, match="Duplicate"):
        verify_manifest(tmp_path, {"files": [record, record]})
    evidence.write_text('{"value": 2}', encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_manifest(tmp_path, {"files": [record]})


def test_manifest_rejects_paths_outside_release(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        verify_manifest(tmp_path, {"files": [{"path": "../outside.json", "sha256": "unused"}]})


def test_cli_demo_has_explicit_expected_output(tmp_path, capsys):
    output = tmp_path / "demo.json"
    assert main(["reference-demo", "--out", str(output)]) == 0
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["synthetic"] is True
    assert result["unique_trials"] == 2
    assert result["posterior_mean"][0] == pytest.approx(0.4)
    assert result["posterior_covariance"][0][0] == pytest.approx(0.5)
    assert "synthetic" in capsys.readouterr().out


def test_cli_reports_missing_release_without_traceback(tmp_path, capsys):
    assert main(["verify", "--root", str(tmp_path)]) == 2
    assert "eeg-diffusion:" in capsys.readouterr().err
