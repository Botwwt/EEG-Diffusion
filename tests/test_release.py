from pathlib import Path
import json
from eeg_diffusion.audit import verify_release

ROOT = Path(__file__).resolve().parents[1]


def test_released_evidence_integrity_and_arithmetic():
    report = verify_release(ROOT)
    assert report['source_assets_verified'] >= 70
    assert abs(report['retrieval_difference_pp'] - 47.5) < 1e-10


def test_every_displayed_retrieval_seed_is_present_once():
    data = json.loads((ROOT/'results/summary.json').read_text(encoding='utf-8'))
    rows = data['retrieval']['rows']
    assert len({(x['method'],x['seed']) for x in rows}) == len(rows) == 6
    assert data['retrieval']['repeats'] == 80

