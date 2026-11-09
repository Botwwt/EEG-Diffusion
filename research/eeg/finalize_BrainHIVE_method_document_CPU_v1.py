"""Apply two reviewed wording corrections and preserve the preceding snapshot."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    document = ROOT / 'BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md'
    previous = json.loads((ROOT / 'audit/BrainHIVE_defect_first_execution_v1.json').read_text())
    assert sha(document) == previous['document_sha256']
    receipt = ROOT / 'audit/BrainHIVE_method_document_final_QA_v1.json'
    backup = ROOT / 'audit/BrainHIVE_method_before_final_review_v1.md'
    assert not receipt.exists() and not backup.exists()
    replacements = {
        '各划分按概念隔离，同概念的刺激和 EEG 重复不会跨划分泄漏。':
        '本轮新增拟合的各划分按概念隔离，同概念的刺激和 EEG 重复不跨 T/Dfit/Dcheck/V 拟合权限。现成先验和历史实验的曝光不会被此次划分消除。',
        '因此必须与同容量普通修正比较，':
        '因此必须与同一主干、同一方向分支的普通修正比较（风险组另外多 257 个参数），',
    }
    body = document.read_text()
    for old, new in replacements.items():
        assert body.count(old) == 1, old
        body = body.replace(old, new)
    shutil.copyfile(document, backup)
    document.write_text(body)
    checked = subprocess.run([sys.executable, str(ROOT / 'audit/check_BrainHIVE_method_document_CPU_v1.py')],
                             cwd=ROOT, text=True, capture_output=True, check=True)
    check = json.loads(checked.stdout)
    assert check['sha256'] == sha(document)
    campaign_path = ROOT / 'campaign_state.json'
    campaign = json.loads(campaign_path.read_text())
    campaign['method_document_sha256'] = sha(document)
    campaign['method_document_final_QA_receipt'] = str(receipt)
    campaign_path.write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + '\n')
    utc = datetime.now(timezone.utc).isoformat()
    result = dict(check, utc=utc, independent_fact_review_completed=True,
                  prior_document_preserved_at=str(backup), prior_document_sha256=sha(backup),
                  changes='Clarify split permissions versus historical exposure, and the 257-parameter risk-head difference.',
                  model_or_training_code_changed=False, queue_registration_changed=False)
    receipt.write_text(json.dumps(result, indent=2) + '\n')
    with (ROOT / 'experiment_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event_id='BrainHIVE_method_document_final_review_v1',
            UTC=utc, receipt=str(receipt), document_sha256=sha(document), GPU_jobs_started=0)) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
