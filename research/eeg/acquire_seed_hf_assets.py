"""Public evaluator weights only; no EEG upload, training, or local disk weights."""
from pathlib import Path
import json
import traceback
from acquire_sdxl_base import acquire_snapshot

ROOT=Path(__file__).resolve().parent
OUT=Path('/dev/shm/eeg_sota_autoresearch_20261018_v1/semantic_evaluation')
GROUPS={'microsoft/git-large-coco':'git_large_coco',
        'sentence-transformers/all-MiniLM-L6-v2':'sentence_minilm',
        'google-bert/bert-base-uncased':'bert_base'}
try:
    manifests=json.loads((ROOT/'audit/seed_hf_selected_assets.json').read_text())
    for manifest in manifests:
        acquire_snapshot(manifest,OUT/GROUPS[manifest['repository']])
    (ROOT/'audit/seed_hf_acquisition_completion.json').write_text(json.dumps(
        dict(status='PUBLIC_HF_EVALUATOR_WEIGHTS_VERIFIED',directories=GROUPS,
             local_large_files_created=0,model_or_metric_calls=0),indent=2)+'\n')
except Exception as exc:
    (ROOT/'audit/seed_hf_acquisition_failure.json').write_text(json.dumps(
        dict(status='FAILED_RESUMABLE',type=type(exc).__name__,message=str(exc),
             traceback=traceback.format_exc()),indent=2)+'\n')
    raise
