"""Bind the ordered research handoff to unchanged scientific results and artifacts."""
from pathlib import Path
import copy
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parent
GALLERY = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/galleries/IDEA1_Frozen_RealPaired_Conditional_complete_20261020_v2')


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def artifact(path):
    path = Path(path)
    return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def main():
    started = time.monotonic()
    target = ROOT/'audit/final_artifact_index_20261020_v2.json'
    assert not target.exists()
    previous_path = ROOT/'audit/final_artifact_index_20261020_v1.json'
    previous = json.loads(previous_path.read_text())
    result = copy.deepcopy(previous)
    changed = {'current_status', 'report'}
    archived = {}
    for name, item in previous['artifacts'].items():
        path = Path(item['path'])
        if name in changed:
            snapshot = ROOT/'audit/author_first_before_20261020_v2'/path.name
            assert sha(snapshot) == item['sha256']
            archived[name] = artifact(snapshot)
            result['artifacts'][name] = artifact(path)
        else:
            assert sha(path) == item['sha256'], ('Protected earlier artifact changed', name)
    result['supersedes_index'] = artifact(previous_path)
    result['previous_mutable_artifacts_preserved_at'] = archived
    paths = {
        'author_based_sequence': ROOT/'configs/author_first_component_sequence_20261020_v2.json',
        'author_based_research': ROOT/'reports/ORDERED_AUTHOR_BASED_COMPONENT_RESEARCH_20261020.md',
        'DreamDiffusion_default_target_diagnosis': ROOT/'audit/DreamDiffusion_default_target_CPU_check_20261020_v1.json',
        'unused_environment_retirement': ROOT/'audit/unused_AVDE_environment_retirement_20261020_v1.json',
        'duplicate_transport_cleanup': ROOT/'audit/duplicate_transport_packet_retirement_20261020_v1.json',
        'thumbnail_deduplication': ROOT/'audit/gallery_thumbnail_deduplication_20261020_v1.json',
        'three_arm_gallery': GALLERY/'index.html',
        'three_arm_gallery_completion': GALLERY/'completion.json',
        'three_arm_gallery_context': GALLERY/'comparison_context.json',
        'three_arm_gallery_independent_check': GALLERY/'independent_materials_check_v1.json',
        'three_arm_gallery_CPU_delivery': ROOT/'audit/three_arm_gallery_CPU_delivery_20261020_v2.json',
        'partial_browser_check_failure_v3': ROOT/'audit/IDEA1_Frozen_RealPaired_Conditional_gallery_browser_QA_v3/failure_observation.json',
        'browser_check_failure_v4': ROOT/'audit/IDEA1_Frozen_RealPaired_Conditional_gallery_browser_QA_v4/report.json',
        'private_browser_QA_service_stopped': ROOT/'audit/three_arm_gallery_private_HTTP_QA_20261020_v3.json',
    }
    for name, path in paths.items():
        result['artifacts'][name] = artifact(path)
    evidence = json.loads((GALLERY/'independent_materials_check_v1.json').read_text())
    assert evidence['queries'] == 16600 and evidence['pairing_rows'] == 49800
    assert evidence['comparison_context_sha256'] == sha(GALLERY/'comparison_context.json')
    service = json.loads((ROOT/'audit/three_arm_gallery_private_HTTP_QA_20261020_v3.json').read_text())
    assert service['status'] == 'STOPPED_LOOPBACK_QA_SERVICE'
    assert not Path('/proc',str(service['pid'])).exists()
    for requirement in result['requirements']:
        if requirement['requirement'] == 'full_unified_gallery':
            requirement.update(status='PARTIAL_REQUIRED_SOURCES_WITH_COMPLETED_THREE_ARM_MATERIALS',
                               evidence_scope='Frozen/RealPaired/Conditional all16600 rows,49800 generated images,415 pages,exact identity and score checks;5required columns absent;browser UI verification incomplete after two navigation timeouts')
    result['requirements'].append(dict(requirement='ordered_author_code_reproduction_before_component',
                                       status='REGISTERED_NOT_YET_EXECUTED',
                                       evidence_scope='ATM first;full-parent out-of-fold contracts and independent mechanism controls;new GPU execution remains gated'))
    retirement = json.loads((ROOT/'audit/unused_AVDE_environment_retirement_20261020_v1.json').read_text())
    packets = json.loads((ROOT/'audit/duplicate_transport_packet_retirement_20261020_v1.json').read_text())
    dedup = json.loads((ROOT/'audit/gallery_thumbnail_deduplication_20261020_v1.json').read_text())
    released = retirement['actual_free_bytes_increase'] + packets['free_bytes_after']-packets['free_bytes_before'] + dedup['free_bytes_after']-dedup['free_bytes_before']
    assert released == 581492736
    result.update(status='AUTHOR_BASED_SEQUENCE_AND_SAVED_THREE_ARM_GALLERY_RECORDED_SCIENTIFIC_GOAL_UNMET',
                  UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                  storage_reclaimed_bytes=released, current_data_disk_free_bytes=shutil.disk_usage(GALLERY).free,
                  browser_UI_QA_complete=False, scientific_goal_complete=False,
                  comparable_SOTA_established=False, new_GPU_jobs=0, new_images=0,
                  experiment_outputs_recomputed=False, protected_protocol_metrics_or_query_seeds_changed=False,
                  source_sha256=sha(__file__), GPU_used=False, seconds=time.monotonic()-started)
    target.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as f:
        f.write(json.dumps(dict(stage='finalize_author_first_research_delivery',
                               seconds=result['seconds'], GPU_used=False, receipt=str(target), receipt_sha256=sha(target)))+'\n')
    print(json.dumps(dict(status=result['status'], artifacts=len(result['artifacts']),
                          storage_reclaimed_bytes=released, index_sha256=sha(target),
                          scientific_goal_complete=False, seconds=result['seconds'])), flush=True)


if __name__ == '__main__':
    main()
