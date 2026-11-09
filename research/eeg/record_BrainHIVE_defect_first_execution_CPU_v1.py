"""Record the user's defect-first decision, actual running jobs and method guide."""
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
WORK = Path('/root/brainhive_reproduction_20261020_v1')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    receipt_path = ROOT / 'audit/BrainHIVE_defect_first_execution_v1.json'
    assert not receipt_path.exists()
    main_recipe = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
    assert sha(main_recipe) == '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
    assert sha(ROOT / 'configs/BrainHIVE_two_ideas_sequence_v1.json') == '809a16b87d44a1dd28301ff8f41cb0d82d84e6f25c77a31024aeff8f7431d93a'
    old_queue = read(WORK / 'audit/two_ideas_queue_v1.json')
    assert old_queue['status'] == 'WITHDRAWN_BEFORE_NEW_TRAINING_PENDING_MECHANISM_REVIEW'
    assert old_queue['automatic_launch_enabled'] is False
    launch = read(WORK / 'audit/native_function_T32_queue_launch_v1.json')
    queue = read(WORK / 'audit/native_function_T32_queue_v1.json')
    assert launch['pid'] == queue['pid']
    os.kill(queue['pid'], 0)
    command = Path(f"/proc/{queue['pid']}/cmdline").read_bytes()
    assert b'queue_BrainHIVE_native_function_T32_v1.py' in command
    os.kill(185943, 0)
    progress = read(WORK / 'components/bounded_v1/risk/progress.json')
    assert progress['status'] == 'RUNNING' and progress['arm'] == 'risk'
    assert not (WORK / 'two_ideas_v1/recipes').exists()
    assert not (WORK / 'runs/brainkan_full_v1').exists()
    assert not (WORK / 'components/two_ideas_v1/response_rank_v1').exists()
    document = ROOT / 'BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md'
    utc = datetime.now(timezone.utc).isoformat()
    client_time = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
    before = ROOT / 'audit/before_BrainHIVE_defect_first_execution_v1'
    assert not before.exists()
    before.mkdir()
    names = ['STATUS.md', 'campaign_state.json', 'results.tsv', 'experiment_ledger.jsonl', 'budget.yaml']
    for name in names:
        shutil.copyfile(ROOT / name, before / name)
    scale = read(ROOT / 'audit/BrainHIVE_scale_and_pairing_CPU_v1.json')['mean']
    geometry = read(ROOT / 'audit/BrainHIVE_condition_geometry_CPU_v1.json')['aggregate']['T_OOF']
    record = dict(
        status='EXISTING_MAIN_RUNNING_KAN_PARKED_FROZEN_T_DIAGNOSTIC_QUEUED',
        UTC=utc, client_time=client_time, client_timezone='America/Los_Angeles',
        latest_user_decision='Use KAN only if it specifically addresses a supported host defect; detailed method Markdown requested.',
        previous_automatic_two_idea_queue=old_queue, frozen_diagnostic_queue=queue,
        existing_main_training=progress, existing_training_pid=185943,
        main_recipe_sha256=sha(main_recipe), new_diagnostic_updates=0, new_diagnostic_images=0,
        KAN_decision='Parked before formal training: generic encoder substitution has no established defect-specific rationale.',
        response_rank_decision='Formal training deferred pending frozen native functional evidence; no automatic fit.',
        established_facts=dict(raw_OOF_matched_cosine=geometry['raw_cosine'],
            raw_OOF_deranged_cosine=geometry['deranged_raw_cosine'], **scale),
        unsupported_claims=['EEG conditioning collapse', 'P/IP destroyed stimulus information',
                            'BrainMLP is too small', 'KAN improves generation or physiological interpretation',
                            'New component image gain', 'Cross-architecture generality'],
        diagnostic_scope=dict(T_queries=32, subject='sub-01', native_time_indices=4,
            condition_arms=5, same_missing_reference_Q=True, parameter_updates=0,
            actual_sampling_trajectory=False, component_training_after_diagnostic_automatic=False),
        requested_document=str(document), document_sha256=sha(document), document_bytes=document.stat().st_size,
        original_EEG_unique_weights_and_historical_images_deleted=0,
        large_local_C_D_files_created=False, new_candidate_configurations=0,
        previous_two_registrations_retained_in_history=True,
    )
    receipt_path.write_text(json.dumps(record, indent=2) + '\n')
    opening = f'''# 当前状态：先验证宿主缺陷，KAN 暂缓

更新 {client_time}（America/Los_Angeles；服务器 UTC {utc}）。

原有 risk→ordinary→parent 生命周期继续，当前 risk 为 {progress['updates']}/24807 更新，正式组件图片比较尚未闭合。原生 R80 单 seed、单图配对干预已闭合，不能写成新组件增益。

旧 KAN→响应排序自动队列已在任何新正式训练前撤下；源代码和原始注册保留。KAN 缺少针对已证实宿主缺陷的依据，暂不使用。响应排序仅保留候选，须先看实际冻结功能诊断，没有自动正式训练。

新队列 PID {queue['pid']} 已核验存活，目前 `{queue['status']}`。它在原主生命周期结束且 GPU 空闲后，只执行一次固定 T32、5 条件、同一缺失参考 Q 的冻结诊断，最多 1200 秒；不拟合、不出图、不看 D/V/最终任务评分，不自动追加训练。

详细方法见 [BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md](BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md)，实际回执见 [audit/BrainHIVE_defect_first_execution_v1.json](audit/BrainHIVE_defect_first_execution_v1.json)。此前状态保留如下；旧“新两组自动启动”和“约13.12GB两组缺口”属于已撤下计划的历史状态。

'''
    (ROOT / 'STATUS.md').write_text(opening + (ROOT / 'STATUS.md').read_text())
    campaign = read(ROOT / 'campaign_state.json')
    campaign.update(last_update_utc=utc, current_main_component='bounded_direction_and_conditional_error_gate',
        KAN_optional_and_parked=True, next_two_ideas_queue=old_queue,
        next_two_ideas_admitted=False, next_two_ideas_formal_training_started=False,
        next_two_ideas_image_gain=False, frozen_T32_diagnostic_queue=queue,
        response_rank_formal_training_enabled=False, next_two_ideas_storage_forecast_active=False,
        requested_method_document=str(document), method_document_sha256=sha(document))
    (ROOT / 'campaign_state.json').write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + '\n')
    result_path = ROOT / 'results.tsv'
    original = result_path.read_bytes()
    lines = original.decode().splitlines(keepends=True)
    fields = next(csv.reader([lines[0]], delimiter='\t'))
    templates = [read(ROOT / 'configs/BrainHIVE_BrainKAN_two_ideas_template_v1.json'),
                 read(ROOT / 'configs/BrainHIVE_response_rank_two_ideas_template_v1.json')]
    changed = []
    for template, status, reason in zip(templates,
            ['parked_before_training_no_defect_specific_evidence', 'pending_frozen_native_functional_diagnostic'],
            ['Optional KAN withdrawn before formal fit; no demonstrated host defect targeted; source and original registration retained.',
             'Unstarted automatic fit withdrawn; native T-only fixed-Q diagnostic queued; no component image gain established.']):
        indices = [i for i, line in enumerate(lines) if line.startswith(template['run_name'] + '\t')]
        assert len(indices) == 1
        index = indices[0]
        row = dict(zip(fields, next(csv.reader([lines[index]], delimiter='\t'))))
        assert row['train_updates'] == '0' and row['generated_images'] == '0'
        row.update(status=status, reason=reason, keep_decision='unresolved_not_trained')
        stream = io.StringIO(newline='')
        csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n').writerow(row)
        lines[index] = stream.getvalue()
        changed.append(index)
    rebuilt = ''.join(lines).encode()
    untouched_before = original.decode().splitlines(keepends=True)
    assert all(lines[i] == untouched_before[i] for i in range(len(lines)) if i not in changed)
    result_path.write_bytes(rebuilt)
    with (ROOT / 'experiment_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event_id='BrainHIVE_defect_first_review_and_method_document_v1',
            UTC=utc, receipt=str(receipt_path), KAN_parked=True,
            response_formal_training_enabled=False, frozen_diagnostic_pid=queue['pid'],
            new_GPU_jobs_started=0, historical_metric_rows_changed=False)) + '\n')
    with (ROOT / 'budget.yaml').open('a') as stream:
        stream.write('\nBrainHIVE_defect_first_review_v1:\n  KAN_formal_training_enabled: false\n'
            '  response_rank_formal_training_enabled: false\n  previous_two_registrations_retained: true\n'
            '  new_candidate_configurations: 0\n  frozen_T32_diagnostic_jobs: 1\n'
            '  diagnostic_maximum_seconds: 1200\n  diagnostic_parameter_updates: 0\n'
            '  diagnostic_generated_images: 0\n  old_two_candidate_storage_forecast_active: false\n')
    print(json.dumps(dict(status=record['status'], UTC=utc, client_time=client_time,
        current_risk_updates=progress['updates'], diagnostic_queue_pid=queue['pid'],
        method_document_bytes=document.stat().st_size, new_formal_training=False)), flush=True)


if __name__ == '__main__':
    main()
