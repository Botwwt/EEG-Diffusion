"""Record a verified resumed producer, preserving old failures and method history."""
from datetime import datetime, timezone
import hashlib
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
    receipt = ROOT / 'audit/BrainHIVE_storage_recovery_running_20261021_v1.json'
    assert not receipt.exists()
    launch = read(ROOT / 'audit/BrainHIVE_storage_recovery_verified_launch_20261021_v1.json')
    state = read(WORK / 'audit/main_storage_continuation_20261021_v1.json')
    diagnostic = read(WORK / 'audit/native_function_T32_queue_v2.json')
    os.kill(state['pid'], 0)
    os.kill(diagnostic['pid'], 0)
    assert state['status'] == 'RUNNING_ORIGINAL_MAIN_STORAGE_CONTINUATION'
    assert state['current_arm'] == 'risk'
    manifest = WORK / 'images/protected_R4_risk_bounded_v1/inference_manifest.jsonl'
    with manifest.open('rb') as stream:
        prefix = stream.read(launch['existing_image_manifest_prefix_bytes'])
    assert hashlib.sha256(prefix).hexdigest() == launch['existing_image_manifest_prefix_sha256']
    rows = [json.loads(line) for line in manifest.read_text().splitlines() if line.strip()]
    assert len(rows) > 11000 and [r['global_query_index'] for r in rows] == list(range(len(rows)))
    assert sha(WORK / 'components/bounded_v1/risk/component_final.pt') == launch['risk_checkpoint_sha256']
    assert all(sha(p) == value for p, value in launch['old_failed_state_sha256'].items())
    recipe_path = ROOT / 'configs/BrainHIVE_bounded_main_recipe_v1.json'
    assert sha(recipe_path) == '55d92772f3e76240799cf69df12f590229a408c7b7a48050b9a0c7ec01f31db3'
    from brainhive_lifecycle_storage_v1 import verify_remaining
    storage = verify_remaining(read(recipe_path))
    utc = datetime.now(timezone.utc).isoformat()
    client = datetime.now(ZoneInfo('America/Los_Angeles')).isoformat()
    record = dict(status='RISK_IMAGE_COUNT_ADVANCING_AFTER_VERIFIED_STORAGE_RECOVERY', UTC=utc, client_time=client,
        risk_training_updates=24807, expected_training_updates=24807, risk_formal_images=len(rows), expected_images=16600,
        original_11000_manifest_prefix_preserved=True, original_risk_checkpoint_preserved=True,
        risk_training_repeated=False, original_failed_states_preserved=True,
        old_failure_cause='Private DATA remaining-life forecast fell short by 7868416 bytes; no model-quality failure.',
        recovery='24 inactive derived ATM feature caches retained byte-for-byte at private RAM symlinks; DATA freed1207615488 bytes.',
        diagnostic_fix='v1 imported diffusers before native runtime setup; v2 only repairs import order, same input/noise/arms.',
        native_runtime_import_CPU_verified=launch['diagnostic_runtime_check'],
        main_queue=state, diagnostic_queue=diagnostic, current_remaining_forecast=storage,
        score_comparison_available=False, component_image_gain_established=False,
        raw_EEG_learned_weights_or_historical_images_deleted=0, RAM_is_persistent_backup=False,
        additional_root_cache_move='Deferred while active generator leaves less than required4GiB RAM headroom; no original cache change at failed admission.')
    backup = ROOT / 'audit/before_BrainHIVE_storage_recovery_20261021_v1'
    assert not backup.exists()
    backup.mkdir()
    for name in ['STATUS.md', 'campaign_state.json', 'experiment_ledger.jsonl', 'BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md']:
        shutil.copyfile(ROOT / name, backup / name)
    guide = ROOT / 'BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md'
    body = guide.read_text()
    title, rest = body.split('\n', 1)
    update = f'''\n> **执行更新 {client}（America/Los_Angeles）**：risk 已完成 24807 次更新；出图曾在 11000 张因存储预算保护暂停，已释放约1.21GB持久空间并核验续跑，当前 {len(rows)}/16600 张。旧图片和检查点保持不变，没有重训 risk。原生 R4 与普通对照尚未完成，没有组件图片增益结论。T32 诊断 v1 在导入依赖时失败，修复为 [v2](diagnose_BrainHIVE_native_function_T32_v2.py)，仍等待主生命周期结束；方法、输入与噪声规则没有修改。详见 [恢复回执](audit/BrainHIVE_storage_recovery_running_20261021_v1.json) 和 [STATUS.md](STATUS.md)。下面保留原方法快照。\n'''
    guide.write_text(title + '\n' + update + rest)
    record['method_document_sha256'] = sha(guide)
    receipt.write_text(json.dumps(record, indent=2) + '\n')
    opening = f'''# 当前状态：存储故障已修复，risk 正在补齐原图\n\n更新 {client}（America/Los_Angeles；UTC {utc}）。risk 训练已完成24807次更新，正式图片当前 {len(rows)}/16600。原队列在11000张触发DATA生命周期空间保护后，普通与原生组也因同一前置条件失败；旧失败回执保留。这是运行故障，不是科学负结果。\n\n已将24个不参与当前主线的可重建特征缓存迁入服务器私有RAM并保留原路径链接，逐文件SHA相同，释放1,207,615,488持久字节。RAM不是持久备份；原始EEG、学习权重和历史原始图片保留。原空间检查通过后，新控制器PID {state['pid']}恢复原生成入口；已验证旧11000行字节前缀及risk检查点均未变化，没有重新训练risk。\n\n后续仍按risk完整出图/评价→ordinary原配方训练/出图/评价→parent出图/评价。T32仅修复导入顺序，v2队列PID {diagnostic['pid']}等待上述主线。当前没有三组配对分数、没有图片增益结论，KAN继续暂缓。详见 [恢复回执](audit/BrainHIVE_storage_recovery_running_20261021_v1.json) 与 [方法说明](BRAINHIVE_CURRENT_METHOD_EXPLAINED_20261020.md)。\n\n此前快照保留如下：\n\n'''
    (ROOT / 'STATUS.md').write_text(opening + (ROOT / 'STATUS.md').read_text())
    campaign = read(ROOT / 'campaign_state.json')
    campaign.update(last_update_utc=utc, current_main_continuation=state,
        risk_training_complete=True, risk_training_updates=24807, risk_formal_images_at_snapshot=len(rows),
        frozen_T32_diagnostic_queue=diagnostic, original_main_failure_preserved=True,
        current_storage_recovery_receipt=str(receipt), method_document_sha256=sha(guide),
        component_image_gain_established=False)
    (ROOT / 'campaign_state.json').write_text(json.dumps(campaign, indent=2, ensure_ascii=False) + '\n')
    with (ROOT / 'experiment_ledger.jsonl').open('a') as stream:
        stream.write(json.dumps(dict(event_id='BrainHIVE_existing_main_storage_only_recovery_20261021_v1',
            UTC=utc, receipt=str(receipt), risk_training_repeated=False, new_candidate_configurations=0,
            risk_formal_images=len(rows), image_gain_established=False)) + '\n')
    print(json.dumps(dict(status=record['status'], client_time=client, risk_images=len(rows),
        main_pid=state['pid'], diagnostic_pid=diagnostic['pid'], scientific_result_complete=False)), flush=True)


if __name__ == '__main__':
    main()
