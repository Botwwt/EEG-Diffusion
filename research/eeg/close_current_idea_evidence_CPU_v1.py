"""Persist completed evidence and an unlaunched reproduction, without new inference."""
from pathlib import Path
import collections
import calendar
import csv
import hashlib
import io
import json
import time

ROOT = Path(__file__).resolve().parent
DATA = Path('/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1')


def sha(path):
    with Path(path).open('rb') as file:
        return hashlib.file_digest(file, 'sha256').hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def main():
    started = time.monotonic()
    now = time.time()
    utc = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(now))
    receipt_path = ROOT/'audit/current_idea_closure_and_unlaunched_decoder_20261020_v1.json'
    assert not receipt_path.exists(), 'Append-only evidence closure; do not repeat'
    closure_path = ROOT/'audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json'
    closure = load(closure_path)
    assert closure['status'] == 'PASSED_ALL16600_COMPLETED_SCORE_IDENTITY_AND_PAIRED_ARITHMETIC'
    assert closure['completed_arms'] == ['Frozen', 'RealPaired', 'Conditional']
    registration_path = ROOT/'configs/ATM_public_decoder_registration_20261020_v1.json'
    cfg = load(registration_path)
    assert cfg['scope'] == 'ALL2000_PUBLISHED_FEATURE_DECODER_REPRODUCTION_ONLY'
    for path, expected in cfg['source_sha256'].items():
        assert sha(path) == expected, path
    excluded = set(cfg['source_sha256']) - set(cfg['inference_source_sha256'])
    assert excluded == {cfg['target_mapping'], cfg['gallery_manifest']}
    for name in ['generate', 'prepare', 'audit', 'run']:
        source = ROOT/(name+'_ATM_published_decoder_v1.py')
        compile(source.read_text(), str(source), 'exec')
    inputs = [json.loads(x) for x in Path(cfg['inference_manifest']).read_text().splitlines()]
    targets = [json.loads(x) for x in Path(cfg['target_mapping']).read_text().splitlines()]
    galleries = [json.loads(x) for x in Path(cfg['gallery_manifest']).read_text().splitlines()]
    permitted = {'query_id', 'subject', 'stimulus_id', 'array_row', 'image_seed', 'prior_seed'}
    assert len(inputs) == len(targets) == len({x['query_id'] for x in inputs}) == 2000
    assert len(galleries) == 200
    expected_subjects = {f'sub-{i:02d}': 200 for i in range(1, 11)}
    assert collections.Counter(x['subject'] for x in inputs) == expected_subjects
    assert all(set(x) == permitted for x in inputs)
    target_index = {x['query_id']: x for x in targets}
    for x in inputs:
        assert all(x[k] == target_index[x['query_id']][k] for k in permitted)
    for g in galleries:
        assert sha(g['image_path']) == g['image_sha256']
    for asset in cfg['subjects'].values():
        assert sha(asset['features']) == asset['features_sha256']
        assert sha(asset['prior']) == asset['prior_sha256']
    out = Path(cfg['output_root'])
    assert not (out/'inference_manifest.jsonl').exists()
    assert not (out/'completion.json').exists()
    assert not (out/'images').exists()
    assert not (ROOT/'audit/ATM_published_decoder_state_20261020_v1.json').exists()
    bounds = load(ROOT/'configs/compute_time_bounds.json')
    assert now > bounds['conservative_occupied_time_deadline_epoch']
    ledger = [json.loads(x) for x in (ROOT/'audit/compute_ledger.jsonl').read_text().splitlines()]
    charged = sum(float(x.get('charged_seconds', 0)) for x in ledger)
    assert not any(x.get('name', '').startswith('ATM_public_decoder_') for x in ledger)
    current = load(ROOT/'audit/current_progress.json')
    assert current['live_GPU_jobs'] == [], 'Observer must verify no live GPU job before closure'
    assert 0 <= now - calendar.timegm(time.strptime(current['utc'], '%Y-%m-%dT%H:%M:%SZ')) < 180
    costs = {}
    for arm in ['Frozen', 'RealPaired', 'Conditional']:
        arm_jobs = [x for x in ledger if x.get('name', '').startswith('IDEA1_'+arm+'_')]
        assert all(x['status'] == 'COMPLETED' and x['exit_code'] == 0 for x in arm_jobs)
        costs[arm] = {x['name']: x['charged_seconds'] for x in arm_jobs}
    result = dict(
        UTC=utc, status='COMPLETE_THREE_ARM_EVIDENCE_AND_NOT_LAUNCHED_AUTHOR_DECODER',
        source_sha256=sha(__file__), GPU_used=False, new_images=0, training_started=False,
        score_closure_sha256=sha(closure_path), queries_per_completed_arm=16600,
        completed_arms=closure['completed_arms'], per_arm_GPU_job_seconds=costs,
        charged_GPU_job_hours_including_historical_allowances=charged/3600,
        conservative_occupied_time_deadline_utc=bounds['deadline_utc'],
        ledger_remaining_hours=max(0, 40-charged/3600), tighter_remaining_authorized_hours=0,
        resource_limit_changed=False, provider_billing_independently_verified=False,
        author_decoder=dict(status='NOT_LAUNCHED_OCCUPIED_TIME_WINDOW_EXPIRED',
            registration_sha256=sha(registration_path), registered_queries=2000,
            source_files_unchanged=True, sources_compile=True, input_target_metadata_join=True,
            all200_gallery_hashes_verified=True, all20_author_asset_hashes_verified=True,
            GPU_jobs=0, generated_images=0, source_encoder_reproduced=False,
            end_to_end_EEG_image_reproduction=False,
            prior_queue_submission_exists=False,
            full_registered_lifecycle_reservation_seconds=sum(cfg['phase_caps_seconds'].values())+cfg['reserve_seconds']+45),
        independent_component_increment_established=False, comparable_SOTA_established=False,
        scientific_goal_complete=False,
        limits=['Closure verifies completed saved evidence, not a new experiment.',
                'New author decoder entry points have not executed a complete pipeline.',
                'No new GPU job may start under the unchanged expired occupied-time bound.',
                'Missing controls and missing reproductions remain incomplete, not negative results.'])

    # Preserve every previous bookkeeping byte before updating stale current summaries.
    archive = ROOT/'audit/current_evidence_before_closure_20261020_v1'
    archive.mkdir()
    for name in ['campaign_state.json', 'STATUS.md', 'research/hypothesis_status.json', 'results.tsv']:
        path = ROOT/name
        destination = archive/name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(path.read_bytes())
    state = load(ROOT/'campaign_state.json')
    state.update(stage='REGISTERED_COMPUTE_WINDOW_EXPIRED_THREE_IMAGE_ARMS_COMPLETE',
        last_update_utc=utc, verified_wait=[], active_controller_pid=None,
        active_state_path=None, remaining_GPU_cap_hours=0, remaining_authorized_hours=0,
        last_goal_turn_classification='progress', new_EEG_image_gain_established=False,
        exact_full_ATM_reproduction_claimed=False, scientific_goal_complete=False,
        author_decoder_status=result['author_decoder']['status'],
        evidence_closure_receipt=str(receipt_path))
    (ROOT/'campaign_state.json').write_text(json.dumps(state, indent=2)+'\n')
    hypotheses = load(ROOT/'research/hypothesis_status.json')
    hypotheses['as_of'] = utc
    for h in hypotheses['hypotheses']:
        if h['id'] == 'H_ERROR_ADAPTED_IMAGE_INTERFACE':
            h.update(status='CURRENT_PROJECTOR_FINETUNING_IMAGE_GAIN_REJECTED_ON_COMPLETE_V',
                scope='All10 subjects,16600 matched R4 images each. Frozen beats RealPaired and Conditional on DINO2AFC,SEED,CapSim,ObjectF1. Conditional minusRealPaired DINO interval crosses0;ObjectF1 decreases. This rejects this registered implementation on V, not all error-calibration mechanisms. Moment/Iso/Clean/full EEG intervention controls remain unrun. No capacity cause established.')
            h['evidence'] += ['audit/idea_Frozen_RealPaired_Conditional_full_score_closure_v1.json',
                              'reports/IDEA_INTERFACE_COMPARISON.md',
                              'reports/THREE_ARM_CLOSURE_AND_REPRODUCTION_NEXT_20261020.md']
        if h['id'] == 'H_NATIVE_ERROR_CHANNEL':
            h.update(status='RELATIVE_FIDELITY_SIGNALS_DO_NOT_PASS_ABSOLUTE_DISTRIBUTION_GATE',
                scope=h['scope']+' Subsequent q-v2 fixed CPU audit fails absolute classifier/support fidelity; no portable simulator established. See IDEA_QV2_CPU_DIAGNOSIS_20261019.md.')
            h['evidence'] += ['reports/IDEA_QV2_CPU_DIAGNOSIS_20261019.md']
    (ROOT/'research/hypothesis_status.json').write_text(json.dumps(hypotheses, indent=2)+'\n')

    table = ROOT/'results.tsv'
    existing = list(csv.DictReader(io.StringIO(table.read_text()), delimiter='\t'))
    fields = next(csv.reader(io.StringIO(table.read_text()), delimiter='\t'))
    additions = []
    for arm in ['Frozen', 'RealPaired', 'Conditional']:
        run_id = 'IDEA1_'+arm+'_full_V_R4_16600'
        assert not any(x['run_id'] == run_id for x in existing)
        summary = load(DATA/'evaluation/V'/('IDEA1_'+arm+'_correct_R4/metrics/summary.json'))
        m = summary['summary']
        row = dict.fromkeys(fields, '')
        row.update(run_id=run_id, parent_run='IDEA1_ATMS_patience50_native_REG_SDXL512',
            hypothesis='Native-projector image adaptation vs unchanged pretrained projector',
            status='COMPLETED_FULL_IMAGES_METRICS_IDENTITY',
            protocol_version='IDEA1_interface_v1_full_V_R4', train_seed=1729,
            subject_scope='all10', R=4, trainable_params=0 if arm=='Frozen' else 8400896,
            train_images=0 if arm=='Frozen' else 13230,
            train_updates=0 if arm=='Frozen' else 8269,
            effective_epochs=0 if arm=='Frozen' else 1,
            gpu_hours=sum(costs[arm].values())/3600, eval_queries=16600,
            expected_images=16600, generated_images=16600, failed_images=0,
            dino_2afc=m['DINO2AFC']['mean'], semantic_metric=m['SEED']['mean'],
            object_f1=m['ObjectF1']['mean'],
            keep_decision='retain_best_matched_reference' if arm=='Frozen' else 'reject_current_image_gain',
            reason='Shared encoder/OOF/evaluator prerequisites charged separately; evidence closure contains immutable per-query/checkpoint hashes. V is development data, no comparable SOTA or independent confirmation.')
        if arm!='Frozen':
            training = load(DATA/'runs'/('IDEA1_interface_'+arm+'_v1/completion.json'))
            row['adapter_ckpt'] = training['checkpoint']
            pair = load(DATA/'evaluation/paired'/('IDEA1_'+arm+'_minus_Frozen_V_R4/summary.json'))
            p = pair['metrics']['DINO2AFC']
            row.update(delta_vs_best_matched_ref=p['paired_delta'],
                       paired_ci_low=p['CI95_subject_concept'][0], paired_ci_high=p['CI95_subject_concept'][1])
        additions.append(row)
    with table.open('a', newline='') as file:
        csv.DictWriter(file, fieldnames=fields, delimiter='\t', lineterminator='\n').writerows(additions)
    status = (f'# 当前已核实状态\n\n更新：{utc}。无正在运行的 GPU 实验。\n\n'
              'Frozen、RealPaired、Conditional 各完成十被试 16,600 张图片、逐 query 评价和配对核验。'
              'Frozen 的 DINO2AFC/SEED 均优于两种本轮投影器微调；当前实现没有图片增益。\n\n'
              '作者 ATM 的二十个匹配公开 prior/EEG 特征已核验，完整 2,000 query 解码入口与映射已登记，尚未提交运行。'
              '它不是已完成的端到端 EEG 复现；没有新解码图片或组件增量。\n\n'
              f'GPU 作业及历史保守额度合计 {charged/3600:.6f} 小时；40 小时占用窗口截止 '
              f'{bounds["deadline_utc"]}，更严格的授权剩余为零。未改变资源上限或启动新 GPU 任务。\n\n'
              '完整结论见 reports/THREE_ARM_CLOSURE_AND_REPRODUCTION_NEXT_20261020.md。'
              '未完成的方法、对照、独立确认和跨架构迁移仍列为未完成，目标未达成。\n')
    (ROOT/'STATUS.md').write_text(status)
    result['seconds'] = time.monotonic()-started
    receipt_path.write_text(json.dumps(result, indent=2)+'\n')
    with (ROOT/'audit/CPU_postprocessing_ledger.jsonl').open('a') as file:
        file.write(json.dumps(dict(event_id='current_idea_closure_and_unlaunched_decoder_20261020_v1',
            UTC=utc, seconds=result['seconds'], GPU_used=False, receipt=str(receipt_path), receipt_sha256=sha(receipt_path)))+'\n')
    with (ROOT/'events.jsonl').open('a') as file:
        file.write(json.dumps(dict(classification='progress', time_utc=utc,
            event_id='complete_three_arm_closure_and_preserve_unlaunched_author_decoder',
            receipt=str(receipt_path), receipt_sha256=sha(receipt_path), scientific_goal_complete=False))+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
