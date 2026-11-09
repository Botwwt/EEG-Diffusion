"""Finite, preregistered readout of three T-only diagnostics; never starts a fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = 'BrainHIVE_decisive_T_direction_readout_20261025_v1'
STATUS = 'COMPLETE_FIXED_T_ONLY_DIRECTION_READOUT'
EXPECTED = {
    'layers': 'COMPLETE_T_ONLY_CPU_HELD_CONCEPT_PROBES',
    'function': 'COMPLETE_FROZEN_T_ONLY_FUNCTIONAL_DIAGNOSTIC',
    'logs': 'COMPLETE_FOUR_ARM_TRAINING_LOG_ANALYSIS_OPTIMIZATION_ONLY',
}
RESULTS = {'SUPPORTED', 'NOT_SUPPORTED', 'UNKNOWN'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def result(value):
    if isinstance(value, dict):
        value = value.get('result', value.get('status', 'UNKNOWN'))
    if value not in RESULTS:
        raise ValueError('Unregistered diagnostic result: ' + str(value))
    return value


def choose(layer_result, decisions, branches=None, log_decisions=None):
    """Priority is fixed before real outputs. Flags can coexist; none proves causality."""
    flags = []
    for name in ('teacher_discrimination', 'paired_EEG_signal', 'PN_over_O',
                 'PN_over_P', 'shuffled_response_control', 'trajectory_transfer',
                 'gradient_conflict', 'radius_saturation'):
        flags.append({'name': name, 'result': result(decisions.get(name, 'UNKNOWN'))})
    values = {item['name']: item['result'] for item in flags}
    # An invalid or uninformative teacher probe cannot establish a response defect.
    branches = branches or {}
    log_decisions = log_decisions or {}
    optimization = result(log_decisions.get('optimization_stability', 'UNKNOWN'))
    beyond_stats = result(branches.get('beyond_preLN_statistics', 'UNKNOWN'))
    pre_stats = result(branches.get('preLN_statistics', 'UNKNOWN'))
    if optimization == 'NOT_SUPPORTED':
        direction = 'RESOLVE_REGISTERED_OPTIMIZATION_ANOMALY_BEFORE_NEW_MECHANISMS'
        explanation = ('完整训练日志触发了事前定义的优化异常。下一步先定位该异常，'
                       '暂不把当前头的功能差异归因于新增机制，也不追加新架构。')
    elif pre_stats == 'SUPPORTED' and beyond_stats != 'SUPPORTED':
        direction = 'TEST_A_MINIMAL_PRE_LAYERNORM_STATISTICS_BYPASS_NEXT'
        explanation = ('LayerNorm 前均值、尺度的补充通过了相应对照；完整 h0 尚未证明具有超出这些统计量的增益。'
                       '下一候选先检验最小统计量旁路，避免把尺度信息写成更丰富的中间层语义。')
    elif layer_result == 'SUPPORTED':
        direction = 'TEST_A_MINIMAL_EEG_INTERMEDIATE_READOUT_HYPOTHESIS_NEXT'
        explanation = ('中间层联合读取通过预设的跨概念与错配对照筛选。下一候选应只检验局部读取或轻量融合，'
                       '匹配容量和预算，并通过实际生成干预确认；当前结果仅支持可预测性假设。')
    elif values['teacher_discrimination'] == 'NOT_SUPPORTED':
        direction = 'VALIDATE_TEACHER_PROBE_BEFORE_ANY_NEW_FUNCTIONAL_FIT'
        explanation = ('先核查教师响应是否衡量了配对语义；公共响应分量或诊断状态可能影响误差。'
                       '不能把该结果解释为生成器忽略 EEG，也不能据此直接训练更强响应头。')
    elif values['trajectory_transfer'] == 'NOT_SUPPORTED':
        direction = 'INVESTIGATE_TRAINING_STATE_VERSUS_SAMPLING_STATE_MISMATCH'
        explanation = ('按固定汇总规则，功能关系没有可靠延续到原生采样轨迹。下一设计优先检验轨迹一致监督，'
                       '先排除状态和坐标差异；不从当前层或时刻中挑选路由。')
    elif (values['gradient_conflict'] == 'SUPPORTED' and
          values['radius_saturation'] == 'SUPPORTED'):
        direction = 'TEST_OBJECTIVE_COMPETITION_WITH_A_FIXED_TRUST_REGION'
        explanation = ('固定 T 重放同时显示损失梯度竞争与半径饱和。下一设计可检验主目标约束下的辅助更新，'
                       '但此诊断不证明扩大半径或调整权重能改善图片。')
    elif all(values[key] == 'SUPPORTED' for key in
             ('PN_over_O', 'PN_over_P', 'shuffled_response_control', 'trajectory_transfer')):
        direction = 'VALIDATE_WHETHER_FUNCTIONAL_PROXY_TRANSFERS_TO_IMAGE_SEMANTICS'
        explanation = ('局部功能代理通过了固定对照；下一问题是代理改善能否转化为图像语义，'
                       '仍需独立的实际图像验证，不继续堆叠同类响应损失。')
    else:
        direction = 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED'
        explanation = ('这些固定诊断尚未给出足够一致的新机制线索。保留 UNKNOWN 和负结果，'
                       '先完成已有实验的完整报告，不自动增加候选或扩大搜索。')
    return {'priority_hypothesis': direction, 'interpretation_zh': explanation,
            'conditional_intermediate_readout': layer_result,
            'beyond_preLN_statistics': beyond_stats, 'preLN_statistics': pre_stats,
            'training_optimization_stability': optimization,
            'functional_evidence': flags, 'automatic_new_fit': False,
            'causal_mechanism_confirmed': False, 'image_benefit_established_by_this_readout': False}


def validate_summary(kind, payload, expected_status):
    if payload.get('status') != expected_status:
        raise ValueError(kind + ' analysis did not close with its registered full status')
    if kind == 'layers':
        assert payload['queries_per_lambda'] == 132300
        assert payload['no_V'] is True and payload['no_D'] is True and payload['no_test'] is True
    if kind == 'function':
        coverage = payload['coverage']
        assert coverage['expected_queries'] == 80
        assert coverage['expected_timesteps_per_domain'] == 4
        assert coverage['expected_domains'] == ['DDPM_real_T', 'native_parent_trajectory']
        assert coverage['observed_query_timestep_domain_cells'] == 640
        assert coverage['expected_native_IP_layers'] == 70
        assert coverage['missing_cells'] == 0
        assert payload['no_optimizer_created'] is True
        assert payload['parameter_updates'] == 0
        assert payload['protected_scores_read'] is False
        assert payload['generated_images'] == 0
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--registration', required=True, type=Path)
    args = parser.parse_args()
    config_sha = sha(args.registration)
    config = read(args.registration)
    assert config['schema'] == SCHEMA
    assert config['automatic_new_fit'] is False
    assert config['analysis_source_sha256'] == sha(__file__)
    output = Path(config['output_directory'])
    output.mkdir(parents=True, exist_ok=True)
    assert not (output / 'direction_summary.json').exists(), 'Preserve existing readouts'
    payloads, refs = {}, {}
    for kind in ('logs', 'layers', 'function'):
        item = config['summaries'][kind]
        path = Path(item['path'])
        assert path.stat().st_size <= 32 * 1024 * 1024
        before = sha(path)
        payloads[kind] = validate_summary(kind, read(path), item['status'])
        assert before == sha(path)
        refs[kind] = {'path': str(path), 'sha256': before, 'status': item['status']}
    decision = choose(result(payloads['layers']['main_decision']),
                      payloads['function']['diagnostic_decisions'], payloads['layers']['branches'],
                      payloads['logs']['diagnostic_decisions'])
    assert sha(args.registration) == config_sha
    report = {'status': STATUS, 'registration': {'path': str(args.registration), 'sha256': config_sha},
              'inputs': refs, 'decision': decision, 'protected_scores_read': False,
              'official_test_inputs_read': False, 'automatic_new_fit': False,
              'scientific_goal_complete': False, 'independent_SOTA_established': False,
              'decision_rule': 'Fixed priority, full registered coverage, no layer/time/sample cherry-picking',
              'limits': ['Frozen T diagnostics and finite readouts cannot establish image improvement.',
                         'Probe threshold passing does not prove mutual information or causal loss of semantics.',
                         'R4 development images, official R80 and unseen confirmation remain separate.']}
    target = output / 'direction_summary.json'
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = ['# 固定 T 诊断后的下一步方向', '', decision['interpretation_zh'], '',
             '- 优先假设：`' + decision['priority_hypothesis'] + '`。',
             '- 中间层联合读取：`' + decision['conditional_intermediate_readout'] + '`。',
             '- 本判读不会启动新拟合；未建立图片收益、因果机制或 SOTA。', '',
             '| 功能证据 | 固定判读 |', '|---|---|']
    lines += ['| ' + row['name'] + ' | ' + row['result'] + ' |' for row in decision['functional_evidence']]
    lines += ['', '完整来源与限制见 [direction_summary.json](direction_summary.json)。', '']
    (output / 'direction_report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps({'status': STATUS, 'priority_hypothesis': decision['priority_hypothesis'],
                      'summary_sha256': sha(target), 'automatic_new_fit': False}))


if __name__ == '__main__':
    main()
