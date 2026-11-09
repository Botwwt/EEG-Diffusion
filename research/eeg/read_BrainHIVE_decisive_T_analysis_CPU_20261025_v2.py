"""Finite, preregistered readout of three T-only diagnostics; never starts a fit."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCHEMA = 'BrainHIVE_decisive_T_direction_readout_20261025_v2'
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
    """Fixed hypothesis triage; unknown prerequisites cannot satisfy a gate."""
    names = ('teacher_discrimination', 'paired_EEG_signal', 'PN_over_O', 'PN_over_P',
             'shuffled_response_control', 'trajectory_transfer', 'gradient_conflict', 'radius_saturation')
    values = {name: result(decisions.get(name, 'UNKNOWN')) for name in names}
    flags = [{'name': name, 'result': value} for name, value in values.items()]
    branches, log_decisions = branches or {}, log_decisions or {}
    optimization = result(log_decisions.get('optimization_stability', 'UNKNOWN'))
    beyond_stats = result(branches.get('beyond_preLN_statistics', 'UNKNOWN'))
    pre_stats = result(branches.get('preLN_statistics', 'UNKNOWN'))
    if optimization == 'NOT_SUPPORTED':
        direction = 'RESOLVE_REGISTERED_OPTIMIZATION_ANOMALY_BEFORE_NEW_MECHANISMS'
        explanation = '完整日志触发事前定义的优化异常。先定位异常，再解释当前头的机制差异。'
    elif layer_result == beyond_stats == 'SUPPORTED':
        direction = 'TEST_A_MINIMAL_EEG_INTERMEDIATE_READOUT_HYPOTHESIS_NEXT'
        explanation = ('c+h0 在两个固定正则下通过了重复 c、错配 h0 和 c+hpre 的条件对照。'
                       '下一候选可检验局部 h0 读取旁路；这只表示固定读出族的跨概念优势，'
                       '不证明额外互信息、最终层因果丢失语义或图片收益。')
    elif pre_stats == 'SUPPORTED':
        direction = 'TEST_A_MINIMAL_PRE_LAYERNORM_STATISTICS_BYPASS_NEXT'
        explanation = ('c+hpre 的匹配维度读出通过了重复 c 对照。hpre 已验证能由 c 及归一化前均值、'
                       '尺度重建，下一候选先检验该数值旁路。当前结果不能证明 LayerNorm 丢失了语义。')
    elif layer_result == 'SUPPORTED':
        direction = 'LAYER_READOUT_GAIN_NOT_LOCALIZED_BY_CONTROLS'
        explanation = ('联合读取通过了基础对照，但没有通过完整的归一化前旁路排除条件。'
                       '保留机制定位未定，不直接推荐“中间层语义补充”新训练。')
    elif values['teacher_discrimination'] != 'SUPPORTED':
        direction = ('VALIDATE_TEACHER_PROBE_BEFORE_ANY_NEW_FUNCTIONAL_FIT'
                     if values['teacher_discrimination'] == 'NOT_SUPPORTED'
                     else 'FUNCTIONAL_TEACHER_EVIDENCE_UNRESOLVED')
        explanation = ('教师的已知噪声对照尚未一致支持。功能响应与梯度测量可描述，'
                       '但不能把它们写成已定位的语义修复机制；先核查探针适用性与公共响应。')
    elif values['trajectory_transfer'] == 'NOT_SUPPORTED':
        direction = 'INVESTIGATE_TRAINING_STATE_VERSUS_SAMPLING_STATE_MISMATCH'
        explanation = ('教师对照通过，且固定 DDPM 的绝对与配对优势没有延续到真实父模型轨迹。'
                       '下一假设是状态一致监督；这仍是有限状态探针，不能声称已证明图片失败的原因。')
    elif values['gradient_conflict'] == values['radius_saturation'] == 'SUPPORTED':
        direction = 'TEST_OBJECTIVE_COMPETITION_WITH_A_FIXED_TRUST_REGION'
        explanation = ('真实加权基础目标与合成辅助目标的局部梯度呈多数冲突，且原训练半径饱和。'
                       '下一候选可检验主目标约束下的辅助更新；不据此扩大半径或承诺图片增益。')
    elif all(values[key] == 'SUPPORTED' for key in
             ('paired_EEG_signal', 'PN_over_O', 'PN_over_P',
              'shuffled_response_control', 'trajectory_transfer')):
        direction = 'VALIDATE_WHETHER_FUNCTIONAL_PROXY_TRANSFERS_TO_IMAGE_SEMANTICS'
        explanation = ('教师、配对增量及固定分支和双状态对照均支持局部代理优势。'
                       '下一问题是代理改善能否转化为实际图片语义，不继续仅凭代理下降添加损失。')
    elif (values['paired_EEG_signal'] == 'NOT_SUPPORTED' and
          all(values[key] == 'SUPPORTED' for key in ('PN_over_O', 'PN_over_P', 'shuffled_response_control'))):
        direction = 'INVESTIGATE_COMMON_RESPONSE_SHIFT_BEFORE_MORE_FUNCTIONAL_OBJECTIVES'
        explanation = ('绝对教师接近与配对信息增量没有一致改善。先检查公共响应偏移，'
                       '不能把更低的正教师误差当作更强的 EEG 配对信息。')
    else:
        direction = 'NO_DECISIVE_NEW_MECHANISM_IDENTIFIED'
        explanation = '固定诊断证据尚不一致，保留 UNKNOWN 和负结果，不自动增加候选或扩大搜索。'
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
        assert payload['schema'] == 'BrainHIVE_decisive_EEG_layers_T_CPU_v2'
        assert payload['queries_per_lambda'] == 132300
        assert payload['no_V'] is True and payload['no_D'] is True and payload['no_test'] is True
        assert payload['GPU_used'] is False and payload['new_EEG_encoder_fit'] is False
        assert payload['branches']['beyond_preLN_statistics']['required_contrasts'] == ['c_h0_minus_c_hpre']
        assert payload['branches']['preLN_statistics']['required_contrasts'] == ['c_hpre_minus_c_c']
    if kind == 'logs':
        assert payload['no_optimizer_created'] is True and payload['parameter_updates'] == 0
        assert payload['protected_scores_read'] is False and payload['generated_images'] == 0
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
