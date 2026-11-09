# 风险头的改善是否只是被试均值：事后缓存审计

2026-10-21，America/Los_Angeles。结论：**简单每被试均值不足以解释已有 Dcheck 平均 MSE 改善。** 风险预测相对这个更严格常数对照的 MSE 仍低约 55.9%。这尚未证明门控能改善图片。

## 为什么补这个对照

原校准记录将风险头与“全体被试共用的 Dfit 常数”比较。如果不同被试的平均角误差不同，头可能只识别了被试，而没有恢复更多 query 误差信息。此处补充一个仅由 Dfit 估计的每被试常数；不在 Dcheck 拟合常数。

风险目标仍为原有缓存中的 `y=1-token_cosine`：未修正 EEG 条件与真实视觉条件经过原生 A/P 后的平均 token 角误差。它是生成条件误差代理，包含视觉目标监督，不是生理 SNR 或对象错误概率。

## 实际使用的数据和计算

- 全部十被试：每被试 Dfit 820 条、Dcheck 830 条，共 8200/8300 条。
- 只载入已有二十份 D 几何缓存，核验 subject、split、array_rows SHA、样本数、有限值及读取前后文件 SHA。
- 用历史 FP32 `1-cosine` 恢复目标，再以 float64 汇总；每被试预测值始终来自其 Dfit 均值。
- 风险头 MSE 直接引用固定最终模型已经保存的校准记录；没有重放头、重新拟合校准、训练或修改 gate。
- 用缓存向量重构的全体常数 MSE 与原记录相差约 1.05×10⁻¹⁰；均值、样本标准差也与原几何总回执吻合。
- 纯 CPU、单线程，实际约 0.97 秒；没有读取 EEG/条件缓存、模型权重、目标图片或 V 分数。

实现：[审计入口](../audit_BrainHIVE_risk_subject_constant_CPU_v1.py)；[合成数学检查](../check_BrainHIVE_risk_subject_constant_CPU_v1.py)。合成检查覆盖被试均值混杂、无均值混杂、Dcheck 均值漂移时仍使用 Dfit 常数，以及按 query 数加权汇总。

## 结果

| Dcheck 预测方法 | MSE |
|---|---:|
| 全体共用的 Dfit 常数 | 0.002943091 |
| 每被试分别估计的 Dfit 常数 | 0.002917318 |
| 已保存的校准风险预测 | 0.001286105 |

每被试常数相对共用常数降低约 **0.876%**；风险预测相对每被试常数降低 **55.9148%**。因此，现有平均 MSE 改善不能仅用这个简单被试均值基线解释。独立使用原有均值和无偏标准差的公式 `MSE=(n-1)/n×std²+(Dcheck均值-Dfit常数)²` 交叉复核，与实际向量计算一致。

## 解释边界

这是**事后解释审计**。审计登记前已从既有聚合矩算出近似值，登记文件明确披露；不能写成事前未知结果或独立确认。原来的 Dcheck 已曝光，不据此调整模型、强度或校准系数。

这里只控制一个固定被试均值，不能排除全部被试效应。风险头的逐 query 预测没有在本审计中读取，因此没有被试内相关、逐被试头 MSE或 head-minus-constant 的配对置信区间。不能把头的聚合 MSE 固定后，仅重采样常数一侧来制造配对区间。

相对 MSE 变化是算术比较，不是因果分解。条件误差可预测也不证明 `gate=1-risk/2` 是合适的修正策略。门控图片增益、感知质量和可比 SOTA 均仍未确认。

实际证据：[输入绑定](../audit/BrainHIVE_risk_subject_constant_input_binding_actual_20261021_v1.json)、[事后方法登记](../configs/BrainHIVE_risk_subject_constant_CPU_v1_registration.json)、[服务器数学检查](../audit/BrainHIVE_risk_subject_constant_CPU_math_server_actual_20261021_v1.json)、[实际缓存结果](../audit/BrainHIVE_risk_subject_constant_CPU_actual_20261021_v1.json)。

同期普通修正已完成全量训练并开始正式出图；原生 HIVE 图片随后顺序执行。保持既有主比较和冻结 T32，不由这项诊断触发新候选训练。
