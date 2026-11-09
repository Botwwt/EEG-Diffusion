# Brain-HIVE 三组 R4 最终配对结果

2026-10-22，America/Los_Angeles。三组均已完整生成并评价16600张图片，共49800张。基础受限条件修正在当前开发集上出现小幅收益；风险门控未证明比普通修正进一步改善图片，未通过冻结配方要求的双对照门槛。

## 完整三组分数

| 方法 | DINO2AFC | DINO Top-1 | SEED | ObjectF1 | SSIM |
|---|---:|---:|---:|---:|---:|
| parent：原生 HIVE | 0.768191 | 0.014639 | 0.130608 | 0.138498 | 0.229010 |
| ordinary：普通受限修正 | 0.778066 | 0.015783 | 0.136787 | 0.141841 | 0.266991 |
| risk：风险门控受限修正 | 0.779900 | 0.015723 | 0.136282 | 0.141227 | 0.258554 |

以上是全部query的均值。Top-1表示1660张候选中的精确目标检索命中；DINO2AFC不能解释成对象正确率。ObjectF1为锁定的COCO+gender代理。

## 配对差与描述性区间

| 对比 | 指标 | 均值差 | subject/concept重采样95%区间 |
|---|---|---:|---|
| ordinary−parent | DINO2AFC | +0.009875 | [0.002762, 0.017299] |
| ordinary−parent | SEED | +0.006179 | [0.003596, 0.008840] |
| ordinary−parent | ObjectF1 | +0.003343 | [−0.001219, 0.008479] |
| risk−parent | DINO2AFC | +0.011709 | [0.005134, 0.018415] |
| risk−parent | SEED | +0.005674 | [0.002913, 0.008373] |
| risk−parent | ObjectF1 | +0.002730 | [−0.001687, 0.007480] |
| risk−ordinary | DINO2AFC | +0.001834 | [−0.000751, 0.004932] |
| risk−ordinary | SEED | −0.000505 | [−0.001920, 0.000885] |
| risk−ordinary | ObjectF1 | −0.000613 | [−0.003259, 0.001961] |

ordinary与risk相对parent的DINO2AFC、SEED区间均高于零，支持当前配置下的开发收益线索。两者的ObjectF1差值区间均跨零，Top-1差值区间也跨零，不能声称已经稳定恢复更多正确对象或精确目标。

risk相对ordinary只有DINO均值略高，SEED、ObjectF1、CapSim、Top-1、PixCorr和SSIM均值略低；这些均值与所报告区间没有证明risk稳定胜出，也不证明风险门控必然有害。

原冻结门槛要求risk相对parent和ordinary均达到DINO2AFC至少+0.01且配对区间下界大于零，SEED同向且区间下界大于零，ObjectF1均值差不小于零。risk只在parent对比满足这些数值条件，在ordinary对比未满足，最终主判据为false。ordinary的DINO差准确值0.009874997355302667，低于0.01；其补充比较的实践门槛也为false，不能四舍五入通过。

## 比较条件与解释范围

全部10被试、166概念、1660刺激、16600个V/R4 query；同一full父编码器、冻结A/P/IP/Turbo及相同query噪声种子，512×512、4步、CFG0、IP1，每query一张，无best-of-N。两个新头均使用完整132300个真实OOF-T query训练三遍、24807次更新，固定训练seed1729与最终检查点。没有根据这些结果改变半径、训练时长、检查点或选择新的生成种子。

配对区间使用既定1000次subject/concept交叉重采样，seed1729，每轮先抽subject再抽concept；它不覆盖训练随机性。V此前已曝光，区间是描述性开发证据，不能充当独立确认。当前结果支持继续研究受限修正；风险配置与ordinary的比较尚不能独立归因到gate，因为risk还包含风险监督与不同的训练后校准。

风险头预测条件误差MSE较低，也不能替代本次实际图片结论。当前未建立风险机制增量、盲评或感知质量非劣、跨架构泛化或官方可比较SOTA。后续功能监督PN尚未正式训练；当前结果不会自动准入新训练，也不触发在V上重复搜索。

## 实际来源与执行闭合

主生命周期于服务器2026-10-22 09:04:02 UTC完成，控制器与parent特征worker已退出。三组图片完成与配对回执SHA、全部特征/评分/低层完成回执，以及原配方和源码不变检查见[实际最终核验](../audit/BrainHIVE_three_arm_final_readout_actual_20261022_v1.json)。这次核验复用既有全量产物，没有重新生成或评分图片。

- [原三组汇总和risk配对比较](../audit/BrainHIVE_three_arm_final_summary_observed_20261022_v1.json)：服务器原件`/root/brainhive_reproduction_20261020_v1/comparison/bounded_v1/summary.json`，SHA `8a0fdc001a60e0f58655f94ba065a34722d3a8eca9a4145cc97cd21cf41cc4d2`。
- [ordinary−parent补充比较](../audit/BrainHIVE_ordinary_parent_final_observed_20261022_v1.json)：完整16600条身份核验后计算，无新增图片或拟合。
- [固定主判据判读](../audit/BrainHIVE_completed_stage_decision_observed_20261022_v1.json)：`CURRENT_RISK_IMAGE_GAIN_NOT_ESTABLISHED`，保留其原判读名；它不否认risk相对parent已有正向差值。
- [主生命周期实际完成状态](../audit/BrainHIVE_main_lifecycle_complete_observed_20261022_v1.json)。

ordinary−parent原等待进程在首次进入比较时，因原512MiB内存余量检查失败而在读取评分之前退出。失败日志与旧状态保留；确认主任务和GPU任务结束、实际余量满足同一条件后，调用原固定源码完成首次评分读取与1000次配对重采样。没有重复选择统计结果或改变比较配方，见[CPU恢复回执](../audit/BrainHIVE_ordinary_parent_CPU_recovery_actual_20261022_v1.json)。
