# 功能监督候选的执行顺序修订与未完成前置

2026-10-21，America/Los_Angeles。PN 尚未拟合，也没有 PN 图片性能结果；已有 CPU 工程证据保留。本次只修订执行计划，没有训练、GPU 前向、生成图片或读取当前 V 分数。

## 为什么修订

[用户 master](../../../EEG_SOTA_AUTORESEARCH_MASTER_PROMPT.md) 第7、169–171、519–524行要求完整正式训练和开发评价，第477行允许小样本工程冒烟检查；没有强制另训一个 T 子集性能 pilot。此前项目协议 §4 要求概念分组的子集训练/验证，DRAFT 却把 pilot 写成 optional/engineering_only，二者没有对齐。

实际 `brainhive_functional_training_contract_v2.py` 的 profile 为固定2×16更新，不保存头或生成图片；正式模式才检查四臂实际 profile 和全生命周期预算。它没有强制检查所有文档中的质量/pilot条件。因而 helper 通过不能被解释为整份研究准入已满足；也不能把正式入口中硬编码的132300/24807/16600改小，假称已有小 pilot 接口。

## 现在的顺序

1. 在拟合前锁定工程判据、完整固定输入、实际入口及源码、功能诊断的准入判读、失败处理与预算。
2. 完成当前 risk/ordinary/parent 三组、已有冻结 T32 功能诊断及零头/关闭的一 query 像素回放。
3. 完成 O/P/PN/S 各自有限的真实 GPU 工程 profile；profile 的小预算与正式四臂完整生命周期预算分别审计。
4. 每臂正式训练全部10被试、132300个真实 OOF-T query、三遍396900次暴露、24807更新；固定最后头，仅用既有 Dfit 选择既定强度。
5. 每臂用同一份既定 sub-01/T32 清单生成32张最终头工程图片。它只核对身份、有限值、原生采样调用、可解码与文件完整性；空间常数/近常数/重复等标志完整报告。不凭对象好坏选头、重训、改强度或裁减 V。
6. 每臂仍生成全部16600张 V 图，完成逐 query 评价、配对与完整性审计，全部完成后一次判读。

取消的是额外的子集性能筛选，不是正式训练、控制臂、V query 或机制验证。模型、损失、半径、时间点分布、种子、Dfit网格和既定主要判据均未修改。候选计数和结构修订上限未重置。

## 尚未具备的入口与结论

固定32条身份与随机流的 [CPU元数据 DRAFT](../configs/BrainHIVE_functional_postfit_T32_queries_DRAFT_v1.json)已保存：原 oracle 顺序、全部 T 归属、两折 OOF/cache 行与同臂种子均核对。cache 行和权重 SHA 来自已保存的实际 CPU 回执转录，本次没有新读远端 tensor 或 image_manifest.csv；完整执行前仍须做实际输入核验。[独立 CPU 回执](../audit/BrainHIVE_functional_postfit_T32_queries_CPU_v1.json)保留这一区别。

实际入口必须使用各 OOF 检查点自己的 affine；这批 OOF-T32 工程图片不能替代正式 V 的 full-parent 加载与数据来源核验。

最终头 T32 [独立生成入口](../generate_BrainHIVE_functional_postfit_T32_v1.py)和[完整V artifact审计](../audit_BrainHIVE_functional_full_V_artifacts_CPU_v1.py)现已实现并绑定源码，通过服务器CPU核验；此前入口为null的工程缺口已补齐。当前没有promoted运行配方、通过准入、v2最终头或真实GPU像素结果，不能据本报告运行正式任务。既有正式 V 生成器只接受16600条，既有零头回放只生成一 query 三种接入情形，仍不能冒充这批最终头T32图片。

每臂最终头 T32 计划上限为1200秒、64MiB全部输出，四臂共128张、256MiB持久空间上限；时限仍待实际工程核验。最初按旧模板的小计由23,648,000,000增加至23,916,435,456字节。随后[实际文件统计](../audit/BrainHIVE_functional_evaluation_storage_file_stat_actual_20261021_v1.json)确认完整risk feature已有275011242字节，超过旧模板220MB/臂；当前候选规划改为feature320MB、metadata256MiB/臂，小计为24,790,177,280字节，仍须安全余量、旧承诺和实际尺寸核验，**未获得完整持久空间准入**。T32图片单独记工程成本，不充作 V 图片或额外挑图机会。

工程 artifact 检查不能证明感知质量非劣；margin仍为null，未定义、未验证，不能宣称通过。没有新增人评、自然度标签或未经验证的质量模型。原生诊断已锁定[描述性判读policy](../configs/BrainHIVE_native_T32_evidence_policy_v1.json)及[独立reader](../read_BrainHIVE_native_T32_evidence_CPU_v1.py)，但它不生成准入：missing-raw Q与parent-Q不同，教师自身J是恒等式，有近零项而缺联合数据时保留UNKNOWN。所有实际执行前置仍须在新拟合前落实；本次不把缺失条件改写成已有结果。

## 证据

新增的[评分调用与来源闭合程序](../run_BrainHIVE_functional_evaluation_CPU_v1.py)和[固定四臂比较](../compare_BrainHIVE_functional_V_CPU_v2.py)已实现并通过28项小型CPU检查与独立源码复核。正式完整生命周期计划仍须绑定它们及冻结scorer的SHA、真实调用输出目录与资源上限；当前DRAFT不是可执行计划。比较要求全部四臂16600条真实评价闭合后一次读取，不以合成fixture代替实际图片结果。

- [执行顺序修订后的 DRAFT](../configs/BrainHIVE_functional_alignment_v2_preregistration_DRAFT.json)
- [CPU字段与来源不变核验](../audit/BrainHIVE_functional_v2_execution_order_revision_CPU_20261021_v1.json)
- [当前项目协议 §4](../BRAINHIVE_NEXT_RESEARCH_PROTOCOL_20261021.md#4-优先候选监督原生条件的功能响应)

八个已固定的功能监督源文件与当前主配方均保持不变。本次没有准入新拟合，也没有组件图片增益或 SOTA 结论。
