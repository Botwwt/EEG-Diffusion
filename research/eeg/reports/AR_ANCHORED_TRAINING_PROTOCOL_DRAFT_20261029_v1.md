# 三臂 EEG 锚定 AR 训练草案：完整 T、步级恢复与有界存储

2026-10-29，America/Los_Angeles。**这是未准入、不会自动启动的协议和实现核心。** 尚无本草案的真实生成器拟合、研究图片或收益。实际全尺寸 VQ/VAR profile、T 图像 code bank、图片门槛和主机资源计划仍需绑定，不能用本地 CPU 检查代替。

[具体协议 JSON](../configs/AR_anchored_training_DRAFT_20261029_v1/protocol.json)绑定了已有 EEG bank 的真实 SHA：sub-01 全部 13,230 条 T／1,323 概念，来自排除对应概念的 oof0／oof1 末轮编码器。full 编码器只为后续正常 Dcheck 推理提供条件。三份编码器各 50 轮，来源 campaign SHA 为 `aa0964d7e94b936459c01005e33ae19b7179c071c287f846d894406e5bd90c56`；原 materialized 1024 维 c、63×1024 memory 不改。

## 三臂和固定数学预算

三臂为 native-global、paired-anchored、same-concept-training-donor-anchored。三者使用同一公开 VAR 骨干张量和同一 seed1729 初始化的已有 EEG `embed_proj`，仅输入通路不同；参数数量相同不等于函数容量相同。VQ、EEG bank 全部冻结，正常推理仅用自身 EEG。

各臂完整 T 十遍，batch16、末批14，每遍827更新；每臂8,270更新／132,300次 query 暴露，三臂共24,810更新／396,900暴露。目标为真实 VQ4096 code 的 B×680 token 平均 CE，无额外教师 loss。AdamW 固定 lr1e−4、wd0.01、betas(0.9,0.999)、eps1e−8、梯度裁剪1；不另加隐含 warmup/cosine。固定完整第十轮，不用 D／V／官方 test 选 epoch。

donor 只在训练时选取同概念其他刺激。每概念按固定 manifest 顺序，使用 `SHA256(1729|AR-local-donor|epoch|concept)` 决定非零循环位移；这是循环错排，不冒称全部排列上的均匀抽样。它不消耗模型或 shuffle RNG。保留 query 自己的 c 和图像目标，仅替换局部偏差：

\[
\widetilde H_i=c_i+(H_j-c_j),\quad j\ne i,\quad concept(j)=concept(i).
\]

推理时 donor 臂也读取自身 memory；推理接口不选择 donor，不读取概念标签或图像。EEG/global/local 共用同一次原生0.1条件 dropout；注意力／投影 dropout 在当前原型中为0，原生 DropPath／FFN 随机流保留。

## 已实现的核心和真实 CPU 范围

[实现核心](../AR_anchored_training_core_DRAFT_20261029_v1.py)没有 CLI、下载或 GPU 派发。调用方必须先完成准入、绑定文件来源和资源，再显式调用。它包含完整 token CE 更新、确定性 donor、完整行覆盖、步级 cursor、model／Adam／CPU／CUDA／shuffle RNG snapshot，以及有界检查点存储。

[23 项 CPU 检查](../audit/AR_anchored_training_DRAFT_20261029_v1/CPU_contract_actual_v1.json)使用作者真实 VAR/VQ 的缩小随机模型。三臂 query 顺序和训练 RNG 消耗一致；在一个 epoch 的短末批之前断开，经实际 tensor-only checkpoint 保存／加载后，续跑得到相同 query／donor／loss、逐位相同模型和 Adam moments、相同 CPU／shuffle RNG；全部短末批保留，VQ 无梯度。另检查了检查点中断、两 payload 上限、未知文件拒绝与独立的不可覆盖逐更新日志。没有真实 EEG／图像训练或 GPU 检查。

分段只允许发生在已完成的 optimizer step 之后。snapshot 保存当前完整 epoch 排列、下一行位置、已暴露 query 数及部分 epoch CE 累积；因此实际 profile 后可改变每个有限作业执行的更新数，不重置 epoch、顺序或训练 RNG，也不改变十遍数学预算。真实 CUDA 内核下的恢复和随机流仍须实际验证。

## 新卡存储不能按“每轮都保存完整状态”估计

310,308,096 个 FP32 参数约1.241GB；模型加两个 Adam moment 的 resume 约3.724GB，尚有序列化和小型状态开销。当前新卡在 materialized EEG bank／运行目录之后只余约14GB，后续还有公开 VQ／VAR约1.7GB。不能让三个臂的完整历轮状态同时驻留，也不能用删除旧结果解决。

草案采用显式登记的、**仅属于新训练臂**的 A/B 两槽 transient resume 策略。稳态保留 current／previous；写下一份前先发布 reservation，保持验证过的 current 可读，再记录并删除自有 previous 的已核验 payload，把新状态写入释放的槽，最后原子提交 manifest。这样写入期间最多也只有两份 resume；普通“先写第三份再删最旧”会达到三份峰值。

固定最终模型另存保留，逐更新 query／donor／loss／gradient、epoch 覆盖和检查点提交／退役哈希元数据全部保留。中断后旧 current 仍可读，未提交 payload 保留；实现拒绝自动重试／清理。未知文件、符号链接或所有权不符会停止；没有递归删除、旧 ordinary／D40／历史结果删除。这个策略必须出现在正式资源注册中，不能隐式使用。

单臂两份 resume 加最终模型粗计约8.69GB。三臂不可能据此直接同时留在当前新卡。正式准入必须选择主机安排，或在继续下一臂前按固定计划将完整状态 SHA 核验转存到旧卡；转存尚未实现或启动。paired／donor 同硬件比较优先，任何跨主机运行都须报告实际硬件与数值边界，不能宣称不同 GPU 的随机轨迹逐位相同。

## 图片验证仍待事前锁定

下一版正式协议应先绑定同一400目标图库、80→400全臂规则、采样参数、query seed、有限资源和完整分母；中间400前缀不读取效果。拟议主比较为 paired−native 及 paired−donor，两者 DINO 均至少+0.01、描述区间下界大于0，同概念 Top1 均值不损害；这些仍是草案，不能拿未绑定门槛给结果放行。

AR 原生256像素、十尺度与 HIVE512像素／四步的分辨率和成本应分列。后续仍需实际 EEG-only 图片、完整指标、多被试和官方 R80；本草案的完整 T 是单被试开发阶段，不能当成十被试完整方法或 SOTA。
