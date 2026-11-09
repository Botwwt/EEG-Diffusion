# 大方向调整：从冻结条件修补转向 EEG 表征与多尺度生成联合验证

> **2026-10-29：完整T编码与锚定EEG注意力**：sub-01 full/oof0/oof1 三模型各固定50轮闭合，合计10400更新／1323000 query暴露；训练循环806.049秒，27个GPU包装1057.071秒、控制器1060.070秒，设计和权重准备另计，不是十被试正式完成。全部150个epoch事务、顺序覆盖、224张量最终模型、Adam及RNG独立CPU复核通过。两卡均实际完成13630条真实OOF-T／Dcheck400局部EEG导出，源角色与归一化排除当前概念；T-only同概念Top1为17.9138%，仅属表征读数。另存锚定原型保留原生图像注意力，只加入局部EEG相对同c的注意力差，零局部信息精确回退；22项CPU检查通过，尚无新生成收益。旧卡00:41:15已派发全T冻结VQ token与后续完整生成器工程队列；第二卡接续自身生成工程与基线槽，CPU等待／传输不能写成GPU训练。旧ordinary未改，正式D仍停止。见[真实执行、公式与证据](EEG_REAL_T_ENCODER_AND_ANCHORED_ATTENTION_PROGRESS_20261029_v1.md)。

> **D40完整400停止**：正确63电极创新＋ordinary DINO为0.786497，匹配ordinary为0.793490；差−0.006992、区间跨零，同概念Top1由27.75%降至23.25%、下降区间全负。正确创新相对错排未有可靠增量，两个背景机制支持均失败。3200图闭合，本地16000指标与11项bootstrap复算通过，不以80条正均值挑赢家。见[完整D40报告](BRAINHIVE_SPATIAL_INNOVATION_RESULTS_20261029_v1.md)。


> **X01完整400实际闭合：2026-10-28 22:24:30，America/Los_Angeles**：全十被试OOF-T三遍、24807更新完成后，两组800张固定图片和评价闭合。ATMS原生/等价ordinary DINO为0.755301/0.757738，差+0.002437、描述区间[-0.016899,+0.024499]；同概念Top1由19.25%降至16.50%。未达原门槛，STOP，不扩V或事后调强度。原特征下载后另行本地4000指标和1000次概念bootstrap复算通过；这不是独立视觉编码器重跑。旧卡本轮GPU已退出，后续公开权重准备属于CPU/网络工作；22:27新卡D40第三臂仍在训练，前两臂完成。HIVE旧ordinary保留，不能据X01声称通用收益。见[完整X01结果与证据](ATMS_STANDARD_SDXL_ORDINARY_FINAL_RESULTS_20261028_v1.md)。

2026-10-28，America/Los_Angeles。本文记录正在实施的方向调整；没有新方法训练成功或 SOTA 结论。旧 ordinary 保留为不可覆盖的 HIVE 基线，D40 与 X01 按原登记继续。

## 为什么调整主线

[D39 完整 400 条](BRAINHIVE_MEASUREMENT_PRIOR_RESULTS_20261028_v1.md)中，测量约束加 ordinary 的 DINO 为 0.781353，匹配 ordinary 为 0.793490，差 −0.012137、描述性区间全负；同概念 Top1 也下降。无约束编码器虽然生成训练损失下降，图片严重退化。事后组合诊断没有支持“ordinary 把小扰动放大失控”这一简单解释。

这些结果支持降低“围绕固定全局 c 反复修补”的优先级，**不能证明 EEG 没有额外信息、BrainHIVE 已到上限或 diffusion 本身不适合 EEG**。改动必须触及学习目标和 EEG→生成器的信息通路；只换一个更大的编码器也没有充分依据。

## 已实际核对的公开路线

| 路线 | 可借鉴之处 | 选择时的边界 |
|---|---|---|
| [AVDE，ICLR 2026](https://arxiv.org/html/2602.22555v1)；[作者代码](https://github.com/ddicee/avde) | 预训练 EEG 编码器与离散多尺度直接生成，有可检查的完整解码器 | 作为生成骨架和强对照；不能因换框架预设胜过 HIVE |
| [ENIGMA 作者代码](https://github.com/Alljoined/ENIGMA) | 多被试时空 CNN、共享训练及完整生成入口 | 已取得作者版本；另取得的 aqasemi 实现是第三方，不能冒充作者代码 |
| [ViEEG，ICML 2026](https://proceedings.mlr.press/v306/liu26ap.html)；[作者代码](https://github.com/LauMason/ViEEG) | 视觉监督的不同成分可分别建模 | 检索改进不能直接写成生成收益 |
| [RPA，2026-09 预印本](https://arxiv.org/html/2609.31698v1) | 视觉侧中间 patch 监督是重要强基线；并非只有扩大 EEG 编码器才能改进 | 其图像重建另训系统和先验；检索值不能替代重建指标。尚未找到论文链接的作者代码，未声称完成复现 |

作者源代码已按不可变 commit 下载并逐文件 SHA 核验。AVDE 为 `3be06a90ae69932c203bf4ae45d90803d8e752ed`，ENIGMA 作者版本为 `460af4b6a0ca6eee11aeaf35c5431e8e3d55c11a`，ViEEG 为 `2365e1b20b8a31ce549a49e24937141ce0c579bf`。源码清单位于 [reproduction_sources](../reproduction_sources/direction_switch_20261028_v1)。没有运行默认训练脚本。

## 已写出的架构原型及推导

令现有全局 EEG 条件为 c，未池化的 EEG patch 表征为 a₁,…,aₘ，构造

$$H_j=c+a_j-\frac1m\sum_i a_i,\qquad \frac1m\sum_j H_j=c.$$

这里 H 是 EEG 电极/时间表征，**不是查询真实图片的视觉 patch**。图像第 k 个尺度的条件分布由

$$q_0(R_k\mid R_{<k},c)$$

改为

$$q_1(R_k\mid R_{<k},H,c).$$

原型把 H 作为持久前缀，复用现有 VAR 的投影、Q/K/V、MLP 和输出头。EEG 前缀自身只读 EEG；图像 token 可读 EEG 及合法的图像上下文。没有新建可学习层；仍增加序列长度和计算量，不能称为免费推理。正常自由生成只接收 EEG 条件、随机种子及固定采样配置，不接收图像、概念标签、文本或图库答案。

若比较的是同一真实分布的最优条件模型，且 c 是 H 的确定性函数，理论上的条件熵差为

$$\mathbb E[-\log p(R_k\mid R_{<k},c)]-\mathbb E[-\log p(R_k\mid R_{<k},H)]
=I(R_k;H\mid R_{<k},c)\ge0.$$

**这是经典条件互信息恒等式，不是原创定理，也不保证有限模型受益。** 对实际学到的 q₀、q₁，配对对数似然差还包含两模型逼近误差的差；正值可能来自拟合或校准差异。训练时真实前缀带来的优势也可能在自由生成时消失，因此必须再做真正的 EEG-only 图片比较。

有限对照设计为：①原作者全局条件；②等 token 数的重复全局条件；③保留自身 EEG patch 差异；④训练时仅同概念错配 patch 差异、保留自身 c。第④组在正常推理时仍使用自己的 EEG，不能读取概念标签挑 donor。所有需要拟合的分支须锁相同预算和固定末轮；目前仅锁定工程范围，没有以此登记正式训练已准入。

实现：[EEG 持久前缀](../eeg_multiscale_prefix_20261028_v1.py)、[工程范围](../configs/multiscale_EEG_20261028_v1/engineering_scope_v1.json)。前缀条件化、保留 token 和基础模型迁移均有既有工作；现阶段不主张首次使用、机制成立或论文创新充分。

## 已真实完成的工程检查

[14 项 CPU 检查](../audit/multiscale_EEG_20261028_v1/CPU_contract_actual_v1.json)使用作者真实 VAR/VQ 实现的缩小随机模型，检查关闭前缀后 logits/像素精确回退、相同全局条件下 patch 差异可见、未来图像不能经 EEG 前缀泄漏、缓存推理和同一生成历史的全序列 logits 一致、异常时缓存释放，以及 EEG memory/原 QKV 梯度。缓存与全序列 logits 最大差为约 4.2e−9。这些不是预训练权重、真实 EEG 或研究图片效果检查。

首次 CPU 检查因作者默认 CUDA RNG 与 CPU 张量冲突失败；后续精确 logits 检查还发现浮点加法结合顺序差异，已修复；随机测试明确启用可见的注意力门值来检查信息通路，未把近零初始化下的微小响应误报为不存在梯度。[初次失败](../audit/multiscale_EEG_20261028_v1/CPU_contract_first_failure_v1.json)及数值失败回执均保留，作者源码未改。

[完整尺寸 meta 构建](../audit/multiscale_EEG_20261028_v1/full_shape_inventory_actual_v1.json)实际计数：VAR 310,308,096 参数，VQ-VAE 108,948,355 参数；未实例化 EEG 编码器、未加载权重。默认图像 token 为 680，增加 63 个 EEG token 后完整注意力矩阵元素数约为原来的 1.194 倍。这不是实测 FLOPs、显存或速度。VAR 的 FP32 参数、梯度及两个 Adam moment 合计约 4.965 GB，尚不含 activation、临时量或其他模型。

[作者源码审查](../audit/multiscale_EEG_20261028_v1/author_source_audit_actual_v1.json)绑定了 10 项具体代码事实，包括默认官方 test validation、按该 loss 保存 best、论文/代码 MSE 权重差异、默认外部日志、全局关闭初始化及 frozen/eval 的区别。新入口须明确处理这些差异，不直接继承默认 test-best 选择，也不能静默改 loss 后称为作者完全复现。

## 当前执行与下一准入条件

两张现有卡仍执行原任务。2026-10-28 22:09 实际观察：D40 的 marginal 与 measurement 均完成各 827 更新，第三臂正在训练；X01 已完成全十被试 24,807 更新，正在补足固定 400 条图片，80 条尚无收益。尚无这两项的完整新图片正结果。

新方向另外实际启动了有限 CPU 公开权重准备：作者 LaBraM base、VAR-d16 和配套 VQ-VAE，共 1,775,804,949 字节，目标为服务器持久盘；不是向 C/D 写入大型权重。初次公网请求超时保留；正在只修传输，作者 revision、大小、Git blob/SHA256 不变。LaBraM 是临床语料预训练权重，不是借用在本次留出概念上拟合过的公开 EEG 最终模型。资源准备没有启动新 GPU 拟合。

正式拟合前仍需闭合：真实 EEG 的单位/通道/时间及 250→200Hz 预处理契约、严格预训练权重加载、真实单卡梯度/内存/速度、固定完整 T 预算及全量评价。当前缓存不能无说明地冒充作者未白化的 200Hz 输入。AVDE 原生输出 256，HIVE 为 512；比较时必须保留原生尺寸与成本，不能用放大图片伪装相同架构。

若这一通路没有留出概念增量或自由图片收益，就停止该实例。最终 SOTA 仍须固定方法后完成相同信息权限下的十被试完整官方 R80 重建与公开指标；单被试开发面板、检索提高、工程检查和 loss 下降均不能替代它。
