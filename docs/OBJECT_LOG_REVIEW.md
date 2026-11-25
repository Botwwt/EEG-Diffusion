# 从历次实验看，怎样让 EEG 重建画对物体

**逐人训练、单人输入、空提示、每个刺激只出一张图时，目前没有证据把 M1a + D2 列为物体重建的胜者。** 官方四次重复中，D2 与按论文投影复现的 Brain‑HiVE 均为 14/2,000 个物体命名积分。八十次时分别为 28 和 37/2,000，配对差 −0.45 个百分点，95% 区间 [−2.30, +1.15]。四次的 SSIM 差为 +0.1185 [+0.1081, +0.1288]。颜色分布和大轮廓接近原图，可以与画错物体同时发生。[十人配对结果及真图评分](PAPER_BENCHMARK_RESULTS.md)

## 哪些结果回答了“画对了吗”

| 实验及输入 | 原记录中的结果 | 对选模型的意义 |
| --- | --- | --- |
| 早期原生 Brain‑HiVE。十人，各用同一张照片的 80 次 EEG | 用正确、错配、缺失 EEG 生成后，DINO 在 200 张原照中找回 320、7、12/2,000。加 ordinary 修正后，DINO 两选一由 0.87277 升至 0.89120，但精确找回只从 320 到 344，差的区间为 −1.60 至 +4.20 个百分点。 | 正确 EEG 确实携带视觉信息。能把真图排在更多干扰图前，对“成图是哪个物体”仍要单独评分。[正确 EEG 与对照](../research/eeg/reports/BrainHIVE_native_R80_comparison_20261020_v10.csv) · [双因素重采样](../research/eeg/audit/HIVE_full1654_original_ordinary_complete_20261101_v1/scientific_readout.json) |
| 视觉中间层，sub‑01，两种训练种子 | CLIP ViT-H/14 第 12 块的共同训练图库，在 200 选一的 EEG 检索中命中 162、166 张，末层命中 71、67 张。另留 128 个训练域物体，每类十张，精确照片检索分别是 343、338 对 74、80/1,280。 | 值得把中间层用作训练老师。旧实验用了不同输入维度、共同学习的视觉坐标和检索计分。旧 I03 中层回归与 I05 扩散引导都未建立可靠成图优势。[原始双种子数据](../results/retrieval.json) · [视觉编码代码](../experiments/archive/visual/eeg_native_single_visual_reference_v1.py) · [旧生成实验](EXPERIMENTS.md) |
| 11 月早期更换骨干、视觉表征和投影，sub‑03 | MLP17 在训练见过的 13,230 张融合特征中精确找回 12,611 张，按整个概念留出后是 70 张。另一项 13,229/13,230 使用的输入是实际照片的融合特征，目标是该特征里的 VAE 分支。训练配对正确的 EEG 编码在 B32 同类十图检索仍比类内错配训练多 8.74 个百分点 [8.07, 9.41]。 | 从早期表格中移除训练内检索和真照片输入作为 EEG 重建的成绩。保留正确/类内错配的训练对照，用成图检验逐图信息。[15 次骨干训练](../research/eeg/evidence/EEG_ALL15_ENCODER_TRAINING_LOGS_20261111_v327.json) · [T 域与留出折](../research/eeg/evidence/EEG_BACKBONE_T_ONLY_RESULTS_20261111_v330.json) · [视觉分支对照](../research/eeg/evidence/EEG_ARCHITECTURE_COMPRESSION_RESULTS_20261109_v312.json) |
| O8、O9、O10、O13 的 R80 生成。每个刺激计四张图 | 200 原照检索依次为 15.81%、15.84%、16.20%（O8、O9、O10）。O10 的语义/布局分支相对 O9 在调参四人多 0.91 个百分点，其余六人持平。O13 的同一照片跨人训练有正向结果，但共享训练、重复次数和测试时适配均与单人 R4 基准不同。 | 跨人监督可以在训练期利用。把两组七种子编码器、按整个测试图库定中心或用测试 EEG 调参并入单查询模型，会改变预算和输入。[O8](../research/eeg_stack_20261112/results/o8_b5_stack.txt) · [O9](../research/eeg_stack_20261112/results/o9_streamlined.txt) · [O10](../research/eeg_stack_20261112/results/o10_factorized.txt) · [O12–O14](../research/eeg_stack_20261112/results/o12_o13_o14_info_ceiling_xbrain.txt) |
| 历史 R4 的 M1a + D2。四次单人 EEG，十人训练，同物体十张候选仅用于评分 | 相同 M1a 读出下，D2 对 Arm1b 是 982 对 939/5,000（19.64% 对 18.78%）。500 张一起检索变为 190 对 180，配对检验 p=0.565。历史评分参考由 VAE 解码，同概念共用采样种子。本轮改用真实 RGB、刺激独立种子，新 50 类上逐人 D2 对逐人 Brain‑HiVE 是 884 对 899/5,000。 | D2 直接训练同一物体的四张照片各自归位。历史增益针对这项区分任务，开放物体身份还需单独验证。[同读出的配对消融](../research/eeg_stack_20261112/results/exp8d_m1a_arm1b_symmetric_results.json) · [本轮对照](PAPER_BENCHMARK_RESULTS.md) |
| 早期无目标菜单。sub‑03 的留出 20 类，sub‑04 的官方 200 类 | sub‑03 的校准视觉头加自由去噪，使生成图在 20 类原照原型间的命中从 25 到 60/200，错配 EEG 为 7；同类十图从 21 到 40，错配为 14。sub‑04 的训练类角度校准使 DINO 原照命中从 10 到 20/200，CLIP 原照从 5 到 17，1,854 词命名仍为 2 对 2。 | 训练类校准是可迁移的线索。这里的 20 类是评分器使用的候选，生成器只看 EEG。两项各来自一人，不能替代十人开放命名。[sub‑03 逐臂汇总](../research/eeg_stack_20261112/results/object_fidelity/summary.json) · [sub‑04 的生成图结果](OFFICIAL_REPEAT_IMAGES.md) |
| 已校准 12 类，每个候选附八张支持照片 | 三人 360 次查询，Fisher 对四次试次编码器加 Fisher，完整词表的成图命名从 145 到 166；DINO 同类具体照片命中从 43 到 42。另有按记录场次匹配、固定词、同词错 EEG 等实验。 | 已校准物体确实更容易画对。目标类和支持照片改变了生成器事先获得的信息，这组数字用于校准任务。[同场次重复试验](TRIAL_OBJECT_PERSONAL12.md) · [逐图评分](../research/eeg_stack_20261112/results/object_fidelity/personal12_trial_replication_images.json) |

共享训练值得保留为独立研究条件。十人提供训练 EEG、测试仍只给一人的 EEG 时，D2 在官方 R4 得 21/2,000，逐人 Brain‑HiVE 得 14；R80 为 51 对 37/2,000，差 +0.70 个百分点 [−1.15, +2.45]。相同共享训练下，R4 的正例相似度对照 D1 已得 22/2,000。论文中应写明训练人数，并按配对差估计加入特定组件的贡献。[共享与逐人训练对照](PAPER_BENCHMARK_RESULTS.md)

追加的十人消融把配对损失的正例系数固定为 0.5。官方 R4 只加同类四图配对损失，成图命名差 +0.20 个百分点 [−0.85, +1.25]，SSIM 差 +0.0132 [+0.0111, +0.0153]。保持相同 D2 权重及布局，语义路径从算术平均换为 M1a，物体命名差 +0.05 个百分点 [−0.90, +1.00]。这些组件在论文中有各自的对照，当前的默认单查询实现可以省去多路读出和布局头。[十人组件消融](PAPER_BENCHMARK_RESULTS.md#单独移除配对损失与重复-eeg-读出)

值得保留的是 EEG 与具体照片的配对监督和错配检验。新 50 类中，把单人 D2 的输入换为同物体另一张照片的 EEG，精确照片命中从 884 降到 438/5,000；正确输入的优势是 +8.92 个百分点 [+6.52, +11.36]。这说明生成图确实保留了单条 EEG 的一些照片信息。原生 Brain‑HiVE 在同一张表上已有 899 次命中，现有结果无法把这部分信息的来源归功于 D2 的新增损失。[同类错 EEG 与基线的逐人结果](PAPER_BENCHMARK_RESULTS.md)

## 两个最能定位错误的对照

1. **顺着同一条 EEG 看图像条件到成图的损失。** 十人官方 R80，逐人 D2 在扩散前有 748 条从 200 个视觉向量中精确找回原照。这 748 条里，生成图再次精确找回原照的有 116，物体命名只得 17 分。完整 1,654 训练类的 Brain‑HiVE 也是 961 条特征找对、176 条图片找对、33.5 分命名。每一行都核对了照片标识和实际检查点。真图排第一只需要略微领先竞争图片，生成器却必须从该条件本身读出足够清楚的物体身份。[逐查询衔接](../research/eeg_stack_20261112/code/paper_error_readout.py) · [命中与未命中组](PAPER_BENCHMARK_RESULTS.md)
2. **固定生成器，只换条件。** 空提示、四步、200 张开发照片。给真实照片的冻结视觉条件，成图命名为 105/200；给训练集平均视觉条件，命名为 0/200。原图自身在同一命名器下为 164/200。EEG 到视觉条件有很大改进空间，哪怕给真实条件，生成和自动命名仍会丢掉一部分目标。[生成接口干预](../research/eeg_stack_20261112/code/paper_image_calibration.py) · [同图库实测](PAPER_BENCHMARK_RESULTS.md)

## 在紧凑模型上检验什么

保留的对照路径为 **四次 EEG 算术平均 → 一个 BrainMLP → 冻结的 Brain‑HiVE 融合适配器及四个图像条件 token → 从噪声起步的 SDXL‑Turbo 四步 → 一张图**。推理输入是一名被试的这次 EEG 和随机种子。主结果另列历史 M1a + D2、仅换布局、同参数损失消融，以及同刺激取错 EEG 的结果。[单次推理入口](../research/eeg_stack_20261112/code/paper_object_infer.py) · [完整实验协议](../research/eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md)

| 训练改动 | 怎样知道改动有没有用 |
| --- | --- |
| 同物体均值与单图残差。在四个生成 token 上，用其余九张训练照片的均值做起点，按训练 EEG 预测单图偏差的可靠性加回偏差。 | 两人各 200 张，原版命中 5、0；全单图 5、1；全均值 2、2；可靠性加权 3、0。三者均未达到“两人各自提高命名且合计同类十图维持或提高”的预设标准。另一条在融合特征上做残差的试验为 7 对 5/400，区间仍跨零。[训练目标](../research/eeg_stack_20261112/code/paper_object_tokens.py) · [逐臂差值](PAPER_BENCHMARK_RESULTS.md) |
| 训练老师换到 CLIP-H 第 12 块。辅助头等容量、同优化预算，对照使用相同视觉模型的末层。训练结束丢弃辅助头。 | 只取注册训练类的 13,230 张原图，固定裁剪和位置。中层取 256 个 patch 的均值，按训练图压到 1,024 维。旧提取器的 CLS 选项没有完整留存，本轮明确定义该选项。[训练与约束](../research/eeg_stack_20261112/code/paper_middle_teacher.py) · [实际成图](PAPER_BENCHMARK_RESULTS.md) |
| 让编码器从实际画错的物体上学习。对真实四步解码 RGB 用冻结 DINOv2 评分，跨物体负例与查询的粗略布局、颜色相近。训练正例分别取当前照片和同一物体的另一张训练照片。 | 两臂使用相同起点、查询、图片、负例、噪声和梯度更新数，只改正例。主要自动命名由独立 CLIP 完成。生成时不增加视觉老师或文字提示。这是小规模开发干预。[两臂训练](../research/eeg_stack_20261112/code/paper_rgb_object.py) · [实测和入选规则](PAPER_BENCHMARK_RESULTS.md) |

每次判定先看 1,854 个词中生成图的正确物体，再看正确 EEG 对错配 EEG、同类十图和像素结构。词表只交给评分器，平局分摊命中分。原照自身的得分限定了这项自动指标的解释，也需要看固定顺序的原图与重建图。传统两选一、原照 Top‑1、同类细节、完整词表命名在论文中各占一列。[评分实现](../research/eeg_stack_20261112/code/paper_score.py)

两人的中层老师共叫对 7/400，末层老师 4/400，原生 5/400。中层老师在两人分别从 5 到 6、从 0 到 1，但同类十图从原生的 79 降为 72/400。直接在四步生成 RGB 上训练，本图和同物体另一照片作正例时分别为 4 和 5/400。这里有相反方向的变化，几张图的正差不足以指定论文最终模型。中层方案仍可用于检查编码器保留了什么视觉信息。[两人配对成绩及控制变量](PAPER_BENCHMARK_RESULTS.md)

另把历史角度校准的思路放回同一个 Brain‑HiVE 生成器，全部校准参数折入原有编码器的最后一层。仅校正训练均值，开发命名从 5 到 7/400，同类十图从 79 到 77；再按训练概念折外预测校正分散程度，命名为 5/400，同类十图为 83。后一步更有利于区分照片，开放命名仍需独立证据。[零额外推理操作的实现](../research/eeg_stack_20261112/code/paper_native_spread.py) · [同生成器消融](PAPER_BENCHMARK_RESULTS.md)

O13 启发的检验同样只在训练时使用其他九人的 EEG。各人的视觉预测按整个训练物体交叉验证，再把其余九人的预测平均。使用同照片教师或同类另一照片教师，两臂的开发命名均为 3/400，同类十图分别为 74 和 65/400。训练人数和数据量在两臂相同，推理各用一个人的四次 EEG。这一轮完成的训练对照均未满足入选八人确认的预设标准，因此保留原生紧凑路径作为比较基线。[跨人教师与配对控制](../research/eeg_stack_20261112/code/paper_peer_teacher.py) · [两人实测](PAPER_BENCHMARK_RESULTS.md)

## 档案范围

校准任务另保存了两人复验及其单查询入口、跨人混淆学习、背景属性和提示、全局照片条件、保护物体身份的局部修正。扩展个人解码器在另两人上把 1,854 词命名由 218 提到 243/700，这里仍提供 12 个已校准物体。背景提示的联合晋级条件未通过；保护照片细节的成图文件记录的是注册设计。设计和真正落盘的成图结果分别标明。[两人校准复验](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_images.json) · [单查询记录](../research/eeg_stack_20261112/results/object_fidelity/fresh_people_single_query.json) · [跨人混淆](../research/eeg_stack_20261112/results/object_fidelity/cross_person_confusions.json) · [背景提示](../research/eeg_stack_20261112/results/object_fidelity/background_prompt_images.json) · [全局照片成图](../research/eeg_stack_20261112/results/object_fidelity/global_photo_images.json) · [保护照片条件的设计](../research/eeg_stack_20261112/results/object_fidelity/protected_photo_images_design.json)

本次核对覆盖十月至十一月的历史报告目录（240 份 Markdown、CSV 和 JSON）、36 份早期结构化证据、O 系列与 SwapDiff/DRaFT/M1/M2 的原始数表，以及工作区物体保真目录的 209 份 JSON（含 77 份设计文件）。同一批生成图的阶段性日志、版本更新和重新评分合并解读。早期有些算力作业只留有报告、哈希或汇总数值；文中的配对区间使用可取回的逐项记录，其余注明原报口径。当前正式基准提供可重新汇总的逐刺激评分包。[早期报告](../research/eeg/reports) · [结构化证据](../research/eeg/evidence) · [十一月结果](../research/eeg_stack_20261112/results) · [按任务索引](EXPERIMENT_REGISTRY.md)
