# 在 CBraMod 内对四次真实 EEG 做联合注意力

上一轮的局部时频残差在成图前能提供少量配对信息。实际成图的主体命名在换用错配残差后没有下降。这一轮改变 EEG 编码器，使它可以在四次呈现之间直接交换特征。[时频实验](LOCAL_SPECTRA_STUDY.md)

公开的 REVE 资料将 THINGS-EEG2 列为预训练覆盖数据。本实验使用原论文报告在 Temple University Hospital EEG Corpus 上训练的 CBraMod 权重。源代码固定在 `b9e961003214326972c567eff390e75b0287e32a`，模型文件固定在发布修订 `500543c7e30bda1b22bfd51a49301b238dee21fd`，加载前逐文件核验。跨电极和片段注意力来自 CBraMod，实验改动是让片段轴表示四次独立呈现，并用同容量控制隔离交换信息的作用。[REVE 预训练数据说明](https://github.com/facebookresearch/neuroai/blob/main/docs/neuralbench/models/reve.rst) · [CBraMod 论文](https://openreview.net/forum?id=NPNUHgHF2w) · [作者权重](https://huggingface.co/weighting666/CBraMod/tree/500543c7e30bda1b22bfd51a49301b238dee21fd)

## 编码与训练

每条查询是同一位被试看同一张图的四次物理记录。使用注册的 17 个后部电极。每次原始保存波形有 250 个采样点，经抗混叠的 4:5 多相重采样得到 200 个值。每个电极的均值和缩放量只从该人的训练记录拟合，按中位单次 RMS 将典型尺度调到 0.2。

设 $x_r$ 为该人第 $r$ 次重采样后的 EEG，$P$ 为 CBraMod 逐次做的波形和频谱片段映射。每次独立调用 $P$，所以片段卷积和位置编码没有读取其他呈现。将得到的 17 电极向量 $z_r=P(x_r)$ 堆叠成 $17\times4\times200$。联合编码器在电极轴及呈现轴上交替传播特征。逐次对照用同一个编码器处理长度为 1 的呈现轴，在后处理中求平均。

$$
h_{\mathrm{joint}}=\operatorname{mean}_{r}\bigl[E(z_1,z_2,z_3,z_4)\bigr]_r,
\qquad
h_{\mathrm{separate}}=\frac14\sum_{r=1}^4 E(z_r).
$$

每组用相同的头把 $17\times200$ 特征映射到原生 Brain-HiVE 的 1,024 维融合空间。第三个对照从随机初始化训练同一联合结构。单次预处理共享同一组权重，训练和评估均保留四次实际测量。只在联合注意力内交换呈现信息，时间轴未把四次呈现解释为一段连续采集。对重复顺序交换的输出差值另行记录。[读出实现](code/paper_cbramod_repeats.py)

每人训练使用登记的 13,230 张图及物理 EEG。每批一张图对应一个不同的训练物体，每个物体的十张图均参与训练。训练损失共用精确照片的融合特征对比和回归项。再以 0.25 权重加入 DINOv2 主体项，把输出通过仅在训练照片拟合的固定映射送到 DINO 空间，与该物体其余九张照片的方向做双向对比。三个训练变体共用批序、步数、照片目标、损失和优化器。正式查询输入只有该人的当次 EEG。训练只在各自独立的注册训练集进行。

第 06、09 名被试的官方 200 张图用于同预算成图。Brain-HiVE 和各新读出共用同一人的前四次物理呈现、空提示、逐刺激随机种子、一次生成和四步 U-Net。错配版本只将联合编码器收到的四次 EEG 整组换成另一个官方刺激的记录。所有测试图像与候选名称只由评分过程读取。

## 保留旧模型的同条件复测

生成了 2,000 张图，另纳入同两位被试已归档的 800 张逐人 D2 与共享 D2 出图。旧模型的四次物理呈现、空提示、种子、刺激和生成步数与本轮一致。共享 D2 使用十人的训练记录，每条推理仍只接收一个人这次的 EEG。旧 D2 的布局与训练目标随模型保留。400 张新跑的 Brain-HiVE PNG 与正式基准逐张散列一致。新编码器的六个训练组各有 7,157,024 个可学习参数、880 次更新。首轮保存时缺少输出目录，修复后这六组完整重训，以下只使用成功保存的权重。[正式基准及旧训练轨](../../docs/PAPER_BENCHMARK_RESULTS.md)

成图后同时用原基准的 CLIP 和冻结的 SigLIP2 在 1,854 个物体名中评分。两个评分器均对并列名称分摊命中分。真原照各得 170.25 和 177.25/200。复评分逐张核对生成 PNG 与原始回执，覆盖所有新旧图。[第二评分器和输入核验](code/paper_independent_naming.py) · [SigLIP2 权重](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2)

| 官方 R4，每组 400 张 | CLIP 主体 | SigLIP2 主体 | CLIP 找回原照 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| Brain-HiVE | 0 | 2 | 12 | .30245 |
| 旧逐人 D2 | 3 | 2 | 12 | .43309 |
| 旧共享 D2 | 5 | 7 | 24 | .41729 |
| CBraMod 联合 | 1 | 3 | 7 | .34536 |
| CBraMod 逐次 | 2 | 0 | 10 | .34595 |
| 随机初始化的联合模型 | 0 | 1 | 4 | .34496 |
| CBraMod 联合，交换 EEG | 0 | 1 | 2 | .33480 |

联合组相对逐次组的命名差在 CLIP 下为 −0.25 个百分点，在 SigLIP2 下为 +0.75 个百分点。两者的配对 95% 区间均包含零。共享 D2 在两个命名器、原照匹配中得分均最高。它相对 Brain-HiVE 的 SigLIP2 命名差为 +1.25 个百分点，95% 区间 [−0.25, +3.50]。这是两人的探索性复评，且共享 D2 用了更多人的训练记录，组件的贡献需用相同训练数据单独检验。[逐人结果、配对区间和文件散列](results/cbramod_repeats.json) · [汇总与核验](code/summarize_cbramod_repeats.py)

逐图看过两个评分器给出命中积分所涉及的 20 个被试–刺激条件，以及固定编号画廊。共享 D2 的第 06 名被试有可辨认的羚羊与橙子，原照里的姿态、背景和视角仍有差异。该组的海藻看起来像陆地暗色地形，两种自动命名都给了分。联合 CBraMod 在第 09 名被试画出了一把可辨认的方刃刀，但航母、香蕉、鹰等固定示例多被映到食物、托盘或工具。第 06 名被试的 Brain-HiVE 画出了金属链，两个评分器只有 SigLIP2 给分。这次目视复核的范围是固定样例和自动命中，表中仍是统一的自动命名指标。

成图前，将各模型的融合向量与 200 个原照向量比较。原生 Brain-HiVE、联合和逐次组分别找回 37、56、66/400。交换四次输入的顺序时，两位被试的预训练联合模型在检查批的融合向量最大坐标差都是 $2.38\times10^{-7}$。预训练学到的信号提高了视觉向量匹配，生成器仍未稳定画对物体。保留旧共享 D2 作为后续复测的候选，并用十人训练的匹配对照检验增益来源。旧的同照片配对训练继续用同物体的不同原照检验。[历史配对收益](../../docs/OBJECT_LOG_REVIEW.md)

## 复跑

按正式基准准备注册数据和权重。CBraMod 外部代码与捆绑权重放在独立目录，DINO 训练图像条件来自既有跨折实验。每位被试设定其个人权重位置。各个 fit 调用可独立在调度器分配的计算节点运行。

```bash
python paper_cbramod_repeats.py fit 6 --upstream "$CBRAMOD_SOURCE" \
  --source-root "$EXISTING_CROSSFIT_RUN" --arm joint
python paper_cbramod_repeats.py fit 6 --upstream "$CBRAMOD_SOURCE" \
  --source-root "$EXISTING_CROSSFIT_RUN" --arm separate
python paper_cbramod_repeats.py fit 6 --upstream "$CBRAMOD_SOURCE" \
  --source-root "$EXISTING_CROSSFIT_RUN" --arm random-joint
python paper_cbramod_repeats.py render 6 --upstream "$CBRAMOD_SOURCE"
```

第 09 名被试沿用上述流程。四次采集的列索引及参与者、图片和生成器回执随结果保存。

将三组 `fit.json`、`pre-gen-sub-*.json` 以及 `images/sub-*/<arm>` 下的 PNG、`render.json`、`scores.json` 和 `observations.npz` 放入同一个归档。旧个人和共享组放到 `historical/<track>/d2/official-r4/sub-*`，旧 Brain-HiVE 只需要 `historical/personal/native-paper/official-r4/sub-*/render.json`。使用包含 `concepts1854.csv`、`test_photos.csv` 和 `test_images.zip` 的核验数据位置打分。独立评分使用 Transformers 4.57.6 和权重修订 `75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2`，长度按模型配置填充。

```bash
python paper_independent_naming.py --model "$SIGLIP2" --assets "$IMAGE_SET" \
  --archive "$RGB_ARCHIVE" --output independent-naming.json --subjects 6 9 \
  --arms native-paper joint separate random-joint joint-wrong personal/d2 shared/d2
python summarize_cbramod_repeats.py --receipts "$EXTRACTED_RGB_ARCHIVE" \
  --independent independent-naming.json --output cbramod_repeats.json
```
