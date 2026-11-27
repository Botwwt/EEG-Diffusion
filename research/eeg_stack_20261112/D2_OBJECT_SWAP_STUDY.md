# 交换 EEG 条件，训练 D2 区分不同物体

共享 D2 的同类照片配对已能传递部分照片信息。单独增加条件幅度减少了重复画面，叠加新读出的方向修正也未提高共享 D2 的主体命名。本轮从原共享 D2 的权重开始，在实际生成图上训练跨物体区分，同时保留同类照片的配对损失。[旧配对结果](../../docs/PAPER_BENCHMARK_RESULTS.md) · [幅度校准对照](CONDITION_CALIBRATION_STUDY.md)

Brain-HiVE 用分层照片特征和融合先验学习脑电与图像的对齐。Perceptogram 已用测试时的 EEG 时间片段互换观察所画物体的变化。EVD-Bench 也用错配 EEG 检查图库检索是否依赖正确输入。本轮把跨物体对换写入训练，在相同布局与噪声下同时训练正配图和错配图，再用独立评分器检查留出物体的正配图是否更容易被识别。[Brain-HiVE](https://arxiv.org/abs/2602.07495) · [Perceptogram](https://2025.ccneuro.org/abstract_pdf/Fei_2025_Perceptogram_Interpreting_Visual_Percepts_from_EEG.pdf) · [EVD-Bench](https://arxiv.org/abs/2611.30061)

## 固定布局与随机数，只交换 EEG 条件

每次更新选同一位被试的两种训练物体，各抽两张不同照片 $a_1,a_2,b_1,b_2$。第二个物体来自低分辨率 VAE 外观上最相近的八个训练物体。两张照片交换对象时按 $a_1\leftrightarrow b_1$、$a_2\leftrightarrow b_2$ 配对。每张照片的脑电输入是四次实际呈现，布局用旧 D2 的训练物体折外岭回归结果。四次实际去噪贯穿训练。两次生成之间固定当前照片的初始布局、噪声和空文本输入。

令 $I_{ij}=G(z_i,E_j,\epsilon)$ 表示从照片 $i$ 的 EEG 布局 $z_i$、照片 $j$ 的 EEG 图像条件 $E_j$ 和相同噪声 $\epsilon$ 得到的图像。冻结 DINOv2 对实际解码 RGB 给出向量 $v(I)$。用训练照片拟合 1,323 个物体原型 $p_k$，本批用到的四张照片从相应原型扣除。令 $S(I,k)=\cos(v(I),p_k)$。对于实际物体和供体物体 $k_i,k_j$，定义方向差

$$
m_{ij}=S(I_{ij},k_i)-S(I_{ij},k_j),\qquad
L_{\rm swap}=\operatorname{mean}_{i}\operatorname{softplus}\!
  \left(\frac{m_{ij}-m_{ii}}{2\tau}\right),\qquad \tau=.07.
$$

布局相同的两次生成都给出相似主体时，$m_{ij}$ 和 $m_{ii}$ 接近。此项监督要求当前真实 EEG 让原物体比供体物体更加突出。两个方向都用同样的公式。正确成图还在全部 1,323 个训练原型上做分类交叉熵，以提高匹配图的绝对主体得分。

同类照片的两组 $2\times2$ 损失沿用 D2 的图像与布局相似度，LAION H/14、B/32 及 16×16 VAE 潜变量低频结构的权重为 .60、.25、.15。比较正确配对相似度与同物体的另一张训练照片。训练更新原融合投影的 `head_f` 和原有四 token 条件层，冻结 EEG 其余参数、布局回归、SDXL-Turbo 和三个视觉老师。两组共用相同初始化、1,323 个登记概念、全部十人的训练记录、120 次更新和八张逐步去噪成图每次。设 $L_{\rm photo}$ 是两组同类损失加匹配相似度，$L_{\rm id}$ 是正确生成的物体交叉熵。

$$
L=L_{\rm photo}+.35L_{\rm id}+\lambda L_{\rm swap}+.05L_{\rm trust},\qquad
\lambda\in\{0,.5\}.
$$

$L_{\rm trust}$ 是新旧图像 token 的均方差除以旧 token 的平方均值。对照在 $\lambda=0$ 时也构造和计算错配的前向。可微地穿过采样过程有 DRaFT 先例。已有 EEG 工作也用真实照片条件的去噪器中间状态监督生成。这里检验的是固定布局和噪声的真实 EEG 对换，及其在最终 RGB 的主体效应。[DRaFT 原论文](https://proceedings.iclr.cc/paper_files/paper/2024/hash/13e8be77982beb73d7ed0bbf122f9f3c-Abstract-Conference.html) · [Beyond Alignment](https://arxiv.org/abs/2611.19232) · [训练及推理](code/paper_d2_object_swap.py)

用归档训练照片的真实缓存，在 CPU 上独立检查 120 个训练批次。每人出现 12 次，共涉及 208 个物体、465 张不同照片。所配两个物体的粗布局中心化余弦均值为 .741，同样锚点的随机配对参照为 .047。把每批真实原照的 DINOv2 向量用作目标输入，在 1,323 个原型上有 405/480 次首位命中，480/480 个对换有正确方向。这一项核验训练照片配对、持出概念以及视觉老师的目标方向。模型训练与开放成图单独据其回执验收。[真实训练批次核验](results/d2_object_swap_episode_audit.json)

## 开放成图验证

沿用 THINGS-EEG2 官方 200 张留出照片。选择第 04、07 名被试作为首轮比较，这两人在上一轮幅度校准没有作为成图对照。各臂输入同一人的前四次观测。空文本、逐刺激种子、布局配置及四步 SDXL-Turbo 固定。比较共享 D2、主体锚定对照、加入交换损失，并给原共享 D2 和交换组各做一个错配对照。错配组只换图像 token 的 EEG 来源，保留该次查询自身的布局。另取正式基准的逐人 Brain-HiVE 原图用于同刺激比较。

图像冻结后，两个评分器各在 1,854 词表上检查主体命名。所有实际推理共用 D2 的已有架构。本轮的训练照片原型和三份图像教师只用于训练，错配只进入对照组。[正式输入核验](PAPER_BENCHMARK_PROTOCOL.md) · [独立命名](code/paper_independent_naming.py)

对保留同一布局的正配和错配图，另外比较 SigLIP2 对当前物体名和供体物体名的余弦差，也比较这两张原照的余弦差。正配减错配测量脑电交换引起的变化，再与原共享 D2 的相同变化比较。另列正配图对正确物体的绝对分数，便于辨别能否实际画对。该项只评估已经完成的图像，1,854 词开放命名仍是主体主指标。[固定布局的交换效应](code/paper_eeg_exchange_effect.py)

## 实际成图结果

两个拟合各更新 2,233,472 个现有参数，采用相同的 120 个训练批次。两人各生成五组，每组 200 张。重新生成的共享 D2 与归档的 400 张 PNG 逐张散列一致。Brain-HiVE 是逐人训练，共享 D2 及其追加训练用十人的训练数据。两种命名器在 200 张真实原照上分别得 170.25 和 177.25 分，词表只在成图之后用于评分。

| 官方 R4，同一批 400 条查询 | CLIP 命名 | SigLIP2 命名 | 找回原照 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| 逐人 Brain-HiVE | 3 | 4 | 26 | .31869 |
| 共享 D2 | 2 | 4 | 30 | .41236 |
| 追加主体训练 | 2 | 3 | 29 | .41336 |
| 追加主体训练和 EEG 交换 | 4 | 4 | 26 | .41400 |
| EEG 交换训练，测试给错配条件 | 0 | 0 | 4 | .39707 |
| 共享 D2，测试给错配条件 | 0 | 0 | 4 | .39503 |

加入交换损失后，相对同预算的主体训练对照，CLIP 命名差为 +0.50 个百分点，95% 配对区间 [0.00, +2.00]。SigLIP2 差为 +0.25 个百分点 [−1.50, +2.00]，原照命中差为 −0.75 个百分点 [−4.00, +2.50]。相对共享 D2，CLIP 命名差为 +0.50 个百分点 [−1.00, +2.00]，SigLIP2 持平，原照命中少 4 张。新增交换项在两种命名器上均缺少可重复的收益。

正配相对保留布局的错配图，原共享 D2 的原照命中从 4 到 30，交换训练从 4 到 26。SigLIP2 对正确物体名的余弦分数在原共享 D2 和交换组分别提高 .02073 和 .02167。用目标名减供体名衡量交换方向，新旧模型的效应之差为 +.00123，95% 区间 [−.00225, +.00493]。用原照片作同样比较为 −.00076 [−.00742, +.00568]。固定布局时，两个模型的图像条件都传递来自 EEG 的信息。本轮对这些效应的额外提升估计接近零。

第 07 人的固定样例中，篮球仍是没有球纹的光滑红球，老鹰的生成图有鸟喙和蓝天，形状与展翅的原图相差较大。自动命中的图中，橙子可辨认，新增交换组和主体训练对照都画出了橙子。平衡木和喷枪的新增 CLIP 命中对应长条和细管等局部线索。样例检查覆盖固定序列和自动命中图，未作盲评。本轮是两人、单次拟合的试验，保留共享 D2 为同预算基线。[逐人命中、配对区间与源文件回执](results/d2_object_swap.json)

## 复现

使用 `EEG_RUNS_ROOT="$RAD_ROOT/fits/shared"`，设置正式基准的代码与模型环境。准备共享 D2、折外布局、H/14 和 B/32 训练目标、DINOv2 训练照片向量，再在计算节点运行两组拟合和两人生成。

```sh
for arm in object-anchor object-swap; do
  python paper_d2_object_swap.py fit --mode "$arm" --bank "$TRAIN_DINO_BANK" --output "$OUTPUT/fits"
done
for person in 4 7; do
  python paper_d2_object_swap.py render "$person" --output "$OUTPUT"
done
```

将 `$OUTPUT` 中的拟合 JSON 与完整成图目录打包，同时保留 `historical/personal/native-paper/official-r4/sub-NN` 下的两名被试成图和 `historical/shared/d2/official-r4/sub-NN/render.json`。以下命令用展开后的回执目录、指定的 SigLIP2 权重和基准图像清单复算。

```sh
python paper_independent_naming.py --model "$SIGLIP" --assets "$PHOTOS" \
  --archive "$BUNDLE" --output "$NAMING" --subjects 4 7 \
  --arms personal/native-paper shared-d2 object-anchor object-swap object-swap-wrong shared-d2-wrong
python paper_eeg_exchange_effect.py --model "$SIGLIP" --assets "$PHOTOS" \
  --images "$RECEIPTS/images" --output "$EFFECT" --subjects 4 7 \
  --pair shared-d2 shared-d2-wrong --pair object-swap object-swap-wrong
python summarize_d2_object_swap.py --receipts "$RECEIPTS" --independent "$NAMING" \
  --exchange "$EFFECT" --output "$REPORT"
```
