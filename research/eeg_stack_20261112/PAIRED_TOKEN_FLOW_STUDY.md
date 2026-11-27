# 用跨物体配对训练 EEG 到图像条件的流

Brain-HiVE 的四个图像 token 可以拆成公共均值和四个相对差异。过去的真图交换中，保留真图的相对差异、替换公共均值，仍能在 200 张图中辨认 70 张主体。只保留真图公共均值时，原主体命中为 0。这一轮保留当前 EEG 的公共均值，使用同一条 EEG 在相对差异的空间里生成一个条件。[四 token 交换实验](TOKEN_EVIDENCE_STUDY.md)

## 训练

训练数据为 1,323 个登记物体、13,230 张训练照片和四位被试的 52,920 条真实四次呈现查询。每条 EEG 条件由没有见过该物体的 Brain-HiVE 编码器生成。图像目标是匹配照片的四个相对 token 在训练照片 PCA 空间里的 512 维标准化坐标 $z_i$。冻结 DINOv2 描述用于在训练侧挑选外观相近、物体不同的照片，并拟合从条件坐标到照片视觉语义的线性映射 $A$。每个训练配对使用同一人的 EEG。

给照片 $a,b$ 同一个标准高斯起点 $\varepsilon$ 和时间 $t$，两条轨迹分别为

$$
x_i(t)=(1-t)\varepsilon+t z_i,\qquad i\in\{a,b\}.
$$

模型接收 $x_i(t)$、时间和 EEG 条件 $e_j$，预测速度 $v_\theta(x_i(t),t,e_j)$。该处给出的预测终点为 $q_{ij}=x_i(t)+(1-t)v_\theta(x_i(t),t,e_j)$。用同一张目标照片的 DINOv2 描述 $u_i$ 测量 $d_{ij}=\operatorname{mean}[(Aq_{ij}-u_i)^2]$。配对损失在轨迹早期权重更高。

$$
L_{\text{pair}}=(1-t)\operatorname{softplus}
\left(\frac{d_{aa}+d_{bb}-d_{ab}-d_{ba}}{2\cdot0.15}\right).
$$

普通流和配对流都使用正确配对的速度回归与图像语义损失。配对流再加 0.4 倍上式。两组的训练输入、误差空间、初始权重、随机噪声、优化步数与每步选取的照片相同。流匹配和对比流均有公开先例。这里检验的改动是，在同一噪声起点对换两条真实 EEG，能否让冻结生成器收到更明确的主体条件。[E2I-Flow](https://arxiv.org/html/2611.04524v1) · [Contrastive Flow Matching，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Stoica_Contrastive_Flow_Matching_ICCV_2025_paper.html) · [实现](code/paper_paired_token_flow.py)

## 推理和对照

每个查询给一人的真实 EEG。生成器仍接收该条 EEG 的公共 token。普通流和配对流共用一个固定种子产生的潜在噪声，经八次 Heun 更新得到相对 token。使用训练照的均方根范数校准相对部分。错配对照只交换流接收的 EEG 条件，公共 token、初始潜在噪声和图像种子沿用接收查询。SDXL-Turbo 使用空提示、一次出图、四次 U-Net 调用。流额外使用 16 次小网络前向。

按照开发 R4 协议，在被试 01 和 03 各自的 200 张整类留出照片成图。正式 R4 与 R80 还记录独立的成图前语义读出。评分阶段才读取原照与 1,854 个名字。[基准协议](PAPER_BENCHMARK_PROTOCOL.md)

## 随机流成图

两组各完成 3,000 步更新，所有训练批次的参与者与照片顺序散列相同。最后 250 个训练批次中，正确对换方向的胜率由普通流的 81.24% 到配对流的 84.99%。开发集的 400 张 Brain-HiVE 图与归档像素逐张匹配。

| 开发 R4，实际成图 | 主体命名 / 400 | 找回原照 / 400 | 同类十图找回原照 / 400 |
| --- | ---: | ---: | ---: |
| Brain-HiVE | 5 | 10 | 79 |
| 普通流 | 3 | 12 | 55 |
| 配对流 | 2 | 11 | 63 |
| 配对流，交换流的 EEG 输入 | 3 | 10 | 46 |

配对流相对原生的命名差为 −0.75 个百分点，按被试和物体配对重采样的 95% 区间为 [−3.00, +0.50]。与交换 EEG 对照相比，差 −0.25 个百分点，区间为 [−2.00, +1.00]。

## 从速度估计条件均值

从标准高斯独立采样时，流在时间零的理想速度为 $v^*(\varepsilon,0,e)=\mathbb E[z\mid e]-\varepsilon$。再用原先权重，在八对正负相反的噪声上直接平均时间零的速度。每对噪声对模型的线性项在平均中相消。这个读出与八步 Heun 采样同为 16 次小网络前向。

$$
\widehat{\mathbb E}[z\mid e]
=\frac1{16}\sum_{k=1}^{8}\left[v_\theta(n_k,0,e)+v_\theta(-n_k,0,e)\right].
$$

用相同噪声交换当前 EEG，再用同一 EEG 反转噪声符号，分别测量相对 token 坐标的变动。用这项诊断判断随机起点与当前 EEG 的影响量级。取均值的两组依然使用相同的单人 EEG 和训练图像条件范数。[条件均值实现](code/paper_token_flow_mean.py)

| 开发 R4，标准化坐标 RMS 差异 | 被试 01 | 被试 03 |
| --- | ---: | ---: |
| 随机流，保持 EEG 并反转噪声 | 1.443 | 1.440 |
| 随机流，保持噪声并交换 EEG | .145 | .138 |
| 条件均值，交换 EEG | .359 | .349 |

在两个参与者上，随机流更受起始噪声影响。条件取均值后，交换 EEG 引起的坐标变动大约是原来两倍半。两人官方 R80 的成图前名称 Top-1 / Top-5 由配对随机流的 0 / 3 到配对均值的 4 / 12。原生参照为 2 / 10，配对均值用错 EEG 时为 0 / 0。这里的分母都是 400，仍是成图前的语义读出。

| 开发 R4，实际成图 | 主体命名 / 400 | 找回原照 / 400 | 同类十图找回原照 / 400 |
| --- | ---: | ---: | ---: |
| Brain-HiVE | 5 | 10 | 79 |
| 普通流条件均值 | 6 | 13 | 69 |
| 配对流条件均值 | 3 | 10 | 81 |
| 配对流条件均值，交换流的 EEG 输入 | 1 | 2 | 50 |

两轮分别重生成的 Brain-HiVE 图都逐张匹配归档的 400 张 PNG。普通均值对原生多 1 个命名积分，配对差 +0.25 个百分点，95% 区间为 [−2.00, +2.00]。配对均值对原生少 2 个积分，差 −0.50 个百分点，区间为 [−2.00, 0.00]。固定样例中，配对均值把护栏画成了推车，把画笔画成了草地上的动物。此前训练配对胜率的增长和条件均值对 EEG 的更强响应，尚未转化为成图主体的稳定优势。[成图、配对区间和输入核验](results/paired_token_flow.json)

## 复现

在已有基准环境中，保留折外 EEG 预测、训练照片的相对 token PCA 和 DINOv2 视觉目标。令环境变量指向当前研究输出根，在符合规定的计算节点运行。

~~~sh
python paper_paired_token_flow.py fit
python paper_paired_token_flow.py infer 1 --render-budgets dev-r4
python paper_paired_token_flow.py score 1 --render-budgets dev-r4
python paper_token_flow_mean.py infer 1 --render-budgets dev-r4
python paper_token_flow_mean.py score 1 --render-budgets dev-r4
~~~
