# 从四次 EEG 的局部时频统计恢复整图条件

上一轮将正确照片的 128 个主成分补到原生条件上，生成器更容易画对物体。直接从平均 EEG 读出这些方向没有超过原生模型。早期用整段非锁相 FFT 功率匹配不同场次的同图脑电，平均为 0.60%，随机水平为 0.50%。这一轮保留局部时频位置，检验逐次统计能否在平均波形之外补充视觉信息。[整图方向诊断](FUSED_POSTERIOR_STUDY.md) · [历史跨场次 FFT](O12_O13_O14_REPORT.md)

## 读出方法

对四次真实呈现分别在 6、9、13、19、27、38、54、76 Hz 应用复解析高斯滤波。原始输入为 17 导、250 Hz 保存的 1 秒波形。每个频带把统计量在五个局部时间窗内求均值，边界为 0、80、160、260、400、640 ms。设某一时刻、通道和频带的四次复系数为 $z_r$，共同系数为 $\bar z$。模型比较三种 680 维特征。

$$
P_{\mathrm{mean}}=|\bar z|^2,\qquad
P_{\mathrm{var}}=\frac{1}{R-1}\sum_r |z_r-\bar z|^2,\qquad
P_{\mathrm{cross}}=|\bar z|^2-\frac{P_{\mathrm{var}}}{R}.
$$

第三项等于不同呈现的复系数乘积取实部后平均，可以为负。第二项包含四次呈现间的振幅、相位和时延变化，也受噪声影响。采用局部时频统计和区分诱发与非锁相反应已有先例。该实验检验这三种统计量在开放 EEG 成图中的作用。[视觉 EEG 时频研究](https://www.jneurosci.org/content/16/13/4240) · [D²-FOSA](https://openaccess.thecvf.com/content/CVPR2026/html/Yu_D2-FOSA_Dual-Diffusion_Guided_EEG-to-Image_Reconstruction_with_Frequency-Oriented_Semantic_Alignment_CVPR_2026_paper.html) · [时频提取](code/paper_trial_spectra.py)

用同四次 EEG 的平均波形分成 40 个 20 ms 箱，拟合到原照融合条件的时域读出 $F_w$。对每个时频统计，先做训练尺度校准，再从中减去可由平均波形线性预测的分量。用所得时频残差预测时域读出的照片误差，得到 $\Delta F_m$。记原生 Brain-HiVE 的整图条件为 $G$，输入冻结生成器的条件为

$$
F_m=G+\beta(F_w-G)+\alpha_m\Delta F_m .
$$

每次修正范数最多为原生条件到训练照片中心距离的 85%。四位校准被试用完全留出物体的四折确定 $\beta=.5$。三种时频统计的 $\alpha_m$ 分别为 1、.25、.5。与成图不同，校准判别使用独立图像方向和该折的物体原型，真类原型排除当前原照。[读出和成图实现](code/paper_spectral_conditions.py)

| 四人整类留出，成图前 | 物体判别 / 52,920 |
| --- | ---: |
| 原生 Brain-HiVE 条件 | 1,919 |
| 混入时域平均波形读出 | 2,093 |
| 加平均波形的局部频带功率 | 2,139 |
| 加跨呈现局部功率 | 2,133 |
| 加呈现间局部时频残差 | 2,143 |

## 同一折的训练侧诊断

保留原生条件、时域读出、照片和校准参数，只交换其他物体的时频残差，命中由 2,143 降至 2,074。对应配对差为 +0.130 个百分点，按被试和整物体重采样的 95% 区间为 [+0.011, +0.248]。完整残差对时域读出的差为 +0.094 个百分点，区间 [−0.004, +0.195]。区间对应已选定校准参数。

用同一已拟合模型提取 80–260 ms 和 260–640 ms 对应窗口的贡献，给两组逐样本使用相同修正幅度。命中分别为 2,123 和 2,111，差值区间跨零。解析滤波的时间支持超出名义窗口。快速呈现的相邻图像可能影响较晚部分。

另将四次呈现分为三种互不重叠双对配置。每组求两个试次差的局部功率几何均值，然后等权平均三组。单个离群试次只有在另一对也出现波形差异时才产生较大贡献。同权重、同容量的独立拟合得到 2,130 次，错配双对贡献为 2,089 次。[时间与配对诊断](code/paper_spectral_consistency.py)

## 实际成图

选择第 07 和 10 名被试作独立成图检查。两人在历史记录中跨次 EEG 匹配较稳定。每人固定官方 200 张照片的同四次物理呈现，各组共用空提示、逐刺激种子与一次四步 SDXL-Turbo。错配组只换新增的时频修正，原生条件与时域读出仍来自当前 EEG。名称与参考图由评分过程读取。结论适用于这两位被试与该四次预算。[正式输入与评分](PAPER_BENCHMARK_PROTOCOL.md)

| 官方 R4，各 400 张 | 自动主体命名积分 | 找回原照 | SSIM |
| --- | ---: | ---: | ---: |
| Brain-HiVE | 4 | 23 | .32574 |
| 时域读出 | 5 | 26 | .34314 |
| 平均波形局部功率 | 6 | 32 | .34109 |
| 平均波形局部功率，交换 | 8 | 26 | .34001 |
| 跨呈现局部功率 | 2 | 26 | .34350 |
| 跨呈现局部功率，交换 | 5 | 30 | .34296 |
| 呈现间局部时频残差 | 7 | 26 | .34332 |
| 呈现间局部时频残差，交换 | 8 | 27 | .34300 |

时频残差相对 Brain-HiVE 的自动命名差为 +0.75 个百分点，95% 配对区间为 [−1.00, +3.00]。正确残差相对错配差为 −0.25 个百分点 [−1.50, +1.00]。平均波形功率组两人各比基线多一次自动命名，补做错配对照得到 8 次，比正确功率多两次。两种自动增加均缺乏正确输入优势。

仅加入时域读出，SSIM 相对基线提高 .01741。再加入时频残差，SSIM 相对时域读出的差为 .00018，95% 区间 [−.00148, +.00186]。400 张重跑的 Brain-HiVE 基线与正式基准逐张散列一致。

在固定编号里，第 07 名被试的老鹰生成成了狗，篮球生成成了碗。还逐图查看了自动命名变化的样例。第 10 名被试的羚羊被两个残差版本都计作命中，画面却呈现猫科外形，真照片有明显的角和长腿。这两次命中保留在统一的自动计分表中，目视检查另行记录。

共核验 3,200 张成图。对应 EEG 的局部时频残差有可测的成图前判别信息。当前读出在主体命名优先的基准中没有可靠增益，保留为研究实验。[逐组分数、错配区间与输入回执](results/local_spectra.json)

## 复跑

沿用正式基准环境并为每个被试设置个人检查点路径。用既有的原生折外条件和 DINO 训练特征文件锁定校准，再在计算节点成图。

```bash
python paper_spectral_conditions.py screen \
  --prep "$RAD_ROOT/data/prep" --eeg-root "$RAD_ROOT/data/prep" \
  --photos "$RAD_ROOT/data/prep/F_train.npy" \
  --source-root "$EXISTING_CROSSFIT_RUN" --output "$EEG_CROSSFIT_ROOT/screen.json"
python paper_spectral_conditions.py render 7 --calibration "$EEG_CROSSFIT_ROOT/screen.json"
python paper_spectral_conditions.py render 10 --calibration "$EEG_CROSSFIT_ROOT/screen.json"
python paper_spectral_consistency.py \
  --prep "$RAD_ROOT/data/prep" --eeg-root "$RAD_ROOT/data/prep" \
  --photos "$RAD_ROOT/data/prep/F_train.npy" --source-root "$EXISTING_CROSSFIT_RUN" \
  --calibration "$EEG_CROSSFIT_ROOT/screen.json" --output "$EEG_CROSSFIT_ROOT/timing-and-pairs.json"
```

平均波形功率错配在初步成图后补跑。保留原校准文件，将 `EEG_CROSSFIT_ROOT` 设为另一个输出目录，使用同一位被试和同一组参数调用 `paper_spectral_conditions.render(subject, calibration_path, ["locked-wrong"])`。
