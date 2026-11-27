# 校准生成器入口的 EEG 图像条件

四次 EEG 的 CBraMod 读出提高了成图前的原照匹配率。实际图里却反复出现面包棒、托盘和工具，第 06 名被试的正确与错配组分别有 17 和 18 张被 SigLIP2 标成面包棒。本轮测量条件在冻结非线性投影前后的幅度，并检验校准后能否补充共享 D2。[上一轮对照](CBRAMOD_REPEAT_STUDY.md)

## 测量条件的逐图变化

每人取 13,230 张注册训练照片的实际四次 EEG 和对应的真实融合特征。用同一冻结适配器及投影得到四个图像 token。令真训练照片的 token 均值和 RMS 分别为 $\mu_V$ 和 $\sigma_V$。对每种读出输入计算它的逐图 RMS $\sigma_E$ 与中心偏移 $\|\mu_E-\mu_V\|$。右两列的官方 200 条仅用于诊断读出在推理时达到的幅度。

| 图像 token 的来源 | 06 训练 $\sigma_E/\sigma_V$ | 09 训练 | 06 官方 | 09 官方 |
| --- | ---: | ---: | ---: | ---: |
| 逐人 Brain-HiVE | 1.959 | 1.956 | 1.893 | 1.912 |
| CBraMod 联合 | .286 | .240 | .256 | .232 |
| CBraMod 逐次 | .290 | .249 | .259 | .237 |
| 共享 D2 | 1.111 | 1.108 | 1.088 | 1.072 |

联合组在两人的正式输入上的中心偏移是自身逐图 RMS 的 1.57 和 1.73 倍。投影前的这一比率只有 .22 和 .19。联合模型的融合条件本身已收缩到训练照片的 .176 和 .160 倍。冻结投影把这批窄范围输出转成了具有较大共同偏移的 token，与成图反复出现相近主体的现象一致。[统计实现](code/paper_condition_probe.py)

## 同一生成器里的两种校准

单独使用联合编码的条件时，整体平移对照使用 $T'=\mu_V+(T-\mu_E)$。第二组按训练照片与训练 EEG 的 RMS 比同时校准幅度。

$$
T'=\mu_V+\frac{\sigma_V}{\sigma_E}(T-\mu_E).
$$

另把该组的逐图偏移方向换成相隔 100 行的另一张图的 EEG 预测，重缩放到原查询偏移的相同范数。两组的中心与每张图的条件幅度完全相同。三组均从高斯噪声出发。

共享 D2 保留原来的 M1a 读出、图像 token、逐人布局和四步生成。令 $R(T)$ 表示每个 token 减去该图四个 token 的均值。先用各自训练照片的 $R$ 估计 D2 中心 $\mu_D$ 和联合组中心 $\mu_C$，并用真实照片和联合组的训练相对幅度得到比例 $g_R$。对当前查询，令 $v=R(T_D)-\mu_D$ 和 $u=g_R[R(T_C)-\mu_C]$，只加入联合组沿 D2 当前方向的正交分量。

$$
T_D'=T_D+\tfrac12\left(u-v\frac{\langle u,v\rangle}{\|v\|^2}\right).
$$

此操作保留 D2 在 $v$ 方向的系数和四个 token 的均值。错配对照保留该查询的 D2 和布局，只替换 CBraMod 的 $u$，再令正交修正的范数与正确查询完全相同。系数 0.5 在生成前固定。这一路多算一个已训练的 CBraMod 编码器，两组同样使用十人训练的 D2。真正生成时只接收一个人的四次实测 EEG，控制组由明确登记的错配产生。[校准和采样实现](code/paper_condition_calibration.py)

选择第 06、09 名被试复查上轮的主体重复现象。每人 200 个相同的官方刺激。固定空文本、逐刺激种子、一次出图和四步 SDXL-Turbo。在原生 CLIP 和独立 SigLIP2 下使用完整 1,854 词表给实际成图评分。用于出图的校准统计只读取注册训练照片与训练脑电。评分读取测试原照和词表。[正式协议](PAPER_BENCHMARK_PROTOCOL.md) · [独立评分入口](code/paper_independent_naming.py)

## 实际成图结果

本轮生成并核对 2,400 张图。共享 D2 的 400 张新跑基线与旧归档逐张 PNG 散列一致。正确和错配 D2 修正的范数在转换成半精度前最大只差 $9.54\times10^{-7}$。两人的整体校准增益分别为 3.494、4.166，相对 token 增益为 3.501、4.171。实际加入 D2 的正交修正中位幅度相当于其原条件偏移的 33.8% 和 37.4%。

| 同两人的官方 R4，每组 400 张 | CLIP 主体积分 | SigLIP2 主体积分 | CLIP 找回原照 | SSIM |
| --- | ---: | ---: | ---: | ---: |
| Brain-HiVE，历史 | 0 | 2 | 12 | .30245 |
| CBraMod 联合，历史 | 1 | 3 | 7 | .34536 |
| CBraMod，只平移训练中心 | 3 | 3 | 6 | .30981 |
| CBraMod，平移及幅度校准 | 0.5 | 1 | 14 | .32961 |
| CBraMod，校准后换另一条 EEG 的方向 | 0 | 0 | 0 | .30136 |
| 共享 D2 重跑 | 5 | 7 | 24 | .41729 |
| 共享 D2，加正配 EEG 的相对修正 | 3 | 3 | 14 | .41366 |
| 共享 D2，加等范数错配修正 | 6 | 6 | 17 | .41733 |

只在联合编码上做幅度校准后，SigLIP2 给两人 200 张图预测的不同标签种数由 113、114 变为 150、175。固定样例中的托盘、面包棒和工具不再反复占据同一画面类型。第 09 名被试的白色连帽衫可以辨认，航母、香蕉和鹰依然画错。幅度校准后正确 EEG 与等范数错配的原照匹配是 14 和 0，差 +3.50 个百分点，95% 区间 [+1.50, +6.00]。相对同轮只平移组的原照差为 +2.00 个百分点，区间 [−0.50, +4.50]。幅度校准的主体命中仍很低。

对共享 D2 加入相对修正后，两个命名器的积分都下降。换成错配修正后，两种命名器的合计积分都高于正配。目视比较中，D2 原有的橙子和四足羚羊形态在错配修正下也能辨认，海藻在这三组仍多像暗色地形。当前候选保留共享 D2 的原成图路径。补充模块需要在布局、种子和幅度一致的配对控制下提供更准确的物体方向。[逐图回执、标签、配对区间](results/condition_calibration.json)

历史 CBraMod 联合参考使用每批 32 条且末批 8 条；本轮探针把 200 条输入近似等分给七批。按旧批次重算后，融合特征的最大绝对差为 $2.38\times10^{-7}$，转换半精度后有 6/400 个查询的图像 token 改变，差值最高为 .00098。历史联合组只作为诊断参考。同轮的两个校准、同轮错配以及三组 D2 比较各自固定了输入批次。两位被试均参加上一轮查看，本轮结果用于定位现象。

## 复现

准备上一轮通过输入核验的个人 CBraMod 检查点、共享 D2 和原生生成器。运行时的 `EEG_RUNS_ROOT` 指向 `fits/shared`。使用同一个 `RAD_ROOT` 和此前的依赖源目录，在计算节点测量条件并成图。

```sh
for sub in 6 9; do
  python paper_condition_probe.py "$sub" --previous "$CBRAMOD_RUN" --output "$THIS_RUN/probe"
  pack="$THIS_RUN/probe/conditions-sub-$(printf '%02d' "$sub").pt"
  python paper_condition_calibration.py "$sub" --probe "$pack" --output "$THIS_RUN/images"
done
EEG_CROSSFIT_ROOT="$THIS_RUN" python audit_condition_batch_precision.py
```

将 `probe/conditions-sub-*.json`、`probe/prior-batches.json`、`images/sub-*/calibration.json`、各组 PNG 和生成及评分回执收齐。独立评分所需归档包含各组的 PNG 与生成回执。用上一轮相同的图像集合和 SigLIP2 修订评分，再核对归档图像、输入散列、布局、种子与逐图差分。

```sh
python paper_independent_naming.py --model "$SIGLIP2" --assets "$IMAGE_SET" \
  --archive "$RGB_ARCHIVE" --output independent-naming.json --subjects 6 9 \
  --arms shared-d2 center spread spread-wrong d2-relative d2-relative-wrong
python summarize_condition_calibration.py --receipts "$THIS_RUN" \
  --independent independent-naming.json --prior-receipts "$CBRAMOD_RECEIPTS" \
  --prior-names "$CBRAMOD_NAMES" --output condition_calibration.json
```
