# 同一张照片的多人 EEG 观测

在两个已经研究过的面板上，每张照片都沿用 sub-04 的同一个物体词预测、八张校准照片、扩散种子和图像条件幅度。变化的是用于估计照片方向的 EEG。三个预算依次为 sub-04 的四次呈现，sub-04、05、01 的十二次，以及再加 sub-02、06、07、08、09、10 的三十六次。每张照片在每个预算下只有一次联合预测。三个预算的物体词均命中 180/240。此前的三人实验还允许多人共同预测物体词。[预先登记的方法和来源哈希](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_photo_dose_design.json)

每人的照片读出在相同的 128 个额外物体上训练，每类十张。训练时分别减去本类 EEG 和 H/14 图像特征均值，用固定 ridge 2 的线性模型回归。预测时，用冻结物体词的八张本折支持照片作原型，再从当前 EEG 预测中减去这些支持的 EEG 预测均值。每人的残差投影到原型的切空间，归一化后等权求和。各预算均沿用原 sub-04 模型为该查询估出的角度。原分类器仍使用 512 个校准物体。九人同词同折错配使用另一条查询的残差方向和接收查询的角度。

## 生成 JPEG 的独立评估

用独立的 DINOv2-S/14 比较真实生成图与原图，九人观测比单人多认对 **8/72 对**，两条查询属于同一物体、同一折且两个物体词都预测正确。把每张生成图单独与同类十张原照比较时，九人只多认对 **5/240 张**，其区间跨零。两种任务的分母、候选数和决策方式不同。[2,160 张 JPEG 的独立评估](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_photo_dose_jpegs.json)

| 已保存 JPEG 的指标 | 1 人 | 3 人 | 9 人 | 仅八图原型 | 9 人错 EEG |
| --- | ---: | ---: | ---: | ---: | ---: |
| DINO 同类十图，全体 240 张 | 25 | 29 | 30 | 17 | 21 |
| DINO 同类十图，物体词正确的 180 张 | 16 | 21 | 18 | 12 | 12 |
| DINO 严格两图，72 对 | 42 | 42 | 50 | 39 | 23 |
| DINO 两张候选中独立判断每张，144 条 | 78 | 77 | 79 | 70 | 60 |
| 补充 H/14 同类十图，240 张 | 23 | 21 | 27 | 19 | 19 |
| 补充 H/14 严格两图，72 对 | 44 | 45 | 55 | 38 | 23 |
| 32×32 RGB 误差，越低越好 | .2985 | .2946 | .2844 | .2997 | .3005 |
| 32×32 空间亮度相关，越高越好 | .0336 | .0362 | .0578 | .0276 | .0248 |

在注册面板，DINO 严格两图从单人 19/33 升至九人 23/33；在复验面板，从 23/39 升至 27/39。合并后九人相对单人高 11.11 个百分点，按物体聚类的 95% 区间为 +1.37 至 +21.05。九人比错 EEG 高 37.50 点（+24.72 至 +49.21），比仅八图原型高 15.28 点（−1.64 至 +30.26）。同类十图中，九人相对单人高 2.08 点（−1.25 至 +5.42），相对仅八图原型高 5.42 点（+2.08 至 +8.75）。三个预算在每张查询上使用相同种子，240 张原型对照在各预算间的 JPEG 字节逐一相同。

增加观测还改善了低频像素相似度。九人相对单人的 RGB 误差下降 .0141（−.0185 至 −.0097），空间亮度相关上升 .0241（+.0023 至 +.0458）。九人的亮度相关本身只有 .0578，留下了较大的布局差距。所有区间固定参与者，在每个面板内重采样整个物体，保留各方法的配对关系。

两张图册各取面板每类官方行号最小的照片。复验图册将换挡杆和拳击沙袋都解码为炉灶，三个观测预算都沿用这个错误词。注册图册也保留了鞋被解码为剪刀的例子。标签注明所看物体和 EEG 预测词。[注册面板的 12 类生成图](../research/eeg_stack_20261112/gallery/fixed_word_dose_registered.jpg) · [复验面板的 12 类生成图](../research/eeg_stack_20261112/gallery/fixed_word_dose_replicate.jpg)

### 九人扩散引导对照

固定九人条件、词、随机种子和四步调度，在每一步保留八图原型的噪声预测，再加上两倍的匹配残差。两种采样的全部 240 张原型 JPEG 字节相同。独立 DINO 同类十图从普通九人的 30/240 升至引导版 37/240（注册 15→20，复验 15→17），增幅 +2.92 个百分点，95% 区间 −0.42 至 +6.67。引导版错 EEG 为 22/240，八图原型为 17/240。严格两图由 50/72 变为 49/72，差值 −1.39 点（−9.33 至 +6.94）。32×32 RGB 误差从 .2844 升至 .2916，增加 .0072（+.0045 至 +.0099）。只看词预测正确的 180 条，引导版同类十图为 24/180，普通版为 18/180。H/14 的单图检索从 27/240 降为 23/240，严格配对从 55/72 降为 48/72。该采样系数在两个主指标上的比较区间均跨零。[引导设计](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_guidance_design.json) · [成图配对评分](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_guided_jpegs.json) · [注册图册含引导列](../research/eeg_stack_20261112/gallery/fixed_word_dose_registered_guided.jpg) · [复验图册含引导列](../research/eeg_stack_20261112/gallery/fixed_word_dose_replicate_guided.jpg)

### 生成图里的物体名

独立 OpenAI CLIP-L/14 把生成的 JPEG 与物体词比较。四种方法（单人、三人、九人、九人引导）在本面板 12 个候选词中的命中都为 180/240，九组普通臂和三组引导臂的命中数相同。在全部 1,854 个 THINGS 词中，匹配 EEG 臂分别为 167、169、168、167/240。用同一 CLIP 对生成图和原图分别按面板中心化后做同类十图检索，匹配臂分别命中 28、21、32、35/240；原型为 20/240，九人普通及引导错配分别为 20、23/240。加观测的证据主要落在照片配对和低频像素，类别词预测在设计中已固定。[独立 CLIP 标签与检索评估](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_photo_dose_names.json)

## 扩散前的 H/14 结果

| 指标 | 1 人 | 3 人 | 9 人 | 9 人错配 | 仅八图原型 | 原单人 512 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原始余弦，同类十张选一，全部 240 条 | 11 | 16 | 15 | 7 | 7 | 12 |
| 原始余弦，同类十张选一，词正确的 180 条 | 4 | 7 | 4 | 2 | 2 | 5 |
| 同类同折两张的联合指认，两个词都预测正确，72 对 | 51 | 59 | 64 | 12 | 36 | 55 |
| 减去支持原型后，同类十张选一，全部 240 条 | 39 | 47 | 61 | 20 | 24 | 53 |
| 减去支持原型后，同类十张选一，词正确的 180 条 | 24 | 35 | 50 | 12 | 18 | 39 |

九人相对一人的联合指认多 13/72 对，差值为 +18.1 个百分点，按物体配对的 95% 区间为 +7.8 至 +27.8。九人相对三人的区间为 −5.6 至 +19.5。全部 72 对照片对九人都具有相同的两次采集场次编号。原始十图检索只多 4/240，区间为 −1.25 至 +4.58 个百分点。预测词正确时，十张候选中包括推断用过的八张支持照片，当前查询是另两张之一。这解释了为什么原型本身在这一检索指标上得分很低。预先登记的减原型十图检索多 22/240，区间为 +1.67 至 +16.25 个百分点。两个面板都是既有刺激，区间描述固定九人对不同物体的变化。[分面板 H/14 汇总](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_photo_dose.json)

## 复现

准备步骤复用九人原始校准包、双面板物理 EEG 和 sub-04 冻结词。所需本地输入位于 `outputs/object-transfer/{generic,halves}`、`outputs/object-transfer/sub-XX.npz`、`outputs/object-fidelity`、`outputs/official-object-readout/sub-03` 和 `outputs/photo-structure/{expanded,within_photo,residual_angle}`。归档检索、行映射与哈希的复现见[物体任务复现页](OBJECT_FIDELITY_REPLAY.md)。从仓库根执行以下命令。

```sh
CODE=research/eeg_stack_20261112/code
python "$CODE/fit_fixed_word_photo_dose.py" register
OPENBLAS_NUM_THREADS=4 python "$CODE/fit_fixed_word_photo_dose.py" prepare
OPENBLAS_NUM_THREADS=4 python "$CODE/fit_fixed_word_photo_dose.py" score
python "$CODE/score_fixed_word_photo_dose.py" --register
ROOT=outputs/fixed-word-photo-dose/render_inputs
for group in single-1 cohort-3 cohort-9; do
  python "$CODE/render_fixed_word_photo_dose.py" --root "$ROOT" --subject "$group" --check-only
  python "$CODE/render_fixed_word_photo_dose.py" --root "$ROOT" --subject "$group" \
    --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY"
done
python "$CODE/score_fixed_word_photo_dose.py"
python "$CODE/gallery_fixed_word_photo_dose.py"
```

每个生成包按原实验的四步 SDXL-Turbo 和 ViT-H IP-Adapter 绘制两面板各 120 条的 own、uniform、wrong 三臂。评分在读取生成包和 JPEG 的 SHA 后，从固定官方 ZIP 取原图，计算独立 DINOv2-S/14 最后层 CLS 的同类十选一和严格两图指认。补充指标包括同一视觉训练空间 H/14 的成图检索、32×32 RGB 均方根误差和空间亮度相关。行级记录和原图留在本地。[独立成图评估登记](../research/eeg_stack_20261112/results/object_fidelity/fixed_word_photo_dose_jpeg_design.json)

复现引导版时，用普通九人的完成登记作对照。以下命令还会重算两个面板的独立词典评分。

```sh
WORK=outputs/fixed-word-photo-dose
GUIDED="$WORK/guided_inputs"
PUBLIC=research/eeg_stack_20261112/results/object_fidelity
mkdir -p "$GUIDED"
cp "$ROOT/design.json" "$ROOT/radius.json" "$ROOT/study_design.json" "$GUIDED/"
cp -R "$ROOT/conditions" "$GUIDED/"
python "$CODE/render_fixed_word_guidance.py" --root "$ROOT" --baseline "$ROOT/renders" \
  --out "$GUIDED" --archive "$ARCHIVE" --adapter "$ADAPTER" --dictionary "$DICTIONARY"
python "$CODE/score_fixed_word_photo_dose.py" --extra-root "$GUIDED" \
  --release "$PUBLIC/fixed_word_guided_jpegs.json"
python "$CODE/gallery_fixed_word_photo_dose.py" --extra-root "$GUIDED"
for panel in registered replicate; do
  mkdir -p "$WORK/names/$panel"
  python "$CODE/score_within_photo_ip.py" --fixed-word-dose --root "$ROOT" \
    --subjects single-1 cohort-3 cohort-9 --panels "$panel" --archive "$ARCHIVE" \
    --dictionary "$DICTIONARY" --reference-dir outputs/object-fidelity \
    --out "$WORK/names/$panel/ordinary.json"
  python "$CODE/score_within_photo_ip.py" --fixed-word-dose --root "$GUIDED" \
    --subjects cohort-9 --panels "$panel" --archive "$ARCHIVE" \
    --dictionary "$DICTIONARY" --reference-dir outputs/object-fidelity \
    --out "$WORK/names/$panel/guided.json"
done
python "$CODE/summarize_photo_dose_names.py"
```
