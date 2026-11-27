# ICME 2027 EEG 重建研究

每个测试查询只输入一名被试的实测 EEG 和固定种子。模型以逐张 EEG 预测的粗布局起步，四步生成一张图。训练时让同一物体的四张成图分别匹配各自的原照。测试用词表和原照只参与生成后的评估。

## 论文和成图

- [英文初稿 PDF](paper/ICME2027-draft.pdf) · [LaTeX 源文](paper/source/manuscript.tex) · [完整源码包](paper/LaTeX-source.zip) · [超参数依据](paper/parameter-notes.md)
- [统一配置 R4 固定对照](experiments/full-metric-benchmark/figures/common-r4-fixed-panel.png) · [统一配置 R80 固定对照](experiments/full-metric-benchmark/figures/common-r80-fixed-panel.png) · [原生系统成图面板](paper/source/fixed-panel.png)
- [当届 ICME 投稿说明和 SPS 模板](https://2027.ieeeicme.org/authors/initial-author-instructions)

## 结果

| 对照 | 范围 | 主表与记录 |
| --- | --- | --- |
| 同一渲染器 | 官方 R4、R80，各 10 人与 200 张照片。六种方法共 24,000 张成图，12 项主指标。 | [完整主表](experiments/full-metric-benchmark/RESULTS.md) · [各方法的配对区间](experiments/full-metric-benchmark/full-benchmark-results.json) |
| 原生端到端流程 | 官方 R80，和 ATM、CognitionCapturer、Brain-HiVE、ENIGMA 比较。每种方法 2,000 张。 | [原生主表](NATIVE.md) · [数值来源](paper/evidence/native-table-audit.json) |
| 同组共享布局 | 十人开发集，每组 2,000 张。共享布局降低 SSIM。 | [完整对照](experiments/shared-training-layout/RESULTS.md) |
| 扣除布局参照交互 | 十人开发集，每组 2,000 张。原照与主体差异的区间均跨零。 | [完整对照](experiments/layout-reference-assignment/RESULTS.md) |

统一配置 R80 的 DINOv2 原照首位为 16.25%，Brain-HiVE 编码器为 7.55%，配对差为 +8.70 个百分点，95% 区间 [5.25, 12.30]。同批成图的 SigLIP2 主体命名为 2.075% 对 1.325%，区间跨零。原生流程中的 CLIP 原照首位由 ENIGMA 领先。

## 复核与复现

在 GitHub 仓库根目录安装 Python 3.11 或更新版本与项目的 dev 依赖。以下命令从逐图存档复核 24,000 张成图的统一结果。

```bash
python -m pip install -e '.[dev]'
python research/icme2027/experiments/full-metric-benchmark/verify_canonical.py \
    --stack-code research/eeg_stack_20261112/code \
    --canonical research/icme2027/paper/evidence/common-renderer-audit.json
python research/icme2027/experiments/full-metric-benchmark/collect_results.py \
    --aggregate-source research/eeg_stack_20261112/code/paper_aggregate.py \
    --canonical research/icme2027/paper/evidence/common-renderer-audit.json \
    --siglip-archive research/icme2027/experiments/full-metric-benchmark/common-siglip2.tar
```

- [评分输入与逐图回执](experiments/full-metric-benchmark/README.md)
- [GPU 训练与推理说明](../eeg_stack_20261112/PAPER_BENCHMARK_PROTOCOL.md) · [公开数据与权重准备](../eeg_stack_20261112/public_inputs/README.md)
- [历次实验索引](../eeg_stack_20261112/README.md) · [主体失败案例与选型记录](../../docs/OBJECT_LOG_REVIEW.md)
- [文件 SHA-256](SHA256SUMS)
