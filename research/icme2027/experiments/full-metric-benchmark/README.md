# 统一配置评分与存档

十名被试的官方 R4 与 R80 各有 200 张测试图。六种方法共享布局、图像适配器、生成器、采样步数、空提示和逐查询种子。每个方法和重复预算有 2,000 张实际成图。

| 文件 | 内容 |
| --- | --- |
| [common-canonical-scores.tar](common-canonical-scores.tar) | 120 套原始评分、Inception 分数、成图回执和 72 张 RGB 样图 |
| [common-dino.tar](common-dino.tar) | 同一批成图的 DINOv2 原照排名与辅助分数 |
| [common-siglip2.tar](common-siglip2.tar) | 同一批成图的 SigLIP2 完整词表命名，覆盖 1,854 个概念 |
| [full-benchmark-results.json](full-benchmark-results.json) | 汇总均值、逐方法的配对区间与散列 |
| [RESULTS.md](RESULTS.md) | 两种试次数的全指标与 DINOv2 附表 |

验证先核对正式刺激、已注册的原始分数和生成回执，再将 DINOv2 和 SigLIP2 绑定到同一组生成像素。DINOv2 和 SigLIP2 使用冻结模型，各方法使用同一图库和词表。详细的原始训练与成图入口见仓库中的基准协议。

Slurm 评分脚本复用基准的计算节点环境。SigLIP2 使用 [固定版本的公开模型](https://huggingface.co/google/siglip2-base-patch16-224/tree/75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2) 与 [依赖散列](runtime.sha256)。用该散列清单在工作目录的 packages 子目录准备 Linux 依赖，按启动脚本所示配置官方图库和成图 tar。主体命名作业的最终队列记录见 [此处](slurm-siglip2-status.txt)。

固定对照图取被试 08 的官方测试顺序前六张。用 make_panels.py 核对成图与原照散列后生成面板。
