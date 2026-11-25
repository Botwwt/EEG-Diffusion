# THINGS-EEG 作者报告的重建结果

这些数值抄录自原论文。公开 ATM 图像的独立重新评分另见本仓库的基准报告。AlexNet、Inception、CLIP 均为两选一识别率，SwAV 为距离。

| 作者主表 | 被试范围 | PixCorr / PCC | SSIM | Alex2 | Alex5 | Inception | CLIP | SwAV ↓ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ATM，表 1 | 被试 08 | 0.160 | 0.345 | 0.776 | 0.866 | 0.734 | 0.786 | 0.582 |
| D²-FOSA，表 3 | 重建表未列被试范围 | 0.193 | 0.350 | — | — | — | — | — |
| Jo 等，表 1 | 10 人均值 | 0.263 | 0.370 | 0.921 | 0.942 | 0.813 | 0.799 | 0.500 |
| ProtoEEG，表 3 | 10 人、5 个种子均值 | 0.154 | 0.390 | 0.779 | 0.855 | 0.743 | 0.768 | 0.506 |
| SPSC，表 17 | 10 人各自训练，SCG-Recon | 0.251 | 0.380 | 0.845 | 0.883 | — | 0.849 | 0.560 |

ATM 公布了十名被试 200 个刺激的多候选成图。共同评分固定取文件名为 `0.png` 的候选，对所有被试采用相同规则。[ATM 表 1](https://arxiv.org/html/2403.07721v7#S3.T1) · [作者成图发布版本](https://huggingface.co/datasets/LidongYang/EEG_Image_decode/blob/dbe34bb2407164f70883c661c76664f0d596e522/generated_imgs.tar.gz)

D²-FOSA 还报告 IS 11.81、FID 146.33 和 KID 0.025。其补充材料按最佳、中位和最差分为三个图版，每组展示 16 个目标。[CVF 主文表 3](https://openaccess.thecvf.com/content/CVPR2026/papers/Yu_D2-FOSA_Dual-Diffusion_Guided_EEG-to-Image_Reconstruction_with_Frequency-Oriented_Semantic_Alignment_CVPR_2026_paper.pdf#page=7) · [补充图 10–12](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Yu_D2-FOSA_Dual-Diffusion_Guided_CVPR_2026_supplemental.pdf#page=15)

Jo 等的生成器每张做 50 次 Euler 函数评估。表 1 报告跨被试均值及标准差，图 4 展示被试 08 的九个刺激。[Jo 等表 1 与图 4](https://arxiv.org/html/2611.19232v1#S4.T1)

ProtoEEG 使用共享的可学习视觉原型。这里抄录作者主表的五种子结果；附表另有采用不同评分器的归档数字。[ProtoEEG 表 3](https://arxiv.org/html/2610.33065v1)

SPSC 用同物体的训练照片作为正例，并用子空间统计选择读出。表 17 是逐人训练的完整重建结果，论文另报告留一被试实验。[SPSC 表 17](https://arxiv.org/html/2611.09380v1)
