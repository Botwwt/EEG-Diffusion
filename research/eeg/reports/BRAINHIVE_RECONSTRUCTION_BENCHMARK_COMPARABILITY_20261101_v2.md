# 重建比较口径、宿主排重与独立路线的下一项证据

2026-11-01，15:36 UTC。本轮新增工作为原始来源复核和后续目标审查；原完整ordinary持续训练，三个既定后续控制器继续等待。没有追加GPU任务或研究图片，本文件不修改其封存协议。

## 公开数字对应什么实验

| 来源 | 本轮实际核对 | 对下一步的影响 |
|---|---|---|
| [ENIGMA原论文4.1节、表1及附录A.3](https://arxiv.org/html/2602.10361v1) | 十被试THINGS-EEG2；每query十次重建取指标均值。单被试配置CLIP为0.8148，多被试为0.8033；不是DINO。标题十五分钟指新被试校准数据时长。 | 保留为后续同口径复现对象；不能用十图均值直接排名当前单图结果，也不把十五分钟当训练耗时。 |
| [NVOL原论文摘要、表III](https://arxiv.org/html/2609.02582v1) | 86.4%是200选一检索。图像重建表仅sub-08：两阶段CLIP为0.799、SSIM为0.366。 | 不将检索准确率或单被试图像表当十被试生成竞争力；本轮尚未核实其完整生成次数、步数和可用权重。 |
| [D²-FOSA的CVF正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Yu_D2-FOSA_Dual-Diffusion_Guided_EEG-to-Image_Reconstruction_with_Frequency-Oriented_Semantic_Alignment_CVPR_2026_paper.pdf) | 原始来源的索引文本表3列IS/FID/KID/SSIM/PCC，主组为11.81/146.33/0.025/0.350/0.193。论文称代码随补充材料提供。直接打开正文和页面返回403；尚未取回核验补充源码。 | 频率编码、双向扩散循环已有正式近邻；不能改名后作为原创。该表不能转换为HIVE的DINO/CLIP表，不据索引文本宣称已完成复现。 |

NVOL全文的生成比较范围，是在先前摘要级筛选基础上补充的证据；ENIGMA的结构差异、指标问题和校准时长解释，早在[2026-10-20记录](AUTHOR_REPRODUCTION_AND_COMPONENT_TRANSFER_20261020.md)中已存在，不记成新发现。OpenReview题名“Learning Brain Representation with Hierarchical Visual Embeddings”对应现有Brain-HIVE，不再计为另一个宿主。

## ENIGMA作者源码没有更新

本轮实际查询[作者仓库main提交API](https://api.github.com/repos/Alljoined/ENIGMA/commits/main)，仍为已保存的`460af4b6a0ca6eee11aeaf35c5431e8e3d55c11a`。同时[GitHub Releases API](https://api.github.com/repos/Alljoined/ENIGMA/releases)返回空列表。后者只说明该发布渠道没有条目，不证明所有网站都没有权重。

精确HTTP响应、读取时间和响应摘要保存在[原始来源核验目录](../audit/benchmark_primary_scope_review_20261101_v1/)。无需重复下载同一源码或重做原先的未改动网络类CPU检查；此前仅有接口检查，完整训练和图片复现仍未完成。

固定版本2AFC函数先作严格小于比较、再减一的脚本问题已有实际自配对检查。继续保留作者口径与统一口径的区别；**不能据此给论文表格机械加1/199**，因为论文实际执行版本尚未建立。分母、编码器、特征比较方向和生成重复数均需绑定实际调用。

## 两套作者指标的真实输入路径

本轮实际解析两个固定作者提交，并将本地HIVE评价runtime的SHA与已经闭合的ordinary执行清单逐项匹配。CLIP两边均为`ViT-L/14`，最终Resize224与Normalize的AST完全一致；这只证明该段变换相同。

ENIGMA生成入口先以JPEG质量75保存重建，再解码缩至224；其评价入口另将真实目标缩至224。HIVE当前作者评价runtime要求每被试200×3×512×512的输入。两边PixCorr／SSIM还分别使用512与425的Resize。即使最终CLIP变换相同，先前序列化和缩放后的像素仍未证明一致；不能把跨论文同名SSIM／CLIP之差当同协议配对效应。

该检查没有运行模型或重新评分任何研究图片，也没有核验ENIGMA论文实际使用的执行版本。最初静态回执保留；v2进一步自动核验作者源清单、两个CLIP加载名称和实际HIVE输入形状断言。[实际源码与执行绑定核验](../audit/benchmark_primary_scope_review_20261101_v1/actual_HIVE_ENIGMA_static_metric_scope_v2.json)。原[第一版报告](BRAINHIVE_RECONSTRUCTION_BENCHMARK_COMPARABILITY_20261101_v1.md)保留原SHA。

当前封存评价器保持原口径；未来若复现ENIGMA，需要同时保留作者输出链与共同评价的原始输出，分别报告。不能为提高某个分数，事后只对某一方法采用更有利的JPEG、缩放或计数流程。

## 独立VAR路线：完整似然之外还需EEG专属增量

[当前分尺度潜类别草案](BRAINHIVE_INDEPENDENT_ROUTE_PRIMARY_AND_MULTISCALE_LATENT_REVIEW_20261101_v1.md)已区分完整680项、旧前5项和类别相关共同项。现有工程队列检查的是原生核与实际成本；均匀先验的图像后验只用于工程复算，不是EEG可预测性证据。

下一份实际训练协议若获得准入，应增加明确的EEG无关先验对照：在相同允许T图像和固定原生核上拟合不接收EEG的视觉边际类链`p_phi(Y)`，并报告

\[
g(Y,E)=\log p_\theta(Y\mid E)-\log p_\phi(Y).
\]

这衡量给定两个已拟合模型的留出log score差。只有两者分别等于真实条件分布与真实边际分布时，其总体期望才等于互信息；一般情况下不能叫已测得的EEG信息量。训练中若只胜均匀先验，可能只是学到了更好的通用类别频率。

另对同一图像固定全部原生似然，比较正确EEG与预先固定的错配EEG：

\[
d(Y,E,E_\pi)=\log p_\theta(Y\mid E)-\log p_\theta(Y\mid E_\pi).
\]

视觉边际项在该配对差中严格抵消。不同概念错配检查粗语义；同概念无固定点错排检查刺激方向。错配身份只作评价干预，正常推理不读取概念或图像答案。必须在训练权限之外评价，不能把训练图的后验集中当泛化收益。

完整后尺度的teacher forcing还会读入真实早尺度前缀；正常生成时前缀由模型自身产生。因此留出似然即使提高，也必须继续通过完整实际图片比较，不能直接宣布解决了生成时的误差累积。

这些是未来协议必须区分的解释问题，**尚未实现或准入新EEG拟合**，也不是新增原创公式。既定的完整目标对粗目标、类链对完整全局类别、配对EEG对错配EEG三项比较继续保留；不从T8工程结果选择有利类别、尺度或温度。

## 当前执行决定

先完成同一完整ordinary、训练内共同方向对照和已挂队原生VAR工程检查。ENIGMA不因旧源码重新被找到而重跑；NVOL暂不依据86.4%启动生成迁移；D²-FOSA的补充实现及公平成本仍未核实，不凭论文名称直接拼接。原生VAR核存在有效类别差异且成本可行之后，才具体实现并审查完整银行和EEG目标。

该顺序保留有实证基础的第一方法，并将第二方向的下一次训练约束为可区分“通用先验改善”与“EEG增量”的实验。目前没有确认的第二方法、SOTA或CCF A结论。
