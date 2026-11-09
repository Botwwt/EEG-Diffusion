# T32 原生四步路径梯度：完整结果与训练目标解释

2026-10-29 10:18:49，America/Los_Angeles。全部32条、32概念、两个固定分段及512次实际UNet前向/重算闭合，无新研究图和方法更新。全部有梯度路径终点latent与原生无梯度回放逐位相同，四次祖先噪声一致，冻结模块前后权重指纹一致。FP16 autograd仍是连续松弛，不是量化文件的数学导数。

事前主读数为旧噪声+0.1token目标与四步latent终点MSE梯度的切空间余弦：均值 **+0.098797**，描述性区间 **[+0.036925,+0.161676]**；8/32条负向。平均方向正向，不支持“旧目标整体反向”的简单解释。局部输入梯度不等于参数/Adam步，latent MSE不等于语义图片效果。

两项事后分解全部报告：epsilon项均值余弦 **+0.131761**，区间 **[+0.058510,+0.207067]**；token项 **+0.001042**，区间 **[−0.027827,+0.027627]**。0.1token梯度范数与epsilon梯度范数之比中位数约0.728，不能说token没有作用；也不能据此证明它损害图片或现场选loss权重。这只支持另行登记匹配损失控制，验证实际图片原因。

本机QR投影、全部32条主余弦和概念bootstrap独立重算，最大余弦差1.665e−16；没有在本机独立重跑UNet/autograd或终点latent。上游T条件为真实概念排除OOF，旧ordinary使用过完整T，整个组合不称全链OOF或独立确认。

两个GPU主体123.721/122.188秒，额外CPU峰值约9.414GB、GPUallocated峰值14.872GB，均在原10/20GiB上限内。控制器含等待1493.757秒，不能称为纯GPU诊断耗时。

证据：[实际全32结果](../audit/HIVE_native_path_adjoint_T32_complete_20261029_v1/metadata/completion.json)、[原始协议](../audit/HIVE_native_path_adjoint_T32_complete_20261029_v1/metadata/protocol.json)、[独立投影与事后分解](../audit/HIVE_native_path_adjoint_T32_complete_20261029_v1/independent_local_gradient_analysis.json)、[文件接收](../audit/HIVE_native_path_adjoint_T32_complete_20261029_v1/receiving_completion.json)。完整采样反传已有DRaFT/AlignProp直接近邻，链式法则和投影不冒称原创；本轮不自动拟合或产生SOTA结论。
