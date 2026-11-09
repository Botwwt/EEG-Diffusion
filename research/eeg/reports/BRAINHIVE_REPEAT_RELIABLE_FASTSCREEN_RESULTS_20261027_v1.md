# D08 重复可靠性解码：真实结果与失败定位

**固定实例停止，不晋级400张，不扫rank/ridge。** 原生与候选的完整80条图片在同一400目标图库上，DINO2AFC为0.808208与0.622964，差−0.185244，描述性概念重采样区间[−0.302043,−0.054535]。7/8概念负向，触发事前v2明确损害规则。目标图片DINO余弦差−0.050259；同概念Top1由12/80到9/80。[真实评分](../audit/repeat_reliable_decoder_20261027_v1/actual_score80_summary.json)、[实际闭合及退出](../audit/repeat_reliable_decoder_20261027_v1/actual_scientific_closure.json)

## 方法、数据与复用

本轮复用D01导出的sub-01全部13230条真实四次物理重复、原F缓存、原生400张图片和评价特征。两折按概念隔离，全部统计和拟合只读相反训练折；推理只读当前被试的物理mean4。没有生成器反传、其他被试query EEG、query视觉输入或V/官方拟合。

先固定连续5点均值形成17×50输入，用训练折的试次残差估计噪声协方差，信号代理为mean4协方差减噪声协方差的1/4。噪声矩阵固定0.1收缩，保留128个正可靠性方向，再用ridge0.1解析拟合完整1024维F，固定无仿射LayerNorm作为输出。没有视觉PCA压缩或按query目标缩放。[冻结配方修正版](../configs/repeat_reliable_decoder_20261027_v1/candidate_spec_bindingfix_v2.json)、[真实源代码](../fit_BrainHIVE_repeat_reliable_decoder_20261027_v1.py)

重复性滤波已有[DSS文献先例](https://pmc.ncbi.nlm.nih.gov/articles/PMC2483698/)。本轮改变了信号提取、回归及输出归一化，不是算法原创或机制归因实验。

## 完整日志说明了什么

两折解析拟合主体8.22秒；可靠性矩阵分别有426与424个正方向，固定保留128，所选特征值约0.098–2.21，线性方程相对残差约6×10⁻¹⁶。额外CPU峰值约2.50GiB，GPU allocated约1.03GiB。训练折同概念F检索约34.5–34.6%，held降至22.4–22.9%，低于原生33.6–33.9%。[全部实际拟合日志](../audit/repeat_reliable_decoder_20261027_v1/actual_fit_progress.jsonl)、[真实拟合汇总](../audit/repeat_reliable_decoder_20261027_v1/actual_fit_summary.json)

held原始F余弦却升至约0.639、MSE降至约0.723；输出范数约32，已避免D02的低范数问题。完整13230条CPU诊断进一步显示：候选正确减同概念其余9刺激的余弦间隔仅0.0183–0.0190，原生为0.0448–0.0452；候选与训练折视觉均值的平均余弦约0.956。候选总体均值向量占输出二阶能量约91.5%，原生约0.07%。这是条件分布的描述量，不是单条EEG的生理信号比例，也不能单独证明因果。[全量条件诊断](../audit/repeat_reliable_decoder_20261027_v1/CPU_condition_geometry/summary.json)

这些读数结合真实图片，支持停止当前“重建条件均值就能改善对象”的实现。提高总余弦、保住范数、保留完整视觉维度仍不充分；不能仅凭MSE大幅下降继续扩训。滤波、回归、时间池化和归一化一起变化，尚不能独立归罪于某一项。

## 实际图片与耗时

全部80条candidate/native/target身份、种子及初始噪声字段、图片与像素SHA已再次CPU核对，零不一致。每个概念固定第一张刺激的八行图已实际查看，候选多出现杆、管、工具或桌面，许多对象不符；这些固定展示不替代全80条评分，也不构成感知质量非劣证明。[全量核验](../audit/repeat_reliable_decoder_20261027_v1/CPU_complete_image_inspection/summary.json)、[固定八概念图片](../audit/repeat_reliable_decoder_20261027_v1/CPU_complete_image_inspection/fixed_all_concepts_first_stimulus_page1.jpg)

首次启动因手工录入的trials SHA多一个字符，在CUDA初始化及模型拟合前被输入检查拒绝。保留原失败、原spec和原执行器目录，随后从原始二进制传输回执读取正确SHA，并独立重哈希真实输入；仅更正绑定，不改公式、rank、ridge、seed或门槛。[输入绑定勘误](../audit/repeat_reliable_decoder_20261027_v1/input_binding_erratum_v2.json)、[原失败](../audit/repeat_reliable_decoder_20261027_v1/actual_initial_failure.json)

修正后远端全段113.75秒，成功GPU包装合计112.07秒；计入首次失败的全部包装115.69秒。从首次远端尝试到最终评分闭合实际297.72秒，包含绑定修正和交接；首次实现、审查、部署另计。400阶段没有启动，所有本轮进程已退出、GPU空闲、锁已释放；正式D仍停止。
