# Frozen 全量生成完成，三组输入一致性通过

2026-10-19（America/Los_Angeles），03:10 UTC观察节点。Frozen已完成全部16,600张V图片，覆盖十被试、每被试1,660张、R4，每query一张。图片保留在服务器；没有覆盖RealPaired、Conditional或历史实验。

## 实际完成的证据

- 生成作业135265/监督135264在03:03:33 UTC退出0。生成内部耗时14,618.323秒，GPU包装器计入14,622.891秒（约4.062 GPU小时）；未复用任何旧query输出。
- 每query20步、CFG5、40次UNet样本评估；全量83,000次调用、664,000次样本评估。分辨率512，与另外两组完全相同。
- 首次身份审计核验全部16,600张保存JPEG、原生EEG条件哈希及释放元数据配对，失败0、重复生成图哈希组0。该阶段`score_rows_checked=0`，尚不能替代评分后的目标核验。
- 新增只读CPU核验完整三组，每组16,600行：37个输入字段逐query一致，包括刺激/概念/数组行、同一EEG及原生条件、初始图像噪声、种子、目标SHA、生成器版本和采样成本。
- 三组保存路径集合完全不相交，各组内部路径唯一；三对比较的相同query生成JPEG哈希相同数都为0。字节不同不证明对象不同或图片改善。
- 独立读取原始IP-Adapter权重与两个训练检查点，转换为实际部署BF16后核对四个投影张量哈希。Frozen确实使用原预训练投影，没有适配器训练检查点；RealPaired及Conditional确实加载各自训练后的8,400,896参数投影。
- 三组输入核验用CPU2.293秒，没有读取图像分数，没有选择样本、种子或检查点。

证据：[完整生成凭据](../audit/idea_interface_Frozen_completed_V_generation_v1.json)、[实际生成作业](../audit/IDEA1_Frozen_generate_job.json)、[首次身份审计](../audit/idea_interface_Frozen_V_identity_before_v1.json)、[三组逐行输入及实际权重核验](../audit/idea_interface_Frozen_RealPaired_Conditional_V_matched_inputs_v1.json)、[核验源代码](../verify_completed_idea_three_V_inputs_v1.py)。

Frozen manifest SHA：`7e3353414abea19df99766b91ac6e6d3c9b8b236cdfa85a854698e28d156d717`。

## 正在进行与未完成项

03:10:26 UTC已实际核对特征提取worker141915、GPU监督141914和驱动器135040的完整命令。驱动器已完成T_readiness、generate、identity_before，正在features阶段；当时CapSim处理1,024/16,600。之后仍需全量评分、低层指标、评分后身份审计和三组配对报告。

当时累计作业及保守历史成本32.798 GPU小时，按更紧的已授权占用期限剩约4.838小时。保留既定后续评价资源，不启动新的图像训练或另选配置。继续十分钟监视。

这一步证明完整图片与比较输入已落实，尚无Frozen最终指标，不能宣称它优于另外两组、已证明误差机制增量或达到SOTA。准备好的[三组完整分数核验程序](../verify_idea_complete_three_arm_scores_v1.py)须等待全部后续评价完成后才能执行。
