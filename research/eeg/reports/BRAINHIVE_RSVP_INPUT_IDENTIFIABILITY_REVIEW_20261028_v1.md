# RSVP相邻响应：先核对输入来源与可识别性

> 2026-10-28 12:48更新：后续D33全部T的四lag一致性检验及独立CPU复核已真实闭合。50点独立段余弦中位数0.915627，但严格恢复边数0，原门槛失败；不能改判事件顺序已恢复。见[实际结果、公式及重采样反例](BRAINHIVE_TEMPORAL_OVERLAP_RESULTS_20261028_v1.md)。

以下保留D32停止后、D33启动前的来源检查快照：没有新模型、EEG数值前向或图片，没有改动ordinary、X01或原预处理；当时尚未注册或执行D33。

[Gifford等人的THINGS-EEG2原论文](https://pmc.ncbi.nlm.nih.gov/articles/PMC9771828/#sec4.8)明确报告200ms的刺激起始间隔，并讨论前后图像响应的重叠；作者同时在整个epoch中获得了可预测的目标响应。因此，短SOA提示需要检查测量混合，不能据此把某个时间点之后全部判成无用或把既有方法失败归因于邻刺激。

## 当前真实缓存究竟保留了什么

旧卡原始训练缓存SHA `c54cf2e8a4f0c122d79bb7d98ddf04698f992d292f3e26ae59bed97230944f63`，4168083005字节。[只读头部／目录实查](../audit/crossdisciplinary_20261028_v1/RSVP_training_metadata_actual_20261028T191120919248Z.json)确认它虽以`.npy`结尾，实际为pickle流。未为了元数据解码整块EEG。

随后[受限符号解析器](../inspect_BrainHIVE_training_pickle_metadata_CPU_20261028_v1.py)通过本地及实际服务器的协议3／4／5小型NumPy对象和不执行REDUCE的四项检查。它只执行自有栈／字典操作，将pickle选定的GLOBAL、REDUCE、BUILD保持为符号记录，不调用这些构造器；遇到不支持的opcode就拒绝。

[真实结果](../audit/crossdisciplinary_20261028_v1/RSVP_symbolic_pickle_metadata_actual_v1.json)：226个opcode、仅读取3005个非EEG元数据字节，以seek跳过偏移85处的4168080000字节载荷，耗时5.503641秒，实际退出码0。顶层只有三个键：

| 字段 | 实际内容 |
|---|---|
| `preprocessed_eeg_data` | 符号数组16540×4×63×250，little-endian FP32，未解码数值 |
| `ch_names` | 完整63通道名字；当前后17通道索引46…62与原输入契约相符 |
| `times` | 300个FP64点，−0.2至0.996秒，步长约0.004秒 |

原发布[ATMS加载器](../reproduction_sources/ATM_main_49cb3a20/Generation/eegdatasets_leaveone.py)对`times`取`[50:]`、保持EEG数组不再截取，按此约定得到250个0至0.996秒时间点。当前缓存本身没有记录原事件到该裁剪结果的完整回放，故需区分发布加载约定与独立事件复核，不能直接将原300点时间向量绑定到250点数据。

缓存没有事件时间戳、呈现顺序或逐重复session字段；原物理R4契约的`independent_session_metadata_available:false`由当前字段实查进一步具体化。数组第一维按图像条件组织，不是已经核验的连续采集顺序。没有读取官方test文件或对D／V数值作分析，密码未写入任何源或回执。

## 为什么不能直接声称已得到去卷积解

将单条epoch抽象为y=u+SΔv+ε，u为目标响应，v为邻刺激响应，SΔ为固定时间平移。对任意允许的g，有：

$$u'=u+S_\Delta g,\quad v'=v-g,
\qquad u'+S_\Delta v'=u+S_\Delta v.$$

因此只给出混合epoch和一个SOA，一般不能唯一恢复两个未知响应。减少重复平均的噪声也不等于解决该分解。经典[Unfold方法](https://doi.org/10.7717/peerj.7838)使用事件时序与设计矩阵进行重叠校正；不能把缺少这些输入的任意滤波称为该方法的真实复现或新颖贡献。

下一项可检验问题是：只用既有、允许的T物理R4，平移后的窗口是否保留同一连续记录的可识别片段，从而恢复部分采集邻接关系？这项检查尚未实施，不能把任意最近邻窗口当成真实事件链；还须排除重复、独立归一化、重采样边界和偶然相似，保留错误lag对照。即使恢复部分T邻接，也不能自动外推到D／V或用相邻query的图像／标签作为正常推理输入。

本轮没有准入新预处理、截短时窗、去卷积器或后续训练。来源与可识别性是下一方向的实际约束，原D32停止结果、ordinary及正在运行的跨框架配方均不变。
