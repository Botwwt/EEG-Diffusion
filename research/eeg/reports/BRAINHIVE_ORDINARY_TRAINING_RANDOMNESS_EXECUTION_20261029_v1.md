# ordinary训练随机性复现：状态恢复、反传重放差异与完整T队列

2026-10-29。原ordinary完整R80的正结果继续保留；当前缺少训练随机性验证，不能把一个训练seed的配对区间当成训练稳定性。本轮预先固定新增seed2027/3407，原seed1729检查点直接复用，不覆盖、不从新seed挑最好者。它是已有方法的复现工作，不是新创新、已完成效果或SOTA。

原ordinary为525568参数、半径0.25、原强度1，SHA `a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03`。所有GPU只用旧卡，第二卡继续留给用户；原正式D、已停止AR及上一轮T80目标几何均不恢复。

## 工程检查发现的真实边界

v1工程采用全部10被试真实OOF输入、原生A/P/UNet、原BF16 autocast和4个科学更新，比较不中断4步与2步保存后继续2步。未来模型/Adam未逐位相同，原严格检查失败，失败保留，未自动准入完整拟合。

v2另增加一个完全不中断的4步重放控制，实际12次临时更新，结果为：

| 比较 | 模型/Adam | CPU/CUDA/噪声RNG、顺序、覆盖 |
|---|---|---|
| 保存时状态→立即反序列化恢复 | 全部逐位相同 | 全部逐位相同 |
| 不中断4步→2步保存后继续2步 | 不逐位相同 | 逐位相同 |
| 不中断4步→重新从头执行相同4步 | 不逐位相同 | 逐位相同 |

三次首次前向loss都为0.24102334678173065，首次direction梯度范数分别0.005329798/0.005336979/0.005330822，随后轨迹开始不同。4步后direction.weight最大绝对差约5.49e−4（保存恢复）及5.24e−4（不中断重放），相对L2差约0.1925/0.1913。由于方向层从零开始，早期相对差不能直接解释成最终图片误差或收益比例。

这排除了“只有checkpoint恢复才造成差异”的解释，与当前GPU混合精度反传路径的非逐位确定性一致；没有逐算子定位，不能指定某个cuDNN或注意力kernel为唯一原因。v2原未来逐位检查仍为失败，**没有修改阈值把它标为通过**。该结果也不推翻原R80实际图片或历史CPU状态核验。

历史“精确恢复”应区分两件事：模型/优化器/RNG/覆盖状态的精确还原，以及后续浮点优化轨迹逐位相同。前者本轮得到实际支持，后者没有。不能将前者的回执扩大为后者。

v2主体70.589秒，显存allocated峰值10722675712字节，额外CPU9412280320字节，原20GiB/10GiB上限未变。v1/v2共25份元数据和源文件按SHA接收；临时checkpoint仅保留服务器用于工程审核，未下载为本机模型、未进入正式复现。证据：[v1保留失败](../audit/HIVE_ordinary_seed_repro_engineering_20261029_v1/actual/failure.json)、[v2完整数值诊断](../audit/HIVE_ordinary_seed_repro_engineering_20261029_v2/actual/resume_numerical_diagnostic.json)、[v2保留严格失败](../audit/HIVE_ordinary_seed_repro_engineering_20261029_v2/actual/failure.json)。

## 实际采用的完整复现方案

因此新研究采用每seed一次不中断的完整拟合，不调用分段恢复路径，也不声称未来轨迹逐位可复现。它测量相同方法的训练随机性，包含当前GPU反传的非确定性，不能唯一归因于名义seed。原检查失败不变；新方案通过消除分段执行来避免将未知恢复影响混入主研究。

| 项目 | 固定设置 |
|---|---|
| 新seed | 2027、3407，全部报告 |
| 数据 | T1323概念、13230视觉刺激、10被试132300条真实OOF条件 |
| 预算 | 每seed3遍、396900次query暴露、24807更新；batch16，末批12 |
| 结构 | 原525568参数ordinary，不加新头、不改半径0.25 |
| 损失 | 原生uniform t0..999 epsilon MSE＋0.1原生token角误差 |
| 优化器 | AdamW lr1e−4、wd0.01、betas(0.9,0.999)、eps1e−8、clip1 |
| 精度 | 原BF16 autocast，冻结宿主FP16权重数值保持；head FP32 |
| 模型选择 | 固定最后24807更新，不读图片选中间权重 |
| 校准 | 固定Dfit8200 token角误差网格{0,.25,.5,.75,1}，同原流程；不能强行设成1 |
| 后续评价 | 每新seed完整R80 2000条、同原生四步/512/CFG0/IP1、同query噪声、单图和完整指标；目前未启动 |

数据、loss、初始化、RNG和Adam科学核心与已运行工程代码的函数AST完全相同；仅改为一次完整循环和对应有限任务上限。加载时完成原生权重核验和固定prompt后，释放不参与cached-latent训练的VAE/文本编码器；没有改变实际UNet/A/P训练计算或舍入冻结权重。不能因此声称新拟合一定复现原seed1729的逐位轨迹。

每个GPU任务仍有限，上限50400秒（14小时），两个固定任务顺序运行，无自动重试或恢复；控制器上限102000秒。原ordinary实际训练本体27897.978秒（约7小时45分），只是历史参照，不能当成本轮实测或保证。两seed完整复现不是分钟级快筛。磁盘为全部未来模型、图片、特征及元数据预留20GiB，启动时实际空闲约219GB，没有删旧产物或购买资源。

全部输入、latent、原实现SHA、临时状态CPU解码、有效资源和新的T-only范围已在实际启动前复核。旧官方1654概念最终拟合入口没有解锁，也没有把已曝光的官方200刺激伪装为独立H。新增head仍只拟合原T1323，R80只是透明、已曝光的匹配benchmark。

实现：[不中断训练](../train_HIVE_ordinary_seed_repro_uninterrupted_GPU_20261029_v3.py)、[有限顺序队列](../queue_HIVE_ordinary_seed_repro_uninterrupted_CPU_20261029_v3.py)、[实际范围审核](../audit/HIVE_ordinary_seed_repro_full_registration_20261029_v3/protocol/actual_scope_review.json)、[完整协议](../audit/HIVE_ordinary_seed_repro_full_registration_20261029_v3/protocol/protocol.json)。

## 实际运行及首断点

控制器323229/start93100313于22:15:13 UTC实际启动。seed2027先运行，seed3407固定等待；截至22:21:33 UTC控制器及GPU包装均存活，公开256/24807更新，无错误。GPU训练PID323235当时约11552MiB显存；不能将这一旧观察当作以后持续状态。

首个128更新／2048次query暴露断点已由独立CPU进程从单一文件描述符读取，SHA、inode和解码来自同一次原子提交，避免下一次rename混入来源。三遍预定顺序SHA、实际覆盖前缀、全部四组Adam步数/有限性/二阶矩非负、模型有限性及随机状态检查通过；没有停止、恢复或重启训练。

证据：[实际首断点CPU审核](../audit/HIVE_ordinary_seed_repro_full_registration_20261029_v3/actual/seed2027/part00/independent_first_checkpoint_CPU_audit.json)、[已核验的7份注册文件](../audit/HIVE_ordinary_seed_repro_full_registration_20261029_v3/receiving_completion.json)。这是运行和状态证据，不是新seed图片收益。

后续须继续同一实际进程。有限本地观察器结束不代表训练结束，不据此重启。两个固定末轮和完整校准闭合后，仍需独立模型/Adam/三遍覆盖审核，再绑定新的完整图片执行器。现有R80脚本多处硬编码seed1729/strength1，不能直接改一个检查点路径就复用；新入口必须尊重新seed的Dfit实际强度、对应来源链和完整指标，不挑表现最好的seed。

## 同时完成的真实图像输入准备

已从原training_images.zip读取固定T8对应的原始JPEG，逐条核对image_member、stimulus_id、array_row、文件SHA、ZIP CRC、RGB解码及像素SHA；完整13230行T索引顺序也与当前T映射一致。没有读取Dcheck/V或官方图像来拟合。

该准备使后续可以直接研究真实图像语义风险，而不是只保持G(F)的语义；当前未新编码CLIP目标或拟合该新损失。OpenAI CLIP默认224 bicubic预处理与官方评价器的512 bilinear前处理并不相同，未来训练surrogate必须单独声明，不能拿同一critic的分数充当独立语义确认。[实际T8输入核验](../audit/HIVE_real_T_semantic_image_inputs_CPU_20261029_v1/completion.json)。

本轮仍无新方法收益、训练稳定性结论、独立确认或CCF A/SOTA结论；目标继续。

## 15:58 实际运行与完整评价队列更新

2026-10-29 15:58:45 America/Los_Angeles（22:58:45 UTC）只读核对：训练控制器323229/start93100313仍在线，seed2027为2240/24807更新，seed3407等待；无失败。第二卡未接触。

先前“新seed评价入口仍待实现”现已由实际实现更新：独立评价控制器324335/start93263426已于22:42:24 UTC启动，只以CPU等待同一组两次固定完整拟合成功退出。随后自动审核模型/Adam/三遍覆盖/RNG及Dfit校准，绑定固定末轮SHA与实际所选强度，各生成完整2000张R80并完成16项指标，再联合原1729及新2027/3407三次训练报告，绝不挑最好seed。新图计划4000张，目前新增评价GPU任务0、无新seed图片分数。

18项CPU检查涵盖五种实际球面强度、错误seed/部分模型/无效校准拒绝、完整6000条三轴描述性统计和仅本队列自有子进程清理。实际1536更新的部分head与全部2000条条件另经五强度独立CPU算术复核，最大绝对误差小于9.62e−7；这不准入部分head出图，也不选择强度。22份注册源码和元数据已逐文件SHA复核，最终新seed权重的真实GPU profile及图片评价尚未运行。

三次训练联合统计只提供有限训练运行的描述性稳定性证据，包含已记录的kernel随机性，仍不是独立H、SOTA或创新充分性证明。原ordinary、旧正式执行门槛及所有保留失败不变。

证据：[完整注册及CPU检查](../audit/HIVE_seed_R80_registration_20261029_v1/receiving_completion.json)、[最新精确在线状态及后续入口](../audit/AR_active_execution_handoff_20261029_v26.json)。
