# 三臂 AR 草案 v2：logical16、micro2，以及可检验的解释边界

2026-10-29，America/Los_Angeles。**正式拟合前的新配方，未准入、未自动启动。** [v1](AR_ANCHORED_TRAINING_PROTOCOL_DRAFT_20261029_v1.md)完整保留。单次 B1 前向不证明全量 stochastic B16 可在20GiB内训练；v2改为逻辑batch16、物理microbatch2。实际匹配的全尺寸 GPU profile仍待闭合。

## 新的随机训练配方

每个逻辑更新将16条query固定切成八个micro2，末批14切成七个micro2。每个micro的token平均CE乘以`本micro query数／当前逻辑batch query数`后反向；全部micro完成后才裁剪一次、AdamW更新一次。固定十遍完整13,230条T、每遍827逻辑更新、每臂8,270更新不变；每臂共有66,150次micro前向。EEG/VQ冻结，各臂原有全部VAR参数训练。

在关闭随机层时，CE对query可加；这**不意味着带条件dropout／DropPath的micro2轨迹与单次physical B16逐位等价**。micro分段改变随机调用顺序和CUDA RNG消耗方式。本轮在任何正式AR训练前将其另存为统一配方，三臂都使用同样的query顺序、micro切分和随机调用约定；不在失败后仅给某一臂偷偷改变batch。

核心：[v2代码](../AR_anchored_training_core_DRAFT_20261029_v2.py)。接口为`run_updates(..., batch=16, micro_batch=2, maximum_updates=..., device=...)`。这里的`maximum_updates`只控制已完成逻辑optimizerstep之间的作业分段；它与改变microbatch这一随机配方修订不同。

累计中标记`logical_update_in_progress=True`，拒绝snapshot、resume或把partial gradients当成完整状态。中途失败必须回到上一个完整optimizerstep，并保留、计费未提交工作。正式producer还须将checkpoint与同attempt的不可覆盖日志前缀绑定；恢复后重算的未提交update写入新attempt目录，不能覆盖原记录。原A/B两槽有界存储策略沿用，旧结果不删。

## 已完成的33项CPU检查

[CPU证据](../audit/AR_anchored_training_DRAFT_20261029_v2/CPU_contract_actual_v1.json)使用作者真实VAR/VQ的两block随机模型，不使用真实EEG或图片，不含GPU／正式模型拟合。

- logical16、16、14确实调用8、8、7个micro2，恰好三次clip／Adam，46条fixture每条一次；
- 关闭随机层后，native及paired两个模型的full16/full14与加权micro2损失最大差约3.28×10⁻⁷、梯度最大绝对差约2.24×10⁻⁸、相对L2误差约2.44×10⁻⁷；
- 训练模式下三臂query顺序和CPU RNG消耗一致；中途epoch完整step的磁盘checkpoint恢复后，模型／Adam／loss／query和donor顺序逐位一致；
- micro累计中断不产生partial Adam／cursor更新，snapshot明确拒绝；换microbatch恢复也拒绝；
- donor权限、冻结VQ、短末批、两payload峰值、checkpoint中断保留current、未知文件拒绝及不可覆盖日志检查继续通过。

这仅验证代码契约。真实CUDA内核、真实预训练VAR、完整680-token/4096-code、matched三臂micro2资源与随机流仍需实际profile。原公开权重strict加载和全部T图像code bank也须绑定后才能训练。

## 三个必须保留的解释边界

第一，`H=c`或关闭localroute精确回退的是**同一组当前VAR权重**下的native路径。paired臂训练全部VAR参数后，关闭localroute不会恢复另训native-global检查点或原公开VAR图片。因此native-global仍须独立匹配训练；post-fit关闭仅是同模型输入干预。

第二，锚点处为零只是一条结构性质，不等于统计上的`E[R|c]=0`，也不证明局部信息与c正交、存在额外条件信息、测量不确定性已校准或得到Doob后验。

第三，同概念donor会同时破坏自身c与local统计的兼容性。paired优于donor支持当前控制下的刺激配对作用，却不能单独排除额外函数容量或可由c预测的局部矩。没有经过验证的`p(H|c)`采样器，不能称为精确条件随机化／knockoff／因果检验。若真实图片有支持，之后才考虑另行验证这些更强主张。

正式图片规则仍需在训练和任何新Dcheck效果读取前绑定：三臂、同400目标图库、80→400、单query单图、正常推理EEG-only、不得读中间400效果。拟议主门槛是paired相对native及donor均至少+0.01 DINO且描述区间下界大于0，同概念Top1均值不损害；当前没有以未绑定草案给结果放行，也没有成功或SOTA结论。
