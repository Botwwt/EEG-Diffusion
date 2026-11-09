# 固定方向R80对照：噪声证据接口修复与实际恢复

2026-11-01 20:01 UTC的实际观察确认：新控制器514576在线，已通过首组完整工程profile并进入图片生成；独立VAR控制器514921在线等待。原成功的完整ordinary训练／评价不重启，两个旧失败控制器及失败文件保留。

## 修复的具体问题

原生R80参考图片清单没有`initial_noise_latents_sha256`和`ancestral_noise_observations`字段。旧入口在原生图片SHA、解码像素和四步调用检查通过后，直接读取这两个不存在的字段而报错。该次GPU包装225.22秒，不是模型效果失败；没有完成机制研究图片或分数。

修正版把两类证据明确分开：图片仍和原生参考图片逐像素比较；噪声证据来自已经完成、具有实际采样观测的同query生成清单。新`noise_reference()`逐项要求query_id、subject、stimulus_id、array_row、image_seed和R80协议相同，再核对四步／280次K和V调用、原生scheduler以及四份FP16噪声记录。如果图片清单本身含有噪声字段，它也必须与来源一致；不向旧清单补造字段，不忽略冲突。

在GPU启动前，真实CPU准入已检查全部2000条原生输入与两组共4000条来源噪声记录：受限F和完整ordinary在每条query上的实际噪声证据一致；6000份来源图片的身份／文件SHA、真实R80条件和十个published affine仍按原规则核验。该阶段没有初始化CUDA或读新控制效果。

## 实际通过的范围

首组`mean_bounded_F`已完成全部2000条条件与CPU64参考的工程核验，最大误差6.7797e−7。十个固定被试query的原生、关闭、零头、来源头和固定方向五条路径共50张工程图完成；原生／关闭／零头共30次像素回放及10次来源头回放通过。主体267.75秒，GPU峰值8,761,631,744字节，额外CPU9,384,599,552字节，均未提高上限。

这些是实际服务器工程回执。当前本地只核对了源文件和回执来源，未将其冒称本地重新跑过全部条件或像素。第二组profile、全部4000研究图片和固定四项比较尚未闭合，因此没有固定方向足够、query依赖修正有效或新语义收益的判定。

本地旧协议15项错误范围拒绝、依赖和退出码检查、31条独立几何计算及128000个合成配对单元继续通过；新增12项噪声身份／内容拒绝，以及一条真实旧原生清单与已审计生成来源的缺字段回归通过。服务器额外完成实际清单准入。VAR新依赖绑定也通过十组服务器CPU检查，真实VAR GPU尚未运行。

## 保留原约束和失败证据

新代码和输出分别保存在`HIVE_training_mean_R80_*_20261101_v2`，旧v1的七份源、失败和部分条件文件不覆盖。旧失败输出与新输出合计仍受原4GiB上限约束；原首次执行的绝对截止2026-11-02 01:28:11 UTC保留，未重新获得六小时。每个GPU作业900秒、GPU20GiB、额外CPU10GiB、启动余量24GiB和运行余量10GiB等约束保留，旧承诺与安全余量照常计入。

数据、两个训练集均值、来源权重、seed1729、强度1、半径0.25、原生四步和固定4000图／四项比较／16指标不变，没有新训练。固定raw方向仍经过依赖当前EEG的切向投影和范数恢复，不能称为删除EEG。

独立VAR新版本只重新绑定修复后的控制队列和保留旧输出的资源记账。四份实际模型／核／概率／数值审计源与v1逐字节一致；全T8×1000类别×680目标、预训练FP32精度、原时间与输出上限均保留。原v1的依赖失败和GPU事件0保留，不把新队列写成旧条件曾经通过。

## 来源

- [实际修复范围与首个工程回执](../audit/HIVE_training_mean_R80_recovery_preparation_20261101_v2/actual_receiving_scope_review.json)
- [均值控制的精确改动](../audit/HIVE_training_mean_R80_recovery_preparation_20261101_v2/exact_source_changes.diff)
- [VAR依赖与记账的精确改动](../audit/VAR_multiscale_native_profile_recovery_preparation_20261101_v2/exact_dependency_and_resource_changes.diff)
- [原始失败及停止范围](../audit/HIVE_mean_R80_schema_failure_20261101_v1/actual_failure_and_scope.json)
- [两条新队列的有限只读观察](../audit/crossdisciplinary_20261028_v1/HIVE_mean_VAR_recovery_readonly_watch_20261101_v2/actual_registration.json)

当前观察会话为93468／本地PID240172，300秒一次、限12小时，只读旧卡。观察结束不触发研究重跑；90452及所有旧观察会话均已终止。当前仍没有第二个独立方法、SOTA或CCF A结论。
