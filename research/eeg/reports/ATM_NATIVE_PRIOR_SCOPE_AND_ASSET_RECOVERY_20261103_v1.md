# ATM原生扩散先验链与ordinary迁移范围

已恢复并逐SHA核验十个作者扩散先验及十份配套测试特征，共395,546,430字节；十模型的全部2000条条件和2000条缺失条件CPU前向通过。这补齐了继续核查原生解码链所需的小型公开资产，不是新增图片收益、完整EEG复现或第二创新。

此前X01的完整400图验证的是ATMS→h94 IP→标准SDXL直接分支。其DINO差+0.002437、区间跨零，仍保留为该实例的结果；它不能代替包含学习到的扩散先验的作者语义链验证。

## 原生链路和已执行范围

固定作者代码提交49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d。作者README将语义、高低层混合和字幕路径分开；语义入口为Generation_metrics_sub8.ipynb。此入口使用DiffusionPriorUNet，再接h94 IP与四步SDXL-Turbo。[作者说明](https://raw.githubusercontent.com/ncclab-sustech/EEG_Image_decode/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/README.md)

本地和旧卡代码SHA匹配历史导出。实际核对的语义notebook使用prior 50步/CFG5，图像分支4步/CFG0/IP1；同仓库低层混合notebook还包含10步/CFG2的例子，不能把两者拼成同一执行配置。prior默认DDPMScheduler开启clip_sample。作者示例每query十图，本研究若进入匹配图片比较，仍须固定每query单图与相同噪声权限。

| 项目 | 本次实际检查 | 解释范围 |
|---|---|---|
| 十个作者prior | 各9,675,648参数，全部状态张量严格加载 | 公开参数与真实类兼容；无优化器记录，训练更新数未知 |
| 十份配套测试特征 | 每份200×1024、FP32、全部有限 | 公共特征到解码器的输入契约；未重放其上游编码器 |
| CPU条件响应 | 全部2000条条件及2000条缺失条件、固定t=500、同噪声前向 | 条件分支能运行；不表示图片收益或正确EEG机制已证实 |
| 调度器 | DDPM、1000训练时间点、clip_sample=True、设置50步通过 | 尚未执行完整50步采样或新图片 |
| prior输出层 | Linear，末端没有学习到的LayerNorm affine | 不能套用HIVE末端gamma/beta或直接搬已学ordinary权重 |

十份公开EEG特征的平均范数为5.9071—9.1989。早期记录中的当前自训ATMS R4输出均值约14.07，二者样本与训练权限不同，这不是能力排名；它足以否定“同为1024维就可无检查接线”的做法。[作者prior实现](https://raw.githubusercontent.com/ncclab-sustech/EEG_Image_decode/49cb3a20a0c1fd35fd2955f78d0e4d0f2939c70d/Generation/diffusion_prior.py)

作者当前develop分支另有统一的高层、低层和混合benchmark入口。本次只读取官方分支信息，固定到efb6b1d5d7dd14bea44ee5a952b72a951fa6da54，没有切换运行源或执行其训练脚本。它与旧notebook须分开报告。[官方当前入口说明](https://raw.githubusercontent.com/ncclab-sustech/EEG_Image_decode/efb6b1d5d7dd14bea44ee5a952b72a951fa6da54/README.md)

## 公开特征和训练权限

资产包只含十模型与十份测试特征，不含EEG编码器权重。测试特征虽然每被试200行，尚未独立核验上游EEG事件、行序和物理重复数；因此不能将本次CPU检查写成“十被试完整R80端到端复现”。

固定数据提交dbe34bb2407164f70883c661c76664f0d596e522的官方元数据共71项，另列十份训练EEG特征，每份约271MB及视觉训练特征。它们本轮没有下载或用于拟合；也不能因文件名含train就将其当作本项目概念OOF条件。需要先核验训练样本顺序、单trial/均值规则、允许概念和上游模型权限。

若使用自训ATMS建立公平的同数据父系统，应将编码器和prior均纳入概念排除关系。对折r，编码器E_r与prior D_r都只能拟合T去掉该折概念；该折组件输入为h_i=D_r(E_r(e_i),ξ_i)。完整T父系统另训练E_full/D_full供开发集推理。prior可在自己的允许训练概念上使用其编码器输出，held概念不得进入其拟合。该安排产生真实的组件折外输入，但仍需报告折外到完整模型的分布差异，不能靠改标签或复用公开测试特征制造OOF。

这是待完成的原生父系统设计，不是本轮已训练的新模型。公开作者prior可以先验证发布的解码支路；未经匹配，不能接到自训ATMS后称原始复现，也不能拿它在完整训练概念上产生的过于乐观输出替代上述折外训练输入。

## ordinary在prior之后的几何定义

HIVE原ordinary使用真实编码器输出affine；ATM prior末端没有这一affine。若将同一受限方向原则放在prior后、原生IP投影前，需要另建权重，明确原生坐标，而不是复制HIVE检查点。

一个保留原输入均值和范数的定义是令m=mean(h)、u=h−m1，将新方向先去均值再去掉u分量，得到d同时垂直于1和u。随后：

$$\Delta=\rho\|u\|\frac{d}{\sqrt{\|u\|^2+\|d\|^2}},\qquad s=\frac{\|u\|}{\|u+\Delta\|},\qquad h'=h+(s-1)u+s\Delta.$$

于是mean(h')=mean(h)、||h'||=||h||；关闭修正或零方向时回到原输入。u为零时须明确定义回退。这里的等价性是受限方向修正原则及零头可回放，不是输出坐标、学习权重或训练损失字节相同。公式是经典球面约束的工程迁移定义，不作为新的第二创新。

后续如验证该定义，原生IP投影替代HIVE的A/P复合接口，视觉监督为该宿主的CLIP-H目标；若声称复现完整ordinary，还须实际实现相同角色的生成噪声目标及匹配F-only/公共方向控制。不能省掉生成目标后仍称完整等价。当前未实施此新头、GPU profile、拟合或图片评价。

## 实际恢复与资源

2026-11-03 21:21 UTC实查，十prior和十特征在旧路径均缺失，原registration/completion仍在。直连下载的有限尝试69.719秒，20项均为网络错误、实际验证字节0；失败记录保留。随后用本机可用网络经内存/SFTP转传，在旧卡持久盘另存原SHA资产；没有在C/D落地大权重，没有覆盖旧登记，也未购买资源或接触第二卡。

内存转传321.25秒；随后全部4000条样本CPU前向检查7.695秒，实际为十模型各条件/缺失条件两次、共20次批前向，额外CPU峰值约0.487GiB，CUDA未初始化。设计、源核查、排错与整理另计。原始字节395,546,430，CPU回执写入前allocated为395,620,352字节；512MiB上限的未使用未来额度在完整资产/CPU核验后结束，真实资产保留，既有50GiB规划底线、8GiB缓冲和其他承诺不变。21:38:28 UTC观察GPU计算列表为空；不是后续占用保证。

本轮推进了两个明确判断：条件几何改变量不能单独预测本轮图片收益；此前跨框架实验尚未覆盖作者原生prior链。下一步应先核验该原生父链及其折外输入，保留HIVE ordinary既有收益与全部负结果，而不是依据本轮CPU成功宣布新方法有效。

来源：[恢复后的逐文件SHA](../audit/crossdisciplinary_20261028_v1/native_ATM_assets_RAM_recovery_actual_20261103_v2.json)、[首次网络失败](../audit/crossdisciplinary_20261028_v1/RAMroutes_old_goal124_native_ATM_pinned_asset_recovery_20261103T212847826862Z.json)、[全部CPU执行原始回执](../audit/crossdisciplinary_20261028_v1/RAMroutes_old_goal124_native_ATM_all10_coordinate_CPU_20261103T213813121776Z.json)、[完整T400条件/图片诊断](HIVE_T400_CONDITION_IMAGE_SEMANTIC_RESULTS_20261103_v1.md)。
