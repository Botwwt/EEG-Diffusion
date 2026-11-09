# X01：另一条开源ATMS／标准SDXL链路的等价ordinary

> **X01完整400实际闭合：2026-10-28 22:24:30，America/Los_Angeles**：全十被试OOF-T三遍、24807更新完成后，两组800张固定图片和评价闭合。ATMS原生/等价ordinary DINO为0.755301/0.757738，差+0.002437、描述区间[-0.016899,+0.024499]；同概念Top1由19.25%降至16.50%。未达原门槛，STOP，不扩V或事后调强度。原特征下载后另行本地4000指标和1000次概念bootstrap复算通过；这不是独立视觉编码器重跑。旧卡本轮GPU已退出，后续公开权重准备属于CPU/网络工作；22:27新卡D40第三臂仍在训练，前两臂完成。HIVE旧ordinary保留，不能据X01声称通用收益。见[完整X01结果与证据](ATMS_STANDARD_SDXL_ORDINARY_FINAL_RESULTS_20261028_v1.md)。

> **时间勘误（22:21）**：原控制器失败UTC为2026-10-29 02:41:29，对应America/Los_Angeles为10月28日19:41:29；此前写为20:41:29是时区换算错误。仅更正显示时间，原日志、40图、模型及接续记录不变。

## 21:33 实际更新：完整训练闭合，修复日志冲突后80图阶段闭合并扩400

原训练完整24,807更新、396,900 query暴露及三次132,300唯一覆盖均闭合，程序记录累计training_seconds=29,780.62秒；最后分段330.26秒不能当成整轮耗时。Dfit8200固定网格选强度0.75，最终模型SHA f13a6854…ef8fd03，校准模型a56797b2…fdb501。旧HIVE ordinary仍为原检查点和强度1。

19:41:29原控制器在第一段40张图片后因`progress() got multiple values for argument 'stage'`退出，原图片和失败完整保留。21:17:45新有限控制器264250只接续图片阶段，原六个科学源、训练目标、模型、校准、400条件和噪声不变。修复仅将日志元数据stage改为image_stage；真实Linux两种事件回归通过。没有重新训练、重校准或重出原40图。

80条两组160图及CPU配对审计、DINO已闭合：native/ordinary的DINO为0.803164/0.802882，差−0.000281955，描述区间[−0.0318084,+0.0261913]；同概念Top1为25.00%/16.25%，差−8.75个百分点，描述区间全负。目标余弦亦下降。当前没有跨框架收益；原80规则只按DINO明确损害提前停止，因此按原规则扩400，不因读到Top1后临时改门槛。400终态尚待。

这是ATMS→h94 IP→标准SDXL-base直接分支的等价ordinary，不是完整作者ATMS→diffusion prior→Turbo重做。20步CFG5与HIVE四步CFG0不直接排名。单被试开发面板、单训练seed不能作独立迁移确认。

[实际接续启动](../audit/goal_iteration_actual_20261028_v1/received_old_20261029T043355.845861_0000/repository/audit/cross_framework_ordinary_20261028_v1/actual_image_continuation_launch_v2.json)、[真实Linux日志回归](../audit/goal_iteration_actual_20261028_v1/received_old_20261029T043355.845861_0000/repository/audit/cross_framework_ordinary_20261028_v1/image_continuation_CPU_check_v2.json)、[此次完整原始回执](../audit/goal_iteration_actual_20261028_v1/received_old_20261029T043355.845861_0000/manifest.json)。以下快照保留。

> **2026-10-28 19:11更新**：19:05:31实际复查：原控制器240115在线、GPU260206运行，23328／24807更新（94.0%）、373240／396900暴露；六个运行源码和旧HIVE ordinary SHA均未变。仍未取得跨框架图片增益。新增D40部署在第二卡并等待D39完整生命周期，不抢占本卡。

> **2026-10-28 18:14:14 America/Los_Angeles 实际更新**：原控制器240115及GPU258193在线，**20992／24807更新（84.6%）、335864 query暴露**，两遍T完整。最新20480／327672断点CPU审计通过；六个执行源及HIVE旧ordinary不变。原训练→Dfit校准→原生／新ordinary配对图片自动队列继续，尚无跨框架效能结果。[实际在线与断点](../audit/crossdisciplinary_20261028_v1/bridge_actual_D39_bound_documentation_X01_live_20261029T011412712263Z.json)。

> **2026-10-28 17:46:05 America/Los_Angeles 实际更新**：原控制器240115和GPU257401在线，**19680／24807更新（79.3%）、314872 query暴露、两遍完整T已完成**。最新19456／311288断点CPU核验通过，SHA为`f48c6f5f…f536042`；六个运行源及HIVE旧ordinary未变。原训练→Dfit8200校准→原生ATMS／新ordinary配对80→400图片队列继续，尚无跨框架图片效果结论。[本次在线原始回执](../audit/crossdisciplinary_20261028_v1/bridge_actual_D39_documentation_X01_live_20261029T004603718190Z.json)。

旧卡上的等价ordinary已实际进入正式训练，控制器PID **240115**；完整T缓存、真实GPU反向传播和六张原生／零头／关闭图片回放均已通过。队列于2026-10-28 09:20:26 America/Los_Angeles启动，后文保留各阶段的日期和实际进度。训练、校准及配对图片评价尚未全部闭合，当前没有跨框架收益结论。

用户本次要求为“其中一张卡在另一个开源框架上实现一套等价的ordinary试试看，看看效果还有没有”。本试验占用原有旧卡；第二张卡上的D30已单独闭合并停止当前实例。两项研究的模型、进度和结果分列。原BrainHIVE ordinary检查点及其强度不变。

## 迁移到哪条链路

使用已发布ATMS编码器与公开原生h94 IP-Adapter，宿主为标准SDXL-base：

```text
原始63电极物理R4均值 → 冻结ATMS → 原生1024维CLIP-H条件
                                      ↓ 新训练ordinary
                             原生P → 原生IP → 标准SDXL-base → 图片
```

这条链路与BrainHIVE的17电极BrainMLP／多模态F／A／学习先验／Turbo不同。这里没有BrainHIVE的F或A，训练教师是原始pooled CLIP-H，生成器为标准SDXL-base，固定512像素、20步EulerDiscrete、CFG5、IP scale1、每query一张。普通头增加一次小型前向，推理UNet调用次数不变。

这是发布ATMS编码器的直接条件分支，**不是作者“ATMS→被试扩散prior→Turbo”完整生成分支的复现**。固定来源、既有基线和这种区分见[跨框架只读审查](../audit/crossdisciplinary_20261028_v1/cross_framework_ordinary_review.json)。现有完整16600条ATMS基线DINO2AFC为0.758269，但其1660图库不能代替本轮新400图库对照，更不能与BrainHIVE400图库分数直接排名。

## ordinary的等价范围

训练新的525568参数1024→256 GELU→1024头，方向输出零初始化，不加载BrainHIVE已学ordinary权重。使用每个真实ATMS角色输出LayerNorm的gamma、beta：

$$u=(c-\beta_{s,r})/\gamma_{s,r},\qquad
c'=c+\gamma_{s,r}\big[(s_0-1)u+s_0\Delta\big].$$

方向去均值、去径向分量，\(\Delta\)和\(s_0\)保持原ordinary的半径0.25及pre-affine范数约束，原角度预算为\(\arctan(0.25)\)。半径约束不保证原始条件、P响应或图片分布不变。零头或显式关闭时应精确回到这条ATMS原生路径。

训练目标同样是真实冻结生成器的噪声预测与原生token约束，但所有坐标均换成当前宿主：

$$L=\operatorname{mean}\|\epsilon_\theta(z_t,t,P(c'))-\epsilon\|^2
+0.1\operatorname{mean}\big[1-\cos(P(c'),P(F_H(I)))\big].$$

\(F_H\)是原始pooled CLIP-H教师，不是BrainHIVE融合F；没有A。使用实际标准SDXL VAE的FP32 posterior mode与原生scaling factor，DDPM前向噪声、均匀t∈0…999。冻结P／IP／UNet，梯度穿过实际计算图到新头。训练只使用正条件噪声预测；正常CFG5推理的负IP条件保持零。不能把仅token回归叫作这套等价ordinary。

## 完整数据和固定预算

复用D15重新导出的132300条真实十被试概念OOF-T条件、30个原ATMS模型及其真实affine。每个T条件来自排除其概念的角色；full-T模型只在后续Dfit／Dcheck推理使用。现成编码器的历史V选择与既有先验曝光仍保留，不把本轮开发读数称独立确认。

| 项目 | 固定设定 |
|---|---|
| 原始EEG权限 | 63×250，物理平均4重复；不增加测试重复 |
| 新训练 | 132300 query×3完整遍，共396900暴露 |
| 更新 | batch16，每遍末批12，共24807 |
| 优化器 | AdamW，lr1e-4，wd0.01，默认betas／eps，clip1 |
| seed／选择 | 1729，固定最后一次；不选中间epoch |
| 新参数 | 525568，只有新ordinary可训练 |
| 校准 | 全8200条Dfit固定{0,.25,.5,.75,1}，token角误差最小、相同取小；强度在半径内部 |
| 单段 | 最多512更新、1800秒；每32更新及段末原子断点 |
| 恢复 | 模型、AdamW、CPU／CUDA／独立噪声RNG、顺序、覆盖与提交日志前缀 |

源入口：[训练器](../train_ATMS_standard_SDXL_ordinary_20261028_v1.py)、[完整队列](../run_ATMS_standard_ordinary_pipeline_CPU_20261028_v1.py)、[用户范围](../configs/cross_framework_ordinary_20261028_v1/user_scope.json)。每段结束都在CPU检查真实断点；失败保留，不因查询连接中断而重启训练。

## 当前实际缺项及队列顺序

服务器实查确认ATMS导出、教师与30个检查点存在；标准SDXL和h94公开权重缓存缺失。恢复过程PID239208已于08:49:35启动，只下载锁定revision并核对原SHA，不读EEG、不训练。09:16左右UNet已完成483／1225个4MiB块，尚非整套权重闭合。

另一项实查发现旧标准T latent完成清单仍在，但52个分块全部缺失：[实际文件检查](../audit/crossdisciplinary_20261028_v1/X01_preflight_actual_v1.json)。队列会另建新版本bank，核对全部13230张原始T图SHA，保持512 RGB bicubic、FP32 VAE mode、batch4／分块256；全部分片重开及两张T图CPU VAE回放后才准入训练。不覆盖旧完成清单或链接，不冒称新分块与旧文件逐字节相同。

实际已挂队列的顺序为：

1. 等待当前有限公开权重恢复，并验证全部来源SHA。
2. 重建并核验完整标准VAE T latent新bank。
3. 锁定训练及图片方案；真实生成器梯度、零头／关闭噪声预测检查。
4. 全20个affine组条件／P token核验，以及两个T query的原生／零头／关闭共6张CFG5工程图片像素回放。
5. 完整24807更新，逐段CPU审计断点，然后仅用Dfit校准。
6. 预先固定全部400条两臂条件、身份与CUDA噪声，再执行真实配对80→400图片及评分。

队列PID与SHA回执见[实际启动](../audit/crossdisciplinary_20261028_v1/X01_launch_actual_v1.json)。工程图、缓存恢复与训练成功均不是科学图片收益。

## 图片效果如何判定

两臂为新生成的`native_ATMS`和`ordinary_ATMS`。同一sub-01物理R4、固定D09 Dcheck400／40概念图库、相同query噪声、同样20步CFG5。只能复用原始目标图的DINO特征，**不使用BrainHIVE生成图充当ATMS原生对照**。

80条为首8个完整概念，两臂160张闭合、身份与像素审计后才评分；DINO差≤−0.02且描述性95%上界<0时停止，其余扩至400。扩展保留原80图／特征，只补640张，不读取中间240条效果。400要求DINO增益至少+0.01、描述区间下界>0、目标余弦同向且同概念Top1不降，才记为值得扩大完整评价的线索。

生成入口每段至多40张，保存每张条件、噪声、token、原生调用数、文件及像素SHA，并包含训练后关闭模块的原生像素回放。准备阶段另做同一实现的CPU设备复放；它不是独立公式实现。图片正面后仍需完整语义评价、多被试／seed确认，不能以小面板DINO声称跨架构普适或SOTA。

入口：[配对生成与审计](../generate_ATMS_standard_SDXL_ordinary_20261028_v1.py)、[真实DINO特征及固定评分](../score_ATMS_standard_SDXL_ordinary_20261028_v1.py)、[执行契约](../audit/crossdisciplinary_20261028_v1/cross_framework_image_execution_contract_v2.json)。

## 已完成检查的实际范围

启动前独立审查发现`train_concepts`实际为T／D／V映射，已修正读取T列表。旧草稿与首轮审查保留；修正发生在这些入口上传和GPU执行前。真实13230条元数据、D15的十个1650行full块、8200 Dfit索引及400条位置均已CPU核对；22项图片契约检查、三遍游标／末批／覆盖检查，以及80停止／400扩展控制器分支模拟通过。

[v2独立源码审查](../audit/crossdisciplinary_20261028_v1/cross_framework_training_source_review_v2.json)没有发现当前启动阻断。尚未实际执行本轮GPU反传、6张工程图片及GPU数值恢复对照；CPU游标检查不能替代这些证据。20GiB GPU allocated／10GiB额外CPU／输出额度属于程序检查，不是OS硬隔离。保留原始结果、所有失败和原BrainHIVE ordinary；不自动恢复旧正式D。

## 10:38实际更新：工程接入通过，正式训练已经开始

2026-10-28 10:38:48 America/Los_Angeles实际观察，原有限控制器PID240115／start_ticks82331630持续运行，正式GPU子进程242424在线，公开进度64／24807更新、1024次主query暴露。首段于10:36:17启动；没有更换科学源、数据或seed，后续自动接Dfit及匹配图片。此时尚未完成完整训练或任何研究效果图。

锁定公开SDXL／h94权重已按原SHA恢复，恢复全段5511.400秒；这是本次资产重建成本，不计成训练速度。新标准VAE T bank已完成全部13230张原图、52块、完整SHA／重开检查和两张CPU VAE重放，主体742.413秒。原旧bank清单保持不变，新bank完成SHA为`6eebf2fdcaa56b6b7bf8d39c38217c17a8cd0821f1844f124f8b0a7a4b6c2829`。

真实GPU接入profile完成58.604秒，两次工程更新均丢弃；单独ε损失对新头的梯度非零，原生成器权重摘要保持不变，零头／关闭的原生噪声预测逐位一致。GPU allocated峰值12323453440字节，额外CPU峰值10348216320字节，均在原程序上限内。正式训练从头重新初始化，不从这两步继续。

另完成全部20组真实ATMS affine条件／P token检查及两个固定T query的原生／零头／关闭共六张512图，20步Euler、CFG5、IP1，72.712秒。实际负IP输入为零、正分支为正确条件，三个路径同噪声并逐像素一致。工程图没有读取目标原图，也没有效果比较分数；它只证明本宿主接入保留了原生行为。

训练spec已在读效果前锁定，SHA `bb92cb615600f2957d1e02daa5550787ba02213697b622b93b300b423ef103de`，图片policy SHA `d88bb905198b7c16bec3169eb652427d9ddd45d6544877056352a1b9d0babd31`。实际证据已取回并逐文件核验：[15件运行产物SHA清单](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T173702233588Z/export_manifest.json)、[真实GPU反向profile](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T173702233588Z/candidate/profile/completion.json)、[六张工程图与CFG路径回执](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T173702233588Z/candidate/engineering_images/completion.json)。前节“尚未反传／工程图”的表述是启动前快照，由本段实际结果更新；独立跨进程GPU恢复对照仍未完成，不能由这两个接入检查替代。

独立CPU断点归档器PID241664已经实际在线，绑定上述同一控制器身份与源SHA；17项真实Linux检查通过。每个已审计的滚动断点只有在与CPU审计SHA吻合后才另存完整字节。未审计的末断点单独标注，晚到且已被覆盖的旧版本明确记为缺失，不把历史路径字符串当作仍存在的权重。归档器不读分数、不修改正式训练及配方。

## 10:49实际更新：512断点闭合，第二段已真实续跑

2026-10-28 10:49:26 America/Los_Angeles，公开进度544／24807更新、8704次主query暴露；原控制器仍为PID240115，第二段GPU子进程242903在线。首段512更新／8192暴露、程序676.198秒，已按原队列自然结束，CPU审计完成后自动进入`train_002 --resume`，没有暂停或重启原科学训练。

首断点9899473字节，SHA `f61bf0c021687bbf1fd198350de9c1454d924fbf42a1c24b1330d88e2b0718c4`。模型、全部AdamW状态、CPU／CUDA／独立噪声及Python／NumPy RNG字段、顺序、覆盖和提交日志前缀均通过CPU结构／有限性检查；第二段实际`EXACT_CHECKPOINT_RESUME`记录读取的也是同一SHA、512更新／游标8192，未提交日志尾长度0，并已新增32次更新。独立归档器已另存同SHA完整字节，确认独立inode及重读SHA。

这些证据证明自然分段、断点保存和实际恢复路径已经贯通，不能替代“同一两步连续运行对比跨进程1＋1”的GPU逐位对照。后者仍是未执行候选；没有为插入它暂停现有长队列。六份锁定源的只读运行时复核已完成，见[接口复核](../audit/crossdisciplinary_20261028_v1/X01_locked_runtime_interface_readonly_review_20261028_v1.json)。校准与部署的head算术精度边界另有记录，尚无实际强度或图片受影响的证据；不据此修改已锁方案。

## 11:35实际更新：2656次更新，五次自然续跑已发生

2026-10-28 11:35:19 America/Los_Angeles，原控制器PID240115／start_ticks82331630仍在线，第六段GPU训练子进程244498在线；公开进度2656／24807更新、42496次暴露。共有五次真实`EXACT_CHECKPOINT_RESUME`记录；最近一次由2560更新／40960游标断点继续，SHA为`48637972cfb1c711859f1f644d69941019ca0c4d7cfaccbbfdd02268daa81087`，9899473字节，未提交日志尾0。

五个自然段末的CPU结构／有限性及提交日志检查均已通过；2560断点已由独立归档器另存同SHA字节，独立inode确认。[本次取回的30件原始产物及SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T183704732791Z/export_manifest.json)包含实际观察、五份CPU审计和归档记录。该检查不包含独立重跑GPU更新；中间断点不是按效果挑选的最终模型。没有修改六份锁定训练／生成／评分核心源、数据、损失、seed或完整24807更新终点。

当前仍在第一遍T训练，Dfit校准与研究图片均未启动，没有跨框架效果分数。完整结束后的匹配native_ATMS／ordinary_ATMS同噪声80→400仍由原队列执行。第二卡正在另行准备[原生HIVE生成环境及工程回放](BRAINHIVE_SECOND_HOST_RUNTIME_PREPARATION_20261028_v1.md)，它不占用本X01 GPU，也不替代ATMS原生对照。

## 12:02实际更新：3936次更新，原长队列持续

2026-10-28 12:02:48 America/Los_Angeles，原控制器240115／start_ticks82331630及第八段GPU245302在线，公开进度3936／24807、62976主query暴露；七次真实恢复，最近由3584／57344断点继续，SHA `fdd895b60e1985a65f9aa58d7b2078bf86b0fd59801f58cf0956c35ce0dedd11`。七份段末CPU检查和完整字节归档记录已保留，[本次36件原始元数据SHA清单](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T190401934958Z/export_manifest.json)包含实际观察及新终态记录。未重启控制器、未改配方或科学源，当前仍无研究效果图或收益分数。

第二卡另行完成[D32全部T的跨重复二阶预测检验](BRAINHIVE_CROSS_REPEAT_QUADRATIC_RESULTS_20261028_v1.md)，结果未过门槛并停止。它没有占用本X01 GPU，也没有改变本轮ordinary的完整24807更新、校准和匹配图片终点。

## 12:28实际更新：5120断点闭合，第十一段在线

2026-10-28 12:28:26 America/Los_Angeles，原控制器240115／start_ticks82331630仍存活，第十一段GPU246507在线；最近公开提交为5120／24807更新、81920暴露。第十段自然完成后，CPU检查及独立归档均已通过，完整9899473字节断点SHA `02ebc2b9e2b2636893c9ef5fc2c6e33acd6fcba3b6416111530e9f18b1778e1a`。第十次真实恢复读取同一5120断点，提交日志前缀2627239字节，未提交尾0；本观察尚未将第十一段未公开的更新补计入数值。

原HIVE生成环境已在第二卡通过四图工程回放及独立CPU原图／噪声核验，GPU随后释放；它证明可分别运行，不是新增图片收益。X01继续原完整训练与后续同条件图片计划，目前没有跨框架收益结论。


## 12:51实际更新：原长队列继续，6176次更新

2026-10-28 12:51:35 America/Los_Angeles，原控制器240115／start_ticks82331630与GPU247311实际在线。公开进度6176／24807更新、98816次主query暴露；12次真实自然恢复。没有重启控制器或改数据、损失、seed、源文件、固定最终终点。Dfit校准和研究图片尚未启动，不能从训练进度宣布迁移收益。

[本次完整原始元数据及SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T195313332429Z/export_manifest.json)保存新断点审计、归档和实际进程观察。第二卡的[D33完整T信号结构检验](BRAINHIVE_TEMPORAL_OVERLAP_RESULTS_20261028_v1.md)及其独立CPU复核另已闭合，未占用X01 GPU。


## 13:12实际更新：7136次更新，原队列持续

2026-10-28 13:12:06 America/Los_Angeles，实际原控制器240115／start_ticks82331630与GPU247774在线；公开进度7136／24807、114176暴露，13次真实自然恢复。完整终点、数据、损失、seed、六份核心源以及原HIVE ordinary均未变。校准和研究图片仍未开始，没有跨框架图片收益。

[本次57件原始产物与SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T201521505704Z/export_manifest.json)包含最新真实观察、段末CPU检查、终态和归档记录。另一台机器的[D34相位机制检验](BRAINHIVE_OVERLAP_PHASE_MECHANISM_RESULTS_20261028_v1.md)另已完成，未占用本GPU；其原始降误差不能改写为本轮ordinary或图片收益。


## 13:43实际更新：8576次更新，原队列继续

2026-10-28 13:43:22 America/Los_Angeles，原控制器240115／start_ticks82331630与GPU248925实际在线；进度8576／24807、137212次暴露，已进入第二遍。16次自然恢复；最新8192断点的模型、优化器、RNG、顺序覆盖和日志前缀另经CPU核验。原完整终点、损失、数据、seed和六份执行源未改；校准、同宿主原生／新ordinary研究图片尚未开始。

[本次66份不可变元数据及SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T204749163875Z/export_manifest.json)保存原始观察。D35仅在第二台机器做CPU冻结诊断，结果未通过原门槛，没有新增删负项训练，不影响本轮跨框架试验。


## 14:21实际更新：10304／24807，原跨框架队列继续

2026-10-28 14:21:27 America/Los_Angeles，原控制器240115／start_ticks82331630实际在线，进度10304、164860次暴露，20次真实恢复；最新CPU检查断点为10240更新。GPU观察为`250599, /root/miniconda3/bin/python, 12944 MiB`。训练、校准、图片门槛及六份执行源均未改变；校准与研究图片尚未完成，没有跨框架收益读数。

[本次75份不可变元数据与SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T212515168729Z/export_manifest.json)。第二台机器的D36只做CPU教师诊断，没有新增GPU训练，不将其读数混入本项效果。

## 15:12实际更新：跨框架完整训练继续

2026-10-28 15:12:10 America/Los_Angeles，原控制器240115、start_ticks82331630实际在线，公开进度12672／24807（51.1%）、202748次query暴露，24次自然恢复。最新12288更新断点经CPU核验模型、AdamW、CPU／CUDA／噪声RNG、顺序覆盖与已提交日志前缀。训练、校准、图片门槛和六份执行源均未改变。ATMS宿主的新ordinary从头拟合，不加载HIVE ordinary权重；校准与研究效果图片仍未完成，不能判断迁移是否有效。

[本次87份不可变元数据及SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T221337033057Z/export_manifest.json)。第二卡另行完成D37的五张HIVE T工程图及像素回退验证，它们不替代本项ATMS原生／ordinary的匹配效果比较。

## 16:23实际更新：15904／24807，原队列继续

2026-10-28 16:23:05 America/Los_Angeles，原控制器 PID240115（start_ticks82331630）和 GPU训练子进程254711实际在线；公开进度 **15904／24807（64.1%）**、254460次主query暴露。最近完整提交15872更新／253948暴露的9899473字节断点 SHA 为 `99d0560526f78d28b724fdfb34d9f5c0b0be7adf8f4afeca1143778bd6dfd49a`，模型、AdamW、CPU／CUDA与噪声RNG、顺序／覆盖和日志前缀经CPU核验。该检查不是独立重跑GPU更新。

六份运行核心源码与原BrainHIVE ordinary检查点SHA未变；没有重启、改损失、提前终止或挑中间checkpoint。ATMS→原生h94 IP→标准SDXL-base直接条件分支仍从头训练新525568参数ordinary；不是把BrainHIVE已学权重直接复制过去，也不是作者完整prior→Turbo分支的复现。固定24807更新后接原Dfit8200校准与同query／同噪声native_ATMS对ordinary_ATMS的20步CFG5、单图80→400评价，当前这些阶段尚未完成，**没有跨框架效果结论**。

见[16:23实际进程、训练与保护SHA原始回执](../audit/crossdisciplinary_20261028_v1/bridge_actual_X01_D37_documentation_final_20261028T232303225996Z.json)及[本轮105份不可变元数据与SHA](../audit/crossdisciplinary_20261028_v1/X01_runtime_actual/20261028T231459758417Z/export_manifest.json)。后者是文件取回快照，内含旧时间的状态文件；当前15904更新来自前者的实时读取，不能把取回时间当成每个文件的生成时间。

第二卡上D37另完成1200张HIVE研究图与五张ordinary回放并按原门槛停止，未胜原HIVE ordinary；其宿主、采样和对照与本X01不同，不替代跨框架的配对图片比较。原正式D继续停止。

## 16:42实际更新：两遍T已完成，16800／24807

2026-10-28 16:42:34 America/Los_Angeles，原控制器240115（start_ticks82331630）及GPU255138实际在线，已16800／24807更新（67.7%）、268792次query暴露，complete_passes=2。最近16384／262140提交断点的模型、AdamW、CPU／CUDA与噪声RNG、顺序／覆盖及日志前缀经CPU核验；SHA为`92a2199b6d810d9976b0396c0d4377db4f737965f2e36482833d9edc6e91af07`。六份锁定核心源与原HIVE ordinary未变，未重启原控制器或更改完整预算。

原训练→Dfit8200固定校准→原生ATMS／新ordinary同噪声20步CFG5的80→400图片仍按原队列继续；校准与研究图片尚未完成，没有跨框架效果结论。见[本次实际进程、断点与保护SHA](../audit/crossdisciplinary_20261028_v1/bridge_actual_D38_documentation_X01_live_20261028T234232483548Z.json)。另一台机器的[D38有向时间交互解析检验](BRAINHIVE_ORDERED_AREA_RESULTS_20261028_v1.md)已按门槛停止，没有占用本GPU；其F预测读数不是本跨框架图片收益。
