# 四臂训练继续、全量图库与官方全训练前置

本文记录新实例上的实际执行，不改变已冻结的 O/P/PN/S 方法、配方、训练 seed 或比较规则。用户取消清理的指令继续有效，本轮没有删除旧模型、图片、EEG 或结果。

## 1. 当前正式训练

2026-10-22 22:58:33 UTC 的[实际观察副本](../audit/preparation_delivery_20261022_v1/live_training_observation.json)确认恢复控制器 PID4150、O 训练 PID4180 在线。O 日志为 1792/24807 更新，随后保存的滚动断点已到1817更新、29072次query暴露并实际 CPU 解码通过；日志和断点是依次读取，不冒充同一瞬时游标。GPU 实测占用13582MiB、利用率92%，扩容盘可用308516958208字节。P/PN/S 尚未开始。原失败作业、178.929 秒 GPU 耗时和单次同 seed 从头重放的记录均保留；没有把重放称为精确恢复。

正式队列继续按每臂完整 132300 个 OOF-T query、三遍固定 24807 更新、最终头 T32、全部 16600 张 V、artifact、特征及实际评分执行，最后固定比较 PN−O、PN−P、PN−S、P−O。新四组尚无完整训练、V 图片或效果结果。资源扩容没有增加候选次数或训练 seed，也没有改变门槛。动态入口仍为服务器 `audit/functional_full_lifecycle_20261022_v1/formal_recovery_queue.json`。

## 2. 原三组完整图库索引

原先仅做 CPU 准备的[索引入口](../build_BrainHIVE_bounded_gallery_index_CPU_v1.py)已在服务器实际运行，输出到 `/root/brainhive_reproduction_20261020_v1/gallery/bounded_component_V_index_20261022_v1/`。其[实际 completion](../audit/gallery_index_actual_20261022_v1/completion.json)记录 16600 条 query、49800 张既有生成图、30 条按被试均值、四份各含全部 16600 条改善/退步排序表；输出约 86.23MB，用时约 12.87 秒。

它引用完整原图和原 166 个分页，不复制、解码或重生成图片，不重新评分。来源表包含生成路径、目标 ZIP/member、配对 ID、seed、清单及评分文件 SHA。排序保留全部分母，不挑选有利个例。DINO、ObjectF1、旧空白代理筛选不等于人工质量标签。

新增核验器两次真实提前失败均已保留：v1 把运行状态的单一源码 SHA 字符串误当字典；v2 又错误要求历史图库目标清单包含数组行号。[第一次失败](../audit/gallery_index_actual_20261022_v1/verification_attempt1_failure.json)与[第二次失败](../audit/gallery_index_actual_20261022_v1/verification_attempt2_failure.json)不改写为成功。v3 根据真实小观测和五类旧表 schema 修正：原 Runner 与恢复控制器各核自己的单一 SHA，数组行号来自 `V_targets`，再与 provenance 及三组生成清单闭合，没有给旧表补虚构字段。

[v3 独立入口](../verify_BrainHIVE_bounded_gallery_delivery_CPU_20261022_v3.py)通过 9 组本地/服务器 CPU 检查和独立源码复核，随后[真实全量交付核验](../audit/gallery_index_actual_20261022_v1/delivery_verification_v3.json)于服务器通过，用时 12.659 秒。它核对全部 16600 行、49800 个生成文件的路径/samefile 身份、30 条均值、四份各16600行排序、旧166分页及三份既有低层完整分母，前后确认同一 O 训练与控制器、29份冻结源及完整剩余空间预算。没有读取图片负载、重新解码、重算分数或改变训练。

浏览器渲染验收仍未完成。该索引依赖服务器原资产，不是可独立搬走的单 HTML 包。没有新增 R4 打乱/缺失条件图片；R80 和 T32 干预不能冒充这些条件。空间完整性也不证明感知质量非劣。

## 3. 官方完整训练阶段的真实元数据入口

新增[元数据入口](../prepare_BrainHIVE_official_fullfit_plan_CPU_v1.py)及[17 项检查](../check_BrainHIVE_official_fullfit_plan_CPU_v1.py)，本地和服务器均实际运行并通过；另有独立只读源码复核。[服务器实际回执](../audit/official_fullfit_metadata_plan_server_20261022_v1/preparation.json)与[跨实例全行核验](../audit/official_fullfit_metadata_cross_instance_QA_20261022_v1.json)保留真实范围。

入口只读取四份固定 SHA 元数据：完整图像清单、原输入绑定、旧概念划分、master 政策原文。它没有打开 EEG、图片、权重或评分数据，也没有构建模型或发起拟合。

| 项目 | 已实际生成的元数据 | 尚未执行的工作 |
|---|---|---|
| 官方训练覆盖 | 1654 概念、16540 图，旧 T/Dfit/Dcheck/V 全部纳入 | 该阶段的真实全量拟合 |
| 新 OOF 划分 | 全部概念按固定 Random1729 排序洗牌、交错两折，各 827 概念、8270 图；概念不跨 fit/held | 20 个被试×折编码器及条件导出 |
| 逐 query 映射 | 165400 行，每条指向未拟合其概念的未来编码器任务，gzip 全行及 CRC 复读 | 实际条件张量、检查点及其 SHA；这些字段仍为空 |
| EEG 任务预算 | 每任务 25 epochs、batch1024、末批78、预期225更新 | 实际参数更新为0 |
| 组件预算 | 3遍、batch16、末批8，预期496200次暴露、31014更新 | 新阶段训练/latent/生成/评分的正式实现与准入 |
| 官方评价 | R80、10被试、200候选、2000 query、每query一图 | 最终头的同协议实际图片和论文指标 |

本地与服务器的折划分、任务表和 query gzip 字节 SHA 相同；跨实例核验再次逐行确认所有 165400 条 query 对应非拟合概念编码器。测试 200 图的全局概念及图像 SHA 均与训练隔离，但既有曝光事实仍保留。原输入依赖发布顺序和元数据/ZIP回执，没有独立重放原始事件标签。

所有输出明确为 `PLAN_ONLY_NO_METHOD_LOCK_NO_FIT_NO_AUTO_LAUNCH`。当前完整四臂比较后才能锁方法和最终校准权限。旧 Dfit/Dcheck/V 进入全训练后不再独立；不得用官方测试或图库答案校准，也不自动复用开发强度。published 全16540父模型须重新审计才能复用，旧 T-only OOF 不能替代新 OOF。

parent/O/P/PN/S 的同等最终拟合机会已明确列入待锁定要求，旧 ordinary/risk、最强保留内部基线和跨架构强参照仍须审查，不能因只实现当前原生四臂便忽略。PN 失败不能改名为 P 成功。当前没有官方完整拟合、最终比较、独立 H 或 SOTA 结论。

## 4. 论文原指标资产与当前 CPU 数学检查

新[持久恢复入口](../restore_BrainHIVE_original_metric_assets_CPU_20261022_v1.py)通过 17 项本地/服务器 CPU 检查及独立源码复核。[实际恢复](../audit/original_metrics_restoration_20261022_v1/persistent_asset_recovery_attempt1.json)用时 77.503 秒，仅从官方地址各请求一次 AlexNet 244408911 字节、CLIP-L/14 932768134 字节；完整原 SHA 一致后保存到扩容持久盘，原逻辑路径仅新增符号链接。Inception、SwAV 和共享 EffNet 复用并核验既有持久文件。

五资产都与旧实际回执的身份一致，没有换评价器。大文件核验后释放其干净页缓存，原文件保留。恢复前后 CPU 实际余量约 23.18GB；扩容盘恢复后可用约 308.52GB，完整四臂存储预算与另 1GiB 资产缓冲仍满足。本轮没有新增 GPU 作业、图片、分数或旧文件删除。

首次服务器小检查因本地镜像形式的测试回执路径不存在而失败；[固定原件复制准备](../audit/original_metrics_restoration_20261022_v1/server_check_fixture_preparation.json)后，17 项服务器检查通过，再启动一次正式资产恢复。此前没有发起下载，未修改产品源码或覆盖旧回执。

[实际 CPU 数学探针](../audit/original_metrics_restoration_20261022_v1/math_CPU_probe.json)使用合成输入核对原 2AFC 方向、解析排名和 skimage0.24 的二维 SSIM 语义，没有读取真实 benchmark 图片或模型权重。它支持数学与当前 CPU 包环境可用，不能替代本实例五模型严格加载、GPU 前向或全量实际评价。这些后续检查仍明确待完成。

## 5. 解释边界与下一步

本轮完成的是正式训练继续、既有全量结果的检索入口和最终完整训练/评价前置。原三组的小幅开发集收益及风险门控未通过双对照门槛的结论不变。新四臂结果、独立机制增量、感知质量非劣、训练 seed 稳定性和官方可比较 SOTA 均未证明。

继续等待固定全量 O/P/PN/S 生命周期，并完成图库浏览器验收及论文原指标模型环境检查；当前四臂比较后才能锁定官方阶段方法和校准权限。元数据计划不启动新拟合，不提前读取未完成的新四臂 V 分数。

## 6. 本实例五模型加载、部分浏览器核验与可导入官方接口

2026-10-22 16:35 America/Los_Angeles（服务器23:35:23 UTC）实际观察：O仍为原训练PID4180、控制器4150，日志3264/24807更新；另取保存的3239更新、51824次query暴露断点，CPU解码与覆盖计数核验通过。日志与断点是不同采样时刻，不把它们混作精确即时cursor。扩容盘约308.517GB空闲，P/PN/S保持原队列，原失败和耗时记录保留。用户停止清理后没有删除旧文件，也没有新增GPU作业或新V图片。

[五原模型CPU入口](../probe_BrainHIVE_original_metric_models_CPU_20261022_v1.py)通过13项CPU检查及独立源码审查后，已在本实例[实际严格加载](../audit/original_metrics_restoration_20261022_v1/CPU_model_probe_attempt1/completion.json)AlexNet、Inception、EffNet、SwAV和原CLIP-L/14，31.418秒、观测最低有效CPU余量19256279040字节。逐模型独立子进程均正常退出，权重/缓冲均在CPU且有限，原资产/源码SHA一致。没有图片、前向、评分或CUDA初始化；原论文指标的本实例GPU前向、原生精度及完整图片评价仍未验证。不得把该结果写成官方分数。

[浏览器实际记录](../audit/gallery_browser_QA_20261022_v1/browser_verdict.json)仅判部分完成：首屏16600总分母、415页分页、sub-10筛选、概念筛选、精确query搜索及目标/parent/ordinary/risk四列原图加载已看到。Chrome阻止单独打开保存的WebP，记录`ERR_BLOCKED_BY_CLIENT`且未绕过；稍后独立SSH转发连接失活，返回HTML出现`ERR_EMPTY_RESPONSE`，两项故障分开保留且不推定原因。临时浏览器页、仅loopback监听和转发均已停止；主训练连接正常。排序、错误筛选、旧页跳转和CSV下载仍未通过浏览器验收。没有人工质量标签或感知非劣结论。

新增可被未来真实producer导入的[编码器/导出契约](../brainhive_official_encoder_job_contract_CPU_v1.py)和[R80身份/目标关联契约](../brainhive_official_benchmark_contract_v1.py)，不是新拟合。前者约束20个真实任务的8270 fit/held、25epoch/225更新、完整fit→cache链和own-affine；后者约束10×200全量盲推理、真实字节观测字段、保存后再join目标、跨方法/EEG干预配对，以及论文七项/代码额外项/common指标分轨。缺失条件要求旁路新增组件，并使原生A实际输入为零；不能代替为零EEG经BrainMLP输出，也不能用seed哈希补写旧清单的噪声观测。

[本实例完整CPU交付检查](../audit/CPU_contract_delivery_20261022_v1/server_execution_v3.json)实际通过14项编码器、13组每条件2000条的合成R80身份检查及20项/165400行真实计划读取，用时14.295秒。合成记录不证明真实hook、权重、图片或评分调用；契约不签发方法锁或准入。首次测试误选本地plan镜像、第二次缺历史CPU回执，均在测试setUp阶段失败且原记录保留；显式选择服务器plan并复制原24118字节历史回执后才通过，没有放松严格校验或修改原计划。

官方阶段仍缺真实单task训练入口、严格全量OOF导出、published当前负载审计、165400条件loader、全训练latent与31014更新流程、最终R80生成和实际评分调用。当前四臂比较未闭合，官方方法/校准权限未锁；没有独立H、独立机制或SOTA结论。完整本轮范围见[进展回执](../audit/CPU_contract_delivery_20261022_v1/actual_progress.json)。

## 7. 官方单任务训练与严格OOF导出的实际代码

2026-10-22 19:30 America/Los_Angeles（服务器2026-10-23 02:30:49 UTC）观察：同一O训练PID4180、控制器4150仍在线，日志10112/24807；保存断点的更新、cursor、已暴露query及覆盖矩阵另见[实际训练观测](../audit/official_executor_implementation_delivery_20261022_v1/live_training_observation.json)。P/PN/S顺序不变。没有新增GPU作业、四臂V图片或新分数读取，用户停止清理后没有删除旧文件。

已实际新增[训练核心](../brainhive_official_encoder_training_v1.py)、[单task训练入口](../train_BrainHIVE_official_encoder_v1.py)、[严格OOF导出](../export_BrainHIVE_official_OOF_conditions_v1.py)和[执行前置核验](../brainhive_official_execution_contract_CPU_v1.py)。它们实现每任务8270 fit/held、25epoch/225更新，batch1024保留末批78，使用原生BrainMLP与冻结但train/dropout.1的实时F.vision_model教师；没有改用确定性F缓存或额外经过A作为训练目标。InfoNCE内部归一化不改变导出的原生raw LayerNorm条件。

训练入口保存model、logit、Adam、cosine进度、CPU/CUDA/shuffle RNG及registration/source SHA。每epoch的resume与真实epoch记录以同文件系统目录事务发布；只有连续完整commit可恢复，未提交目录和旧检查点保留。最终文件不覆盖。保留25份model+Adam+RNG状态要求未来每task输出cap至少2.5GiB，20项最低50GiB；这是新增实现的规划下限，尚未获得该官方阶段完整资源准入，不能沿用只计最终权重的旧估计。

导出代码严格核对训练完成、实际成功wrapper、全部epoch事务、checkpoint、EEG/视觉/F输入溯源和own-affine，然后只读完整16540训练EEG中的8270 held行，保存float32条件及精确全局array_rows。只有现有cache和receipt的完整实际SHA、所有数值/行及own-affine均通过才接受；不使用`exists→skip`，不读官方测试或检索目标。

[本实例CPU交付回执](../audit/official_executor_implementation_delivery_20261022_v1/server_execution_v2.json)确认训练8项、执行/导出10组与两条真实CLI的缺review提前拒绝。检查包括新CPU解释器间连续与恢复运行逐参数/Adam/scheduler/RNG一致、8270条合成8维输入单epoch9更新及末批78、tiny全局held顺序/own-affine落盘，以及真实训练事务→导出扫描的互通。独立复核发现的scanner漏参、同物理盘重复预算、内存检查源码漏pin及train/export action不对称均已窄修；旧源和早期实际回执保留。服务器交付程序最初漏给导出命令`--execution-review`，该项只到argparse；补参数后只重跑这一个拒绝检查，复用已通过的8+10项，不覆盖原失败。

[原生源码CPU实证](../audit/official_encoder_fidelity_review_server_20261022_v1.json)另确认冻结/no_grad的F在train模式仍随机，eval重复相同；恢复缺CPU或shuffle RNG会实际发散。这些是随机CPU输入与小维度软件检查，不证明真实CUDA/BF16完整拟合、恢复或8270条件导出。当前不存在正式官方method-lock/review，因此两个真实入口均在导入模型、读实际fit payload和写训练目录前拒绝；没有据此启动新拟合。

[published父模型实际负载审计](../audit/published_parent_payloads_CPU_20261022_v1/current_payloads_attempt2.json)对十个完整16540父检查点完成原SHA、strict native state、25epoch/425更新日志和own-affine核验。十份旧R80 cache均为目标缺失的悬空链接，cache验证数为0，未重算EEG导出、未恢复复用准入。首个审计因手抄registration SHA多一个字符，在读权重前失败；原注册未变，修正取自原归档并新增原件SHA检查，首次源和失败回执保留。当前四臂使用的full/OOF缓存不依赖这些published R80链接。

还需完成当前四臂正式图片比较、官方方法/最终校准与等机会锁定、该阶段实际profile/整体资源审定和20任务调度；真实全量fit/export、165400条件组件loader、16540 latent、31014更新、最终R80生成及真实论文/common评分均未闭合。published R80条件须在相应权限下从已审计模型重新导出或恢复原SHA副本。跨架构、独立机制与SOTA目标均保留为未完成。见[本轮进展范围](../audit/official_executor_implementation_delivery_20261022_v1/actual_progress.json)。


## 8. 全训练输入和独立工程profile的实际入口

2026-10-22 21:36 America/Los_Angeles 实际观察：O为15040/24807更新，原训练PID4180和控制器4150在线，保存断点另经CPU解码核验；P/PN/S按固定顺序等待。没有新增GPU作业、四臂V图片、分数读取或旧文件删除。 当前所有新入口仅完成CPU工程核验；原四臂配方、固定seed1729及最终比较门槛均不变。

[完整OOF读取器](../brainhive_official_component_data_v1.py)实现按20个新官方任务拼接165400条raw float32条件、20组各自检查点affine及16540个F目标，不将目标复制十份。它逐行核对真实完整query-map的global array_row、subject、held fold和cache_row；旧T-only/full/published缓存不能替代新官方OOF。真实加载仍须先有方法锁、完整资源与专用阶段review，再核全部25epoch事务、成功fit/export wrapper、实际文件和完整张量/affine SHA。export记录中的历史RUNNING job SHA不会冒充最终closed SHA；读取器另按注册job、精确命令和实际完成回执核验。既有全16540 F目标缓存仍需一个实际source-bound身份/负载核验producer，该producer本轮尚未实现或运行，不能手填通过。

[全训练latent入口](../build_BrainHIVE_official_train_latents_v1.py)使用完整16540训练图，保持原生FP32 VAE、MODE而非采样、Resize512 BILINEAR/CenterCrop和原scaling factor。VAE config的精确SHA从旧冻结decoder注册读取，不能由新review换成同结构配置。新bank为16540×4×64×64 float32，数据约1083965440字节，规划stage cap至少2GiB；65个完整分块可核验后续跑，未提交块可重算，旧13230 bank与任何旧结果均不改。文件和元数据写入前按实际allocation unit、唯一inode核算空间；canonical/pending硬链接不重复计费。

[编码器工程profile入口](../profile_BrainHIVE_official_encoder_GPU_v1.py)具有独立T-only工程权限检查，避免“先有正式profile才能运行正式profile”的循环。将来只使用原T的固定前8270条global row、原生BrainMLP/F教师，做一个工程epoch的9次更新（保留末批78，scheduler horizon仍225），再对同8270条做float32 raw-LN导出；这些行与工程训练重合，不能称held OOF效果。工程头不用于重建，条件不落为正式缓存，可选单次epoch事务仅测一份I/O。一个warm GPU作业的train/export两个分段不能冒充两项独立作业、25epoch或20任务实测。此程序尚无真实permission、GPU排队或CUDA测量，输出cap上限1GiB仅是权限上限。

[服务器实际31项核验](../audit/official_input_pipeline_delivery_20261022_v1/server_CPU_attempt1/completion.json)用时12.130秒：10项loader、12项latent、9组profile全部退出0，前后同一4180/4150进程在线，29份当前固定源、上一轮6份官方入口源、4配方和旧分数表SHA不变。覆盖完整165400行真实元数据、tiny实际张量拼接/own-affine、真实npy分块中断/损坏/低cap拒绝、原预处理与MODE数学，以及缺review/permission真实CLI提前拒绝。它没有构建真实全量条件或latent，也没有读真实EEG/图像/模型负载、新V分数或使用GPU。

独立复核中发现的context.runtime漏字段、原VAE configSHA漏绑定、profile报告写前预算及历史RUNNING引用问题均已修复并核验。latent首次Windows只读fd执行fsync失败，以及窄测试把POSIX元数据路径转反斜杠导致KeyError，原源与失败记录保留，分别改r+b和PurePosixPath；这些失败发生在工程检查，未产生实际GPU产物。原profile v1检查未覆盖的低cap分支也以独立新v2回执保留。

后续仍须闭合当前四臂图片比较、锁定官方方法/最终校准与强基线同等机会、实际工程profile/完整资源审定、20个真OOF拟合及导出、F目标来源核验、全训练输入和31014更新、R80真实生成及论文/common完整评分。单seed、已曝光官方200刺激、无独立H和未跨架构的限制均保留；没有新图片收益、独立机制、感知质量非劣或SOTA结论。
