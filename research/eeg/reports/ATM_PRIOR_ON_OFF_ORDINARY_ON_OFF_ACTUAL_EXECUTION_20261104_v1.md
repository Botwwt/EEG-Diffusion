# 原生ATM的prior开关与冻结ordinary开关：实际图片队列

本项已在旧卡实际启动，同一控制器PID653049、start_ticks139069421。2026-11-04T06:09:02.813646UTC实查，原控制器及400阶段direct第160起始块的GPU子进程653706在线；完整80条初筛及其CPU审计和DINO阶段成功，已按冻结联合规则扩400。没有读取本次80数值或中间160/240/320效果，也没有完整400结论。观察时刻不是未来运行保证。

这是一项补齐结构比较的开发实验，不因去掉一个模块就声称第二项创新。上一轮完整T显示，较好的EEG视觉条件并未提高同一冻结prior的时刻刺激区分；本次改用正常EEG推理的实际图片检查prior的净作用及其与ordinary的交互。

## 两个独立开关

| 分支 | 条件路径 | 新图片 | 每query prior前向 | UNet前向 |
|---|---|---:|---:|---:|
| parent | R4 EEG→同一full ATMS→原生50步prior→h94/Turbo | 原400复用 | 100 | 4 |
| ordinary | 上述prior输出→冻结ATM ordinary→同一h94/Turbo | 原400复用 | 100 | 4 |
| direct | R4 EEG→同一full ATMS→同一h94/Turbo | 最多400 | 0 | 4 |
| direct_ordinary | 上述直接EEG条件→同一冻结ATM ordinary→同一h94/Turbo | 最多400 | 0 | 4 |

全部sub-01、物理R4、63电极、相同40概念400刺激/400候选图库、512像素、4步、CFG0、IP1、每query一张及同一实际噪声。直接分支不读真图、类别标签、图库答案，不重缩放或截断编码器输出；编码器、prior、ordinary和图像生成器全部冻结。训练更新及新增模型均0。

这里的ordinary是此前原生ATM迁移时已完成的525568参数头，检查点SHA34e294b4…939844、2481更新/39690暴露，强度1、半径0.25。它不是最早HIVE a66869e5…9e11cc03权重；最早原件仍不可覆盖。本次改变该ATM头的输入分布却不重新训练，因此比较回答固定头的组合效果，不能说已经优化出绕过prior时最好的ordinary。

prior开关同时改变输入条件、截断与采样计算，不能凭一个指标唯一定位某一项原因。直接分支的prior成本更低；没有偷偷给它更多图像或best-of-N。

## 事前固定比较与继续规则

固定报告direct−parent、direct_ordinary−ordinary、direct_ordinary−direct、ordinary−parent，以及交互差 `(direct_ordinary-direct)-(ordinary-parent)`。交互可判断ordinary的增量是否随prior改变；不存在预设两个收益可相加的保证。

每个阶段完整报告DINO目标余弦、DINO2AFC、400图库Top1、同概念Top1、同概念2AFC、RGB像素相关、SSIM和平均RGB绝对误差。1000次概念重采样、seed1729，保留全部逐query差值与40项区间/阶段；本轮仍是单seed、已反复曝光的开发面板，未控制顺序观察的名义错误率。

80条只在两个绕过配置均相对各自prior版本出现明确联合损害时停止：DINO差≤−0.01且区间上界<0，同时同概念2AFC差<0且上界<0。其他情形两个配置一起扩400，不从80挑赢家。现在已进入400；这仅说明没有触发上述整体停止规则，不能当作收益或SOTA。

## 已实际完成的输入与工程证据

400条条件已在任何新图之前锁定。原两组400条FP32条件逐位保留；新直接条件来自同一T-only full ATMS，全部400条Dcheck概念均未进入其拟合集。真实EEG八条CPU/GPU前向scaled差4.50e−7；全部400条ordinary独立NumPy64重算最大差1.9133e−6，零头及关闭路径逐位相同。准备程序11.330秒，不把它写成完整出图耗时。

图像程序从旧原生四臂已运行程序另存后明确修改导入、分支和溯源字段。旧源及原图片不改；新源六份与注册副本SHA逐位相同。直接分支生成前先回放parent第0条，要求同一真实初始噪声、四次ancestral噪声及解码像素SHA与旧图一致；其80图及后续审计已经成功。

新例子字段明确标记prior_bypassed，prior_sample_seed为null；没有把不存在的prior采样记作已执行。生成、身份/实际条件FP32-FP16/全部噪声/像素CPU审计、冻结DINO、低层指标和配对统计均已实现并串入同一个有限控制器。完整400及终态接收仍待完成。

本地检查用完整历史400真实schema复核基线八项区间，误差0；另外用明确的合成数据检查两配置联合损害规则及交互恒等式。这些不是新的bypass效果。终态接收和独立复核工具已实现：完整阶段读取全部排名、所有5×8区间、完整400头输出和固定32张图片；实际OLD liveguard已通过，当前不下载效果数组，不创建本地结果目录。正向完整接收分支尚未执行，不能写成已复核新效果。

## 执行与资源

实际启动05:56:44.792461UTC，原截止09:56:44UTC不自动延长。每个GPU作业900秒、CPU阶段900秒；图片每块80条，失败保留前缀，不自动重训、重出已有图或搜索参数。独占现有旧卡；第二卡不联系、不启动。

512MiB覆盖新code、条件、最多800张图、特征、日志与元数据。已存在800张基线图片及其全部特征直接复用。准入前实际磁盘余量满足全部50GiB旧底线、8GiB缓冲、既有所有承诺及新的512MiB。已闭合128MiB T支持、256MiB true-forward、256MiB条件曝光组未使用的未来额度仍继续计费；没有删除文件或降低任何上限。

当前本地Python客户端一度在启动阶段返回0xc0000142，随后同一Python重新进入正常、原SSH只读读取恢复。客户端故障没有导致重启服务器研究作业。用户暂停和恢复goal期间原队列未被本地重挂；此次06:09的实查证明原控制器仍在线，不能据本地观察丢失宣告科学失败。

原正式HIVE D仍停止，27个未运行原生EEG拟合不恢复。当前没有新图片效果、第二项独立创新、官方SOTA或CCF A发表性结论。后续应先完成本固定2×2比较，再依据全部指标决定是否重新设计prior中的EEG信息保持机制；普通旁路、x0/EDM预条件或h-transform本身均不能自动作为原创。

[六源注册副本](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/ATM_prior_factorial_registration_20261104_v1)、[与旧程序的实际差异](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/ATM_prior_factorial_source_derivation_20261104_v1.json)、[真实准入与启动](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/RAMroutes_old_goal133_native_prior_on_off_ordinary_on_off_actual_admission_launch_20261104T055622586049Z.json)、[400条件与真实GPU阶段](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/RAMroutes_old_goal133_same_factorial_controller_and_actual_prepare_or_images_20261104T055738539013Z.json)、[暂停恢复后的同一控制器实查](D:/扩散模型/research/eeg_sota_autoresearch_20261018_v1/audit/crossdisciplinary_20261028_v1/RAMroutes_old_goal133_resume_same_exact_factorial_no_research_restart_20261104T060840331239Z.json)保留可核查证据。
