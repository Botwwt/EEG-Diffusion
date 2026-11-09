# 第二卡原生BrainHIVE运行环境：实际准备与有限后续队列

最新真实终态：2026-10-28 12:19:16 America/Los_Angeles。整套公开权重传输、CPU安装与四张GPU工程回放均已闭合，原接续控制器退出。原生／零头／显式关闭三图逐像素匹配旧固定T参考，旧ordinary第四图亦已真实生成。没有新研究效果图或模型更新；旧卡X01继续其原训练队列。下面11:34“尚未完成”段落为该时间的历史记录。

状态快照：2026-10-28 11:34:56 America/Los_Angeles。跨框架X01仍占用旧卡；第二卡的本队列只准备原生HIVE生成环境和工程回放，没有自动启动新科学候选。权重仍在传输，尚未完成CPU模型安装或GPU图片验证。

两机只读实查确认Torch2.3.0+cu121、torchvision0.18.0+cu121和NumPy1.26.4一致；第二卡缺少原生HIVE公开权重与显式Python附加运行目录，持久盘可用44,032,462,848字节。没有修改基础Torch／NumPy环境、第三方项目或provider账户。原HIVE ordinary的权重、半径和强度1保持不变。

旧机CPU导出已真实完成，主体19.925842秒。21项逻辑映射对应19个唯一持久目标，总唯一字节8,490,982,669；包含锁定SDXL-Turbo／原生先验公开资产、排除基础Torch／NumPy／CUDA的显式附加依赖，以及旧固定T工程回放证据。[原始传输计划](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_transfer_plan_v1.json)SHA为`1d845a56007089dd5683cf50e7ecb353ed512816570226a29af671b4472fa8ba`。

首次传输v1在VALIDATE阶段因重复目标守卫停止，写入权重字节0；两个tokenizer的merges／vocab分别完全同SHA同字节，实际指向同一持久文件。失败保留。v2仅允许这种完全相同别名，冲突别名仍拒绝；原传输计划、模型SHA和信息权限不变。v2于11:23:41实际启动PID14481／start_ticks1465981419，7200秒有限直接SFTP，不经本地中转大文件、不保存密码。不能将目前已传前缀视作整套恢复完成。

11:34:56第二卡已实际启动有限接续控制器，timeout父进程PID14804／start_ticks1466048894。绑定上述唯一传输身份，等待真实终态成功后依次运行：

1. [CPU安装与严格prior加载](../install_HIVE_second_host_runtime_CPU_20261028_v1.py)，至多600秒主体／660秒包装；逐文件SHA检查，不覆盖已有不同文件，基础环境不变。
2. [四图GPU工程回放](../verify_HIVE_second_host_native_GPU_20261028_v1.py)，独占第二卡、900秒包装、原20GiB allocated／10GiB额外CPU上限。

安装器[8项实际Linux路径与renameat2 NOREPLACE检查](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_actual/20261028T183702719121Z/81ba097a05_HIVE_runtime_installer_actual_CPU_contracts_v1.json)已通过，覆盖授权路径、相对／越界／非规范路径、符号链接父目录、已有文件拒绝覆盖及新文件原子提交。它们不等于真实运行环境已安装。

四图固定旧T query `sub-01/moose_05s.jpg/R4`，同一seed5533825044790129053：原生、零头、训练头显式关闭，以及旧冻结ordinary。前三条须与原历史RGB像素、A输入、IP token和初始UNet输入一致；第四条只检验旧ordinary真实接入，不比较效果。加载采用已存在meta分支及四个转换模块逐张量核对，不修改原FP16权重。预计4张、16次UNet／16份ancestral噪声均只是待实际验证的执行契约，不能预填成功。

[队列源](../queue_HIVE_second_host_runtime_20261028_v1.py)与[事前锁定spec](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_followon_spec_v1.json)已落第二卡。任一步失败保留、不自动重试、不追加科学训练。当前新研究图片和新训练更新均0；X01的ATMS native／ordinary正式效果仍须其完整训练、校准和配对80→400图片闭合。

## 实际完成与时间边界

v2直接传输全部19个唯一文件、8490982669字节，所有原SHA通过；21个逻辑映射的重复别名没有重复写大文件。传输3228.058789秒，于12:17:30闭合，原v1准入失败保留。CPU实际安装6758个文件及公开资产路径，57.929934秒；原F＋A／P／IP的严格CPU加载参数分别19403776／8400896／340787200，额外CPU峰值3297026048字节。没有更换基础Torch／NumPy版本或旧科学源、权重字节。

新卡实际GPU程序44.956371秒，GPU包装46.028288秒；同一历史T query完成四张新图，共16次UNet和16份实际ancestral噪声。四组件转换加载逐张量匹配，原生／零头／关闭三路径的RGB像素、A输入、IP token和初始UNet输入全部与历史参考一致。额外CPU峰值9414647808字节、GPU allocated峰值8746399744字节，均在原上限内。旧ordinary权重、实现、半径与强度1保持不变，第四张仅为接入验证。

两个阶段的原包装退出码均为0；接续控制器实测2659.605894秒，含等待既有传输。传输与控制器的等待区间重叠，不能相加当成总耗时，也不能把资产迁移称为分钟级新方法迭代。首次设计、导出和文档另计。

实际证据：[全部恢复文件与原SHA](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_actual/20261028T191917714902Z/5ef0328282_HIVE_runtime_pull_actual_completion_v2.json)、[6758文件安装及严格加载](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_actual/20261028T191917714902Z/6964a23e93_completion.json)、[四图／转换权重／真实噪声回执](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_actual/20261028T191917714902Z/bd2bc2252b_completion.json)、[两阶段终态与真实控制器时间](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_actual/20261028T191917714902Z/75190bcc67_completion.json)。

随后另将四张实际新图和一张历史参考取回本地，独立用Pillow／NumPy重算全部文件及RGB像素SHA；三个应回退路径与旧图逐像素相同，四组初始噪声及全部16份ancestral噪声记录一致，五文件inode互不复用。四组件2641个转换张量摘要保持匹配。旧ordinary图相对native存在像素变化，仅表明接入产生作用，不代表收益。[独立CPU原图／噪声复核](../audit/crossdisciplinary_20261028_v1/HIVE_second_host_pixel_actual_v1/completion.json)同时取回原GPU包装COMPLETED／退出0，并实际确认GPU空闲及独占锁已释放；这次复核没有重出图或新GPU作业。
