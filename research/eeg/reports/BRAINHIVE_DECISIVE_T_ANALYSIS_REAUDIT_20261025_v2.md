# Brain-HIVE 已挂分析的第二轮复审与 v2 修正

本轮按用户“再次确认挂上的实验”的要求，对科学对照、真实训练路径、控制器和配置互通作了独立复审。**v1 不能称为完美无问题**：发现了下表中的实际缺口。2026-10-25 09:17:53 America/Los_Angeles，已确认 v1 没有任何分析阶段或 GPU 作业启动，仅显式停止等待控制器 PID105917。原 S 训练及主控制器未收到停止信号；v1 源码、配置、日志和状态全部保留。

## 1. 发现及修正

| 问题 | 为什么会影响结论或执行 | v2 的具体修正 |
|---|---|---|
| 加性统计量对照不完整 | 线性读取 `[c, mean(hpre), log sigma(hpre)]` 无法完成尺度与方向的乘性交互；c+h0 优于它，不能排除归一化前均值、尺度的解释 | 增加相同 2048 维的 c+hpre 对照，验证 hpre 可由 c、真实 γ/β 及行均值/尺度重建；“超出归一化前旁路”必须胜过 c+hpre，原加性对照仅作描述 |
| 梯度精度与真实训练不一致 | v1 在 BF16 autocast 内计算修正头，而真实训练在 autocast 外计算 FP32 头；梯度重放的固定 prompts 也需匹配训练精度 | 头在 autocast 外计算，A/P/IP/UNet 使用原 BF16 autocast；梯度阶段使用 BF16 prompts 副本，轨迹探针继续保持原生 FP16 |
| 分支冲突不等于实际目标冲突 | token 梯度可能改变基础目标方向，正教师与排序梯度相加也可能抵消分支冲突 | 主冲突量使用 `g_base = g_epsilon + g_token_weighted` 与各臂实际 `g_aux` 的余弦、点积和范数比；分支余弦保留描述，不当作总更新结论 |
| UNKNOWN 与公共偏移可能误放行 | 教师 UNKNOWN 或配对间隔没有增益时，绝对教师误差仍可能变好 | 教师未支持时阻止功能机制推荐；“代理通过”必须同时满足教师、配对增量、三个分支对照与状态迁移。CPU 表征可读取性仍独立判读 |
| 控制器有未覆盖的挂起/清理路径 | GPU 空闲查询缺 timeout；Popen 后首次保存状态失败、或 wrapper 先退出时，可能遗漏已创建的 child | nvidia-smi 查询各限 10 秒；进程创建后的状态保存、轮询及 GPU 闭合核验放在同一清理范围。只清理身份匹配的自有进程组，并实际验证 CPU 模拟 wrapper/child 的分离进程组清理 |

没有修改原四臂训练配方、模型、数据权限、采样设置或评分规则。没有根据新分析结果调整阈值，因为尚无新分析结果。

## 2. 修正后的决定性对照

### 2.1 EEG 中间层读取

仍使用 20 个冻结 OOF 编码器各自从未拟合的概念池，在池内二分读出拟合/评价，交换方向；40 个单元、每个 λ 覆盖全部 132300 个 T query。不得拼接全部 OOF 输出后再做普通外层交叉验证。

新增对照是：

$$X_{h0}=\frac{1}{\sqrt 2}[c,h_0],\qquad
X_{pre}=\frac{1}{\sqrt 2}[c,h_{pre}],\qquad
X_{dup}=\frac{1}{\sqrt 2}[c,c].$$

各分支使用读出拟合池自己的逐坐标标准化；拼接后作相同能量缩放。原 c+h0 对重复 c、异概念打乱 h0 的主条件不变。两个固定 λ、至少 1% 相对 MSE 降幅、0.005 正确减错误目标余弦间隔增量、至少 8/10 被试两项同向的筛选线不变。

LayerNorm 的精确数值关系为：

$$u=(c-\beta)\oslash\gamma,\quad
\mu=\operatorname{mean}(h_{pre}),\quad
\sigma=\sqrt{\operatorname{var}(h_{pre})+10^{-6}},\quad
h_{pre}=\sigma u+\mu\mathbf 1.$$

因此 c+hpre 是完整归一化前数值旁路的匹配维度对照；加性读取 c、μ、logσ 无法替代这个乘性关系。即使 c+h0 胜过 c+hpre，结论也只限于固定读出族的跨概念可预测性，不能写成额外互信息或语义丢失的因果证明。

实际入口：[表征 v2](../analyze_BrainHIVE_EEG_layers_T_CPU_20261025_v2.py)；[反例与权限检查](../check_BrainHIVE_EEG_layers_T_CPU_20261025_v2.py)。

### 2.2 功能与梯度

固定 80 query、8 个共享 T 刺激、两状态域、4 时刻、70 层以及原有完整 UNet 对照不增加。DDPM 的噪声 MSE 只在有已知 ε 时计算；真实父模型轨迹不使用真实 latent 更新，也没有真实 ε 误差。

状态迁移主判读既要求绝对教师接近，也要求正确减错配的配对增量。DDPM 本身未支持，不能称为“迁移失败”；教师退化、缺测或方向混合均保留 UNKNOWN。80 query 不是 80 个独立刺激。

梯度固定在每被试来自两个 OOF 角色的 2 个 query、4 个时刻上重放。每条梯度记录是一个 batch=2 的联合损失梯度，不能冒充逐 query 梯度或正式优化器实际更新。主冲突量为：

$$g_{base}=\nabla_\phi(L_\epsilon+0.1L_{token}),$$

$$g_{aux,P}=\nabla_\phi(0.1L_{pos}),\quad
g_{aux,PN/S}=\nabla_\phi[0.1(L_{pos}+L_{rank})],$$

$$\operatorname{cos}_{base,aux}=
\frac{\langle g_{base},g_{aux}\rangle}{\|g_{base}\|\|g_{aux}\|}.$$

P/PN/S 的被试×时刻记录固定等权汇总；非有限或零梯度使相关判读保持 UNKNOWN。分解梯度、合成梯度的范数比及总梯度方向用于描述。没有 optimizer、backward 写入 `.grad` 或参数更新；没有声称原 AdamW 更新方向已被重放。

实际入口：[功能 v2](../diagnose_BrainHIVE_decisive_T_function_GPU_20261025_v2.py)；[精度、梯度和状态检查](../check_BrainHIVE_decisive_T_function_CPU_20261025_v2.py)。训练日志入口 v1 仍可复用，它只读取已闭合日志和最终头元数据，不调用 v1 梯度诊断。

## 3. 执行与判读边界

独立 [控制器 v2](../queue_BrainHIVE_decisive_T_analysis_20261025_v2.py) 使用新注册目录、新状态、新日志、新持久输出根和唯一 GPU job 名；原 v1 不覆盖或自动重试。仍须等待原四臂完整生命周期及固定比较闭合、原进程退出和生命周期锁，再运行日志→表征→功能→总判读。

仍保留 48 小时等待上限，阶段时限 1800/14400/7200/1800 秒，单 GPU、CPU/cgroup 和双盘余量保护，4.75GiB 总输出额度及失败完整保留规则。v2 没有自动启动新候选的分支。

总判读使用 [reader v2](../read_BrainHIVE_decisive_T_analysis_CPU_20261025_v2.py)。CPU 表征的独立正信号不会被 GPU 教师 UNKNOWN 抹掉；功能机制推荐不能跳过教师未定，公共偏移也不能被写成配对信息改善。

## 4. 复审可以确认到哪里

本次检查能核验源码逻辑、合成反例、真实 CPU 输入、源文件 SHA、队列配置互通和实际等待进程。**完整 GPU 诊断尚须在上游结束后运行**；7200 秒和 20GiB 是事前上限，不是全量已通过的测量。实际失败会保留原因和完整分母，不把失败或缺数据改成成功。

实时证据存入 [v2 audit 目录](../audit/decisive_T_analysis_20261025_v2/)。v1 的显式停止证据见 [retirement 回执](../audit/decisive_T_analysis_20261025_v1/retirement_for_review_corrections.json)。

## 5. 实际交付与重新挂队

2026-10-25 **09:36:35 America/Los_Angeles**（16:36:35 UTC），修正版独立控制器 **PID107402** 已实际启动，状态为 `WAITING_FOR_COMPLETE_FOUR_ARM_LIFECYCLE_AND_COMPARISON`。旧 PID105917 已退出且从未启动分析阶段。原 S 生命周期继续，未抢占其 GPU。

本地 89 项检查通过，3 项 Linux 进程检查因 Windows 环境跳过；服务器 **92 项检查全部通过、跳过数为 0**。其中 3 项真实 CPU 进程检查验证了：正常 TERM、wrapper 先退出但独立 child 仍活时的清理、忽略 TERM 后 KILL 并取得 -9 退出码。它们没有创建 GPU 上下文。

另外，服务器真实重放了 20 个 OOF 编码器的小批 T 数据、缓存一致性和完整归一化前重建，并核验固定 80 query 选样；这些不是完整表征分析或完整 GPU 运行。S 最终头尚未闭合时，四臂日志兼容检查保持待定，不能据此提前执行正式分析。

40 份执行依赖及四份子配置 SHA 已绑定，原 30 份科学执行源、退役 v1 的 39 份源均复查一致。新注册 SHA：`082802dc2cd8384107030a8f591ea940b4e4377239a1ec7f61b2cebac4f5e9ce`。

证据：[启动回执](../audit/decisive_T_analysis_20261025_v2/actual_launch_receipt.json)、[服务器检查](../audit/decisive_T_analysis_20261025_v2/server_CPU_checks_actual.json)、[真实进程清理](../audit/decisive_T_analysis_20261025_v2/actual_Linux_process_cleanup_checks.json)、[真实输入](../audit/decisive_T_analysis_20261025_v2/actual_input_CPU_preflight.json)、[最终在线复查](../audit/decisive_T_analysis_20261025_v2/actual_progress.json)。
