# ordinary固定：独立研究采样协方差

> 本页以下为D25当时的历史方案。用户随后要求创新更贴合EEG，已撤下D25作为主创新及后续80图计划。工程结果保留，未运行研究图片效果；当前转向EEG测量与原生注意力/输入投影的机制检验，ordinary继续固定。详见BRAINHIVE_EEG_REPEAT_RESPONSE_INVESTIGATION_20261028_v1.md与BRAINHIVE_EEG_COVARIANCE_IDENTIFIABILITY_20261028_v1.md。


2026-10-28，America/Los_Angeles。按用户要求，保留ordinary这条贡献线，固定原检查点、源实现和强度1；不再修改其损失、权重、半径或校准。第二条研究线改生成器的随机采样过程，可以分别关闭、独立使用和组合。当前已经实现并完成真实工程验证，尚无新采样器的研究图片收益结论。

## ordinary保留什么

- 原检查点SHA：`a66869e5676ccea27f326f4f0b86cd446526b42460734905990758e49e11cc03`。
- 原组件代码SHA：`2b93c0644bde268ae876f0e2b722106852b25910fedc1cffc97820e96cadf5d7`。
- 强度1，原半径0.25；没有重新训练、重校准或搜索强度。
- 全16600条旧开发结果：DINO0.768191→0.778066，SEED0.130608→0.136787。它支持小幅开发收益，但没有通过原定实用门槛，也不自动构成原创性或独立确认。

[冻结约定](../configs/ordinary_frozen_base_20261028_v1.json)保存用户指令和固定身份。把ordinary作为论文的一条贡献候选，并不要求后续改动继续围绕它添加新头。

## 两部分怎样分别存在

```text
EEG → 原BrainMLP → [ordinary可关闭] → 原A/P/IP → [新采样规则可关闭] → 图片
```

| 条件修正 | 随机采样 | 用途 |
|---|---|---|
| 关闭ordinary | 原采样 | 原生基线，已有完整缓存 |
| 固定ordinary | 原采样 | 固定强参考，已有完整缓存 |
| 关闭ordinary | 新采样 | 检验新规则能否单独工作 |
| 固定ordinary | 新采样 | 检验叠加后是否还有图片增量 |

“独立”指实现和开关可分离、无需共同训练，不是预设效果统计独立或一定相加。新采样器只接收标准IP tokens，不导入ordinary实现，不需要知道条件来自哪个修正头。四组结果还要报告交互项$[(O+S)-O]-[(P+S)-P]$，不能拿叠加均值改善当成二者独立贡献。

## 第二部分的底层逻辑与闭式解

原生Euler ancestral在每步先计算条件预测对应的更新，再注入高斯噪声：

$$z_{t-1}=\mu_{\mathrm{native}}(z_t,c)+\sigma_{\mathrm{up},t}\eta_t,\qquad\eta_t\sim\mathcal N(0,I).$$

本候选不修改给定当前状态和条件时的原生预测返回值，也不修改`scheduler.step`或它的签名。在相同UNet输入和时间点，另算一次“缺失原始1024维条件”的参考响应，定义每个latent位置的四通道方向：

$$d_t=\epsilon_\theta(z_t,t;c)-\epsilon_\theta(z_t,t;0),\qquad v_t=d_t/\|d_t\|.$$

这里的0是A输入的原始条件零向量，不是零EEG，也不是已训练的真正无条件score。$d_t$只是原生条件响应差，不是EEG生理信号、信息量或真实似然梯度。

候选约束随机扰动满足$v_t^\top\delta_t=0$，同时保持每个四通道位置的**期望**噪声能量为4。在这个三维子空间上，最大熵高斯的三个非零协方差特征值相等，故

$$\boxed{\Sigma_t=\frac43(I-v_tv_t^\top),\qquad
\delta_t=\sqrt{\frac43}(I-v_tv_t^\top)\eta_t.}$$

这个解来自固定trace下的三维协方差行列式最大化。不是把每次噪声实现的范数强行调成相同，也没有新增可训练组件。若$d_t=0$，约束为空，精确返回原始$\eta_t$；显式关闭新采样器也直接走原路径。

ordinary改变传入的条件；新规则改变随机转移的协方差。组合后未来状态会变化，因此不能把“当前状态的均值函数未改”扩大成“整个生成轨迹或最终均值不变”。协方差改变了采样分布，这不是精确EEG后验，也不保证保存的响应方向对应正确视觉细节。

## 为什么需要错误方向控制

使用固定的四通道反对称正交变换$J(x_1,x_2,x_3,x_4)=(-x_2,x_1,-x_4,x_3)$，有$Jv\perp v$且范数相同。错误方向组用$Jv$替代$v$，保持相同协方差谱、trace、额外UNet次数及基础高斯随机数。

如果正确方向没有胜过这个控制，就不能把结果归因于保护了条件响应；可能只是改变随机过程或有效维度。该控制不会修改ordinary或使用query目标图。

## 实际验证已经完成什么

本地和服务器九项数学检查通过，包含trace=4、rank=3、采样协方差、错误方向正交、零响应／关闭精确回退、不额外消耗RNG等。[本地检查](../audit/conditional_noise_geometry_20261028_v1/local_math_checks.json)。

真实GPU工程包含八张图片：原生／ordinary各自的等方差原采样、保护方向和错误方向，以及ordinary的人工零响应与显式关闭。四种应回退的路径与已有原生／ordinary缓存的文件字节和像素完全一致。全部400条原生／ordinary条件也用原始full编码器和冻结ordinary独立重算，最大相对误差为9.387e−7／9.187e−7。

独立CPU核对八张图片和28次实际噪声变换，NumPy重算最大相对差为1.569e−5。FP16返回后的方向点积残差约8e−5，而原始各向同性噪声约0.5；这是数值实现诊断，不能当成图片收益。

GPU程序67.695秒，包装69.826秒；CPU核验9.026秒，控制器全段150.731秒。共60次UNet调用、8张工程图，新增训练与研究图片均为0。全部ordinary身份在前后重新核验未变。[真实profile](../audit/conditional_noise_geometry_20261028_v1/actual/candidate/profile/completion.json)、[独立CPU核验](../audit/conditional_noise_geometry_20261028_v1/actual/server/profile_CPU_audit/summary.json)、[实际闭合](../audit/conditional_noise_geometry_20261028_v1/actual/server/profile_scientific_closure.json)。

正常新采样仍为四个去噪时间点，但每点多一次参考UNet前向，成本为8次UNet，原基线为4次；不把它称为相同推理成本。正确／错误方向两组的计算量相同，没有额外采样种子或best-of-N。

## 下一项固定图片比较

保留原生、ordinary、risk三组参考缓存，新增“原生+新采样”“ordinary+新采样”“ordinary+错误方向”三臂，固定80→400和同一400目标图库。主比较事前固定为ordinary+新采样相对ordinary；同时完整报告另外三组组合关系和交互项。ordinary权重、强度、EEG编码器、A/P/IP/UNet/VAE均不更新。

80条只按既有v2明确损害规则提前停止，其余扩400；不因工程验证通过或方向非零宣布成功。400有支持后才讨论更多被试、seed、成本匹配与完整评价。

[NoiseCtrl](https://openaccess.thecvf.com/content/CVPR2025/html/Dai_NoiseCtrl_A_Sampling-Algorithm-Agnostic_Conditional_Generation_Method_for_Diffusion_Models_CVPR_2025_paper.html)已有条件噪声研究，[APG](https://arxiv.org/abs/2410.02416)已有投影导引研究。高斯投影和最大熵公式也不是原创定理。当前可检验的问题是：在冻结ordinary的Brain-HIVE中，不加导引漂移、保持期望噪声能量、只控制局部条件响应方向上的随机协方差，是否带来可独立复现的图片增量。它目前是研究候选，尚未建立第二条创新贡献。
