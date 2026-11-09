本文按2026-11-09本地已封存源码展开当前架构。原生Brain-HIVE、ordinary、EEG端A/B候选与新锚定C候选分别列出；它们是不同实验分支，不是全部同时串入一个已成功的模型。本文不修改训练、队列或任何检查点。

当前判断：ordinary有历史积极图片证据；新C是ordinary原则向EEG特征提取端的机制扩展；A/B/C尚未形成第二项已确认的独立创新。下面的数学性质不替代留出图片效果，也不构成整个非凸训练问题的闭式最优解。

**符号与输入。** 令原始EEG为 \(e\in\mathbb R^{63\times250}\)。当前重建使用同一刺激4次重复的物理平均R4，不能把它写成单次试验或R80。原生选17个后部电极，记 \(x=\operatorname{vec}(e_P)\in\mathbb R^{4250}\)；其余46电极记 \(e_N\in\mathbb R^{46\times250}\)。当前新实验的可用原始EEG是sub-03；不能由此声称十被试或正式R80验证已经完成。

\(I\) 是允许训练阶段的真实刺激图像；\(c\in\mathbb R^{1024}\) 是EEG条件；\(T(c)\in\mathbb R^{4\times2048}\) 是IP条件token。\(\odot\) 和除法对向量均逐元素运算。\(\operatorname{center}(v)=v-\mathbf1(\mathbf1^Tv)/1024\)。所有正常重建条件只来自当前query自己的EEG，不读取目标图、概念、字幕或图库答案。

**原生视觉教师：训练时提供目标。** 三路已有视觉表征为

\[
b(I)\in\mathbb R^{512},\qquad h(I)\in\mathbb R^{1024},\qquad v(I)\in\mathbb R^{1024}.
\]

它们分别来自CLIP-B/32、CLIP-H/14和作者的VAE视觉嵌入。这里的1024维VAE视觉嵌入不是后面512像素图像的 \(4\times64\times64\) 生成latent。

\[
q_I=W_b b(I)+a_b+W_h h(I)+a_h+W_v v(I)+a_v,
\]

\[
F(I)=\operatorname{LN}_{\gamma_F,\beta_F}
\left[q_I+W_{F2}\operatorname{GELU}(W_{F1}q_I)\right].
\]

其中 \(W_{F1}:1024\to4096\)，\(W_{F2}:4096\to1024\)。两个残差MLP矩阵无bias；三路输入投影有bias。原生EEG父模型拟合时，冻结教师权重但保留实际dropout=0.1的随机教师前向；后续生成器适配损失使用已核验的确定性F目标。上式是去掉训练dropout记号后的确定性前向。

**原生EEG编码器。** 实际BrainMLP为

\[
h_0=W_0x+a_0,\qquad
r_0=h_0+W_2\operatorname{GELU}(W_1h_0),
\]

\[
c_0=E_0(x)=\gamma\odot
\frac{r_0-\overline r_0\mathbf1}
{\sqrt{\operatorname{Var}(r_0)+10^{-6}}}+\beta.
\]

\(W_0:4250\to1024\) 有bias，\(W_1,W_2:1024\to1024\) 无bias。原生训练中GELU后有dropout=0.1；当前A/C生成器适配阶段为确定性eval前向，dropout关闭。总参数6,452,224，其中末层LayerNorm的 \(\gamma,\beta\) 共2048参数。

父编码器原生对齐使用对称InfoNCE。令 \(\bar c_i=c_i/\|c_i\|\)、\(\bar f_i=F(I_i)/\|F(I_i)\|\)、\(s_{ij}=\exp(\ell)\bar c_i^T\bar f_j\)，则

\[
L_{\mathrm{NCE}}=-\frac1{2n}\sum_i
\left[
\log\frac{\exp s_{ii}}{\sum_j\exp s_{ij}}
+\log\frac{\exp s_{ii}}{\sum_j\exp s_{ji}}
\right].
\]

\(\ell\) 是可学习logit_scale。原生父模型先做EEG—视觉表征对齐，没有直接对其EEG权重优化当前的UNet噪声重建损失。

**原生A、P与IP注意力。** 为避免名字混淆，以下A是作者已有的残差映射，不是我们的A_pair实验分支。

\[
A(c)=c+W_{A2}\operatorname{GELU}(W_{A1}c),
\qquad W_{A1}:1024\to4096,\quad W_{A2}:4096\to1024.
\]

A无bias、无LayerNorm。

\[
T(c)=P(A(c))=
\operatorname{LN}_{2048}\!\left[
\operatorname{reshape}_{4\times2048}(W_P A(c)+a_P)
\right],\qquad W_P\in\mathbb R^{8192\times1024}.
\]

P只有一次线性投影1024→8192，再重排为4个2048维token，最后逐token做LayerNorm；不是额外两层MLP。

在第 \(l\) 个IP注意力处理器、每个head内，令 \(H_l\) 为UNet当前空间特征：

\[
Q_l=H_lW_{Ql},\qquad K_l=T(c)W^{IP}_{Kl},\qquad V_l=T(c)W^{IP}_{Vl},
\]

\[
O^{IP}_l(c)=\operatorname{softmax}
\left(\frac{Q_lK_l^T}{\sqrt{d_l}}\right)V_l.
\]

它与原生文本分支注意力相加，再走作者原有输出投影、dropout及所在层的残差/缩放：

\[
H'_l=\mathcal R_l\left[W_{Ol}
\left(O^{text}_l+\lambda_{IP}O^{IP}_l(c)\right)\right],
\qquad\lambda_{IP}=1.
\]

这里 \(\mathcal R_l\) 只是原生层剩余处理的记号，没有新增学习模块。正常重建使用固定空文本条件；没有使用query目标字幕。70个IP处理器及A/P/SDXL-Turbo都是已有宿主。

**原生图像生成与我们共同使用的训练误差。** 记冻结条件UNet为 \(\epsilon_\psi\)。训练latent为

\[
z_0=0.13025\,\operatorname{mode}(\operatorname{Enc}_{VAE}(I)),
\qquad z_0\in\mathbb R^{4\times64\times64},
\]

\[
\epsilon\sim\mathcal N(0,I),\qquad t\sim\operatorname{Unif}\{0,\ldots,999\},\qquad
z_t=\sqrt{\bar\alpha_t}z_0+\sqrt{1-\bar\alpha_t}\epsilon.
\]

对任意EEG条件产生函数 \(C(e)\)，定义每query噪声能量

\[
\mathcal E_i(C)=\frac1{4\cdot64\cdot64}
\left\|\epsilon_\psi(z_{t_i},t_i,T(C(e_i)))-\epsilon_i\right\|_2^2,
\]

以及token误差

\[
L_{tok}(C)=\frac1{4n}\sum_{i,k}
\left[1-\cos\left(T(C(e_i))_k,T(F(I_i))_k\right)\right].
\]

共同正例目标为

\[
L_+(C)=\frac1n\sum_i\mathcal E_i(C)+0.1L_{tok}(C).
\]

A/P/IP/UNet冻结，不意味着停止其输入梯度；误差仍通过它们回传至训练中的ordinary或EEG编码器。只有视觉教师目标停止梯度。

正常生成为原生4步Euler ancestral，512×512、CFG=0、IP scale=1，每query一张。为说明采样器没有新增学习部分，设当前噪声尺度 \(\sigma_k\)、下一尺度 \(\sigma_{k+1}\)，原生epsilon预测形式可写为

\[
\hat\epsilon_k=\epsilon_\psi
\left(\frac{x_k}{\sqrt{1+\sigma_k^2}},t_k,T(c)\right),
\]

\[
\sigma_{up}=\min\!\left(\sigma_{k+1},
\sqrt{\frac{\sigma_{k+1}^2(\sigma_k^2-\sigma_{k+1}^2)}{\sigma_k^2}}\right),
\qquad\sigma_{down}=\sqrt{\sigma_{k+1}^2-\sigma_{up}^2},
\]

\[
x_{k+1}=x_k+(\sigma_{down}-\sigma_k)\hat\epsilon_k+
\sigma_{up}\xi_k,\quad \xi_k\sim\mathcal N(0,I).
\]

最后用原生VAE解码 \(x_4/0.13025\)。采样时间点、随机数消费和数值细节以冻结Diffusers/scheduler实现为准；上式是其epsilon形式的数学展开，不是本轮新采样算法。用 \(G_\psi(c;\xi)\) 简记整条原生生成链。

**ordinary：在既有条件内做有界的生成器适配。** 对 \(c=E_0(x)\)，先撤销真实EEG编码器的输出affine：

\[
u=(c-\beta)\oslash\gamma,
\qquad a_\phi(u)=V_2\operatorname{GELU}(V_1u+b_1)+b_2,
\]

\(V_1:1024\to256\)、\(V_2:256\to1024\)，共525,568个参数，第二层零初始化。令

\[
d_0=\operatorname{center}(a_\phi(u)),\qquad
d=d_0-\frac{d_0^Tu}{\|u\|^2}u,
\]

\[
\Delta=\rho\|u\|\frac{d}{\sqrt{\|u\|^2+\|d\|^2}},
\quad s=\frac{\|u\|}{\|u+\Delta\|},\quad\rho=0.25,
\]

\[
\boxed{O_\phi(c)=c+\gamma\odot[(s-1)u+s\Delta].}
\]

其标准化输出等于 \(s(u+\Delta)\)，保持标准化范数；零头给出原生条件。ordinary没有风险头或风险gate，不能把历史risk实验混进ordinary公式。

\[
\phi^*=\arg\min_\phi L_+(O_\phi\circ E_0),
\qquad \boxed{\hat I_O=G_\psi(O_{\phi^*}(E_0(x));\xi).}
\]

只有新增ordinary参数训练，原生EEG编码器及整个生成器冻结。新sub03匹配ordinary训练使用真实概念折外EEG条件和对应各折affine，头本身在全部T训练，因此整个组合不是全链OOF。固定Dfit网格只选择半径内全局强度，当前选1；历史ordinary权重不覆盖。

ordinary的贡献候选是“在真实宿主条件坐标中，以有界方向修正接受冻结生成器监督”。MLP、切空间投影或球面归一化各自不是全新数学工具；其独立机制价值需要与匹配的视觉目标、均值方向等对照区分。

**A：把生成器误差变成同概念EEG区分目标。** 对同概念的不同训练刺激，固定一个循环负例 \(j=\pi(i)\ne i\)。正负条件必须面对同一 \((z_{t_i},t_i,\epsilon_i)\)，仅EEG条件替换：

\[
\mathcal E_i^+(C)=\operatorname{mean}
\|\epsilon_\psi(z_{t_i},t_i,T(C(e_i)))-\epsilon_i\|^2,
\]

\[
\mathcal E_i^-(C)=\operatorname{mean}
\|\epsilon_\psi(z_{t_i},t_i,T(C(e_{\pi(i)})))-\epsilon_i\|^2.
\]

注意负例目标依然是 \(I_i\) 的同一个latent和噪声，不是换成负例自己的目标图。

\[
\boxed{L_{pair}(C)=L_+(C)+\frac{\tau}{n}\sum_i
\operatorname{softplus}\left(\frac{\mathcal E_i^+-\mathcal E_i^-}{\tau}\right).}
\]

\(\tau\) 是拟合前固定256条T、64概念原生能量差的RMS，下限 \(10^{-4}\)，各分支共用且不由Dcheck选取。A使用 \(C(e)=E_\theta(e_P)\)，训练原生编码器除末层affine外的6,450,176参数，没有新增推理模块。

对任意与EEG条件无关的加性能量项 \(b_i\)，\((\mathcal E_i^++b_i)-(\mathcal E_i^-+b_i)=\mathcal E_i^+-\mathcal E_i^-\)。这只消去条件无关的加性能量偏移，不能证明消除全部公共视觉特征。其配对项梯度为

\[
\nabla_\theta L_{contrast,i}=\operatorname{sigmoid}
\left(\frac{\mathcal E_i^+-\mathcal E_i^-}{\tau}\right)
(\nabla_\theta\mathcal E_i^+-\nabla_\theta\mathcal E_i^-).
\]

三个对照是：A_positive把配对项乘0；A_pair使用上式；A_reverse保留正确正例目标，仅把softplus内差值反向。A_positive仍执行匹配的正负图计算。

实际完整T训练诊断中，A_pair同概念Top1为82.8949%，A_positive为77.6342%，原生full-T为99.3802%。因此当前只能说配对项减轻了普通生成器训练造成的训练内区分损失。它尚不是留出图片增益，更不能称已成功的第二创新。

**B：在第一层接入额外46电极的条件残差。** 只由T EEG求逐通道/时点均值 \(\mu_P,\mu_N\)，再将所有T样本和250时间点汇总为空间协方差 \(C_{PP},C_{NP}\)。定义

\[
B^*=\arg\min_B\mathbb E_T
\|(e_N-\mu_N)-B(e_P-\mu_P)\|_F^2+\lambda\|B\|_F^2
=C_{NP}(C_{PP}+\lambda I)^{-1},
\]

\[
\lambda=10^{-3}\operatorname{tr}(C_{PP})/17,\qquad
r_N=(e_N-\mu_N)-B^*(e_P-\mu_P).
\]

修改原生第一层为

\[
h_B=W_0\operatorname{vec}(e_P)+a_0+U\operatorname{vec}(r_N),
\quad U\in\mathbb R^{1024\times11500},\quad U_{init}=0,
\]

\[
\boxed{C_B(e)=\operatorname{NativeResidualMLP\!+\!LN}(h_B).}
\]

仅U的11,776,000参数训练，旧编码器其余参数冻结。B_residual使用上述残差；B_raw令B=0、使用中心化的额外电极；B_shuffle仅训练时给正确后部电极配同概念错误刺激的残差，正常推理仍使用自己63电极。

它可精确折叠为一次63电极线性投影。在按P/N分组的展平坐标中，令 \(\mathcal B=B^*\otimes I_{250}\)：

\[
W_P'=W_0-U\mathcal B,\qquad W_N'=U,
\qquad a'=a_0-U\operatorname{vec}(\mu_N-B^*\mu_P).
\]

\[
h_B=W_P'\operatorname{vec}(e_P)+W_N'\operatorname{vec}(e_N)+a'.
\]

实际折叠再恢复原63通道顺序。折叠不代表参数量或计算量与17电极原生完全相同；也不证明端到端GPU加速。岭回归、残差化和折叠均是已有数学工具。残差不自动等于独立神经信息，尤其 \(-U\mathcal B\) 同时重新加权原17电极，因此还需要强63电极/后部重加权控制才能归因。

**C：当前新增的原生EEG锚定编码器。** 根据A的训练诊断，新C同时保留冻结的 \(E_0\) 和从同一父权重初始化的 \(E_\theta\)。后者训练，前者提供每条输入自己的参考：

\[
c_0=E_0(x),\qquad q_\theta=E_\theta(x),\qquad
u_0=(c_0-\beta)\oslash\gamma,\qquad
u_\theta=(q_\theta-\beta)\oslash\gamma.
\]

\(E_\theta\) 的末层 \(\gamma,\beta\) 固定为同一原生参数。这里的方向直接来自重新读取4250维EEG的编码器：

\[
d_0=\operatorname{center}(u_\theta-u_0),\qquad
d=d_0-\frac{d_0^Tu_0}{\|u_0\|^2}u_0,
\]

\[
\Delta=\rho\|u_0\|\frac{d}{\sqrt{\|u_0\|^2+\|d\|^2}},\quad
s=\frac{\|u_0\|}{\|u_0+\Delta\|},\quad \rho=0.25,
\]

\[
\boxed{C_\theta(x)=c_0+\gamma\odot[(s-1)u_0+s\Delta].}
\]

这与ordinary共享约束映射，但方向来源不同：ordinary是 \(a_\phi(c_0)\)，C是 \(E_\theta(x)-E_0(x)\) 在真实标准化坐标中的差。C没有新增可训练的c后MLP；它训练原生编码器已有的6,450,176参数，同时保留额外一套冻结父编码器。推理需要两个EEG前向，UNet仍四次。

其结构动机可精确写为

\[
W_0v=0\ \Longrightarrow\ E_0(x+v)=E_0(x)
\ \Longrightarrow\ O_\phi(E_0(x+v))=O_\phi(E_0(x)).
\]

而允许训练第一层后，\(W_\theta v\) 可以非零。因此C的函数类能利用只接收原生c的ordinary无法区分的某些输入变化。4250→1024的原生第一层核维度至少3226；核存在不证明自然EEG中确有可泛化的视觉信号。真实矩阵构造检查已验证通路区别，尚不是学到有用信息的证据。

约束性质的推导为

\[
u_0^Td=0,\qquad u_0^T\Delta=0,\qquad
\|\Delta\|\le\rho\|u_0\|,\qquad \|s(u_0+\Delta)\|=\|u_0\|.
\]

因此

\[
\tan\angle(u^+,u_0)=\|\Delta\|/\|u_0\|\le\rho,
\qquad \angle(u^+,u_0)\le\arctan(0.25)\approx14.04^\circ.
\]

令 \(y^+=u^+/\|u^+\|\)、\(y_0=u_0/\|u_0\|\)，有

\[
\|y^+-y_0\|\le\eta=
\sqrt{2-\frac2{\sqrt{1+\rho^2}}}\approx0.24437,
\]

\[
\left|\|y_i^+-y_j^+\|-\|y_{0i}-y_{0j}\|\right|\le2\eta,
\qquad
\left\|\frac1n\sum_i y_i^+\right\|
\le\left\|\frac1n\sum_i y_{0i}\right\|+\eta.
\]

这是精确算术、撤销真实affine后的标准化单位坐标性质；浮点实现另报告实测误差。它没有保证F检索排名、原始c均值能量、A/P注意力或生成图片质量，也没有保证保留每个刺激的全部信息。

C_positive最小化 \(L_+(C_\theta)\)；C_pair最小化 \(L_{pair}(C_\theta)\)；C_reverse只反转配对项。三组从同一原生父模型出发，各完整T一遍827更新、13230次query暴露、batch16末批14、AdamW lr=1e-5/weight decay=0.01/clip1、单seed1729，不根据Dcheck选epoch或强度。

候选整体可紧凑写为

\[
\boxed{\theta^*=\arg\min_\theta L_{pair}(C_\theta),\qquad
\hat I_C=G_\psi(C_{\theta^*}(x);\xi).}
\]

若再组合固定ordinary，则是另一个显式实验分支

\[
\boxed{\hat I_{C+O}=G_\psi(O_{\phi^*}(C_{\theta^*}(x));\xi).}
\]

新C不是在训练时自动接上ordinary联合优化；当前组合复用既有固定ordinary做交互检验。C后叠加ordinary不能继续声称整体相对 \(c_0\) 仍满足同一个14.04°边界。

对任意固定评价量M，组合增量需实际计算

\[
\Delta_{int}=M(C+O)-M(C)-M(O)+M(parent).
\]

接口可组合不等于效果独立或收益相加；只有对照结果才能判断。

**创新应如何计数。**

| 内容 | 改动位置 | 当前可以说什么 | 不能据此说什么 |
|---|---|---|---|
| 原生E/F/A/P/IP/Turbo | 继承宿主 | 基模型与教师/生成器 | 我们重新发明的组件 |
| ordinary | 编码器输出c之后 | 有界宿主适配的贡献候选，有历史积极图片线索 | MLP或投影公式本身首次提出、跨宿主均已成立 |
| A_pair | EEG编码器的生成器训练目标 | 检验同概念配对是否改善刺激区分 | 普通对比学习公式就是第二原创、已有留出图片成功 |
| B_residual | 第一层输入与46额外电极 | 检验原生输入选择遗漏的条件增量 | 岭回归/残差就是创新，已证明独立神经信息 |
| C锚定编码器 | 原始EEG特征提取及逐输入函数约束 | ordinary的机制扩展候选，有结构通路与边界推导 | 已确认第二独立创新、14°保证图像不退化 |
| GPU/折叠/缓存 | 实现与计算 | 已实现的等价工程可另测速度 | 未测端到端收益就列第三独立贡献 |

“约束微调时表示漂移”已有 [Better Fine-Tuning by Reducing Representational Collapse](https://arxiv.org/abs/2008.03156) 等近邻；“扩散重建监督可能损害区分能力、引入对比目标”也已有 [DCR, CVPR2026](https://arxiv.org/abs/2603.04803)。这些工作不是我们当前EEG实现的逐项复刻，但足以说明一般问题和工具不能冒称首次提出。本文没有完成穷尽性新颖性检索。

目前最可检验的方法主张是：**让生成器监督更新EEG特征提取，同时用同输入原生神经条件限制漂移，并用同概念错配EEG检验是否真正保留刺激级条件作用。** 这一组合若在完整留出图片上胜过普通微调、反向控制、ordinary及匹配强基线，才值得发展为更强论文贡献。

**实现与证据对应。**

- 原生作者类：[BrainMLP](https://github.com/ssshamiii/Brain-HIVE/blob/f06db69fcc4a029439e369c5c3dba6099e57560e/main/models_brain.py)、[F/残差MLP](https://github.com/ssshamiii/Brain-HIVE/blob/f06db69fcc4a029439e369c5c3dba6099e57560e/main/layers.py)、[P/IP](https://github.com/ssshamiii/Brain-HIVE/blob/f06db69fcc4a029439e369c5c3dba6099e57560e/main/models_adapter.py)。
- 实际宿主加载：[PKU_native_generator_20261109_v253.py](../PKU_native_generator_20261109_v253.py)。
- ordinary：[brainhive_bounded_component_v1.py](../brainhive_bounded_component_v1.py)、[当前匹配训练](../PKU_train_matching_ordinary_GPU_20261109_v259.py)。
- A/B结构：[PKU_eeg_candidate_core_20261109_v266.py](../PKU_eeg_candidate_core_20261109_v266.py)、[目标/训练](../PKU_train_eeg_candidates_GPU_20261109_v267.py)。
- C结构：[PKU_anchored_encoder_core_20261109_v286.py](../PKU_anchored_encoder_core_20261109_v286.py)、[目标/训练](../PKU_train_anchor_GPU_20261109_v286.py)、[固定注册](../configs/EEG_anchored_encoder_study_20261109_v286.json)。
- 已保存实际状态：[v298交接](../audit/EEG_ACTIVE_EXECUTION_HANDOFF_20261109_v298.json)。截至该回执六组A/B已完成拟合；新C数学/真实输入检查通过、完整训练与评价已排依赖队列，不能把CPU检查写成C训练收益。本文未重新查询实时作业状态。

上面展开的是模型与训练目标。当前只有B的岭回归子问题有经典闭式最优解；ordinary/C的约束映射有显式可行公式；EEG网络与生成器损失的联合适配仍是非凸梯度训练，没有整个问题的解析最优解，也没有由公式推出SOTA或录用结论。
