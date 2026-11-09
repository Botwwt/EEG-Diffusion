# 原生末步条件的视觉诊断与EEG训练工程结果

2026-10-29 12:41，America/Los_Angeles。固定80条视觉诊断、160张新图和80条ordinary回放完成；独立图片/统计核验及本机290份文件SHA接收完成。后续真实T8工程检查通过，四次临时更新已丢弃。**视觉诊断使用查询真实图像特征，不是EEG方法收益；尚无新EEG模型的正式图片结果。**

ordinary的前三步保持原样，只在第四次UNet前向将IP条件换为正确视觉F或同概念循环错配F。原生scheduler、四步、512、CFG0、IP1和每query单图不变。面板为旧D09的sub-01物理R4、8个完整概念共80条，图库仍为400张。

| 条件 | DINO2AFC | 400图库Top1 | 同概念十选一Top1 |
|---|---:|---:|---:|
| 原ordinary | 0.858083 | 7.50% | 23.75% |
| 末步正确视觉F | 0.957206 | 23.75% | 38.75% |
| 末步同概念错配F | 0.939818 | 6.25% | 15.00% |

正确F减ordinary的DINO差+0.099123，描述性95%概念区间[+0.077468,+0.120397]；同概念Top1差+15个百分点，区间[+8.75,+21.25]。正确F减同概念错配F的同概念Top1差+23.75个百分点，区间[+15.00,+31.25]。预先登记的接口支持门槛通过。错配F也能大幅提高整体DINO，却降低精确刺激区分，说明整体相似度不能代替细节验证。

这个实验支持“ordinary生成前三步后，末步条件仍能影响刺激细节”。它没有证明这些视觉细节可由EEG预测，没有给新EEG训练自动放行400或官方任务，也没有证明新颖性或SOTA。

证据：[全部80分数](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/visual/score80.json)、[预先登记](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/visual_protocol/protocol.json)、[独立图片及统计](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/visual/independent_CPU_audit_v2.json)、[本机接收及另一种bootstrap复算](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/receiving_completion.json)。

80条ordinary回放全部逐像素匹配旧图；三分支的全部四次UNet输入、初始及ancestral噪声一致，前三次IP token一致，全部原生K/V调用匹配实际条件。160张新图的身份、路径/inode、原始像素SHA和F目标行均检查。独立排名及bootstrap最大统计差2.78e-17，本机另一种einsum实现最大差5.56e-17。没有另行重跑视觉编码器。

必须保留数值边界：CPU半精度scheduler回放240次均未通过原设绝对1e-3/相对1e-5严格线，最大绝对误差0.001953125、相对0.000447795；原失败与诊断保留，没有放宽线或重出图。随后在原GPU上用保存的prefix和epsilon调用真实scheduler.step，**240次全部逐位一致**，没有额外UNet调用。CPU与GPU运算后端差异不能被写成CPU严格检查通过。独立图像和评分检查不依赖这项CPU末步近似。

## 训练改动的实际位置

候选冻结原parent+ordinary前三步，另复制同一个原生BrainMLP负责末步条件；冻结其输出LayerNorm的gamma/beta，保留ordinary的真实坐标及其原权重、强度1、半径0.25。训练已有输入层和残差MLP共6,450,176参数，推理仍四次UNet，另多一次小EEG编码器前向。没有新适配器或采样算法。

令冻结前三步生成状态为z，末步sigma=0.6932054162。连续算术下，epsilon型Euler终步sigma_next=0时为

\[
z_{\rm final}=z-\sigma\epsilon_\theta(z/\sqrt{1+\sigma^2},249,P(A(O(f_\phi(E))))).
\]

因此终点MSE对应噪声目标(z-z0)/sigma，且带sigma平方权重。**实际代码调用真实scheduler.step，不用此简式代替FP16路径**；对浮点cast的autograd是连续近似，不是量化图像文件的数学导数。

两臂比较拟采用相同模型、初始化、顺序、优化器、更新数及末步损失，仅训练状态来源不同：A为冻结ordinary真实前三步状态；B为同sigma下真实T latent的正向高斯加噪。两者均用真实原生末步终点MSE加0.1 token角误差。它检验状态/目标匹配，而不是在不同loss单位间直接排名。

## 已完成的真实工程检查

独立旧卡作业314001已完成退出。8条真实T输入经原parent和ordinary生成原生prefix，未生成研究图片。零更新、梯度启用的单query末步逐位回到原生终点；EEG输入权重梯度范数0.987316，后续输入层和两个残差权重梯度均有限非零。

两臂各两次batch8临时更新，AdamW lr1e-4/wd0.01、clip1、固定loss-scale1024，输出LN不动。两臂初始化摘要相同，原parent/ordinary/A/P/UNet/VAE权重逐张量摘要不变。临时权重没有保存或进入研究拟合。单步0.538–0.870秒；峰值GPU allocated18,990,017,024字节（17.69GiB）、额外CPU峰值9,529,389,056字节（8.87GiB），低于原20/10GiB上限。整个工程程序58.813秒，8条完整四步生成2.559秒；这些不是完整T训练耗时保证。

证据：[完整工程结果](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/profile/completion.json)、[240次真实GPU末步重放](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/profile/all240_GPU_native_terminal_replay.json)、[工程注册](../audit/HIVE_terminal_visual_and_EEG_profile_complete_20261029_v1/profile_protocol/protocol.json)。

后续完整T拟合必须另登记和预算；当前工程注册没有放行正式拟合。所用父编码器已在T训练，后续也只能称T监督warm-start，不能写成全链OOF。Dcheck只用于冻结的图片评价，不能据此选择epoch或强度。分阶段条件和截断采样反传已有近邻，不能以链式法则或末步训练本身主张原创。

全部GPU只用旧卡，第二卡未接触，原正式D保持停止。原ordinary检查点SHA仍为a66869e5…9e11cc03。当前没有新EEG图片收益、独立确认或CCF A/SOTA结论。

## 工程闭合后的完整T开发队列

2026-10-29 12:48另行完成真实源文件、全部输入SHA和资源范围核验，签发[新的单被试开发注册](../audit/HIVE_terminal_fullT_registration_20261029_v1/protocol.json)及[实际范围检查](../audit/HIVE_terminal_fullT_registration_20261029_v1/actual_scope_review.json)。它不修改旧官方准入标记，也不复用临时工程权重。

控制器314506/start92219247已实际运行：7段冻结prefix缓存覆盖全部13,230条T，再顺序训练real_prefix/Gaussian两臂，各一遍1,654更新、13,230次暴露，batch8末批6。两臂同初始化、同固定顺序、AdamW1e-4/wd0.01、clip1、末轮模型；不读取Dfit/Dcheck或官方分数来选模型。每任务900秒、13个任务；输出cap4GiB，含余量保留8GiB，注册时空闲约205GiB。有限上限不是实际耗时保证。

12:54:10实际观测：第一段2,048条prefix完成，第二段904/2,048，控制器、当前包装及GPU进程均存活，无错误。批量8生成prefix的数值不声称与逐query单独采样逐位一致；结构、权重、scheduler和每query种子均固定。该队列尚未完成两臂训练，没有新EEG图片或收益读数。后续须检查全量覆盖、优化器与最终状态，再另锁正常EEG图片比较。见[最新交接v20](../audit/AR_active_execution_handoff_20261029_v20.json)。
