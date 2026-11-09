# ordinary 两项完整 T 损失对照：实际执行

2026-10-29。原 ordinary 的完整 R80 图片比较已通过预设实用门槛；本轮核验它的收益是否可由更廉价的视觉对齐解释。两项均是机制对照，不作为新创新点或 SOTA 结果。

两组均保留原 525568 参数受限修正、真实宿主输出 affine、半径0.25与零方向初始化；只改变训练目标：

\[
L_F=0.1\operatorname{mean}[1-\cos(c',F(I))],\qquad
L_P=0.1\operatorname{mean}[1-\cos(P(A(c')),P(A(F(I))))].
\]

原 ordinary 的目标为噪声 MSE 加0.1token误差。三者的数据、初始化方法、优化器、query顺序和更新预算相同；新对照不运行UNet/VAE，**计算量没有匹配**。不能由这个实验排除额外生成器计算的所有解释。

每组全部十被试132300个真实OOF-T条件、完整三遍、396900次暴露、24807次AdamW更新；batch16、末批12、seed1729、lr1e-4、weight decay0.01、clip1。三个pass顺序SHA实际与原ordinary一致。最终强度只按原Dfit8200的固定token角误差网格选择，不读Dcheck或官方目标来拟合、选epoch或调强度。固定最后检查点。

旧卡控制器305234已实际启动。18:06:30 UTC观察：F-only已完成24807更新，训练循环188.683秒；token-only在19200更新。各段保存模型、Adam、CPU/CUDA RNG和全量覆盖，旧ordinary及冻结A/P指纹保持一致。此处时间是实际已完成训练循环，不能代表首次实现、加载和全轮图片验证。

图片控制器306675已实际启动并确认存活，等待同一训练控制器成功退出。固定D09单被试物理R4、40概念×10刺激、同一400图库、512原生四步、CFG0/IP1、每query单图。每段首先回放旧parent与ordinary首图的像素及全部采样噪声。80条按原明确损害门槛逐组停止，其余扩400，不把普通对照未显著差异称为等价。当前新研究图和效果结果为0。

两次派发前错误保留：训练派发脚本字符串引号错误在启动进程前退出；首版图片注册引用了错误位置的R80程序，在创建图片控制器/CUDA之前退出。修正只处理脚本与依赖位置，未改变训练、面板、种子、损失和门槛。第二卡未接触，原正式D未恢复。

实际状态以[最新交接](../audit/AR_active_execution_handoff_20261029_v16.json)、[训练派发](../audit/crossdisciplinary_20261028_v1/proxy3_old_HIVE_two_full_T_loss_controls_launch_v1_fixed_20261029T175911963885Z.json)、[图片派发](../audit/crossdisciplinary_20261028_v1/proxy3_old_HIVE_loss_control_images_actual_launch_v2_20261029T180545300546Z.json)和[进程观测](../audit/crossdisciplinary_20261028_v1/proxy3_old_HIVE_loss_controls_actual_progress_v4_20261029T180629545485Z.json)为准。尚无新机制胜出、独立确认或CCF A贡献成立的结论。
