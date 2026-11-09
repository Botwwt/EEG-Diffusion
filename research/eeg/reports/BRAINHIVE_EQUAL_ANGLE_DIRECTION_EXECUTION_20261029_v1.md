# ordinary 等角度方向干预：实际执行

2026-10-29。上一轮完整400的损失对照支持保留生成监督配方的同概念区分线索，但DINO优势未明确；完整T方向分解又显示总体分量较大。本轮只验证总体方向和样本变化方向的作用，不新增训练或声称新方法贡献。

原ordinary保持SHA a66869e5…9e11cc03、强度1、半径0.25。总体均值使用此前全部132300个OOF-T条件的原head中心化方向；Dcheck目标F、图像和评分不参与均值或新条件构造。当前条件c和旧ordinary输出均为真实EEG推理缓存，不是图像评价答案。

对每个query，撤销真实gamma/beta得到u和旧ordinary的v，取实际角度theta=angle(u,v)。q分别取Pi_u(mu_T)和Pi_u(d0(u)−mu_T)。新条件为

\[
u'_q=\cos\theta\,u+\sin\theta\,\|u\|q/\|q\|,\qquad c'_q=\beta+\gamma u'_q.
\]

每个分支保持旧ordinary的修正角度和标准化范数，减少幅度差异的解释。若分量范数≤1e−12或角度≤1e−12，明确回到该条旧ordinary条件并记录。经典投影和球面公式不作为原创。两个控制仍使用当前EEG几何和原head幅度，不能叫完全EEG无关的基线。

本机13条合成几何和两组零分量回退检查通过；服务器全部400×2条件CPU导出、另一种归一化步公式核验通过，最大绝对误差2.34e−7。CPU原head与旧GPU缓存相对误差2.05e−7，不冒充逐位相同。

旧卡控制器309775/start91786221实际启动。18:38:33UTC确认存活，GPU子进程309970，80阶段已保存143/160张新图；此前parent与ordinary参考图片逐像素及噪声回放通过。没有新训练或图片效果结论。

固定D09单被试物理R4、40概念×10刺激、同一400图库；512原生四步、CFG0/IP1、每query单图，同初始及四次ancestral噪声。每组80条只因相对ordinary的DINO≤−.02且区间上界<0提前停止，其余扩400。主对比为完整ordinary减总体方向，变化方向剔除为固定次级对比。区间跨零不称等价，也不预设总体分量占能量就必然保住图片收益。

400扩展用80→240→400两个有限生成分段，减少重复宿主加载。每GPU作业仍限900秒、GPU20GiB/额外CPU10GiB，全部输出2GiB及缓冲4GiB。第二卡未接触，旧正式D及此前完成的控制器不恢复。

最新证据以[交接v18](../audit/AR_active_execution_handoff_20261029_v18.json)、[实际派发](../audit/crossdisciplinary_20261028_v1/proxy3_old_HIVE_equal_angle_direction_actual_launch_v1_20261029T183611721440Z.json)和[实际进程观察](../audit/crossdisciplinary_20261028_v1/HIVE_equal_angle_latest_RAM_watch.json)为准。当前没有SOTA或CCF A结论。
