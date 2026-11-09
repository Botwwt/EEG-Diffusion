# 已保存图片的空间完整性补充诊断

## 全量完成更新：2026-10-21 11:16 PDT

risk 的全部 16600 张原图已完成扫描，并用独立程序逐行连接原 manifest、首次表和补齐表，复核 ID、顺序、路径、文件 SHA、像素 SHA 及完整分母。十个被试各 1660 条、共 1660 个刺激。结果如下：

| 检查 | 全量结果 |
|---|---:|
| 文件 SHA 核验通过且可解码为 512×512 RGB | 16600/16600 |
| 未处理或空间标记未知 | 0 |
| 缺失、损坏、哈希或尺寸错误 | 0 |
| 精确空间恒色 | 0 |
| 近空间恒色（最大通道空间标准差≤1/255） | 0 |
| 旧全 RGB 空白代理命中 | 0 |
| 解码像素完全重复 | 0 |

首次作业达到 900 秒预算时，完成 14743 条，其余 1857 条明确标记为 `not_processed_time_limit`；没有从分母中删除。新增[一次性补齐入口](../finish_BrainHIVE_spatial_integrity_CPU_v1.py)先核验旧进程已结束、原注册/源码/manifest/结果 SHA 及逐行身份，只读取这 1857 张尚未处理的图片，耗时约 114 秒。原 14743 行逐字段不变，首次表和摘要字节不变；补齐输出写入独立目录，拒绝再次续跑或覆盖。服务器 CPU 契约已检查只打开未处理图片，以及源漂移、活动进程和重复续跑拒绝。见[实际契约](../audit/BrainHIVE_spatial_finish_CPU_contract_server_actual_20261021_v1.json)。

两段累计 1014.34 秒 CPU；没有导入训练模型、占用 GPU、生成新图或重算语义分数。全量逐 query 表留在服务器：`/root/brainhive_reproduction_20261020_v1/audit/spatial_image_integrity_risk_CPU_v1_finish_once_v1/per_query.csv.gz`。本地只保存小型 JSON 回执，不搬运大文件。详见[合并摘要](../audit/BrainHIVE_risk_spatial_integrity_full_summary_actual_20261021_v1.json)及[独立闭合复核](../audit/BrainHIVE_spatial_integrity_risk_full_closure_actual_20261021_v1.json)。

因此，这批 risk 图片未观察到恒色/近恒色塌缩、文件损坏或像素重复。这些定义不能识别所有纹理噪声，也不能评价对象正确、自然度或感知质量非劣；紫色纹理噪声发生率仍未判定。新组件是否优于原生 HIVE，仍取决于完整同协议图片比较。同期普通对照为 8384/24807 更新、第二遍数据，20 分钟巡检最近第七次记录无告警；原主配方、检查点及固定源码未改。

以下保留扫描运行中的历史快照。

2026-10-21 11:00，America/Los_Angeles。本轮发现并验证了旧空白代理的解释缺口，尚未发现实际 risk 图片存在同一问题。主训练、主指标、训练 seed、模型和数据划分不变。

## 已证实的检查缺口

[image_evaluation.py](../image_evaluation.py) 的 `quality_statistics` 对整幅 RGB 数组求标准差，以 `<0.01` 标记空白代理；[score_low_level.py](../score_low_level.py) 的 `generated_pixel_std` 也混合了空间与通道差异。各通道空间恒定、但通道均值不同的图仍可具有较高全局标准差。

服务器上的固定合成反例为整幅 RGB=(255,0,255)：旧标准差约 0.4714045，旧空白标记为 false；各通道空间极差和标准差都为零。新增检查正确识别该反例，并通过了渐变、棋盘格、缺失、损坏、哈希错误、尺寸错误、不同文件编码但相同像素、完整分母及一个灰度级边界检查。见[实际服务器 CPU 回执](../audit/BrainHIVE_spatial_image_integrity_CPU_contract_server_actual_20261021_v1.json)。

这限制的是旧空白代理的解释范围，不使既有 DINO、SEED、ObjectF1 或配对结果失效。没有修改旧评价源码或覆盖旧结果。

## 扫描前固定的定义

[注册文件](../configs/BrainHIVE_spatial_image_integrity_CPU_v1_preregistration.json) 已于实际扫描前提交：原始 512×512 RGB，不缩放；逐通道空间极差均为零时标记精确恒色，最大逐通道空间总体标准差不超过 1/255 时标记近恒色。保留旧代理作比较，另记录像素 SHA、通道均值、黑白饱和比例、邻像素差分及拉普拉斯数值。

这些都是工程标记和连续诊断，不能自动识别紫色噪声、判断对象正确、证明自然度或感知质量非劣。人工评价未开展，感知非劣容忍界仍未确定。

## 实际执行与未完成项

已有 risk 的 16600 条 manifest 由固定主配方及原配对回执解析、核验 SHA；不猜路径、不读目标图或语义得分。新增[只读 CPU 扫描](../audit_BrainHIVE_spatial_image_integrity_CPU_v1.py)的真实 PID 为 215509，最多 900 秒、输出上限 16 MiB，独立新目录，全部 query 保留在分母。

11:00 快照：4000/16600 条已核验文件 SHA 并成功解码，进程仍存活，未取得全量空间标记结论。前 4000 条用时约 245 秒，线性外推超过当前 900 秒上限；若触发预算保护，须保留未处理条目为 unknown，并在原 manifest 下补齐后才能声称完成。预算停止不是模型方法失败。

同期普通对照为 7552/24807 更新；20 分钟巡检已完成六次，最近区间增加 1024 次更新、无告警、主生命周期空间检查通过。原主源、配方、risk 检查点及八个功能监督 v2 源文件保持不变。见[运行回执](../audit/BrainHIVE_spatial_integrity_audit_running_20261021_v1.json)。

后续对普通组与原生组应用相同检查。组件的实际图片增益、机制独立增量、感知质量与可比 SOTA 均尚未确认。
