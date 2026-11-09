# 完整配对画廊浏览器核验补充

2026-10-19（America/Los_Angeles）。补充[此前材料报告](IDEA1_CONDITIONAL_REALPAIRED_GALLERY_20261019.md)：原Chrome启动失败及当时报告保持原样；本次使用已有Edge 155.0.4283.62完成相同检查，没有安装浏览器、重建画廊、重新生成或评分。

## 实际通过的检查

1. 完整16,600 query、十被试、166概念、415页及缺失对照提示。
2. 翻页、被试/概念/刺激筛选、固定首query及三张原图链接。
3. 单被试最后一页及翻页边界。
4. 415个静态分页链接存在，实际打开首个静态页。
5. 全部提升/下降案例CSV链接可用；底层文件已由全量材料审计核验各16,600行。
6. 完整盲评材料隐藏方法与自动分数，被试筛选可用。

没有JS错误、失败请求或外部站点请求。新旧检查器在参数改名后，全部assert的AST一致；未删减检查来通过。浏览器核验耗时48.75秒，两张截图共239,662字节，已实际查看。本地没有完整图片库、数据集或模型下载。

证据：[实际浏览器报告](../audit/IDEA1_Conditional_RealPaired_gallery_browser_QA_v2/report.json)、[检查器](../check_idea_multi_gallery_browser_v2.py)、[完整材料机械核验](../audit/IDEA1_Conditional_RealPaired_gallery_materials_check_v1.json)。首次静态页并不代表逐页人工检查；全量文件配对及覆盖来自独立材料审计。

## 固定首样本与原始配对

首样本由稳定字典排序固定为 `sub-01/acorn_01b.jpg/R4`，没有按分数或生成种子选择。原生数组行是30，概念是`00004_acorn`，刺激是`acorn_01b.jpg`；目标及评分目标均为服务器gallery中的`00000030.jpg`。两组同一图像种子，生成文件路径不同。

| 方法 | DINO2AFC | DINO Top1 | ObjectF1 | SEED |
|---|---:|---:|---:|---:|
| RealPaired | 0.638336 | 0 | 0.014925 | 0.124510 |
| Conditional | 0.964436 | 0 | 0.071429 | 0.118820 |

查看截图：目标是橡果，两组生成均为长杆状物体，没有重建出目标对象。这个固定案例说明高DINO2AFC不能直接解释为对象正确；一个案例不能估计总体对象准确率，也不改变完整16,600条比较结论。

实际重查了EEG刺激、概念、数组行、目标路径、生成路径、评分目标、JPEG SHA及gallery原图硬链接，并与原逐query评分一致。详见[配对凭据，包含完整服务器路径](../audit/IDEA1_Conditional_RealPaired_gallery_browser_QA_v2/first_stable_pair_provenance_v1.json)。该额外只读核验耗时0.420秒，没有修改图片或指标。

[固定首query截图](../audit/IDEA1_Conditional_RealPaired_gallery_browser_QA_v2/first_stable_query_pair.jpg) · [完整画廊首屏截图](../audit/IDEA1_Conditional_RealPaired_gallery_browser_QA_v2/unblinded_first_stable_page.jpg)

## 边界

临时服务仅监听服务器127.0.0.1，通过加密SSH转发访问，已停止并核对进程不存在；[服务凭据](../audit/idea_matched_gallery_private_HTTP_v2.json)。浏览器及本地转发均已正常退出。

没有开展人工评分，不声称画质非劣、机制有效或SOTA。当前双列画廊的Frozen等缺项仍明确展示；Frozen正在独立完成全量评价，完成后需要新版本合并，不能覆盖该历史材料。
