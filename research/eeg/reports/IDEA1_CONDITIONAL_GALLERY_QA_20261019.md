# Conditional 完整图库及实际浏览检查

新编码器下 Conditional 的完整 V 图库已在服务器生成，覆盖全部十名被试、16,600 个 query、415 个静态分页。完整 JPEG 使用原文件硬链接；18,260 个原图及生成图资产、缩略图、16,600 行配对索引和独立盲评目录均已核对。缺失、损坏及重复生成 JPEG hash 为零。此次图库构建使用 CPU，耗时 113.92 秒；没有重新生成图片或改变评分。

实际 Chrome 155.0.8059.84 浏览检查通过：完整人数/概念/行数显示，前后分页及边界，被试/概念/精确 query 筛选，原图与生成图链接，415 个静态页面入口及首页渲染，缺失配对比较提示，以及盲评页面的方法名和自动分数隐藏。没有 JS 错误、意外失败请求或外站请求。浏览通过临时加密 SSH 转发和服务器回环 HTTP 完成；没有公开服务，没有向 C/D 复制完整图库。浏览检查耗时 86.95 秒，临时 HTTP 服务已停止。

## 实际查看的固定顺序样本

浏览检查默认按稳定 query ID 排序，没有按分数挑选案例，也没有选择生成种子。实际查看了首屏截图及首个 query 的独立截图：

| query | 目标及目视观察 | DINO2AFC | SEED | ObjectF1 |
|---|---|---:|---:|---:|
| sub-01/acorn_01b.jpg/R4 | 目标为橡果；生成图为棕色棍状物，未重建目标对象 | 0.964436 | 0.118820 | 0.071429 |

首屏中第二行目标同为橡果，生成图也表现为长杆/柄状物，未见目标橡果。该首屏观察只是非盲、定性的局部诊断，不构成全量对象准确率或人工质量评分。首例的高 DINO2AFC 与明显对象错误同时出现，再次说明该相对图库偏好分数不能解释为对象重建成功。

## 保存位置与范围

服务器完整入口为 `/root/autodl-tmp/eeg_sota_autoresearch_20261018_v1/galleries/IDEA1_Conditional_V_complete_20261019T165807Z_v1/index.html`；独立盲评入口在同目录的 `blinded/index.html`。完整图片、页面和索引留在服务器。本地仅保存小型回执、两张浏览截图和首个 query 的 JSON。

- [图库完整性回执](../audit/IDEA1_Conditional_gallery_completion_v1.json)
- [独立材料核查](../audit/IDEA1_Conditional_gallery_materials_check_v1.json)
- [实际浏览检查回执](../audit/IDEA1_Conditional_gallery_browser_QA_v1/report.json)
- [固定首屏截图](../audit/IDEA1_Conditional_gallery_browser_QA_v1/unblinded_first_stable_page.jpg)
- [固定首例截图](../audit/IDEA1_Conditional_gallery_browser_QA_v1/first_stable_query_pair.jpg)
- [首例路径与分数](../audit/IDEA1_Conditional_gallery_browser_QA_v1/first_stable_query_case.json)

当前图库只有已完整评价的 Conditional 一列。Frozen、RealPaired、Moment、Iso、Clean、Conditional 置换 EEG 和缺失 EEG 列在页面明确标为缺失，没有把缺失写成负结果。尚无完整同协议配对差值，无人工盲评、画质非劣结论或独立确认。初始材料回执中的 `browser_UI_QA_performed=false` 保留其创建时状态；后续实际浏览证据单独保存，不回写或改变已绑定 SHA 的原回执。
