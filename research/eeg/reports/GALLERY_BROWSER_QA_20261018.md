# 全量画廊的实际浏览器检查

历史 patience20 REF_REG 的完整画廊已载入浏览器：16,600行、415页、十被试、166个概念。实际检查通过被试/概念/query筛选、前后页、首页和被试末页边界、目标原图链接及盲评页的方法/分数隐藏。没有 JavaScript 错误、失败页面请求或外部页面请求。

画廊仍完整保存在服务器。检查通过短时回环 HTTP 服务和加密 SSH 转发进行，服务已经关闭；没有下载整套图片。本地只保存110,685字节的首个稳定页面截图和小型记录。原浏览器工具初始化报路径错误，重置重试仍失败；实际检查使用已安装的 Playwright 和 Chrome，无新增软件安装。

[实际首屏截图](../audit/gallery_browser_QA_v1/unblinded_first_stable_page.jpg)

## 首个稳定样本：高 DINO 不等于正确对象

本案例是稳定 ID 顺序的第一行，没有按分数或生成种子选择。

| 已核对项目 | 实际内容 |
|---|---|
| query | `sub-01/acorn_01b.jpg/R4` |
| EEG stimulus_id | `acorn_01b.jpg` |
| 概念 ID | `00004_acorn` |
| 数组行 | `30` |
| 原图 ZIP member | `training_images/00004_acorn/acorn_01b.jpg` |
| 生成文件 | `.../REF_REG_correct_R4/sub-01/00000030.jpg` |
| 评分目标 | `.../evaluation/gallery/V/images/00000030.jpg` |
| DINO2AFC / DINO余弦 | `0.981916845 / 0.418001175` |
| DINO Top1 / Top5 | `0 / 0` |
| ObjectF1 / SEED | `0.092827004 / 0.154116584` |

刺激、概念、数组行、原图字节 SHA、生成文件 SHA 和真实评分目标已交叉核对。完整路径与哈希见 `audit/gallery_browser_QA_v1/first_stable_query_case.json`。

首屏目视检查中，该生成物体的形态明显不同于目标橡子。DINO2AFC 衡量它相对固定真实 gallery 干扰图的偏好，不能解释成98.2%的对象正确率。该案例不是评价函数错误的证据，也不代表全部图片的对象准确率；本轮保持既定评价不变，同时保留独立语义指标和真实图片。

本次完成的是浏览器功能及首个稳定页面的界面目视检查。全部文件和配对另有机械核查。没有逐图人工科学评分，没有测试独立 file:// 打开，也没有把历史图片称为新编码器或新 idea 的结果。

实际记录：`audit/gallery_browser_QA_v1/report.json`、`audit/gallery_browser_QA_v1/completion.json`；服务器画廊中的新增证明为 `actual_browser_UI_QA_v1.json`。原构建时“尚未进行浏览器检查”的记录按历史事实保留。
