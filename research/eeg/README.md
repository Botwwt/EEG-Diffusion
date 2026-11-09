# EEG 研究代码与结果快照

此目录归档 2026-10-18 至 2026-11-09 的 Brain-HIVE / ATM 研究代码、配置和报告。先读 [当前研究摘要](../../docs/CURRENT_RESEARCH.md)。

- `brainhive_bounded_component_v1.py`：ordinary 的受限方向修正。
- `brainhive_component_generator_v1.py`：冻结原生生成器与输入梯度通路。
- `train_BrainHIVE_bounded_component_v1.py`：原普通/风险组件训练实现。
- `eeg_margin_matrix_free_20261106_v233.py`：无矩阵求导算子原型。
- `PKU_*v239.py` / `PKU_*v241.py`：新集群原生编码器重建与独立 CPU 审计。
- `configs/`：各次协议与来源约束；DRAFT 不代表已运行。
- `reports/`：结果、负结果、推导和解释边界；每份按其日期阅读。
- `evidence/`：少量关键交接证据，不含原始 EEG 或权重。

代码来自多个已冻结实验版本，使用过旧服务器目录和不同运行环境。它们是可审计的研究源文件，不是一键复现实验入口。不要直接运行历史队列、重建已退出的控制器或将不同父模型/宿主的缓存和检查点混用。

原生 HIVE 作者代码固定到 `ssshamiii/Brain-HIVE` 的 `f06db69fcc4a029439e369c5c3dba6099e57560e`；原生类通过 AST 读取，需从原项目取得并按记录核验。公开模型/数据仍受各自许可约束，本快照不重分发其权重。

大二进制文件和逐图大特征不在 Git 中。公开文件清单与哈希位于 [快照来源清单](../../provenance/research_snapshot_20261109.json)。扫描通过只说明发布范围内未发现匹配的凭据格式，不能代替全部实验的科学复核。
