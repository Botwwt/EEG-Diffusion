# 用人类物体相似性维度选择 EEG 图像条件的修正方向

前一轮的流模型把生成器的相对 token 作为整体预测。随机起点在这些坐标里造成比替换实际 EEG 大约十倍的变动。取条件均值能提高 EEG 的相对影响，成图命名仍少。新实验只修改原生条件中与对象属性相关的方向，保留该条 EEG 的其他 token 信息。[配对流与条件均值](PAIRED_TOKEN_FLOW_STUDY.md)

## 图像目标

外部作者从约 470 万次三图相似性判断学到 66 个物体维度，随后通过冻结视觉特征的回归把每个维度延伸到 THINGS 的单张照片。保存的照片向量是该公开回归的预测。按文件名和发布版散列核对，26,107 张照片中有且仅有登记训练集的 13,230 张进入拟合。试验保留训练和开发的整类划分。[原始行为数据论文](https://doi.org/10.7554/eLife.82580) · [逐图对象维度](https://doi.org/10.1038/s41562-024-01980-y) · [固定数据发布](https://github.com/ViCCo-Group/dimension_encoding/tree/b1843e7eb92858c9c4a9ef88b4a5faea24752b49/data/66d)

将四个相对图像 token 写成相同的训练照片 PCA 坐标 $Z$。行为组对 $Z$ 和标准化后的 66 维图像描述计算交叉协方差并取正交列空间 $U$。等维视觉对照用 DINOv2 图像特征的前 66 个主成分作描述。测得的 EEG 输入是已校对的 17 导、0–800 毫秒的 20 毫秒箱，以及从这一条 EEG 得到的 Brain-HiVE 预测。每位被试用一个共享岭回归给出相对 token 估计 $\hat z(e)$。令原生图像条件的相对坐标是 $z_N$，修正为

$$
z_{*}=z_N+(\hat z(e)-z_N)U U^\top.
$$

投影之外的坐标、PCA 剩余量与四个 token 的公共均值来自实际查询。错配对照用预先固定的另一物体查询给出 $\hat z(e)$。两种子空间的参数规模与 EEG–照片配对相同，推理只用当前的一名被试 EEG。正式名称和原照由后来运行的评分代码读取。[子空间训练与推理](code/paper_behavioral_token_subspace.py)

## 整类折外检查

在登记训练物体内的四个整类折上独立重估子空间和岭回归。验证采用没有见过对应物体的 EEG 编码器，每条查询的 DINOv2 正物体原型排除对应照片。两名被试各提供 13,230 条查询。

| 相对 token | 主体 Top-1 / 26,460 | 主体 Top-5 / 26,460 |
| --- | ---: | ---: |
| 原生 | 127 | 704 |
| 替换 66 个 DINOv2 方向 | 128 | 580 |
| 替换 66 个行为属性方向 | 67 | 287 |

视图控制使用 DINOv2 目标，折外评分也在 DINOv2 空间。另用相同的训练照片到语言状态映射，从两个被试的连续条件读出完整 1,854 词表。开发和正式 R4 的所有组在首位均为 0/400。正式 R80 的原生、视觉替换、行为属性替换的 Top-1 / Top-5 分别为 2 / 10、1 / 6、1 / 6，行为组使用错 EEG 时为 0 / 1。这组候选的表现止于成图前读出。[四折与开放词表评分](results/behavioral_token_subspace.json)

## 复现

下载固定发布中的文件名、66 个维度名称和预测矩阵。在已备好训练划分和物理 EEG 的环境中，重建训练子集并在计算节点运行。

~~~sh
python paper_prepare_behavior_dimensions.py "$SOURCE" "$RAD_ROOT/data/prep" "$INPUTS"
python paper_behavioral_token_subspace.py fit 1 --attributes "$INPUTS/registered-training-dimensions.npz"
python paper_behavioral_token_subspace.py infer 1
python paper_behavioral_token_subspace.py score 1
~~~
