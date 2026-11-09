# 当前 Brain-HIVE 实际模型清单

2026-10-21 UTC 的 CPU 审计。重新核验40个EEG父模型的文件SHA并严格加载；实际加载公开F/A/P，检查真实OOF条件的A/P前向；IP按safetensors真实张量形状计数。Turbo主干按当前加载器的准确配置在meta设备计数，并与实际权重文件UNet/VAE header核对。没有启动GPU实验或改变正在训练的组件。

| 模块 | 实际结构与参数量 | 来源与当前状态 |
| --- | --- | --- |
| EEG编码器 | 作者 `main.models_brain.BrainMLP`，6,452,224；4250→1024，单层残差1024→1024→1024 GELU、LayerNorm | 本次从头训练；不是ATM，也不是ATM缩小版。published/full/oof0/oof1各十被试，25epoch、seed1729；每被试分别425/325/175/175更新。组件阶段全部冻结 |
| 视觉融合目标F | `FusionEncoder`，11,015,168；B32(512)、H14(1024)、VAE视图特征(1024)分别映射1024后求和，再经1024→4096→1024残差与LayerNorm | 作者已有预训练权重；用于建立训练目标，不在正常EEG生成时读取目标图像。父模型训练按发布入口保留视觉目标dropout0.1，权重冻结 |
| 原生条件适配A | `ResNet1DLayer(norm=False)`，8,388,608；1024→4096→1024、GELU、无偏置、残差 | 作者已有预训练权重，当前冻结；输入原始EEG预测1024维，没有用单位向量替换原生输入 |
| 原生token投影P | `IPProjectionModel`，8,400,896；1024→8192→4×2048、LayerNorm | 作者已有预训练权重，当前冻结；与A分开计数 |
| 原生图像注意力IP | `IPAttnAdapterModel/IPAttnProcessor2_0`，340,787,200；70个跨注意力处理器的K/V | 作者已有预训练权重，当前冻结；每图4步实际调用280次，工程和原生图片已有调用核验 |
| Turbo纯UNet | `UNet2DConditionModel`，2,567,463,684；原生SDXL-Turbo配置 | 已有 `stabilityai/sdxl-turbo` 权重，版本 `71153311d3dbb46851df1931d3ca6e939de83304`；冻结。此行未计上行额外IP，也未计外部P |
| Turbo VAE | `AutoencoderKL`，83,653,863 | 同一已有Turbo权重，冻结。图像训练缓存使用原生FP32编码、mode与缩放 |
| 文本编码器1/2 | `CLIPTextModel` 123,060,480；`CLIPTextModelWithProjection` 694,659,840 | 已有Turbo权重，冻结，正常生成使用相同空文本提示 |
| 普通受限适配器 | `BoundedConditionAdapter(predict_risk=False)`，525,568；1024→256 GELU→1024，零初始化方向头 | 本次新建，正式训练排在风险组之后；与风险组相同完整OOF数据、固定三遍、24807更新。尚未完成训练时不能填写为已有最终checkpoint |
| 风险组件 | `BoundedConditionAdapter(predict_risk=True)`，525,825；上述结构加256→1风险头 | 本次从头训练中。风险头仅多257参数；输入EEG预测条件及本父模型仿射，正常推理不读目标、类别、caption。训练固定三遍、24807更新 |

父模型训练还优化一个独立的对比温度标量 `logit_scale`；它不是EEG编码器前向的一部分，也不参与图片生成，因此不混入6,452,224的编码器参数数目。

采样器的纯UNet、额外IP、VAE和两文本编码器合计 **3,809,625,067** 参数；A、P、单被试EEG编码器及新组件另计。F仅用于视觉训练目标，不能把40个替代父模型、F或重复引用的权重路径全部相加当作单次推理模型。当前P在UNet外部；不要套用其他加载器把P注册进UNet后的模块计数方式。

## 输入与生成配置

EEG实际选择17个后部电极：P7、P5、P3、P1、Pz、P2、P4、P6、P8、PO7、PO3、POz、PO4、PO8、O1、Oz、O2；每电极250点，17×250=4250。训练与保护划分V使用四次真实重复的均值；已完成的原生官方轨使用80次真实重复均值。名义时间窗0–0.996秒、250Hz沿用发布预处理及时间元数据；缓存本来已有250点，不二次裁剪。未声称独立重放原始采集事件。

原生生成器固定512×512、EulerAncestral trailing、4步、CFG0、IP尺度1、每query一图；单训练seed1729，每query噪声按固定规则派生，无最佳种子挑选。冻结权重保持FP16原字节；组件训练采用BF16 autocast，推理重新加载原生FP16权重。工程阶段新加载入口已对固定原生query逐像素重放成功，最大像素差为0。

当前正式条件链为 **EEG→BrainMLP→有界组件→A→P→原生IP K/V→Turbo**。原生对照跳过新增组件。当前没有旧 `DiffusionPriorUNet` 的1024维DDPM语义采样；旧REF-PRIOR及旧A1/A2/C1残差控制器属于历史实验，不能把它们的参数量、训练步数或CFG填进本轮表格。

## 检查点与训练状态

每个EEG父模型、F/A/P/IP的实际路径及SHA、元配置参数数目和当前组件进度见 `audit/BrainHIVE_current_model_inventory_CPU_v21.csv` 与同名JSON，共50条。它引用保留的v19权重/张量审计，明确区分预定训练遍数与已完成遍数：运行中的第1遍不算完成1遍，普通组未启动就记录0遍；最终checkpoint不存在就明确记为不存在。原生父模型正式目录为 `/root/brainhive_reproduction_20261020_v1/runs/{published,full,oof0,oof1}_source_dropout_v2/sub-XX/brain_final.pt`；公开条件模块位于同工作目录的 `assets/prior_H14/`。

风险组件的训练恢复点为 `components/bounded_v1/risk/resume.pt`；只有固定24807更新与Dfit校准完成后才写 `component_final.pt`。普通组同理。最终文件尚未存在时，只报告运行中的实际更新，不把预期训练量冒充完成量。实时状态为服务器 `components/bounded_v1/risk/progress.json`；每两分钟保存可恢复状态，首次恢复点已核验参数有限、方向非零、覆盖计数和随机状态。

风险表示条件预测误差，不能称为生理EEG SNR。组件只接收1024维编码结果，不能据此解释为已经定位了具体电极或时间段的因果贡献。KAN和阶段路由仍未实施。

## 容量判断

当前没有受控扩容实验，**没有容量不足的具体证据**。原生生成主干并非玩具缩小版；EEG编码器与本轮组件参数量小于生成器也不直接证明瓶颈。25epoch原生对齐与三遍组件训练是实际固定配方，不是收敛或图片成功证明。

先完成原生、普通适配器、风险组件在同父模型、数据、权限和采样成本下的全量图片比较。若再检验容量，必须指定扩容模块，固定训练数据/目标/更新预算与评价，报告训练与开发泛化及实际图片差值；不能用差图或下降的训练损失代替这个对照。

来源：`audit_BrainHIVE_current_model_inventory_CPU_v19.py`、`audit/BrainHIVE_current_model_inventory_CPU_v19.json`、`clarify_BrainHIVE_inventory_training_state_CPU_v21.py`、`train_BrainHIVE_parent_v2.py`、`brainhive_component_generator_v1.py`、原生模型源文件与固定配置。时间轴依据服务器 `/root/mced_eeg_full_montage_cache_v3/time_alignment_contract.json` 及已有逐被试输入审计。完整图片结论另见 `BRAINHIVE_NATIVE_CLOSURE_AND_COMPONENT_EXECUTION_20261021.md`。
