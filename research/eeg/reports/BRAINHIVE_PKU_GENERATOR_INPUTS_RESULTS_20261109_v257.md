# 原生生成接口与训练图像恢复：实际工程结果

更新：2026-11-09 UTC。本轮完成生成接口和输入资产检查，尚未训练新的 ordinary，也没有新方法图片收益。

## 已完成的验证

- GPU 作业 5097204 在一张 A800 上完成原生 A/P/IP/SDXL-Turbo 接入，程序 114.65 秒，其中冷加载 107.89 秒；峰值 allocated 显存 8.78 GB。A/P/IP 使用严格加载的原生权重，70 个 IP 层；Turbo 使用已校验的固定公开单文件版本。未逐张量独立重验转换后的全部 UNet。
- 对新 sub-03 编码器的两个真实 T 折外条件，各生成原生和零初始化 ordinary 图片，共四张 512×512、四步、CFG0 图片。两对逐像素 SHA 相同，关闭路径和零头条件也逐位回到原生 c。另由本机 Pillow 独立解码四张 PNG 并复核像素 SHA。
- 合成 latent/noise 的反向图检查得到方向层梯度范数 0.0693983，冻结 A/P/UNet 未产生参数梯度、参数版本未变。这只验证计算图，不是实际图像训练 loss 或方法效果。优化器更新为零。
- CPU 作业 5097322 恢复 Torch 2.3.0+cu121 / torchvision 0.18.0+cu121，导入和 pip check 通过。沿用 Resize512、Bilinear、CenterCrop512、RGB ToTensor、Normalize0.5。
- OSF y63gw 的原训练图像 ZIP，文件 ID `60d837c4f83de2006bb02ccc`、版本 1、655,039,265 字节，已直接传到共享存储。CPU 作业 5097488 在 6.37 秒内复核整 ZIP SHA，以及全部 16,540 张 JPEG 的唯一成员名、大小、SHA 和图像头；均为 500×500。官方测试图像未读取。D/V 仅参与文件身份校验，未参与模型前向或拟合。

ZIP SHA-256：`dc9cc627141ba230641f729f84b80eb01f0a963d288c971eefcf76b4a7581f3a`。首次传输超时保留前缀，同源有限续传完成；该失败不属于科学负结果。

## 当前输入缓存任务

GPU 作业 5097501 已实际启动，只对允许的 13,230 张 T 图像构建原生 VAE mode 缓存。由固定 FP16 VAE 权重拓宽到 FP32，TF32 关闭，原生 scaling=0.13025，输出为 13,230×4×64×64。缓存放在共享存储，不下载到 D 盘。

本报告时最新观察为 RUNNING，尚无全量缓存完成或独立复核结论。每个作业有有限时间；若只完成前缀，即便 Slurm 返回成功也不能称全量完成。此次新硬件缓存不宣称与旧缓存逐位相同。

## 对研究结论的影响

以上是新父编码器匹配 ordinary 训练的前置工作。旧 ordinary 权重不覆盖；新父模型需要另训自己的 ordinary，不能按维度相同直接接旧权重。尚未取得该新组合的实际图片效果，第二项独立创新、跨宿主增量、端到端加速和 SOTA 均未确认。

下一步先完成并独立核验全部 T latent，再登记匹配 ordinary 的实际训练和对照。保留历史 ordinary 的完整结果，继续按全部指标解释后续实验。

原生生成器代码：[PKU_native_generator_20261109_v253.py](../PKU_native_generator_20261109_v253.py)；工程入口：[GPU profile](../PKU_profile_native_generator_GPU_20261109_v253.py)；图像身份核验：[CPU audit](../PKU_audit_training_images_CPU_20261109_v254.py)；缓存入口：[T VAE](../PKU_build_native_T_latents_GPU_20261109_v256.py)。
