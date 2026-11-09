# 本地存储整理与恢复

完成时间：2026-11-09T05:13:17.003954+00:00。本次整理保持实验计算、权重内容和效能结论不变。

- 41 个至少 50 MiB 的历史大文件，共 **22.47 GB**，已归档到用户的 `wwt/eeg/backups/local_D_large_v245`。
- 每个文件先上传，再由 Slurm CPU 计算全文件 SHA-256，删除前再次计算本地 SHA；全部通过后才移除本地副本。
- 每个原路径旁保留 `.remote.json`，记录大小、哈希、远端路径和验证任务。已独立复核全部 41 份恢复指针。
- 200 组完全相同的文件整合为硬链接，保留 329 个原路径；真实文件 ID 和内容哈希已复核。新增确认的实体副本合并量约 1.18 GB；更早的 485 MB 仅是逻辑重复量，不重复计入实际释放。
- D 盘可用空间从盘点时约 0.22 GB，升至完成时约 **23.57 GB**。这是两个时点的观测，会受盘上其他活动影响。

GitHub 保存约 17 MB 的源码、配置、报告和关键证据。原始 EEG、模型权重、优化器与大型特征不进入 Git；`.gitignore` 阻止常见二进制和传输文件被误加入。集群归档与已有实验位于同一共享存储系统，不宣称是异地备份。

## 恢复单个文件

本机根目录 `STORAGE_INDEX.json` 以及原文件旁的 `.remote.json` 是完整索引，含用户自己的远端位置。公开仓库不保存登录密码。

```console
python -m pip install -e '.[archive]'
python scripts/restore_archived_file.py --workspace WORKSPACE --pointer FILE.remote.json --host HOST --user USER --host-key-sha256 EXPECTED_FINGERPRINT
```

主机指纹应使用已有可信记录。Windows 若需指定网络接口，可显式传 `--windows-interface-index INDEX`；不改全局路由。密码通过隐藏输入读取，只留在内存中。

恢复器拒绝覆盖现有文件，保留至少 2 GiB 磁盘余量，按小块下载并验证整文件 SHA，成功后才建立原文件名。中断的 `.restore.partial` 可在核对指针后显式 `--resume`。已实测 11,852 字节文件的下载与 SHA，以及覆盖/空间不足的提前拒绝；未重新下载多 GiB 权重。

清理记录：[汇总 JSON](../provenance/storage_cleanup_20261109.json)。当前方法和证据：[研究摘要](CURRENT_RESEARCH.md)。
