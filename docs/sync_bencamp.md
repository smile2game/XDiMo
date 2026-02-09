# 大本营同步说明（本机 ↔ 超算）

本机（2080Ti 等）为**大本营**：代码以 **xdimo/** 为准，训练/开发在此验证后，再推送到超算或 GitHub。

## 目录约定

| 位置     | 代码布局     | 说明 |
|----------|--------------|------|
| 大本营   | **xdimo/**   | 模型、数据集、扩散、工具（train.py 等依赖此路径） |
| 超算旧版 | 可能仍有 src/ | 拉取时已排除，不会覆盖本地 xdimo/ |

文档统一在 **docs/**（如 STRUCTURE.md、perf.md 在 docs/ 内）；根目录不再保留 STRUCTURE.md、perf.md，避免与超算旧版冲突。

## 排除规则（.rsync_exclude）

- 与 **.gitignore** 对齐：output/、share_ckpts/、wandb/、缓存等不同步。
- **大本营保护**：拉取时排除远程的 **src/**、根目录 **STRUCTURE.md**、**perf.md**，只保留本地 xdimo/ 与 docs/ 布局。

## 常用命令（在 dits/ 下）

```bash
./fetch.sh          # 超算 → 本机（拉回代码与配置，不拉 output/、src/）
./push.sh           # 本机 → 超算（推送代码与配置，不推 output/、权重）
REMOTE=用户@超算 REMOTE_DIR=/path/dits/XDiMo/ ./fetch.sh   # 临时指定超算
```

推送后超算会得到 **xdimo/**；若超算上仍有旧版 **src/**，可手动删除或保留不用。
