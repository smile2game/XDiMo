# 路径与同步配置说明

本仓库内尽量使用**相对路径**；只有落在仓库外的数据、权重等需要**绝对路径**。以下说明如何主动选择/修改路径。

---

## 1. push.sh / fetch.sh（与超算同步）

脚本在 **dits/** 目录下，通过相对路径定位 **dits/XDiMo/**，无需写死本机绝对路径。

### 需要主动配置的项（仅这两项通常为“绝对”或机器相关）

| 变量 | 含义 | 默认示例 | 如何改 |
|------|------|----------|--------|
| `REMOTE` | 超算登录串 | `liuhuijie@10.150.2.4` | 改成你的 `用户@超算地址` |
| `REMOTE_DIR` | 超算上 XDiMo 所在目录 | `/public/home/liuhuijie/dits/XDiMo/` | 改成你在超算上的路径，末尾保留 `/` |

### 使用方式

- **不改脚本**：直接运行 `./push.sh` 或 `./fetch.sh`，使用脚本内默认的 `REMOTE`、`REMOTE_DIR`。
- **临时覆盖**（推荐，不改脚本内容）：
  ```bash
  REMOTE="你的用户@超算IP" REMOTE_DIR="/你的/home/目录/dits/XDiMo/" ./push.sh
  REMOTE="你的用户@超算IP" REMOTE_DIR="/你的/home/目录/dits/XDiMo/" ./fetch.sh
  ```
- **密码**：默认使用脚本内密码；若用环境变量可设 `SSHPASS=你的密码` 再执行脚本。

### 排除 output

- **push**：不推送本地的 **output/**（日志、checkpoint），避免把本机训练状态推到超算。
- **fetch**：不拉取远程的 **output/**，避免用超算的日志/权重覆盖本地。

两边训练各自独立，仅代码与配置通过 push/fetch 同步。

---

## 2. XDiMo 内部：数据与权重路径

训练/采样用到的路径在 **configs/** 的 yaml 里配置，例如：

- **configs/ffs/ffs_train.yaml**
  - `data_path`：训练数据目录（视频等）
  - `pretrained_model_path`：共享权重（如 VAE）目录
  - `results_dir`：实验输出根目录（默认 `./output`，相对仓库根）

这些在仓库内一般用 **相对路径**（如 `./share_ckpts`、`./output`）。  
当数据或权重放在**仓库外**时，改为**绝对路径**即可，例如：

- 本机：`data_path: "/data/preprocess_ffs/train/videos"`
- 超算：`data_path: "/public/home/xxx/datasets/ffs/train/videos"`

**建议**：本机与超算各保留一份配置（或同一 yaml 里用环境变量），按机器选择数据/权重路径，其余尽量保持相对路径。详见 [README](../README.md) 中的「环境」「2×2080Ti 训练」「超算训练」等小节。
