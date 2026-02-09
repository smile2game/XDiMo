# XDiMo 训练脚本

## 目录结构

```
scripts/
├── 4gpu/                 # 4 卡对比实验（6000ada/h200/pro6000 各 4 张）
│   ├── 6000ada/         # baseline.sh, bs2.sh
│   ├── h200/            # baseline.sh, bs4.sh
│   ├── pro6000/         # baseline.sh, bs2.sh
│   └── submit_compare.sh # 一键提交全部对比任务
├── 8gpu/                 # 8 卡单节点
│   ├── 6000ada.sh
│   └── pro6000.sh
├── 1-node.sh             # 6000ada 8卡（旧）
├── h200-1node.sh         # H200 2卡（旧）
├── pro6000-1node.sh      # pro6000 8卡（旧）
└── ...
```

## 提交命令

- 4 卡对比: `cd XDiMo && bash scripts/4gpu/submit_compare.sh`
- 单任务: `cd XDiMo && jsub < scripts/4gpu/6000ada/baseline.sh`
- 查看: `jjobs`
- 杀任务: `jctrl kill <id>`
