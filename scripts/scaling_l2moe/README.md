# Latte-L/2-MoE 单卡到多卡加速比实验（表一）

- **6000ada**：可同时跑 4 个任务（1/2/4/8 卡）。
- **H200**：本次只跑 1 个 4 卡任务；若因 PEND 上限未提交，等 6000ada 有任务跑完后补交。
- **pro6000**：环境未就绪，暂不跑。

提交前建议先 `jhosts`（节点 27=H200，28=pro6000，前面=6000ada）。

## 提交全部（5 个任务）

```bash
cd /public/home/liuhuijie/dits/XDiMo
bash scripts/scaling_l2moe/submit_all.sh
```

若 H200 因「待提交任务数达上限」未提交，等 8gpu 任务开跑后单独提交：

```bash
jsub < scripts/scaling_l2moe/h200/4gpu.sh
```

查看任务：`jjobs`；杀任务：`jctrl kill <id>`。日志：`output/logs/<JOBID>.output`、`<JOBID>.error`。
