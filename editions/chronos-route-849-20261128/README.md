# Chronos-849 实验结果目录

本实验研究两个问题。在相同的付费观测与 Chronos 预测中心下，路线记忆能否更准确地判断波束还可以保持多久？把 849 笔定位预约分给不同用户后，这种信息能否提高全机群的净吞吐？确认集使用 48 个全新物理仿真世界（6000–6047），每场 16 架无人机，从零预热起运行 180 秒。[完整报告](REPORT.md) · [实验规划](package/EXPERIMENT_PLAN.md) · [冻结设计](design/frozen.json)

**结论。** 在同向重访主分析层，相对时间模型 TIME，ROUTE 平均少错失 0.098 秒可用波束时间，配对 95% 区间为 [0.073, 0.126] 秒。额外放行的不安全时间上界为 0.0103 秒，低于预定的 0.02 秒容限。共同工作点 0.75 下，VALUE-route 相对冻结强对手 VALUE-time 的净吞吐差为 +1.387 Mbps，区间为 [−1.568, +4.334]，跨过零。相对风险差为 −0.189 个百分点，区间为 [−0.291, −0.085]，通过非劣判据。VALUE-route 的绝对风险为 6.737%，区间为 [6.636%, 6.838%]，通过 7% 目标。因此论文可主张路线记忆改善保持期判断；同费净吞吐增益仍缺乏确认。[全部配对差值](contrasts.csv) · [最终判定回执](report_receipt.json)

## 从问题找到文件

| 想看什么 | 文件及读法 |
|---|---|
| 论文图表 | [图 1 固定中心下的波束保持期](figures/Fig1_fixed_center_lifetime.pdf)、[表 1 同一 849 笔预约下的净吞吐与风险](表1_同一849笔预约下的净吞吐与风险.csv)、[图 2 选人分歧带来的通信收益](figures/Fig2_recipient_goodput.pdf)。图 2 右侧的同状态反事实来自开发集，16 个抽定状态中有 2 次选人分歧。 |
| 正式比较 | [world_scores.csv](world_scores.csv) 保留物理世界、指纹、实验臂、分层和工作点。[contrasts.csv](contrasts.csv) 给出差值方向、均值、上下界和世界数。E2 先在同一世界内平均 DQN 的三个种子，再按完整世界配对，进行 20,000 次自助抽样。 |
| E1 保持期 | [主层逐世界评分](E1/primary_grid.csv)、[主层覆盖](E1/primary_coverage.csv)、[分层汇总](E1/strata_summary.csv) 和 [风险头](E1/calibration/)。U 表示错失的安全保持秒数，O 表示额外放行的不安全秒数。主层覆盖 48 个世界的 1,182 个前缀、3,095 个查询。 |
| E2 吞吐与风险 | [端点表](E2/endpoints.csv) 按世界、选人策略、种子和倍率列出结果。G 为全机群净吞吐（Mbps），表中 R 为风险百分比，ΔR 的单位为百分点。共同工作点含 480 次闭环，冻结的额外工作点含 816 次。[预约表](E2/calendar_849.json) · [各臂工作点](E2/development_frozen_workpoints.csv) · [两个价值头](E2/labels/) |
| 核验和定位差异 | [总审计](audits.json) 汇总开发世界 1400 初核、最终模型核验及逐世界回执。[独立交付复核](audits/delivery_validation.json) 从原始记录重算 U、O、净吞吐和主对照区间，并核验 1,296 次闭环的 1,100,304 笔定位。每次闭环前 16 笔按 uid 0–15 初始化，定位共占 2.6319 秒，四项账本合计 180 秒。 |
| 复现依据 | [原始规划包](provenance/Chronos-849-server-plan.zip)、[协议](protocol.json)、[代码](code/)、[运行资产与哈希](manifest.json)、[冻结模型和世界清单](design/frozen.json)、[物理世界指纹](design/worlds/)。[Git bundle](provenance/frozen-source.bundle) 保留独立实验库的冻结提交 `e6fe73b01fe47f78a8dfdc13f591309170ed8031`。服务器路径是原始运行记录。 |
| 执行规模 | [进度终态 TXT](实时进度.txt) 区分 180 秒闭环、4.464 秒反事实短分支和 E1 世界面板；共完成 6,302 个评分实验，计入训练、核验及交付后共 6,344 个工作单元。[运行日志](logs/) |

## 原始数据与逐笔账本

压缩包解开后沿用实验目录的相对路径。[下载本版 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/chronos-route-849-20261128) · [20 个分包的大小、文件数和 SHA-256](archives/assets.json) · [逐文件路径、位置、大小和 SHA-256](archives/contents.tsv.gz)

| 下载文件 | 解开后的位置与用途 |
|---|---|
| `e1-scores.tar.gz` | `E1/scores/` 下各阶段的逐查询评分和前缀一致性回执。 |
| `e1-prefix-index.tar.gz`、`e1-fit-calibration-inputs.tar.gz` | `E1/prefix_index/`、`E1/calibration/inputs/` 中的付费前缀索引、拟合和独立校准输入。 |
| `e2-receipts.tar.gz` | `E2/receipts/` 中各闭环的实际选人、849 笔定位起止时刻、交付比特与无线账本。 |
| `e2-decisions-*.tar.gz` | `E2/decision_logs/` 中的逐预约决策轨迹。确认集按连续 8 个世界分包，开发和两次单世界核验单独成包。 |
| `e2-counterfactual-labels.tar.gz`、`e2-same-state-rankings.tar.gz` | 价值头训练、验证和开发诊断的反事实分支，以及同状态下不同选人方法的排序。 |
| `worlds-*.tar.gz` | `design/worlds/` 的 48 个物理世界输入，每包 8 个。按同名 JSON 核对文件及物理指纹。 |
| `task-receipts-and-handoff.tar.gz` | `jobs/` 的逐任务回执及 `audits/freeze_worker_handoff/` 下隔离的旧缓存片段。 |

服务器保留运行时推理缓存、NUMA 权重副本、锁和 Python 字节码。预训练 Chronos、微调权重与原生误差管沿用已有服务器资产，其身份由清单中的 SHA-256 固定。本版随附新拟合的四个风险头和两个价值头。[资产清单](manifest.json)

## 独立复核主结果

在本目录下运行以下命令。两个下载包足以核对全部确认集端点、主层保持期和主要配对区间；脚本需要 Python 3.11 或更高版本，以及 NumPy、pandas、Pillow。

```bash
repo_url=https://github.com/mingc5961-alt/five-mpc/releases/download/chronos-route-849-20261128
for archive in e1-scores.tar.gz e2-receipts.tar.gz; do
  curl -fL -O "$repo_url/$archive" || exit 1
done
grep -E '  (e1-scores|e2-receipts)\.tar\.gz$' archives/SHA256SUMS | sha256sum -c - &&
  tar -xzf e1-scores.tar.gz &&
  tar -xzf e2-receipts.tar.gz &&
  python3 logs/verify_delivery.py
```

最后一行 JSON 的 `status` 应为 `PASS`，`worlds` 应为 48，`E2_actual_clock_pairs` 应为 1,100,304。逐世界复核使用上面的两个原始包。完整还原时，可把 20 个包放到同一目录，运行 `sha256sum -c archives/SHA256SUMS`，然后逐包解开。[原始交付复核回执](audits/delivery_validation.json)
