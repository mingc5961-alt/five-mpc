# G1 通信调度实验 · 服务器归档（2026-11-28）

本版收录同预算调度、预测器与真实航迹三个问题的实验布置、冻结配置、逐场结果和无线账本。服务器按每场成功交付的比特数除以服务时长计算净吞吐。风险为用户探针遇到链路不可用或波束指向损失超过 3 dB 的比例。

[服务器实验计划](plans/SERVER_EXPERIMENT_PLAN.md) · [确认实验设计冻结件](results/paired_linux_v2/design/frozen.json) · [逐世界结果](results/paired_linux_v2/confirmation/world_scores.csv) · [随机实飞与吞吐结果 TXT](report/random-flight-results.txt) · [2026-11-28 服务器状态快照](report/server-status-snapshot.txt)

## 实际执行范围

| 批次 | 规模 | 设计与结果 |
|---|---:|---|
| 本机主实验 · 开发 | 1,504 场 | 8 个开发世界 × 4 个倍率确定主配额和固定周期，随后比较原生、等费、风险与学习调度策略。[32 个开发点](results/paired_linux_v2/freeze/all_32.csv) · [冻结结果](results/paired_linux_v2/freeze/freeze.json) |
| 本机主实验 · 确认与压力 | 2,352 场 | 包含主场景、预算、配额和机群构成变化。主比较覆盖 48 个配对世界。[主对比](results/paired_linux_v2/confirmation/contrasts.json) · [次要与压力对比](results/paired_linux_v2/confirmation/secondary_contrasts.json) |
| 开发倍率扩展 | 656 场 | 补测 0.30、1.25，并保留原定执行范围。[设计](results/paired_linux_v2/design/extension_plan.json) · [回执](archives/multiplier-extension.tar.gz) |
| 实飞轨迹驱动的通信闭环 | 160 场 | 8 个轨迹组合，每组 20 个预测器、调度和种子设置。含联合与固定时刻 AoI-DQN、G1、Chronos、RAFT、固定融合和运动外推。[结果 CSV](results/paired_linux_v2/field_extension/paired_throughput.csv) · [每组明细](results/paired_linux_v2/field_extension/panel_metrics.csv) |
| 前期 Linux 重建协议 | 1,384 场 | 640 个开发端点、744 个留出端点，使用重训 G1。单列 24 个留出世界 4000–4023。[该协议报告、图和统计](results/rebuild_v1/RESULTS.md) |

本机主实验合计 3,856 场。AoI-DQN 另有 912 个训练回合与 720 个验证回合。原有实飞几何覆盖 314 架次，统计每个指标时依实际可评估架次取数。历史回放、时钟核验、模型前向压测、训练回合和几何查询各自计数。[学习协议](results/paired_linux_v2/learning/training_protocol.json) · [学习结果](results/paired_linux_v2/learning/summary.json) · [原有实飞评价](results/paired_linux_v2/field/comparisons.json)

## 三项结果

1. 在 48 个主确认世界中，同一个 G1 预测器使用截止期调度，相对联合 AoI-DQN 的平均净吞吐差为 **+228.94 Mbps**，95% 配对区间为 **[225.12, 232.72] Mbps**，风险差为 **−8.08 个百分点**。在相同 AoI-DQN 定位事件下，G1 相对纯运动外推的平均净吞吐差为 **+101.72 Mbps**，95% 区间为 **[95.12, 108.31] Mbps**。G1 相对固定融合的吞吐差为 **+1.70 Mbps**，区间 **[−0.80, 4.28]** 跨过零。这三项使用各臂冻结工作点。[逐对结果](results/paired_linux_v2/confirmation/contrasts.json)
2. 在 16 架次低重复随机航迹组成的 30 秒机群回放中，同倍率、同配额下 G1 截止期调度达到 **945.24 Mbps**，联合 AoI-DQN 为 **826.01 Mbps**，差 **+119.23 Mbps（+14.43%）**，风险分别为 **9.81%** 和 **13.04%**。用 Chronos 时，两种调度分别为 **1,353.10** 和 **1,294.54 Mbps**，风险分别为 **19.54%** 和 **12.84%**。这个全随机组合仅有一组。不同时间采集的航次合成机群，28 GHz、400 MHz 无线链路和交付比特由仿真计算。[逐组合结果](results/paired_linux_v2/field_extension/panel_metrics.csv) · [完整配对差值](results/paired_linux_v2/field_extension/paired_throughput.csv)
3. 随机实飞几何组包含 14 架次 AMOVFLY 和 10 架次 NeuroBEM，后者由 9 套随机航点程序飞得。在各航次可用的 2、3、5 秒预测查询中，G1 与 Chronos 的平均视线角误差分别为 **1.185° 和 1.152°**（AMOVFLY）、**1.183° 和 0.561°**（NeuroBEM）。按相同方向几何重复筛查，24 架次中 23 架次通过低重复阈值。[误差、聚合规则与架次明细](results/paired_linux_v2/field_extension/recurrence_and_angles.json)

## 原参考平台验收

原版 G1 历史世界 0 ×0.45 的 848 个定位事件和无线账本已对齐，Chronos 世界 0 ×0.75 的 811 个事件也已对齐。Linux 物理数组指纹与原平台仍有差异。世界 1400 ×0.45 回放有 847 个事件，净吞吐与原参考相差 0.001445 Mbps。原参考平台 N01 尚未验收，其余 31 个坐标待提交。本机配对协议在自己的冻结物理世界上完成 32 个开发坐标，得出配额 849 和固定周期 3.394 秒。两套回执各保留原状态。[G1 回放核验](audits/recovery_01/anchor_g1/verification.json) · [Chronos 回放核验](audits/recovery_01/anchor_c/verification.json) · [跨平台首场](results/paired_linux_v2/cross_platform_canary.json) · [本机协议](results/paired_linux_v2/protocol.json)

## 从结果追到原始回执

[端点索引](indexes/endpoints.csv) 的每一行给出协议分卷、回执在归档中的路径、世界或机群、吞吐、风险、定位次数和 SHA-256。五个闭环分卷共 6,056 行。原始回执保存了逐次定位和四项无线时间账本。[归档成员索引](indexes/archive-members.csv) 同时覆盖几何评分和历史诊断。根目录的 SHA256SUMS 覆盖本版本其他全部文件。

在本版目录运行以下命令，使用 Python 3.10 或更新版本的标准库即可校验文件、每条归档成员、五个批次的计数、逐场无线账本，以及实飞组的 48 个配对吞吐结果。

```bash
python3 tools/verify.py
```

完整重跑模拟需要配套的物理世界生成器、原预测模型权重和相应科学环境。归档保留服务器实际使用的运行脚本、六个冻结 AoI-DQN 检查点、验证曲线、规划矩阵和模型指纹。脚本中的绝对路径记录原服务器布局，迁移时需配置对应资源。[服务器代码](source/g1-original-20261127/independent/) · [交接代码](source/g1-server-handoff/) · [原 G1 研究版本](../g1-paid-cold-start-20261127/) · [恢复说明](plans/START_SERVER.md)

实飞部分提供原始文件哈希、架次划分、各模型派生误差和最终通信结果。原始遥测和第三方模型权重应从各发布方获取。DeepSense Scenario 23 原包需要授权登录，本版的实飞吞吐结果来自 AMOVFLY 和 NeuroBEM 轨迹。[AMOVFLY](https://github.com/YujiaoHu/AMOVFLY-Dataset) · [NeuroBEM 数据说明](https://rpg.ifi.uzh.ch/neuro_bem/Readme.html) · [NeuroBEM 架次清单](https://download.ifi.uzh.ch/rpg/NeuroBEM/Flights.txt) · [DeepSense Scenario 23](https://www.deepsense6g.net/scenarios/Scenarios%2020-29/scenario-23)
