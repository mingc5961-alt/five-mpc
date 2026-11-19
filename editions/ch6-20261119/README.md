# 第六章实验版本 · 2026-11-19

本版在首轮 FIVE 平台上修改路线融合门控、误差校准、调度和实验编排，使用独立世界重新检验通信吞吐与保持可靠性。可直接查看[十页完整报告](report/RouteClock_第六章完整实验分析报告.pdf)、[Word 文档](report/RouteClock_第六章完整实验分析报告.docx)及[逐场与配对工作簿](report/实验数据与配对结果.xlsx)。

## 与 11 月 18 日版本对照

| 内容 | 首轮 results-20261118 | 本版 ch6-20261119 |
|---|---|---|
| 代码 | 顶层 code/ 中的服务器平台与正式运行编排 | 独立目录 platform/，保留实际发布时的科学源码和测试 |
| 方法 | 旧门控、全局包络及首轮动作设置 | G1 上下文与时域交互，按候选期限校准，全中心统一 full 误差头，比较三种调度 |
| 样本 | 645 场 FIVE、120 场 MPC；离线路线 2,400 条 | 复用已有离线输入。20 个世界共 840 场网络验证及 3 组 F 网格；30 个独立测试世界共 600 + 60 场。Chronos 子集为其中 10 个世界 |
| 实验问题 | FIVE 原始基线与 MPC 的感知开销、吞吐和可靠性 | 固定单次成本缩减共享预算，以及固定 5% 预算增加单次成本。12.4 ms 于主实验完成 538 场时登记追加 |
| 结果口径 | 首轮 30 世界主表的 RC−I 为 −18.02 Mbps | 新基点为 +6.03 Mbps。协议、模型与世界均有变动，两轮之差不能单独归给 G1 |

源文件对照中，[horizon_gate.py](platform/routeclock/horizon_gate.py) 为新增门控，five、holding、predictors、radio、scheduler、simulator、training 和 validity 八个模块有修改。工具侧改动 fixed_grid 并新增 11 个 ch6 脚本，新增两份回归测试。可对照[首轮平台](../../code/routeclock_server/routeclock_five_20261115/)与[本版平台](platform/)。

## 如何核查结果

- [主实验](results/primary/)收录冻结选择依据、600 场测试和验证清单，以及 Chronos、调度和等间隔对照。[追加实验](results/extension/)收录 12.4 ms 的登记、清单和四档压力配对结果。四档压力表共 240 个世界×方法行，其中 180 行来自主实验。
- [逐场数据](report/数据表/逐场指标1500.csv)含 840 + 600 + 60 场。独立测试按 task 配对。5% 预算下，成本依次为 3.1、6.2、9.3、12.4 ms，RC−I 净吞吐均值依次为 +6.03、+51.62、+230.24、+163.19 Mbps。9.3 → 12.4 ms 的配对变化为 −67.04 Mbps，95% 区间 [−95.13, −41.89]。
- [全部派生表](report/数据表/)及[分析脚本](report/分析脚本/)支持追溯到物理世界、用户、候选期限和模型权重。[首轮 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/results-20261118) 提供历史训练数据与 MPC 结果。

## 下载与重算

从[本版 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/ch6-20261119)下载三个数据或报告附件及 SHA256SUMS.txt。在同一目录运行以下命令，校验后将实验包解压到新目录。

```bash
sha256sum -c SHA256SUMS.txt
mkdir -p chapter6-data
tar -xzf RouteClock_ch6_20261119.tar.gz -C chapter6-data
(cd chapter6-data/routeclock_archives/paired600 && sha256sum -c SHA256SUMS)
```

包内保留 routeclock_archives/paired600、routeclock_runs/paired600_pressure12_4 及中文报告目录的相对位置。主归档包含全部已校验源码、配置、冻结 LightGBM 与 Chronos 权重、模型选择、逐场数据及日志；追加归档包含全部 60 场产物。报告目录的两个相对链接指向对应批次。使用 Linux 并保留这些目录关系，分析脚本即可从归档读取世界结果。完整模型和原始运行结果以 Release 数据包为准。

原始命令与指标中的 /data/chen 和 /data/coding 路径用于保留运行记录。在其他机器重跑科学实验时，需映射原始训练数据、权重、脚本与新输出目录。生成报告的绘图字体放在包内 teacher_report_20261118/assets，重建 PDF 时可按实际解压路径调整字体位置。
