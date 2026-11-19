# 固定起始历史修订版 · 2026-11-19

同一世界的各组都从相同时间戳与量测值出发，全站共 2,872 次观测，每架 179–180 次。服务阶段继续运行闭环。三个批次合计 2,080 条科学运行记录，物理测试世界 30 个，Chronos 对照使用其中相同的 10 个。581 次新执行包含 520 次原设计补充、60 次同动作 F 对照和 1 次基点数值复核。基点的 140 条旧记录在两套协议之间复用。

- [14 页修订报告](report/RouteClock_第六章完整实验分析报告.pdf) · [Word](report/RouteClock_第六章完整实验分析报告.docx) · [数据工作簿](report/实验数据与配对结果.xlsx)
- [本次新增源码](tools/) · [相同版本的基础仿真平台](../ch6-20261119/platform/) · [任务清单与源码指纹](evidence/) · [原分析脚本](report/分析脚本/) · [逐场数据与其他明细](report/数据表/)
- [给独立复核者的最小包](https://github.com/mingc5961-alt/five-mpc/releases/download/ch6-fixed-history-20261119/RouteClock_Claude_review.zip) · [阅读顺序和复核任务](review/先读我.txt) · [实际重算输出](review/verified_output.txt)

## 如何解释修正

5% 预算下，从 9.3 到 12.4 ms，RC−I 的变化量在原协议下为 −67.04 Mbps，统一初始历史后为 −25.45 Mbps，后者的逐世界 95% 区间为 [−59.50, +6.36]。两协议的变化量之差为 +41.59 Mbps [26.23, 55.63]。在相同的 12.4 ms 服务成本下，替换预热历史使 RC 平均增加 97.24 Mbps。最紧预算的 10 世界 Chronos 子集中，固定历史后的 CR_raw−C 为 −24.17 Mbps [−38.59, −9.91]。起始历史的比较包含历史量、间隔和末次观测新鲜度的共同变化。

## 下载与原始数据复核

[本版 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/ch6-fixed-history-20261119) 的 RouteClock_fixed_history_20261119.tar.gz 包含本次新增代码、逐场原始指标和 RC 权重、起点预测、日志、协议、修订分析代码、更新后的合并数据和报告。SHA256SUMS.txt 列出各附件校验值。基础权重、初版 1,500 场原始记录和已冻结的平台见[上一版完整实验包](https://github.com/mingc5961-alt/five-mpc/releases/tag/ch6-20261119)。

如需核对全部原始记录，将上版 RouteClock_ch6_20261119.tar.gz 和本版补充包依次解压到同一个新目录。可在 routeclock_runs/fixed_history_20261119 内运行 sha256sum -c SHA256SUMS，在中文报告目录内运行 sha256sum -c 报告交付_SHA256SUMS。原始仿真与归档分析脚本保留服务器绝对路径，异机完整复跑需要映射目录。小包内的 verify.py 可用相对路径从 CSV 复算核心结果。

## 复核范围

从 PDF、逐场表和两份诊断明细开始，再查预热入口、缓存、调度、测量、指标定义和配对分析。若要按飞行器类型做新的分层分析，可另取 [逐机资源合并表](report/数据表/逐机资源合并.csv)。所有结论都应注明样本数、配对单位和预热协议。两个阶段的解释和第三方复核优先问题见复核包的“先读我.txt”。
