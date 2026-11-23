# 0 dB 基线、先进预测模型与实飞回放

这一轮的 2,505 份逐场完成记录由 360 场基线、180 场开发、1,080 场确认和 885 场 AMOVFLY 单机回放组成。包括 Chronos-2、VECTOR、RAFT 的训练曲线、冻结种子权重、误差管与 0 dB MCS 链路。AMOVFLY 提供真实轨迹；感知和无线链路由协议模拟。

本页收录 145 个可在 GitHub 查看和比较的源文件。[打开代码目录](code/)

## 报告、参数与工作簿

- [RouteClock_0dB_结果分析报告_首版_20261123.docx](results/refine-logs/RouteClock_0dB_%E7%BB%93%E6%9E%9C%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A_%E9%A6%96%E7%89%88_20261123.docx)
- [RouteClock_0dB_结果分析报告_首版_20261123.pdf](results/refine-logs/RouteClock_0dB_%E7%BB%93%E6%9E%9C%E5%88%86%E6%9E%90%E6%8A%A5%E5%91%8A_%E9%A6%96%E7%89%88_20261123.pdf)
- [scope.json](protocol-and-tables/routeclock_runs/confirm_0db/scope.json)

## 原始文件下载

- [04-confirmation-cache-01.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-cache-01.tar.zst)
- [04-confirmation-cache-02.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-cache-02.tar.zst)
- [04-confirmation-cache-03.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-cache-03.tar.zst)
- [04-confirmation-cache-04.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-cache-04.tar.zst)
- [04-confirmation-cache-05.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-cache-05.tar.zst)
- [04-confirmation-experiment-01.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-experiment-01.tar.zst)
- [04-confirmation-weights-01.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-weights-01.tar.zst)
- [04-confirmation-weights-02.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-weights-02.tar.zst)
- [04-confirmation-weights-03.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/04-confirmation-weights-03.tar.zst)
- [analysis-tools.tar.zst](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-stages-20261123/analysis-tools.tar.zst)

原始工作路径、SHA-256 和所属分包可在[逐文件清单](../../files.csv)中按本阶段标识筛选。浏览用代码按原路径分组；运行时以归档内的完整布局为准。
