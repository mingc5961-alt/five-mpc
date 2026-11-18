# RouteClock / FIVE-MPC 实验与结果

本仓库保存 2026-11-18 完成的 RouteClock 科学代码、完整分析报告与派生结果。大体积正式实验数据和模型权重通过 Releases 提供。

## 阅读报告

- [完整报告 PDF（27 页）](report/RouteClock_实验结果与机制综合报告.pdf)
- [可编辑 Word](report/RouteClock_实验结果与机制综合报告.docx)
- [HTML 报告源码，下载后用浏览器打开](report/RouteClock_实验结果与机制综合报告.html)
- [40 份派生数据表](report/tables)
- [14 张图及 SVG](report/figures)

## 完整实验包

前往 [Releases](https://github.com/mingc5961-alt/five-mpc/releases) 下载三个 `RouteClock_results_20261118.zip.partXX`、`parts_manifest.json` 和 `merge_parts.py`，放在同一目录后运行：

```bash
python merge_parts.py
```

脚本会检查各分卷和合并 ZIP 的 SHA-256。三个分卷缺一不可，不能分别解压。Release 页面自动生成的 Source code (zip/tar.gz) 仅是本仓库快照，不含完整实验数据。完整 ZIP 约 4.08 GB，解压约 13.36 GiB，包含原始结果、模型、代码、预训练权重和证据清单。

## 结果范围与结论

- FIVE：645 场策略运行；MPC：120 场策略运行。策略运行数不等于独立物理世界数。
- 离线数据：1200 条训练、400 条验证、400 条校准、400 条测试路线。
- 143 次刷新回放仅来自两个世界，属于相关事件的探索性诊断。
- 45 场 TabPFN 复核按事后批准的时间预算延后；批准范围完成，原始 690 场 FIVE 矩阵未全部完成。
- RC 的长期预测明显优于 IMM，但当前主闭环通信吞吐量仍低于 IMM；MPC 相对轮询显著节省感知时间，同时存在几何可靠性代价。请结合报告的负结果与适用边界理解。

## 目录

- `code/routeclock_server/`：原科学源码与配置；本轮主要使用 `routeclock_five_20261115` 与 `routeclock_server_full`。
- `code/formal_execution/`：作业编排、资源控制、审计及测试。
- `report/analysis/`：只读分析与报告渲染脚本。
- `provenance/`：完成状态、事后范围调整、依赖记录和源码哈希。
- `download/`：分卷清单、合并校验脚本与操作说明。

仓库不包含虚拟环境。分析脚本的默认路径指向原服务器；复现时传入解压后的正式运行目录，并参照报告第十七节。科学程序按各子项目说明配置环境及权重路径。
