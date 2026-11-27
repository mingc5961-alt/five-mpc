# 原版 G1：首次付费定位下的预测与波束调度

本版本收录 7/9 机群的硕士论文主线、四组中心与包络实验，以及对既有预测的窄束保持期复核。研究结论是保留原版 G1。新诊断读取同一批已完成的仿真，在既有中心和管宽上增加物理读数，未训练新模型。

**阅读顺序**　先读[论文主线](THESIS.md)，再读[保持期的定义、直接检验与推导边界](HOLD.md)。[完整实验附录](appendix/FACTORIAL.md)保留逐项公式、实验设置、倍率开关、按已有定位数的优势秒数与负结果。[附录 PDF](appendix/factorial-report.pdf)及[结果总图](appendix/factorial-results.png)可直接下载。

## 本次上传了什么

| 文件或目录 | 内容与用途 |
|---|---|
| [THESIS.md](THESIS.md) | 硕士论文正文。按首条定位、运动与路线专家、已兑现反馈、保持期准入、多机调度展开，说明创新归属和四组分别支持什么。 |
| [G1-thesis.docx](G1-thesis.docx) | 与正文对应的可编辑 Word，保留数学公式与实验表格。 |
| [HOLD.md](HOLD.md) | 新的机制核验。分别报告真实寿命、准入、漏用、超额与交集，比较原 G1 与 Chronos，推导有限条件下的预算阈值。 |
| [appendix](appendix/) | 原四组配对实验的全文、PDF、总图。正文使用相同的风险、角度和计时定义。 |
| [evidence/work/factorial_run](evidence/work/factorial_run/) | 世界级 CSV、共同前缀评分数组、逐回合回执与冻结配置。文件说明见下表。 |
| [evidence/work/g1mix7](evidence/work/g1mix7/) | 32 份复用的原 G1 与 Chronos 倍率开发回执。与同场四组比较时使用相同世界指纹。 |
| [evidence/work](evidence/work/) | NumPy 复算脚本；另附实验生成、误差头、调度调用和物理诊断代码，便于核对算法与实现。 |
| [模型](evidence/model_sha256.json)、[场景](evidence/world_hashes.json)指纹及 [SHA256SUMS](SHA256SUMS) | 定位原 G1、种子 7 微调 Chronos 的版本，并核验本版本发布文件的字节。 |

## 如何读结果表

以下表格均位于 [factorial_run](evidence/work/factorial_run/)。

| 文件 | 指标口径 |
|---|---|
| [confirm_center.csv](evidence/work/factorial_run/confirm_center.csv)、[confirm_coverage.csv](evidence/work/factorial_run/confirm_coverage.csv)、[confirm_area_contrasts.csv](evidence/work/factorial_run/confirm_area_contrasts.csv) | 同前缀的端点角误差、路径覆盖、同覆盖面积和配对区间。 |
| [confirm_frozen.csv](evidence/work/factorial_run/confirm_frozen.csv)、[confirm_paired.csv](evidence/work/factorial_run/confirm_paired.csv) | 开发集按 7% 选好倍率后，冻结策略在新世界的实际风险和交付。固定 ×1 单列。 |
| [confirm_equalrisk_levels.csv](evidence/work/factorial_run/confirm_equalrisk_levels.csv)、[confirm_equalrisk.csv](evidence/work/factorial_run/confirm_equalrisk.csv)、[samegrid_result.csv](evidence/work/factorial_run/samegrid_result.csv) | 测试场景按实际风险插值的端点、混合概率、配对区间；原 G1 与 R1 的相同倍率网格敏感性检验。 |
| [prefix_winners.csv](evidence/work/factorial_run/prefix_winners.csv)、[prefix_crossings.csv](evidence/work/factorial_run/prefix_crossings.csv) | 给定已有定位条数，在未来哪些半秒节点各模型领先，Chronos 在何处反超。 |
| 以 shadow、noise 开头的表 | 原 G1 自主刷新时间表上的共同前缀评分，以及仅在仿真中可算的新息分解。 |
| [hold_prefixes.csv](evidence/work/factorial_run/hold_prefixes.csv)、[hold_worlds.csv](evidence/work/factorial_run/hold_worlds.csv)、[hold_levels.csv](evidence/work/factorial_run/hold_levels.csv)、[hold_contrasts.csv](evidence/work/factorial_run/hold_contrasts.csv) | 6444 个起点、五种既有方案、两个相同倍率的 64440 条保持期读数及其统计结果。 |
| [hold_chronos_panel.csv](evidence/work/factorial_run/hold_chronos_panel.csv)、[hold_panel_levels.csv](evidence/work/factorial_run/hold_panel_levels.csv)、[hold_panel_contrasts.csv](evidence/work/factorial_run/hold_panel_contrasts.csv) | 1152 个相同起点，原 G1 与微调 Chronos 各用两个原工作倍率，逐发布读取有效期并与闭环指标配对。 |
| run_confirm、run_risk、sensitivity_grid 开头的 JSON | 104 个主确认回合、64 个新增倍率开发回合、16 个同网格敏感性回合。加上旧开发回执共 216 份。最终确认仍为 8 个独立世界。 |

## 复算

下载此目录，在一个副本的 `evidence/work` 中，用带 NumPy 的 Python 依次执行以下命令。评分数组和闭环回执均已随版提供。脚本会在同目录的 `factorial_run` 中重建对应的 CSV。

```sh
python factorial_stats.py
python factorial_grid_stats.py
python hold_summary.py
python hold_panel_summary.py
```

前两项复算四组和倍率敏感性，后两项从逐前缀物理读数复算保持期。若要从原仿真再生成逐前缀读数，还需在原科学运行环境中恢复世界、G1 与 Chronos 检查点、原误差头和已拟合的条件头。本版发布统计复算所需资料，未随版复制大型预训练权重。[基础模型与依赖说明](../gru-chronos-irregular-4505/REPRODUCE.md)。

当前结论基于从空历史开始、每次 3.1 ms、16 机共享 180 秒服务期内约 3 秒定位预算的设置。本版本与[同日 8/8 三预算研究](../formal-8x8-20261127/report/REPORT.md)分别报告。
