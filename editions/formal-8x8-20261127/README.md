# 8/8 正式实验 · 2026-11-27

16 架飞机共享一个通信基站，8 架有可复用路线，8 架随机机动。本轮比较 14 种预测与误差管配置。最终清单包含 2,760 项任务、2,454 场正式闭环；主结果按十个确认世界配对，历史窗口按五个确认世界配对。

**[完整中文报告](report/REPORT.md) · [可下载 PDF](report/RouteClock-8x8-report.pdf) · [GitHub 仓库既有内容盘点](archive/repository.md) · [完整世界与训练分片下载](https://github.com/mingc5961-alt/five-mpc/releases/tag/formal-8x8-20261127)**

中档预算下，在四核心方案的 3.285% 共同风险处，G8P 达 1,550.15 Mbps，比双方均可调管的 Cnative 高 48.43 Mbps。松档同口径下低 31.71 Mbps。紧档四核心无共同风险点；仅保留这两者另行计算，G8P 在 3.757% 的共同风险下亦落后。各口径的区间、实际感知开销、有效窄束时间、分组误差及 14 种方案的实现细节见报告。

## 新增内容目录

| 路径 | 内容与用途 |
|---|---|
| [report/REPORT.md](report/REPORT.md)、[PDF](report/RouteClock-8x8-report.pdf) | 同一份报告的仓库阅读版与打印版，含过程、机制、全方法主表、跨预算配对、门控与历史消融 |
| [report/plot.py](report/plot.py)、[throughput-delta.png](report/throughput-delta.png) | 十世界配对差及 95% 区间图，用归档表重绘 |
| [report/tight_pair.py](report/tight_pair.py)、[tight_pair.csv](report/tight_pair.csv) | 本次复盘新增的紧档两方案事后对照，沿用原端点、重选工作点及配对重抽方法 |
| [verify.py](verify.py)、[report/audit.txt](report/audit.txt) | 终态、逐任务、账本及统计重算程序与运行输出 |
| [run/](run/) | 正式运行目录的原始 Python、环境/启动脚本、[原计划](run/manifest.full_20261125.json)、[最终清单](run/manifest.json)及[变更协议](run/protocol.txt)。center、fitting、experiments、manifest、supervisor 分别负责模型、拟合、评估、排程设计与执行 |
| [run/results/all_main/](run/results/all_main/) | 14 方案主结果。tight_full 为紧档全扫，middle_loose_native 为中松档十世界原倍率，middle_loose_sweep 为中松档三世界全倍率 |
| [run/results/core_main/](run/results/core_main/) | C、IMM、G8P、Cnative 十世界三档完整端点、工作点、均值和配对区间；与全方法表有重叠 |
| [run/results/core_history/](run/results/core_history/) | 五世界历史窗口、路线独立截短与误差管迁移的世界级配对、样本数和均值 |
| [run/results/main_T/](run/results/main_T/) | 调整前已生成的 13 方案紧档历史快照，终版全方法使用 all_main |
| [run/receipts.tar.gz](run/receipts.tar.gz) | 逐任务 JSON 4,149 份与逐起点 NPZ 450 份，含清单外已完成记录。最终任务 ID 以 manifest.json 筛选 |
| [run/weights/](run/weights/)、[run/worlds/](run/worlds/) | 本轮产生的网络、门控、管形状与尺度因子，以及原始世界元数据、种子、类别、偏离事件和数组指纹 |
| [run/live/](run/live/)、[run/logs/](run/logs/) | 最后心跳、逐任务尝试次数、最终进度文本、生成进度文本的服务器脚本、调度日志及 CPU/CUDA 回放核对 |
| [archive/repository.md](archive/repository.md)、[prior-tree.csv](archive/prior-tree.csv)、[prior-release-assets.csv](archive/prior-release-assets.csv) | 上传前仓库结构、各阶段关系、逐文件基线及历史 Release 附件索引 |
| [archive/source-files.csv](archive/source-files.csv)、[release.sha256](archive/release.sha256)、[expert-pilot-env.sh](archive/expert-pilot-env.sh) | 4,753 个来源文件的逐项记录、存放位置和 SHA-256，Release 包摘要，原环境脚本依赖的上一级入口 |
| [Release 附件](https://github.com/mingc5961-alt/five-mpc/releases/download/formal-8x8-20261127/formal-8x8-state.tar.zst) | formal-8x8-state.tar.zst，493,743,029 字节，内含 14 组世界数组及对应 JSON（含清单外的 3000）、训练/验证缓存和 queue.sqlite3 终态 |
| 仓库根 README.md | 新增本版入口，已有版本仍由原路径访问 |

Release 同时提供可直接下载的 [12 页报告 PDF](https://github.com/mingc5961-alt/five-mpc/releases/download/formal-8x8-20261127/RouteClock-8x8-report.pdf)，与 report/ 中的版本相同。

## 统计复核与恢复

在本版目录运行以下命令。Python 3.12、NumPy、pandas 即可从已存回执重建 38 张正式 CSV，无需 GPU 或 Release 大包。脚本先解包到临时目录，按世界验证共同面板起点一致，重算逐场账本和三个最终统计任务；绝对/相对数值容差各为 10⁻⁹（CSV 重建）。原环境使用 Python 3.12.13、NumPy 2.5.4、pandas 3.1.1。图形重绘另需 Matplotlib 以及仓库根 report/assets 下的字体。

```bash
python verify.py
python report/tight_pair.py
python report/plot.py
```

若需读取完整运行状态，将 Release 附件下载到本版本目录，以如下命令恢复。逐文件 SHA-256 见 archive/source-files.csv，解包前可先核验压缩包。

```bash
sha256sum -c archive/release.sha256
tar --zstd -xf formal-8x8-state.tar.zst -C run
tar -xzf run/receipts.tar.gz -C run
```

若需从模型重新运行物理回合，还需原平台数据、权重、CUDA 和代码路径。正式脚本保留了服务器上的绝对路径。先按 11 月 23 日分阶段归档的 files.csv 恢复 03/04 的观测数据、训练库、模型和校准包、05b/06/07 的模型头及回放依赖，再将本轮 run 放至 /data/chen/formal_8x8_20261125，或等效调整路径。run/env.sh 所需的 /data/chen/expert_pilot_20261125/env.sh 副本在 archive/expert-pilot-env.sh。原任务使用 PyTorch 2.10.0+cu128、SciPy 1.18.1、LightGBM 4.7.0、chronos-forecasting 2.3.2。已存在回执时 worker 跳过对应任务，重新执行单场可在独立副本上直接调用 experiments.cell。此次验证限于归档与统计重算。[前置归档目录](../routeclock-stages-20261123/) · [前置分包 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/routeclock-stages-20261123)

run 中保留已产出的清单外历史回执、结果和权重，便于解释版本调整；Python 字节码、PID、空锁和动态服务进程不作为科学产物发布。先导试验与旧版已发布数据不计入本轮样本量。
