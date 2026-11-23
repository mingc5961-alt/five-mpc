# RouteClock 实验档案 · 2026-11-23

从早期 FIVE/MPC，到 0 dB 链路确认、路线记忆、RC 里程学习和公共管复核。下面的每一页列出实验设计、代码入口、报告或原始分包。

| 阶段 | 研究内容与结果入口 | 页内源文件数 |
|---|---|---:|
| 01 | [FIVE / MPC 初版与服务器复核](stages/01-foundation/) | 207 |
| 02 | [第六章与固定历史修订](stages/02-fixed-history/) | 89 |
| 03 | [2,280 场系统实验与前沿搜索](stages/03-refine/) | 7 |
| 04 | [0 dB 基线、先进预测模型与实飞回放](stages/04-confirmation/) | 145 |
| 05a | [三种先进方法的 LightGBM 对照](stages/05a-matched-tubes/) | 4 |
| 05b | [公共特征与模型专用尺度管](stages/05b-specialized-tubes/) | 13 |
| 05c | [中心准确率、MCS 与跨版本差距归因](stages/05c-gap-analysis/) | 17 |
| 05d | [中心误差和部署时延测量](stages/05d-center-latency/) | 6 |
| 06 | [先进模型的显式路线记忆](stages/06-route-adapters/) | 20 |
| 07 | [RC 沿线进度与路线分支](stages/07-progress-branches/) | 21 |
| 08 | [候选专属里程、辅助监督和新世界复验](stages/08-candidate-mileage/) | 10 |
| 09 | [最新 RC 对公共管和 Chronos 原生管](stages/09-common-tubes/) | 2 |

## 最新结果怎么读

阶段 07 比较原 RC 与新增路线分支/进度机制。阶段 08 比较上轮 RC 与本轮候选里程模型。阶段 09 固定中心再换管，区分 Chronos 原生分位管、有相同输入定义的公共 LightGBM 管，以及含内部诊断的专用管。闭环三档感知上限分别为每 180 秒的 3、4.5、9 秒，价值调度、最早到期刷新，MCS 余量 0 dB。相同历史的中心误差、固定窄束保持和自适应闭环的实际保持分别计分。

实验曾复用相同世界和系统。阶段 09 工作簿中的 150 行含 60 场补测和 90 场复用，不能与阶段 08 的 150 场直接相加。实飞回放以实测航迹为真值，感知噪声及无线信道按配置模拟。各阶段具体口径以该页报告、冻结参数和逐场记录为准。

## 取数与复现

1. [逐文件索引](files.csv) 的每行给出阶段、实验类型、原工作路径、字节数、SHA-256、Release、附件和包内路径。清单有 36,906 条本机文件记录。本次 43 个独立压缩包收录 9,668 份文件记录。历史同内容文件按哈希指向既有正式附件；历史版本另有自己的完整清单。
2. 在[本次 Release](https://github.com/mingc5961-alt/five-mpc/releases/tag/routeclock-stages-20261123)下载相应阶段的分包和 SHA256SUMS.txt。代码、报告可直接从阶段页面查看或下载。模型权重、训练曲线、校准资产、逐锚点评分、逐场账本与固定世界缓存保留在分包内。
3. 在同一个新目录进行校验和解压。已下载子集可用下面的命令恢复。权重采用原始 PyTorch/LightGBM 格式，使用与冻结版本匹配的环境加载。

```sh
sha256sum --check --ignore-missing SHA256SUMS.txt
mkdir -p restored
for f in *.tar.zst; do tar --zstd -xf "$f" -C restored; done
```

4. 科学代码中保留了服务器的绝对路径。运行时将本轮归档的根目录对应到 `/data/chen`，历史包按索引中记录的包内路径恢复；`analysis-tools/` 中的临时分析工具原位于 `/tmp`。环境约束可见正式包的 `requirements-resolved.txt` 和各轮 `env.sh`。原始 11 月 21 日训练观测、上游权重和源码快照由该日 Release 提供。AMOVFLY 的外部原始数据由 `routeclock_runs/confirm_0db/prepare.sh` 的 `fetch_amovfly_index` 固定到提交 `67069ed00ddbebd62b71aa9bb1272415e9b15ff8`，随本版保存了选窗清单、源文件摘要和回放输出。[上游轨迹版本](https://github.com/YujiaoHu/AMOVFLY-Dataset/tree/67069ed00ddbebd62b71aa9bb1272415e9b15ff8)
5. 核对最近两轮，无需从头训练。环境和冻结结果就位后，在 `refine-logs/RC候选里程` 运行 `source env.sh` 和 `"$PY" report.py`；在其 `非专用管对照` 子目录运行 `"$PY" summarize.py`。后者会核对全部逐场账本与起始中心指纹再生成工作簿。重训则从各阶段代码中的训练入口执行，例如阶段 08 的 `mileage.py`；训练和回放应在单独工作副本中进行。
6. `external` 表示外部依赖。本机旧版第三方 TabPFN 预训练检查点若属于此类，可依工作路径与摘要从原提供方获取。打包范围是实验源码、科学输入/缓存和结果；操作系统、Python 虚拟环境、Git 登录资料、下载中间文件与重复迁移镜像不在本版分包中。

## 历史版本

- [results-20261118](https://github.com/mingc5961-alt/five-mpc/releases/tag/results-20261118)，9 个附件
- [ch6-20261119](https://github.com/mingc5961-alt/five-mpc/releases/tag/ch6-20261119)，4 个附件
- [ch6-fixed-history-20261119](https://github.com/mingc5961-alt/five-mpc/releases/tag/ch6-fixed-history-20261119)，7 个附件
- [routeclock-refine-20261121](https://github.com/mingc5961-alt/five-mpc/releases/tag/routeclock-refine-20261121)，38 个附件

[历史附件大小、下载地址和 GitHub 摘要](historical-releases.json) · [本版附件校验](SHA256SUMS.txt)
