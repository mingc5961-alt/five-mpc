# GitHub 仓库盘点（本次上传前）

2026-11-27 核对公开仓库 mingc5961-alt/five-mpc 的 Git 树以及 GitHub 分支、标签、Release 与附件 API。上传前唯一分支为 main，提交 1d7d752af7ee671158f92a8fd592f24c62a453ad，5 个标签对应 5 个已发布 Release。本次创建的版本单列在[新增目录](../README.md)。[基线提交](https://github.com/mingc5961-alt/five-mpc/tree/1d7d752af7ee671158f92a8fd592f24c62a453ad)

| 基线位置 | 跟踪文件数 | 内容 |
|---|---:|---|
| editions/ | 817 | 5 个版本目录，列于下表 |
| code/ | 213 | 早期 RouteClock 服务端科学代码与 formal_execution 编排 |
| report/ | 85 | 首轮中文 PDF、Word、HTML、派生表、制图分析与字体 |
| refine-logs/ | 7 | 历史实验计划、执行与跟踪记录 |
| provenance/ | 4 | 来源、完成状态和归档记录 |
| download/ | 3 | 首轮分卷索引、合并脚本及说明 |
| 根目录 | 2 | README.md 与 .gitignore |
| **合计** | **1,131** | 每个文件对应的 Git blob、字节数见逐文件索引 |

| 已有版本 | 已有内容 | 与本轮的关系 |
|---|---|---|
| results-20261118 | 首轮 FIVE 与 MPC，根目录报告、原始结果与模型分卷 | 早期平台与指标来源 |
| ch6-20261119 | 第六章闭环平台、网络验证、独立测试与成本压力实验 | 共享闭环评估的背景 |
| ch6-fixed-history-20261119 | 固定起始观测修订、逐场结果与复核包 | 区分共同起点与自适应刷新 |
| routeclock-refine-20261121 | refine 实验、PDF/Word/Excel、文件清单 | 训练及校准档案的前置版本 |
| routeclock-stages-20261123 | 01 基础、02 固定历史、03 refine、04 确认、05a 匹配管、05b 专用管、05c 差距、05d 中心与延迟、06 路线适配、07 进度与分支、08 候选里程、09 公共管 | 本轮调用 04 的 Chronos/VECTOR/RAFT、05b 的尺度头、06 的回放和 07 的联合模型 |
| gru-chronos-irregular-4505 | main 中独立目录；10/16 无复用、两个世界、32 个闭环回执，4.505208% 工作点 | 历史对照。GRU 比固定原生管高 162.73 Mbps，允许 Chronos 一起调管则低 86.57 Mbps |

前五行均有 Release，首轮材料放在仓库根目录。后五行在 editions/ 下有目录。以下是上传前所有已发布 Release 附件（不计 GitHub 自动生成的源码快照）。

| 标签 | 附件数 | 总字节 |
|---|---:|---:|
| results-20261118 | 9 | 4,082,819,073 |
| ch6-20261119 | 4 | 681,281,767 |
| ch6-fixed-history-20261119 | 7 | 936,456,780 |
| routeclock-refine-20261121 | 38 | 5,505,091,577 |
| routeclock-stages-20261123 | 47 | 10,472,522,351 |

合计 105 个附件。逐项索引列出标签、文件名、大小、GitHub 返回的摘要及下载 URL；本次盘点读取了远端元数据，未逐一下载历史附件。[基线文件索引](prior-tree.csv) · [历史附件索引](prior-release-assets.csv)
