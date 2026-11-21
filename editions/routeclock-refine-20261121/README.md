# RouteClock 无人机波束跟踪实验 · 2026-11-21

本次发布归档 2,280 场合成闭环实验、610 个前沿搜索评分、两组公开航迹数据、预训练与适配权重，以及报告和逐场工作簿。

- [完整分析报告（PDF）](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-refine-20261121/RouteClock-report.pdf)
- [可编辑报告（Word）](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-refine-20261121/RouteClock-report.docx)
- [逐场结果与配对分析（Excel）](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-refine-20261121/RouteClock-results.xlsx)
- [逐文件索引（原始路径、所属分包、字节数和 SHA-256）](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-refine-20261121/files.csv)
- [附件校验值](https://github.com/mingc5961-alt/five-mpc/releases/download/routeclock-refine-20261121/SHA256SUMS.txt)

| Release 附件 | 内容 |
| --- | --- |
| rc-simulation-results | 全部 2,280 场结果、独立搜索、复现、离线结果、日志、CMU 历史实验及短窗无线回放 |
| rc-reports-and-analysis | 中文原名报告、实验计划、逐场进度与资源记录，以及分析表、校正后的 CMU 数据和生成脚本 |
| rc-trained-weights | 自行训练或适配的权重、误差管、融合头与校准资料，含 VECTOR 种子 7、11、19 |
| rc-upstream-* | Chronos-2、Moirai-2、含原始 VECTOR 检查点的源码快照、uni2ts 和本轮下载的依赖轮子 |
| rc-cmu-flights、rc-px4-flights-* | CMU 原始飞行 ZIP、元数据，及 PX4 的 200 份原始 ULog |
| rc-inputs-metadata、third-party-notices.zip | 来源、抽样和下载清单、模型卡与上游许可说明 |
| rc-training-observations | 合成训练、验证与校准观测 |
| rc-world-cache-*、rc-warm-history-* | 本次生成的世界物理场与共同预热缓存 |
| rc-experiment-code | 本轮运行的实验及调度代码快照 |

带数字后缀的每个 tar.zst 均可独立解压。将需要的附件下载到同一目录，使用下列命令检查对应的文件，再解压所需分包。

```sh
sha256sum --check --ignore-missing SHA256SUMS.txt
mkdir -p restored
for f in rc-*.tar.zst; do tar --zstd -xf "$f" -C restored; done
```

逐场 CSV 和工作簿保留原服务器的绝对路径；恢复后将 /data/chen/ 映射至 restored/ 即可定位结果。代码快照在 restored/code/routeclock_refine，个别运行脚本引用先前已归档的 paired600。源文件保留原许可证和版权信息。CMU 本次只在一个开发架次做无线回放；PX4 主要完成筛选。真实数据的验证范围、合成结果与统计口径见报告。
