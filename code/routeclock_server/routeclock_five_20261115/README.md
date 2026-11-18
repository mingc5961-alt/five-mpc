# 路线记忆＋短期状态门控的五模型平台

本目录是2026-11-15的新版本。旧版和其权重、实验结果保留在同级 routeclock_platform_20261113。现在的主入口是 tools/five_campaign.py；原有工具随项目保留供溯源。请按 [服务器运行清单](SERVER_RUNBOOK.md) 执行正式实验。[本地验证结果](LOCAL_RESULTS.md) 收录六个共同起点、两组三机服务期、固定基线与风险估计结果。

## 本地结果位置

- runs/fixed_validation/mission_00.json 至 mission_19.json 是正式固定基线全部场景、全部候选；winner.json 为冻结的选择，full_reference_check.json 为未采用快速评分的事件仿真复核。
- runs/five_pilot 的训练、验证、校准及网络pkl含仿真位置与带噪观测生成的因果路线快照。results/forecast.csv、gate_events.csv、historical_gate_events.csv、deadlines.csv 分别是逐时域预测、共同起点门控诊断、测试轨迹内部的时序诊断和窄束期限。
- results/closed_loop.csv 及 loop_* 保存主比较和同一门控的动作消融；results/communication_decomposition.csv 为各任务的时间与速率分解。results/risk_heads.csv 是同一门控位置中心下的真实TabPFN、本地／全量特征LightGBM、简单误差包络和直接期限模型对照。
- 进入Word、PPT或论文的数字宜先对照独立样本数。当前两个网络场景与400条独立测试航线、30个16机网络的正式实验分别报告。

## 复现本地验证

以下命令从本目录运行，使用有依赖的Python。在本机该解释器是 ../routeclock_platform_20261113/.venv/Scripts/python.exe。Chronos使用旧平台 .cache/huggingface/hub 中的真实本地快照；TabPFN通过环境变量ROUTECLOCK_TABPFN_CHECKPOINT指定旧平台 .cache/tabpfn 下的回归器。两项均可直接读取已授权的模型文件。

```text
python tools/fixed_grid.py --workers 4 --verify
python tools/pilot_histories.py --workers 2
python tools/five_pilot.py --checkpoint <本地Chronos-2快照目录>
python tools/pilot_risk.py
python tools/local_report.py --previous ../routeclock_platform_20261113/runs/long_history_deadline_pilot
python -m pytest -q
```

fixed_grid在同一观测间隔下共用事件时间线和IMM预测，对九种阵列逐一计算正确的SNR、MCS和有效比特。其预测使用无新观测情况下各模式联合一阶矩的线性传播。回归测试对18个小场景设置逐一核对，同时有一个独立的完整16机场景复核。运行结束与核对结果见fixed_validation目录。

