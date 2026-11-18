# 空间路线记忆与波束有效期实验平台

本目录独立于旧版 `multi_uav_beam`。方法、数据、训练和结果均不从旧实验目录隐式读取。研究依据见 `implement-stage/RESEARCH_REPORT.md` 与 `configs/research.json`。

## 实现的数据流

1. `world.py` 生成六类外生三维飞行：固定节奏、每圈变速、分段变速、停留、偏离归位、无可复用路线。统一限制速度、加速度和 jerk。量测包含距离、方位、俯仰、径向速度，返回位置协方差。世界实例由独立命名空间确定，方法与模型种子不改变世界。
2. `filters.py` 实现三维 CV／左转／右转自适应IMM与一维巡航／减速／停留进度IMM。后者用连续时间模式转移及OU精确离散化处理不等间隔观测。
3. `memory.py` 只从真实观测建立有向片段，以位置、航向、进度连续性匹配；保留交叉点多候选、观测到的片段连接、路径闭合证据。正权重sigma点把进度与匹配歧义传播到位置。
4. `predictors.py` 提供I、P、H、C、CR_raw及RC/CR门控。H是周期与漂移网格的在线状态空间专家。门控是16维、随预测时长变化的logistic融合器，直接优化融合位置的二维角误差。
5. `foundation.py` 接入真实 `amazon/chronos-2` 与指定 `TabPFN V2_5`。Chronos冻结权重，历史路线协变量来自当时发布的快照；未来协变量为当前路线模型的预测。无路线C完全移除路线字段。模型失败直接报错，不使用其他输出代替。
6. `evidence.py` 保存发布时的预测包。新量测先检查旧预测，再更新状态和地图。无新量测时只推进预测年龄，不重置原点。
7. `validity.py` 训练轴—时长共享的误差头：60个诊断特征，追加h/10与两个轴one-hot，共63维。标签是整段最大角误差；独立任务校准两轴与多个时长的共同最大归一化误差。另有经验包络、LightGBM、直接期限头与Zhang式SNR下界触发对照。
8. `radio.py` 实现半波长UPA、9种子阵、总功率固定、7档MCS、BLER与实际有效比特。角误差矩形转为方向余弦外包范围，两个轴的增益乘积形成二维共同损失下界。
9. `scheduler.py` 生成候选束宽与期限，按共享感知负载和预测通信价值选择动作。EDF执行观测，统一加宽与单动作限制用于消融。Token bucket计入每次3.1ms真实观测成本。
10. `simulator.py` 执行不与20ms真值网格混淆的事件时间线。感知、切换、传输分开计时；实际传输比特除以服务墙钟时间。每次观测与每个传输片段均有可追踪记录。

## 本地启动

在本目录使用 `.venv/Scripts/python.exe`（Windows）或 `.venv/bin/python`（Linux）。默认使用CPU，GPU环境显式传 `--device cuda`。不要在旧虚拟环境安装新依赖。

Linux GPU环境可用 `uv venv .venv --python 3.12`，然后 `uv pip install --python .venv/bin/python -e . pytest --torch-backend auto`。模型权重放在同一 `.cache` 布局，或通过 `ROUTECLOCK_TABPFN_CHECKPOINT` 指定回归器权重。

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m routeclock.cli generate --profile local --data data/acceptance
.\.venv\Scripts\python.exe -m routeclock.cli train --data data/acceptance --head lightgbm --out runs/acceptance_models
.\.venv\Scripts\python.exe -m routeclock.cli local --bundle runs/acceptance_models/bundle.pkl --methods I,RC --out runs/acceptance_loop
```

`local` 是60秒建图、12秒服务、3机的小规模执行配置，缩小任务量而保留完整方法。数据为24训练、12验证、12校准、12测试任务；学习结果只用于本地流程与行为诊断。`model-smoke` 是12秒建图、3秒服务的真实大模型接口配置，专门减少CPU调用数量。`formal` 使用报告的180秒建图＋180秒服务及2400个独立单机任务。

## 真实模型

Chronos权重在新目录 `.cache/huggingface`。运行 `tools/probe_models.py` 可查看两个真实模型的独立通过／失败记录。该脚本不把失败伪装为模型输出。

TabPFN源码不是模型权重。需要账户已接受许可的 `tabpfn-v2.5-regressor-v2.5_default.ckpt`，放入 `.cache/tabpfn`，或设置 `ROUTECLOCK_TABPFN_CHECKPOINT` 为本地权重路径。当前显式固定V2_5，不使用安装包的新版默认模型。不要把账户令牌放进结果文件或提交到代码中。

```powershell
.\.venv\Scripts\python.exe -m routeclock.cli generate --profile model-smoke --chronos --data data/real_models
.\.venv\Scripts\python.exe -m routeclock.cli train --data data/real_models --head tabpfn --out runs/real_models
.\.venv\Scripts\python.exe -m routeclock.cli local --profile model-smoke --bundle runs/real_models/bundle.pkl --methods I,C,CR,RC --out runs/real_model_loop
```

所有正式方法都有各自的误差头，不能拿RC的误差头冒充C/CR。用LightGBM完成的真实Chronos小闭环会明确标为LightGBM，不计作TabPFN主方法结果。

## 正式执行顺序

P0：运行测试、`tools/geometry_diagnostics.py`、真实模型探针及 `tools/profile_models.py`。模型性能配置指定Chronos720历史步、TabPFN7200行、4个estimators及16查询；工具不会自动缩小规模。

A：`generate --profile formal --chronos` 生成全部split；分别以 `--head tabpfn` 与 `--head lightgbm`，以及种子7/11/19训练。数据生成与模型训练为分开的阶段，已经计算的真实专家预测可以供相同预算的下游头使用。

```powershell
.\.venv\Scripts\python.exe -m routeclock.cli generate --profile formal --chronos --device cuda --data data/formal
.\.venv\Scripts\python.exe -m routeclock.cli train --data data/formal --head tabpfn --device cuda --seed 7 --out models/tabpfn_seed7
.\.venv\Scripts\python.exe tools/offline_ablation.py --data data/formal --bundle models/tabpfn_seed7/bundle.pkl --device cuda --out runs/A_ablations
.\.venv\Scripts\python.exe -m routeclock.cli select-fixed --profile formal --bundle models/tabpfn_seed7/bundle.pkl --count 20
```

先在网络validation任务上选择固定方案F，再运行测试。正式网络任务使用新的命名空间，绝不重用单机训练实例。

B/C/D：`python -m routeclock.cli matrix --out runs/design` 写出830项正式网络工作单；其中包含报告补充的20项Zhang式对照。按块执行，不扫描笛卡尔积：

```powershell
.\.venv\Scripts\python.exe tools/campaign.py --bundles models --out runs/formal --block B --device cuda
.\.venv\Scripts\python.exe tools/campaign.py --bundles models --out runs/formal --block B_Z --device cuda
.\.venv\Scripts\python.exe -m routeclock.cli summarize --out runs/formal
```

其他块：C_N、C_budget、C_actions、C_scheduler、C_reuse、D_archive、D_lgb、D_seed。种子11/19与LightGBM需先训练对应目录。实验失败会直接中止，不会静默跳过后继续汇总为完整矩阵。

额外80次单机动作诊断使用 `--block single_actions`，先运行 `select-fixed --fixed-center RC --uavs 1` 在验证任务上选择同一RC预测器的固定动作。四种动作使用这个相同参考宽度/间隔。网络C块的单动作消融预先固定16×16子阵及1秒间隔。

## 消融与可解释性

- A1比较7个中心，按六种行为及预测时长输出位置与角度误差。
- A2固定中心比较完整／局部／经验包络／无档案尺度头。`offline_ablation.py` 另在局部状态近邻中置换长期证据，并重拟合头。
- A3直接期限头对比误差区域产生的期限，评价期限失效与实际单机传输收益。
- D的无档案版本同时删除门控与尺度头历史证据，分别重训；不同于A2固定中心消融。
- `tools/real_trajectory.py` 接收有时间戳的xyz或GPS CSV，按时间顺序匹配真实轨迹。GPS轨迹只验证路径匹配，不作为28GHz链路吞吐证据。
- 进度点估计消融用 `generate --point-progress`；多边形OOD用 `generate --shape polygon --split test --count 20`。两者写入独立数据目录，重新训练需要重训的模块。

预测标签只在离线训练或评价器中读取真值。线上Tracker不接收World对象、真实类别、参考路线和真实圈数。

## 输出与解释

`bundle.pkl`：门控、独立校准的各方法误差头、消融头和训练元数据。

`centers_raw.csv` / `centers_summary.csv`：预测中心误差；`fixed_center_validity.csv`：固定中心覆盖与区域宽度；`actions_A3.json`：期限与传输诊断。

每次闭环运行目录包含 `metrics.json`、`events.json`、`predictions.json`。时间账本自动检验“感知＋切换＋数据＝服务期”；事件数据可重算比特、超期、失联与资源贡献。统计采用独立任务配对bootstrap，不能把同一轨迹的时刻当成独立任务。

正式结果需要完整测试任务运行后产生。本地数字表明模块在给定小场景中的实际行为，不代替会议论文主实验。

## 本次交付的验收入口

最终实施说明见 `IMPLEMENTATION_REPORT.md`；机器可读状态见 `runs/FINAL_VERIFICATION.json`。以 `data/final_real`、`runs/final_tabpfn_models`、`runs/final_lgb_models` 和 `runs/final_*_verification` 为本次最终验收版本。其余目录保留开发过程。

```powershell
.\.venv\Scripts\python.exe tools/verify_closed_loop.py --bundle runs/final_tabpfn_models/bundle.pkl --out runs/recheck --methods I,C,CR,RC --uavs 6
```
