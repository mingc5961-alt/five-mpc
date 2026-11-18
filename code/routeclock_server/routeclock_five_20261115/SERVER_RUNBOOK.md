# 24 GB 单卡服务器执行清单

本版主模型为长历史重训的空间路线＋短期运动门控。五中心依次为 I（自适应IMM）、P（空间路径＋进度）、RC（本文门控）、C（纯Chronos-2）和 CR_raw（使用同一份路径预测的Chronos-2）。固定策略F使用IMM中心。正式主表统一使用验证集选型的局部特征LightGBM，每个预测中心分别训练、分别校准。旧版CR代表另一个门控，不列入五模型主表。

## 迁移前准备

把本目录的 routeclock、tools、configs、tests、pyproject.toml，以及 runs/fixed_validation、runs/five_pilot/results/bundle.pkl 一同复制。基线优胜设置用20个独立的16机验证场景产生，源数据和全部候选分数同目录。带上先导模型用于首日计时；正式网络测试读取重新训练的正式模型。权重复制已获授权的TabPFN **回归** checkpoint和Chronos-2本地快照（配置与safetensors），或按原仓库布局复制HF缓存。工程根目录为以下命令的工作目录。

使用带CUDA的PyTorch环境。Windows CPU依赖清单中的torch带有+cpu，不能照搬为GPU包。当地验收版本为Python 3.12、chronos-forecasting 2.3.2、tabpfn 9.2.0、lightgbm 4.7.0。下面为Linux shell示例，示例模型目录按实际挂载点替换。

```bash
python -m pip install -e . pytest
export ROUTECLOCK_TABPFN_CHECKPOINT=/models/tabpfn-v2.5-regressor-v2.5_default.ckpt
export CHRONOS_DIR=/models/chronos-2
export OPENBLAS_NUM_THREADS=1
python -c "import torch; print(torch.__version__,torch.cuda.is_available(),torch.cuda.get_device_name())"
python -m pytest -q
python tools/five_campaign.py matrix
```

## 首日测速，先决定续租多久

1. 单模型实测。以下小程序测720个历史步的两版Chronos、7200行上下文的真实TabPFN，分别计初始化、首次调用、热调用中位数和P95、峰值显存。输入是用于加载测时的形状样本，不产生学术结果。共同局部误差头16列；完整路线诊断头63列。
2. 端到端实测。timing另建day1命名空间，跑六条不同观测密度和行为的完整离线航线，以及五个预测器各一场完整的16机180秒预热＋180秒服务。用先导LightGBM控制策略估算墙钟占用，首个正式任务完成后以正式观测次数和时长再更新。小样本策略会改变观测次数。
3. 记下模型与CUDA版本、GPU型号、CPU核数、峰值显存、磁盘占用、冷启动与热调用秒数。分组按实际耗时外推，公式为各类剩余任务数乘该类平均秒数之和，另算拟合、校准与写盘。固定基线验证已在CPU完成。主测试各方法30场，动作消融90场，比例扫描120场；后两块只有物理专家或门控，不能按Chronos速度计价。

```bash
python tools/day1_profile.py chronos --checkpoint "$CHRONOS_DIR"
python tools/day1_profile.py tabpfn --features 16
python tools/day1_profile.py tabpfn --features 63
python tools/five_campaign.py timing --checkpoint "$CHRONOS_DIR" --bundle runs/five_pilot/results/bundle.pkl
```

单模型结果在 runs/day1；端到端结果在 runs/five_formal/day1/times.json。单模型探针和一次完整任务的耗时请分别记录。仿真通信吞吐目前不扣除模型执行延迟；真实耗时另表报告。

## 正式顺序

### 1. 离线航线与长历史门控训练

```bash
python tools/five_campaign.py generate --checkpoint "$CHRONOS_DIR"
python tools/five_campaign.py train --head lightgbm --seed 7
```

1200训练、400验证、400校准、400测试，按航线任务划分。每条航线单独落盘，支持指定 --split train --start 0 --stop 20（左闭右开）及断点续跑。不同分片不要并发写同一任务。每次历史观测更新空间地图、物理专家和发布档案；两个冻结Chronos在评测锚点读取当时已有历史预测快照。五中心中的门控不用Chronos兑现误差，五个共同误差头仅用局部状态。在线阶段则在本机实际需要刷新对应预测器时运行Chronos。正式误差训练每个锚点抽一个时长、两个角度轴，得到7200行训练上下文。门控按任务三折产生下游训练时使用的样本外预测。首轮保留全部门控权重和测试逐时域误差，无需因胜负调整基线。

读取 runs/five_formal/lightgbm_7 的 forecast.csv、gate_events.csv 和 deadlines.csv。分别看位置/角度误差、按时长的路线权重和两个专家误差、32×32窄束事后可用期限与模型准入期限。gate_events按实际模拟偏离与局部转向标签统计；缺少某类事件时报告样本数。多个锚点先在同一航线聚合，再做配对比较。

### 2. 固定中心，比较误差模型

LightGBM训练同时生成主模型的简单包络、直接期限以及完整诊断对照。用同一份离线航线做TabPFN复核。

```bash
python tools/five_campaign.py train --head tabpfn --seed 7
```

分别查看对应的deadlines.csv，以固定中心和相同信息量比较误差头。共同局部尺度与带路线诊断的完整尺度分开列；直接期限使用完整诊断，宜与完整尺度头对照。比较获准比例、联合覆盖、平均半宽和获准窄束提前失效。直接期限没有角度覆盖或宽度指标。

### 3. 16机主闭环

```bash
python tools/five_campaign.py network --block main --checkpoint "$CHRONOS_DIR" --bundle runs/five_formal/lightgbm_7/bundle.pkl
python tools/five_campaign.py summary
```

30个外生网络任务，每个任务跑F与五种预测器，共180场。统一读取已选32×32、2秒固定方案。主场景16机、感知占时预算5%、180秒预热＋180秒服务。实际观测按各自策略发生；按场景ID做配对。死亡线为持有段内阵列损失超过3 dB；同时区分波束过期、SNR中断和无可用动作。合并统计报告每个网络的失效比率差和任务级95%区间，可与预定±2个百分点的等效范围比较。

### 4. 收益归因及复核

```bash
python tools/five_campaign.py network --block actions --bundle runs/five_formal/lightgbm_7/bundle.pkl
python tools/five_campaign.py network --block reuse --bundle runs/five_formal/lightgbm_7/bundle.pkl
python tools/five_campaign.py summary
```

actions用同一个RC中心，只换固定两项、仅调宽、仅调间隔三种方式；联调行复用main。三种方式用完全相同的固定参考值。reuse在0%、50%、100%可复用比例分别比较RC与IMM，每点20个网络。标准主场景有14/16具备可复用结构。各方法共用场景身份和噪声规则，每个新比例作为独立实验条件。

补充复核先用相同20个场景跑第二误差头；门控随机性按7/11/19重新拟合。以下用11为例，19同理。

```bash
python tools/five_campaign.py network --block repeat --head tabpfn --checkpoint "$CHRONOS_DIR" --bundle runs/five_formal/tabpfn_7/bundle.pkl
python tools/five_campaign.py summary --head tabpfn
python tools/five_campaign.py train --head lightgbm --seed 11
python tools/five_campaign.py network --block repeat --head lightgbm --seed 11 --checkpoint "$CHRONOS_DIR" --bundle runs/five_formal/lightgbm_11/bundle.pkl
python tools/five_campaign.py summary --head lightgbm --seed 11
```

network支持按块内作业序号 --start / --stop 分片。仅在指标与账本写完后生成completed.json。summary按任务配对，有效场数见paired_tasks；不把不完整块当正式全量结果。每块各方案首个场景保留events/predictions/choices；后续场景保留指标和RC逐时长权重压缩档案。所有结果写在新的 runs/five_formal。不要用旧版tools/campaign.py替代五模型入口。

