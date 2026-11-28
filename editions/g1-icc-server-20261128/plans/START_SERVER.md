# 给服务器 Codex 的 G1 恢复指令

> 使用本包找回的原始模块、冻结权重和误差管恢复 G1。先用空目录独立重放 7/9 冷启动确认世界 0 的 G1 ×0.45，记录逐次定位、物理世界指纹和验收差异。通过后启动 N01 的八个新开发世界和四个倍率。用全量归档或服务器已有的同指纹 Chronos 资产，另行核验 Chronos ×0.75。按之前交接的实验规划继续其他对照。保留已完成的 Chronos 压测记录，正式仿真和压测分别计数。

## 1　启动 G1

在服务器已有科学环境下运行。参考环境为 Python 3.12，PyTorch、NumPy、LightGBM 等版本见原 CPU 包的说明和 `gru4505/requirements-replay.txt`。将下面的 PATCH 指向本包解压目录，RUN 指向服务器新建的实验目录。RUN 与解压目录分开。

上传 ZIP 后可用 `python -m zipfile -e G1-runtime-recovery.zip .` 解压，保留源目录的 UTF-8 文件名。在解压后的 `g1-runtime-recovery` 下运行 `sha256sum --quiet -c SHA256SUMS` 可一次核对所有交付文件。

```bash
PATCH=/absolute/path/to/g1-runtime-recovery
RUN=/absolute/path/to/g1_runs/recovery_01
REPO="$PATCH/reference"
CPU="$PATCH/gru4505"
mkdir -p "$RUN"
python -B "$PATCH/tools/experiment_env.py" --repo "$REPO" --cpu-root "$CPU" --runtime "$RUN/runtime"
python -B "$PATCH/tools/replay_anchor.py" --repo "$REPO" --cpu-root "$CPU" --runtime "$RUN/runtime" --out "$RUN/anchor_g1" --arm G1
```

`reference/` 带有从提交 `96b6e630e02c5fc709aef32e5a655b33ddf2463c` 导出的 7/9 脚本、模型指纹和历史参照。也可以将 REPO 指向含该提交的 five-mpc Git 仓库。辅助程序把科学脚本装配到 RUN 下，并在回放进程内设置 Linux 资源路径。使用过的 anchor 输出目录须保留，复跑时用新目录。

**G1 的验收值**为 848 次定位、2.6288 s 感知、1338.95195 Mbps、7.42847% 总风险、58.9878 s 合格窄束数据。检查 `anchor_g1/verification.json` 中的 PASS，同时核对世界指纹、逐事件用户和完成时间、各机交付比特。发生差异时保存首个分歧，再排查版本和底噪。

## 2　首批正式开发作业

`run_n01.py` 接入原 G1 的预测、无线调度与账本，生成一个从零开始的新开发世界。首先执行一个坐标确认资源消耗。

```bash
python -B "$PATCH/tools/run_n01.py" --repo "$REPO" --cpu-root "$CPU" --runtime "$RUN/runtime" --out "$RUN/dev_n01" --world 1400 --multiplier 0.45
```

完整开发清单是 world=1400…1407 与 multiplier=0.45、0.60、0.75、1.00 的笛卡尔积，共 32 个坐标。把剩余坐标交给现有队列。每个作业调用同一命令，替换 world 和 multiplier。每个成功回合写一份含事件、账本、逐机比特、组别、命名空间、有效配置、模型与源文件指纹的 JSON。工具对已有回执和同一作业的并发写入均给出明确提示。失败时保留日志并重排该坐标。

本实现用 `World(cfg,"g1_icc_server_v1/main/development",world_id,...)` 生成物理场景，用 `SeedSequence([20261127,world_id])` 单独排列六类用户。后续组必须共享这些世界的用户顺序、轨迹、噪声和阴影指纹。若服务器已经冻结了不同的排列或版本，保留两个生成协议各自的记录，并在同一个协议内完成所有配对。只有整个开发集聚合后，才能据计划选工作点、K★ 和匹配固定周期。

本机使用轻量包实跑的独立开发坐标 1400 ×0.45 为 847 次定位、2.6257 s 感知、1340.93323 Mbps、7.20347% 总风险。物理世界 SHA-256 为 `e5bab7f31adfd70c9ace880099974b5f891279534a1949d0d34cca18404c1678`。完整回执在 `verification/N01_development_1400_0.45.json`。服务器可用此坐标检查新世界的生成和计费，再提交其余 31 个。单个开发结果只用于这一步。

## 3　Chronos 资产和后续实验

小包附带 G1 的全套原科学输入，以及完整 CPU 包中适用于两臂的公共代码。Chronos 还要加载微调权重和底座快照。若服务器已有这两件原文件，可在另一个可写的资源副本中按下列布局链接并核验。提供的整包可以直接补齐全部资产。

| 归档内路径，相对 gru4505 | 含义 |
|---|---|
| `confirm_0db/training/Chronos-2/7/best.pt` | 微调种子 7 的权重 |
| `chronos_snapshot/model.safetensors` | Chronos-2 底座 |
| `confirm_0db/tubes/Chronos-2/7/bundle.pkl` | 包中已带的原生误差管 |

整包名为 `gru-chronos-4505-cpu.tar.gz`，共 900,277,552 字节，SHA-256 为 `e1c5f1a168c5d7de3e7599eac772e51b2b331b43a3a2e0a3e1afce49c203cf79`。解压到新目录，使用其中的模型和公共仿真器，继续走本包的 **7/9、零预热**入口。

```bash
ARCHIVE=/absolute/path/to/gru-chronos-4505-cpu.tar.gz
mkdir -p "$RUN/full"
echo "e1c5f1a168c5d7de3e7599eac772e51b2b331b43a3a2e0a3e1afce49c203cf79  $ARCHIVE" | sha256sum -c -
tar -xzf "$ARCHIVE" -C "$RUN/full"
FULL="$RUN/full/gru4505"
python -B "$PATCH/tools/experiment_env.py" --repo "$REPO" --cpu-root "$FULL" --runtime "$RUN/runtime" --with-chronos
python -B "$PATCH/tools/replay_anchor.py" --repo "$REPO" --cpu-root "$FULL" --runtime "$RUN/runtime" --out "$RUN/anchor_c" --arm C
```

原归档自己的 `replay.py six/common` 对应旧机群和预热实验。本研究采用上面的入口。其余 19 组矩阵、配额和工作点冻结、同龄面板及统计标准继续依照上一份 `EXPERIMENT_PLAN.md`。本恢复包提供历史双臂回放和 N01 开发入口，AoI-DQN、固定成功时刻等新对照仍按规划分别实现。

## 来源

两个模块在本机旧实验工作目录和历史“四组实验”源码归档中逐字节相同。四个 G1 模型文件按 7/9 版公布的长度及 SHA-256 逐项验证。为方便已有部署，`models/G1/` 提供清单中的标准文件名。同时在 `gru4505/` 中保留原加载器所需的目录结构。Linux 路径适配写在工具中，原模型文件与两个原科学模块均按找回的字节交付。[7/9 模型指纹](https://github.com/mingc5961-alt/five-mpc/blob/96b6e630e02c5fc709aef32e5a655b33ddf2463c/editions/g1-paid-cold-start-20261127/evidence/model_sha256.json) · [原 CPU 复刻包说明](https://github.com/mingc5961-alt/five-mpc/blob/96b6e630e02c5fc709aef32e5a655b33ddf2463c/editions/gru-chronos-irregular-4505/README.md)

若实现反馈门控消融，请显式从此次恢复的同一 G1 检查点派生。历史 `grurun.Predictor` 中直接传 `Static` 或 `NoRoute` 会实例化 `grutrain.Cold`，继而读取另一套 `cold_gru.pt` 和 `residual.json`，会把检查点变化也引入对照。
