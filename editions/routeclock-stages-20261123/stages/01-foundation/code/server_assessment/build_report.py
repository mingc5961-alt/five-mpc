import datetime
import json
from pathlib import Path
import shutil

OUT = Path('/data/chen/server_assessment')
def read(name):
    return json.loads((OUT/name).read_text())

hardware = read('hardware.json')
checks = read('checks.json')
profiles = {p.stem: json.loads(p.read_text()) for p in (OUT/'profiles').glob('*.json')}
network = read('pilot_16uav_timing.json')
metrics = read('pilot_16uav/network/lightgbm_7/main/003/completed.json')
smoke = read('mpc_smoke_pinned/verification.json')
batch = read('batch_probe.json')
summary = dict(created_at=datetime.datetime.now().isoformat(), hardware=hardware, checks=checks,
               profiles=profiles, large_query_probe=batch, mpc_smoke=smoke,
               rc_16uav_timing=network, rc_16uav_metrics=metrics,
               conclusion='Suitable for current single-GPU serial training and offline experiments; full formal campaign not executed.')
(OUT/'assessment.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))

lines = [
'RouteClock 服务器训练适配评估',
f'检查时间：{summary["created_at"]}，Asia/Shanghai',
'',
'结论',
'这台服务器具备运行当前代码配置下的单卡串行训练与离线仿真的条件。RTX 3060 12GB 已通过文档要求的全部模型形状探针，以及额外的正式验证集批量探针。文档中的 24GB 是原规划，当前实测负载无需据此认定 12GB 不够。完整正式实验尚未全量执行，因此本报告不保证所有场景的峰值资源或总完成时间。',
'',
'一、检查范围与项目关系',
'依据 routeclock_server/服务器从这里开始.md 与 routeclock_five_20261115/SERVER_RUNBOOK.md，检查服务器入口、四个子项目、离线权重、依赖、数据生成、门控与误差头训练、模型适配器和仿真调度代码。',
'1,189 个交付文件 SHA-256 校验通过；156 个 Python 文件语法检查通过。检查结束后再次通过交付校验，原始源码及交付模型未修改。',
'主实验为 routeclock_five_20261115：IMM、路线 P、门控 RC、纯 Chronos C、带路线 Chronos CR_raw；网络另包含固定基线 F。Chronos 冻结参数，只执行预测，不训练大模型本体。门控使用 SciPy 优化，LightGBM 在 CPU 拟合，TabPFN 使用预训练回归器与缓存上下文。',
'routeclock_server_full 是另一条 MPC 实验：重新训练门控、LightGBM 角误差管、ExtraTrees 风险与条件速率模型，主要在 CPU 运行。其 MPC 当前不包含 Chronos/TabPFN。paired 是已交付分叉的复现入口。其余两个目录为独立历史版本。',
'',
'二、实际资源',
'GPU：NVIDIA GeForce RTX 3060，12,288 MiB；PyTorch 可见约 11.63 GiB。驱动 595.112.07，已验证 PyTorch 2.10.0+cu128 / CUDA 12.8 可用。nvidia-smi 的 CUDA 13.2 不是 PyTorch 自带运行时版本。',
'CPU：宿主机 AMD EPYC 7K62，显示 96 线程；容器 cpu.max=1300000/100000，实际算力配额相当于 13 个逻辑 CPU，不能按独占 96 线程安排工作。',
'内存：容器 memory.max=32 GiB，无交换空间；free 显示的约 251 GiB 属于宿主机。',
f'磁盘：/data 当前剩余 {shutil.disk_usage("/data/chen").free/2**30:.2f} GiB。独立验证环境约 559 MiB，复用已有 CUDA PyTorch。',
'',
'三、环境修复与验证',
'默认 python 指向 /data/miniconda/bin/python（3.13.12），没有 torch。已有 /data/miniconda/envs/torch/bin/python 为 3.12.13，GPU 可用，但缺少 Chronos/TabPFN 等依赖。',
'旧环境 NumPy 1.26.4 / scikit-learn 1.8.0 / LightGBM 4.6.0 与交付模型不一致。实际出现旧 pickle 加载失败及 LightGBM eval_X 参数不兼容。按 requirements-linux.txt 对齐到 NumPy 2.5.4、scikit-learn 1.9.1、LightGBM 4.7.0 等版本后，这些问题已消失，训练 API 小样本检查通过。',
'已创建 /data/chen/server_assessment/venv，使用 Python 3.12 并继承原 CUDA PyTorch，其余所需依赖安装在独立环境。pip check 通过；完整解析版本见 requirements-resolved.txt。',
'测试：五模型 23 passed；完整 MPC 10 passed；旧平台 19 passed；旧 MPC 6 passed、1 skipped（缺少演示缓存）。总计 58 passed、1 skipped。',
'',
'四、GPU 实测（测速输入不用于准确率结论）',
]
for key, p in sorted(profiles.items()):
    lines.append(f'{key}: 初始化/fit {p["init_and_fit_s"]:.3f}s，峰值分配显存 {p["peak_allocated_GiB"]:.3f} GiB，峰值保留显存 {p["peak_reserved_GiB"]:.3f} GiB。')
    for name,t in p['timings'].items():
        lines.append(f'  {name}: 首次 {t["first_s"]:.3f}s，重复调用中位数 {t["median_s"]:.3f}s。')
lines += [
'Chronos 使用 720 历史步，分别测纯模型与路线协变量模型；两个分支共享同一模型实例。TabPFN 使用 7,200 行上下文、4 个 estimators、fit_with_cache，分别 16/63 维，查询 16 行。每项重复 2 次，仅能作为启动验收，不是稳定 P95 的统计估计。',
f'补充 7,200×63 上下文、2,400×63 查询测试：fit {batch["fit_s"]:.3f}s，查询 {batch["predict_s"]:.3f}s，峰值分配 {batch["peak_allocated_GiB"]:.3f} GiB、峰值保留 {batch["peak_reserved_GiB"]:.3f} GiB，输出全部有限。',
'',
'五、真实流程验收',
f'正式参数的一条 train/0 长历史任务已生成，含 3 个锚点及真实离线 Chronos 预测。整个命令含启动耗时 {next(x["wall_seconds"] for x in checks if x["name"]=="generate_one"):.2f}s；文件 274,885 字节。这一条采样协议为 0.5 秒，不能直接代表全部六种行为与观测密度。',
f'完整 MPC smoke 在对齐后的环境从零完成数据、门控、误差管、风险/速率拟合、两策略仿真和审计。verification.passed={smoke["passed"]}；共 {smoke["ticks"]} 快轮，最大时钟误差 {smoke["max_clock_error_s"]:.3g}s。',
f'16 机 RC 闭环使用交付先导 LightGBM 模型，按正式 180 秒预热＋180 秒服务配置完成一场。墙钟耗时 {network["seconds"]:.2f}s，采样到的进程树 RSS 峰值 {network["peak_process_tree_rss_GiB"]:.3f} GiB，实际观测 {metrics["observations_total"]} 次；该 RSS 为 .5 秒采样估计。此为容量/计时验证，不是正式重训后的论文结果。',
'',
'六、正式运行约束与剩余工作',
'主离线任务为 1200 训练＋400 验证＋400 校准＋400 测试，共 2400 条；训练 3 锚点×2 角度轴得到 7200 行。网络代码 jobs() 的 main/actions/reuse 分别为 180/90/120 场，另有多种子和第二误差头复核，不能沿用旧版本配置中其他实验矩阵的总数。',
'建议按当前配置单 GPU 任务串行执行；不要据单任务显存值直接扩大 GPU 并发。MPC workers=4 用于独立数据生成，当前闭环仿真循环本身仍按场景和策略顺序执行。',
'五模型 train 会一次加载四个 split，并对训练数据 deepcopy。已验收任务的内存较低，但正式全量训练的实际峰值仍需监控；本次没有运行 2400 条全量训练。',
'TabPFN 性能注意：routeclock/validity.py 中 Regressor._activate() 会清除前一个头的模型缓存；predict() 在切回时重新 fit。routeclock/five.py 的 evaluate_anchors() 按锚点交替遍历多个头，会反复重建缓存。因此单头热推理耗时会显著低估完整 TabPFN 评估耗时；可考虑按模型批量评估，保持统计口径不变。本次只记录风险，未修改代码。',
'磁盘应按真实输出监控。仅 MPC 正式 30×4×2904×16=5,575,680 行 ledger，按现有演示每行字节粗估约 1.3 GiB；该数字不含场景缓存、其他日志、五模型明细和重复实验。',
'本次未执行完整 five timing 的六类任务加五个中心全闭环，也未完成全部正式训练、390 场基础网络实验或 143 个 paired 分叉。不能把当前样本计时当作全量工期承诺。仿真配置未将真实模型执行延迟计入通信时间，因此可运行实验不等于已满足实际通信实时部署期限。',
'',
'七、使用已验收环境',
'source /data/chen/server_assessment/activate.sh',
'cd /data/chen/routeclock_server',
'python server.py gpu',
'之后按“服务器从这里开始.md”执行。activate.sh 同时将 BLAS/OpenMP 线程设为 1；具体模型的线程参数仍按项目代码配置。',
'',
'证据文件：checks.json、hardware.json、profiles/*.json、batch_probe.json、mpc_smoke_pinned/verification.json、pilot_16uav_timing.json、各检查 *.log；结构化汇总见 assessment.json。',
]
(OUT/'服务器训练适配评估.txt').write_text('\n'.join(lines)+'\n')
print(OUT/'服务器训练适配评估.txt')
