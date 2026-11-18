import json,pickle,re,sys
from pathlib import Path

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
def read(p):return json.loads((root/p).read_text(encoding='utf-8'))
probes=read('runs/model_probe.json')
primary=read('runs/final_tabpfn_verification/verification.json')
assert {x['method'] for x in primary}=={'I','C','CR','RC'}
assert all(x['head']=='tabpfn' and x['event_bits_verified'] for x in primary)
controls=read('runs/final_lgb_verification/verification.json')
fixed=read('runs/final_fixed_verification/verification.json')
assert len(controls)==11 and len(fixed)==1
testlog=(root/'runs/final_pytest.log').read_text(encoding='utf-8')
tests=int(re.search(r'(\d+) passed',testlog)[1])
probe2=read('runs/tabpfn_local_probe.json')
with (root/'runs/final_lgb_models/bundle.pkl').open('rb') as f:lgb=pickle.load(f)
correction=lgb['controls']['RC']['snr'].correction
names={'I':'自适应IMM运动预测（I）','C':'无路线输入的Chronos预测（C）','CR':'路线辅助Chronos与IMM学习融合（CR）','RC':'空间路径与IMM学习融合（RC）'}
rows='\n'.join(f"| {names[r['method']]} | {r['N']} | {r['net_mbps']:.3f} | {r['observations_total']} | 通过 |" for r in primary)
method_guide=(root/'implement-stage/METHOD_GUIDE.md').read_text(encoding='utf-8')
report=f'''# 完整代码平台实施与本地验收报告

日期：2026-11-13。平台目录：`D:/低空论文/routeclock_platform_20261113`。

## 完成范围

已实现研究报告的主闭环：真实无线量测 → 位置IMM与空间路径记忆 → 进度模式分布 → 专家预测与学习门控 → 未来整段角误差区域 → 各束宽可用期限 → 共享预算调度 → 实际通信比特 → 新观测核验旧预测。所有内容位于新目录，旧项目代码和结果保留。

本轮按用户要求以代码逻辑和闭环运行验收为目标，没有根据效果优劣重新设计研究方法。

{method_guide.replace('# 方法名称与完整数据流', '## 方法名称与完整数据流', 1).replace('## 学习门控', '### 学习门控').replace('## 位置预测', '### 位置预测').replace('## 怎样阅读', '### 怎样阅读')}

## 模块与实现

| 模块 | 文件 | 实现内容与核验 |
|---|---|---|
| 物理世界与量测 | world.py、geometry.py | 六种行为、三维动力学、速度/加速度/jerk限制、球坐标无线量测、绝对时间噪声场；随机世界与方法种子分离 |
| 状态与进度 | filters.py | CV/左右转弯位置IMM；巡航/减速/停留进度IMM；连续时间模式转移与OU离散化 |
| 空间记忆 | memory.py | 有向片段、位置/航向/连续性匹配、多分支、闭合、末端接续、进度不确定性传播；中途偏离不伪装成原路段末端 |
| 预测与融合 | predictors.py | I/P/H/C/CR_raw/RC/CR七中心，16维随时长变化的学习门控，任务分组OOF |
| 真实基础模型 | foundation.py | amazon/chronos-2真实权重；TabPFN-2.5真实回归器、4 estimators；过去与未来协变量实际调用通过 |
| 兑现证据 | evidence.py | 保存真实发布预测、先对账后更新；P/I/CR/H短长期残差；无新观测不重置原点 |
| 有效性与对照头 | validity.py、training.py | 63维共享轴/时长输入、TabPFN与LightGBM、经验包络、直接期限头、SNR下界头；独立任务校准 |
| 无线通信 | radio.py | 9种子阵、二维联合方向图、MCS/BLER、有效比特与最差指向损失 |
| 多机调度 | scheduler.py | 束宽和期限候选、EDF、按通信代价调整、统一加宽、单变量动作；精确感知时间预算 |
| 闭环与节点记录 | simulator.py | 感知/切换/数据事件、预测包、实际动作、SNR、比特、失联、超期；账本核对通过 |
| 实验执行 | dataset.py、experiments.py、ablations.py、evaluation.py | 分组数据划分、A1/A2/A3、830项网络矩阵、80项单机诊断、消融、配对统计 |
| 真实数据接口 | tools/real_trajectory.py | 时间戳xyz/GPS CSV的因果轨迹匹配；尚未执行AERPAW实测数据验收 |

## 实际验证结果

- 自动测试：**{tests}项通过**。覆盖物理限制、进度滤波、分支传播、模式点估计消融、角度区域增益界、观测预算、动作筛选顺序、档案因果性、校准时序、矩阵规模等。
- Chronos真实权重加载与协变量推理：**{probes['chronos']['status']}**。
- TabPFN-2.5真实回归器拟合与预测：**{probes['tabpfn']['status']}**；序列化后重新加载的最大预测差异为 **{probe2['reload_max_abs_difference']}**。
- 最终验收数据：24训练任务、12验证任务、12校准任务、12测试任务。训练每任务3个锚点，校准每任务1个锚点；每个共享误差头144个训练查询行。这里缩小的是任务规模，模型结构、4个estimators和真实权重不变。
- 完整LightGBM分支验收：**11个分支通过**，覆盖I/C/CR/RC/Z、无档案、直接期限、EDF、统一加宽、仅调间隔、仅调束宽。另完成F固定基线的验证集选参及闭环。
- 真实TabPFN主闭环：**6架无人机、六种行为、12秒建图、3秒服务，四个主方法均完成**。每个运行都用事件有效比特重算净吞吐，并检查感知＋切换＋数据等于服务时间。

| 方法 | 无人机数 | 净吞吐 Mbps | 全任务观测次数 | 事件与汇总一致性 |
|---|---:|---:|---:|---|
{rows}

这些数值是闭环验收输出，主用途是证明真实预测结果进入了波束、调度和通信账本。统计性能比较仍需正式任务矩阵。

## 已修复的实现问题

1. 同一路段上的多个交叉候选原先没有分别移动预测中心，现已各自传播。
2. 动作先做全局筛选会错误删除“只调宽度/间隔”的候选，现已先限制后筛选。
3. RC/I训练档案与部署时可得信息不一致，现已移除未执行专家的档案输入。
4. 无档案消融原先只删除误差头证据，现已连门控一起重训和校准。
5. 校准锚点与测试锚点此前的观测时序不一致，现采用共同因果观测协议。
6. 谐波兑现误差、末端连接的桥段距离、事件末尾时间和实际动作图均已补齐。

详细设计对应与两轮审查记录位于 `implement-stage`。第二轮审查后修复的项由执行人通过回归检查确认，原始审查意见保留。

## 效果不足的现象与后续方向

LightGBM的三机验收中，RC未获得持续吞吐优势。该配置只建图12秒，45条RC观测记录中尚无路径闭合记录，长期几何记忆还不成熟。正式180秒建图条件下应重新判断其价值。

同一验收中的Z下界触发产生零吞吐；记录到SNR校准扣减为 **{correction:.2f} dB**，使候选动作难以通过门限。后续可先增加训练与校准任务，检查跨束宽的残差分布及区间覆盖，而非直接调低门限。本轮保留失败现象，没有据此改动算法来改善排名。

## 留待后续资源的工作

代码已提供以下执行入口，本轮未执行其完整计算：2400个正式单机任务；720步Chronos和7200行TabPFN上下文的24GB设备测量；种子7/11/19；830次网络测试与80次单机动作诊断；32机规模与预算/复用比例扫描。

24GB GPU主要用于这些正式规模计算与显存测量。当前真实模型和闭环已经在CPU完成验收。AERPAW验证还需要实际数据文件，这一项不是显卡问题。

## 从哪里开始使用

说明书：`README.md`；设计依据：`implement-stage/RESEARCH_REPORT.md`；运行配置：`configs/research.json`。

最终真实模型：`runs/final_tabpfn_models/bundle.pkl`；完整分支轻量模型：`runs/final_lgb_models/bundle.pkl`。

最终主闭环：`runs/final_tabpfn_verification/verification.json`；完整分支：`runs/final_lgb_verification/verification.json`；真实模型探针：`runs/model_probe.json`；测试日志：`runs/final_pytest.log`。

复现命令（在本目录运行）：

```powershell
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe tools/verify_closed_loop.py --bundle runs/final_tabpfn_models/bundle.pkl --out runs/recheck --methods I,C,CR,RC --uavs 6
```
'''
(root/'IMPLEMENTATION_REPORT.md').write_text(report,encoding='utf-8')
status={'unit_tests_passed':tests,'real_model_probe_status':{k:v['status'] for k,v in probes.items()},'tabpfn_closed_loop_methods':[r['method'] for r in primary],'tabpfn_uavs':6,'lightgbm_branches':len(controls),'fixed_baseline_runs':len(fixed),'formal_network_jobs_defined':830,'formal_single_jobs_defined':80,'formal_campaign_executed':False}
(root/'runs/FINAL_VERIFICATION.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
print('IMPLEMENTATION_REPORT.md and FINAL_VERIFICATION.json generated')
