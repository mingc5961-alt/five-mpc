"""Render the supplementary report from audited paired summaries."""
import json
from pathlib import Path
import pandas as pd
from server_paths import RUN,ROOT

def load(p):return json.loads(p.read_bytes())
def value(row,unit=''):
    return f"{row['mean']:+.3f}{unit}［{row['lo']:+.3f}, {row['hi']:+.3f}］"
def main():
    out=RUN/'analysis';pr=pd.read_csv(out/'panel_summary.csv');closed=pd.read_csv(out/'closed_contrasts.csv')
    age=pr[(pr.label=='same_direction_revisit')&(pr.exposure=='imposed')].set_index('metric')
    def contrast(a,b,metric,mode):
        match=closed[(closed.left==a)&(closed.right==b)&(closed.metric==metric)&(closed['mode']==mode)]
        return value(match.iloc[0]) if len(match) else '待汇总'
    other=pd.read_csv(out/'revisit_minus_first.csv')
    inter=other[(other.exposure=='imposed')&(other.metric=='d_M_minus_A')].iloc[0]
    arms=pd.read_csv(out/'closed_arms.csv')
    a=load(out/'audit.json');field=load(RUN/'field/summary.json');bos=load(RUN/'field_bosello/summary.json')
    phys=load(RUN/'stress_strict/audit.json');q=load(RUN/'models/independent_tubes.json')
    lines=['# Linux 补充实验结果','',
      '原 G1 的恢复、历史核验和正式 N01 开发由新目录记录。本报告中的 G1 使用本机重训的 GRU、重建门控和独立拟合的误差管。补充留出世界为 4000–4023。原协议的 2000–2023 预留给正式确认。','',
      '## 三个对照回答的问题','',
      '| 控制条件 | 能从中判断的内容 |','|---|---|',
      '| 固定同机付费历史、观测年龄和未来真值 | 路线是否改善三维视线方向，在线反馈相对训练集固定融合权重是否再有增益 |',
      '| 固定时刻、定位用户、定位次数和倍率 | 相同观测流下，整个预测系统（中心和误差管）带来的吞吐与风险变化 |',
      '| 固定总定位次数，按预测截止期选时及选人 | 预测与刷新相互作用后的净效果。正式计划用联合 DQN 的跨预测器复用事件进一步拆分 |','',
      '## 几何与闭环','',
      f"按预定 n=6/8/12/24/40、年龄 2/3/5 秒、额外期限 0 秒，24 个世界有 {a['panel']['primary_revisit_queries']:,} 次同向重访查询。全部面板有 {a['panel']['distinct_prefixes']:,} 个共同前缀、{a['panel']['paired_queries']:,} 个查询。每个世界先将有数据的 n 与年龄格等权，再对世界等权。区间为 20,000 次按世界配对自助法双侧 95% 区间。",'',
      '| 同向重访的角误差差（°） | 平均差［95% 区间］ |','|---|---:|']
    for metric,label in [('d_M_minus_A','纯运动 − 自适应路线融合'),('d_Fstar_minus_A','训练集固定融合 − 自适应路线融合'),
                          ('d_IMM_minus_RAFT','IMM − RAFT'),('d_no_age_minus_with_age','去除年龄 − 包含年龄的最后快照')]:
        lines.append(f'| {label} | {value(age.loc[metric])} |')
    lines+=['',f'在同时覆盖初访和重访的 n、年龄格中，路线增益的“重访减初访”为 {value(inter)}°。这组世界支持路线相对纯运动的贡献，尚难把它限定为重访专属，也难确定在线反馈优于固定融合。','',
       '共同倍率为 0.75，每世界每臂 844 次定位、2.6164 秒。P03、P_M、P_Fstar 在全部 24 个世界的每次定位对象与时刻哈希一致。','',
       '| 闭环差值 | 吞吐增量 Mbps［95% 区间］ | 风险差 pp［95% 区间］ |','|---|---:|---:|']
    for x,y,label in [('P03','P_M','同事件流，自适应融合减纯运动'),('P03','P_Fstar','同事件流，自适应减固定融合'),
                       ('Q01','Q02','等次数、各自按截止期刷新，自适应减纯运动'),('Q01','Q05','等次数、各自按截止期刷新，自适应减固定融合')]:
        lines.append(f'| {label} | {contrast(x,y,"G_Mbps","common075")} | {contrast(x,y,"R_pct","common075")} |')
    lines+=['',f"在各臂独立开发冻结的工作点，Q01−Q05 的吞吐差为 {contrast('Q01','Q05','G_Mbps','frozen')} Mbps。共同倍率与独立工作点回答不同问题，二者均已列入逐世界表。重训版在三个预算 B=3/4.5/9 秒上得到的 K 为 844/981/1296，匹配周期为 3420/2941/2231 毫秒。这些值仅用于本补充协议。",'',
      '[逐世界回执汇总](analysis/closed_worlds.csv) · [所有闭环差值](analysis/closed_contrasts.csv) · [同龄、尾部和准入统计](analysis/panel_summary.csv) · [配对效果图](analysis/paired_effects.png)','',
      '## 原权重 Chronos 与独立基线','',
      '原权重 Chronos 的旧世界 0 ×0.75 回放，811 组用户及完成时刻在 1 微秒内全部对齐，吞吐为 1270.491852 Mbps、风险 8.511806%、定位 2.5141 秒。Linux 生成的物理数组指纹与旧回执不同。重建 G1 从第 19 次事件起偏离，得到 830 次，历史原版为 848 次。压测仅记前向计算。','']
    cc=arms[(arms.row=='N05')&(arms['mode']=='frozen')].set_index('metric')
    if len(cc):lines += [f"Chronos 共 48 个开发端点、48 个补充留出端点。开发冻结概率为倍率 0.75 占 49.411%，倍率 1.00 占 50.589%。24 个补充留出世界的吞吐为 {value(cc.loc['G_Mbps'])} Mbps，风险为 {value(cc.loc['R_pct'])}%。", '']
    lines+=['RAFT 使用 64 个训练世界、16 个验证世界，检索库含 1024 条轨迹的 4096 个窗口，训练屏蔽同轨迹匹配。两个快照头仅共享最后一次付费位置、径向速度、不确定度和查询期限，分别保留或移除观测年龄。16 个世界独立校准整段角误差管，每个世界预选一个锚点。在 24 个留出世界的 10 秒窗口，RAFT 覆盖 20/24，IMM 覆盖 24/24。IMM 的放大因子约 11.51；高覆盖须结合宽度阅读。','',
      '## 实飞外部验证','',
      f"Grigoriou 室外数据含 {field['source_flight_ids']} 个航次标签、{field['source_clusters']} 个采集日。源时间精度为整数秒，保留每秒首条、按缺口和跳变断段。虚拟基站在首点以西 300 米。连续 10 米同向重访主条件仅覆盖 2 个航次同一天的 {field['primary_queries']} 个查询。IMM−RAFT 为 {field['d_IMM_minus_RAFT_deg']:+.3f}°，单日无法估计跨日不确定性。五个等距点近似判据曾得到 3 个查询，已保留在 sampled5 文件中。[作者室外数据](https://zenodo.org/records/13682870) · [逐航次结果](field/per_flight.csv)",'',
      f"Bosello v1.0.0 的 24 个完整航次以公开相机时戳动捕表回放。主条件覆盖 12 个人工飞行航次、{bos['primary_queries']} 个查询，两类场地路线按组等权后 IMM−RAFT 为 {bos['d_IMM_minus_RAFT_deg']:+.3f}°。只有两种有覆盖的模式与路线组合，外推需保留这一限制。全部实飞零样本，未用实飞选择模型。无线吞吐与风险来自仿真。[作者数据发布](https://github.com/tii-racing/drone-racing-dataset/releases/tag/v1.0.0) · [逐航次结果](field_bosello/per_flight.csv)",'',
      '## 压力场景及调度接口','',
      f"12 个种子各有 0/16、9/16、16/16 结构用户和配对偏航四个视图。3 个种子共 9 个原始视图有非注入的加加速度超限。保留原始视图，另在对应小窗做最小改动，最大位置差 {phys['max_position_change_m']*1000:.3f} 毫米。修正后的 48 个视图全程符合 12 m/s、4 m/s²、8 m/s³ 容差 1e−6。36 个受扰航段横向峰值 5 米，持续 12 秒。参照和偏航在处理窗口外逐样本相同。测量噪声与阴影指纹全部核对。",'',
      '在固定外生观测流下，偏航中受扰机的 IMM 角误差相对配对无扰动增加 0.047°［0.012, 0.082］，RAFT 增加 0.006°［−0.026, 0.037］。偏航前两者差值为零。原版策略下的购买、无线风险与恢复时间仍需正式闭环。[压力场景全程核验](stress_strict/audit.json) · [压力几何结果](stress_results/summary.csv)','',
      'AoI-DQN 的共享用户编码、预算合法动作、终端和下一状态自举掩码已测试。用开发世界和随机初始化完成 8 种协议/预算组合的重复回放。正式三种子的学习与两个主对照 Q03−Q04、Q01−Q03 待原版材料与开发配额。','',
      f"本报告已核验 {a['verified_holdout_endpoints']} 个 180 秒留出端点，每个端点检查 16 机、每机 9000 探针、定位付费、180 秒账本与净比特恒等式。[核验计数](analysis/audit.json) · [模型文件指纹](analysis/model_hashes.csv) · [服务器实验计划](../../SERVER_EXPERIMENT_PLAN.md)"]
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
