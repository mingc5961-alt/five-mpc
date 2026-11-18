from prepare import OUT, COST, N, PERIOD, SLOTS, ROUNDS, QUANTUM
import numpy as np
import pandas as pd

def main():
    frames = pd.concat([pd.read_csv(OUT/f"test_{i}_ledger.csv") for i in range(6)],ignore_index=True)
    forks = pd.concat([pd.read_csv(OUT/f"test_{i}_forks.csv") for i in range(6)],ignore_index=True)
    train = pd.concat([pd.read_csv(OUT/f"train_{i}_train.csv") for i in range(4)])
    times = frames.groupby(["mission","policy","slot"])
    assert (times["observed"].sum()==1).all() and (times.uid.nunique()==N).all()
    assert (frames.groupby(["mission","slot"]).t.nunique()==1).all()
    assert (frames.groupby(["mission","policy"])["observed"].sum()==SLOTS).all()
    for policy in ["bits","risk"]:
        picked = frames[(frames.policy==policy)&(frames.observed==1)]
        paired = forks[forks.history==policy].merge(picked,on=["mission","slot"])
        assert np.array_equal(paired.uid,paired[policy+"_uid"])
    same = forks.risk_uid==forks.bits_uid
    assert np.allclose(forks[same].extra_Mbit,0)
    assert np.allclose(forks.extra_Mbit,forks.bits_pick_refresh_Mbit-forks.risk_pick_refresh_Mbit)
    merged = forks.pivot(index=["mission","slot"],columns="history",values="extra_Mbit")
    assert np.allclose(merged.xs(0,level="slot").risk,merged.xs(0,level="slot").bits)
    route = frames.groupby(["mission","uid","group","policy"]).agg(Mbit=("Mbit","sum"),
        probes=("observed","sum"),unavailable=("unavailable","sum")).reset_index()
    route.to_csv(OUT/"route_ledger.csv",index=False)
    groups = frames.groupby(["mission","policy"])[["Mbit","over3","unavailable","quanta","observed"]].sum()
    groups["Mbps"] = groups.Mbit/(SLOTS*PERIOD)
    groups["over3_active_pct"] = 100*groups.over3/(groups.quanta-groups.unavailable)
    groups["unavailable_pct"] = 100*groups.unavailable/groups.quanta
    expired={}
    for (m,s),sub in frames.groupby(["mission","policy"]):
        order=np.argsort(np.random.default_rng(8100+m).permutation(N))
        last=178 + np.arange(N)*.125 + COST
        counter=0
        for k,step in sub.groupby("slot"):
            t=180+k*PERIOD
            last[int(step.loc[step.observed==1,"uid"].iloc[0])]=t+COST
            mid=t+COST+(order[:,None]+N*np.arange(ROUNDS))*QUANTUM+(QUANTUM+.0001)/2
            mask=(mid>last[:,None]+10+1e-9).sum(axis=1)
            assert (mask <= step.sort_values("uid").unavailable.to_numpy()).all()
            counter+=mask.sum()
        expired[m,s]=counter
    groups["expired_quanta"] = [expired[idx] for idx in groups.index]
    groups.to_csv(OUT/"by_network.csv")
    net = groups.Mbps.unstack()
    net["bits_minus_risk"]=net.bits-net.risk
    net["bits_minus_edf"]=net.bits-net.edf
    network = []
    for history, frame in forks.groupby("history"):
        for mission, sub in frame.groupby("mission"):
            d = sub[sub.bits_uid!=sub.risk_uid]
            network.append(dict(history=history,mission=mission,decisions=len(sub),differ=len(d),
                sum_Mbit=d.extra_Mbit.sum(),win=(d.extra_Mbit>1e-6).sum(),loss=(d.extra_Mbit< -1e-6).sum()))
    pd.DataFrame(network).to_csv(OUT/"forks_by_network.csv",index=False)
    attribution = []
    for row in forks[~same].itertuples():
        for uid,sgn,value in [(row.bits_uid,1,row.bits_pick_refresh_Mbit),(row.risk_uid,-1,row.risk_pick_refresh_Mbit)]:
            attribution.append((row.history,row.mission,uid,sgn*value))
    a = pd.DataFrame(attribution,columns=["history","mission","uid","Mbit"])
    a=a.groupby(["history","mission","uid"]).Mbit.sum().reset_index()
    a.to_csv(OUT/"fork_route_attribution.csv",index=False)
    route_wide=route.pivot(index=["mission","uid","group"],columns="policy",values="Mbit")
    rd=(route_wide.bits-route_wide.risk).sort_values(ascending=False)
    rd.rename("bits_minus_risk_Mbit").to_csv(OUT/"closed_route_difference.csv")
    names={"edf":"到期优先","risk":"风险优先","bits":"预计挽回比特优先"}
    def percent(x):return f"{100*x:.1f}%"
    def top(x,k):
        positives=x[x>0].sort_values(ascending=False)
        return percent(positives.head(k).sum()/positives.sum()) if len(positives) else "无正贡献"
    lines=["# 同一观测时钟下的 16 机排序实验", "",
      f"4 个独立训练网络，{len(train)} 条配对训练记录。6 个独立测试网络，共 96 条测试航线。各场先观察 180 秒，每机每 2 秒一次并错开。随后服务 {SLOTS*PERIOD:.4f} 秒，每隔 {PERIOD:.4f} 秒观测一架，每场恰好 {SLOTS} 次。每次占 {COST*1000:.1f} 毫秒，实际感知占比 {percent(COST/PERIOD)}。其余时段每架固定分得两次 10 毫秒数据机会，每次扣 0.1 毫秒切束。","",
      "轨迹中心和误差管沿用冻结的路线记忆＋短期状态门控及 LightGBM。发布预测时使用平台波束候选集，从非支配候选中选预计速率最高者，其有效期用于到期排序。三者使用同一束宽规则。风险和比特各用一台同配置随机森林，输入相同，目标分别为继续使用旧预测时的失配或不可服务比例、观测相对等待一轮可增加的有效 Mbit。输入包括预测年龄、到期差、束宽、误差管、距离、姿态几何与路线诊断。决策时不读取实际位置、未来阴影、航线类型或新观测。","",
      "## 三种策略独立运行","",
      "|策略|平均净速率 Mbps|已传输时隙中超 3 dB|无可用预测的数据时隙|从未获得服务期观测的航线数 / 96|",
      "|---|---:|---:|---:|---:|"]
    for s in names:
        g=groups.xs(s,level="policy")
        lines.append(f"|{names[s]}|{g.Mbps.mean():.2f}|{percent(g.over3.sum()/(g.quanta.sum()-g.unavailable.sum()))}|{percent(g.unavailable.sum()/g.quanta.sum())}|{(route[route.policy==s].probes==0).sum()}|")
    lines+=["","|策略|年龄超 10 秒的时隙 / 全部数据时隙|其余无法传输时隙|有预测时每秒数据传输的有效比特 Mbps|","|---|---:|---:|---:|"]
    for s in names:
        g=groups.xs(s,level="policy")
        lines.append(f"|{names[s]}|{int(g.expired_quanta.sum())} / {int(g.quanta.sum())}|{int((g.unavailable-g.expired_quanta).sum())}|{g.Mbit.sum()/((g.quanta.sum()-g.unavailable.sum())*(QUANTUM-.0001)):.2f}|")
    lines+=["","|测试场景|到期 Mbps|风险 Mbps|比特 Mbps|比特减风险 Mbps|","|---|---:|---:|---:|---:|"]
    for m,r in net.iterrows():lines.append(f"|{m}|{r.edf:.2f}|{r.risk:.2f}|{r.bits:.2f}|{r.bits_minus_risk:+.2f}|")
    d=net.bits_minus_risk.to_numpy()
    ci=np.quantile(np.random.default_rng(230).choice(d,(10000,len(d)),replace=True).mean(axis=1),[.025,.975])
    lines.extend(["",f"比特减风险平均 {d.mean():+.2f} Mbps，配对场景 bootstrap 描述性 95% 区间 [{ci[0]:+.2f}, {ci[1]:+.2f}]，逐一去掉一个场景后的均值范围 {(d.sum()-d.max())/5:+.2f} 至 {(d.sum()-d.min())/5:+.2f} Mbps。六场中有 {(d>0).sum()} 场为正。","",
       "## 在同一决策状态是否选了不同的飞机","",
       "分别沿风险策略和比特策略已经形成的历史，在每轮观测前保存现场，同时试算两种选择。只比较到下一次观测前的真实仿真有效比特，两架以外的飞机在该轮完全相同。两组分别报告，避免重复计算共同起点。正负以 1 bit 为分辨率。","",
       "|共同历史|分歧数 / 决策数|分歧时比特排序赢 / 输 / 平|多传合计 Mbit|每次分歧平均 Mbit|中位 Mbit|赢时累计 / 输时累计 Mbit|",
       "|---|---:|---:|---:|---:|---:|---:|"])
    for s in ["risk","bits"]:
        sub=forks[(forks.history==s)&~same]
        ds=sub.extra_Mbit
        wins=ds>1e-6; losses=ds< -1e-6
        lines.append(f"|{names[s]}|{len(ds)} / {6*SLOTS}|{wins.sum()} / {losses.sum()} / {(~(wins|losses)).sum()}|{ds.sum():+.2f}|{ds.mean():+.3f}|{ds.median():+.3f}|{ds[wins].sum():+.2f} / {ds[losses].sum():+.2f}|")
    actual=frames[frames.observed==1].pivot(index=["mission","slot"],columns="policy",values="uid")
    actual_diff=actual.bits!=actual.risk
    lines+=["",f"直接对齐两套独立闭环的真实时钟，比特与风险排序在 {actual_diff.sum()} / {len(actual)} 轮选择不同飞机（{percent(actual_diff.mean())}）。后续轮的各机观测历史会分开，这个次数与上面的同状态分叉次数分别计数。","",
       "|同一风险优先历史上的分歧例子|观测轮开始时刻 s|风险选择（场景 / 机号）|价值选择（场景 / 机号）|分别刷新相对等待的实际收益 Mbit|选择价值优先的净差 Mbit|",
       "|---|---:|---|---|---:|---:|"]
    sub=forks[(forks.history=="risk")&~same]
    for label, s in [("获益样本",sub[sub.extra_Mbit>1e-6]),("损失样本",sub[sub.extra_Mbit< -1e-6])]:
        if len(s):
            r=s.iloc[np.argmin(abs(s.extra_Mbit.to_numpy()-s.extra_Mbit.median()))]
            lines.append(f"|{label}（该组中位数附近）|{r.t:.4f}|{r.mission} / {r.risk_uid}|{r.mission} / {r.bits_uid}|{r.risk_pick_refresh_Mbit:+.3f} / {r.bits_pick_refresh_Mbit:+.3f}|{r.extra_Mbit:+.3f}|")
    lines+=["","## 航线贡献是否集中","",
       "逐航线把比特策略与风险策略的整场实际 Mbit 相减。以下占比的分母是所有正贡献航线增加的比特之和，负贡献另列，以免正负相抵夸大集中度。","",
       f"96 条航线中，{(rd>1e-6).sum()} 条增加，{(rd< -1e-6).sum()} 条减少，其余持平。正贡献 {rd[rd>0].sum():.2f} Mbit，负贡献 {rd[rd<0].sum():.2f} Mbit。前 1 / 3 / 10 条正贡献航线占 {top(rd,1)} / {top(rd,3)} / {top(rd,10)}。剔除贡献最大的三条后，余下航线的合计差为 {rd.iloc[3:].sum():+.2f} Mbit（事后归因）。","",
       "|航线类型|航线数|闭环净差 Mbit|","|---|---:|---:|"]
    for g,v in rd.groupby(level="group"):lines.append(f"|{g}|{len(v)}|{v.sum():+.2f}|")
    lines+=["","对于同状态分歧，把获选飞机的实际刷新收益记为正项，放弃飞机的实际刷新收益记为负项。"]
    for s in ["risk","bits"]:
        v=a[a.history==s].Mbit
        lines.append(f"- {names[s]}历史上，前 1 / 3 / 10 条航线分别占正贡献的 {top(v,1)} / {top(v,3)} / {top(v,10)}。")
    lines+=["","## 核查与复现","","按场、策略、观测轮核对了观测时刻、次数、每机数据份额和观测身份。两策略选同一飞机时，反事实差为零。吞吐量为平台链路模型按 BLER 折算的期望有效比特除以整段服务时间。超过 3 dB 指实际阵列增益较同一波束完美对准降低超过 3 dB。无可用预测的时隙传输量为零，单独列示。","","依次运行 fixed_slots/prepare.py、fixed_slots/compare.py train、fixed_slots/compare.py test、fixed_slots/report.py。明细在同目录中的 *_ledger.csv、*_forks.csv 和逐航线归因表。"]
    lines+=["","## 研究判断","",
        f"比特排序相对风险排序平均增加 {net.bits_minus_risk.mean():.2f} Mbps，相对到期排序平均差 {net.bits_minus_edf.mean():+.2f} Mbps。本试验将本次收益定义为下一次公共观测机会之前的有效比特。平台沿已发布的位置预测更新指向，超过所选波束的建议期限仍可尝试发送，但预测年龄超过 10 秒即无可用预测。在有预测的时隙中，比特排序的有效速率最高，全时段吞吐则由到期排序领先。","",
        "建议保留预计可挽回比特作为排序信号。先在独立新场景验证一个简单组合：当继续等待将导致未来观测名额不足以在 10 秒前刷新一批飞机时，优先解决最早到限的飞机；有余量时按预计比特排序。与到期优先同时比较总有效比特、各机失去服务的比例，以及 5% 预算下的表现。如果组合已经获得主要收益，多步指数策略需要在相同条件下进一步超越这个基线，才值得成为论文的核心。"]
    (OUT/"对照结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("\n".join(lines).encode("ascii","backslashreplace").decode()[:64])

if __name__=="__main__":
    main()
