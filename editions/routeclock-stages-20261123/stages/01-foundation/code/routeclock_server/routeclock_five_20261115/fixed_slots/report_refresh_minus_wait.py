from refresh_minus_wait import DEST
from prepare import SLOTS, PERIOD, N, ROUNDS, COST
import numpy as np
import pandas as pd

NAMES={"edf":"到期优先","bits":"原比特排序","split":"刷新减等待"}
def all_files(pattern):
    paths=sorted(DEST.glob(pattern))
    return pd.concat([pd.read_csv(path) for path in paths], ignore_index=True)

def main():
    ledger=all_files("network_*.csv")
    probes=all_files("audit_*.csv")
    training=all_files("extra_*.csv")
    replay=all_files("replay_*.csv")
    sec=PERIOD*SLOTS
    base=ledger.groupby(["mission","policy"]).agg(Mbit=("Mbit","sum"),missing=("unavailable","sum"),
        expired=("expired","sum"),observations=("observed","sum"),over3=("over3","sum"),rows=("uid","size"))
    base["Mbps"]=base.Mbit/sec
    base["n_quanta"]=base.rows*ROUNDS
    base["expired_pct"]=100*base.expired/base.n_quanta
    base.to_csv(DEST/"by_network.csv")
    assert len(base)==18 and (base.observations==64).all() and (base.n_quanta==2048).all()
    stale=probes[probes.age>10].copy()
    expected=ledger[(ledger.age_before>10)&ledger.policy.isin(["bits","split"])]
    keys=["policy","mission","slot","uid"]
    assert set(map(tuple,expected[keys].to_numpy()))==set(map(tuple,stale[keys].to_numpy()))
    assert (stale.wait_est==0).all() and (stale.passive==0).all()
    audit=[]
    for policy,frame in stale.groupby("policy"):
        for model in ["oldscore","newscore"]:
            err=frame[model]-frame.gain
            audit.append(dict(policy=policy,model=model,n=len(frame),routes=len(frame.groupby(["mission","uid"])),
                 predicted=frame[model].mean(),actual=frame.gain.mean(),bias=err.mean(),
                 mae=err.abs().mean(),rmse=np.sqrt((err**2).mean()),under_pct=100*(err<0).mean(),
                 route_equal_bias=frame.assign(err=err).groupby(["mission","uid"]).err.mean().mean()))
    diag=pd.DataFrame(audit)
    diag.to_csv(DEST/"expired_score_audit.csv",index=False)
    info=ledger.assign(old=ledger.age_before>10,refresh_old=(ledger.age_before>10)&(ledger.observed==1))
    clocks=info.groupby(["policy","mission","slot"])[["old","refresh_old"]].any().groupby("policy").sum()
    clocks.to_csv(DEST/"expired_decisions.csv")
    rates=base.Mbps.unstack()
    avg=rates.mean()
    means=base.groupby("policy").sum()
    gap=avg.edf-avg.bits
    lines=["# 16 机固定观测时钟下的比特排序修正实验","",
      f"每个策略重新运行相同的六场测试。各场 16 机、180 秒历史、{sec:.4f} 秒服务、64 次观测。实际观测记录与时钟一致，感知占时 {100*COST/PERIOD:.3f}%。旧排序与到期优先的每机、每轮收益复现到 1e-8 Mbit 以内。轨迹中心、误差管、波束选择、信道噪声和 TDMA 数据机会沿用原平台。","",
      f"从原训练场景重放 {len(replay)} 条记录，分别恢复主动刷新和被动等待的真实标签。另用不同命名空间生成 4 个训练网络、64 条独立新航线，每条构造 5 种等待时长，共 {len(training)} 条记录。其中决策前已超时 {sum(training.x0>10)} 条，两个数据机会中仅有一个仍有效 {sum(training.valid==1)} 条。每条航线的多个时长共享历史。","",
      "使用同参数随机森林分别估计本轮刷新后的总有效 Mbit、每个仍有效的数据机会继续等待时的平均有效 Mbit。后者乘以本轮旧预测仍覆盖的数据机会数（0、1 或 2）。相减得到排序分数。两个头均只使用决策时已知量。训练超参数沿用原比特模型，测试时冻结。","",
      "## 通信结果","","|策略|净吞吐 Mbps|过期时隙 / 12288|过期比例|可传时隙中波束损失超 3 dB|有超时机的轮数 / 384|其中选择已超时飞机的轮数|","|---|---:|---:|---:|---:|---:|---:|"]
    for p,name in NAMES.items():
        g=means.loc[p]; c=clocks.loc[p]
        lines.append(f"|{name}|{avg[p]:.2f}|{int(g.expired)}|{100*g.expired/g.n_quanta:.2f}%|{100*g.over3/(g.n_quanta-g.missing):.2f}%|{int(c.old)}|{int(c.refresh_old)}|")
    lines+=["",f"新版相对原版 {avg.split-avg.bits:+.2f} Mbps，相对到期优先 {avg.split-avg.edf:+.2f} Mbps。六场对到期优先 {sum(rates.split>rates.edf)} 胜 / {sum(rates.split<rates.edf)} 负。原先与到期优先的差距弥合比例为 {100*(avg.split-avg.bits)/gap:.1f}%。","",
      "|场景|到期优先|原比特排序|刷新减等待|新版减到期 Mbps|","|---|---:|---:|---:|---:|"]
    for m,r in rates.iterrows():
        lines.append(f"|{m}|{r.edf:.2f}|{r.bits:.2f}|{r.split:.2f}|{r.split-r.edf:+.2f}|")
    lines+=["","将净吞吐拆成「实际能传输的数据时间占比 × 能传输时的平均有效速率」。用两种策略的中点作对称分解，二者相加等于全网差值。","",
            "|新版相对的基线|有效传输时间变化贡献 Mbps|传输时平均速率变化贡献 Mbps|合计 Mbps|","|---|---:|---:|---:|"]
    available_time=(means.n_quanta-means.missing)*(.01-.0001)
    share=available_time/(6*sec)
    rate=means.Mbit/available_time
    for ref in ["bits","edf"]:
        time_term=(share.split-share[ref])*(rate.split+rate[ref])/2
        rate_term=(rate.split-rate[ref])*(share.split+share[ref])/2
        np.testing.assert_allclose(time_term+rate_term,avg.split-avg[ref])
        lines.append(f"|{NAMES[ref]}|{time_term:+.2f}|{rate_term:+.2f}|{time_term+rate_term:+.2f}|")
    lines+=["","## 相同状态对账","","在原比特与新版策略各自的实际历史上，对每一架已经超时的飞机都独立模拟一次刷新。等待为零，用同一架飞机同一轮刷新得到的有效比特核对两种预测。两种历史分开比较；连续超时的同一架会贡献多行。所有量以 Mbit/架/轮为单位，偏差为预测减实际。","",
      "|策略历史|估值方法|记录 / 航线数|预测均值|实际均值|偏差|平均绝对误差|低估比例|","|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in diag.itertuples():
        lines.append(f"|{NAMES[r.policy]}|{NAMES['bits' if r.model=='oldscore' else 'split']}|{r.n} / {r.routes}|{r.predicted:.2f}|{r.actual:.2f}|{r.bias:+.2f}|{r.mae:.2f}|{r.under_pct:.1f}%|")
    lines+=["","## 原场景 4 第 30 轮复查","","|机号|年龄 s|原打分 Mbit|新打分 Mbit|实际刷新收益 Mbit|","|---|---:|---:|---:|---:|"]
    sample=probes[(probes.policy=="bits")&(probes.mission==4)&(probes.slot==30)]
    for r in sample.itertuples():
        lines.append(f"|{r.uid}|{r.age:.2f}|{r.oldscore:.2f}|{r.newscore:.2f}|{r.gain:.2f}|")
    example=pd.read_csv(DEST/"example_all_candidates.csv")
    best=example.iloc[0]
    legacy=example.loc[example.old_score.idxmax()]
    assert len(example)==16
    lines+=["",f"16 架都参与同一现场的重评分。原模型首选 {int(legacy.uid)} 号，新模型首选 {int(best.uid)} 号，替换这一次观测实际多传 {best.actual_gain-legacy.actual_gain:.2f} Mbit。上表列出原策略已选飞机及当轮全部超时飞机。新版独立闭环会形成不同历史。","",
      "本轮针对已发现问题的同场景回归，结论适用于这六场、该固定感知占时。原两条基线与上轮逐行核查；两处改动联合评估。训练、测试均在本地 CPU 完成。","","依次运行 fixed_slots/refresh_minus_wait.py train、fixed_slots/refresh_minus_wait.py test、fixed_slots/replay_expired_example.py、fixed_slots/report_refresh_minus_wait.py。"]
    (DEST/"本地对照结果.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

if __name__=="__main__":
    main()
