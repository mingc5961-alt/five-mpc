"""将控制器接到物理世界；真值只用于链路结算、离线标签和最终评分。"""
import copy
import csv
import io
import json
import pickle
import numpy as np
from .control import slow_plan,fast_plan,sensing_clock
from .data import GRID,publish,labels
from .forecast_training import train_models as train,load_models,residuals
from .risk import RouteRisk
from .setup_run import atomic,log,load_split,save_json
from routeclock.geometry import angles
from routeclock.radio import Radio,array_gain

def dump_csv(path,rows):
    buf=io.StringIO(newline="")
    writer=csv.DictWriter(buf,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    atomic(path,lambda f:f.write(buf.getvalue().encode("utf-8-sig")))

def settle(world,radio,uid,profile,pair,start,seconds,deadline):
    """按真实发送窗口结算。跨过 10 秒时划分有效段与过期段，过期秒数仍占总时钟。"""
    valid=np.clip(profile.pkg.origin+deadline-start,0.,seconds)
    if valid<=1e-12:
        return 0.,seconds,0.
    mid=start+valid/2
    p=profile.pkg.at("RC",mid-profile.pkg.origin)
    direction=angles(p)
    truth=world.truth(uid,mid)[0]
    received=radio.snr(truth,direction,pair,world.shadow(uid,mid))
    mcs=radio.mcs(radio.snr(p,direction,pair))
    rate=radio.goodput(received,mcs)[0]/1e6
    loss=10*np.log10(np.prod(pair)/array_gain(direction,angles(truth),pair))
    return float(rate*valid),seconds-valid,valid if loss>3. else 0.

def run_policy(world,initial,packages,model,policy,c,out):
    """慢层定未来观测名额，快层读入真实的新测量，然后分配本快轮通信资源。"""
    n=world.n
    trackers=copy.deepcopy(initial)
    profiles=[model.profile(pkg,policy=="age_mpc") for pkg in packages]
    radio,gate=Radio(world.cfg),model.gate
    sim=world.cfg["simulation"]
    cost,switch=sim["observation_cost_s"],sim["beam_switch_s"]
    dt=c["dt"]
    opportunities=sensing_clock(c["steps"],dt,c["sensing_budget"],cost)
    serial=np.cumsum(opportunities)-1
    debt=np.zeros(n)
    bits=np.zeros(n)
    total_sensing=total_data=total_expired=total_over3=0.
    records,plans=[],[]
    for k in range(c["steps"]):
        t=c["warmup_s"]+k*dt
        remaining=c["steps"]-k
        if k%c["slow_every"]==0 or policy=="myopic":
            horizon=min(c["slow_horizon"],remaining)
            if policy=="rotation":
                plan=[int(serial[k+j]%n) if opportunities[k+j] else -1 for j in range(horizon)]
                objective=None
            elif policy=="myopic":
                plan,objective=slow_plan(profiles,t,debt,1,c,cost,switch,opportunities[k:k+1])
            else:
                plan,objective=slow_plan(profiles,t,debt,horizon,c,cost,switch,opportunities[k:k+horizon])
            plan_origin=k
            plans.append(dict(step=k,time=t,objective=objective,actions=" ".join(map(str,plan))))
        offset=k-plan_origin
        picked=plan[offset]
        if picked>=0 and not opportunities[k]:
            raise RuntimeError("本轮没有观测名额")
        if picked>=0:
            trackers[picked].observe(world.observe(picked,t+cost))
            pkg=publish(trackers[picked],gate)
            profiles[picked]=model.profile(pkg,policy=="age_mpc")
            total_sensing+=cost
        # 单步基线只承诺当前动作，尚未安排的未来轮按等待估值；快层 LP 时域保持相同。
        horizon=min(remaining,c["fast_horizon"])
        future=plan[offset:offset+horizon]
        future=future+[-1]*(horizon-len(future))
        airtime,beams=fast_plan(profiles,t,debt,future,c,cost,switch)
        cursor=t+(cost if picked>=0 else 0.)
        actual=np.zeros(n)
        for i in range(n):
            cursor+=switch
            pr=profiles[i]
            pair=radio.pairs[beams[i]]
            secs=float(airtime[i])
            deadline=float(pr.deadlines[beams[i]])
            delivered,expired,over3=settle(world,radio,i,pr,pair,cursor,secs,deadline)
            age=cursor+secs/2-pr.pkg.origin
            p=float(np.interp(min(age,10.),GRID,pr.p)) if age<=10 else None
            valid=secs-expired
            if valid>1e-12:
                effective_age=cursor+valid/2-pr.pkg.origin
                pointing=np.rad2deg(angles(pr.pkg.at("RC",effective_age)))
                planned_rate=float(np.interp(effective_age,GRID,pr.rates[:,beams[i]]))
            else:
                pointing=(None,None)
                planned_rate=0.
            actual[i]=delivered
            records.append(dict(step=k,time=t,uid=i,observed=int(picked==i),age_mid_s=age,
                p_route_failed=p,beam_y=pair[0],beam_z=pair[1],data_s=secs,expired_s=expired,
                over3_s=over3,delivered_Mbit=delivered,debt_before_Mbit=debt[i],
                pointing_az_deg=pointing[0],pointing_el_deg=pointing[1],est_valid_Mbps=planned_rate,
                beam_deadline_s=deadline,age_at_start_s=cursor-pr.pkg.origin,
                route_weight=float(np.interp(age,pr.pkg.hs,pr.pkg.diagnostics["gate_weights"]["RC"])),
                tube_az_deg=float(np.rad2deg(np.interp(age,GRID,pr.envelope[:,0]))),
                tube_el_deg=float(np.rad2deg(np.interp(age,GRID,pr.envelope[:,1])))))
            total_data+=secs
            total_expired+=expired
            total_over3+=over3
            cursor+=secs
        if abs(cursor-(t+dt))>1e-7 or min(airtime)<-1e-9:
            raise RuntimeError("时隙账本失衡")
        bits+=actual
        # 反馈量是基站实际交付量的 ACK 抽象；决策器收不到该轮位置真值。
        debt=np.maximum(0.,debt+c["target_network_mbps"]*dt/n-actual)
        if (k+1)%max(1,c["steps"]//5)==0:
            log(out,policy,f"场景 {world.mission}，快轮 {k+1}/{c['steps']}，累计 {bits.sum()/((k+1)*dt):.2f} Mbps")
    duration=c["steps"]*dt
    guards=c["steps"]*n*switch
    if abs(total_sensing+total_data+guards-duration)>1e-6 or total_sensing>duration*c["sensing_budget"]+1e-9:
        raise RuntimeError("总时间或感知预算超限")
    counts=np.count_nonzero(bits)
    return records,plans,dict(mission=world.mission,policy=policy,n=n,service_s=duration,
        Mbps=bits.sum()/duration,observations=int(round(total_sensing/cost)),sensing_pct=100*total_sensing/duration,
        data_s=total_data,guard_s=guards,expired_s=total_expired,over3_pct=100*total_over3/max(total_data-total_expired,1e-12),
        worst10_Mbps=np.sort(bits/duration)[:int(np.ceil(.1*n))].mean(),served=int(counts),
        jain=float(bits.sum()**2/(n*np.dot(bits,bits))) if np.any(bits) else 0.,terminal_debt_Mbit=debt.sum())

def score_risk(world,trackers,model,out):
    """统一测试发布集，与各策略后续主动选了哪架无关。保留每条预测的概率与事后事件。"""
    radio=Radio(world.cfg)
    rows=[]
    for i,tr in enumerate(trackers):
        for pkg in tr.evidence.archive:
            profile=model.profile(pkg)
            y,_=labels(world,i,pkg,radio)
            p=profile.p
            truth=np.stack([world.truth(i,pkg.origin+h)[0] for h in GRID])
            error=residuals(pkg,truth)
            covered=np.logical_and.accumulate((error<=profile.envelope+1e-12).all(axis=1))
            for j,h in enumerate(GRID):
                rows.append(dict(mission=world.mission,uid=i,origin=pkg.origin,h=h,
                                 event=int(y[j]),prob=p[j],age_base=model.base[j],tube_covered=int(covered[j]),
                                 tube_az_deg=np.rad2deg(profile.envelope[j,0]),tube_el_deg=np.rad2deg(profile.envelope[j,1])))
    dump_csv(out/f"risk_test_{world.mission}.csv",rows)

def simulate(c,out,rerun=False,model_dir=None):
    source=model_dir or out
    model=load_models(source)
    identity=json.loads((source/"models.json").read_text(encoding="utf8"))
    marker=out/"evaluation_models.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf8"))!=identity:
        raise ValueError("此测试目录已绑定另一组模型。请为新模型使用新的测试输出目录。")
    save_json(marker,identity)
    for m in c["test_ids"]:
        with (out/f"test_{m}.pkl").open("rb") as f:
            world,trackers,packages=pickle.load(f)
        if not (out/f"risk_test_{m}.csv").exists():
            score_risk(world,trackers,model,out)
        for policy in c["policies"]:
            folder=out/f"scene_{m}"/policy
            folder.mkdir(parents=True,exist_ok=True)
            pending=folder/"in_progress.json"
            if (folder/"metrics.json").exists() and not pending.exists() and not rerun:
                log(out,policy,f"场景 {m} 已完成，复用账本")
                continue
            log(out,policy,f"开始场景 {m}")
            save_json(pending,{"mission":m,"policy":policy})
            rows,plans,summary=run_policy(world,trackers,packages,model,policy,c,out)
            dump_csv(folder/"ledger.csv",rows)
            dump_csv(folder/"plans.csv",plans)
            save_json(folder/"metrics.json",summary)
            pending.unlink()

def report(c,out):
    summaries=[]
    risk=[]
    for m in c["test_ids"]:
        for p in c["policies"]:
            path=out/f"scene_{m}"/p/"metrics.json"
            if (path.parent/"in_progress.json").exists():
                raise RuntimeError(f"场景 {m} 策略 {p} 尚未完成，先运行 simulate 续跑。")
            summaries.append(json.loads(path.read_text(encoding="utf8")))
        with (out/f"risk_test_{m}.csv").open(encoding="utf-8-sig") as f:
            risk.extend(csv.DictReader(f))
    dump_csv(out/"network_results.csv",summaries)
    y,p,b,h=(np.array([float(row[key]) for row in risk]) for key in ("event","prob","age_base","h"))
    quality=[]
    for length in (1,2,5,10):
        mask=h==length
        quality.append(dict(h_s=length,samples=int(mask.sum()),observed=y[mask].mean(),
            predicted=p[mask].mean(),brier=np.mean((p[mask]-y[mask])**2),age_brier=np.mean((b[mask]-y[mask])**2)))
    dump_csv(out/"risk_quality.csv",quality)
    tube_rows=[]
    for length in (1,2,5,10):
        subset=[r for r in risk if float(r["h"])==length]
        tube_rows.append(dict(h_s=length,publications=len(subset),**{
            k:float(np.mean([float(r[k]) for r in subset])) for k in ("tube_covered","tube_az_deg","tube_el_deg")}))
    dump_csv(out/"tube_quality.csv",tube_rows)
    lines=["# 双时间尺度 MPC 运行结果","",f"独立测试网络 {len(c['test_ids'])} 个，每个网络 {c['uavs']} 架，每次服务 {c['steps']*c['dt']:.3f} 秒。","",
        "|策略|净吞吐 Mbps|感知次数/场|感知占时 %|最弱 10% 单机 Mbps|Jain 公平指数|可传时间中超 3 dB %|",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for policy in c["policies"]:
        subset=[s for s in summaries if s["policy"]==policy]
        vals=[np.mean([s[key] for s in subset]) for key in ("Mbps","observations","sensing_pct","worst10_Mbps","jain","over3_pct")]
        lines.append("|"+policy+"|"+"|".join(f"{x:.3f}" for x in vals)+"|")
    lines+=["","|预测时长 s|实际累计失效率|预测均值|Brier|仅年龄 Brier|","|---:|---:|---:|---:|---:|"]
    for row in quality:
        lines.append("|"+"|".join(f"{row[k]:.3f}" for k in ("h_s","observed","predicted","brier","age_brier"))+"|")
    lines+=["","Brier 为概率与实际 0/1 事件的平均平方误差，越低越好。风险样本为各策略共享的预热末段发布，多时域和重复发布来自相同航线。每场、每机及每轮的真实仿真账本见 network_results.csv、scene_*/策略/ledger.csv。"]
    lines+=["","|预测时长 s|两轴同时覆盖整段的比例|平均方位半宽 °|平均俯仰半宽 °|","|---:|---:|---:|---:|"]
    for row in tube_rows:
        lines.append("|"+"|".join(f"{row[k]:.3f}" for k in ("h_s","tube_covered","tube_az_deg","tube_el_deg"))+"|")
    atomic(out/"结果报告.md",lambda f:f.write(("\n".join(lines)+"\n").encode("utf8")))
