"""独立读取已写账本复核实际资源，用于部署后的验收。"""
from collections import defaultdict
import csv
import json
import numpy as np
from .data import load_config
from .setup_run import save_json,log

def verify(c,out):
    sim=load_config()["simulation"]
    cost,switch=sim["observation_cost_s"],sim["beam_switch_s"]
    largest_clock_error=0.
    audited=0
    for mission in c["test_ids"]:
        for policy in c["policies"]:
            folder=out/f"scene_{mission}"/policy
            if (folder/"in_progress.json").exists():
                raise RuntimeError(f"场景 {mission} 策略 {policy} 尚未完成，先续跑。")
            summary=json.loads((folder/"metrics.json").read_text(encoding="utf8"))
            with (folder/"ledger.csv").open(encoding="utf-8-sig") as f:
                groups=defaultdict(list)
                for row in csv.DictReader(f):
                    groups[int(row["step"])].append(row)
            assert len(groups)==c["steps"]
            sensed=total_bits=0.
            for k in range(c["steps"]):
                step=groups[k]
                assert len(step)==c["uavs"] and len({r["uid"] for r in step})==c["uavs"]
                count=sum(int(r["observed"]) for r in step)
                assert count<=1
                sensed+=count*cost
                assert sensed<=(k+1)*c["dt"]*c["sensing_budget"]+1e-9
                data=sum(float(r["data_s"]) for r in step)
                err=abs(data+count*cost+c["uavs"]*switch-c["dt"])
                largest_clock_error=max(err,largest_clock_error)
                assert err<1e-7
                for r in step:
                    a=float(r["data_s"])
                    assert -1e-9<=float(r["expired_s"])<=a+1e-9
                    deadline=float(r["beam_deadline_s"])
                    age=float(r["age_at_start_s"])
                    assert -1.<=deadline<=10.
                    # 无证的时间不能记入有效发送；逐条用实际起点和束宽期限反推有效窗口。
                    assert abs((a-float(r["expired_s"]))-np.clip(deadline-age,0.,a))<1e-7
                    assert 0<=float(r["route_weight"])<=1
                    assert min(float(r["tube_az_deg"]),float(r["tube_el_deg"]))>=0
                    assert a+1e-9>=c["min_airtime_fraction"]*data/c["uavs"]
                    total_bits+=float(r["delivered_Mbit"])
            assert abs(total_bits/summary["service_s"]-summary["Mbps"])<1e-6
            audited+=1
        with (out/f"risk_test_{mission}.csv").open(encoding="utf-8-sig") as f:
            paths=defaultdict(list)
            for r in csv.DictReader(f):
                paths[(r["uid"],r["origin"])].append((float(r["h"]),float(r["prob"]),int(r["event"])))
        for rows in paths.values():
            grid,p,y=np.asarray(sorted(rows)).T
            assert grid[0]==0 and grid[-1]==10 and len(grid)==41
            assert min(p)>=0 and max(p)<=1 and np.all(np.diff(p)>=-1e-12)
            assert np.all(np.diff(y)>=0)
    save_json(out/"verification.json",dict(scenes_and_policies=audited,
        ticks=audited*c["steps"],max_clock_error_s=largest_clock_error,passed=True))
    log(out,"独立复核",f"{audited} 条闭环逐轮通过，总时钟最大误差 {largest_clock_error:.2e} 秒")
