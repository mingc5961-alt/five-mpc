"""Summarize paired physical validity diagnostics from exported receipts and prefixes."""
from pathlib import Path
import csv
import json
import numpy as np

root=Path(__file__).resolve().parent/"factorial_run"
def read(file):
    with (root/file).open(encoding="utf-8-sig") as f:return list(csv.DictReader(f))
def save(file,rows):
    with (root/file).open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def main():
    from collections import defaultdict
    groups=defaultdict(list)
    for r in read("hold_prefixes.csv"):
        groups[int(r["world"]),r["arm"],float(r["scale"])].append([float(r["admitted_s"]),float(r["realized_s"])])
    assert len(groups)==80 and sum(map(len,groups.values()))==64440
    rows=[]
    for (k,a,m),v in groups.items():
        p,t=np.asarray(v).T
        vuse=np.minimum(p,t);miss=t-vuse;over=p-vuse
        if a in ("O0","O1","R1"):
            base="R0" if a=="R1" else "G1"
            assert np.array_equal(t,np.array(groups[k,base,m])[:,1])
        rows.append(dict(world=k,arm=a,scale=m,n=len(v),admitted_s=p.mean(),realized_s=t.mean(),usable_s=vuse.mean(),missed_s=miss.mean(),excess_s=over.mean(),admit_pct=100*np.mean(p>0)))
    save("hold_worlds.csv",rows)
    ep={(int(r["world"]),r["arm"],float(r["scale"])):r for r in read("confirm_endpoints.csv")}
    for f in root.glob("sensitivity_grid*.json"):
        d=json.loads(f.read_text());ep[d["k"],"G1",d["m"]]={"Mbps":d["result"]["G_mbps"],"risk_pct":100*d["result"]["R"],"narrow_good_s":d["result"]["narrow_good_data_s"]}
    def get(k,a,m):
        return next(r for r in rows if (r["world"],r["arm"],r["scale"])==(k,a,m))
    # Test-mixture sensitivity. Re-estimate each mix on every resample.
    rng=np.random.default_rng(270228);boot=rng.integers(8,size=(20000,8))
    fields=["admitted_s","realized_s","usable_s","missed_s","excess_s","admit_pct","Mbps","risk_pct","narrow_good_s"]
    def mix(arr):
        x,y=arr[...,0,:],arr[...,1,:]
        p=(7-y[...,7])/(x[...,7]-y[...,7])
        return p[...,None]*x+(1-p[...,None])*y,(p>=0)&(p<=1)
    result=[];est={};reps={};valid={}
    for a in ("G1","O0","O1","R0","R1"):
        arr=np.array([[[float((get(k,a,m)|ep[k,a,m])[key]) for key in fields] for m in (.4,.85)] for k in range(8)])
        key=a
        est[key],ok=mix(arr.mean(0));assert ok
        reps[key],valid[key]=mix(arr[boot].mean(1))
        for tag,vector in (("0.40",arr[:,0].mean(0)),("0.85",arr[:,1].mean(0)),("matched_7",est[key])):
            result.append({"arm":a,"operating_point":tag,**dict(zip(fields,vector))})
    save("hold_levels.csv",result)
    comparisons=[]
    for a,b in (("R1","R0"),("O1","O0"),("R1","G1"),("R1","O0"),("R0","O0")):
        ok=valid[a]&valid[b]
        for j,key in enumerate(fields):
            lo,hi=np.quantile(reps[a][ok,j]-reps[b][ok,j],[.025,.975])
            comparisons.append(dict(A=a,B=b,metric=key,difference=est[a][j]-est[b][j],lo=lo,hi=hi,feasible=ok.mean()))
    save("hold_contrasts.csv",comparisons)

if __name__=="__main__":main()
