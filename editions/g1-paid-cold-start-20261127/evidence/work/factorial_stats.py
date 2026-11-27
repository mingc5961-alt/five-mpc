"""Rebuild tables solely from paired receipts and common-prefix scores."""
from pathlib import Path
from itertools import combinations
import csv,json
import numpy as np
W=Path(__file__).resolve().parent
D=W/"factorial_run"
ARMS=("O0","O1","R0","R1")
H=np.array([.5,1,2,3,5,8,10])
B=np.random.default_rng(271127).integers(8,size=(20000,8))
def write(name,header,rows):
    with (D/name).open("w",encoding="utf-8-sig",newline="") as f:csv.writer(f).writerows([header,*rows])
def ci(z):
    return np.quantile(z[B].mean(1),[.025,.975])
def equalquant(samples,prob):
    z=np.concatenate(samples);weights=np.concatenate([np.full(len(x),1/len(x)/len(samples)) for x in samples])
    perm=np.argsort(z)
    return float(z[perm][np.searchsorted(np.cumsum(weights[perm]),prob)])
def offline(split):
    paths=[D/f"eval_{split}{k}.npz" for k in range(8)]
    worlds=[dict(np.load(p)) for p in paths]
    factors=json.loads((D/"factors.json").read_text())
    rows=[];cent=[];conditions=[];persist={}
    for k,x in enumerate(worlds):
        first=np.isclose(np.expm1(x["z"][:,0]),1.)
        assert sum(first)==16 and np.array_equal(x["err"][first,0],x["err"][first,1])
        assert np.all(np.diff(x["score"],axis=2)>=0)
        for b in (0,1):
            for h in (.5,1.,2.,5.,10.):
                r=x["err"][:,b,int(2*h)]
                cent.append([k,("O","R")[b],h,len(r),r.mean(),np.mean(r*r)])
        assert len(x["area"])==len(x["z"]) and np.isfinite(x["score"]).all()
        for n0,n1 in [(1,1),(2,3),(4,8),(9,10000)]:
            n=np.rint(np.expm1(x["z"][:,0]))
            sel=(n>=n0)&(n<=n1)
            for j,a in enumerate(ARMS):
                e=x["err"][sel,j//2,20]
                conditions.append([k,n0,n1,a,int(sel.sum()),e.mean(),np.mean(x["score"][sel,j,-1]<=factors[a][-1])])
    for j,a in enumerate(ARMS):
        ss=[x["score"][:,j,-1] for x in worlds]
        aa=np.array([x["area"][:,j].mean() for x in worlds])
        q=equalquant(ss,.90)
        persist[a]={"q":q,"area90":aa.mean()*q*q}
        for k,x in enumerate(worlds):
            for ih,h in enumerate(H):
                f=factors[a][ih]
                rows.append([k,a,h,np.mean(x["score"][:,j,ih]<=f),f,aa[k]*f*f if h==10 else "",aa[k]*q*q if h==10 else ""])
    original=json.loads((D/"g1_original_factors.json").read_text())
    for k,x in enumerate(worlds):
        for ih,h in enumerate(H):
            f=original[ih];area=x["area"][:,0].mean()
            rows.append([k,"G1",h,np.mean(x["score"][:,0,ih]<=f),f,area*f*f if h==10 else "",area*persist["O0"]["q"]**2 if h==10 else ""])
    write(f"{split}_coverage.csv",["world","arm","h","coverage","frozen_factor","area_deg2_10s","area_at_observed_90pct"],rows)
    write(f"{split}_center.csv",["world","center","h","issues","mean_deg","mse_deg2"],cent)
    write(f"{split}_prefix.csv",["world","n_min","n_max","arm","issues","10s_deg","path_coverage"],conditions)
    # Re-estimate 90% scores for both arms in each paired Bayesian-free multinomial world bootstrap.
    # Only distinct bootstrap count vectors need a quantile lookup.
    patterns,inv=np.unique(np.stack([(B==k).sum(1) for k in range(8)],1),axis=0,return_inverse=True)
    byarm={}
    for j,a in enumerate(ARMS):
        s=np.sort(np.concatenate([x["score"][:,j,-1] for x in worlds]))
        cdf=np.stack([np.searchsorted(np.sort(x["score"][:,j,-1]),s,side="right")/len(x["u"]) for x in worlds])
        ar=np.array([x["area"][:,j].mean() for x in worlds]); qq=[]
        for block in np.array_split(patterns,32):
            c=(block@cdf)/8
            ix=np.argmax(c>=.90,axis=1)
            qq.extend((s[ix]**2*(block@ar/8)).tolist())
        byarm[a]=np.array(qq)[inv]
    contrasts=[]
    for a,b in [("O1","O0"),("R0","O0"),("R1","R0"),("R1","O1"),("R1","O0")]:
        c=np.quantile(byarm[a]-byarm[b],[.025,.975])
        contrasts.append([a,b,persist[a]["area90"]-persist[b]["area90"],*c])
    signs={"R1":1,"R0":-1,"O1":-1,"O0":1}
    eff=sum(sign*persist[a]["area90"] for a,sign in signs.items())
    interval=np.quantile(sum(sign*byarm[a] for a,sign in signs.items()),[.025,.975])
    contrasts.append(["interaction","",eff,*interval])
    write(f"{split}_area_contrasts.csv",["A","B","difference","lo","hi"],contrasts)
    return persist

def best(pts,t):
    # Input parameter shapes are (..., endpoints, metrics); all arrays retain their world pairing.
    candidates=[]
    for i,j in combinations(range(pts.shape[-2]),2):
        g,r=pts[...,i,0],pts[...,i,1];v,s=pts[...,j,0],pts[...,j,1]
        p=np.divide(t-s,r-s,out=np.full_like(r,np.nan),where=np.abs(r-s)>1e-12)
        candidates.append(np.where((p>=-1e-8)&(p<=1+1e-8),p*g+(1-p)*v,-np.inf))
    return np.max(candidates,axis=0)

def closed():
    pol=json.loads((D/"freeze.json").read_text())
    raw={};finger={};details=[]
    for f in D.glob("run_confirm_*_*.json"):
        d=json.loads(f.read_text());k,a,m=d["k"],d["arm"],d["m"];r=d["result"]
        assert finger.setdefault(k,d["world"])==d["world"]
        assert abs(sum(r["ledger_s"])-180)<1e-6 and abs(len(d["events"])*.0031-r["ledger_s"][0])<1e-7
        assert r["ledger_s"][0]<=3.00001 and r["probes_per_user"]==9000 and len(r["bits"])==16
        assert abs(sum(r["bits"])/180e6-r["G_mbps"])<1e-6
        raw[k,a,m]=np.array([r["G_mbps"],r["R"]*100,r["narrow_good_data_s"],r["ledger_s"][0],r["unavailable_fraction"]*100])
        details.append([k,a,m,*raw[k,a,m],d["world"]])
    assert len(set(finger.values()))==8 and len(details)==104
    dev_hashes=set()
    for k in range(8):
        dev_paths=list(D.glob(f"run_risk_{k}_*.json"))
        for a,scales in (("G1",(.45,.75)),("C",(.75,1.))):
            dev_paths.extend(W/"g1mix7"/f"run_test_{k}_{a}_{m:.2f}.json" for m in scales)
        hashes={json.loads(path.read_text())["world"] for path in dev_paths}
        assert len(dev_paths)==12 and len(hashes)==1
        dev_hashes.update(hashes)
    assert len(dev_hashes)==8 and dev_hashes.isdisjoint(finger.values())
    hdr=["Mbps","risk_pct","narrow_good_s","sense_s","unavailable_pct"]
    write("confirm_endpoints.csv",["world","arm","scale",*hdr,"world_sha"],details)
    rr=[];vals={};curves={};paired=[];equal=[];elig=[]
    for a,p in (pol|{"G1x1":{"1.0":1},"Cx1":{"1.0":1}}).items():
        orig=a.split("x")[0]
        vals[a]=np.stack([sum(v*raw[k,orig,float(m)] for m,v in p.items()) for k in range(8)])
        rr.extend([[k,a,*z] for k,z in enumerate(vals[a])])
        if a in pol:
            ms=sorted(m for k,b,m in raw if k==0 and b==a)
            assert all(set(ms)=={m for kk,b,m in raw if kk==k and b==a} for k in range(8))
            curves[a]=np.stack([[raw[k,a,m] for m in ms] for k in range(8)])
    write("confirm_frozen.csv",["world","arm",*hdr],rr)
    contrasts=[("R0","O0"),("O1","O0"),("R1","O1"),("R1","R0"),("R1","O0"),("R1","G1"),("R1","C"),("G1","C"),("C","Cx1")]
    for a,b in contrasts:
        diff=vals[a]-vals[b]
        for j,label in enumerate(hdr):paired.append([a,b,label,diff[:,j].mean(),*ci(diff[:,j]),int((diff[:,j]>0).sum())])
    interaction=vals["R1"]-vals["O1"]-vals["R0"]+vals["O0"]
    for j,label in enumerate(hdr):paired.append(["interaction","",label,interaction[:,j].mean(),*ci(interaction[:,j]),int((interaction[:,j]>0).sum())])
    write("confirm_paired.csv",["A","B","metric","delta","lo","hi","positive_worlds"],paired)
    lo=max(v.mean(0)[:,1].min() for v in curves.values());hi=min(v.mean(0)[:,1].max() for v in curves.values())
    if lo>hi:raise ValueError(f"disjoint held-out ranges {lo} {hi}")
    t=float(np.clip(7.,lo,hi))
    point={a:float(best(v.mean(0),t)) for a,v in curves.items()}
    matched=[]
    for a,v in curves.items():
        scales=sorted(m for k,arm,m in raw if k==0 and arm==a)
        options=[]
        for i,j in combinations(range(len(scales)),2):
            u,z=v.mean(0)[[i,j]]
            if u[1]==z[1]:continue
            p=(t-z[1])/(u[1]-z[1])
            if -1e-8<=p<=1+1e-8:options.append(((p*u+(1-p)*z)[0],i,j,p))
        _,i,j,p=max(options)
        value=(p*v[:,i]+(1-p)*v[:,j]).mean(0)
        matched.append([a,scales[i],scales[j],p,*value])
    write("confirm_equalrisk_levels.csv",["arm","m1","m2","p1",*hdr],matched)
    reps={a:best(v[B].mean(1),t) for a,v in curves.items()}
    ok=np.logical_and.reduce([np.isfinite(v) for v in reps.values()])
    for a,b in contrasts[:-1]:
        delta=(reps[a][ok]-reps[b][ok]);l,h=np.quantile(delta,[.025,.975]);equal.append([a,b,t,point[a],point[b],point[a]-point[b],l,h,ok.mean()])
    v=reps["R1"][ok]-reps["O1"][ok]-reps["R0"][ok]+reps["O0"][ok]
    equal.append(["interaction","",t,"","",point["R1"]-point["O1"]-point["R0"]+point["O0"],*np.quantile(v,[.025,.975]),ok.mean()])
    write("confirm_equalrisk.csv",["A","B","risk_pct","A_Mbps","B_Mbps","delta","lo","hi","feasible_boot"],equal)
    return vals,point,t

if __name__=="__main__":
    for s in ("confirm","shadow"):print(s,offline(s),flush=True)
    vals,p,t=closed();print(t,p,{a:v.mean(0).tolist() for a,v in vals.items()})
