"""Paired estimates for the original G1 and Chronos at their primary grid points."""
from hold_summary import read,save,np

def main():
    data=read("hold_chronos_panel.csv")
    assert len(data)==4608
    ends=read("confirm_endpoints.csv")
    fields=("admitted_s","realized_s","usable_s","missed_s","excess_s","risk_pct","Mbps","narrow_good_s","abs_mismatch_s")
    rng=np.random.default_rng(270228);boot=rng.integers(8,size=(20000,8))
    results=[];samples={};est={};feasible={}
    def mix(z):
        x,y=z[...,0,:],z[...,1,:];p=(7-y[...,5])/(x[...,5]-y[...,5])
        return x*p[...,None]+y*(1-p[...,None]),(p>=0)&(p<=1)
    for arm,ms in (("G1",(.45,.75)),("C",(.75,1.))):
        worlds=[]
        for k in range(8):
            grid=[]
            for m in ms:
                r=[v for v in data if v["arm"]==arm and int(v["world"])==k and float(v["scale"])==m]
                assert len(r)==144 and len({(v["u"],v["n"]) for v in r})==144
                t,p=np.array([[float(v["realized_s"]),float(v["admitted_s"])] for v in r]).T
                u=np.minimum(t,p)
                e=next(v for v in ends if v["arm"]==arm and int(v["world"])==k and float(v["scale"])==m)
                grid.append([p.mean(),t.mean(),u.mean(),(t-u).mean(),(p-u).mean(),float(e["risk_pct"]),float(e["Mbps"]),float(e["narrow_good_s"]),np.abs(p-t).mean()])
            worlds.append(grid)
        z=np.asarray(worlds)
        est[arm],ok=mix(z.mean(0));assert ok
        samples[arm],feasible[arm]=mix(z[boot].mean(1))
        results.append(dict(arm=arm,**dict(zip(fields,est[arm]))))
    save("hold_panel_levels.csv",results)
    ok=feasible["G1"]&feasible["C"];interval=np.quantile((samples["G1"]-samples["C"])[ok],[.025,.975],axis=0)
    save("hold_panel_contrasts.csv",[dict(metric=key,G1_minus_C=est["G1"][j]-est["C"][j],lo=interval[0,j],hi=interval[1,j],feasible=ok.mean()) for j,key in enumerate(fields)])
if __name__=="__main__":main()
