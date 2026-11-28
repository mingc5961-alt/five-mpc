"""Paired G1/Chronos validity read-out on the existing nine-history panel."""
import csv
import pickle
from hold_diagnostic import ROOT,GRID,HS,DENSE,Radio,array_gain,truth,angle,np,raw,bundle,fresh_world
from paid_full import chrono,BASE,difference
from grudata import schedule

def one(radio,w,u,t,c,width,factors,m):
    point=angle(c); env=np.maximum(width,np.vstack([width[:1],width[:-1]]))
    ok=[np.max(radio.worst_losses(point[GRID<=h],(env*m*f)[GRID<=h],(32,32)))<=3 for f,h in zip(factors,HS)]
    adm=max(HS*np.logical_and.accumulate(ok))
    dense=np.column_stack([np.interp(DENSE,GRID,col) for col in c.T])
    good=np.logical_and.accumulate(array_gain(angle(dense),angle(truth(w,u,t+DENSE)),(32,32))/1024>=10**(-3/10))
    obs=max(HS*good[np.rint(HS*50).astype(int)])
    return adm,obs

if __name__=="__main__":
    with (ROOT/"chronos_panel.csv").open() as f:
        prior={(int(r["world"]),int(r["u"]),int(r["n"]),r["arm"]):float(r["10.0"]) for r in csv.DictReader(f)}
    cf=pickle.loads((BASE/"tubes/Chronos-2/7/bundle.pkl").read_bytes())["factors"]["native"]
    with (ROOT/"hold_chronos_panel.csv").open("w",newline="") as f:
        out=csv.writer(f);out.writerow(["world","u","n","arm","scale","admitted_s","realized_s"])
        for k in range(8):
            w=fresh_world("confirm",k);radio=Radio(w.cfg)
            d=np.load(ROOT/f"data_confirm{k}.npz");width=raw(bundle()["tree"],d["f"])
            histories=[[] for _ in range(16)];selected=[]
            for row,(u,t) in enumerate(schedule(w,41000+k)):
                h=histories[u];h.append(w.observe(u,t))
                if len(h) in (1,2,3,4,6,8,12,24,40):selected.append((row,u,t,list(h)))
            for j in range(0,len(selected),16):
                sub=selected[j:j+16]
                preds=chrono().batch([v[3] for v in sub])
                for (row,u,t,h),(cc,ww) in zip(sub,preds):
                    for a,c,s,fa,scales in (("G1",d["c"][row],width[row],bundle()["final"],(.45,.75)),("C",cc,ww,cf,(.75,1.))):
                        err=np.rad2deg(np.linalg.norm(difference(angle(c[-1]),angle(truth(w,u,np.array([t+10]))[0]))))
                        assert abs(err-prior[k,u,len(h),a])<.001
                        for m in scales:out.writerow([k,u,len(h),a,m,*one(radio,w,u,t,c,s,fa,m)])
            print(k,flush=True)
