"""Post-hoc validity diagnostic on existing common-prefix forecasts. No refitting."""
import csv
import json
from factorial_heads import ROOT, HS, GRID, DENSE, ARMS, bundle, raw, inputs, centers, cond_width, fresh_world, truth, angle, np, pickle
from routeclock.radio import Radio, array_gain

def world_rows(k):
    w=fresh_world("confirm",k)
    radio=Radio(w.cfg)
    with (ROOT/"heads.pkl").open("rb") as f: beta,heads=pickle.load(f)
    d=np.load(ROOT/f"data_confirm{k}.npz")
    basis=raw(bundle()["tree"],d["f"])
    xx=inputs(d["f"],d["e"],d["z"],basis)
    orig=d["c"]
    corrected=centers(d,beta)
    fac=json.loads((ROOT/"factors.json").read_text())
    fac["G1"]=json.loads((ROOT/"g1_original_factors.json").read_text())
    widths={"G1":basis,"O0":basis,"R0":basis,"O1":cond_width(heads[0],xx,basis),"R1":cond_width(heads[1],xx,basis)}
    rows=[]
    for mode,c in (("O",orig),("R",corrected)):
        passed=[]
        for center,u,t in zip(c,d["u"],d["t"]):
            dense=np.column_stack([np.interp(DENSE,GRID,col) for col in center.T])
            loss_gain=array_gain(angle(dense),angle(truth(w,u,t+DENSE)),(32,32))/1024
            ok=np.logical_and.accumulate(loss_gain>=10**(-3/10))
            passed.append(ok[np.rint(HS*50).astype(int)])
        actual=np.asarray(passed)
        vtrue=(actual*HS).max(axis=1)
        for arm in (("G1","O0","O1") if mode=="O" else ("R0","R1")):
            wid=widths[arm]
            env=np.maximum(wid,np.concatenate([wid[:,:1],wid[:,:-1]],axis=1))
            for m in (.4,.85):
                checks=[]
                for j,h in enumerate(HS):
                    loss=radio.worst_losses(angle(c).reshape(-1,2), (env*m*fac[arm][j]).reshape(-1,2),(32,32)).reshape(len(c),-1)
                    checks.append(np.max(loss[:,GRID<=h],axis=1)<=3.)
                eligible=np.logical_and.accumulate(np.stack(checks,axis=1),axis=1)
                v=(eligible*HS).max(axis=1)
                if k==0:
                    from simulate import frontier
                    for idx in (0,1,83):
                        options=frontier(radio,c[idx],wid[idx],m*np.array(fac[arm]))
                        ref=max((o.interval for o in options if o.pair==(32,32)),default=0.)
                        assert ref==v[idx]
                for row,(u,t,p,q) in enumerate(zip(d["u"],d["t"],v,vtrue)):
                    rows.append([k,int(u),float(t),int(np.rint(np.expm1(d["z"][row,0]))),arm,m,p,q])
    return rows

if __name__=="__main__":
    with (ROOT/"hold_prefixes.csv").open("w",newline="") as f:
        out=csv.writer(f);out.writerow(["world","u","issue_s","n","arm","scale","admitted_s","realized_s"])
        for k in range(8):out.writerows(world_rows(k));print(k,flush=True)
