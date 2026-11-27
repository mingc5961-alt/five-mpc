from factorial_heads import *
from paid_full import chrono
import csv

if __name__=="__main__":
    waitfile(ROOT/"run_confirm_0_C_1.00.json")
    rows=[]
    for k in range(8):
        w=fresh_world("confirm",k)
        d=np.load(ROOT/f"eval_confirm{k}.npz")
        center_errors=d["err"]
        histories=[[] for _ in range(16)]
        batch=[]
        def score():
            if not batch:return
            preds=chrono().batch([s[2] for s in batch])
            for (u,t,h,row),(c,_) in zip(batch,preds):
                y=angle(truth(w,u,t+GRID[1:]))
                err=np.rad2deg(np.linalg.norm(difference(y,angle(c[1:])),axis=1))
                for name,values in (("C",err),("G1",center_errors[row,0,1:]),("R",center_errors[row,1,1:])):
                    rows.append([k,u,len(h),name,*values])
            batch.clear()
        for row,(u,t) in enumerate(schedule(w,41000+k)):
            h=histories[u];h.append(w.observe(u,t))
            if len(h) in (1,2,3,4,6,8,12,24,40):
                batch.append((u,t,list(h),row))
            if len(batch)==16:score()
        score()
    assert len(rows)==8*16*9*3
    with (ROOT/"chronos_panel.csv").open("w",newline="") as f:
        wr=csv.writer(f);wr.writerow(["world","u","n","arm",*GRID[1:]]);wr.writerows(rows)
