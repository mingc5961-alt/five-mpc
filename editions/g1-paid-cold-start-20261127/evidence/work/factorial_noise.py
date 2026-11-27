"""Oracle diagnostic only: split the received angular innovation into measurement and latent terms."""
from factorial_heads import *
import csv
def diagnose(split):
    rows=[];beta=get_beta()[[2,4,10,20]]
    for k in range(8):
        with np.load(ROOT/f"data_{split}{k}.npz") as a:
            w=fresh_world("confirm",k)
            acc=[];last={}
            for u,t,c,e,z in zip(a["u"],a["t"],a["c"],a["e"],a["z"]):
                old=last.get(u);last[u]=(t,c)
                if not z[1]:continue
                pos=np.array([np.interp(t-old[0],GRID,axis) for axis in old[1].T])
                actual=truth(w,u,np.array([t]))[0]
                latent=difference(angle(actual),angle(pos))
                noise=difference(angle(w.observe(u,t).position),angle(actual))
                assert np.max(abs(difference(e,latent+noise)))<1e-5
                y=angle(truth(w,u,t+np.array([1,2,5,10])))
                r=difference(y,angle(c[[2,4,10,20]]))
                losses=[]
                for v in (np.zeros(2),e,latent,noise):
                    losses.append(np.sum(np.rad2deg(difference(r,beta[:,None]*v))**2,axis=1))
                acc.append(losses)
            ave=np.mean(acc,0)
            for j,h in enumerate((1,2,5,10)):
                rows.append([k,h,len(acc),*ave[:,j],ave[0,j]-ave[1,j]])
    with (ROOT/f"noise_{split}.csv").open("w",newline="") as f:
        ww=csv.writer(f);ww.writerow(["world","h","issues","base_mse","observed_mse","oracle_latent_mse","oracle_noise_mse","observed_gain_deg2"]);ww.writerows(rows)
if __name__=="__main__":
    for s in ("confirm","shadow"):
        for k in range(8):waitfile(ROOT/f"eval_{s}{k}.npz")
        diagnose(s)
