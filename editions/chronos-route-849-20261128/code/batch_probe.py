"""Numerical and throughput diagnostic only, outside the experiment cohorts."""
from base849 import *
import time
from grudata import schedule
def main():
    w,wm=load_checked('confirmation',2000)
    m=Model(w,'Chronos');hs=[[] for _ in range(16)]
    for u,t in schedule(w,43000):
        hs[u].append(w.observe(u,t))
        if t>55:break
    net=m.base.net;t=time.monotonic();solo=[net.batch([h])[0] for h in hs];single=time.monotonic()-t
    tests={}
    for b in [2,4,8,16]:
        t=time.monotonic();bs=[]
        for i in range(0,len(hs),b):bs.extend(net.batch(hs[i:i+b]))
        tests[b]={'seconds':time.monotonic()-t,'center_equal':sum(np.array_equal(x[0],y[0]) for x,y in zip(solo,bs)),'width_equal':sum(np.array_equal(x[1],y[1]) for x,y in zip(solo,bs)),'center_max_abs':max(float(np.max(abs(x[0]-y[0]))) for x,y in zip(solo,bs))}
    put(OUT/'audits/batch_benchmark.json',dict(world=2000,samples=len(hs),single_seconds=single,tests=tests,used_for_science=False))
if __name__=='__main__':main()
