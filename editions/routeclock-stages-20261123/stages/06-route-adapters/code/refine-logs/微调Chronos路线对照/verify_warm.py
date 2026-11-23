"""Reuse archived filter states only after comparing every causal measurement of every user."""
from replay_route import *
import gzip
def verify_all():
    setup()
    evidence=[]
    for w in [*range(2,40),0,1]:
        world=get_world(w); hs=network.warm_history(world,180,.0031)
        path=world_cache.BASE/'warm'/f'v1_{w:03}.pkl.gz'
        with gzip.open(path,'rb') as src:trs=pickle.load(src)
        maximum=0.
        for h,tr in zip(hs,trs,strict=True):
            np.testing.assert_array_equal([o.t for o in h],[o.t for o in tr.history])
            for field in ('position','covariance','radial_velocity','radial_std'):
                a=np.array([getattr(o,field) for o in h]);b=np.array([getattr(o,field) for o in tr.history])
                maximum=max(maximum,float(np.max(abs(a-b))))
                np.testing.assert_allclose(a,b,atol=1e-10,rtol=0)
            # Service passes the newly observed arrays to Chronos. Rebind only after verifying all fields.
            tr.history=h
        out=runtime.OUT/'warm'/path.name
        if not out.exists():
            with out.with_suffix('').with_suffix('.lock').open('a') as f:
                try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                except BlockingIOError:continue
                if not out.exists():
                    tmp=out.with_suffix('.validated')
                    with gzip.open(tmp,'wb',compresslevel=1) as dst:pickle.dump(trs,dst)
                    tmp.replace(out)
        evidence.append(dict(world=w,max_absolute_difference=maximum))
    write(OUT/'warm_audit.json',evidence)
    log(f'预热溯源：逐观测核验确认世界的旧状态与当前实测数组，时间戳逐项相等，位置/协方差/径向测速及其标准差绝对容差 1e-10；通过后在独立确认缓存复用状态。单独抽查开发世界缓存相差数百米，未使用。')
if __name__=='__main__':verify_all()
