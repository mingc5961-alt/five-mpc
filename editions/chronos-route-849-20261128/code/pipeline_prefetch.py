from parallel849 import *
from prefix849 import collect
def work(spec):
    split,k=spec;tag=busy(f'{split}/{k} 提前计算后续需要的付费前缀')
    try:
        if split=='development':
            return collect(split,k)['prefixes']
        # The mixed behavior uses only EDF-native and its registered RNG, hence
        # the purchased history is completely independent of either fitted risk head.
        w,wm=world(split,k);cache=Cache(w,wm,split);eng=Engine(w,cache,.75,False,'EDF-native')
        number=8 if split=='train' else 1
        rng=np.random.default_rng(201127+k);eligible=np.flatnonzero((np.arange(len(TICKS))>=16)&(np.array(TICKS)+24<grid.opportunities))
        chosen={int(rng.choice(v)) for v in np.array_split(eligible,number)}
        for i in range(max(chosen)+1):
            eng.preservice(i)
            uid=i if i<16 else (int(rng.integers(16)) if rng.random()<.35 else int(np.argmin(eng.deadline())))
            eng.buy(i,uid,heads=False)
        return len(eng.events)
    finally:tag.unlink(missing_ok=True)
if __name__=='__main__':
    assert get(OUT/'audits/smoke.json')['status']=='PASS'
    w,_=world('development',1400);Model(w,'Chronos')
    # Once fitting finishes, the label workers take over the CPUs.
    jobs=[('development',k) for k in range(1401,1408)]+[('train',k) for k in range(1000,1007)]
    with ProcessPoolExecutor(14,mp_context=mp.get_context('fork')) as p:
        for f in as_completed([p.submit(work,j) for j in jobs]):print(f.result(),flush=True)
