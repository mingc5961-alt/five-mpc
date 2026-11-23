import replay_route as rr
from replay_route import *
from fast_memory import vectorized_forecast

def check(arm,w,b):
    setup();world=get_world(w);cfg=load_config();cfg['simulation']['sensing_budget']=runtime.BUDGET[b]
    radio=Radio(cfg);radio.threshold-=5
    ref=json.loads((OUT/'cells'/f'{w:03}_{b}_{arm}.json').read_text())['result']
    eng=Routed() if arm.startswith('CR') else None
    model=Model(arm,world,eng)
    start=time.monotonic()
    with runtime.controller(radio,10.),patch.object(network,'Ledger',HoldLedger),vectorized_forecast():
        r,_=network.run_world(cfg,world,model,{'factors':{arm:FACTORS[arm]}},arm)
    assert r.keys()==ref.keys()
    maxdiff=0.
    for k in r:
        if isinstance(r[k],str):assert r[k]==ref[k]
        else:
            np.testing.assert_allclose(r[k],ref[k],rtol=1e-12,atol=1e-9,err_msg=k)
            maxdiff=max(maxdiff,float(np.abs(np.asarray(r[k])-np.asarray(ref[k])).max()))
    write(OUT/('fast_check_'+arm+'.json'),dict(arm=arm,elapsed=time.monotonic()-start,max_abs_difference=maxdiff))
if __name__=='__main__':check(sys.argv[1],0,'T')
