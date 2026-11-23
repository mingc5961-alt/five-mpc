import csv
import fcntl
import time
from contextlib import contextmanager
from unittest.mock import patch
from types import SimpleNamespace
import study as st
from study import *
from features import Advanced, rc_features, rc_params, apply_gate
from runtime.fast_ledger import DeferredLedger
from routeclock.radio import Radio
from routeclock.memory import Fragment,RouteMemory
from parallel_frontier import frontier

NS='confirm_0db_20261122/v1'
ARMS=['C-native','C-special','CR-native','CR-transfer','CR-special','CRG-transfer','CRG-special','RC-special']
ASSETS=pickle.loads((OUT/'fitted.pkl').read_bytes())
CN=pickle.loads((ROOT/'tubes/Chronos-2/7/bundle.pkl').read_bytes())
RCHEAD=pickle.loads((SPECIAL/'heads/RC/special.pkl').read_bytes())
HEADS={'C-special':ORIGINAL,'RC-special':RCHEAD,**{k:ASSETS[k] for k in ARMS if k in ASSETS and k!='CR-native'}}
FACTORS={**{k:v['factors'] for k,v in HEADS.items()},'C-native':CN['factors']['native'],'CR-native':ASSETS['CR-native']}

@contextmanager
def memoized_arc():
    if (OUT/'native_check_CR-transfer.json').exists():
        from native_map import native_forecast
        with native_forecast():yield
        return
    if (OUT/'fast_check_RC-special.json').exists():
        from fast_memory import vectorized_forecast
        with vectorized_forecast():yield
        return
    old=RouteMemory.forecast
    arc=Fragment.arc
    def forecast_memory(mem,*args,**kwargs):
        cached={}
        def length(f):
            key=id(f)
            if key not in cached:cached[key]=arc(f)
            return cached[key]
        with patch.object(Fragment,'arc',length):return old(mem,*args,**kwargs)
    with patch.object(RouteMemory,'forecast',forecast_memory):yield

class HoldLedger(DeferredLedger):
    def result(self,cost,rho):
        result=super().result(cost,rho)
        state=np.array(self.states); ends=np.asarray(self.ends); starts=np.r_[self.warm,ends[:-1]]
        holds=[];safe_seconds=0.;count=0
        for uid in range(self.world.n):
            a=state[:,uid]; expiry=a[:,0]+a[:,2]
            stop=np.minimum(ends,expiry)
            valid=(a[:,2]>0)&(stop>starts)
            key=a[:,[0,3,4]]
            continuation=valid[1:]&valid[:-1]&(stop[:-1]==starts[1:])&(key[1:]==key[:-1]).all(1)
            first=np.where(valid & ~np.r_[False,continuation])[0]
            last=np.where(valid & ~np.r_[continuation,False])[0]
            assert len(first)==len(last)
            spans=stop[last]-starts[first]
            assert abs(spans.sum()-np.maximum(0,np.where(valid,stop-starts,0)).sum())<1e-6
            holds.extend(spans)
        result.update(hold_segments=len(holds),hold_sum_s=float(np.sum(holds)),hold_mean_s=float(np.mean(holds)),hold_p50_s=float(np.median(holds)),hold_p90_s=float(np.quantile(holds,.9)))
        return result

class Model:
    def __init__(self,arm,world,engine=None):
        self.arm=arm;self.routed=arm.startswith('CR')
        self.only_route=arm.startswith('CR-')
        self.events=0;self.world_id=world.mission
        self.net=engine
        if engine is not None and (OUT/'graph_validation.json').exists():
            from graph_encoder import enable
            backbone=engine.engine.net if self.routed else engine.net.net
            enable(backbone)
        self.track={}
        if self.routed or arm=='RC-special':
            self.warm=runtime.warm_trackers(world)
        if arm=='RC-special':self.gate=rc_params()['gate']
    def history_packages(self,histories):
        if not self.track:
            for h,tr in zip(histories,self.warm,strict=True):
                assert np.array_equal([v.t for v in h],[v.t for v in tr.history])
                assert np.array_equal([v.position for v in h],[v.position for v in tr.history])
                self.track[id(h)]=tr
        pkgs=[]
        for h in histories:
            tr=self.track[id(h)]
            package=tr.snapshots[-1]
            for o in h[len(tr.history):]:
                if getattr(self,'only_route',False):
                    tr.imm.observe(o)
                    tr.memory.update(o,tr.imm.state()[0][3:])
                    tr.history.append(o)
                    center,cov,support,_,_=tr.memory.forecast(HS,tr.point_progress)
                    package=SimpleNamespace(origin=o.t,centers={'P':center},covariance={'P':cov},diagnostics={'support':support})
                    tr.snapshots.append(package)
                else:
                    tr.observe(o);package=tr.issue()
            pkgs.append(package)
        return pkgs
    def batch(self,histories):
        self.events+=1
        if self.events%250==0:print(self.arm,self.world_id,'updates',self.events,'observation time',round(histories[0][-1].t,3),flush=True)
        if self.arm=='C-native':return self.net.net.batch(histories)
        if self.arm=='C-special':
            rows=self.net.extract(histories,True)
        else:
            pkgs=self.history_packages(histories)
            if self.routed:
                widths=[]
                def remember(q,c,w):
                    widths.append(w)
                    return private_features(q,c,w)
                with patch.object(st,'private_features',remember):
                    rows=self.net.batch(histories,pkgs)
                if self.arm=='CR-native':
                    return [(c,w) for (c,_),w in zip(rows,widths,strict=True)]
                if self.arm.startswith('CRG'):
                    augmented=[]
                    for h,p,(c,z) in zip(histories,pkgs,rows,strict=True):
                        i=p.centers['I']
                        fused,w=fused_values(ASSETS['gate'],gate_x(p,p.features,h),c,i)
                        if self.arm=='CRG-special':z=np.column_stack([z,w[K],c[K]-i[K]])
                        augmented.append((fused,z))
                    rows=augmented
            else:
                rows=[]
                for p in pkgs:
                    apply_gate(p,self.gate,p.features,'RC','P')
                    rows.append((p.centers['RC'],rc_features(p,p.features)))
        x=np.stack([common(h,c) for h,(c,_) in zip(histories,rows,strict=True)])
        scales=predict(HEADS[self.arm],x,np.stack([z for _,z in rows]))
        return [(r[0],s) for r,s in zip(rows,scales,strict=True)]

def setup():
    world_cache.BASE=ROOT/'work/baseline_cache'
    runtime.OUT=OUT/'confirm_cache'
    runtime.OUT.mkdir(exist_ok=True)
    network.frontier=frontier
def get_world(w):
    cfg=load_config();cfg['simulation']['sensing_budget']=runtime.BUDGET['T']
    return world_cache.paired_world(cfg,NS,w)

def run_cell(arm,w,b,engine):
    start=time.monotonic()
    world=get_world(w)
    cfg=load_config();rho=runtime.BUDGET[b];cfg['simulation']['sensing_budget']=rho
    radio=Radio(cfg);radio.threshold-=5
    model=Model(arm,world,engine)
    with runtime.controller(radio,10.),patch.object(network,'Ledger',HoldLedger),memoized_arc():
        result,_=network.run_world(cfg,world,model,{'factors':{arm:FACTORS[arm]}},arm)
    assert abs(sum(result['bits'])/180e6-result['G_mbps'])<1e-7
    assert result['ledger_s'][0]<=rho*180+.0031001 and abs(result['ledger_s'][0]-.0031*result['updates'])<1e-6
    assert result['probes_per_user']==9000 and 0<=result['unavailable_fraction']<=result['R']<=1
    assert 0<result['hold_mean_s']<=10.000001 and result['hold_sum_s']<=16*180+.00001
    path=ROOT/'results/confirmation/Chronos-2/7'/f'{w:03}_{b}_native/complete.json'
    if arm in ('C-native','C-special') and path.exists():
        ref=json.loads(path.read_text())['result']
        assert result['initial_center_sha256']==ref['initial_center_sha256']
        if arm=='C-native':
            # Archived runs may use different GPU generations. Quantify rather than ignore drift.
            result['reference_delta_mbps']=result['G_mbps']-ref['G_mbps']
    receipt=dict(arm=arm,world=w,budget=b,namespace=NS,seed=7,mcs_margin_db=0,fitted_sha256=hashlib.sha256((OUT/'fitted.pkl').read_bytes()).hexdigest(),center_sha256=CFG['center_sha256']['Chronos-2'],result=result,elapsed_s=time.monotonic()-start)
    write(OUT/'cells'/f'{w:03}_{b}_{arm}.json',receipt)
    count=len(list((OUT/'cells').glob('*.json')))
    log(f'闭环 {count}/240，世界 {w}、{b}、{arm}，净吞吐 {result["G_mbps"]:.2f} Mbps，总风险 {100*result["R"]:.3f}%，连续保持均值 {result["hold_mean_s"]:.3f} s；180 s 账本及实际感知核对通过。')

def worker(which):
    setup();engines={}
    tasks=[(arm,w,b) for w in range(10) for b in ('T','M','L') for arm in ARMS]
    if which=='cpu':tasks=[v for v in tasks if v[0]=='RC-special']
    else:tasks=[v for v in tasks if v[0]!='RC-special']
    (OUT/'locks').mkdir(exist_ok=True)
    for arm,w,b in tasks:
        key=f'{w:03}_{b}_{arm}'
        with (OUT/'locks'/(key+'.lock')).open('a') as f:
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:continue
            if (OUT/'cells'/(key+'.json')).exists():continue
            kind='route' if arm.startswith('CR') else ('C' if arm.startswith('C') else 'rc')
            if kind not in engines:
                if kind=='route' and 'C' in engines:
                    engines[kind]=Routed.__new__(Routed);engines[kind].engine=engines['C'].net
                    engines[kind].ids=[engines[kind].engine.net.chronos_config.quantiles.index(q) for q in (.1,.5,.9)]
                elif kind=='C' and 'route' in engines:
                    engines[kind]=Advanced.__new__(Advanced);engines[kind].name='Chronos-2';engines[kind].net=engines['route'].engine
                else:engines[kind]=Routed() if kind=='route' else Advanced('Chronos-2') if kind=='C' else None
            run_cell(arm,w,b,engines[kind])
if __name__=='__main__':worker(sys.argv[1])
