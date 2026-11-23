"""Isolated chapter-six runs; each CLI invocation has its own output directory."""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from routeclock.config import dump_json,load_config
from routeclock.dataset import load_anchors,observation_schedule
from routeclock.five import train_five,evaluate_anchors
from routeclock.foundation import ChronosAdapter
from routeclock.geometry import angles,angle_error
from routeclock.predictors import Tracker
from routeclock.scheduler import TokenBucket
from routeclock.simulator import simulate
from routeclock.training import inject_gate,fit_gate
from routeclock.world import World,rng_for

FORMAL = Path('/data/chen/formal_execution/run_20261117_001732')
COSTS = (3.1,6.2,9.3,12.4)
BUDGETS = (5.,2.5,5./3)


def prewarm(cfg,world,cost,adapter=None,expert=False,warm_budget=None):
    """Issue causal packages on a shared rotation ending at the service boundary."""
    sim = cfg['simulation']
    rho = sim['sensing_budget'] if warm_budget is None else warm_budget
    step = max(1.,world.n*cost/rho)
    bucket = TokenBucket(cost,rho)
    due = np.arange(world.n)*cost/rho
    trackers = [Tracker(cfg,adapter) for _ in due]
    now = 0.
    while True:
        uid = int(np.argmin(due))
        start = max(due[uid],bucket.ready(now))
        now = start+cost
        if now>sim['warmup_s']: break
        bucket.consume(start)
        trackers[uid].observe(world.observe(uid,now))
        trackers[uid].issue(include_chronos=expert)
        due[uid]=now+step
    return trackers


def protocol(args):
    cfg = load_config()
    sim = cfg['simulation']
    if args.smoke: sim.update(warmup_s=8.,service_s=6.)
    sim['observation_cost_s']=args.cost_ms/1000
    sim['sensing_budget']=args.budget_pct/100
    warm_cost=.0031 if args.warm=='fixed' else sim['observation_cost_s']
    return cfg,warm_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage',choices=['train','offline','gate-check','fit-gate-check','case0','recalibrate','network','select-fixed'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=FORMAL/'five_data')
    parser.add_argument('--bundle',type=Path,default=FORMAL/'five/lightgbm_7/bundle.pkl')
    parser.add_argument('--gate',choices=['full','scale','weighted','g1'],default='g1')
    parser.add_argument('--head',choices=['local','full'],default='local')
    parser.add_argument('--calibration',choices=['global','deadline'],default='global')
    parser.add_argument('--method',choices=['F','I','P','RC','C','CR_raw'],default='RC')
    parser.add_argument('--scheduler',choices=['value','edf','uniform'],default='value')
    parser.add_argument('--warm',choices=['fixed','lifecycle'],default='lifecycle')
    parser.add_argument('--use-calibration',choices=['global','deadline'],help='Evaluate fitted heads with the specified calibration')
    parser.add_argument('--cost-ms',type=float,choices=COSTS,default=12.4)
    parser.add_argument('--budget-pct',type=float,choices=BUDGETS,default=5.)
    parser.add_argument('--validation-start',type=int,default=1)
    parser.add_argument('--split',choices=['validation','test'],default='validation')
    parser.add_argument('--mission',type=int,default=0)
    parser.add_argument('--winner',type=Path)
    parser.add_argument('--checkpoint')
    parser.add_argument('--compact',action='store_true',help='Keep metrics and gate weights only')
    parser.add_argument('--smoke',action='store_true')
    a = parser.parse_args()
    if a.method in ('C','CR_raw') and not a.checkpoint: parser.error('supply local Chronos --checkpoint')
    if a.stage=='network' and a.method=='F' and not a.winner: parser.error('F requires --winner')
    cfg,warm_cost=protocol(a)
    a.out.mkdir(parents=True,exist_ok=False)
    dump_json(a.out/'arguments.json',vars(a))
    def anchors(split):
        count=cfg['data']['single_uav_tasks'][split]
        rows=[]
        for i in range(count):
            route=load_anchors(a.data/split/f'{i:04}.pkl')
            if 'recent_spacing_s' not in route[0]['package'].diagnostics:
                # Historical packages omit timestamps. Replay the generator's
                # observation clock only and check it against the saved count.
                rng=rng_for(f'offline/{split}',i,777)
                candidates=[rng.uniform(lo,hi) for lo,hi in cfg['data']['anchor_intervals_s']]
                selected=[float(rng.choice(candidates))] if split=='calibration' else candidates
                clock=np.array(observation_schedule(candidates,selected,[.5,1,2,5,10,None][i%6],rng))
                for row in route:
                    pkg=row['package']
                    history=clock[clock<=pkg.origin]
                    if history[-1]!=pkg.origin or len(history)!=round(row['features']['P'][0,26]*100):
                        raise ValueError(f'Anchor timestamp mismatch: {row["task"]}')
                    pkg.diagnostics['recent_spacing_s']=float(np.diff(history[-9:]).mean())
            rows.extend(route)
        return rows
    if a.stage=='train':
        train_five(anchors('train'),anchors('validation'),anchors('calibration'),[],cfg,a.out,
                   extras=False,gate_mode=a.gate,head_mode=a.head,calibration=a.calibration)
        return
    if a.stage=='select-fixed':
        from fixed_grid import fixed_scores
        rows=[]
        for mission in range(a.validation_start,a.validation_start+20):
            w=World(cfg,'chapter6/'+('smoke/' if a.smoke else '')+'validation',mission)
            trackers=prewarm(cfg,w,warm_cost,warm_budget=.05 if a.warm=="fixed" else None)
            rows.append(fixed_scores(cfg,w,trackers))
        candidates=[dict(pair=g[0]['pair'],interval=g[0]['interval'],net_mbps=np.mean([r['net_mbps'] for r in g]))
                    for g in zip(*rows)]
        dump_json(a.out/'grid.json',rows)
        dump_json(a.out/'winner.json',dict(**max(candidates,key=lambda r:r['net_mbps']),warm=a.warm,
                                          cost_ms=a.cost_ms,budget_pct=a.budget_pct,validation_start=a.validation_start,smoke=a.smoke))
        return
    with a.bundle.open('rb') as f: bundle=pickle.load(f)
    for head in bundle['heads'].values():
        if head.reg.kind=='lightgbm': head.reg.model.set_params(n_jobs=1)
    if a.stage=='recalibrate':
        rows=anchors('calibration')
        if len(rows)!=len({r['task'] for r in rows}):raise ValueError('One anchor per calibration route required')
        inject_gate(rows,bundle['gates']['RC'],'RC','P')
        for method,head in bundle['heads'].items():
            previous=head.factor
            head.calibrate(rows,method)
            np.testing.assert_allclose(head.factor,previous,rtol=1e-10)
        bundle['calibration']=a.calibration
        with (a.out/'bundle.pkl').open('wb') as f:pickle.dump(bundle,f)
        return
    if a.use_calibration:
        if a.use_calibration=='deadline' and not all(hasattr(h,'deadline_factors') for h in bundle['heads'].values()):
            raise ValueError('Retrain or recalibrate legacy heads before using deadline factors')
        bundle['calibration']=a.use_calibration
    if a.stage in ('gate-check','fit-gate-check'):
        if a.stage=='fit-gate-check':
            if a.split!='validation': raise ValueError('Development check uses validation only')
            bundle['gates']['RC'],_=fit_gate(anchors('train'),anchors('validation'),'P',7,mode=a.gate)
        rows=anchors(a.split)
        inject_gate(rows,bundle['gates']['RC'],'RC','P')
        records=[]
        for row in rows:
            pkg=row['package']
            for h in (.5,1.,2.,5.,10.):
                truth=[np.interp(h,row['truth_h'],axis) for axis in row['truth'].T]
                r=dict(task=row['task'],event=row['event'],h=h,
                       w=pkg.diagnostics['gate_weights']['RC'][int(2*h)])
                for method in ('I','P','RC'):
                    residual=angle_error(angles(pkg.at(method,h)),angles(truth))
                    r[method]=np.linalg.norm(np.rad2deg(residual))
                records.append(r)
        df=pd.DataFrame(records)
        df.to_csv(a.out/'anchors.csv',index=False)
        df.groupby(['event','h'])[['I','P','RC','w']].mean().to_csv(a.out/'events.csv')
        df.groupby(['task','h'])[['I','P','RC','w']].mean().groupby('h').mean().to_csv(a.out/'horizons.csv')
        return
    if a.stage=='offline':
        rows=anchors(a.split)
        inject_gate(rows,bundle['gates']['RC'],'RC','P')
        evaluate_anchors(rows,bundle,a.out)
        return
    if a.winner:
        winner=json.loads(a.winner.read_text())
        if winner.get('budget_pct',5.)!=a.budget_pct: raise ValueError('Fixed validation winner budget mismatch')
        if any(winner[k]!=v for k,v in [('warm',a.warm),('cost_ms',a.cost_ms),('smoke',a.smoke)]):
            raise ValueError('Fixed validation winner protocol mismatch')
        bundle['fixed']=winner
    if a.stage=='case0':
        if a.method not in ('I','RC') or a.smoke or a.cost_ms!=3.1 or a.budget_pct!=5.:
            raise ValueError('Historical case0 requires I/RC, 3.1 ms and full length')
        simulate(cfg,World(cfg,'network/test',20000),bundle,a.method,out=a.out)
        return
    namespace='chapter6/'+('smoke/' if a.smoke else '')+a.split
    expert=a.method if a.method in ('C','CR_raw') else False
    adapter=ChronosAdapter('cuda',a.checkpoint) if expert else None
    warm_budget=.05 if a.warm=='fixed' else cfg['simulation']['sensing_budget']
    if a.compact and not a.smoke and (not expert or Path(a.checkpoint).is_dir()):
        from ch6_cache import prepare
        w,trackers,timing=prepare(cfg,namespace,a.mission,warm_cost,warm_budget,adapter,expert,prewarm)
        dump_json(a.out/'cache.json',timing)
    else:
        w=World(cfg,namespace,a.mission)
        trackers=prewarm(cfg,w,warm_cost,adapter,expert,warm_budget)
    simulate(cfg,w,bundle,a.method,adapter,a.out,scheduler=a.scheduler,service_trackers=trackers,
             compact=a.compact or a.mission!=0)


if __name__=='__main__': main()
