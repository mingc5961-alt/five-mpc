"""CLI contract from EXPERIMENT_PLAN.md, with explicit verified dependency gates."""
import argparse
from design849 import *
from audit849 import validate
from gate849 import poolmap
from chronos_life_panel import inputs,fit
from labels849 import labelworld,trainvalue
def args():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['preflight','smoke','fit-life','fit-value','develop','freeze','confirm','report']);p.add_argument('--out',type=Path,required=True);p.add_argument('--world',type=int,default=1400);p.add_argument('--experiment',choices=['E1','E2']);return p.parse_args()
def main():
    a=args();assert a.out.resolve()==OUT
    if a.command=='preflight':
        if (OUT/'design/frozen.json').exists():check(get(OUT/'design/frozen.json'))
        else:preflight();anchor()
        verified_cal();return
    if a.command=='smoke':
        assert a.world==1400 and (OUT/'audits/legacy.json').is_file()
        phase='final-smoke' if (OUT/'design/frozen.json').exists() else 'smoke'
        poolmap(execute_endpoint,[(phase,'development',1400,r,s,.75) for r,s in MATRIX],10);panel((phase,'development',1400));validate(phase);return
    assert get(OUT/'audits/smoke.json')['status']=='PASS'
    w,_=world('development',1400);Model(w,'Chronos')
    if a.command=='fit-life':
        assert not (OUT/'design/frozen.json').exists()
        jobs=[('tube_fit',k) for k in range(1200,1216)]+[('tube_calibration',k) for k in range(1300,1316)]
        poolmap(shard,[(split,k,g) for split,k in jobs for g in (0,1)],capacity());poolmap(inputs,jobs,32);poolmap(fit,LEARN,4)
    elif a.command=='fit-value':
        assert not (OUT/'design/frozen.json').exists()
        jobs=[('train',k,8) for k in range(1000,1032)]+[('train_validation',k,1) for k in range(1100,1116)]+[('development',k,2) for k in range(1400,1408)]
        poolmap(labelworld,jobs,capacity());poolmap(trainvalue,['time','route'],2)
    elif a.command=='develop':
        poolmap(execute_endpoint,[('development','development',k,r,s,m) for m in SCALES for r,s in MATRIX for k in range(1400,1408)],capacity())
        poolmap(panel,[('development','development',k) for k in range(1400,1408)],8)
    elif a.command=='freeze':freeze()
    else:
        d=get(OUT/'design/frozen.json');check(d);assert get(OUT/'audits/final-smoke.json')['status']=='PASS'
        if a.command=='report':
            from report849 import report
            report()
        elif a.experiment=='E1':poolmap(panel,[('confirmation','confirmation',r['world']) for r in d['worlds']],capacity())
        elif a.experiment=='E2':poolmap(execute_endpoint,[('confirmation','confirmation',k,r,s,m) for k,r,s,m in d['base_endpoints']+d['extra_endpoints']],capacity())
        else:raise ValueError('confirm requires --experiment')
if __name__=='__main__':main()
