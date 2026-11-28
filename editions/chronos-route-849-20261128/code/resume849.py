"""Verified restart by stage, including after an interrupted supervisor process."""
from design849 import *
from gate849 import poolmap
from audit849 import validate
from chronos_life_panel import inputs,fit
from labels849 import labelworld,trainvalue
def resume():
    from filelock import FileLock
    with FileLock(str(OUT/'logs/resume.lock'),timeout=0):
        if (OUT/'report_receipt.json').exists():
            r=get(OUT/'report_receipt.json');assert all(sha_file(OUT/p)==s for p,s in r['artifacts'].items());return
        dfile=OUT/'design/frozen.json'
        if dfile.exists():
            d=get(dfile);check(d)
        else:
            if not (OUT/'audits/smoke.json').is_file():
                poolmap(execute_endpoint,[('smoke','development',1400,r,s,.75) for r,s in MATRIX],10)
                panel(('smoke','development',1400));validate('smoke')
            w,_=world('development',1400);Model(w,'Chronos')
            fj=[('tube_fit',k) for k in range(1200,1216)]+[('tube_calibration',k) for k in range(1300,1316)]
            poolmap(inputs,fj,32)
            missing=[]
            for a in LEARN:
                p=OUT/f'E1/calibration/{a}.pkl';r=p.with_suffix('.json')
                if r.exists():assert sha_file(p)==get(r)['sha256']
                else:missing.append(a)
            if missing:poolmap(fit,missing,len(missing))
            ls=[('train',k,8) for k in range(1000,1032)]+[('train_validation',k,1) for k in range(1100,1116)]+[('development',k,2) for k in range(1400,1408)]
            (OUT/'logs/status.txt').write_text('独立风险头已校准。正在训练与验证世界并行生成同状态反事实标签。\n')
            poolmap(labelworld,ls,capacity())
            for side in ('time','route'):
                p=OUT/f'E2/labels/VALUE-{side}.txt';r=p.with_suffix('.json')
                if r.exists():assert sha_file(p)==get(r)['sha256']
                else:trainvalue(side)
        development_freeze_confirm()
if __name__=='__main__':resume()
