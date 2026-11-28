from audit849 import *
from chronos_life_panel import inputs,fit
from concurrent.futures import ProcessPoolExecutor,as_completed
import multiprocessing as mp
def poolmap(fn,jobs,n):
    if n>=12 and fn.__name__ not in ('inputs','audit_confirm_world'):
        from numapool849 import parallel
        return parallel(fn,jobs,n)
    with ProcessPoolExecutor(n,mp_context=mp.get_context('fork')) as p:
        futures=[p.submit(fn,j) for j in jobs]
        for f in as_completed(futures):print(fn.__name__,f.result(),flush=True)
def main():
    while not ((OUT/'E1/scores/smoke/1400.json').exists() and len(list((OUT/'E2/receipts/smoke').glob('*.json')))==10):time.sleep(4)
    validate('smoke')
    (OUT/'logs/status.txt').write_text('世界 1400 初核通过。正在拟合并独立校准风险头。\n')
    w,_=world('development',1400);Model(w,'Chronos')
    jobs=[('tube_fit',k) for k in range(1200,1216)]+[('tube_calibration',k) for k in range(1300,1316)]
    from parallel849 import shard,capacity
    poolmap(shard,[(split,k,group) for split,k in jobs for group in (0,1)],capacity())
    poolmap(inputs,jobs,min(32,capacity()))
    poolmap(fit,LEARN,4)
    from labels849 import labelworld,trainvalue
    jobs=[('train',k,8) for k in range(1000,1032)]+[('train_validation',k,1) for k in range(1100,1116)]+[('development',k,2) for k in range(1400,1408)]
    (OUT/'logs/status.txt').write_text('正在同一付费状态上生成候选用户的短窗反事实标签。\n')
    poolmap(labelworld,jobs,capacity())
    poolmap(trainvalue,['time','route'],2)
    from design849 import development_freeze_confirm
    development_freeze_confirm()
if __name__=='__main__':main()
