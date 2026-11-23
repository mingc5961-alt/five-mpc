from collections import Counter
from datetime import datetime
from pathlib import Path
import json
import time
from protocol import OUT,MODELS,log
if __name__=='__main__':
    last=-1
    while True:
        paths=list((OUT/'cells').glob('*/*/*.json'))
        n=len(paths)
        if n!=last:
            counts=Counter(p.parent.parent.name for p in paths)
            log(f'逐场核验 {n}/240；'+ '，'.join(f'{m} {counts[m]}/60' for m in MODELS)+'。每格均计分 180 秒，列入汇总前再次检查配对、误差头指纹、比特与四项时间账本。')
            last=n
        if n==240:break
        time.sleep(180)
