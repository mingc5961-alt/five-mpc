import pickle,numpy as np
from unittest.mock import patch
from heads_new import packed_rows,paths,records,ARCHIVE,common,rc_features,errors,fit_heads
from protocol_local import OLD,OUT,log
from mileage import selected,infer
def train():
    m=selected('cpu')
    for split in ('train','validation','calibration'):
        rows=[]
        for f in paths(split):
            if split=='train' and int(f.stem)%5:continue
            a=pickle.loads((ARCHIVE/split/(f.stem+'.pkl')).read_bytes())
            packed=packed_rows(OLD/'packed'/split/f.name,a);history=list(records(f))
            cs=infer(m,packed,[h for h,t in history])
            for c,(h,t),r in zip(cs,history,a,strict=True):
                rows.append((int(f.stem),common(h,c),rc_features(r['package'],r['features']['P']),errors(c,t)))
        dst=OUT/'features/v2';dst.mkdir(exist_ok=True,parents=True)
        np.savez(dst/(split+'.npz'),**dict(zip(('ids','common','special','error'),map(np.stack,zip(*rows)),strict=True)))
    with patch.object(fit_heads,'OUT',OUT),patch.object(fit_heads,'KINDS',('special',)),patch.object(fit_heads,'log',log):fit_heads.fit('v2')
if __name__=='__main__':train()
