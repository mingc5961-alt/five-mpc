from replay_route import *
import pandas as pd
results=[]
def record(label,cover,mean_width):
    w=np.asarray(mean_width)
    if w.ndim>1:w=w.mean(axis=1)
    for h,a,b in zip(QUERY_H[1:],cover,w,strict=True):results.append((label,h,100*a,b))
def imported(label,path):
    d=json.loads(path.read_text())['validation']
    record(label,d['coverage'],d['angular_halfwidth_deg'])
def summarize(label,err,raw,fac):
    record(label,(tubes.scores(err,raw)<=fac).mean(0),np.rad2deg(raw[:,K[1:]]*fac[None,:,None]).mean(0))
for model,label in [('RC','RC-special'),('Chronos-2','C-special'),('RAFT','RAFT-base'),('VECTOR','VECTOR-base')]:
    imported(label,SPECIAL/'heads'/model/'special.json')
imported('C-native',SPECIAL/'original/Chronos-2.json')
for label,v in json.loads((OUT/'calibration.json').read_text())['validation'].items():
    record(label,v['coverage'],v['halfwidth_deg'])
for model in ('RAFT','VECTOR'):
    imported(model+'-refit',OUT/'route_heads/heads'/model/'special.json')
    with np.load(OUT/'route_heads/features'/model/'validation.npz') as d:
        head=pickle.loads((SPECIAL/'heads'/model/'special.pkl').read_bytes())
        raw=predict(head,d['common'],d['special'])
        summarize(model+'-transfer',d['error'],raw,head['factors'])
eng=Routed(); widths=[];errs=[];batch=[]
def capture(q,c,w):
    widths.append(w);return private_features(q,c,w)
def flush():
    if not batch:return
    h,p,t=zip(*batch,strict=True)
    with patch.object(st,'private_features',capture):rows=eng.batch(h,p)
    errs.extend(tubes.errors(c,tr) for (c,_),tr in zip(rows,t,strict=True))
    batch.clear()
for file in paths('validation'):
    a=pickle.loads((ARCHIVE/'validation'/(file.stem+'.pkl')).read_bytes())
    for (h,t),r in zip(records(file),a,strict=True):
        batch.append((h,r['package'],t))
        if len(batch)==8:flush()
flush();assert len(widths)==1200
summarize('CR-native',np.stack(errs),np.stack(widths),FACTORS['CR-native'])
res=pd.DataFrame(results,columns=['方案','期限s','双角前缀覆盖%','两轴平均半宽度'])
assert len(res)==14*7
res.to_csv(OUT/'误差管验证.csv',index=False)
log('14 臂已在相同 400 条验证路线、1,200 个起点上核对 0.5–10 秒双角前缀覆盖与半宽。')
