"""Equal-cost locally paired deterministic rows and lower/higher acquisition budgets."""
from concurrent.futures import ProcessPoolExecutor,as_completed
import hashlib,json,multiprocessing,os,sys,time
from history import ROOT,put
from paired_grid import BASE,SCALES,load_checked
from paired_models import Predictor,OUT as MODELS
from physical_inputs import sha_file
from policy_env import initialize
from slots import compute_slot
OUT=BASE/'deterministic_matrix'
KINDS={'P01':('Acal','fixed_due'),'P03':('Acal','oldest'),'Q01':('Acal','due'),
       'Q02':('M','due'),'Q05':('Fstar','due')}
def budgets():
    primary=json.loads((BASE/'summary.json').read_bytes())['K']
    with (BASE/'native_matrix/budget_freeze.json').open() as stream:scan=json.load(stream)['rows']
    return [(3,primary),(3,700),(3,950)]+[(b,scan[f'N01_B{b:g}']['K']) for b in (4.5,9)]
def jobs():
    out=[]
    for idx,(b,kq) in enumerate(budgets()):
        rows=list(KINDS) if idx==0 else ['Q01']
        out.extend((k,r,b,kq,m) for r in rows for k in range(1400,1408) for m in SCALES)
    return list(dict.fromkeys(out))
def key(j):
    k,r,b,q,m=j;return f'{r}_{k}_B{b:g}_K{q}_m{m:.2f}'
def work(j):
    import paid_full as pf
    from closed import simulate_controls
    k,row,b,q,m=j;tag=key(j);dst=OUT/'receipts'/(tag+'.json')
    if dst.is_file():
        record=json.loads(dst.read_bytes())
        if record['key']!=tag or record['quota']!=len(record['events']):raise ValueError(tag)
        return tag
    with compute_slot():
        mark=OUT/'active'/dst.name;put(mark,dict(pid=os.getpid(),started=time.time()))
        try:
            w,meta=load_checked('development',k);w.cfg['simulation']['sensing_budget']=b/180
            kind,clock=KINDS[row];model=Predictor(w,kind)
            es,r,forced=simulate_controls(w,clock,model.predict,model.factors,pf,b,m,q)
            if len(es)!=q or abs(r['ledger_s'][0]-q*.0031)>1e-7:raise ValueError('cost')
            eh=hashlib.sha256(json.dumps([(u,round(t*1e6)) for u,t in es],separators=(',',':')).encode()).hexdigest()
            put(dst,dict(key=tag,protocol='g1_linux_paired_v2',world=meta['physical_sha256'],world_id=k,scale=m,
                row=row,clock=clock,model=kind,model_sha256=sha_file(MODELS/f'{kind}.pkl'),budget_s=b,quota=q,events=es,
                event_sha256=eh,forced=forced,result=r,elapsed_s=time.time()-json.loads(mark.read_bytes())['started']))
        finally:mark.unlink(missing_ok=True)
    return tag
def main():
    initialize()
    from closed import freeze
    for s in ('active','receipts'):(OUT/s).mkdir(parents=True,exist_ok=True)
    start=time.time();todo=jobs()
    put(OUT/'batch.json',dict(total=len(todo),workers=30,pid=os.getpid(),start=start,phase='development'))
    with ProcessPoolExecutor(30,mp_context=multiprocessing.get_context('fork')) as p:
        for f in as_completed([p.submit(work,j) for j in todo]):f.result()
    policies={}
    for row,b,q in dict.fromkeys((r,b,q) for _,r,b,q,_ in todo):
        by={m:[json.loads((OUT/'receipts'/(key((k,row,b,q,m))+'.json')).read_bytes()) for k in range(1400,1408)] for m in SCALES}
        policies[f'{row}_B{b:g}_K{q}']=freeze(by)
        if row=='P03':
            for i in range(8):
                if len({by[m][i]['event_sha256'] for m in SCALES})!=1:raise ValueError('oldest stream differs across scales')
    put(OUT/'summary.json',dict(total=len(todo),policies=policies,wall_s=time.time()-start))
    put(OUT/'batch.json',dict(total=len(todo),workers=30,pid=os.getpid(),start=start,phase='done'))
if __name__=='__main__':main()
