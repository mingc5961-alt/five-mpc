"""Independent deliverable check using raw per-world scores and endpoint receipts."""
import hashlib,json,time
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image
root=Path(__file__).resolve().parents[1]
def get(p):return json.loads(p.read_bytes())
def hashfile(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def run():
    while not (root/'report_receipt.json').exists():time.sleep(4)
    frozen=get(root/'design/frozen.json');out=get(root/'report_receipt.json');tab=pd.read_csv(root/'contrasts.csv')
    assert {w['world'] for w in frozen['worlds']}==set(range(6000,6048))
    for path,digest in out['artifacts'].items():assert hashfile(root/path)==digest,path
    for p in (root/'figures').glob('*.png'):
        with Image.open(p) as f:assert min(f.size)>400;f.verify()
    for p in (root/'figures').glob('*.pdf'):assert p.read_bytes().startswith(b'%PDF')
    d2=[];clock=get(root/'E2/calendar_849.json');expected={(k,a,s,m) for k,a,s,m in frozen['base_endpoints']+frozen['extra_endpoints']}
    got=set()
    for p in sorted((root/'E2/receipts/confirmation').glob('*.json')):
        r=get(p);z=r['result'];got.add((r['world'],r['arm'],r['seed'],r['scale']))
        assert len(r['events'])==849 and [e[1:] for e in r['events']]==clock and [e[0] for e in r['events'][:16]]==list(range(16))
        assert abs(z['ledger_s'][0]-2.6319)<1e-7 and abs(sum(z['ledger_s'])-180)<1e-6 and z['probes_per_user']==9000
        g=sum(z['bits'])/180e6;assert abs(g-z['G_mbps'])<1e-6
        if r['scale']==.75:d2.append([r['world'],r['arm'],g,100*z['R']])
    assert got==expected
    d2=pd.DataFrame(d2,columns=['world','arm','G','R']).groupby(['world','arm'])[['G','R']].mean()
    rows=[];queries=0;peak=0.
    for w in frozen['worlds']:
        k=w['world'];p=root/f'E1/scores/confirmation/{k}.csv.gz';r=get(p.with_name(f'{k}.json'))
        assert hashfile(p)==r['sha256'] and len(set(r['row_digest_by_arm'].values()))==1 and r['mismatches']==0
        q=pd.read_csv(p,usecols=['scale','n','age','h','label','arm','true_life','hold','U','O'])
        l=np.maximum(q.true_life-q.age,0);a=np.maximum(q.hold-q.age,0);eu=np.maximum(l-a,0);eo=np.maximum(a-l,0)
        peak=max(peak,float(np.max(np.abs(q.U-eu))),float(np.max(np.abs(q.O-eo))));assert peak<1e-10
        q=q.assign(U=eu,O=eo)
        selected=q[(q.scale==.75)&q.n.isin([6,8,12,24,40])&q.age.isin([2,3,5])&(q.h==0)&(q.label=='same_direction_revisit')]
        queries+=len(selected[selected.arm=='TIME'])
        for arm,v in selected.groupby('arm'):
            cells=v.groupby(['n','age'])[['U','O']].mean().mean();rows.append((k,arm,cells.U,cells.O))
    d1=pd.DataFrame(rows,columns=['world','arm','U','O']).set_index(['world','arm'])
    deltas=[];verified={}
    def check(family,label,metric,values):
        row=tab[(tab.family==family)&(tab.comparison==label)&(tab.metric==metric)].iloc[0]
        v=np.asarray(values,float);assert len(v)==row.worlds
        ix=np.random.default_rng(271127).integers(0,len(v),size=(20000,len(v)))
        lo,hi=np.quantile(v[ix].mean(axis=1),[.025,.975]);real=np.array([v.mean(),lo,hi]);reported=np.array([row['mean'],row.lo95,row.hi95])
        err=float(np.max(abs(real-reported)));deltas.append(err);assert err<1e-8,(family,label,metric,err)
        verified[f'{family}:{label}:{metric}']=real.tolist()
    for a in sorted(set(('TIME','SHUFFLE',frozen['strong_E1']))):
        for tag,var,first,last in [('missed_s_saved','U',a,'ROUTE'),('extra_overadmission_s','O','ROUTE',a)]:
            pivot=d1[var].unstack();check('E1',a,tag,pivot[first]-pivot[last])
    for label,first,last in [('VALUE-time','VALUE-route','VALUE-time'),('EDF-route','VALUE-route','EDF-route'),('nonroute_vs_AoI',frozen['strong_nonroute_scheduling'],frozen['strong_AoI'])]:
        for tag,var in [('delta_G_Mbps','G'),('delta_R_pp','R')]:
            pivot=d2[var].unstack();check('E2',label,tag,pivot[first]-pivot[last])
    check('E2-absolute','VALUE-route','R_pct',d2.R.unstack()['VALUE-route'])
    audit=get(root/'audits.json');assert all(get(root/f'audits/confirmation/{k}.json')['status']=='PASS' and hashfile(root/f'audits/confirmation/{k}.json')==audit['confirmation'][str(k)] for k in range(6000,6048))
    result=dict(status='PASS',worlds=48,E2_endpoints=len(got),E2_actual_clock_pairs=849*len(got),E1_primary_unique_queries=queries,max_recomputed_lifetime_loss_difference=peak,max_recomputed_bootstrap_difference=max(deltas),paired_results=verified)
    tmp=root/'audits/delivery_validation.tmp';tmp.write_text(json.dumps(result,indent=2));tmp.replace(root/'audits/delivery_validation.json');print(json.dumps(result),flush=True)
if __name__=='__main__':run()
