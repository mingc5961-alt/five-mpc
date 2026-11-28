"""Audit the complete 8-world x 4-scale original N01 grid and freeze K and PM."""
import argparse,csv,hashlib,json,math,statistics,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'g1-server-handoff/tools'))
from check_g1_receipts import counts_at_period

SCALES=(.45,.60,.75,1.)
WORLDS=range(1400,1408)


def main(index,out):
    jobs=json.loads(index.read_bytes())
    pairs={(j['world'],float(j['scale'])):Path(j['receipt']) for j in jobs}
    if len(jobs)!=32 or set(pairs)!={(k,m) for k in WORLDS for m in SCALES}:
        raise ValueError('require exactly 32 unique N01 coordinates')
    summaries=[];physical={};by={m:[] for m in SCALES}
    for (k,m),p in sorted(pairs.items()):
        r=json.loads(p.read_bytes());v=r['result'];e=r['events'];s=v['ledger_s'][0]
        if int(r['k'])!=k or float(r['m'])!=m or r['arm']!='G1' or r['world']!=physical.setdefault(k,r['world']):
            raise ValueError(f'incorrect identity or mixed physical world for {p}')
        if len(e)<16 or [t[0] for t in e[:16]]!=list(range(16)) or not all(a[1]<b[1] for a,b in zip(e,e[1:])):
            raise ValueError(f'event order for {p}')
        if (abs(sum(v['ledger_s'])-180)>1e-6 or abs(len(e)*.0031-s)>1e-7 or s>3.0000001
            or v['probes_per_user']!=9000 or len(v['bits'])!=16 or abs(sum(v['bits'])/180e6-v['G_mbps'])>1e-6):
            raise ValueError(f'unbalanced ledger for {p}')
        row=dict(world=k,scale=m,physical_sha256=r['world'],events=len(e),S_s=s,
                 G_mbps=v['G_mbps'],R_pct=100*v['R'],W_s=v['narrow_good_data_s'],
                 receipt=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
        summaries.append(row);by[m].append(row)
    if len(set(physical.values()))!=8:raise ValueError('development worlds not distinct')
    points={m:{key:statistics.mean(r[key] for r in rows) for key in ('G_mbps','R_pct','S_s','events')} for m,rows in by.items()}
    options=[]
    for a in SCALES:
        x=points[a]
        if x['R_pct']<=7:options.append((x['G_mbps'],-x['R_pct'],1,{a:1.}))
        for b in SCALES:
            y=points[b]
            if a>=b or (x['R_pct']-7)*(y['R_pct']-7)>=0:continue
            prob=(7-y['R_pct'])/(x['R_pct']-y['R_pct'])
            options.append((prob*x['G_mbps']+(1-prob)*y['G_mbps'],-7.,0,{a:prob,b:1-prob}))
    out.mkdir(parents=True,exist_ok=True)
    with (out/'all_32.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=summaries[0]);writer.writeheader();writer.writerows(summaries)
    if not options:
        (out/'development_points.json').write_text(json.dumps(points,indent=2))
        raise ValueError('7% target unreachable in base grid; consult START_SERVER and development extension rule')
    g,r,_,mix=max(options,key=lambda c:c[:3])
    target=sum(p*points[m]['events'] for m,p in mix.items())
    quota=math.floor(target+.5)
    grid=range(3000,6001);cost={p:abs(sum(counts_at_period(p))-target) for p in grid}
    best_cost=min(cost.values())
    minima=[p for p in grid if cost[p]==best_cost]
    ms=statistics.median_low(minima)
    freeze=dict(input_worlds=list(WORLDS),input_multipliers=SCALES,development_only=True,
        protocol='N01 original G1, native clock B=3 s',world_sha256=physical,points=points,
        target_risk_pct=7,mixture=mix,expected_goodput_mbps=g,expected_risk_pct=-r,
        expected_fixes=target,expected_sense_s=target*.0031,K=quota,exact_quota_cost_s=quota*.0031,
        PM_ms=ms,period_fixes=sum(counts_at_period(ms)),period_cost_s=sum(counts_at_period(ms))*.0031,
        tie_rule='best expected G, lower risk then one endpoint; nearest count rounded half-up; median_low of best integer-ms periods 3000..6000',
        period_tied_ms=minima,grid_csv_sha256=hashlib.sha256((out/'all_32.csv').read_bytes()).hexdigest())
    temp=out/'freeze.tmp';temp.write_text(json.dumps(freeze,indent=2));temp.replace(out/'freeze.json')
    return freeze

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('index',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(main(a.index,a.output),indent=2))
