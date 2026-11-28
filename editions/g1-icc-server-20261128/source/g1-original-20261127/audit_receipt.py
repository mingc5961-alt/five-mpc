"""Complete field-by-field comparison of historical or development receipts."""
import argparse,hashlib,json,math
from pathlib import Path

def audit(expected,actual):
    ref=json.loads(expected.read_bytes());obs=json.loads(actual.read_bytes())
    if not all(k in obs and k in ref for k in ('events','result','world')):raise ValueError('incomplete receipt')
    event_diff=[];max_dt=0.;user_diff=0;exact=0;within_us=0
    for i,(old,new) in enumerate(zip(ref['events'],obs['events'])):
        du=int(old[0])!=int(new[0]);dt=abs(float(old[1])-float(new[1]))
        max_dt=max(max_dt,dt);user_diff+=du;exact+=(not du and not dt);within_us+=(not du and dt<=1e-6)
        if du or dt>1e-6:event_diff.append(dict(index=i+1,expected=old,observed=new))
    fields={}
    for key in sorted(set(ref['result'])|set(obs['result'])):
        if key not in ref['result'] or key not in obs['result']:raise ValueError('missing ledger field '+key)
        a,b=ref['result'][key],obs['result'][key]
        if isinstance(a,list) != isinstance(b,list):raise ValueError(key)
        aa=a if isinstance(a,list) else [a];bb=b if isinstance(b,list) else [b]
        if len(aa)!=len(bb) or not all(math.isfinite(x) for x in aa+bb):raise ValueError(key)
        deltas=[y-x for x,y in zip(aa,bb)]
        fields[key]=dict(expected=a,actual=b,delta=deltas if isinstance(a,list) else deltas[0],
            absmax=max(map(abs,deltas)),relative_max=max(abs(d)/max(1e-30,abs(x)) for x,d in zip(aa,deltas)))
    s=obs['result']
    invariants=dict(users=len(s['bits']),probes_per_user=s['probes_per_user'],
        duration_balance_s=sum(s['ledger_s'])-180,charged_balance_s=s['ledger_s'][0]-.0031*len(obs['events']),
        bit_balance_Mbps=sum(s['bits'])/180e6-s['G_mbps'])
    return dict(expected_sha256=hashlib.sha256(expected.read_bytes()).hexdigest(),actual_sha256=hashlib.sha256(actual.read_bytes()).hexdigest(),
       expected_world=ref['world'],actual_world=obs['world'],physical_match=ref['world']==obs['world'],
       expected_events=len(ref['events']),actual_events=len(obs['events']),exact_event_pairs=exact,
       matching_event_pairs_1us=within_us,different_users=user_diff,max_completion_time_diff_s=max_dt,
       mismatching_events_1us=event_diff,fields=fields,invariants=invariants)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('expected',type=Path);p.add_argument('actual',type=Path);p.add_argument('out',type=Path)
    args=p.parse_args();out=audit(args.expected,args.actual)
    args.out.write_text(json.dumps(out,indent=2,allow_nan=False))
    print('physical_match',out['physical_match'],'event pairs at 1µs',out['matching_event_pairs_1us'],'/',out['expected_events'])
