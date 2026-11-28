"""Match an entirely random fleet to an all-planned fleet of the same source mix."""
import json
import re
from collections import Counter
import numpy as np
from field_random_geometry import recurrence
from field_worlds import OUT
from history import put

def group_key(r):
    if r['source']=='neurobem' and r['label']=='random':
        return re.search(r'random points (\d+)',r['task']).group(1)
    return r['id']

def choose():
    counts={'amov':10,'neurobem':6}
    diagnostic=json.loads((OUT/'recurrence_and_angles.json').read_bytes())
    full={(r['source'],r['flight']):r['repeat_score'] for r in diagnostic['tracks']}
    tracks={s:json.loads((OUT/(s+'_inputs.json')).read_bytes())['tracks'] for s in counts}
    rng=np.random.default_rng(271130)
    for duration in (180,120,90,60,45,30):
        groups={}; notes={}
        for s,n in counts.items():
            for label in ('random','planned'):
                candidates=[r for r in tracks[s] if r['label']==label and r['seconds']>=duration]
                if label=='random':
                    candidates=[r for r in candidates if full[(s,r['id'])] is not None and full[(s,r['id'])]<.70]
                picked=[];keys=set()
                for i in rng.permutation(len(candidates)):
                    r=candidates[i];key=group_key(r)
                    if key in keys:continue
                    if label=='random':
                        with np.load(r['track']) as d:t,p=d['t'],d['p']
                        offset=(t[-1]-duration)/2;use=(t>=offset)&(t<=offset+duration)
                        score=recurrence(t[use],p[use])
                        if score is None or score>=.70:continue
                        notes[s+'_'+r['id']]=dict(full=full[(s,r['id'])],window=score)
                    keys.add(key);picked.append(r)
                    if len(picked)==n:break
                groups[s,label]=picked
        if all(len(group)==counts[s] for (s,label),group in groups.items()):break
    else:raise ValueError('insufficient labeled, non-repeating tracks')
    order=rng.permutation(16);panels=[]
    for view,label in [('random','random'),('planned','planned')]:
        rows=[r for s in counts for r in groups[s,label]];rows=[rows[i] for i in order]
        assert Counter(r['source'] for r in rows)==Counter(counts)
        panels.append(dict(id='combined_0_'+view, source='combined',view=view,cohort=0,duration=duration,
                           quota=round(849*duration/180),noise_seed=27113000,flights=rows,station_offset_m=[300.,0.,0.]))
    put(OUT/'combined_inputs.json',dict(source='combined',panels=panels,random_geometry=notes,
        source_counts=counts,rule='Seed 271130. Longest supported horizon in 180/120/90/60/45/30 s. Exclude strong repeats in both complete random track and service window. Distinct waypoint programs within fleet. Baseline uses source-designated routes. This sensitivity analysis reuses flights from the source panels.'))

if __name__=='__main__':choose()
