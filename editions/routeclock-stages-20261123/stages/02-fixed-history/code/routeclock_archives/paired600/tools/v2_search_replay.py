"""Replay recorded joint decisions with search instrumentation and verify the selected actions."""
import argparse
import json
from pathlib import Path
import pandas as pd
from pilot_histories import configuration
from routeclock.scheduler import Action,choose_actions


def replay(folder):
    read=lambda name:json.loads((folder/name).read_text())
    packages={(p['uav'],p['origin']):p for p in read('predictions.json')}
    cfg=configuration()['simulation']
    metrics=read('metrics.json')
    if metrics['scheduler']!='value' or metrics['action_mode']!='joint':
        raise ValueError('Expected value scheduler and joint width/interval decisions')
    rows=[]
    for frame in read('choices.json'):
        actions=frame['choices']
        fronts=[]
        for uid,action in enumerate(actions):
            candidates=packages[(uid,action['origin'])]['frontier']
            fronts.append([Action(tuple(c['pair']),c['interval'],c['rate'],c['loss_db']) for c in candidates])
        trace={}
        choices=choose_actions(fronts,cfg['observation_cost_s'],metrics['sensing_budget'],trace=trace)
        for choice,action in zip(choices,actions,strict=True):
            observed=(action['pair'],action['interval'])
            expected=(list(choice.pair),choice.interval) if choice else (None,0)
            if observed!=expected:raise ValueError(f'Decision changed at {frame["t"]}: {observed} vs {expected}')
        rows.append(dict(t=frame['t'],n=len(choices),raw=sum(trace['raw']),pareto=sum(trace['pareto']),
                         vehicles_with_multiple_widths=sum(v>1 for v in trace['available_widths']),
                         budget_width_moves=sum(a!=b for _,a,b in trace['budget_moves']),
                         tested_width_changes=trace['tested_width_changes'],accepted_width_changes=trace['accepted_width_changes']))
    return pd.DataFrame(rows)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('folder',type=Path)
    p.add_argument('output',type=Path)
    args=p.parse_args()
    replay(args.folder).to_csv(args.output,index=False)
