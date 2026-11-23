"""Audit saved FIVE case logs without feeding realized errors back to the policy."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from routeclock.config import load_config, dump_json
from routeclock.geometry import angles, angle_error
from routeclock.radio import Radio, array_gain
from routeclock.scheduler import Action, choose_actions


def replay(folder, stride):
    read = lambda name: json.loads((folder / (name + '.json')).read_text())
    metrics = read('metrics')
    cfg = load_config()
    radio = Radio(cfg)
    cost, budget = cfg['simulation']['observation_cost_s'], metrics['sensing_budget']
    packages = {(p['uav'], p['origin']): p for p in read('predictions')}
    frames = read('choices')
    events = read('events')
    observations=[e for e in events if e['type']=='observation']
    costs={e['cost'] for e in observations}
    if len(costs)!=1: raise ValueError('Replay expects a fixed service observation cost')
    cost=costs.pop()
    frame_times=np.array([f['t'] for f in frames])
    backlog=token_wait=0.
    waits=[]
    for previous,current in zip(observations,observations[1:]):
        if previous['finish']<metrics['warmup_s']:continue
        index=np.searchsorted(frame_times,current['start']+1e-10,side='right')-1
        if index<0:continue
        choice=frames[index]['choices'][current['uav']]
        due=choice['origin']+choice['interval']
        decision=previous['finish']
        backlog+=max(0,decision-due)
        token_wait+=max(0,previous['start']+cost/budget-max(due,decision))
        waits.append(max(0,current['start']-due))
    over = changed = tested = no_action = 0
    for index, frame in enumerate(frames):
        choices = frame['choices']
        no_action += any(not a['pair'] for a in choices)
        over += sum(cost/a['interval'] if a['pair'] else 1 for a in choices) > budget + 1e-12
        if index % stride:
            continue
        fronts = []
        for uid, selected in enumerate(choices):
            p = packages[uid, selected['origin']]
            age = frame['t'] - p['origin']
            hs = np.array(p['h'])
            pos = np.array(p['center'])
            pointing = angles(pos)
            rates = {}
            actions = []
            for a in p['frontier']:
                pair, delta = tuple(a['pair']), a['interval']
                if age > delta:
                    continue
                ratio=(p.get('deadline_ratios') or {}).get(str(float(delta)),1.)
                key=(pair,ratio)
                if key not in rates:
                    loss = [radio.worst_loss(d, e, pair) for d, e in zip(pointing, ratio*np.array(p['envelope_rad']))]
                    snr = radio.snr(pos, pointing, pair) - np.minimum(loss, 40)
                    rates[key] = [radio.goodput(s, radio.mcs(s))[0] for s in snr]
                knots = np.r_[age, hs[(hs > age) & (hs <= delta)]]
                remaining = np.interp(knots, hs, rates[key])
                rate = np.sum(np.diff(knots) * (remaining[:-1]+remaining[1:])/2)/(delta-age) if delta > age else remaining[0]
                actions.append(Action(pair, delta, rate, a['loss_db']))
            fronts.append(actions)
        proposal = choose_actions(fronts, cost, budget)
        for a, s in zip(proposal, choices):
            changed += (list(a.pair) if a else None, a.interval if a else 0) != (s['pair'], s['interval'])
            tested += 1

    rows = []
    for e in events:
        if 'prediction_age' not in e:
            continue
        age = e['prediction_age']
        p = packages[e['uav'], e['prediction_origin']]
        pair = tuple(e['pair'])
        target = np.array(e['pointing']) + e['actual_angle_error']
        residual = []
        for expert in ('I','P'):
            xyz = np.array(p['experts'][expert])
            pred = [np.interp(age, p['h'], axis) for axis in xyz.T]
            residual.extend(angle_error(angles(pred),target))
        env = [np.interp(age,p['h'],axis) for axis in np.array(p['envelope_rad']).T]
        ratio = p.get('deadline_ratios') or {}
        delta = round(e['deadline']-p['origin'],8)
        env = np.asarray(env)*ratio.get(str(float(delta)),1.)
        conservative = radio.mcs(e['predicted_snr_db'] - min(40,radio.worst_loss(e['pointing'],env,pair)))
        locked_rate,_=radio.goodput(e['snr_db'],conservative)
        certificate = max((a['interval'] for a in p['frontier'] if tuple(a['pair']) == pair),default=0)
        actual_loss = -10*np.log10(array_gain(e['pointing'],target,pair)/np.prod(pair))
        rows.append([e['data_s'],age,*residual,conservative != e['mcs'],
                     age > e['deadline']-p['origin'], age > certificate, actual_loss > 3,locked_rate-e['delivered_bits']/e['data_s']])
    data = np.array(rows)
    corr = []
    for lo, hi in ((0,1),(1,2),(2,10)):
        z = data[(data[:,1]>lo)&(data[:,1]<=hi)]
        if len(z)<2:
            continue
        axes = [np.cov(z[:,[2+j,4+j]].T,aweights=z[:,0],ddof=0) for j in (0,1)]
        corr.append(dict(age_s=[lo,hi],slots=len(z),axis_corr=[v[0,1]/np.sqrt(v[0,0]*v[1,1]) for v in axes]))
    summary = np.average(data[:,6:],axis=0,weights=data[:,0])
    return dict(task=metrics['task'],method=metrics['method'],frames=len(frames),over_budget_frames=int(over),
                frames_missing_action=int(no_action),aged_stride=stride,aged_changed=int(changed),aged_decisions=tested,
                mcs_disagreement_share=summary[0],past_requested_share=summary[1],
                past_certified_grid_share=summary[2],actual_over3_data_share=summary[3],
                locked_mcs_delta_net_mbps=summary[4]*data[:,0].sum()/metrics['service_s']/1e6,
                horizon_exhausted_holds=metrics['expired_segments'],unavailable_s=metrics['unavailable_s'],
                overdue_requests=metrics['overdue_requests'],wait_count=len(waits),
                mean_late_ms=1000*np.mean(waits),p95_late_ms=1000*np.quantile(waits,.95),
                mean_already_late_at_replan_ms=1000*backlog/len(waits),
                mean_additional_token_wait_ms=1000*token_wait/len(waits),expert_correlation=corr)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('folder',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--stride',type=int,default=100)
    args = p.parse_args()
    if args.stride<1: p.error('stride must be positive')
    dump_json(args.out,replay(args.folder,args.stride))
