"""Fit independent uncertainty heads and measure their executed service policies."""
import csv
import json
import pickle
import sys
import time
from long_history_data import ROOT, OUT, CENTERS, KINDS, HORIZONS, configuration
import numpy as np
from routeclock.validity import ScaleHead, GlobalEnvelope, QUERY_H, interval_errors, feature_rows
from routeclock.geometry import angles, angle_error
from routeclock.radio import array_gain
from routeclock.simulator import simulate


def load(path):
    with path.open('rb') as handle:
        return pickle.load(handle)


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def errors(anchor, center):
    return interval_errors(anchor['package'], center, HORIZONS, anchor['truth'])


def fit(train, calibration, center, kind):
    if kind == 'envelope':
        return GlobalEnvelope().fit(train, center).calibrate(calibration, center)
    x = np.concatenate([feature_rows(a['features'][center], a['package'].hs) for a in train])
    y = np.concatenate([errors(a, center).ravel() for a in train])
    head = ScaleHead(kind, device='cpu', seed=7, params={'num_leaves':15,'min_child_samples':20})
    head.fit(x, y)
    return head.calibrate(calibration, center)


def interpolated(center_points, times):
    return np.stack([np.interp(times, np.arange(21)*.5, center_points[:,j]) for j in range(3)],axis=-1)


def gain_loss(pointing, actual, pair):
    return -10*np.log10(array_gain(pointing, actual, pair)/np.prod(pair))


def evaluate(directory, world, anchors, center, head, mission):
    read = lambda name: json.loads((directory/name).read_text(encoding='utf-8'))
    metrics, events, forecasts, choices = [read(name+'.json') for name in ['metrics','events','predictions','choices']]
    warm, end = 180., 198.
    packages = {(p['uav'],p['origin']):p for p in forecasts}
    matched = []
    for uid,a in enumerate(anchors):
        pkg = a['package']
        prediction = interpolated(pkg.centers[center], HORIZONS)
        angular = angle_error(angles(prediction), angles(a['truth']))
        forecast = packages[(uid,pkg.origin)]
        envelope = np.array(forecast['envelope_rad'])
        narrow = max((v['interval'] for v in forecast['frontier'] if v['pair']==[32,32]),default=0.)
        actual_loss = gain_loss(angles(prediction), angles(a['truth']), (32,32))
        matched.append(dict(mission=mission,uid=uid,behavior=a['group'],center=center,head=head,
                            ADE_m=np.linalg.norm(prediction-a['truth'],axis=1).mean(),
                            angle_deg=np.sqrt(np.mean(np.sum(np.rad2deg(angular)**2,axis=1))),
                            covered=int(np.all(errors(a,center)<=envelope[(QUERY_H*2).astype(int)])),
                            width_at_2s_deg=np.rad2deg(envelope[4]).tolist(),narrow_cert_s=narrow,
                            unsafe_narrow_certificate=int(narrow>0 and np.any(actual_loss[HORIZONS<=narrow]>3))))
    intervals = []
    for uid in range(world.n):
        current = None
        for frame in choices:
            choice = frame['choices'][uid]
            key = (choice['origin'], tuple(choice['pair'] or ()))
            if current and current['key']==key:
                continue
            if current:
                current['finish']=frame['t']
                intervals.append(current)
            current=dict(uid=uid,key=key,start=frame['t'])
        current['finish']=end
        intervals.append(current)
    spans=[]
    for seg in intervals:
        uid=seg['uid'];origin,pair=seg['key'];start,finish=seg['start'],seg['finish']
        if not pair or finish<=start:
            continue
        age=np.r_[np.arange(start,finish,.02),finish]-origin
        p=packages[(uid,origin)]
        pos=interpolated(np.array(p['center']),age)
        truth=np.array([world.truth(uid,origin+h)[0] for h in age])
        loss=gain_loss(angles(pos),angles(truth),pair)
        spans.append(dict(uid=uid,origin=origin,start=start,finish=finish,seconds=finish-start,
                          pair=str(pair),narrow=int(pair==(32,32)),failed=int(max(loss)>3)))
    delivered=bad_time=valid_time=0.
    for e in events:
        if e['type']!='data':
            continue
        delivered+=e['delivered_bits']
        if 'pair' not in e:
            continue
        duration=e['data_s']; valid_time+=duration
        point=np.array(e['pointing']);actual=point+e['actual_angle_error']
        if gain_loss(point,actual,e['pair'])>3:
            bad_time+=duration
    np.testing.assert_allclose(delivered/18e6,metrics['net_mbps'],rtol=1e-9)
    np.testing.assert_allclose(sum(metrics[k] for k in ['sensing_s','switch_s','data_s']),18,atol=1e-8)
    narrow_spans=[s for s in spans if s['narrow']]
    unavailable=18*metrics['unavailable_service_fraction']
    return matched, dict(mission=mission,center=center,head=head,net_mbps=metrics['net_mbps'],
        senses=metrics['observations_service'],sensing_pct=100*metrics['sensing_fraction'],
        certified_narrow_s=np.mean([a['narrow_cert_s'] for a in matched]),
        narrow_s=sum(s['seconds'] for s in narrow_spans),narrow_runs=len(narrow_spans),
        narrow_share_pct=100*sum(s['seconds'] for s in narrow_spans)/(18*world.n),
        held_failed=sum(s['failed'] for s in spans),held_total=len(spans),
        narrow_failed=sum(s['failed'] for s in narrow_spans),
        data_bad_s=bad_time,data_valid_s=valid_time,unavailable_s=unavailable,
        outage_pct=100*metrics['outage_data_fraction'],switch_s=metrics['switch_s'],
        conditional_mbps=delivered/max(valid_time,1e-12)/1e6)


def main():
    import torch
    torch.set_num_threads(3)
    counts={'train':24,'cal':24,'network':4}
    while not all((OUT/f'{k}_{i:02}.pkl').exists() for k,n in counts.items() for i in range(n)):
        time.sleep(3)
    splits={split:[r for file in sorted(OUT.glob(f'{split}_*.pkl')) for r in load(file)[0]] for split in counts}
    hashes=[a['route_hash'] for split in counts for a in splits[split]]
    assert len(hashes)==len(set(hashes))==60
    loop_rows=[];anchor_rows=[]
    for center in CENTERS:
        for kind in KINDS:
            model_file=OUT/f'head_{center}_{kind}.pkl'
            if model_file.exists():
                fitted=load(model_file)
            else:
                fitted=fit(splits['train'],splits['cal'],center,kind)
                with model_file.open('wb') as f:
                    pickle.dump(fitted,f)
            bundle={'kind':kind,'seed':7,'heads':{center:fitted}}
            for mission in range(4):
                directory=OUT/f'loop_{center}_{kind}_{mission}'
                anchors,trackers,world=load(OUT/f'network_{mission:02}.pkl')
                if not (directory/'choices.json').exists():
                    simulate(configuration(),world,bundle,method=center,out=directory,service_trackers=trackers)
                sample, network=evaluate(directory,world,anchors,center,kind,mission)
                anchor_rows.extend(sample);loop_rows.append(network)
                print(center,kind,mission,round(network['net_mbps'],2),network['senses'],flush=True)
                write_rows(OUT/'closed_loop.csv',loop_rows)
                write_rows(OUT/'common_anchors.csv',anchor_rows)


if __name__=='__main__':
    main()
