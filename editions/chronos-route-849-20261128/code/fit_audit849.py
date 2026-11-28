"""Recompute the 16 per-horizon calibration ranks directly from the stored independent anchors."""
from chronos_life_panel import *
def audit():
    fitids=list(range(1200,1216));calids=list(range(1300,1316));by={}
    fitphysics={get(OUT/f'E1/calibration/inputs/{k}.json')['physics'] for k in fitids}
    calphysics={get(OUT/f'E1/calibration/inputs/{k}.json')['physics'] for k in calids}
    assert len(fitphysics)==16 and len(calphysics)==16 and not fitphysics&calphysics
    anchor_keys=[]
    for a in LEARN:
        fp=OUT/f'E1/calibration/{a}.pkl';m=bundle(a);meta=get(fp.with_suffix('.json'));assert sha_file(fp)==meta['sha256']
        vals=[];keys=[]
        for k in calids:
            [(o,n,gap,q,width,y,prefix)]=pickle.loads((OUT/f'E1/calibration/inputs/{k}.pkl').read_bytes())
            x=features(a,width,o,n,gap,q);pres=raw(m['tree'],x[None])
            vals.append(scores(y[None],pres)[0]);keys.append([k,prefix,o.t])
        rank=int(np.ceil(.9*(len(keys)+1)))
        actual=np.maximum.accumulate(np.sort(vals,axis=0)[rank-1])
        assert rank==16 and keys==meta['anchors'] and np.array_equal(actual,m['final']) and np.array_equal(np.asarray(vals),meta['ratios'])
        anchor_keys.append(keys);by[a]=dict(training_prefixes=meta['training_prefixes'],rank=rank,score_max_abs_difference=0,factor_max_abs_difference=0,model_sha=meta['sha256'])
    assert all(v==anchor_keys[0] for v in anchor_keys)
    result=dict(status='PASS',fit_worlds=fitids,independent_calibration_worlds=calids,overlapping_physical_fingerprints=0,all_heads_same_anchors=True,anchors=anchor_keys[0],models=by)
    put(OUT/'audits/life_fit.json',result)
    audit_index=get(OUT/'audits.json');audit_index['life_fit']=dict(path='audits/life_fit.json',sha256=sha_file(OUT/'audits/life_fit.json'));audit_index['early_branch']=get(OUT/'audits/early_branch.json');put(OUT/'audits.json',audit_index)
if __name__=='__main__':audit()
