"""Independent accounting and split checks from saved raw scores and issue-time inputs."""
from experiment import OUT,load,errors
import numpy as np
import pandas as pd
import json
def main():
    df=load((OUT/'routes').glob('target_test_*.csv')).set_index('qid',drop=False)
    pred=pd.read_csv(OUT/'query_predictions.csv')
    assert len(df)==1008 and df.route.nunique()==48
    pairs=pd.read_csv(OUT/'matched_cases.csv')
    for split,g in pairs.groupby('dataset'):
        assert g.route.is_unique and len(g)==2*g.pair.nunique()
        for _,p in g.groupby('pair'):
            win=p[p.side=='benefit'].iloc[0];lose=p[p.side=='harm'].iloc[0]
            assert win.route!=lose.route and win.step==lose.step and win.h==lose.h
            assert max(win.support,lose.support)<.5 and abs(win.support-lose.support)<=.05+1e-12
            assert win.gain_deg2>=1 and lose.gain_deg2<=-4
    account=pd.read_csv(OUT/'pairwise_accounts.csv')
    max_residual=0.
    for model,g in pred.groupby('model'):
        d=df.loc[g.qid];assert len(g)==len(df) and g.qid.is_unique
        e,_=errors(d,g.weight.to_numpy());np.testing.assert_allclose(e,g.deg2,atol=1e-8)
        if model.startswith(('switch','zero','pair_classifier')):
            assert np.allclose(g.loc[d.support.to_numpy()>=.5,'weight'],.89)
        for ref in ['fixed89','route','current48']:
            c=account[(account.model==model)&(account.baseline==ref)].set_index('subset')
            remainder=abs(c.loc['all','net']-c.loc['switched','net']-c.loc['remaining','net'])
            max_residual=max(max_residual,remainder,float(np.abs(c.saved-c.lost-c.net).max()))
            assert max_residual<1e-6
    info=dict(test_routes=48,test_queries=len(df),policies=pred.model.nunique(),max_account_residual=max_residual,
              independent_train_pairs=int((pairs.dataset=='train_LOPO').sum()/2),
              independent_test_pairs=int((pairs.dataset=='fresh_test').sum()/2))
    (OUT/'verification.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    print(info)
if __name__=='__main__':main()

