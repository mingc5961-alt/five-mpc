from core import ROOT
import json,pandas as pd
OLD=ROOT.parent/'微调Chronos路线对照'
def historical(new):
    labels={'C-special':'Chronos-2','C-native':'Chronos-native','RC-special':'RC','VECTOR-base':'VECTOR','RAFT-base':'RAFT'}
    rows=[]
    for folder in ('cells','other_cells'):
        for f in (OLD/folder).glob('*.json'):
            v=json.loads(f.read_text());name=labels.get(v['arm'])
            if not name:continue
            assert v['mcs_margin_db']==0 and v['seed']==7
            r=v['result'];rows.append((v['world'],v['budget'],name,r['G_mbps'],100*r['R'],r['hold_mean_s']))
    old=pd.DataFrame(rows,columns=['world','budget','arm','G','risk','hold'])
    assert len(old)==150 and set(old.world)==set(range(10))
    keys=['G','risk','hold'];a=old.groupby(['budget','arm'])[keys].mean()
    b=new[new.arm.isin(labels.values())].groupby(['budget','arm'])[keys].mean()
    return pd.concat([a.add_suffix('_old'),b.add_suffix('_new'),(b-a).add_suffix('_difference')],axis=1).reset_index()
