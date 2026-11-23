"""Verify alternative batched evaluation against a completed C3 replay."""
import json
import pickle
import time
from pathlib import Path
from unittest.mock import patch
import numpy as np
from parallel_frontier import frontier
from run_flights import MeasuredFlight,WORK,ROOT,warm,CausalModel,load_config,BUDGET,Radio,network,controller

def main():
    win=json.loads((WORK/'flight_windows.json').read_text())[0]
    bundle=pickle.loads((WORK/'baseline_cache/classical.pkl').read_bytes())['I']
    checks=[]
    for budget in ('T','M','L'):
        cfg=load_config();cfg['simulation'].update(primary_N=1,sensing_budget=BUDGET[budget]/16)
        radio=Radio(cfg);radio.threshold-=5
        world=MeasuredFlight(cfg,win,0)
        net=CausalModel('I',bundle,cfg)
        with patch.object(network,'warm_history',warm),patch.object(network,'frontier',frontier),controller(radio,10):
            tic=time.monotonic()
            result,_=network.run_world(cfg,world,net,bundle,'default')
        old=json.loads((ROOT/'results/amovfly/I'/win['flight_id']/budget/'complete.json').read_text())
        assert old['result']==result
        checks.append(dict(budget=budget,all_fields_exact=True,elapsed_sec=time.monotonic()-tic,reference_sec=old['elapsed_s']))
    (ROOT/'state/frontier_verified.json').write_text(json.dumps(checks,indent=2))
    print(checks)
if __name__=='__main__':main()
