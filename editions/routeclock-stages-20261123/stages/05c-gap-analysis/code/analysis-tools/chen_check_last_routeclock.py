from pathlib import Path
import sys
import time
import json
import faulthandler
import signal
from unittest.mock import patch
import numpy as np
sys.path.insert(0,'/data/chen/refine-logs/RC_Chronos2_差距定位/跨版本归因')
import old_state
from routeclock.memory import Fragment,RouteMemory
from routeclock.predictors import Tracker
faulthandler.register(signal.SIGUSR1)
old_arc=Fragment.arc;old_at=Fragment.at;old_forecast=RouteMemory.forecast;old_map=RouteMemory.map_distribution
def forecast(mem,*args,**kwargs):
    arcs={}; positions={}; maps={}
    def arc(f):
        k=id(f)
        if k not in arcs:arcs[k]=old_arc(f)
        return arcs[k]
    def at(f,s):
        k=(id(f),s)
        if k not in positions:positions[k]=old_at(f,s)
        return positions[k]
    def mapping(f,fi,s,depth=0):
        k=(fi,s,depth)
        if k not in maps:maps[k]=old_map(f,fi,s,depth)
        return maps[k]
    with patch.object(Fragment,'arc',arc),patch.object(Fragment,'at',at),patch.object(RouteMemory,'map_distribution',mapping):
        return old_forecast(mem,*args,**kwargs)
dest=Path('/tmp/chen_routeclock_perf_check_'+sys.argv[1]+'.json')
old_state.write=lambda ignored,row:dest.write_text(json.dumps(row))
issued=Tracker.issue
counter=0
def issue(tr,*a,**kw):
    global counter
    counter+=1
    if counter%50==1:print(counter,round(tr.history[-1].t,2),flush=True)
    return issued(tr,*a,**kw)
with patch.object(RouteMemory,'forecast',forecast),patch.object(Tracker,'issue',issue):
    old_state.run('C',int(sys.argv[1]))
