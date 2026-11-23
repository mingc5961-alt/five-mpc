"""Local single-worker prototype. Memoize arc length only while a forecast reads fixed fragments."""
from contextlib import contextmanager
from unittest.mock import patch
from pathlib import Path
import gzip
import pickle
import numpy as np
import bench
from routeclock.memory import RouteMemory,Fragment

@contextmanager
def reuse_arc():
    forecast=RouteMemory.forecast
    arc=Fragment.arc
    def cached_forecast(memory,*args,**kwargs):
        values={}
        def cached(fragment):
            k=id(fragment)
            if k not in values:values[k]=arc(fragment)
            return values[k]
        with patch.object(Fragment,'arc',cached):
            return forecast(memory,*args,**kwargs)
    with patch.object(RouteMemory,'forecast',cached_forecast):yield

def check():
    n=0
    for w in (0,1,2,7,9):
        with gzip.open(bench.PRIOR/'cache/warm'/f'v1_{w:03}.pkl.gz','rb') as f:trackers=pickle.load(f)
        for tr in trackers:
            uncached=tr.memory.forecast(bench.network.HS)
            with reuse_arc():cached=tr.memory.forecast(bench.network.HS)
            for a,b in zip(uncached,cached,strict=True):np.testing.assert_array_equal(a,b)
            n+=1
    return n
if __name__=='__main__':
    n=check()
    bench.OUT=Path(__file__).resolve().parent/'弧长缓存';bench.OUT.mkdir(exist_ok=True)
    print('Verified at',n,'archived states',flush=True)
    with reuse_arc():bench.bench('RC','original',1024,0)
