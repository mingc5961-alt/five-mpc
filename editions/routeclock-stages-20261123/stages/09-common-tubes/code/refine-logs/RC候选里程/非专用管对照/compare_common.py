"""Use the same two frozen centers and scheduling code as the primary confirmation."""
import sys
from pathlib import Path
from unittest.mock import patch
import pickle
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent))
import matrix
from protocol_local import log
from fit_heads import predict
import fit_heads
prev=matrix.prev

def fit():
    with patch.object(fit_heads,'OUT',OUT.parent),patch.object(fit_heads,'KINDS',('common',)),patch.object(fit_heads,'log',log):
        fit_heads.fit('v2')

def worker(arm):
    matrix.setup()
    src=(prev.rr.SPECIAL/'heads/Chronos-2/common.pkl' if arm=='Chronos-2' else OUT.parent/'heads/v2/common.pkl')
    head=pickle.loads(src.read_bytes())
    assert head['kind']=='common'
    label='Chronos-公共管' if arm=='Chronos-2' else '新RC-公共管'
    def publish(path,obj):
        obj.update(arm=label,center=arm,tube='common',head_path=str(src))
        original_write(path,obj)
    original_write=prev.rr.write
    def pred(h,x,z):return predict(h,x,None)
    with patch.object(prev,'ROOT',OUT),patch.object(prev,'head_for',lambda _: (head,head['factors'])),patch.object(matrix,'predict',pred),patch.object(prev,'predict',pred),patch.object(prev.rr,'write',publish),patch.object(prev,'log',lambda msg:log('非专用管补测 '+msg.replace('/210','/60').replace(arm,label))):
        engines={}
        for w in range(50,60):
            for b in 'TML':prev.run(arm,w,b,engines)
if __name__=='__main__':
    if sys.argv[1]=='fit':fit()
    else:worker(sys.argv[1])
