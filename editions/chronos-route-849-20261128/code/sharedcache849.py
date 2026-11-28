"""Serialize only colliding E2 forecast computations; complete paid prefixes remain the data keys."""
import fcntl,os,time
from chronos_recipients import Cache,OUT
class SharedPrefixCache(Cache):
    def __init__(self,w,wm,split):
        super().__init__(w,wm,split)
        self.lockdir=OUT/'cache/prefix_locks'/split/self.spec
        self.lockdir.mkdir(parents=True,exist_ok=True)
    def __call__(self,uid,mem):
        # A collision only waits. The underlying database still verifies its complete
        # world, model, UID, timestamp and paid-prefix key and the payload digest.
        lane=os.environ.get('CHRONOS849_OVERFLOW_LANE')
        if lane is not None:
            permit=OUT/'logs'/os.environ.get('CHRONOS849_OVERFLOW_PERMIT','overflow_enabled.txt')
            while int(lane)>=int(permit.read_text()):
                if time.time()-permit.stat().st_mtime>30:break
                time.sleep(.5)
        shard=int(mem.key[:8],16)%256
        with (self.lockdir/str(shard)).open('a') as gate:
            fcntl.flock(gate,fcntl.LOCK_EX)
            return super().__call__(uid,mem)
