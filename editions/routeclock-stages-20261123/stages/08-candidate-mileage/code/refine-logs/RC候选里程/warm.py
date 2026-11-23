import sys
from matrix import setup,prev
from protocol_local import log
setup()
for w in map(int,sys.argv[1:]):
    world=prev.rr.get_world(w)
    with prev.rr.memoized_arc():ts=prev.runtime.warm_trackers(world)
    assert len(ts)==16
    log(f'新确认世界 {w} 的 16 个预热状态已就绪。')
