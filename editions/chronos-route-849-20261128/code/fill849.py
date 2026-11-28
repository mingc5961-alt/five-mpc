"""Fill otherwise idle CPU slots while the last E1 input worlds finish."""
from pipeline_prefetch import work
from parallel849 import *
from concurrent.futures import wait,FIRST_COMPLETED
def drivers():
    active=set()
    for f in (OUT/'jobs').glob('*.json'):
        d=get(f);pid=d.get('pid')
        if pid and Path(f'/proc/{pid}').exists():active.add(pid)
    for f in (OUT/'workers').glob('*'):
        if f.name.isdigit() and Path('/proc',f.name).exists():active.add(int(f.name))
    return len(active)
def main():
    w,_=world('development',1400);Model(w,'Chronos')
    jobs=[('train',k) for k in range(1007,1032)]+[('train_validation',k) for k in range(1100,1116)]
    with ProcessPoolExecutor(30,mp_context=mp.get_context('fork')) as pool:
        active=set()
        while jobs or active:
            if (OUT/'E1/calibration/TIME.pkl').is_file():jobs.clear()
            # Include diagnostics and monitoring in the reserved capacity.
            room=min(30-len(active),max(0,capacity()-3-drivers()))
            for _ in range(min(room,len(jobs))):active.add(pool.submit(work,jobs.pop(0)))
            if active:
                done,active=wait(active,timeout=6,return_when=FIRST_COMPLETED)
                for f in done:print('prefetched',f.result(),flush=True)
            else:time.sleep(5)
if __name__=='__main__':main()
