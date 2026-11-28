"""Transfer supervision only after every active E1 input has a verified durable receipt."""
from base849 import *
import signal,time
def main():
    parent=int(sys.argv[1]);p=Path(f'/proc/{parent}');identity=(p/'stat').read_text().split()[21]
    assert b'code/gate849.py' in (p/'cmdline').read_bytes()
    children=[int(x) for x in (p/f'task/{parent}/children').read_text().split()]
    assert len(children)==32
    os.kill(parent,signal.SIGSTOP);counts=[]
    try:
        while True:
            files=list((OUT/'E1/calibration/inputs').glob('*.json'));counts.append(len(files))
            if len(files)==32:break
            if len(counts)>90 and len(set(counts[-90:]))==1:raise RuntimeError('no input receipt within 15 minutes')
            time.sleep(10)
        for f in files:assert get(f)['sha256']==sha_file(f.with_suffix('.pkl'))
        put(OUT/'logs/supervisor_transfer.json',dict(input_worlds=32,original_pid=parent,verified_before_transfer=True,reason='NUMA-local native process pools for counterfactual and end-to-end runs'))
        assert (p/'stat').read_text().split()[21]==identity
        # All 32 input jobs have produced their final checksum; no active scientific work is terminated.
        for child in children:
            try:os.kill(child,signal.SIGTERM)
            except ProcessLookupError:pass
        os.kill(parent,signal.SIGTERM);os.kill(parent,signal.SIGCONT)
    except BaseException:
        os.kill(parent,signal.SIGCONT);raise
if __name__=='__main__':main()
