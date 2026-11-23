"""Isolated warm-throughput probe; never writes scientific outputs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('/data/chen')
HERE = Path(__file__).resolve().parent
SOURCE = ROOT/'routeclock_server/routeclock_five_20261115'
RUN = ROOT/'formal_execution/run_20261117_001732'
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                  HF_HUB_OFFLINE='1', TABPFN_DISABLE_TELEMETRY='1',
                  ROUTECLOCK_TABPFN_CHECKPOINT=str(ROOT/'routeclock_server/weights/tabpfn/tabpfn-v2.5-regressor-v2.5_default.ckpt'))


def worker(folder, index, kind):
    import numpy as np
    import torch
    import pickle
    sys.path.insert(0, str(SOURCE))
    from routeclock.foundation import ChronosAdapter
    torch.set_num_threads(4)  # Match the unmodified scientific network entry point.
    if kind == 'tabpfn':
        with (RUN/'five/tabpfn_7/bundle.pkl').open('rb') as f:
            bundle = pickle.load(f)
        with (RUN/'five_data/test/0000.pkl').open('rb') as f:
            anchors = pickle.load(f)
        head = bundle['heads']['I']
        head.reg.device = 'cuda'
        cursor = 0
        def query():
            nonlocal cursor
            a = anchors[cursor % len(anchors)]
            cursor += 1
            return head.predict(a['package'], a['features']['I'])
    else:
        hub = ROOT/'routeclock_server/weights/huggingface/hub/models--amazon--chronos-2'
        model = ChronosAdapter('cuda', str(hub/'snapshots'/(hub/'refs/main').read_text().strip()))
        rng = np.random.default_rng(7)
        names = ['measurement_mask','measurement_age','radial_velocity','radial_std',
                 *[f'position_cov_{i}{j}' for i,j in zip(*np.triu_indices(3))],
                 *[f'position_{i}' for i in range(3)]]
        past = {k:rng.normal(size=720).astype('float32') for k in names}
        target = rng.normal(size=(3,720)).astype('float32')
        keys = ['route_x','route_y','route_z','route_std_x','route_std_y','route_std_z','route_support']
        plain = dict(target=target,past_covariates=past)
        route = dict(target=target,past_covariates={**past,**{k:rng.normal(size=720).astype('float32') for k in keys}},
                     future_covariates={k:rng.normal(size=20).astype('float32') for k in keys})
        cursor = 0
        def query():
            nonlocal cursor
            inp = (plain,route)[cursor % 2]
            cursor += 1
            return model.predict_inputs([(inp,np.zeros(3),np.eye(3))])
    for _ in range(4):
        query()
    torch.cuda.synchronize()
    (folder/f'{index}.ready').touch()
    deadline = time.monotonic()+180
    while not (folder/'go').exists():
        if time.monotonic()>deadline:
            raise TimeoutError('Probe barrier timeout')
        time.sleep(.05)
    start = time.monotonic()
    count = 0
    while time.monotonic()-start < 20:
        query()
        count += 1
    torch.cuda.synchronize()
    elapsed = time.monotonic()-start
    (folder/f'{index}.json').write_text(json.dumps(dict(kind=kind,queries=count,seconds=elapsed,
        queries_per_second=count/elapsed,peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)))


def run():
    sys.path.insert(0,str(ROOT/'formal_execution'))
    import resource_control as resources
    results = []
    for n in (2,4,6):
        folder = HERE/f'mixed_{n}_{time.time_ns()}'
        folder.mkdir()
        children=[]
        logs=[]
        try:
            for i in range(n):
                log=(folder/f'{i}.log').open('w')
                logs.append(log)
                children.append(subprocess.Popen([sys.executable,__file__,'worker',str(folder),str(i),
                    'tabpfn' if i%2 else 'chronos'],stdout=log,stderr=subprocess.STDOUT))
            deadline=time.monotonic()+180
            while len(list(folder.glob('*.ready')))<n:
                if any(p.poll() is not None for p in children):
                    raise RuntimeError(f'Probe worker failed; inspect {folder}')
                if time.monotonic()>deadline:
                    raise TimeoutError(f'Probe initialization timeout; inspect {folder}')
                time.sleep(.5)
            before=resources.sample()
            (folder/'go').touch()
            samples=[]
            while any(p.poll() is None for p in children):
                if time.monotonic()>deadline:
                    raise TimeoutError('Probe measurement timeout')
                samples.append(dict(cpu=resources.sample(),gpu=resources.sample_gpu()))
                time.sleep(1)
            if any(p.returncode for p in children):
                raise RuntimeError(f'Probe measurement failed; inspect {folder}')
            after=resources.sample()
            workers=[json.loads((folder/f'{i}.json').read_text()) for i in range(n)]
            record=dict(concurrency=n,folder=str(folder),workers=workers,
                throughput={k:sum(w['queries_per_second'] for w in workers if w['kind']==k) for k in ('chronos','tabpfn')},
                cpu=resources.sample_rates(after,before),
                peak_vram_gib=max(s['gpu']['gpu_memory_used_gib'] for s in samples),
                peak_memory_gib=max(s['cpu']['memory_current_bytes'] for s in samples)/2**30,
                mean_gpu_util=sum(s['gpu']['gpu_utilization_percent'] for s in samples)/len(samples))
            (folder/'samples.json').write_text(json.dumps(samples,indent=2))
            results.append(record)
            (HERE/'probe_results.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(record),flush=True)
        finally:
            for p in children:
                if p.poll() is None:
                    p.terminate()
            for p in children:
                try:
                    p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
            for f in logs:
                f.close()


if __name__=='__main__':
    if len(sys.argv)>1:
        worker(Path(sys.argv[2]),int(sys.argv[3]),sys.argv[4])
    else:
        run()
