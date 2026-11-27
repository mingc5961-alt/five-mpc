import os,time,traceback
from base import *
# Linux CPUs from the two-core concurrency trial on this experiment host.
CUDA_HOST_CPUS = {1, 2}

def init(cpu):
    if cpu:
        os.environ['CUDA_VISIBLE_DEVICES']=''
        allowed = os.sched_getaffinity(0)
        if CUDA_HOST_CPUS < allowed:
            os.sched_setaffinity(0, allowed - CUDA_HOST_CPUS)
    for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
def run(task):
    key=task['id'];a=task['args'];kind=task['kind'];p=BASE/'receipts'/(key+'.json')
    tic=time.time()
    affinity = None
    try:
        if task.get('inference_device') == 'cuda':
            allowed = os.sched_getaffinity(0)
            if CUDA_HOST_CPUS < allowed:
                os.sched_setaffinity(0, CUDA_HOST_CPUS)
                affinity = allowed
        if p.exists():return True,'',0
        if kind=='world':
            from worldgen import generate
            generate(**a);data=json.loads(worldfile(a['w']).with_suffix('.json').read_text())
        elif kind=='anchor':
            data=dict(provenance='用户提供的 2026-11-25 EXPERIMENT_PLAN.md，历史参照，不作本次仿真实测',archived_native_risk_pct=4.505208,archived_native_delta_mbps=162.73,archived_tuned_delta_mbps=-86.57)
        elif kind in ('train','cal','gate','shard'):
            import fitting
            data=getattr(fitting,{'cal':'calibrate'}.get(kind,kind))(**a)
        elif kind=='tables':
            from tables import report
            data=report(task)
        else:
            import experiments
            data=experiments.cell(**a,inference_device=task.get("inference_device","cpu")) if kind=='cell' else experiments.panel(**a,path=p)
        record = dict(id=key,args=a,elapsed=time.time()-tic,finished=time.time(),data=data)
        if 'inference_device' in task:
            record['inference_device'] = task['inference_device']
        js(p,record)
        return True,'',time.time()-tic
    except Exception:
        err=traceback.format_exc()
        with (BASE/'logs'/(key+'.log')).open('a') as f:f.write(err+'\n')
        return False,err.splitlines()[-1],time.time()-tic
    finally:
        if affinity is not None:
            os.sched_setaffinity(0, affinity)
