"""Durable execution of the supplied RouteClock protocol; no model substitutions."""
import argparse
import collections
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import traceback

import psutil
import resource_control as resources

HERE = Path(__file__).resolve().parent
SOURCE = Path('/data/chen/routeclock_server')
PYTHON = Path('/data/chen/server_assessment/venv/bin/python')
FIVE = SOURCE/'routeclock_five_20261115'
MPC = SOURCE/'routeclock_server_full'
RUN = HERE/'run_20261117_001732'
START = '2026-11-17T00:17:32+08:00'

def now():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',dir=path.parent,delete=False) as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        temp=Path(f.name)
    temp.replace(path)

def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def code_hashes():
    paths = [SOURCE/'server.py', SOURCE/'requirements-linux.txt']
    for folder in [FIVE, MPC, SOURCE/'routeclock_platform_20261113', SOURCE/'routeclock_mpc']:
        for part in ['routeclock','mpc','tools','configs','tests']:
            base = folder/part
            if base.exists():
                paths += [p for p in base.rglob('*') if p.suffix in ('.py','.json')]
        if (folder/'run.py').exists(): paths.append(folder/'run.py')
    return {str(p.relative_to(SOURCE)):digest(p) for p in sorted(set(paths))}

def environment():
    return dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1',
                MKL_NUM_THREADS='1', TABPFN_DISABLE_TELEMETRY='1',
                PYTHONUNBUFFERED='1', HF_HUB_OFFLINE='1', CUDA_VISIBLE_DEVICES='0')

def build_jobs():
    jobs=[]
    def add(id, args, deps=(), cpu=4, gpu=0, kind='command', meta=None, cwd=SOURCE):
        jobs.append(dict(id=id, command=[str(PYTHON),*map(str,args)], cwd=str(cwd),
                         dependencies=list(deps), cpu=cpu, gpu=gpu, kind=kind, meta=meta or {}))
        return id
    verify=add('verify_delivery',['server.py','verify'],cpu=1)
    tests=[]
    for name in ['routeclock_five_20261115','routeclock_server_full','routeclock_platform_20261113','routeclock_mpc']:
        tests.append(add('tests_'+name,['-m','pytest','-q','-p','no:cacheprovider','tests'],[verify],cpu=2,cwd=SOURCE/name))
    prereq=tests
    # Full day-one timing has a distinct world namespace and is never counted as a test result.
    add('day1_timing',['server.py','five','timing','--bundle','runs/five_pilot/results/bundle.pkl',
                      '--out',RUN/'timing'],prereq,gpu=1,kind='timing')
    counts=read(FIVE/'configs/research.json')['data']['single_uav_tasks']
    datajobs=[]
    for split,count in counts.items():
        for start in range(0,count,20):
            stop=min(start+20,count)
            datajobs.append(add(f'data_{split}_{start:04}',
                ['server.py','five','generate','--split',split,'--start',start,'--stop',stop,
                 '--data',RUN/'five_data','--out',RUN/'five'],prereq,gpu=1,kind='data',
                meta=dict(split=split,start=start,stop=stop)))
    dataaudit=add('audit_dataset',[HERE/'audit.py','dataset',RUN],datajobs,cpu=1)
    models={}
    for head,seed in [('lightgbm',7),('tabpfn',7),('lightgbm',11),('lightgbm',19)]:
        models[head,seed]=add(f'train_{head}_{seed}',
            ['server.py','five','train','--head',head,'--seed',seed,'--data',RUN/'five_data','--out',RUN/'five'],
            [dataaudit],gpu=2 if head=='tabpfn' else 0,kind='train',meta=dict(head=head,seed=seed))
    sys.path.insert(0,str(FIVE/'tools'))
    from five_campaign import jobs as matrix
    networks=[]; summaries=[]
    for head,seed in models:
        blocks=['main','actions','reuse'] if (head,seed)==('lightgbm',7) else ['repeat']
        local=[]
        for block in blocks:
            for index,(mission,method,mode,frac) in enumerate(matrix(block)):
                id=add(f'network_{head}_{seed}_{block}_{index:03}',
                    ['server.py','five','network','--head',head,'--seed',seed,'--block',block,
                     '--start',index,'--stop',index+1,'--bundle',RUN/f'five/{head}_{seed}/bundle.pkl',
                     '--out',RUN/'five'],[models[head,seed]],
                    gpu=2 if head=='tabpfn' else (1 if method in ('C','CR_raw') else 0),
                    kind='network',meta=dict(head=head,seed=seed,block=block,index=index,mission=mission,
                                             method=method,mode=mode,reuse=frac))
                local.append(id);networks.append(id)
        summaries.append(add(f'summary_{head}_{seed}',
            ['server.py','five','summary','--head',head,'--seed',seed,'--out',RUN/'five'],local,cpu=1,kind='summary',meta=dict(head=head,seed=seed)))
    # Separate output roots preserve supplied pilot models and historical ledgers.
    mpc_train=add('mpc_train',['server.py','mpc','train','--config','configs/formal.json',
                 '--out',RUN/'mpc_train','--workers','4'],prereq,kind='mpc_train')
    mpc_sim=add('mpc_simulate',['server.py','mpc','simulate','--config','configs/formal.json',
               '--models',RUN/'mpc_train','--out',RUN/'mpc_formal','--workers','4'],[mpc_train],kind='mpc_simulate')
    mpc_report=add('mpc_report',['server.py','mpc','report','--out',RUN/'mpc_formal'],[mpc_sim],cpu=1)
    mpc_verify=add('mpc_verify',['server.py','mpc','verify','--out',RUN/'mpc_formal'],[mpc_report],cpu=1,kind='mpc_verify')
    paired=add('paired_run',['server.py','paired','run','--workers','4','--out',RUN/'paired'],prereq,kind='paired_run')
    pa=add('paired_audit',['server.py','paired','audit','--out',RUN/'paired'],[paired],cpu=2,kind='paired_audit')
    ps=add('paired_search',['server.py','paired','search','--out',RUN/'paired'],[pa],cpu=2,kind='paired_search')
    pend=add('paired_summary',['server.py','paired','summary','--out',RUN/'paired'],[ps],cpu=1,kind='paired_summary')
    add('formal_comparison_report',[HERE/'audit.py','report',RUN],
        [*summaries,mpc_verify,pend,'day1_timing'],cpu=1,kind='final_report')
    return jobs

def initialize():
    RUN.mkdir(parents=True,exist_ok=True)
    if (RUN/'plan.json').exists():
        raise RuntimeError('Plan already exists; use run/status to resume the same campaign.')
    codes=code_hashes()
    save(RUN/'source_hashes.json',codes)
    shutil.copy2(SOURCE/'manifest.json',RUN/'delivery_manifest.json')
    shutil.copy2('/data/chen/server_assessment/requirements-resolved.txt',RUN/'requirements-resolved.txt')
    for p in [SOURCE/'服务器从这里开始.md',FIVE/'SERVER_RUNBOOK.md',FIVE/'V2执行方案.md']:
        shutil.copy2(p,RUN/p.name)
    save(RUN/'plan.json',dict(started_at=START,extension_deadline=(dt.datetime.fromisoformat(START)+dt.timedelta(hours=10)).isoformat(),
         protocol='服务器从这里开始.md + SERVER_RUNBOOK.md; V2 extensions conditional on core completion within ten hours',
         source=str(SOURCE),python=str(PYTHON),cpu_capacity=12,gpu_capacity=2,
         gpu_policy='At most two Chronos processes; TabPFN uses both slots exclusively.',
         counts=dict(single_uav_tasks=2400,model_fits=4,five_network_runs=690,mpc_network_runs=120,paired_events=143),
         no_mechanism_changes=True,jobs=build_jobs()))
    print(RUN)

def jobdir(id):
    return RUN/'jobs'/id

def process_alive(record):
    try:
        p=psutil.Process(record['pid'])
        return p.is_running() and p.status()!=psutil.STATUS_ZOMBIE and abs(p.create_time()-record['create_time'])<.1
    except (psutil.Error,KeyError): return False

def verify_job(job):
    # All checks inspect outputs; none modify model predictions or thresholds.
    import importlib.util
    spec=importlib.util.spec_from_file_location('campaign_audit',HERE/'audit.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.verify_job(job,RUN,SOURCE)

def worker(id):
    job=next(x for x in read(RUN/'plan.json')['jobs'] if x['id']==id)
    folder=jobdir(id);folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (folder/'receipt.json').exists(): return
        me=psutil.Process()
        save(folder/'running.json',dict(pid=os.getpid(),create_time=me.create_time(),started_at=now(),job=id))
        started=time.time();peak=0;child=None;code=None;error=None;validation=None
        try:
            if code_hashes()!=read(RUN/'source_hashes.json'):
                raise RuntimeError('Source/config fingerprints changed; refusing a mixed-version experiment.')
            with (folder/'stdout.log').open('a') as log:
                log.write(f'\n[{now()}] command={json.dumps(job["command"])}\n');log.flush()
                child=subprocess.Popen(job['command'],cwd=job['cwd'],env=environment(),stdout=log,stderr=subprocess.STDOUT)
                last=0
                while child.poll() is None:
                    try:
                        root=psutil.Process(child.pid)
                        procs=[root,*root.children(recursive=True)]
                        rss=sum(p.memory_info().rss for p in procs if p.is_running())
                        peak=max(peak,rss)
                        if time.time()-last>=15:
                            save(folder/'resource.json',dict(at=now(),child_pid=child.pid,elapsed_s=time.time()-started,rss_bytes=rss,peak_rss_bytes=peak))
                            last=time.time()
                    except psutil.Error: pass
                    time.sleep(1)
                code=child.returncode
            if code!=0: raise RuntimeError(f'Native command exited {code}; inspect stdout.log. No model fallback was used.')
            validation=verify_job(job)
        except Exception:
            error=traceback.format_exc()
        save(folder/'receipt.json',dict(job=id,status='failed' if error else 'passed',started_at=dt.datetime.fromtimestamp(started).astimezone().isoformat(),
            finished_at=now(),wall_s=time.time()-started,exit_code=code,peak_process_tree_rss_bytes=peak,validation=validation,error=error))

def states(plan):
    result={}
    for job in plan['jobs']:
        folder=jobdir(job['id'])
        if (folder/'receipt.json').exists(): result[job['id']]=read(folder/'receipt.json')['status']
        elif (folder/'running.json').exists():
            result[job['id']]='running' if process_alive(read(folder/'running.json')) else 'interrupted'
        else: result[job['id']]='pending'
    # Transitive blocked status is explicit, never counted as successful.
    for _ in range(len(plan['jobs'])):
        changed=False
        for job in plan['jobs']:
            if result[job['id']]=='pending' and any(result[d] in ('failed','blocked','interrupted') for d in job['dependencies']):
                result[job['id']]='blocked';changed=True
        if not changed: break
    return result

def extension_gate(completed_times, expected_jobs, deadline):
    """Judge actual completion timestamps, including after a supervisor restart."""
    if len(completed_times)!=expected_jobs: return None,False
    finished=max(dt.datetime.fromisoformat(x) for x in completed_times)
    return finished.isoformat(),finished<=dt.datetime.fromisoformat(deadline)

def publish_status(plan,state,reason=None,resource_status=None):
    counts=collections.Counter(state.values())
    active=[j['id'] for j in plan['jobs'] if state[j['id']]=='running']
    data={s:len(list((RUN/'five_data'/s).glob('*.pkl'))) for s in ('train','validation','calibration','test')}
    net=list((RUN/'five/network').glob('*/*/*/completed.json'))
    completed=all(v=='passed' for v in state.values())
    finished,eligible=extension_gate(
        [read(jobdir(j['id'])/'receipt.json')['finished_at'] for j in plan['jobs']] if completed else [],
        len(plan['jobs']),plan['extension_deadline'])
    status=dict(at=now(),supervisor_pid=os.getpid(),counts=dict(counts),active_jobs=active,
        failed_jobs=[k for k,v in state.items() if v in ('failed','interrupted')],
        data_tasks_completed=data,data_tasks_expected=dict(train=1200,validation=400,calibration=400,test=400),
        five_network_completed=len(net),five_network_expected=690,
        mpc_network_completed=len(list((RUN/'mpc_formal').glob('scene_*/*/metrics.json'))),mpc_network_expected=120,
        disk_free_GiB=shutil.disk_usage(RUN).free/2**30,core_complete=completed,
        extensions_eligible=eligible,extension_deadline=plan['extension_deadline'],reason=reason)
    if resource_status is not None: status['resources']=resource_status
    save(RUN/'status.json',status)
    lines=[f'更新时间：{status["at"]}',f'任务状态：{dict(counts)}',
           f'正式航线：{sum(data.values())}/2400；五模型网络：{len(net)}/690；MPC网络：{status["mpc_network_completed"]}/120',
           f'正在运行：{", ".join(active)}',f'扩展条件截止：{plan["extension_deadline"]}',
           '核心全量完成。' if completed else '核心全量尚未完成，当前输出不能表述为完整正式结果。']
    if reason: lines.append(reason)
    if resource_status is not None:
        lines.append(f'资源：CPU网络并发 {resource_status["cpu_network_jobs"]}/{resource_status["max_cpu_network_jobs"]}；'
                     f'CPU使用 {resource_status.get("cpu_used_cores", "采样中")} / {resource_status["cpu_quota_cores"]} 核；'
                     f'内存 {resource_status["memory_current_gib"]:.2f} GiB（新任务准入上限 {resource_status["memory_ceiling_gib"]:.2f} GiB）')
    for id in status['failed_jobs']:
        receipt=jobdir(id)/'receipt.json'
        lines.append(f'失败或中断：{id}；日志 {jobdir(id)/"stdout.log"}')
        if receipt.exists(): lines.append(read(receipt).get('error') or '无 traceback；请查进程日志。')
    (RUN/'执行状态与问题.txt').write_text('\n'.join(lines)+'\n')
    if completed:
        save(RUN/'core_completion.json',dict(completed_at=finished,within_ten_hours=eligible,extensions_require_implementation=eligible))
    return status

def supervise():
    plan=read(RUN/'plan.json')
    previous_sample=None;last_resource_log=0
    with (RUN/'supervisor.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        while True:
            state=states(plan)
            free=shutil.disk_usage(RUN).free
            reason=None
            if code_hashes()!=read(RUN/'source_hashes.json'): reason='source_changed; no new jobs started'
            elif free<5*2**30: reason='disk_below_5_GiB; no new jobs started'
            policy=None;resource_status=None;current_sample=None
            running=[j for j in plan['jobs'] if state[j['id']]=='running']
            memory_ceiling=float('inf');memory_committed=0
            cpu_capacity=plan['cpu_capacity']
            try:
                policy=resources.read_policy(RUN)
                if policy is not None:
                    current_sample=resources.sample()
                    cpu_capacity=min(plan['cpu_capacity'],policy['cpu_capacity'],current_sample['cpu_quota_cores'] or plan['cpu_capacity'])
                    memory_ceiling,memory_committed=resources.memory_budget(RUN,running,policy,current_sample)
                    resource_status=dict(cpu_quota_cores=current_sample['cpu_quota_cores'],cpu_capacity=cpu_capacity,
                        max_cpu_network_jobs=policy['max_cpu_network_jobs'],
                        memory_current_gib=current_sample['memory_current_bytes']/2**30,
                        memory_ceiling_gib=memory_ceiling/2**30,oom_kill=current_sample['oom_kill'],
                        **resources.sample_rates(current_sample,previous_sample))
            except Exception as exc:
                reason=f'resource_policy_error: {exc}; no new jobs started'
                policy=None;resource_status=None
            cpu=sum(resources.cpu_cost(j,policy) for j in running)
            cpu_networks=sum(j['kind']=='network' and not j['gpu'] for j in running)
            gpu=sum(j['gpu'] for j in plan['jobs'] if state[j['id']]=='running')
            if reason is None:
                for job in plan['jobs']:
                    id=job['id']
                    if state[id]!='pending' or not all(state[d]=='passed' for d in job['dependencies']): continue
                    if not resources.can_admit(job,policy,cpu,cpu_capacity,gpu,plan['gpu_capacity'],
                                               cpu_networks,memory_committed,memory_ceiling): continue
                    folder=jobdir(id);folder.mkdir(parents=True,exist_ok=True)
                    with (folder/'worker.log').open('a') as log:
                        p=subprocess.Popen([str(PYTHON),str(Path(__file__).resolve()),'worker',id],
                            env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    # Supervisor also writes the initial lease to make restart recovery safe.
                    save(folder/'running.json',dict(pid=p.pid,create_time=psutil.Process(p.pid).create_time(),started_at=now(),job=id))
                    cpu+=resources.cpu_cost(job,policy);gpu+=job['gpu'];state[id]='running'
                    cpu_networks+=int(job['kind']=='network' and not job['gpu'])
                    if policy is not None: memory_committed+=resources.memory_reservation(job,policy)
                    print(now(),'started',id,flush=True)
            if resource_status is not None:
                resource_status.update(cpu_network_jobs=cpu_networks,cpu_reserved=cpu,gpu_slots_reserved=gpu,
                                       memory_committed_gib=memory_committed/2**30)
                if time.monotonic()-last_resource_log>=30:
                    with (RUN/'resource_monitor.jsonl').open('a') as log:
                        log.write(json.dumps(dict(at=now(),**resource_status))+'\n')
                    last_resource_log=time.monotonic()
            if current_sample is not None: previous_sample=current_sample
            status=publish_status(plan,state,reason,resource_status)
            if not status['active_jobs'] and not any(v=='pending' for v in state.values()):
                if not status['core_complete']:
                    subprocess.run([str(PYTHON),str(HERE/'audit.py'),'partial',str(RUN)],env=environment(),check=False)
                print(now(),'terminal',json.dumps(status['counts']),flush=True)
                return
            if reason and not status['active_jobs']:
                print(now(),reason,flush=True);return
            # Reap only direct children; scientific child commands have their own durable workers.
            try:
                while os.waitpid(-1,os.WNOHANG)[0]: pass
            except ChildProcessError: pass
            time.sleep(5)

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','run','status','worker','retry']);p.add_argument('id',nargs='?')
    args=p.parse_args()
    if args.command=='init': initialize()
    elif args.command=='run': supervise()
    elif args.command=='worker': worker(args.id)
    elif args.command=='retry':
        plan=read(RUN/'plan.json');state=states(plan)
        if state.get(args.id) not in ('failed','interrupted'):
            raise SystemExit('Only a failed/interrupted job may be retried; passed/running jobs are protected.')
        folder=jobdir(args.id);archive=folder/f'attempt_{time.time_ns()}';archive.mkdir()
        if (folder/'running.json').exists() and process_alive(read(folder/'running.json')):
            raise SystemExit('Worker still exiting; wait before retrying it.')
        for name in ['receipt.json','running.json','resource.json','stdout.log','worker.log']:
            if (folder/name).exists(): shutil.move(folder/name,archive/name)
        print('Original attempt preserved:',archive)
    else: print(json.dumps(read(RUN/'status.json'),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
