"""Publish the full planned matrix, verified outputs, live work and resource samples."""
import argparse
from collections import Counter
import csv
from datetime import datetime
import fcntl
from functools import lru_cache
import io
import json
import math
import os
from pathlib import Path
import subprocess
import time

from resource_monitor import BASE,RUNTIME,measure

LOG=BASE.parents[1]/'refine-logs'
VIEW=LOG/'实时实验进度.txt'
DETAIL=LOG/'2280场逐场进度.csv'
HISTORY=LOG/'实验资源采样.csv'
STATE=RUNTIME/'monitor_state.json'
PLAN=BASE/'planned_cells.csv'
MODELS=('chronos','moirai','vector')
GROUPS=('两端工况验证 T/L','中间工况验证 M','MCS 余量验证','T 主工况确认','固定尺度与 CR_raw 归因','M/L 预算对照','VECTOR 跨种子复核','统一 LightGBM 误差管')
STATUSES=('已完成','运行中','已排队','未排队','待核查')
FIELDS=['场次ID','实验组','阶段','世界ID','工况','模型','误差管','训练种子','MCS余量_dB','状态','进程PID','开始时间','完成时间','运行耗时_s','进程CPU_s','PyTorch峰值已分配显存_MiB','PyTorch峰值保留显存_MiB','结果文件','依据']
def stamp(t):
    return datetime.fromtimestamp(t).astimezone().strftime('%Y-%m-%d %H:%M:%S %z') if t else ''

def atomic(path,value):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(value)
    tmp.replace(path)

def load_json(path):
    return json.loads(path.read_text())

def group(r):
    if r['phase']=='validation':
        if r['mcs_margin_db']!='5': return 2
        return int(r['condition']=='M')
    if r['tube']=='lightgbm': return 7
    if r['seed']!='7': return 6
    if r['condition']!='T': return 5
    if r['tube']=='fixed' or r['model']=='CR_raw': return 4
    return 3

@lru_cache(maxsize=3000)
def verified(path,mtime,phase,model,idx,tube,condition="T"):
    r=load_json(path)
    assert (r['phase'],r['model'],r['world'],r['tube'])==(phase,model,idx,tube)
    cfg=r['config']
    assert cfg['primary_N']==16 and cfg['warmup_s']==cfg['service_s']==180
    assert math.isclose(cfg['sensing_budget'],{'T':.05/3,'L':.05,'M':.025}[condition]) and r['probes_per_user']==9000
    assert len(r['bits'])==16 and all(math.isfinite(v) and v>=0 for v in r['bits'])
    assert 0<=r['R']<=1 and 0<=r['unavailable_fraction']<=1
    assert math.isfinite(r['G_mbps']) and r['G_mbps']>=0 and len(r['initial_center_sha256'])==64
    assert abs(sum(r['ledger_s'])-180)<1e-6 and r['wall_s']>0
    return r['wall_s']

def matrix(procs,controller):
    with PLAN.open() as f: plan=list(csv.DictReader(f))
    assert len(plan)==len({tuple(r.values()) for r in plan})==2280
    active={}
    for p in (RUNTIME/'active').glob('*.json'):
        try: a=load_json(p)
        except FileNotFoundError: continue
        if a['pid'] in procs and procs[a['pid']]['task'].startswith('runtime.dispatch --worker'):
            active[a['model'],a['world'],a['tube']]=a
    extra_jobs={};extra_active={}
    for folder,module in [('learned_queue','runtime.learned_queue'),('remaining_queue','runtime.full_queue')]:
        extra=BASE/folder
        manifest=load_json(extra/'manifest.json') if (extra/'manifest.json').exists() else {}
        running=procs.get(manifest.get('pid'),{}).get('task')==module+' run'
        for row in manifest.get('cells',[]):
            assert row['id'] not in extra_jobs
            extra_jobs[row['id']]=(row,extra,running)
        for path in (extra/'active').glob('*.json'):
            try: item=load_json(path)
            except FileNotFoundError: continue
            if procs.get(item['pid'],{}).get('task','').startswith(module+' worker '):
                extra_active[item['id']]=item
    output=[]; counts=[Counter() for _ in GROUPS]; recent=[]; pairs={}
    for i,r in enumerate(plan,1):
        g=group(r); model=r['model']; phase=r['phase']; idx=int(r['world'])
        candidate=model in MODELS and r['seed']=='7' and r['condition']=='T' and r['mcs_margin_db']=='5'
        tube='native' if r['tube']=='default' and model in MODELS else r['tube']
        assigned=candidate and phase=='confirmation'
        commissioning=candidate and phase=='validation' and idx==0 and tube=='native'
        path=BASE/'tube_network'/phase/model/f'{idx:03}'/(tube+'.json')
        stat='未排队'; note='计划已登记，当前执行器未覆盖'; pid=start=end=wall=cpu=alloc=reserved=file=''
        if assigned or commissioning:
            a=active.get((model,idx,tube)) if assigned else None
            if path.is_file():
                try:
                    at=path.stat().st_mtime
                    wall=verified(path,at,phase,model,idx,tube)
                    stat='已完成'; file=str(path); end=stamp(at)
                    start=stamp(at-wall)
                    note='按落盘时间及耗时回推开始时间'
                    if commissioning: note+='；完整启动验证与计划格配置一致，复用一次'
                    if assigned:
                        recent.append(at)
                        pairs.setdefault((model,idx),set()).add(tube)
                        meta=RUNTIME/'cell_resources'/model/f'{idx:03}'/(tube+'.json')
                        if meta.is_file():
                            m=load_json(meta)
                            pid=m['pid']; cpu=round(m['cpu_seconds'],3); alloc=round(m['peak_allocated_mib'],2); reserved=round(m['peak_reserved_mib'],2)
                    wall=round(wall,3)
                except (KeyError,ValueError,AssertionError,OSError,TypeError) as exc:
                    stat='待核查';note='结果校验异常 '+type(exc).__name__;file=str(path)
            elif a:
                stat='运行中';pid=a['pid'];start=stamp(a['started']);note='执行进程已进入该场闭环'
            elif assigned and controller:
                stat='已排队';note='E22 执行器负责此场'
            elif assigned:
                note='等待恢复 E22 执行器'
        cell=f'RC-{i:04}'
        if cell in extra_jobs:
            spec,extra,extra_live=extra_jobs[cell]
            dest=extra/'results'/(cell+'.json')
            if dest.exists():
                try:
                    meta=load_json(dest)
                    assert all(meta[k]==v for k,v in spec.items())
                    at=dest.stat().st_mtime
                    wall=round(verified(dest,at,phase,model,idx,tube,r['condition']),3)
                    stat='已完成';file=str(dest);end=stamp(at);start=stamp(meta['started'])
                    pid=meta['pid'];cpu=round(meta['cpu_seconds'],3)
                    alloc=round(meta['peak_allocated_mib'],2);reserved=round(meta['peak_reserved_mib'],2)
                    note='逐场结果与计划键、工况和时间账本已核对'
                except (KeyError,ValueError,AssertionError,OSError,TypeError) as exc:
                    stat='待核查';note='结果异常 '+type(exc).__name__;file=str(dest)
            elif cell in extra_active:
                a=extra_active[cell];stat='运行中';pid=a['pid'];start=stamp(a['started']);note='续跑执行器已进入该场'
            elif extra_live:
                stat='已排队'
                note='续跑执行器已接管'
                if spec['seed']!=7 and not (extra/f"seed{spec['seed']}"/'ready.json').exists():
                    note+='；等待本种子训练与校准'
                if extra.name=='remaining_queue':
                    fitting={'F':'F','I':'classical','P':'classical','RC':'classical'}.get(model,model)
                    if not (extra/(fitting+'_ready.json')).exists(): note+='；等待模型或工作点准备'
                    elif phase=='confirmation': note+='；按本模型验证依赖放行'
            else:
                note='调度器已退出，请检查 '+str(extra)
        counts[g][stat]+=1
        output.append([f'RC-{i:04}',GROUPS[g],phase,idx,r['condition'],model,tube,r['seed'],r['mcs_margin_db'],stat,pid,start,end,wall,cpu,alloc,reserved,file,note])
    return output,counts,recent,pairs

def display(x,n=1):
    return '未采到' if x is None else f'{x:.{n}f}'

def size_gib():
    return int(subprocess.check_output(['du','-s','-B1',str(BASE)],text=True).split()[0])/2**30

def snapshot(prev,state,write):
    metrics,procs,prev=measure(prev)
    ts=metrics['time']
    status=load_json(BASE/'tube_campaign_status.json')
    controller=procs.get(status['pid'],{}).get('task','').split(' ',1)[0] in ('refine.campaign','runtime.dispatch') and status['stage']=='freeze_and_confirmation'
    rows,groups,recent,pairs=matrix(procs,controller)
    total=sum(groups,Counter())
    sig=tuple((r[0],r[9],r[10],r[14],r[18]) for r in rows if r[9]!='未排队')
    if write and (sig!=state.get('sig') or not DETAIL.exists()):
        buff=io.StringIO(); w=csv.writer(buff); w.writerow(FIELDS); w.writerows(rows)
        atomic(DETAIL,'\ufeff'+buff.getvalue());state['sig']=sig
    if ts-state.get('disk_sample',0)>60:
        state['output_gib']=size_gib();state['disk_sample']=ts
    m=metrics
    state.setdefault('since',ts)
    for key in ('experiment_cores','experiment_gpu_mib','device_vram_mib','cgroup_memory_gib','gpu_pct'):
        if m[key] is not None: state['peak_'+key]=max(m[key],state.get('peak_'+key,0))
    if write:
        state['cpu_seconds']=state.get('cpu_seconds',0)+m['experiment_cores']*m['dt_s']
    else:
        state.setdefault('cpu_seconds',0)
    metrics['completed']=total['已完成'];metrics['active']=total['运行中'];metrics['queued']=total['已排队']
    if write:
        new=not HISTORY.exists() or HISTORY.stat().st_size==0
        with HISTORY.open('a') as f:
            w=csv.DictWriter(f,fieldnames=['timestamp',*metrics])
            if new:w.writeheader()
            w.writerow(dict(timestamp=stamp(ts),**metrics))
        atomic(STATE,json.dumps({k:v for k,v in state.items() if k!='sig'}))
    text=[f'RouteClock 全部实验实时进度    {stamp(ts)}    每 5 秒刷新','',
          f"总计划 2,280 场  |  已完成 {total['已完成']}（{total['已完成']/2280:.1%}）  |  运行中 {total['运行中']}  |  已排队 {total['已排队']}  |  未排队 {total['未排队']}  |  待核查 {total['待核查']}",
          f"尚未完成 {2280-total['已完成']} 场。完整启动验证中可复用的 3 场原生管按配置计入验证，各计一次。",
          '', '实验组                       总量   完成   运行   排队   未排   待核查']
    for name,c in zip(GROUPS,groups):
        pad=' '*max(1,29-sum(2 if ord(ch)>127 else 1 for ch in name))
        text.append(f'{name}{pad}{sum(c.values()):5}'+''.join(f'{c[x]:7}' for x in STATUSES))
    recent_total=sum(1 for r in rows if r[9]=='已完成' and Path(r[17]).stat().st_mtime>=ts-300)
    text+=['',f'全计划近 5 分钟完成 {recent_total} 场。续跑清单共 397 + 1,250 场；模型训练、校准与前沿搜索单列。',
           '工况 T/M/L 的感知预算分别为 0.05/3、0.025、0.05，每次观测成本 3.1 ms。验证取 20 个世界，确认取另 70 个世界。',f"专项 E22 已落盘 {len(recent)}/630 场；近 5 分钟完成 {sum(t>=ts-300 for t in recent)} 场。阶段 {status['stage']}，调度进程 {status['pid']}。",
           '模型           原生管        LightGBM         固定管       原生/LightGBM成对世界']
    for model in MODELS:
        done=[tubes for (name,i),tubes in pairs.items() if name==model]
        nums=[sum(t in s for s in done) for t in ('native','lightgbm','fixed')]
        text.append(f'{model:10}'+''.join(f'{v:8}/70   ' for v in [*nums,sum({"native","lightgbm"}<=s for s in done)]))
    text+=['','服务器资源',
           f"CPU    实验 {display(m['experiment_cores'],2)} 核，容器全部 {display(m['cgroup_cpu_cores'],2)} / {display(m['cpu_limit'],0)} 核（采样间隔 {m['dt_s']:.1f} 秒）",
           f"GPU    {m['gpu_name'] or '未采到'}，整卡利用率 {display(m['gpu_pct'],0)}%，功耗 {display(m['gpu_power_w'])} W",
           f"显存   实验进程 {display(m['experiment_gpu_mib'],0)} MiB，整卡 {display(m['device_vram_mib'],0)} / {display(m['device_capacity_mib'],0)} MiB",
           f"内存   容器含页缓存 {display(m['cgroup_memory_gib'],2)} / {display(m['memory_limit_gib'],2)} GiB",
           f"磁盘   /data 剩余 {m['disk_free_gib']:.2f} / {m['disk_total_gib']:.2f} GiB，本轮输出含缓存 {state['output_gib']:.2f} GiB（每分钟更新）",
           f"自 {stamp(state['since'])} 采样，实验累计约 {state['cpu_seconds']/3600:.3f} 核时，采样最高实验显存 {display(state.get('peak_experiment_gpu_mib'),0)} MiB、实验 CPU {display(state.get('peak_experiment_cores'),2)} 核、容器内存 {display(state.get('peak_cgroup_memory_gib'),2)} GiB。",
           '', '当前实验进程     CPU 核   RSS GiB   GPU MiB   任务 / 场次']
    by_pid={r[10]:f'{r[0]} {r[5]} / 世界 {r[3]:03} / {r[6]}' for r in rows if r[9]=='运行中'}
    for pid,p in sorted(procs.items()):
        task=by_pid.get(pid,p['task'].replace(str(RUNTIME)+'/',''))
        text.append(f"{pid:10} {display(p['cores'],2):9} {p['rss']:9.2f} {display(p['gpu_mib'],0):9}   {task}")
    text+=['','已完成场次的资源记录（按实验组）',
           '逐场耗时相加含并行重叠；CPU 核时和 PyTorch 显存峰值只统计带逐场记录的结果。']
    for name in GROUPS:
        done=[r for r in rows if r[1]==name and r[9]=='已完成']
        if not done: continue
        measured=[r for r in done if r[14]!='']
        if not measured:
            text.append(f"{name}：{len(done)} 场，合计运行 {sum(r[13] for r in done)/60:.1f} 场分钟，暂无逐场 CPU 和显存记录。")
            continue
        peak=max(r[16] for r in measured)
        core_hours=sum(r[14] for r in measured)/3600
        text.append(f"{name}：{len(done)} 场，合计运行 {sum(r[13] for r in done)/60:.1f} 场分钟，"
                    f"CPU/显存记录 {len(measured)}/{len(done)} 场，"
                    f"已测 CPU {core_hours:.3f} 核时，单进程峰值保留显存 {display(peak)} MiB。")
    metrics_path=BASE/'cmu_history_dose/holdout/summary.json'
    dose=load_json(metrics_path) if metrics_path.exists() else {}
    cmu=dose.get('RC3_minus_I3_deg',{}).get('flights','进行中')
    intake=BASE/'intake/flight_manifest.csv'
    with intake.open() as f: audited=Counter(row['source'] for row in csv.DictReader(f))
    with (BASE/'q1_frontier/F_scores.csv').open() as f: fdone=sum(1 for r in csv.DictReader(f))
    rcdone=(BASE/'q1_frontier/RC_scores.csv')
    nrc=len(list((BASE/'remaining_queue/ancillary').glob('grid-*.json')))
    if rcdone.is_file():
        with rcdone.open() as f:nrc=sum(1 for r in csv.DictReader(f))
    prefetch=RUNTIME/'prefetch_validation.jsonl'
    prepared=sum(1 for line in prefetch.open() if line.strip()) if prefetch.is_file() else 0
    warnings=[p for p in (BASE/'remaining_queue/error.json',) if p.is_file()]
    if warnings:
        text+=['','调度异常']+[str(load_json(p)) for p in warnings]
    text+=['','配套工作',f"T 工况验证世界缓存已备 {1+prepared}/20 个（含启动验证已用的世界 0），预生成单列。",f"训练 / 校准：VECTOR 种子 7 中心权重 {'已落盘' if (BASE/'weights/vector7.pt').is_file() else '待完成'}，三个模型误差管 {sum((BASE/'weights'/(name+'_tubes.pkl')).is_file() for name in MODELS)}/3 套。",
           f"前沿评分 F {fdone}/360 格，RC 已完成 {nrc}/250 格。原归档复现 {', '.join(load_json(BASE/'reproduction/summary.json'))}。",
           f"留出的独立确认世界共 70 个，三种误差管共用同一模型的起始中心。新结算的完整服务窗复核 {sum((RUNTIME/'validation_replay'/(name+'_verified.json')).is_file() for name in MODELS)}/3 个模型，每模型两场。",
           f"真实航迹审计清单含 CMU {audited['CMU']}、PX4 {audited['PX4']} 条；CMU 留出集历史剂量对照 {cmu} 个共同架次。",
           '完整启动验证 6 场。3 场原生管已计入总表，另 3 场 LightGBM 启动验证单列。开发重放、资源复核和真实航迹另计。',
           '',f'逐场明细  {DETAIL}',f'资源时间序列  {HISTORY}',
           '逐场 CSV 可按状态筛选。空白的 CPU 秒与显存峰值表示该场缺少监测记录。显存峰值是 PyTorch 进程内口径，设备总量以 nvidia-smi 为准。']
    return '\n'.join(text)+'\n',prev

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--publish',action='store_true');a=ap.parse_args()
    prev=None
    if a.publish:
        lock=(RUNTIME/'full_progress.lock').open('a')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=load_json(STATE) if STATE.exists() else {}
    while True:
        content,prev=snapshot(prev,state,a.publish)
        if not a.publish:
            print(content,end='');break
        atomic(VIEW,content)
        time.sleep(5)
