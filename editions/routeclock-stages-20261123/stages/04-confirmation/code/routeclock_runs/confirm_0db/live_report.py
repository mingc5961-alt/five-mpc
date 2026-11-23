"""Render the C3 and synthetic queues from durable cell receipts."""
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
import shutil
import subprocess
import time

MODELS=('Chronos-2','VECTOR','RAFT')
STAGES={'baselines':'正式基线','development':'误差管开发','confirmation':'新世界确认','amovfly':'AMOVFLY 实飞轨迹回放'}
CACHE={}

def stamp(value):
    return datetime.fromtimestamp(value).astimezone().strftime('%m-%d %H:%M:%S')

def duration(seconds):
    minutes,sec=divmod(int(max(0,seconds)),60);h,m=divmod(minutes,60)
    return f'{h} 小时 {m:02d} 分 {sec:02d} 秒'

def ratio(n,total):
    return f'{n}/{total}（{100*n/total:.1f}%）' if total else f'{n}/0'

def compact(p,stage):
    s=p.stat()
    if p not in CACHE or CACHE[p][0]!=s.st_mtime_ns:
        v=json.loads(p.read_text());b=v.get('frozen_config',v.get('contract',v))
        CACHE[p]=(s.st_mtime_ns,(stage,b['model'],b.get('budget',b.get('condition')),b.get('seed',0),
                    b.get('tube','default'),b.get('world',b.get('flight_id')),s.st_mtime,v['elapsed_s']))
    return CACHE[p][1]

def label(tag):
    if tag=='baselines':return 'I/F/RC 正式基线（世界 00–39）'
    if tag=='summary':return '全量核验及配对统计'
    kind,model,*rest=tag.split('_')
    if kind=='select':return model+' 误差管选型'
    if kind=='flight':return model+' 实飞回放（种子 7，三档预算）'
    if kind=='dev':return f'{model} 开发，种子 7，世界 {int(rest[0]):02d}–{int(rest[0])+4:02d}'
    return f'{model} 确认，种子 {rest[0]}，世界 {int(rest[1]):02d}–{int(rest[1])+19:02d}'

def render(root,batch,finished,active,errors,cores,elapsed,sample_dt):
    now=time.time(); rows=[];counts=Counter()
    wins=json.loads((root/'work/flight_windows.json').read_text())
    totals=dict(baselines=360,development=180,confirmation=1080,amovfly=15*len(wins))
    for stage in STAGES:
        for p in (root/'results'/stage).rglob('complete.json'):
            r=compact(p,stage);rows.append(r);counts[stage]+=1
    def number(stage,model=None,budget=None,seed=None,tube=None):
        return sum(r[0]==stage and all(x is None or x==y for x,y in zip((model,budget,seed,tube),r[1:5])) for r in rows)
    n=len(rows);total=sum(totals.values())
    resume=(root/'state/migration_resume.lock').stat().st_mtime
    cg=Path('/sys/fs/cgroup');q,per=map(int,(cg/'cpu.max').read_text().split())
    gpu=subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True)
    disk=shutil.disk_usage(root);gib=2**30
    lines=['RouteClock　0 dB 实验实时进度',f'更新于 {datetime.now().astimezone():%Y-%m-%d %H:%M:%S %Z}　约每 10 秒自动刷新',
           f'正式闭环　{ratio(n,total)} 场　剩余 {total-n} 场',
           f'本机接续于 {stamp(resume)}，已过 {duration(now-resume)}；接续时 248 场，本机新增 {n-248} 场。',
           '', '各阶段完成情况']
    for stage,title in STAGES.items():
        lines.append(f'  {title}　{ratio(counts[stage],totals[stage])} 场　剩余 {totals[stage]-counts[stage]} 场')
    summary=root/'results/summary/complete.json'
    lines.append('  配对统计　'+('已生成，详见 results/summary' if summary.exists() else '等待四个阶段全部落盘后执行；逐世界先平均种子，再作 5,000 次配对重抽，实飞按架次配对。'))
    base=root/'state'
    assets=('restore_code','initial_audit','amovfly_index','amovfly_quality')
    pilots=('i_0db_pilot','f_0db_pilot','rc_0db_pilot')
    lines+=['','资产、训练与离线校准',
            f'  准备 {ratio(sum((base/(s+".done")).exists() for s in assets),4)} 项（代码恢复、归档核查、轨迹下载与初筛）；0 dB 试跑 {ratio(sum((base/(s+".done")).exists() for s in pilots),3)} 场（I/F/RC）。']
    for model in MODELS:
        stats=[]; fitted=[]
        for seed in (7,11,19):
            p=root/'training'/model/str(seed)/'complete.json'
            if p.exists():
                v=json.loads(p.read_text());stats.append(f"{seed}={v['validation_rad']:.5f} rad（{v['epochs']} 轮）")
            if (root/'tubes'/model/str(seed)/'offline.json').exists():fitted.append(str(seed))
        lines.append(f'  {model}　中心权重 {len(stats)}/3　验证误差按种子　'+ '，'.join(stats))
        lines.append('    误差管离线校准　'+', '.join(fitted)+f' 已完成（{len(fitted)}/3 个种子）')
    lines+=['','正式基线 · 40 个共同世界 × T/M/L × 3 种方法']
    for m in ('I','F','RC'):
        lines.append(f'  {m:<3}　{ratio(number("baselines",m),120)} 场　'+ '　'.join(f'{b} {number("baselines",m,b)}/40' for b in 'TML'))
    lines+=['','误差管开发与新世界确认 · 三个模型分别选型']
    for m in MODELS:
        tube_dir=root/'tubes'/m
        tubes=json.loads((tube_dir/'7/offline.json').read_text())['candidates']
        selection=tube_dir/'selection.json'
        select=json.loads(selection.read_text())['tube'] if selection.exists() else '等待本模型 60 场开发齐全'
        lines.append(f'  {m}　开发 {ratio(number("development",m),60)}；确认 {ratio(number("confirmation",m),360)}')
        lines.append('    候选管　'+'　'.join(f'{k} {number("development",m,tube=k)}/30' for k in tubes)+'　选型　'+select)
        lines.append('    新世界确认　'+'　'.join(f'种子 {s} {number("confirmation",m,seed=s)}/120' for s in (7,11,19)))
    lines+=['','AMOVFLY · 单机无线链路模拟',
            f'  270 架次 → 127 个质量候选 → 冻结 {len(wins)} 架次（'+ '，'.join(f'{k} {v}' for k,v in Counter(w['route'] for w in wins).items())+'）。',
            '  每架次 180 秒预热、180 秒计分，每种方法跑三档预算。']
    for m in ('I','RC',*MODELS):
        completed=Counter(r[5] for r in rows if r[0]=='amovfly' and r[1]==m)
        lines.append(f'  {m:<9}　{ratio(number("amovfly",m),len(wins)*3)} 场　三档齐全 {sum(v==3 for v in completed.values())}/{len(wins)} 架次')
    lines+=['','正在运行与后续排程']
    waiting=[tag for tag in batch if tag not in active and tag not in finished]
    lines.append(f'  批任务　运行 {len(active)}，已结束 {len(finished)}/{len(batch)}，等待 {len(waiting)}。')
    completed_keys={r[:6] for r in rows}
    for tag in active:
        lines.append('  运行　'+label(tag))
        parts=tag.split('_')
        pending=None
        if parts[0] in ('dev','confirm'):
            kind,m=parts[:2]
            stage='development' if kind=='dev' else 'confirmation'
            lo=int(parts[-1]);sz=5 if kind=='dev' else 20;seed=7 if kind=='dev' else int(parts[2])
            if kind=='dev':
                tubes=json.loads((root/'tubes'/m/'7/offline.json').read_text())['candidates']
            else:
                tubes=[json.loads((root/'tubes'/m/'selection.json').read_text())['tube']]
            pending=next((f'世界 {u:02d}，预算 {b}，误差管 {t}' for u in range(lo,lo+sz)
                          for b in 'TML' for t in tubes if (stage,m,b,seed,t,u) not in completed_keys),None)
        elif parts[0]=='flight':
            m=parts[1]
            done={(r[5],r[2]) for r in rows if r[0]=='amovfly' and r[1]==m}
            pending=next((f'架次 {w["flight_id"]}，预算 {b}' for w in wins for b in 'TML' if (w['flight_id'],b) not in done),None)
        if pending:lines.append('    批内首个未落盘场次　'+pending)
    if 'baselines' in active:
        folders=sorted(d.name for d in (root/'results/baselines').iterdir() if d.is_dir() and not (d/'complete.json').exists())
        lines.append('    已建目录、待落盘的基线场次　'+('、'.join(folders) or '等待工作进程分配'))
    for title,prefix in [('开发','dev_'),('选型','select_'),('确认','confirm_'),('实飞','flight_')]:
        pending=[j for j in waiting if j.startswith(prefix)]
        if pending:
            conditions={'开发':'空闲工作进程按世界分批领取','选型':'对应模型 60 场开发齐全后择优',
                        '确认':'对应模型冻结误差管后释放；同种子的校准包在批间共享',
                        '实飞':'I/RC 可直接领取，先进模型等待各自选型冻结'}
            lines.append(f'  等待 · {title}（{len(pending)} 批）　'+conditions[title])
            for i in range(0,len(pending),3):
                lines.append('    '+'；'.join(label(j) for j in pending[i:i+3]))
    if 'summary' in waiting:lines.append('  等待 · 统计　全量验收后生成逐场表、各方法均值、配对差值与区间。')
    lines+=['','最近落盘的 8 场（完成时刻；耗时为单场墙钟时间）']
    for s,m,b,seed,t,u,mtime,wall in sorted(rows,key=lambda r:r[6],reverse=True)[:8]:
        details=f'架次 {u}' if s=='amovfly' else f'世界 {int(u):02d}'
        options=f'，种子 {seed}，{t}' if seed else ''
        lines.append(f'  {stamp(mtime)}　{STAGES[s]}　{m}　{details}，预算 {b}{options}　{wall:.1f} 秒')
    lo=max(resume,now-900);recent=sum(r[6]>=lo for r in rows);minutes=(now-lo)/60
    lines+=['','本机资源与运行速度',
            f'  CPU　{cores:.2f}/{q/per:.2f} 核（近 {sample_dt:.1f} 秒均值）' if sample_dt>=1 else f'  CPU　等待首个采样区间，配额 {q/per:.2f} 核',
            f'  内存　{int((cg/"memory.current").read_text())/gib:.1f}/{int((cg/"memory.max").read_text())/gib:.1f} GiB（含缓存）',
            f'  磁盘　已用 {disk.used/gib:.1f} GiB，可用 {disk.free/gib:.1f} GiB，数据卷共 {disk.total/gib:.1f} GiB']
    if gpu.returncode==0:
        name,util,used,cap=map(str.strip,gpu.stdout.splitlines()[0].split(','))
        lines.append(f'  GPU　{name}　利用率 {util}%　显存 {used}/{cap} MiB（瞬时采样）')
    lines.append(f'  近 {minutes:.1f} 分钟新落盘 {recent} 场，合计 {recent/max(minutes,.01):.2f} 场/分。')
    lines+=['','重试与产物位置','  当前调度周期重试次数　'+('，'.join(f'{label(k)} {v} 次' for k,v in errors.items()) or '0'),
            '  逐场结果　'+str(root/'results'),'  运行日志　'+str(root/'logs'),'  窗口清单　'+str(root/'work/flight_windows.json')]
    return '\n'.join(lines)+'\n'
