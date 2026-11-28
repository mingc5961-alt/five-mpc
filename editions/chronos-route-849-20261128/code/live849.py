"""Live task ledger. Task identities and completion proofs are independent of process count."""
import datetime,json,os,sys,time
from pathlib import Path
O=Path(__file__).resolve().parents[1];D=O/'jobs';D.mkdir(exist_ok=True)
def write(p,x):
    tmp=p.with_suffix('.'+str(os.getpid())+'.tmp');tmp.write_text(x);os.replace(tmp,p)
def save(key,**fields):write(D/(key+'.json'),json.dumps(fields))
def start(key):save(key,pid=os.getpid(),start=Path('/proc/self/stat').read_text().split()[21])
def finish(key,receipt):save(key,receipt=str(receipt))
def proc(pid):
    try:return Path(f'/proc/{pid}/stat').read_text().split()[21]
    except OSError:return None
def openworkers():
    # Initial smoke was submitted before the live ledger. An open forecast database
    # in an endpoint worker proves that a world has been entered.
    n=0
    for p in Path('/proc').glob('[0-9]*'):
        try:
            cmd=(p/'cmdline').read_bytes()
            if b'code/chronos_recipients.py\x00' not in cmd:continue
            if any('cache/development/' in os.readlink(fd) and os.readlink(fd).endswith('.sqlite') for fd in (p/'fd').iterdir()):n+=1
        except (FileNotFoundError,PermissionError,ProcessLookupError):pass
    return n
def rows():
    groups=[
      ('pre','归档、资产和旧开发锚点',2),('s1','初核 E1 世界面板',1),('s2','初核 E2 闭环',10),
      ('lifeinputs','风险头拟合及校准输入世界',32),('lifehead','独立拟合并校准的风险头',4),
      ('labels','训练反事实短分支',4096),('val','验证反事实短分支',256),('devcf','开发诊断反事实短分支',256),('value','价值头',2),
      ('dev1','开发 E1 四倍率面板（按世界）',8),('dev2','开发 E2 闭环（含四倍率）',320),('freeze','设计冻结',1),
      ('f1','终核 E1 世界面板',1),('f2','终核 E2 闭环',10),
      ('c1','留出 E1 面板（六臂共用一面板）',48),('c2','留出 E2 闭环（共同倍率）',480),('report','配对汇总、图表和报告',1)]
    if (O/'design/frozen.json').exists():
        frozen=json.loads((O/'design/frozen.json').read_text())
        groups=[(g,label,frozen['n'] if g=='c1' else len(frozen['base_endpoints']) if g=='c2' else total) for g,label,total in groups]
        groups.append(('extra','冻结的额外倍率闭环',len(frozen['extra_endpoints'])))
    records={}
    for f in D.glob('*.json'):
        try:
            d=json.loads(f.read_text());group=f.stem.split('_')[0]
            comp=bool(d.get('receipt') and Path(d['receipt']).is_file())
            if comp or ('pid'in d and proc(d['pid'])==d['start']):
                pair=records.setdefault(group,[0,0]);pair[0 if comp else 1]+=1
        except (OSError,ValueError):pass
    records['pre']=[int((O/'manifest.json').exists() and (O/'logs/archive_verify.log').is_file() and 'OK:' in (O/'logs/archive_verify.log').read_text())+int((O/'audits/legacy.json').exists()),0]
    if 's1' not in records:
        ongoing=any(b'code/prefix849.py\x00development\x001400' in readcmd(p) for p in Path('/proc').glob('[0-9]*'))
        records['s1']=[0,int(ongoing)]
    smoke=list((O/'E2/receipts/smoke').glob('*.json'))
    records['s2']=[len(smoke),min(10-len(smoke),max(openworkers(),records.get('s2',[0,0])[1]))]
    return [(label,total,*records.get(g,[0,0])) for g,label,total in groups]
def readcmd(p):
    try:return (p/'cmdline').read_bytes()
    except OSError:return b''
_cpu_sample=None
def refresh():
    global _cpu_sample
    data=rows();tot=sum(r[1] for r in data);don=sum(r[2] for r in data);run=sum(r[3] for r in data)
    ex=[r for r in data if any(x in r[0] for x in ('E1','E2','短分支','额外倍率'))]
    et,ed,er=[sum(r[i] for r in ex) for i in (1,2,3)]
    numerator,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    maxcores=int(numerator)//int(period)
    stat=dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    sample=(time.monotonic(),int(stat['usage_usec'])*1.e-6)
    per=f"配额 {maxcores} 核"
    if _cpu_sample:
        cores=(sample[1]-_cpu_sample[1])/(sample[0]-_cpu_sample[0])
        per+=f'，最近 5 秒实际使用 {cores:.1f} 核（{100*cores/maxcores:.0f}%）'
    _cpu_sample=sample
    txt=['Chronos-849 实验实时进度',datetime.datetime.now().astimezone().isoformat(timespec='seconds')+'   每 5 秒刷新','',
        f'全部实验工作单元 {tot:,}    已完成 {don:,}    正在运行 {run:,}    排队 {tot-don-run:,}',
        f'其中评分实验 {et:,}    已完成 {ed:,}    正在运行 {er:,}    排队 {et-ed-er:,}',
        '评分实验包含 180 秒闭环、4.464 秒短分支及 E1 世界面板。工作单元另含拟合输入、模型训练、核验与交付。',
        f'服务器 CPU {per}。单个工作单元内可以有额外并行分片。',
        '运行中按已领取的实验计，包含等待相同付费前缀的任务，因此可多于 48。排队包含等待前置核验和冻结的任务。完成以通过校验的回执为准。',
        ('确认规模及额外倍率已经冻结，见下方各阶段。' if (O/'design/frozen.json').exists() else '正式确认暂列 48 个物理世界、480 个共同倍率闭环。冻结时按开发方差与重访覆盖复核样本量，额外倍率同时计入。'),
        '短分支是一次 4.464 秒反事实仿真。全程闭环是一次 180 秒仿真。两者分别列示。','',
        '   计划    完成    运行    排队   实验 / 阶段']
    for label,total,done,active in data:txt.append(f'{total:7,d} {done:7,d} {active:7,d} {total-done-active:7,d}   {label}')
    pref=O/'E1/prefix_index/development/1400'
    if (pref/'receipt.json').exists():detail=json.loads((pref/'receipt.json').read_text())['prefixes']
    else:detail=len(list(pref.glob('*.part')))
    workers=[]
    for path in (O/'workers').glob('*'):
        if path.name.isdigit() and proc(int(path.name)):
            try:workers.append(path.read_text())
            except OSError:pass
    primary_workers=sum('反事实主计算' in x for x in workers)
    early_closed=sum('开发闭环提前运行' in x for x in workers)
    derived=len(workers)-primary_workers-early_closed
    txt+=['',f'并行明细：额外前缀计算分片 {derived} 个，反事实世界主进程 {primary_workers} 个，提前放行开发闭环 {early_closed} 个。闭环与候选分支已计入上表。',f'开发世界 1400 已落盘的 E1 前缀 {detail} 个。',f'结果根目录 {O}']
    fixes=[]
    for f in (O/'active_closed').glob('*/*.json'):
        try:
            d=json.loads(f.read_text())
            if proc(d['pid']) and not (O/'E2/receipts'/f.parent.name/f.name).exists():fixes.append(d['completed_fixes'])
        except (OSError,ValueError):pass
    if fixes:txt+=['',f'有逐笔心跳的 {len(fixes)} 个运行中闭环，最近一次落盘合计 {sum(fixes):,}/{849*len(fixes):,} 笔。每 64 笔更新。']
    if (O/'logs/status.txt').exists():txt+=['',(O/'logs/status.txt').read_text().strip()]
    state=lambda test:'已通过' if test else '待完成'
    pick={name:(target,complete,active) for name,target,complete,active in data}
    def stages(*names):
        cells=[pick[n] for n in names if n in pick];n=sum(r[0] for r in cells);done=sum(r[1] for r in cells);running=sum(r[2] for r in cells)
        return f'{done}/{n}，运行中 {running}'
    r=[('R00','归档、环境与权重',state((O/'manifest.json').exists()),'manifest.json'),
       ('R01','原预约表、旧锚点与新旧选人入口',state((O/'audits/smoke.json').exists()),'audits/smoke.json'),
       ('R02','E1 初始六臂逐行对齐',stages('初核 E1 世界面板'),'E1/scores/smoke/1400.json'),
       ('R03','E1 风险头与独立校准',state((O/'audits/life_fit.json').exists()),'audits/life_fit.json'),
       ('R04','E2 八臂及固定 DQN 三个种子',stages('初核 E2 闭环'),'audits/smoke.json'),
       ('R05','反事实训练、验证、开发诊断与两价值头',stages('训练反事实短分支','验证反事实短分支','开发诊断反事实短分支','价值头'),'E2/labels/'),
       ('R06','开发网格、冻结及最终单世界核验',stages('开发 E1 四倍率面板（按世界）','开发 E2 闭环（含四倍率）','设计冻结','终核 E1 世界面板','终核 E2 闭环'),'design/'),
       ('R07','留出 E1',stages('留出 E1 面板（六臂共用一面板）'),'E1/scores/confirmation/'),
       ('R08','留出 E2',stages('留出 E2 闭环（共同倍率）','冻结的额外倍率闭环'),'E2/receipts/confirmation/'),
       ('R09','配对汇总、图、表及报告',stages('配对汇总、图表和报告'),'REPORT.md')]
    md=['# Chronos-849 服务器实验跟踪','','更新时间 '+datetime.datetime.now().astimezone().isoformat(timespec='seconds'),'','|编号|工作单元|状态|证据位置|','|---|---|---|---|']
    for tag,label,stat,path in r:md.append(f'|{tag}|{label}|{stat}|'+(f'[{path}]({path})' if (O/path).exists() else path)+'|')
    write(O/'EXPERIMENT_TRACKER.md','\n'.join(md)+'\n')
    write(O/'实时进度.txt','\n'.join(txt)+'\n')
if __name__=='__main__':
    while True:
        refresh()
        if (O/'REPORT.md').exists() and (D/'report_final.json').is_file():break
        time.sleep(5)
