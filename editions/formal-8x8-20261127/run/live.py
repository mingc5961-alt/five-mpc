import csv,io,datetime,os,collections,json,time,shutil
from pathlib import Path
from base import *
EXPORT=Path('/mnt/data')
def put(p,b):
    q=p.with_suffix('.tmp');q.write_bytes(b);q.replace(p)
def clock():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def publish(tasks,state,active,notes,attempts,started):
    now=time.time();byphase={};counts=collections.Counter();rows=[]
    for t in tasks:
        k=t['id'];s=state.get(k,'pending');desc=''
        if s=='pending':
            failed=[q for q in t['deps'] if state.get(q) in ('failed','blocked')]
            if failed:s='blocked';desc=failed[0]
            elif any(state.get(q)!='done' for q in t['deps']):s='waiting'
            else:s='ready'
        counts[s]+=1;ph=byphase.setdefault(t['batch'],collections.Counter());ph[s]+=1
        rows.append([k,t['batch'],t['resource'],s,attempts.get(k,0),round(now-active[k],1) if k in active else '',notes.get(k,desc)])
    snap=dict(updated=clock(),pid=os.getpid(),elapsed_s=int(now-started),total=len(tasks),counts=counts,phases=byphase,running=[r for r in rows if r[3]=='running'],recent_errors=[r for r in rows if r[3]=='failed'][:20],memory_GiB=round(int(Path('/sys/fs/cgroup/memory.current').read_text())/2**30,2),free_GiB=round(shutil.disk_usage(BASE).free/2**30,1))
    js(BASE/'live/status.json',snap);js(EXPORT/'RouteClock_status.json',snap)
    buf=io.StringIO();w=csv.writer(buf);w.writerow(['任务ID','批次','资源','状态','尝试','运行秒','最近错误或依赖']);w.writerows(rows)
    data=buf.getvalue().encode('utf-8-sig');put(BASE/'live/tasks.csv',data);put(EXPORT/'RouteClock_tasks.csv',data)
    symbols={'done':'完成','running':'运行','ready':'可执行','waiting':'等前置','failed':'失败','blocked':'前置失败'}
    def ct(c,s):return c.get(s,0)
    tr=''.join('<tr><td>'+g+'</td>'+''.join(f'<td>{ct(v,s)}</td>' for s in symbols)+'</tr>' for g,v in sorted(byphase.items()))
    import html
    lst=''.join('<li><code>'+html.escape(r[0])+'</code> · '+str(r[5])+' s</li>' for r in snap['running'])
    err=''.join('<li>'+html.escape(r[0]+' '+r[-1])+'</li>' for r in snap['recent_errors'])
    page=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><meta http-equiv="refresh" content="10"><title>RouteClock 实时进度</title><style>body{{max-width:1040px;margin:32px auto;padding:0 16px;font:17px/1.55 system-ui;color:#14354b}}table{{border-collapse:collapse;width:100%}}td,th{{padding:6px;text-align:left;border-bottom:1px solid #ccc}}progress{{width:100%}}</style><h1>RouteClock 实验队列</h1><p>服务器心跳 {snap['updated']} · PID {os.getpid()}</p><progress max="{len(tasks)}" value="{counts['done']}"></progress><p>完成 {counts['done']} / {len(tasks)} · 正在运行 {counts['running']} · 失败 {counts['failed']} · 内存 {snap['memory_GiB']} GiB · 剩余磁盘 {snap['free_GiB']} GiB</p><table><tr><th>批次</th>{''.join('<th>'+v+'</th>' for v in symbols.values())}</tr>{tr}</table><h2>正在运行</h2><ul>{lst}</ul><h2>最近失败</h2><ul>{err}</ul><p><a href="tasks.csv">全部任务明细 CSV</a> · <a href="status.json">状态 JSON</a></p><p>网页每10秒重读服务器文件。请检查心跳时间。下载的离线副本只含下载时的状态。</p></html>'''
    put(BASE/'live/index.html',page.encode());put(EXPORT/'RouteClock_live.html',page.replace('href="tasks.csv"','href="RouteClock_tasks.csv"').replace('href="status.json"','href="RouteClock_status.json"').encode())
