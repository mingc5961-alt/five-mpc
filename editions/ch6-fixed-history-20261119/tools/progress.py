from pathlib import Path
import json,time
r=Path(__file__).resolve().parent
plan=json.loads((r/'manifest.json').read_text())
while True:
 s=json.loads((r/'status.json').read_text())
 series={}
 for j in plan['new']:
  key=('基点数值复核' if j['policy']=='baseline_check' else '固定动作 F' if j['policy']=='fixed_action' else '成本压力' if j['cost_ms']>3.1 else '调度器' if j['scheduler']!='value' else 'Chronos' if j['method'] in ('C','CR_raw') else '共享预算')
  g=series.setdefault(key,[0,0]);g[0]+=int((Path(j['out'])/'completed.json').exists());g[1]+=1
 lines=['# 固定历史补充实验','',time.strftime('%Y-%m-%d %H:%M:%S %Z'),'',f"通过审计 {sum(g[0] for g in series.values())}/581，新任务运行中 {len(s.get('running',[]))}，失败 {len(s.get('failed',[]))}。另复用基点 140 场。",'','| 实验 | 已完成 | 计划 |','|---|---:|---:|']+[f'| {k} | {v[0]} | {v[1]} |' for k,v in series.items()]
 (r/'进度.tmp').write_text('\n'.join(lines)+'\n');(r/'进度.tmp').replace(r/'实验进度.md')
 if s['stage']=='simulation_complete':break
 time.sleep(20)
