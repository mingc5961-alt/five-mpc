"""Read-only scientific-output checks and comparisons for the frozen campaign."""
import csv
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import pandas as pd

SOURCE=Path('/data/chen/routeclock_server')
FIVE=SOURCE/'routeclock_five_20261115'

def read(path): return json.loads(Path(path).read_text())

def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def digest(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f: return list(csv.DictReader(f))

def check(condition,message):
    if not condition: raise ValueError(message)

def load_data(path,split,index):
    sys.path.insert(0,str(FIVE))
    with path.open('rb') as f: data=pickle.load(f)
    check(len(data)==(1 if split=='calibration' else 3),f'{path}: anchor count')
    task=f'offline/{split}/{index}'
    for a in data:
        check(a['task']==task,f'{path}: wrong task identity')
        pkg=a['package']
        for center in ('I','P','C','CR_raw'):
            check(center in pkg.centers,f'{path}: missing {center}')
            check(np.isfinite(pkg.centers[center]).all(),f'{path}: nonfinite {center} prediction')
        check(np.isfinite(a['truth']).all(),f'{path}: nonfinite truth')
        check(a['truth'].shape==(len(a['truth_h']),3),f'{path}: truth shape')
    return data

def verify_job(job,root,source=SOURCE):
    root=Path(root);kind=job['kind'];meta=job['meta']
    if kind=='data':
        anchors=0
        for i in range(meta['start'],meta['stop']):
            anchors+=len(load_data(root/f'five_data/{meta["split"]}/{i:04}.pkl',meta['split'],i))
        return dict(tasks=meta['stop']-meta['start'],anchors=anchors)
    if kind=='timing':
        data=read(root/'timing/day1/times.json')
        check(len(data)==11,'Expected six offline and five network day-one timings')
        check(all(r['seconds']>0 for r in data),'Invalid timing')
        return dict(timed_tasks=len(data))
    if kind=='train':
        sys.path.insert(0,str(FIVE))
        folder=root/f'five/{meta["head"]}_{meta["seed"]}'
        with (folder/'bundle.pkl').open('rb') as f: b=pickle.load(f)
        check(b['counts']==[1200,400,400,400],'Training split counts differ from protocol')
        check((b['kind'],b['seed'])==(meta['head'],meta['seed']),'Model identity mismatch')
        check(set(b['heads'])=={'I','P','RC','C','CR_raw'},'Five independent error heads required')
        for name,h in b['heads'].items():
            check(len(h.calibration_scores)==400,f'{name}: independent calibration count')
            check(np.isfinite(h.factor),f'{name}: nonfinite calibration factor')
        expected={'forecast.csv':42000,'gate_events.csv':4800,'deadlines.csv':9600}
        for name,count in expected.items(): check(len(rows(folder/name))==count,f'{name}: expected {count} rows')
        findings=[]
        for name,g in b['gates'].items():
            if not g.fit_status.get('success'): findings.append(dict(gate=name,optimizer=g.fit_status))
        return dict(bundle_sha256=digest(folder/'bundle.pkl'),counts=b['counts'],findings=findings)
    if kind=='network':
        folder=root/f'five/network/{meta["head"]}_{meta["seed"]}/{meta["block"]}/{meta["index"]:03}'
        data=read(folder/'completed.json');metrics=read(folder/'metrics.json')
        expected_policy=meta['method'] if meta['mode']=='joint' else meta['mode']
        check((data['mission'],data['policy'],data['block'])==(meta['mission'],expected_policy,meta['block']),'Network job identity mismatch')
        check(data['target_reuse']==meta['reuse'],'Reuse setting mismatch')
        check(data['N']==16 and data['warmup_s']==180 and data['service_s']==180,'Formal network scale mismatch')
        check(abs(data['accounted_s']-180)<1e-6,'Network time ledger mismatch')
        check(data['net_mbps']==metrics['net_mbps'],'Completion marker disagrees with metrics')
        if meta['mission']==0:
            for name in ['events.json','predictions.json','choices.json']: check((folder/name).exists(),f'Missing {name}')
        return dict(completed_sha256=digest(folder/'completed.json'),net_mbps=data['net_mbps'],
                    hold_segments=data['hold_segments'],hold_failures=data['hold_failures'])
    if kind=='summary':
        folder=root/f'five/network/{meta["head"]}_{meta["seed"]}'
        expected=390 if (meta['head'],meta['seed'])==('lightgbm',7) else 100
        check(len(rows(folder/'jobs.csv'))==expected,'Incomplete network summary')
        check(len(rows(folder/'paired.csv'))>0,'Missing paired comparisons')
        return dict(jobs=expected)
    if kind=='mpc_train':
        folder=root/'mpc_train';manifest=read(folder/'models.json')
        for name,sha in manifest.items(): check(digest(folder/name)==sha,f'MPC model checksum mismatch: {name}')
        return dict(models=manifest,training=read(folder/'training.json'))
    if kind=='mpc_simulate':
        paths=list((root/'mpc_formal').glob('scene_*/*/metrics.json'))
        check(len(paths)==120,'Expected all 30 networks x 4 MPC policies')
        check(read(root/'mpc_formal/evaluation_models.json')==read(root/'mpc_train/models.json'),'MPC training/evaluation model mismatch')
        return dict(completed=120)
    if kind=='mpc_verify':
        data=read(root/'mpc_formal/verification.json');check(data['passed'],'MPC ledger verification failed');return data
    if kind in ('paired_run','paired_audit','paired_search'):
        file={'paired_run':'paired_events.csv','paired_audit':'refresh_value_audit.csv','paired_search':'search_trace.csv'}[kind]
        data=rows(root/'paired'/file)
        check(len(data)==143,f'{file}: expected 143 intervention points')
        check(len({(x['scene'],x['step']) for x in data})==143,f'{file}: duplicate intervention points')
        return dict(events=143)
    if kind=='paired_summary':
        check(len(rows(root/'paired/paired_with_audit.csv'))==143,'Paired audit merge incomplete')
        check(len(rows(root/'paired/summary.csv'))==24,'Paired descriptive groups incomplete')
        return dict(events=143)
    if kind=='final_report':
        data=read(root/'comparison/completion_audit.json');check(data['complete'],'Final completeness check failed');return data
    return dict(command_completed=True)

def audit_dataset(root):
    counts=read(FIVE/'configs/research.json')['data']['single_uav_tasks']
    inventory=[];tasks=set()
    for split,count in counts.items():
        files=list((root/'five_data'/split).glob('*.pkl'))
        check(len(files)==count,f'{split}: expected exactly {count} task files')
        for i in range(count):
            path=root/f'five_data/{split}/{i:04}.pkl';data=load_data(path,split,i)
            task=data[0]['task'];check(task not in tasks,'Task leaked across splits');tasks.add(task)
            inventory.append(dict(task=task,split=split,index=i,anchors=len(data),sha256=digest(path),bytes=path.stat().st_size))
    write(root/'dataset_inventory.json',inventory)
    write(root/'dataset_audit.json',dict(passed=True,tasks=len(tasks),anchors=sum(x['anchors'] for x in inventory),split_counts=counts,
        limitation='Validates route identities and namespaces; a separate geometry-hash audit was not added to the original protocol.'))
    print('Dataset verified:',len(tasks),'independent task IDs',flush=True)

def bootstrap(a,b):
    diff=np.asarray(a,float)-np.asarray(b,float);r=np.random.default_rng(271828)
    means=diff[r.integers(0,len(diff),(2000,len(diff)))].mean(axis=1)
    return float(diff.mean()),*map(float,np.quantile(means,[.025,.975]))

def comparison_report(root):
    out=root/'comparison';out.mkdir(exist_ok=True)
    plan=read(root/'plan.json')
    for name,sha in read(root/'delivery_manifest.json')['sha256'].items():
        check(digest(SOURCE/name)==sha,f'Original delivered file changed: {name}')
    check(read(root/'dataset_audit.json')['passed'],'Dataset audit missing')
    allrows=[];comparisons=[]
    metrics=['net_mbps','sensing_fraction','worst10_mbps','narrow_share','narrow_hold_s',
             'hold_failure_fraction','outage_data_fraction','unavailable_s']
    for head,seed in [('lightgbm',7),('tabpfn',7),('lightgbm',11),('lightgbm',19)]:
        folder=root/f'five/network/{head}_{seed}'
        data=[read(p) for p in sorted(folder.glob('*/*/completed.json'))]
        check(len(data)==(390 if (head,seed)==('lightgbm',7) else 100),'Incomplete comparison matrix')
        frame=pd.DataFrame(data);frame['head']=head;frame['seed']=seed;allrows.extend(frame.to_dict('records'))
        for (block,reuse),g in frame.groupby(['block','target_reuse'],dropna=False):
            if block=='actions': g=pd.concat([g,frame[(frame.block=='main')&(frame.policy=='RC')]])
            for metric in metrics:
                table=g.pivot(index='mission',columns='policy',values=metric)
                if 'RC' not in table: continue
                for method in table:
                    if method=='RC': continue
                    pair=table[['RC',method]].dropna()
                    if not len(pair):
                        comparisons.append(dict(head=head,seed=seed,block=block,reuse=reuse,baseline=method,metric=metric,paired_tasks=0,issue='No defined paired metric'))
                        continue
                    mean,low,high=bootstrap(pair.RC,pair[method])
                    comparisons.append(dict(head=head,seed=seed,block=block,reuse=reuse,baseline=method,metric=metric,
                        paired_tasks=len(pair),expected_pairs=20 if block in ('repeat','reuse') else 30,
                        difference=mean,ci95_low=low,ci95_high=high,
                        within_prespecified_2pp=(low>=-.02 and high<=.02) if metric=='hold_failure_fraction' else None))
    full=pd.DataFrame(allrows);full.to_csv(out/'all_network_jobs.csv',index=False)
    pairs=pd.DataFrame(comparisons);pairs.to_csv(out/'paired_network_comparisons.csv',index=False)
    full.groupby(['head','seed','block','target_reuse','policy'],dropna=False)[metrics].mean().to_csv(out/'network_means.csv')
    offline=[]
    for head,seed in [('lightgbm',7),('tabpfn',7),('lightgbm',11),('lightgbm',19)]:
        f=root/f'five/{head}_{seed}'
        d=pd.read_csv(f/'deadlines.csv')
        # pandas may infer an object column when duration rows have no coverage.
        d['covered']=d['covered'].map({True:1.,False:0.,'True':1.,'False':0.})
        # Average each task's anchors before population means; retain undefined direct-duration coverage.
        cols=['issued','early_failure','covered','mean_halfwidth_deg','approved_s','oracle_s']
        for (method,variant),g in d.groupby(['method','head']):
            route=g.groupby('task')[cols].mean(numeric_only=True)
            check(len(route)==400,'Offline testing must contain all 400 independent tasks')
            row=dict(error_model=head,seed=seed,method=method,variant=variant,tasks=400,anchors=len(g))
            row.update(issued_anchors=int(g.issued.sum()),early_failure_anchors=int(g.early_failure.sum()),
                early_failure_given_issued=float(g.early_failure.sum()/g.issued.sum()) if g.issued.sum() else None)
            row.update(route.mean().to_dict());offline.append(row)
    pd.DataFrame(offline).to_csv(out/'offline_validity_task_means.csv',index=False)
    mpc=pd.read_csv(root/'mpc_formal/network_results.csv');check(len(mpc)==120,'Incomplete MPC matrix')
    mp=[]
    for metric in ['Mbps','worst10_Mbps','sensing_pct','expired_s','over3_pct']:
        table=mpc.pivot(index='mission',columns='policy',values=metric)
        for method in ['rotation','myopic','age_mpc']:
            a=table[['joint_mpc',method]].dropna();check(len(a)==30,'Incomplete MPC pairs')
            mean,low,high=bootstrap(a.joint_mpc,a[method]);mp.append(dict(metric=metric,baseline=method,paired_tasks=30,difference=mean,ci95_low=low,ci95_high=high))
    pd.DataFrame(mp).to_csv(out/'mpc_paired_comparisons.csv',index=False)
    paired=pd.read_csv(root/'paired/summary.csv');check(len(rows(root/'paired/paired_with_audit.csv'))==143,'Paired reproduction incomplete')
    checks=dict(complete=True,single_uav_tasks=2400,five_network_jobs=len(full),mpc_network_jobs=len(mpc),paired_interventions=143,
        source_unchanged=True,original_protocol=plan['protocol'])
    write(out/'completion_audit.json',checks)
    lines=['RouteClock 正式实验结果与分析','',
        '原始系统模型和调度机制未改动。本报告只在全部固定任务完成后生成；未依输赢调整种子、阈值或基线。',
        '任务规模：2400 条单机航线、四组正式拟合、690 场五模型网络、120 场 MPC、143 个既定分叉。',
        '统计单位为独立航线或网络；网络区间采用固定种子 271828 的 2000 次配对 bootstrap。','',
        '主实验：LightGBM seed=7，RC 减各基线的净吞吐量差（Mbps）']
    main=pairs[(pairs['head']=='lightgbm')&(pairs.seed==7)&(pairs.block=='main')&(pairs.metric=='net_mbps')]
    for _,r in main.iterrows():
        conclusion='区间支持 RC 吞吐更高' if r.ci95_low>0 else ('区间支持 RC 吞吐更低' if r.ci95_high<0 else '区间跨过零，未证明差异方向')
        lines.append(f'{r.baseline}: 差值 {r.difference:.3f}，95% CI [{r.ci95_low:.3f}, {r.ci95_high:.3f}]，{int(r.paired_tasks)} 对；{conclusion}。')
    lines+=['','失效率等效：仅在差值的整个 95% 区间位于预设 [-0.02,0.02] 时记录为符合“相近失效率”；不以不显著代替等效。',
            '动作消融、复用比例、第二误差头与种子复核的全部正负差值见 paired_network_comparisons.csv。',
            '直接期限头没有角度管覆盖和半宽定义，相关空值保留；任何缺少成对指标的条目均保留 paired_tasks 和 issue。','',
            'MPC：joint_mpc 减基线的净吞吐量差（Mbps）']
    for r in mp:
        if r['metric']=='Mbps': lines.append(f"{r['baseline']}: {r['difference']:.3f}，95% CI [{r['ci95_low']:.3f}, {r['ci95_high']:.3f}]，30 对。")
    lines+=['','143 分叉复现：刷新减等待的实际交付差值（Mbit），同一网络内分叉相关，仅作描述统计。']
    for _,r in paired[paired.group=='all'].iterrows():
        lines.append(f'{r.horizon}: 均值 {r.mean_delta_Mbit:.4f}，中位数 {r.median_delta_Mbit:.4f}，刷新获胜 {int(r.refresh_wins)}/{int(r.n)}。')
    lines+=['','解释边界：本轮数据为固定模拟世界；仿真未把模型真实计算延迟扣入通信账本。预测超订、迟到、失效和无可用动作按原实现保留，不视为已经满足实时部署要求。',
            '工程改动清单见运行根目录 修改说明.txt；原始模型文件指纹与 source_hashes.json 对照。',
            'V2 新模型与学习调度尚不属于上述对照；只有满足用户的十小时条件并另外完成实现与验证，才能作为新增结果报告。']
    (out/'实验结果与分析.txt').write_text('\n'.join(lines)+'\n')
    print('Full comparison report completed.',flush=True)

if __name__=='__main__':
    action=sys.argv[1];root=Path(sys.argv[2])
    if action=='dataset': audit_dataset(root)
    elif action=='report': comparison_report(root)
    else: raise SystemExit('Expected dataset/report')
