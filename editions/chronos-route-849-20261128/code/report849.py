from design849 import *
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def ci(values):
    z=np.asarray(values,dtype=float)
    if not len(z):return [None,None,None,0]
    assert np.isfinite(z).all()
    if len(z)==1:return [float(z[0]),None,None,1]
    idx=np.random.default_rng(271127).integers(0,len(z),(20000,len(z)))
    lo,hi=np.quantile(z[idx].mean(1),[.025,.975])
    return [float(z.mean()),float(lo),float(hi),len(z)]
def csvout(df,path):
    atom(OUT/path,df.to_csv(index=False).encode())
def fmt(row,digits=3):
    x,lo,hi,n=row
    if x is None:return '无合格世界'
    if lo is None:return f'{x:.{digits}f}（1 世界，区间不可估）'
    return f'{x:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]'
def figsave(fig,name):
    for suffix in ('.png','.pdf'):fig.savefig(OUT/'figures'/(name+suffix),dpi=160,bbox_inches='tight')
    plt.close(fig)
def report():
    from live849 import start as mark,finish
    mark('report_final')
    v=subprocess.run([sys.executable,'-B','tools/verify.py'],cwd=ED,capture_output=True,check=True)
    atom(OUT/'logs/archive_verify_post.log',v.stdout+v.stderr)
    d=get(OUT/'design/frozen.json');check(d);assert get(OUT/'audits/final-smoke.json')['status']=='PASS'
    exp=set(tuple(v) for v in d['base_endpoints']+d['extra_endpoints']);ids=set(w['world'] for w in d['worlds'])
    from audit849 import audit_confirm_world
    from gate849 import poolmap
    poolmap(audit_confirm_world,sorted(ids),min(32,capacity()))
    audit_index=get(OUT/'audits.json');audit_index['confirmation']={str(k):sha_file(OUT/f'audits/confirmation/{k}.json') for k in sorted(ids)};put(OUT/'audits.json',audit_index)
    df=receipts('confirmation');assert set(zip(df.world,df.arm,df.seed,df.scale))==exp and len(df)==len(exp)
    assert set(df.world)==ids
    phys={m['world']:m['physical_sha256'] for m in d['worlds']}
    assert (df.physics==df.world.map(phys)).all()
    e1=life_summary('confirmation')
    scores=[];coverage=[];counts=[];byage=[];effective_worlds=set();effective_prefixes=set();all_prefixes=0
    for k in sorted(ids):
        p=OUT/f'E1/scores/confirmation/{k}.csv.gz';r=get(p.with_name(f'{k}.json'))
        assert r['sha256']==sha_file(p) and r['physics']==phys[k] and r['mismatches']==0
        all_prefixes+=r['issued'];d1=pd.read_csv(p);unique=d1[(d1.arm=='ROUTE')&(d1.scale==.75)]
        pp=primary(unique);valid=pp[(pp.permutable>0)&(pp.route_support>0)]
        if len(valid):effective_worlds.add(k);effective_prefixes.update((k,p) for p in valid.prefix)
        counts.append(dict(world=k,issued=r['issued'],primary_prefixes=pp.prefix.nunique(),primary_flights=pp.uid.nunique(),primary_queries=len(pp),permutable_prefixes=valid.prefix.nunique(),missing_reason='' if len(pp) else 'no eligible same-direction revisit cell'))
        q=d1[(d1.scale==.75)&(d1.h==0)]
        subsets={'all':q,'primary':primary(q),'first_pass':q[q.label=='first_pass'],'low_reuse':q[q.group=='no_reusable_path'],'detour_rejoin':q[q.label.isin(['detour','rejoin'])],'route_support':q[q.route_support>0],'no_route_support':q[q.route_support==0],'natural':q[q.natural==1]}
        for label,data in subsets.items():
            ix=data[data.arm=='ROUTE']
            coverage.append(dict(world=k,group=label,queries=len(ix),users=ix.uid.nunique(),prefixes=ix.prefix.nunique(),censored=float(ix.censored.mean()) if len(ix) else None,no_time=float((ix.time_support==0).mean()) if len(ix) else None,time_only=float(((ix.time_support>0)&(ix.route_support==0)).mean()) if len(ix) else None,permutable=float(ix.permutable.mean()) if len(ix) else None))
            for arm,v in data.groupby('arm'):
                cells=v.groupby(['n','age'])[['U','O']].mean()
                scores.append(dict(experiment='E1',world=k,physics=phys[k],stratum=label,arm=arm,scale=.75,eligible_cells=len(cells),**cells.mean().to_dict()))
        for (arm,age),v in primary(q).groupby(['arm','age']):
            byage.append(dict(world=k,arm=arm,age=age,U=v.groupby('n').U.mean().mean()))
    s2=df.groupby(['world','arm','scale']).mean(numeric_only=True).reset_index()
    nw=s2[s2.scale==.75].copy();nw['experiment']='E2';nw['stratum']='all';nw['physics']=nw.world.map(phys)
    csvout(pd.concat([pd.DataFrame(scores),nw],ignore_index=True),'world_scores.csv');csvout(df,'E2/endpoints.csv')
    csvout(e1,'E1/primary_grid.csv');csvout(pd.DataFrame(counts),'E1/primary_coverage.csv');csvout(pd.DataFrame(coverage),'E1/strata.csv')
    cover=pd.DataFrame(coverage);slices=[]
    for group,v in cover.groupby('group'):
        ok=v[v.queries>0]
        slices.append(dict(stratum=group,worlds=len(ok),prefixes=int(v.prefixes.sum()),queries=int(v.queries.sum()),aircraft_worlds=int(v.users.sum()),no_time=float(ok.no_time.mean()) if len(ok) else None,time_only=float(ok.time_only.mean()) if len(ok) else None,permutable=float(ok.permutable.mean()) if len(ok) else None,censored=float(ok.censored.mean()) if len(ok) else None))
    csvout(pd.DataFrame(slices),'E1/strata_summary.csv')
    cs=[];answers={}
    def contrast(family,label,col,vector):
        val=ci(vector);cs.append(dict(family=family,comparison=label,metric=col,mean=val[0],lo95=val[1],hi95=val[2],worlds=val[3]));answers[family,label,col]=val;return val
    def paired(tab,arm,other,metric):
        if tab.empty:return []
        z=tab.pivot(index='world',columns='arm',values=metric)
        return (z[arm]-z[other]).dropna().values if arm in z and other in z else []
    yes1={};yes2={}
    main=e1[e1.scale==.75];main2=nw
    for a in sorted(set(['TIME','SHUFFLE',d['strong_E1']])):
        u=contrast('E1',a,'missed_s_saved',paired(main,a,'ROUTE','U'))
        o=contrast('E1',a,'extra_overadmission_s',paired(main,'ROUTE',a,'O'))
        yes1[a]=bool(u[1] is not None and u[1]>0 and o[2]<.02)
    permok=len(effective_worlds)>=d['effective_shuffle_min_worlds'] and len(effective_prefixes)>=d['effective_shuffle_min_prefixes']
    e1pass=all(yes1.values()) and permok
    def comp2(label,a,b,tab=main2,family='E2'):
        g=contrast(family,label,'delta_G_Mbps',paired(tab,a,b,'G'));r=contrast(family,label,'delta_R_pp',paired(tab,a,b,'R'))
        return bool(g[1] is not None and g[1]>0 and r[2]<.5)
    for a in sorted(set(nw.arm)-{'VALUE-route'}):yes2[a]=comp2(a,'VALUE-route',a)
    alt=comp2('nonroute_vs_AoI',d['strong_nonroute_scheduling'],d['strong_AoI'])
    absolute={a:contrast('E2-absolute',a,'R_pct',v.R) for a,v in nw.groupby('arm')}
    tables=[]
    for a,v in nw.groupby('arm'):
        rr=absolute[a];row=dict(arm=a,worlds=len(v),absolute_R_7pct=rr[2]<=7)
        for tag in ['G','R','unavailable','mispointed','P5','Jain','narrow_s','max_gap_s','purchases','paid_s','switch_s','data_s','unavailable_s']:
            q=ci(v[tag]);row[tag]=q[0];row[tag+'_lo']=q[1];row[tag+'_hi']=q[2]
        tables.append(row)
    table=pd.DataFrame(tables);csvout(table,'表1_同一849笔预约下的净吞吐与风险.csv')
    def weighted_scalar(v,ws,metric):
        vs=v.set_index('scale');return sum(float(weight)*vs.loc[float(m),metric] for m,weight in ws.items())
    matched=[]
    for setting,ww in [('risk_matched',d['E2_matched']['weights']),('individual_target',{a:v['weights'] for a,v in d['E2_individual_target'].items()})]:
        for (k,a),v in s2.groupby(['world','arm']):
            if a not in ww:continue
            matched.append(dict(setting=setting,world=k,arm=a,G=weighted_scalar(v,ww[a],'G'),R=weighted_scalar(v,ww[a],'R')))
    mdf=pd.DataFrame(matched);csvout(mdf,'E2/development_frozen_workpoints.csv')
    for setting,tab in mdf.groupby('setting'):
        for a in sorted(set(['VALUE-time',d['strong_E2'],'EDF-route'])):comp2(a,'VALUE-route',a,tab,setting)
    lm=[]
    for (k,a),v in e1.groupby(['world','arm']):
        if a in d['E1_matched']['weights']:
            ws=d['E1_matched']['weights'][a];lm.append(dict(world=k,arm=a,U=weighted_scalar(v,ws,'U'),O=weighted_scalar(v,ws,'O')))
    lm=pd.DataFrame(lm);csvout(lm,'E1/development_frozen_workpoints.csv')
    csvout(pd.DataFrame(cs),'contrasts.csv')
    ageframe=pd.DataFrame(byage,columns=['world','arm','age','U']);fig,ax=plt.subplots(1,2,figsize=(10,4));colors={'TIME':'#3865aa','ROUTE':'#13806c','SHUFFLE':'#b5632d'}
    age_differences=[]
    for age in (2,3,5):
        block=ageframe[ageframe.age==age]
        for other in ('TIME','SHUFFLE'):
            effect=ci(paired(block,other,'ROUTE','U'))
            age_differences.append(dict(age_s=age,comparison=f'{other}-ROUTE',delta_U_s=effect[0],low95=effect[1],high95=effect[2],worlds=effect[3]))
    csvout(pd.DataFrame(age_differences),'E1/age_contrasts.csv')
    for a,color in colors.items():
        means=[];errs=[]
        for age in (2,3,5):
            v=ageframe[(ageframe.arm==a)&(ageframe.age==age)];q=ci(v.U)
            means.append(np.nan if q[0] is None else q[0]);errs.append([0,0] if q[1] is None else [max(0,q[0]-q[1]),max(0,q[2]-q[0])])
        ax[0].errorbar([2,3,5],means,yerr=np.asarray(errs).T,marker='o',color=color,label=a,capsize=3)
        v=e1[e1.arm==a].groupby('scale')[['O','U']].mean()
        if len(v):
            ax[1].plot(v.O,v.U,'o-',color=color,label=a);ax[1].scatter(v.loc[.75,'O'],v.loc[.75,'U'],s=100,facecolors='none',edgecolors=color)
    ax[0].set(xlabel='Age since paid fix (s)',ylabel='Missed usable lifetime U (s)',xticks=[2,3,5]);ax[1].set(xlabel='Overadmission O (s)',ylabel='Missed usable lifetime U (s)')
    if d['E1_matched']['overlap']:ax[1].axvspan(*d['E1_matched']['overlap'],alpha=.09,color='black')
    ax[0].legend();fig.suptitle(f'Fixed Chronos center | {len(ids)} worlds | {all_prefixes:,} issued prefixes');fig.tight_layout();figsave(fig,'Fig1_fixed_center_lifetime')
    same=[]
    for k in sorted(ids):
        rows=json.loads(gzip.decompress((OUT/f'E2/same_state_rankings/{k}.json.gz').read_bytes()))
        h=pd.DataFrame(rows,columns=['world','appointment','route_uid','alternate','alt_uid','age_gap','within_one_opportunity','route_support','alt_support'])
        for rival,v in h.groupby('alternate'):
            diff=v[v.route_uid!=v.alt_uid];close=diff[diff.within_one_opportunity]
            same.append(dict(world=k,comparison=rival,scored=len(v),conflicts=len(diff),near_AoI_conflicts=len(close),supported_conflicts=int((diff.route_support>0).sum())))
    same=pd.DataFrame(same);csvout(same,'E2/same_state_rankings_by_world.csv')
    diag=[];rank=[]
    for path in sorted((OUT/'E2/labels/development').glob('*/*/state.json')):
        rec=get(path);b=np.asarray(rec['bits']);chosen={}
        for side in ('time','route'):
            x=np.asarray(rec['features'][side]);s=value_head(side).predict(x,num_threads=1);u=tie(s,-np.asarray(rec['ages']))[0];chosen[side]=u
            corr=spearmanr(s,b).statistic
            rank.append(dict(world=rec['world'],appointment=rec['appointment'],side=side,spearman=float(corr) if np.isfinite(corr) else None,regret_bits=float(max(b)-b[u]),improvement_over_EDF_bits=float(b[u]-b[rec['edf']])))
        u,v=chosen['route'],chosen['time']
        diag.append(dict(world=rec['world'],appointment=rec['appointment'],route_uid=u,time_uid=v,conflict=u!=v,route_support=rec['support'][u][1]>0,age_diff=abs(rec['ages'][u]-rec['ages'][v]),delta_bits=b[u]-b[v]))
    diag=pd.DataFrame(diag);csvout(diag,'E2/same_state_conflicts.csv');csvout(pd.DataFrame(rank),'E2/development_value_diagnostics.csv')
    fig,ax=plt.subplots(1,2,figsize=(11,4.6))
    for i,(a,v) in enumerate(nw.groupby('arm')):
        point=[v.R.mean(),v.G.mean()];col=plt.cm.tab10(i)
        ax[0].scatter(*point,color=col,label=a)
        matchrows=mdf[(mdf.setting=='risk_matched')&(mdf.arm==a)]
        if len(matchrows):ax[0].plot([point[0],matchrows.R.mean()],[point[1],matchrows.G.mean()],'x--',lw=.7,color=col)
    ax[0].axvline(7,color='.5',ls=':');ax[0].set(xlabel='Risk (%)',ylabel='Net goodput (Mbps)');ax[0].legend(fontsize=7,ncol=2)
    groups=diag[diag.conflict].groupby('route_support');y=[]
    for j,(supp,v) in enumerate(groups):
        z=ci(v.groupby('world').delta_bits.mean()/1e6);y.append('Supported' if supp else 'Fallback')
        ax[1].bar(j,z[0],color='#13806c');ax[1].text(j,z[0],f'{len(v)} decisions',ha='center',va='bottom',fontsize=8)
        if z[1] is not None:ax[1].errorbar(j,z[0],yerr=np.array([[max(0,z[0]-z[1])],[max(0,z[2]-z[0])]]),fmt='none',color='black',capsize=3)
    if not y:ax[1].text(.5,.5,'No conflicting sampled decisions',transform=ax[1].transAxes,ha='center')
    ax[1].set(xticks=range(len(y)),xticklabels=y,ylabel='Paired rescued bits (Mbit)',title='4.464 s counterfactual | development');ax[1].axhline(0,color='.6',lw=.8)
    fig.tight_layout();figsave(fig,'Fig2_recipient_goodput')
    e2pass=yes2['VALUE-time'] and yes2[d['strong_E2']]
    if e1pass and e2pass:claim='路线记忆改善波束保持期判断，并在相同定位成本、风险非劣条件下带来净吞吐收益。'
    elif e1pass:claim='只主张固定 Chronos 中心下的保持期判断改善。同费净吞吐收益尚未获得预冻结双判据支持。'
    elif alt:claim='主线收敛为强预测器下的多用户付费刷新调度。路线记忆改善保持期的机制证据不足。'
    elif yes2['VALUE-time']:claim='可报告选人层相对同构 VALUE-time 的同费经验收益。保持期机制归因未获支持。'
    else:claim='本轮证据不足以主张路线保持期改善或同风险吞吐改善，论文应报告风险与吞吐的权衡、支持覆盖与失效条件。'
    rr=absolute['VALUE-route'];lines=['# Chronos-849 配对确认实验报告','',claim,'',f'确认集共 {len(ids)} 个新物理世界。E1 共有 {all_prefixes:,} 个唯一签发前缀，主层含 {sum(x["primary_prefixes"] for x in counts)} 个前缀、{main.world.nunique()} 个有效世界。E2 完成 {len(df)} 个全程端点，其中共同倍率端点 {len(d["base_endpoints"])} 个。共同工作点为 0.75。每个世界内先平均 DQN 的三个冻结种子。区间采用完整物理世界配对自助法，20,000 次，种子 271127。','','## 图 1　固定中心下的波束保持期','','![图1](figures/Fig1_fixed_center_lifetime.png)','']
    for a in sorted(yes1):
        u=answers['E1',a,'missed_s_saved'];o=answers['E1',a,'extra_overadmission_s'];lines.append(f'- ROUTE 对 {a} 的错失秒数改善为 {fmt(u)} s，过度准入增量为 {fmt(o)} s。0.02 秒非劣双判据{"通过" if yes1[a] else "未通过"}。零容忍敏感性{"通过" if o[2] is not None and o[2]<0 else "未通过"}。')
    lines+=['',f'有效空间置换覆盖 {len(effective_worlds)} 个世界、{len(effective_prefixes)} 个主层前缀。开发冻结的非空间风险头为 {d["strong_E1"]}。缺失世界及原因见 E1/primary_coverage.csv。','','## 表 1　同一 849 笔预约下的净吞吐与风险','','|臂|G Mbps|R %|不可用 %|失向 %|用户 P5 Mbps|实付笔数|占时 s|绝对 7%|','|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for r in table.itertuples():lines.append(f'|{r.arm}|{r.G:.3f}|{r.R:.3f}|{r.unavailable:.3f}|{r.mispointed:.3f}|{r.P5:.3f}|{r.purchases:.0f}|{r.paid_s:.4f}|{"通过" if r.absolute_R_7pct else "未通过"}|')
    lines+=['',f'开发集冻结的最强非空间对手为 {d["strong_E2"]}。以下 ΔG 为 VALUE-route 减对手，ΔR 单位为百分点。','']
    for a in dict.fromkeys(['VALUE-time',d['strong_E2'],'EDF-route']):
        g=answers['E2',a,'delta_G_Mbps'];r=answers['E2',a,'delta_R_pp'];lines.append(f'- 对 {a}，ΔG 为 {fmt(g)} Mbps，ΔR 为 {fmt(r)} 个百分点。风险非劣{"通过" if r[2]<.5 else "未通过"}。吞吐与风险双判据{"通过" if yes2[a] else "未通过"}。')
    lines+=['',f'VALUE-route 的绝对风险为 {fmt(rr)}%，其上界与 7% 比较，{"通过" if rr[2]<=7 else "未通过"}。开发倍率网格的 E2 共同可达风险区间为 {d["E2_matched"]["overlap"]}。各臂边际可达区间和风险匹配权重见 design/frozen.json，确认集加权点见 E2/development_frozen_workpoints.csv。',f'无路线调度 {d["strong_nonroute_scheduling"]} 对冻结 AoI 强对照 {d["strong_AoI"]} 的 ΔG 为 {fmt(answers["E2","nonroute_vs_AoI","delta_G_Mbps"])} Mbps，ΔR 为 {fmt(answers["E2","nonroute_vs_AoI","delta_R_pp"])} 个百分点。','','## 图 2　选人分歧带来的通信收益','','![图2](figures/Fig2_recipient_goodput.png)','',f'右图为离线开发诊断。分层抽定的 {len(diag)} 个共同付费状态中，两价值臂选人不同时才计入，共 {int(diag.conflict.sum())} 个分歧预约。各候选随后均走 EDF-native，比较 4.464 秒全机群实际交付比特。置信区间按开发世界聚类。','','首次单世界核验使用临时接口测试头。最终单世界核验、开发工作点及确认集使用拟合并独立校准的模型。原始场景为 16 机物理仿真、180 秒、零预热。定位成本、波束控制、风险与净速率均取自本轮逐场账本。']
    lines+=['','## E1 分层覆盖与回退','','比例先在有合格查询的世界内计算，再按世界等权平均。','',
            '|分层|有效世界|前缀|查询|无时间支持回退|仅时间支持回退|10 s 右删失|','|---|---:|---:|---:|---:|---:|---:|']
    for row in slices:
        rate=lambda name:('—' if row[name] is None else f'{100*row[name]:.1f}%')
        lines.append(f'|{row["stratum"]}|{row["worlds"]}|{row["prefixes"]}|{row["queries"]}|{rate("no_time")}|{rate("time_only")}|{rate("censored")}|')
    lines+=['','风险匹配若涉及两个倍率，使用开发冻结的权重，对同一物理世界两个完整闭环回执的净吞吐和风险计算加权期望。共同倍率的主结论和绝对 7% 结论均保留各自的确认集区间。']
    if yes1.get('TIME') and yes1.get(d['strong_E1']) and not e1pass:
        lines+=['','带检索特征的头相对于冻结的非空间比较臂改善了保持期判断。空间置换对照或有效置换覆盖未达预设条件，因而路线地理对应关系的独立贡献仍待验证。']
    v=same[same.comparison=='VALUE-time']
    lines+=['',f'共同倍率的 VALUE-route 线上状态中，VALUE-time 在 {int(v.conflicts.sum())}/{int(v.scored.sum())} 个可选预约会改选用户，其中 {int(v.near_AoI_conflicts.sum())} 笔的两个候选信息年龄相差至多 0.186 秒。该统计只比较同一个付费状态的排名，短窗因果净比特另由图 2 的反事实诊断计算。每场两类风险均以 16×9,000 个探针为分母。']
    rawinfo=[get(OUT/f'E1/prefix_index/confirmation/{k}/receipt.json') for k in ids]
    closedinfo=[get(p) for p in (OUT/'E2/receipts/confirmation').glob('*.json')]
    telemetry=OUT/'logs/cpu_memory.csv'
    compute=dict(E1_formal_worlds=len(rawinfo),E1_formal_prefixes=sum(r['prefixes'] for r in rawinfo),E1_worker_seconds_per_1000_prefixes=1000*sum(r['wall_seconds'] for r in rawinfo)/sum(r['prefixes'] for r in rawinfo),E2_confirm_endpoints=len(closedinfo),E2_median_wall_s=float(np.median([r['elapsed_s'] for r in closedinfo])),E2_endpoint_wall_quartiles=np.quantile([r['elapsed_s'] for r in closedinfo],[.25,.75]).tolist(),E2_cache_hits=sum(r['cache_hits'] for r in closedinfo),E2_native_forwards=sum(r['cache_misses'] for r in closedinfo),counterfactual_branches=dict(train=4096,validation=256,development_diagnostic=256))
    if telemetry.is_file():
        monitor=pd.read_csv(telemetry)
        compute['sampled_resources']={phase:dict(samples=len(tab),cpu_cores_mean=float(tab.study_cpu_cores.mean()),peak_observed_pss_GiB=float(tab.study_pss_bytes.max()/2**30),peak_shared_cgroup_memory_GiB=float(tab.cgroup_memory_bytes.max()/2**30)) for phase,tab in monitor.groupby('phase')}
    put(OUT/'resources.json',compute)
    lines+=['',f'正式 E1 每 1,000 个签发前缀消耗 {compute["E1_worker_seconds_per_1000_prefixes"]:.1f} 个世界任务墙钟秒，E2 正式端点墙钟中位数 {compute["E2_median_wall_s"]:.1f} 秒。采样 CPU 和进程比例内存峰值见 resources.json。训练、验证和开发诊断短分支分列，均未计作正式端点。']
    planning=d['power_planning']
    lines+=['',f'样本量在读取确认结果前按开发配对方差与重访覆盖确定。计算需求为 {planning["requirements"]} 个物理世界，选择 {planning["selected"]} 个。预约算力上限对全部预设效应的规划精度{"充足" if planning["capacity_sufficient"] else "未完全覆盖"}。可检出性估算采用开发方差，正式结论以确认配对区间为准。']
    runtime=get(OUT/'design/runtime.json');manifest=get(OUT/'manifest.json')
    lines+=['','## 复核入口','','|项目|实际提交或 SHA-256|','|---|---|',f'|five-mpc 基线提交|{manifest["commit"]}|',f'|本轮冻结提交|{d["new_code_commit"]}|',f'|本轮源代码集合|{d["code_hash"]}|',f'|Chronos 原底座|{runtime["native"]["original_backbone"]["sha256"]}|',f'|Chronos 种子 7 微调权重|{runtime["native"]["Chronos/best.pt"]["sha256"]}|',f'|Chronos 原生误差管|{runtime["native"]["Chronos/tube.pkl"]["sha256"]}|',f'|展开的物理场景配置|{manifest["config_sha"]}|',f'|849 笔预约表|{manifest["calendar_sha256"]}|','','模型与全部加载源码路径见 manifest.json。世界清单、比较对象、统计阈值及每个新增模型哈希见 design/frozen.json。两次 1400 核验与逐留出世界审计索引见 audits.json。']
    atom(OUT/'REPORT.md',('\n'.join(lines)+'\n').encode())
    put(OUT/'report_receipt.json',dict(worlds=len(ids),endpoints=len(exp),E1_pass=e1pass,E2_pass=e2pass,absolute_risk_pass=bool(rr[2]<=7),simple_increment=yes2['EDF-route'],scheduling_pass=alt,claim=claim,artifacts={str(p.relative_to(OUT)):sha_file(p) for p in [OUT/'REPORT.md',OUT/'world_scores.csv',OUT/'contrasts.csv',*sorted((OUT/'figures').glob('*'))]}));finish('report_final',OUT/'report_receipt.json')
    (OUT/'logs/status.txt').write_text('全部冻结实验、审计与报告已生成。结论见 REPORT.md。\n')
