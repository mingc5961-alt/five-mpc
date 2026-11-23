import hashlib,html,json,sys
import numpy as np,pandas as pd
from protocol_local import OUT,OLD,log
SPLITS=('fresh','replicate','prior','test','flight')
NAMES={'replicate':'第二批新世界 60–69','fresh':'新世界 50–59','prior':'旧世界 40–49','test':'离线 400 路线','flight':'AMOVFLY 59 架次','prev':'上轮 RC','RC':'原始 RC','continued':'仅继续训练','distance':'共享进度＋里程监督','tempo':'候选专属进度','both':'候选专属进度＋里程监督','joint-refit':'上轮 RC＋上轮管','v2-transfer':'本轮中心＋上轮管','v2-refit':'本轮中心＋重训管','Chronos-native':'Chronos-2 原生管','Chronos-2':'Chronos-2','VECTOR':'VECTOR','RAFT':'RAFT'}
def ci(a,b,ratio=False):
    a,b=np.asarray(a),np.asarray(b);assert a.shape==b.shape and np.isfinite([a,b]).all()
    index=np.random.default_rng(2381).integers(len(a),size=(5000,len(a)))
    aa,bb=a[index].mean(1),b[index].mean(1)
    stat=aa/bb if ratio else aa-bb
    return [(a.mean()/b.mean()) if ratio else (a.mean()-b.mean()),*np.quantile(stat,[.025,.975])]
def run():
    sel=json.loads((OUT/'selection.json').read_text());win=sel['selected']
    results={};pair=[];verified={};group=[];worlds=[]
    for split in SPLITS:
        d=pd.read_csv(OUT/(split+'.csv'));results[split]=d
        assert len(d)=={'replicate':160,'fresh':160,'prior':160,'test':1200,'flight':236}[split]*63
        assert not d.duplicated(['id','anchor','arm','horizon']).any()
        if split in ('prior','test','flight'):
            previous=pd.read_csv(OLD/(('world' if split=='prior' else split)+'.csv'))
            a=d[d.arm.isin(['prev','Chronos-2'])].replace({'arm':{'prev':'joint'}})
            b=a.merge(previous,on=['id','anchor','horizon','arm'],validate='one_to_one')
            assert len(b)==len(a)
            err=abs(b.position-b.position_m).max()
            assert err<.0002,(split,err)
            verified[split]=err
        if split in ('fresh','replicate','prior'):
            grouped=d[d.arm.isin([win,'prev','Chronos-2'])].groupby(['id','arm','horizon']).position.mean().unstack('arm')
            for w in d.id.unique():
                r={'批次':split,'世界':w}
                for h in (5,10):
                    g=grouped.loc[w,h];r.update({f'{a}_{h}':g[a] for a in [win,'prev','Chronos-2']})
                    r[f'误差比_{h}']=g[win]/g['Chronos-2']
                r['双期限均接近']=max(r['误差比_5'],r['误差比_10'])<=1.10
                worlds.append(r)
        keys=['position','angle','hold','safe32','arc_error','along','cover']
        for h in (5,10):
            s=d[d.horizon==h].groupby(['id','arm'])[keys].mean()
            for a in [win,'continued','distance','tempo','both']:
                for base in ['prev','continued','Chronos-2','VECTOR','RAFT']:
                    if a==base:continue
                    for k in ['position','angle','hold','safe32']:
                        av=s.xs(a,level='arm')[k];bv=s.xs(base,level='arm')[k]
                        pair.append((split,h,a,base,k,*ci(av,bv),*(ci(av,bv,True) if k=='position' else [None]*3)))
        g=d[d.horizon.isin([5,10])].groupby(['group','arm','horizon'])[keys].mean().reset_index().assign(split=split);group.append(g)
    group=pd.concat(group);ws=pd.DataFrame(worlds).sort_values('世界')
    sensitivity=[]
    for split in ['prior','fresh','replicate','fresh20','combined30']:
        w=ws if split=='combined30' else (ws[ws['世界']>=50] if split=='fresh20' else ws[ws['批次']==split])
        for h in (5,10):
            a=w[f'{win}_{h}'];b=w[f'Chronos-2_{h}'];loo=(a.sum()-a)/(b.sum()-b)
            sensitivity.append([split,h,*ci(a,b,True),int((a<b).sum()),int((a<=1.10*b).sum()),len(w),loo.min(),loo.max()])
    sens=pd.DataFrame(sensitivity,columns='批次 秒 RC比Chronos 下限 上限 反超世界 达到1.10世界 世界数 删一最小比 删一最大比'.split())
    pair=pd.DataFrame(pair,columns='split 秒 模型 对照 指标 均差 差下限 差上限 比值 比下限 比上限'.split()).drop_duplicates()
    cell=[]
    for f in (OUT/'cells').glob('*.json'):
        v=json.loads(f.read_text());r=v['result']
        assert len(r['bits'])==16 and r['probes_per_user']==9000 and abs(sum(r['ledger_s'])-180)<1e-6
        cell.append([v['world'],v['arm'],v['budget'],r['G_mbps'],100*r['R'],r['hold_mean_s'],r['ledger_s'][0],r['narrow_good_data_s'],100*r['unavailable_fraction'],r['initial_center_sha256']])
    raw=pd.DataFrame(cell,columns='world arm budget Mbps 风险百分数 实际保持秒 感知秒 优质32阵列数据秒 不可用百分数 fingerprint'.split())
    assert len(raw)==150 and set(raw.world)==set(range(50,60)) and not raw.duplicated(['world','arm','budget']).any()
    fingerprint=raw.pivot(index=['world','budget'],columns='arm',values='fingerprint')
    for a,b in [('v2-refit','v2-transfer'),('Chronos-2','Chronos-native')]:assert (fingerprint[a]==fingerprint[b]).all()
    closed=raw.drop(columns=['world','fingerprint']).groupby(['budget','arm']).mean().reset_index()
    cpair=[]
    for b in 'TML':
        s=raw[raw.budget==b].set_index(['world','arm'])
        for a in ['v2-transfer','v2-refit']:
            for base in ['joint-refit','v2-transfer','Chronos-2','Chronos-native']:
                if base==a:continue
                for k in ['Mbps','风险百分数','实际保持秒','感知秒']:
                    av=s.xs(a,level='arm')[k].sort_index();bv=s.xs(base,level='arm')[k].sort_index()
                    cpair.append([b,a,base,k,*ci(av,bv)])
    cpair=pd.DataFrame(cpair,columns='预算 模型 对照 指标 均差 下限 上限'.split())
    avg=pd.concat([d.drop(columns=['id','anchor','group']).groupby(['arm','horizon']).mean().reset_index().assign(split=k) for k,d in results.items()])
    fits=json.loads((OUT/'traces.json').read_text())
    trials=pd.DataFrame([dict(arm=m,seed=r['seed'],lr=r['lr'],aux=r['coefficient'],epochs=len(r['validation']),min_validation=min([sel['previous'],*r['validation']])) for m,rs in fits.items() for r in rs])
    assert len(trials)==12
    tables={'中心均值':avg,'中心配对区间':pair,'世界敏感性':sens,'逐世界难度':ws,'轨迹类型':group,'闭环均值':closed,'闭环配对区间':cpair,'闭环逐场':raw,'训练组合':trials}
    with pd.ExcelWriter(OUT/'配对结果.xlsx',engine='openpyxl') as xl:
        for k,v in tables.items():
            v.to_excel(xl,index=False,sheet_name=k);sheet=xl.sheets[k];sheet.freeze_panes='B2';sheet.auto_filter.ref=sheet.dimensions
            for col in sheet.columns:sheet.column_dimensions[col[0].column_letter].width=min(32,max(13,max(len(str(c.value)) for c in col[:20])+2))
    parts=['<!doctype html><meta charset="utf-8"><title>候选里程与 Chronos 配对实验</title><style>body{font:16px/1.65 system-ui;max-width:1150px;margin:3em auto;color:#172d36}table{border-collapse:collapse;font-size:14px}th,td{border-bottom:1px solid #d4dddd;padding:.3em .6em}th{background:#eaf2f2}h2{margin-top:2em;color:#17435a}</style><h1>候选里程学习与 Chronos-2 的配对比较</h1>']
    def p(s):parts.append('<p>'+html.escape(s)+'</p>')
    def title(s):parts.append('<h2>'+html.escape(s)+'</h2>')
    def table(d):parts.append(d.replace(NAMES).to_html(index=False,float_format=lambda x:f'{x:.3f}',border=0,na_rep='—'))
    s=sens[sens['批次']=='fresh'];meanclose=s['RC比Chronos'].max()<=1.10;cert=s['上限'].max()<=1.10
    p(f"验证选择 {NAMES[win]}。新世界预定双期限接近标准（误差比≤1.10）均值达标：{meanclose}，两个配对 95% 区间上界达标：{cert}。")
    for h in (5,10):
        sub=pair[(pair.split=='fresh')&(pair['秒']==h)&(pair['模型']==win)&(pair['指标']=='position')].set_index('对照')
        for k in ['prev','Chronos-2']:
            r=sub.loc[k];p(f"新世界 {h} 秒相对 {NAMES[k]} 的位置误差差值 {r['均差']:+.2f} 米，配对 95% 区间 [{r['差下限']:+.2f}, {r['差上限']:+.2f}] 米。")
    t=avg[(avg.split=='fresh')&(avg.horizon==10)].set_index('arm')
    prev,new,chron=t.loc['prev'],t.loc[win],t.loc['Chronos-2']
    p(f"主确认的 10 秒候选投影里程误差由 {prev.arc_error:.2f} 米降至 {new.arc_error:.2f} 米，同时最终位置误差由 {prev.position:.2f} 米变为 {new.position:.2f} 米。相对于上轮与 Chronos 的差，缩小比例为 {(prev.position-new.position)/(prev.position-chron.position):.1%}。候选投影包含多路径近似、最终中心还经过分支加权和门控，里程误差与中心误差是不同口径。")
    combined=ws[ws['世界']>=50]
    for h in (5,10):
        delta=ci(combined[f'{win}_{h}'],combined[f'prev_{h}'])
        p(f"合并两个新批次的 20 个世界，{h} 秒相对上轮 RC 的均差为 {delta[0]:+.2f} 米，95% 配对区间 [{delta[1]:+.2f}, {delta[2]:+.2f}] 米。")
    title('1　同一起点的训练与消融')
    p('从上轮 48 维 GRU 的同一冻结点起步；原 IMM、旧门控、候选生成规则固定。增设继续训练对照以扣除再训本身的影响。四组都微调原网络可学习参数。候选专属速度头读取路径形状、候选身份、元数据，以及历史匹配点后已实际观测到的五段 2 秒里程速度；据此分别修正四条路径的速度，再按原软选择方式加权。新头零初始化，训练前验证损失核对一致。')
    p('里程监督从真实未来相对起点的位移构造，只作训练目标。每条发布时可见路径在 0–120 米上每 2 米一个网格；动态规划要求从零出发、每半秒前进 0–6 米，以全段位置残差对齐；在偏离路线或非局部弧长投影含混处降低权重。在 1、3、5、8、10 秒增加 Huber 损失。候选的几何与历史速度都来自发布前。辅助标签是可行路径上的估计，并非测得的真实里程。')
    p('中心训练 960 条路线、2880 个锚点，验证 400 条、1200 个锚点，两者与校准/测试分离。沿用角误差×40＋位置误差×0.04 作验证选择；每组 3 套学习率/种子组合，至多 65 轮，降学习率与早停。选出三种改动中验证最优者后冻结，不按确认结果重选。专用管另用 240 条训练路线、400 条独立校准路线，双角前缀标称 90%。')
    table(trials)
    title('2　主确认与复验')
    p('50–59 为新主确认；连续世界 60–69 是预先补入的独立中心复验，同样保留所有世界；40–49、离线 test、AMOVFLY 在上一轮分析过，本轮只作复验。不按模型效果剔除世界。16 用户共同 180 秒每秒预热，一次发布按相同观测比较；中心表不受调度影响。实飞为 59 架次、每架次四个固定观测起点，感知与阵列损耗模拟。位置单位米，角度为度，保持为固定 32×32 首次超过 3 dB 的秒数，10 秒截尾。along 为误差在真值局部运动方向的绝对分量，仅对速度≥0.5 m/s 的样本计算。arc_error 为候选投影置信度加权里程差，不能视为真实里程。')
    for k in SPLITS:
        title(NAMES[k]);sub=avg[(avg.split==k)&avg.horizon.isin([5,10])]
        table(sub[['arm','horizon','position','angle','along','arc_error','hold','safe32','cover']])
    title('3　逐世界结论与换世界敏感性')
    p('世界/路线/架次内先取均值，再对独立单位配对重抽样 5000 次。比值为误差均值之比；删一范围分别遍历每一个世界，与主样本同时报告。工作簿给出所有轨迹类型。')
    from world_plot import draw
    parts.append(draw(ws,win));p("每点代表一个世界 16 架次的平均误差。等值线下方表示新 RC 更低；绿色带表示新 RC 高出不到 10%。")
    table(sens);table(ws)
    for cohort in ['fresh','replicate']:
        u=ws[ws['批次']==cohort];r=sens[(sens['批次']==cohort)&(sens['秒']==10)].iloc[0]
        best=u.loc[u['误差比_10'].idxmin()];worst=u.loc[u['误差比_10'].idxmax()]
        p(f"{NAMES[cohort]} 的 10 秒指标，最小比值在世界 {int(best['世界'])}，为 {best['误差比_10']:.2f}；最大在世界 {int(worst['世界'])}，为 {worst['误差比_10']:.2f}。逐个尝试删去一个世界，剩余世界整体误差比在 {r['删一最小比']:.2f}–{r['删一最大比']:.2f}；两期限同时达到接近线的世界有 {int(u['双期限均接近'].sum())}/{len(u)} 个。")
    p('一两个世界的反超仅适用于该批场景。若实际应用要限制到某一类运行环境，应在看结果前按可观察条件定义，例如规则闭环或变速复访，再用另一批该类航迹验证。')
    title('按机动类型检查 10 秒差距')
    for cohort in ['fresh','replicate','flight']:
        d=results[cohort];d=d[d.horizon==10]
        g=d.pivot_table(index='group',columns='arm',values='position')[['prev',win,'Chronos-2']].copy()
        g['样本数']=d[d.arm==win].groupby('group').size()
        g['新RC减上轮']=g[win]-g['prev'];g['新RC减Chronos']=g[win]-g['Chronos-2']
        g['对整体差值的贡献米']=g['新RC减Chronos']*g['样本数']/g['样本数'].sum()
        p(NAMES[cohort]+'。组内均值只用于说明场景构成。贡献按完整批次内的样本占比计算，各类相加等于该批总体差。')
        table(g.reset_index())
    title('4　通信收益与风险')
    p('世界 50–59，180 秒预热、180 秒服务，16 用户。T/M/L 的感知上限依次为 3/4.5/9 秒（1.667%/2.5%/5%）；每次感知 3.1 ms、价值调度、最早到期刷新、10 ms TDMA、MCS 余量 0 dB。每用户 9000 次统一风险评分，含不可用或有效指向损耗超过 3 dB。上轮中心使用上轮联合模型的专用管；新中心分别转移上轮管或单独重训；Chronos 使用相同微调权重分别配专用或原生管。每个预算逐世界核对同中心两管的起始指纹。')
    table(closed)
    table(cpair[(cpair['模型']=='v2-refit')&(cpair['对照'].isin(['joint-refit','Chronos-2']))&cpair['指标'].isin(['Mbps','风险百分数'])])
    for b in 'TML':
        idx=cpair[(cpair['预算']==b)&(cpair['模型']=='v2-refit')&(cpair['指标']=='Mbps')].set_index('对照')
        a=idx.loc['joint-refit'];c=idx.loc['Chronos-2']
        p(f"{b} 档，新版重训管相对上轮 RC 的吞吐差 {a['均差']:+.1f} Mbps，配对区间 [{a['下限']:+.1f}, {a['上限']:+.1f}]；相对 Chronos 专用管 {c['均差']:+.1f} Mbps。中心评分与闭环刷新历史不同，不能以端点位置差直接推算吞吐差。")
    (OUT/'结果分析.html').write_text('\n'.join(parts))
    dig=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (OUT/'验收记录.json').write_text(json.dumps(dict(historical_baseline_max_difference_m=verified,selected=win,worlds=list(range(50,60)),cells=len(raw),assets={str(p.name):dig(p) for p in [OUT/(win+'.pt'),OUT/'heads/v2/special.pkl',OUT/'结果分析.html',OUT/'配对结果.xlsx']}),ensure_ascii=False,indent=2))
    log(f'验收完成，12/12 训练、5 个共同历史数据集、150/150 闭环；新世界 {win} 的 5/10 秒相对 Chronos 误差比 '+ '/'.join(f"{v:.3f}" for v in s['RC比Chronos'])+'；逐世界、留一分析、闭环账本、报告和工作簿已保存。')
if __name__=='__main__':run()
