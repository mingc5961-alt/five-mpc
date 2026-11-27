from base import *
tasks={}
def add(kind,batch,prio,resource,deps=(),**args):
    key=kind+'__'+tag(args)
    if key not in tasks:tasks[key]=dict(id=key,kind=kind,batch=batch,prio=prio,resource=resource,deps=list(deps),args=args)
    return key
def wjob(w):return add('world','R04' if w<1000 else ('R03' if w>=3000 else 'R09'),1 if w<1000 else 35,'cpu',w=w)
def cohort(v):return [] if v=='arch' else [wjob((1000 if v=='half' else 2000)+k) for k in range(28)]
def res(a):return 'gpu' if a in ('Cnative','Cexternal','Czero','VECTOR','RAFT') else 'cpu'
def weights(a,v):
    n={'T8P':'T8','G8P':'G8'}.get(a,a)
    if n not in ('T8','T180','G8','G180'):return []
    shard=[add('shard','R02' if v=='arch' else 'R09',3 if v=='arch' else 40,'cpu',cohort(v),v=v,split=s) for s in ('fit','val')]
    net=add('train','R02' if v=='arch' else 'R09',4 if v=='arch' else 41,'train',shard,a=n,v=v)
    if n==a:return [net]
    return [add('gate','R02' if v=='arch' else 'R09',6 if v=='arch' else 42,'cpu',[net,*cohort(v)],a=a,v=v)]
def tube(a,H,v,mode,priority):
    return add('cal','R02' if H==180 and v=='arch' else ('R09' if v!='arch' else 'R08'),priority,res(a),[*weights(a,v),*cohort(v)],a=a,H=H,v=v,mode=mode)
def grid(arms,H,v,mode,budgets,ids,batch,rank,transfer=False,scales=SCALES):
    allcells=[]
    for a in arms:
        fit=tube(a,180 if transfer else H,v,'all' if transfer else mode,rank)
        for w in ids:
            panel=add('panel',('R06' if a.endswith('8P') else 'R05') if batch=='R07' else batch,rank+1,res(a),[wjob(w),*weights(a,v)],a=a,H=H,v=v,mode=mode,w=w)
            for b in budgets:
                for m in scales:
                    allcells.append(add('cell',batch,rank+2+(2 if w%3000>=100 else 0)+(1 if m!=1. else 0),res(a),[wjob(w),fit],a=a,H=H,v=v,mode=mode,w=w,b=b,m=m,transfer=transfer))
    return allcells
def bundle(arms,H,v,mode,budgets,ids,batch,rank,code,transfer=False):
    c=grid(arms,H,v,mode,budgets,ids,batch,rank,transfer)
    panels=[k for k,t in tasks.items() if t['kind']=='panel' and t['args']['H']==H and t['args']['v']==v and t['args']['mode']==mode and t['args']['w'] in ids and t['args']['a'] in arms]
    return add('tables','R10',rank+6,'light',c+panels,code=code)
def build():
    ids=DEV+CONF
    add('anchor','R01',0,'light',source='provided')
    bundle(ARMS,180,'arch','all',['T'],ids,'R07',8,'main_T')
    bundle(ARMS,180,'arch','all',['M','L'],ids,'R07',20,'budgets_ML')
    bundle(ARMS,180,'arch','all',['T'],[3000+i for i in ids],'R03',22,'random10')
    for H in [30,60,120]:
        primary=['T8P','Cnative','G8P','Joint07']
        bundle(primary,H,'arch','all',['T'],ids,'R08',25,f'h{H}_priority')
        bundle(ARMS,H,'arch','all',['T'],ids,'R08',30,f'h{H}_all')
        bundle(['T8P','G8P','Joint07'],H,'arch','route',['T'],ids,'R08',32,f'h{H}_route')
        bundle(primary,H,'arch','all',['T'],ids,'R08',34,f'h{H}_tube_transfer',True)
    for v in ('original','half'):
        bundle(['T8','T180','G8','G180','T8P','G8P'],180,v,'all',['T'],ids,'R09',43,'training_'+v)
    for t in list(tasks.values()):
        if t['kind']=='cell':
            a=t['args']; ref=dict(a,w=3000 if a['w']>=3000 else 0,m=1.)
            first='cell__'+tag(ref)
            if first!=t['id'] and first in tasks:t['deps'].append(first)
        if t['kind']=='tables' and t['args']['code']!='main_T':
            t['deps'].append('tables__main_T')
    return list(tasks.values())
def revised_plan():
    tasks.clear()
    methods=['G8P','Cnative','IMM','C']
    bundle(ARMS+['C'],180,'arch','all',['T','M','L'],DEV+CONF,'R07',8,'all_main')
    bundle(methods,180,'arch','all',['T','M','L'],DEV+CONF,'R07',8,'core_main')
    # Native comparisons keep all confirmation worlds. Scale sweeps for the
    # secondary CPU models use the same three preselected confirmation worlds.
    secondary={'T8','T180','G8','G180','T8P','Joint07'}
    omit=set()
    for t in tasks.values():
        a=t['args']
        if (t['kind']=='cell' and a['a'] in secondary and a['b'] in ('M','L')
                and a['w'] in CONF[3:] and a['m']!=1.):
            omit.add(t['id'])
    for k in omit:
        del tasks[k]
    main=tasks['tables__all_main']
    main['deps']=[k for k in main['deps'] if k not in omit]
    worlds=[0,100,101,102,103,104]
    for h in (30,120):
        grid(methods,h,'arch','all',['T'],worlds,'R08',12,scales=[1.])
    grid(['G8P','Cnative'],30,'arch','all',['T'],worlds,'R08',12,transfer=True,scales=[1.])
    for w in worlds:
        for a,h,mode in [(a,60,'all') for a in methods]+[('G8P',30,'route')]:
            add('panel','R08',13,res(a),[wjob(w),*weights(a,'arch')],a=a,H=h,v='arch',mode=mode,w=w)
    history_deps=[]
    for t in tasks.values():
        a=t['args']
        if t['kind'] not in ('cell','panel') or a['a'] not in methods or a['w'] not in worlds:
            continue
        if t['kind']=='panel' or (a['b']=='T' and a['m']==1.):
            history_deps.append(t['id'])
    add('tables','R10',16,'light',history_deps,code='core_history')
    for t in tasks.values():
        a = t['args']
        if (t['kind'] == 'cell' and a['a'] in {'G180', 'T180'}
                and a['b'] in {'M', 'L'} and a['m'] != 1 and a['w'] in (100, 101, 102)):
            t['resource'] = 'gpu'
            t['inference_device'] = 'cuda'
    # Keep the same world-zero reference prerequisite as the original experiment.
    for t in tasks.values():
        if t['kind']=='cell':
            a=t['args'];first='cell__'+tag(dict(a,w=0,m=1.))
            if first!=t['id']:
                t['deps'].append(first)
    return list(tasks.values())

if __name__=='__main__':js(BASE/'manifest.json',revised_plan())
