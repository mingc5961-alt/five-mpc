"""Train and validate fixed-recipient and joint-WAIT AoI DQNs on frozen local inputs."""
from collections import deque
from concurrent.futures import ProcessPoolExecutor,as_completed
import json,multiprocessing,os,time
from pathlib import Path
import numpy as np
import torch
from history import put
from policy_env import initialize,episode
from paired_grid import BASE
from deterministic_matrix import budgets
from learning_ready import OUT
from slots import compute_slot

SEEDS=(17,29,41)
def save(path,data):
    tmp=path.with_suffix(f'.{os.getpid()}.tmp');torch.save(data,tmp);tmp.replace(path)
def eval_one(args):
    from aoi_policy import Policy
    path,protocol,k,kind,b,q=args
    model=Policy().eval();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True))
    with compute_slot():r,events,_=episode('train_validation',k,kind,protocol,model,b,q,.75)
    return dict(world=k,predictor=kind,G_mbps=r['G_mbps'],risk=r['R'],fixes=len(events))
def validation(directory,step,protocol,policy,pilot):
    dest=directory/f'val_{step}.json'
    if dest.exists():return json.loads(dest.read_bytes())
    path=directory/f'model_{step}.pt';save(path,policy.state_dict())
    b,q=budgets()[0]
    jobs=[(path,protocol,k,kind,b,q) for k in range(1100,1108 if pilot else 1116) for kind in ('Acal','M')]
    with ProcessPoolExecutor(4,mp_context=multiprocessing.get_context('fork')) as pool:rows=list(pool.map(eval_one,jobs))
    result=dict(step=step,episodes=rows,count=len(rows),G_mbps=float(np.mean([x['G_mbps'] for x in rows])),
                risk=float(np.mean([x['risk'] for x in rows])))
    put(dest,result);return result

def rank(v):
    return (1,v['G_mbps'],-v['risk']) if v['risk']<=.07 else (0,-v['risk'],v['G_mbps'])

def schedule(protocol,n,seed):
    rng=np.random.default_rng([20261127,seed,165 if protocol=='fixed' else 177])
    conditions=budgets()
    if protocol=='fixed':conditions=conditions[:1]
    count=[n] if protocol=='fixed' else [n//2]+[n//8]*4
    counts={}
    jobs=[]
    for (b,q),pairs in zip(conditions,count):
        for idx in range(pairs):
            kind=('Acal','M')[idx%2]
            offset=counts.get(kind,0)
            if offset%64==0:counts[(kind,'order')]=rng.permutation(64).tolist()
            w=1000+counts[(kind,'order')][offset%64]
            counts[kind]=offset+1
            jobs.append((w,kind,b,q))
    rng.shuffle(jobs);return jobs
def record_transitions(payload,lam,eta,b,q,norm):
    action=payload['action'];T=len(action);users=payload['users'];public=payload['public'];legal=payload['legal']
    reward=payload['bits']/norm-lam*payload['fail']/(16*9000/T)-eta*payload['queue']/q*((action<16)&(np.arange(T)>=16))
    result={'u':users,'p':public,'m':legal,'a':action.astype(np.int64),'r':reward.astype(np.float32),
            'nu':np.concatenate([users[1:],np.zeros_like(users[:1])]),'np':np.concatenate([public[1:],np.zeros_like(public[:1])]),
            'nm':np.concatenate([legal[1:],np.zeros_like(legal[:1])]),'done':np.arange(T)==T-1}
    return result
class Replay:
    def __init__(self):self.d={};self.at=0;self.size=0
    def append(self,data):
        n=len(data['a']);idx=(self.at+np.arange(n))%50000
        for key,arr in data.items():
            if key not in self.d:self.d[key]=np.empty((50000,*arr.shape[1:]),dtype=arr.dtype)
            self.d[key][idx]=arr
        self.at=(self.at+n)%50000;self.size=min(50000,self.size+n)
    def sample(self,rng):
        idx=rng.integers(0,self.size,64)
        return {k:torch.from_numpy(v[idx]) for k,v in self.d.items()}

def train(spec):
    from aoi_policy import Policy,td_target
    protocol,seed,lam,eta,pilot=spec
    key=f'{protocol}_s{seed}_l{lam:g}_q{eta:g}';directory=OUT/('trials' if pilot else 'formal')/key
    directory.mkdir(parents=True,exist_ok=True)
    if (directory/'summary.json').exists():return json.loads((directory/'summary.json').read_bytes())
    ready=json.loads((OUT/'ready.json').read_bytes())
    n=16 if pilot else 128;limit=4000 if pilot else 30000
    milestones=[limit] if pilot else [10000,20000,30000]
    rng=np.random.default_rng(seed);torch.manual_seed(seed)
    policy=Policy();target=Policy().eval();target.load_state_dict(policy.state_dict())
    optimizer=torch.optim.Adam(policy.parameters(),lr=.001)
    replay=Replay();step=0;done=0;curve=[];checkpoint=directory/'resume.pt'
    jobs=schedule(protocol,n,seed)
    if checkpoint.exists():
        last=torch.load(checkpoint,weights_only=False,map_location='cpu');done=last['done'];step=last['step']
        policy.load_state_dict(last['policy']);target.load_state_dict(last['target']);optimizer.load_state_dict(last['optim'])
        rng.bit_generator.state=last['rng'];curve=last['curve']
        for i in range(done):
            with np.load(directory/f'episode_{i:03}.npz') as d:replay.append({k:d[k] for k in d.files})
    for i in range(done,n):
        w,kind,b,q=jobs[i]
        epsilon=max(.05,1.-.95*i/max(1,.65*(n-1)))
        with compute_slot():
            result,events,payload=episode('train',w,kind,protocol,policy.eval(),b,q,.75,rng,epsilon,True)
            data=record_transitions(payload,lam,eta,b,q,ready['bits_per_window'][f'{b:g}_{q}'])
            tmp=directory/f'episode_{i:03}.tmp'
            with tmp.open('wb') as fp:np.savez_compressed(fp,**data)
            tmp.replace(directory/f'episode_{i:03}.npz');replay.append(data)
            old=step;losses=[]
            while step<(i+1)*limit//n:
                x=replay.sample(rng);policy.train()
                pred=policy(x['u'],x['p']).gather(1,x['a'][:,None]).flatten()
                y=td_target(target,x['nu'],x['np'],x['nm'],x['r'],x['done'],gamma=.99)
                loss=torch.nn.functional.smooth_l1_loss(pred,y)
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(policy.parameters(),2.)
                optimizer.step();step+=1;losses.append(loss.item())
                if step%500==0:target.load_state_dict(policy.state_dict())
                if step in milestones:save(directory/f'model_{step}.pt',policy.state_dict())
        curve.append(dict(episode=i+1,world=w,predictor=kind,budget_s=b,quota=q,fixes=len(events),
            epsilon=epsilon,G_mbps=result['G_mbps'],risk=result['R'],gradient_steps=step,
            reward_sum=float(data['r'].sum()),mean_loss=float(np.mean(losses)),
            forced_rate=float((payload['legal'].sum(1)==1).mean())))
        save(checkpoint,dict(done=i+1,step=step,policy=policy.state_dict(),target=target.state_dict(),
            optim=optimizer.state_dict(),rng=rng.bit_generator.state,curve=curve))
        put(directory/'progress.json',dict(completed_train_episodes=i+1,planned_episodes=n,gradient_steps=step))
        for at in milestones:
            if old<at<=step:
                snapshot=Policy();snapshot.load_state_dict(torch.load(directory/f'model_{at}.pt',weights_only=True))
                validation(directory,at,protocol,snapshot,pilot)
    reports=[]
    for at in milestones:
        snapshot=Policy();snapshot.load_state_dict(torch.load(directory/f'model_{at}.pt',weights_only=True))
        reports.append(validation(directory,at,protocol,snapshot,pilot))
    best=max(reports,key=lambda x:(*rank(x),-x['step']))
    result=dict(protocol=protocol,policy_seed=seed,lambda_risk=lam,eta_queue=eta,run=key,train_episodes=n,gradient_steps=step,
        validation_episodes=sum(v['count'] for v in reports),selected_step=best['step'],validation=best,
        model=str(directory/f"model_{best['step']}.pt"),pilot=pilot,curve=curve)
    put(directory/'summary.json',result)
    return result
def run_group(tasks):
    with ProcessPoolExecutor(len(tasks),mp_context=multiprocessing.get_context('fork')) as pool:
        return [f.result() for f in as_completed([pool.submit(train,t) for t in tasks])]
def main():
    initialize()
    put(OUT/'training_protocol.json',dict(gamma=.99,batch=64,replay_capacity=50000,optimizer='Adam lr .001',gradient_clip=2,
        target_update_steps=500,epsilon='1 to .05 over first 65% of training episodes; deterministic masked argmax in validation',
        pilot_gradient_updates=4000,formal_gradient_updates=30000,pilot_episodes=16,formal_episodes=128,
        risk_penalties=[.3,1,3],joint_queue_penalties=[0,.1],tuning_seed=53,seeds=SEEDS,
        checkpoint_rule='main validation pairs: feasible risk <=.07 then max throughput; else min risk. ties earlier checkpoint; hyperparam ties smaller risk and queue penalties.',
        train_validation_only=True))
    candidates=[(p,53,lam,eta,True) for p in ('fixed','joint') for lam in (.3,1.,3.) for eta in ((0.,) if p=='fixed' else (0.,.1))]
    pilots=run_group(candidates)
    best={p:max((x for x in pilots if x['protocol']==p),key=lambda x:(*rank(x['validation']),-x['lambda_risk'],-x['eta_queue'])) for p in ('fixed','joint')}
    put(OUT/'hyperparameters.json',{p:{k:r[k] for k in ('lambda_risk','eta_queue','run')} for p,r in best.items()})
    formal=run_group([(p,seed,best[p]['lambda_risk'],best[p]['eta_queue'],False) for p in best for seed in SEEDS])
    all_runs=pilots+formal
    summary=dict(formal={f"{r['protocol']}_{r['policy_seed']}":r['model'] for r in formal},trials=len(pilots),formal_models=len(formal),
        training_episodes=sum(r['train_episodes'] for r in all_runs),validation_episodes=sum(r['validation_episodes'] for r in all_runs),
        gradient_steps=sum(r['gradient_steps'] for r in all_runs))
    assert (summary['training_episodes'],summary['validation_episodes'],summary['gradient_steps'])==(912,720,216000)
    put(OUT/'summary.json',summary)
if __name__=='__main__':main()
