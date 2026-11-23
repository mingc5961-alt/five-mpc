"""慢层分配感知名额，快层在剩余物理时间内联合选择波束与各机 TDMA 份额。"""
import numpy as np
from scipy.optimize import linprog
from .data import GRID

def sensing_clock(count,dt,budget,cost):
    """公共预算时钟：到第 k 轮末累计可用的观测数不超过 floor(k·dt·预算/成本)。

    每个快轮至多观测一架；空出的观测时间立即退给通信。本版本不跨轮囤积未用名额。
    """
    cap=np.floor(np.arange(count+1)*min(1.,dt*budget/cost)+1e-10).astype(int)
    return np.diff(cap).astype(bool)

def rate_table(profiles, t, horizon, dt, cost):
    """生成滚动规划查找表 [未来快轮, 飞机, 最近刷新编号]。

    编号 0 表示继续使用已发布包，j+1 表示在展望窗口的第 j 轮刷新。
    对尚未收到的新测量作确定性近似：保持最近可见的路线环境，重置预测年龄。
    真正执行刷新后，快层立即用新测量重新求解，不沿用这个假设的波束。
    """
    n=len(profiles)
    result=np.zeros((horizon,n,horizon+1))
    resets=t+np.arange(horizon)*dt+cost
    # 所有候选计划共用同一张表；未来年龄按本轮中点近似，实际账本逐机计时。
    mid=t+(np.arange(horizon)+.5)*dt
    for i,pr in enumerate(profiles):
        ages=mid[:,None]-np.r_[pr.pkg.origin,resets][None,:]
        z=np.clip(ages/0.25,0,len(GRID)-1)
        lo=z.astype(int)
        frac=(z-lo)[...,None]
        vals=(1-frac)*pr.rates[lo]+frac*pr.rates[np.minimum(lo+1,len(GRID)-1)]
        result[:,i]=np.where((ages>=0)&(ages<=10.),vals.max(axis=-1),0.)
    return result

def one_step_allocation(rates,available,due,minimum,weight):
    """慢层搜索内部的可行通信分配，批量处理多条候选分支。

    每机先分到固定最低时间；在服务欠账清零前，一秒通信同时带来吞吐和欠账
    减少两项收益。按分段边际价值由高到低分时间，等价于单快轮线性规划。
    """
    count,n=rates.shape
    airtime=np.broadcast_to((minimum*available/n)[:,None],rates.shape).copy()
    remaining=available-airtime.sum(axis=1)
    needed=np.maximum(due-rates*airtime,0.)
    useful=np.divide(needed,rates,out=np.zeros_like(rates),where=rates>0)
    slopes=rates*(1+weight)
    best=np.argmax(rates,axis=1)
    idx=np.arange(count)
    for ids in np.argsort(-slopes,axis=1,kind="stable").T:
        piece=np.where(slopes[idx,ids]>rates[idx,best],np.minimum(useful[idx,ids],remaining),0.)
        airtime[idx,ids]+=piece
        remaining-=piece
    airtime[idx,best]+=remaining
    bits=airtime*rates
    return bits.sum(axis=1),np.maximum(due-bits,0.)

def slow_plan(profiles,t,debt,horizon,c,cost,switch,opportunities=None):
    """搜索未来多轮的「等待或刷新某架飞机」序列，每层保留总收益最高的若干条。

    刷新要占用本轮通信时间。搜索价值同时计入各机已交付的 Mbit 和服务欠账
    的下降量；同一张期望速率表用于两层控制。返回完整候选最优计划供快层前瞻。
    """
    n=len(profiles)
    dt=c["dt"]
    table=rate_table(profiles,t,horizon,dt,cost)
    histories=np.zeros((1,0),dtype=int)
    last=np.zeros((1,n),dtype=int)
    debts=debt[None,:]
    score=np.zeros(1)
    for j in range(horizon):
        actions=np.arange(-1,n) if opportunities is None or opportunities[j] else np.array([-1])
        prior=np.repeat(np.arange(len(score)),len(actions))
        act=np.tile(actions,len(score))
        last_next=last[prior].copy()
        active=act>=0
        last_next[np.flatnonzero(active),act[active]]=j+1
        rates=table[j,np.arange(n)[None,:],last_next]
        available=dt-cost*active-n*switch
        due=debts[prior]+c["target_network_mbps"]*dt/n
        bits,new_debt=one_step_allocation(rates,available,due,c["min_airtime_fraction"],c["deficit_weight"])
        values=score[prior]+bits+c["deficit_weight"]*(debts[prior].sum(axis=1)-new_debt.sum(axis=1))
        keep=np.argsort(-values,kind="stable")[:c["search_width"]]
        histories=np.column_stack([histories[prior[keep]],act[keep]])
        last,debts,score=last_next[keep],new_debt[keep],values[keep]
    return histories[0].tolist(),float(score[0])

def fast_plan(profiles,t,debt,actions,c,cost,switch):
    """在给定慢层计划后，优化多个未来快轮，执行第一轮的时间与波束。

    决策变量为每轮每机的数据秒数 a 和期末服务欠账 q。约束为每轮时间守恒、
    最低时间份额，以及 q[k] >= q[k-1]+目标到达量-rate*a。目标为总预计
    有效比特减终端欠账代价。给定年龄后，各束宽均消耗相同切束时间，因此可
    先选预计有效速率最高的束宽，和 LP 联合最大化的解等价。
    """
    n=len(profiles)
    m=len(actions)
    dt=c["dt"]
    origins=np.array([p.pkg.origin for p in profiles])
    rates=np.empty((m,n))
    first_beams=[]
    available=np.array([dt-n*switch-cost*(a>=0) for a in actions])
    for j,a in enumerate(actions):
        if j and a>=0:
            origins[a]=t+j*dt+cost
        # 当前轮的刷新已经执行，profiles 内含新观测；未来刷新仍按确定性近似。
        ages=t+(j+.5)*dt-origins
        for i,pr in enumerate(profiles):
            rates[j,i],beam,_=pr.at(ages[i])
            if j==0:
                first_beams.append(beam)
    size=m*n
    objective=np.r_[-rates.ravel(),np.zeros(size)]
    objective[-n:]=c["deficit_weight"]
    eq=np.zeros((m,2*size))
    bounds=[]
    for j in range(m):
        eq[j,j*n:(j+1)*n]=1.
        bounds.extend([(c["min_airtime_fraction"]*available[j]/n,None)]*n)
    bounds.extend([(0,None)]*size)
    ub=np.zeros((size,2*size))
    k=np.arange(size)
    ub[k,k]=-rates.ravel()
    ub[k,size+k]=-1.
    ub[k[n:],size+k[n:]-n]=1.
    rhs=np.full((m,n),-c["target_network_mbps"]*dt/n)
    rhs[0]-=debt
    sol=linprog(objective,A_ub=ub,b_ub=rhs.ravel(),A_eq=eq,b_eq=available,
                bounds=bounds,method="highs")
    if not sol.success:
        raise RuntimeError("快层时间分配求解失败："+sol.message)
    return sol.x[:n],first_beams
