"""Deferred vectorized accounting. The scheduler and model never read the ledger."""
import numpy as np
from refine.network import HS, Ledger, interpolate
from routeclock.geometry import angles
from routeclock.radio import array_gain

class DeferredLedger(Ledger):
    def __init__(self,*args):
        super().__init__(*args)
        self.ends=[]
        self.states=[]
        self.centers=[]
        self.last_centers=[None]*self.world.n
        self.center_ids=[-1]*self.world.n
        self.slots=[]

    def advance(self,start,finish,state,sensing):
        if finish<=start:return
        epoch=len(self.ends)
        self.ends.append(finish)
        snapshot=[]
        for uid,p in enumerate(state):
            if self.last_centers[uid] is not p.center:
                self.last_centers[uid]=p.center
                self.center_ids[uid]=len(self.centers)
                self.centers.append(p.center)
            a=p.action
            snapshot.append((p.origin,self.center_ids[uid],a.interval if a else 0.,
                             a.pair[0] if a else 0, a.pair[1] if a else 0))
        self.states.append(snapshot)
        if sensing:
            self.times[0]+=finish-start
            return
        count=max(1,int(np.ceil((finish-start)/self.quantum-1e-10)))
        slots=start+np.arange(count)*self.quantum
        ends=np.minimum(slots+self.quantum,finish)
        starts=np.minimum(slots+self.switch,ends)
        users=(self.rr+np.arange(count))%self.world.n
        self.rr=(self.rr+count)%self.world.n
        self.times[1]+=sum(starts-slots)
        self.slots.append((epoch,users,starts,ends))

    def result(self,cost,rho):
        state=np.asarray(self.states)
        centers=np.asarray(self.centers)
        public=self.probes
        probe_epoch=np.searchsorted(self.ends,public,side='right')
        self.probe_count=len(public)
        if self.slots:
            slot_epoch=np.concatenate([np.full(len(ids),epoch) for epoch,ids,_,_ in self.slots])
            users,starts,ends=(np.concatenate([s[j] for s in self.slots]) for j in (1,2,3))
        else:
            slot_epoch=users=np.empty(0,int)
            starts=ends=np.empty(0)
        for uid in range(self.world.n):
            ref=state[:,uid,:]
            expiry=ref[:,0]+ref[:,2]
            good=(ref[probe_epoch,2]>0)&(public<expiry[probe_epoch])
            self.unavailable_count[uid]=self.risk_count[uid]=np.count_nonzero(~good)
            sel=users==uid
            ep=slot_epoch[sel];left=starts[sel];right=ends[sel]
            stop=np.where(ref[ep,2]>0,np.maximum(left,np.minimum(right,expiry[ep])),left)
            self.times[3]+=(right-stop).sum()
            used=stop>left
            dt=(stop-left)[used]
            self.times[2]+=dt.sum()
            mids=(stop[used]+left[used])*.5
            times=np.r_[public[good],mids]
            epochs=np.r_[probe_epoch[good],ep[used]]
            split=np.count_nonzero(good)
            if not len(times):continue
            keys=ref[epochs,1].astype(int)
            age=times-ref[epochs,0]
            # Use the original np.interp operation for each forecast/axis.
            order=np.argsort(keys,kind='stable')
            cuts=np.flatnonzero(np.diff(keys[order]))+1
            xyz_pred=np.empty((len(times),3))
            for ix in np.split(order,cuts):
                center=centers[keys[ix[0]]]
                xyz_pred[ix]=np.column_stack([np.interp(age[ix],HS,v) for v in center.T])
            pointing=angles(xyz_pred)
            xyz=interpolate(self.world.positions[uid],times,self.world.dt)
            pairs=ref[epochs,3:].astype(int)
            gain=array_gain(pointing,angles(xyz),pairs.T)
            loss=10*np.log10(np.prod(pairs,axis=1)/gain)
            self.risk_count[uid]+=np.count_nonzero(loss[:split]>3.)
            if not len(mids):continue
            pair=pairs[split:].T
            nominal=self.radio.snr(xyz_pred[split:],pointing[split:],pair)
            mcs=np.searchsorted(self.radio.threshold,nominal,side='right').clip(1,len(self.radio.se))-1
            shadow=interpolate(self.world.shadows[uid],mids,self.world.dt)
            snr=self.radio.snr(xyz[split:],pointing[split:],pair,shadow)
            rate,_=self.radio.goodput(snr,mcs)
            self.bits[uid]=rate@dt
            self.narrow+=dt[(pair[0]==32)&(pair[1]==32)&(loss[split:]<=3.)].sum()
        return super().result(cost,rho)
