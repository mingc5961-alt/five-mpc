"""Exogenous continuous physical worlds; predictors only receive Observation."""
from dataclasses import dataclass
import zlib
import numpy as np
from .geometry import angles, spherical_to_xyz, position_covariance, unit, clip_norm

GROUPS = ["stable_path_stable_tempo", "stable_path_global_speed", "stable_path_segment_speed",
          "stable_path_segment_speed_stops", "local_deviation_rejoin", "no_reusable_path"]

def rng_for(namespace, mission, stream=0, root=20261112):
    return np.random.default_rng(np.random.SeedSequence([root,zlib.crc32(namespace.encode()),int(mission),int(stream)]))

@dataclass
class Observation:
    t: float
    position: np.ndarray
    covariance: np.ndarray
    radial_velocity: float
    radial_std: float
    raw: np.ndarray

class World:
    def __init__(self, cfg, namespace, mission, n=None, groups=None, reusable=None, shape=None):
        self.cfg=cfg; self.namespace=namespace; self.mission=mission
        self.n=n or cfg["simulation"]["primary_N"]
        self.dt=cfg["simulation"]["truth_step_s"]
        self.duration=cfg["simulation"]["warmup_s"]+cfg["simulation"]["service_s"]+11
        self.times=np.arange(0,self.duration+self.dt/2,self.dt)
        if groups is None:
            weights=np.array([.25,1/6,1/6,1/6,.125,.125])
            if reusable is not None: weights=np.r_[reusable*np.array([2/7,4/21,4/21,4/21,1/7]),1-reusable]
            raw=weights*self.n; counts=np.floor(raw).astype(int)
            order=np.argsort(-(raw-counts),kind="stable")
            counts[order[:self.n-counts.sum()]]+=1
            if self.n==16 and reusable is None:
                counts=np.r_[4,np.roll([3,3,2],mission%3),2,2]
            groups=[g for g,c in zip(GROUPS,counts) for _ in range(c)]
        self.groups=groups
        self.positions=[]; self.velocities=[]; self.accelerations=[]
        self.references=[]; self.progress=[]; self.noise=[]; self.shadows=[]
        wind_rng=rng_for(namespace,mission,999,cfg['seeds']['data_root'])
        pars=cfg['data']['route_parameters']
        wind=np.zeros((len(self.times),3)); phi=np.exp(-self.dt/pars['common_wind_corr_s'])
        for k in range(1,len(wind)): wind[k]=phi*wind[k-1]+pars['common_wind_accel_std_m_s2']*np.sqrt(1-phi*phi)*wind_rng.normal(size=3)
        for i,g in enumerate(groups):
            r=rng_for(namespace,mission,i+1,cfg['seeds']['data_root'])
            self._generate(r,g,wind,shape)
            # A fixed absolute-time noise field: no dependence on method or query count.
            self.noise.append(r.normal(size=(len(self.times),4)))
            sh=np.zeros(len(self.times)); ph=np.exp(-self.dt/cfg['radio']['shadow_correlation_s'])
            sh[0]=r.normal()*cfg['radio']['shadow_std_db']
            for k in range(1,len(sh)): sh[k]=ph*sh[k-1]+np.sqrt(1-ph*ph)*cfg['radio']['shadow_std_db']*r.normal()
            self.shadows.append(sh)
        self.positions=np.array(self.positions); self.velocities=np.array(self.velocities)
        self.accelerations=np.array(self.accelerations)

    def _generate(self,r,g,wind,shape):
        pars=self.cfg['data']['route_parameters']; theta=np.deg2rad(r.uniform(*pars['initial_azimuth_deg'])); radius=r.uniform(*pars['initial_horizontal_range_m'])
        center=np.array([radius*np.cos(theta),radius*np.sin(theta),r.uniform(*pars['initial_altitude_m'])])
        phase=np.linspace(0,2*np.pi,1201)
        a,b=r.uniform(*pars['ellipse_semiaxes_m'],size=2); kind=shape or r.choice(['ellipse','figure8','scan'])
        if kind=='figure8':
            a,b=r.uniform(*pars['figure8_half_m'],size=2);curve=np.c_[a*np.sin(phase), b*np.sin(2*phase)/2, r.uniform(*pars['vertical_amplitude_m'])*np.sin(phase)]
        elif kind=='scan':
            a=r.uniform(*pars['scan_long_half_m']);b=r.uniform(*pars['scan_transverse_half_m'])
            vertices=np.array([[-a,-b],[a,-b],[a,-b/3],[-a,-b/3],[-a,b/3],[a,b/3],[a,b],[-a,b],[-a,-b]])
            lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(vertices,axis=0),axis=1))]
            ar=np.linspace(0,lengths[-1],len(phase)); curve=np.c_[np.interp(ar,lengths,vertices[:,0]),np.interp(ar,lengths,vertices[:,1]),np.zeros(len(ar))]
        elif kind=='polygon':
            v=r.uniform(-1,1,(7,2))*[a,b]; v=v[np.argsort(np.arctan2(v[:,1],v[:,0]))]; v=np.vstack([v,v[0]])
            lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(v,axis=0),axis=1))]; ar=np.linspace(0,lengths[-1],len(phase))
            curve=np.c_[np.interp(ar,lengths,v[:,0]),np.interp(ar,lengths,v[:,1]),np.zeros(len(ar))]
        else: curve=np.c_[a*np.cos(phase),b*np.sin(phase),r.uniform(*pars['vertical_amplitude_m'])*np.sin(2*phase)]
        rot=r.uniform(0,2*np.pi); R=np.array([[np.cos(rot),-np.sin(rot),0],[np.sin(rot),np.cos(rot),0],[0,0,1]])
        curve=curve@R.T+center; ss=np.r_[0,np.cumsum(np.linalg.norm(np.diff(curve,axis=0),axis=1))]; length=ss[-1]
        def at(s): return np.array([np.interp(s%length,ss,curve[:,j]) for j in range(3)])
        nom=length/r.uniform(*pars['nominal_traversal_s']); s=r.uniform(0,length); lap=-1; stop_until=-1; done_stops=set(); det=None
        p=np.zeros((len(self.times),3)); v=p.copy(); acc=p.copy(); prog=np.zeros(len(p)); p[0]=at(s)
        target=center+r.normal(size=3)*15; next_target=0; maneuver=np.zeros(3); nonperiodic_speed=r.uniform(*pars['nonperiodic_speed_m_s'])
        for k,t in enumerate(self.times[:-1]):
            new_lap=int(s//length)
            if new_lap!=lap:
                lap=new_lap; glob=1 if g==GROUPS[0] else r.uniform(*pars['global_speed_multiplier'])
                mult=r.uniform(*pars['segment_speed_multiplier'],size=int(r.integers(pars['segment_count'][0],pars['segment_count'][1]+1))) if g in GROUPS[2:5] else np.ones(1)
                stops=np.sort(r.uniform(0,length,r.poisson(pars['stop_count_per_traversal_poisson']))) if g==GROUPS[3] else np.array([])
                done_stops=set(); det=None
                if g==GROUPS[4] and r.random()<pars['detour_probability_per_traversal']:det=(r.uniform(0,length),r.uniform(*pars['detour_duration_s']),r.uniform(*pars['detour_amplitude_m'])*r.choice([-1,1]),None)
            frac=(s%length)/length; speed=nom*glob*mult[min(int(frac*len(mult)),len(mult)-1)]
            for si,sp in enumerate(stops):
                if si not in done_stops and s%length>=sp: done_stops.add(si); stop_until=t+r.uniform(*pars['stop_duration_s'])
            if t<stop_until: speed=0
            if g==GROUPS[5]:
                if t>=next_target:
                    radius=pars['nonperiodic_target_region_radius_m'];target=center+np.r_[r.uniform(-radius,radius,2),r.uniform(-12,12)]; next_target=t+r.uniform(*pars['nonperiodic_target_refresh_s'])
                desired=unit(target-p[k])*min(nonperiodic_speed,np.linalg.norm(target-p[k]))
            else:
                tangent=unit(at(s+.2)-at(s)); goal=at(s)
                if det is not None:
                    start,dur,amp,start_t=det
                    if start_t is None and s%length>=start: start_t=t; det=(start,dur,amp,start_t)
                    if start_t is not None:
                        u=(t-start_t)/dur
                        if 0<=u<=1: goal+=amp*64*u**3*(1-u)**3*unit(np.cross([0,0,1],tangent))
                desired=speed*tangent+1.2*(goal-p[k])
            phi=np.exp(-self.dt/pars['random_accel_corr_s']);maneuver=phi*maneuver+pars['random_accel_std_m_s2']*np.sqrt(1-phi*phi)*r.normal(size=3)
            disturbance=maneuver if g==GROUPS[5] else np.zeros(3)
            vmax=self.cfg['data']['max_speed_m_s'];amax=self.cfg['data']['max_acceleration_m_s2'];jmax=self.cfg['data']['max_jerk_m_s3']
            desired=clip_norm(desired,vmax)
            requested=clip_norm((desired-v[k])/0.7+wind[k]+disturbance,amax)
            aa=acc[k]+clip_norm(requested-acc[k],jmax*self.dt); aa=clip_norm(aa,amax)
            vv=v[k]+aa*self.dt
            if np.linalg.norm(vv)>vmax: vv=clip_norm(vv,vmax); aa=(vv-v[k])/self.dt
            v[k+1]=vv; acc[k+1]=aa; p[k+1]=p[k]+(v[k]+vv)*self.dt/2
            prog[k]=s; s+=speed*self.dt if g!=GROUPS[5] else np.linalg.norm(vv)*self.dt
        prog[-1]=s
        self.positions.append(p); self.velocities.append(v); self.accelerations.append(acc)
        self.references.append(curve); self.progress.append(prog)

    def interpolate(self, array, t):
        j=min(int(t/self.dt),len(self.times)-2); u=(t-self.times[j])/self.dt
        return array[j]*(1-u)+array[j+1]*u

    def truth(self,i,t): return self.interpolate(self.positions[i],t),self.interpolate(self.velocities[i],t)

    def observe(self,i,t):
        p,v=self.truth(i,t); d=np.linalg.norm(p); ae=angles(p)
        m=self.cfg['simulation']['measurement_std']; std=np.array([m['range_m'],np.deg2rad(m['azimuth_deg']),np.deg2rad(m['elevation_deg']),m['radial_velocity_m_s']])
        j=min(int(round(t/self.dt)),len(self.times)-1)
        z=np.r_[d,ae,np.dot(unit(p),v)]+std*self.noise[i][j]
        return Observation(t,spherical_to_xyz(z),position_covariance(z,std),float(z[3]),std[3],z)

    def shadow(self,i,t): return float(self.interpolate(self.shadows[i],t))
