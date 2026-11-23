import numpy as np
from itertools import product
from .geometry import angles

def array_gain(pointing, actual, pair):
    """Half-wavelength UPA in the yz plane; fixed total transmit power."""
    a,e=np.moveaxis(np.asarray(actual),-1,0); a0,e0=np.moveaxis(np.asarray(pointing),-1,0)
    du=np.cos(e)*np.sin(a)-np.cos(e0)*np.sin(a0); dv=np.sin(e)-np.sin(e0)
    gain=1.0
    for delta,n in zip([du,dv],pair): gain=gain*n*np.square(np.sinc(n*delta/2)/np.sinc(delta/2))
    return np.maximum(gain,1e-12)

class Radio:
    def __init__(self,cfg):
        self.cfg=cfg['radio']; self.pairs=list(product(self.cfg['subarray_axes'],repeat=2))
        self.se=np.array(self.cfg['spectral_efficiencies']); self.threshold=10*np.log10(2**self.se-1)+5
    def snr(self,p,pointing,pair,shadow=0):
        loss=20*np.log10(4*np.pi*np.maximum(np.linalg.norm(p,axis=-1),1)*self.cfg['carrier_hz']/299792458)
        noise=-174+10*np.log10(self.cfg['bandwidth_hz'])+self.cfg['noise_figure_db']
        return self.cfg['total_tx_dbm']+10*np.log10(array_gain(pointing,angles(p),pair))+self.cfg['rx_gain_dbi']-loss+shadow-noise
    def mcs(self,snr): return int(np.clip(np.searchsorted(self.threshold,snr,side='right')-1,0,len(self.se)-1))
    def goodput(self,snr,mcs):
        bler=1/(1+9*np.exp(np.clip((snr-self.threshold[mcs])/1.5,-60,60)))
        return self.cfg['bandwidth_hz']*.85*self.se[mcs]*(1-bler),bler
    def worst_loss(self,pointing,envelope,pair):
        # Enclose the angular rectangle in direction-cosine coordinates. The
        # product of 1D minimum factors is a conservative *joint* array bound.
        def bounds(lo,hi,func):
            critical=np.arange(np.floor(lo/(np.pi/2)),np.ceil(hi/(np.pi/2))+1)*(np.pi/2)
            points=np.r_[lo,hi,critical[(critical>=lo)&(critical<=hi)]];values=func(points)
            return values.min(),values.max()
        a,e=pointing;da,de=envelope
        sa=bounds(a-da,a+da,np.sin);ce=bounds(e-de,e+de,np.cos);se=bounds(e-de,e+de,np.sin)
        products=np.array([x*y for x in sa for y in ce]);u0=np.cos(e)*np.sin(a);v0=np.sin(e)
        max_d=[max(abs(products.min()-u0),abs(products.max()-u0)),max(abs(se[0]-v0),abs(se[1]-v0))]
        factor=1.
        for d,n in zip(max_d,pair):
            if d>=2/n:return 120. # bound includes a first-null direction
            factor*=float((np.sinc(n*d/2)/np.sinc(d/2))**2)
        return float(-10*np.log10(max(factor,1e-12)))
