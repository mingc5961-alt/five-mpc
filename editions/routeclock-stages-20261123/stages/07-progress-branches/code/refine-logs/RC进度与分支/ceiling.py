import numpy as np
from routeclock.geometry import angles,angle_error
def diagnostic(d,weights,travel,scale,actual,y):
    # Truth is used only in this evaluation function, after the model has returned.
    v=d['valid'].astype(bool)
    curves=d['curves'][v].astype('float64')
    if len(curves):curves=np.concatenate([curves,(curves*weights[v,None,None]).sum(0)[None]])
    out=[]
    for h in (5,10):
        j=int(2*h);target=y[round(h/.02)];a=scale[j]
        base=d['base'][j]-a*(d['pathbase'][j]-d['pathbase'][0])
        def best(pts):
            pts=pts.reshape(-1,3)
            xyz=np.linalg.norm(pts-target,axis=1)
            ang=np.rad2deg(np.linalg.norm(angle_error(angles(pts),angles(target)),axis=1))
            return xyz.min(),ang.min()
        xy=actual[j,None];fixed=along=both=xy
        if len(curves):
            s=travel[j]/2;lo=min(59,int(s));r=s-lo
            fixed=np.vstack([base+a*(curves[:,lo]*(1-r)+curves[:,lo+1]*r),xy])
            # Add actual soft expectation at the real-valued learned distance.
            grid=curves[:,:int(12*h/2)+1];points=base+a*grid
            along=np.vstack([points[-1],xy]);both=np.vstack([points.reshape(-1,3),fixed,along])
        for tag,pts in [('learned',xy),('oracle_branch',fixed),('oracle_progress',along),('oracle_both',both)]:
            out.append((h,tag,*best(pts)))
    return out
