"""Minimum-change lateral deviation under discrete kinematics and norm limits (Clarabel 0.11)."""
import numpy as np
from scipy import sparse as sp
try:
    import clarabel
except ModuleNotFoundError:
    import sys
    from server_paths import RUN
    sys.path.append(str(RUN/'deps'))
    import clarabel


def solve(position_profile, direction, velocity, acceleration, step, caps):
    n = len(velocity)
    eye = sp.eye(n,format='csr')
    diff = sp.diags([-np.ones(n-1),np.ones(n)],[-1,0],format='lil')
    diff[0,0] = 0.
    diff = diff.tocsr()
    average = sp.diags([np.ones(n-1)*.5,np.ones(n)*.5],[-1,0],format='csr')[1:]
    dim = sp.eye(3)
    kron = lambda a:sp.kron(a,dim,format='csr')
    # x = (offset in position, offset in velocity) at every simulation time.
    kin = sp.hstack([kron(diff[1:]),kron(-step*average)])
    left = lambda a:sp.hstack([a,sp.csr_matrix((a.shape[0],3*n))],format='csr')
    right = lambda a:sp.hstack([sp.csr_matrix((a.shape[0],3*n)),a],format='csr')
    ends = eye[[0,-1]]
    eq = sp.vstack([kin,left(kron(ends)),right(kron(ends)),right(kron(diff[[-1]]/step))])
    if direction is None:
        blocks = [eq]
        bs = [np.zeros(eq.shape[0])]
        cones = [clarabel.ZeroConeT(eq.shape[0])]
        target = np.zeros(3*n)
    else:
        lateral = left(sp.kron(eye,sp.csr_matrix(direction[None,:]),format='csr'))
        eq = sp.vstack([eq,lateral[[n//2]]])
        peak = position_profile.max()
        rhs = np.zeros(eq.shape[0]);rhs[-1] = peak
        blocks = [eq,lateral,-lateral]
        bs = [rhs,np.full(n,peak),np.zeros(n)]
        cones = [clarabel.ZeroConeT(len(rhs)),clarabel.NonnegativeConeT(2*n)]
        target = (position_profile[:,None]*direction).ravel()
    accel_op = diff / step
    jerk_op = (diff @ diff) / step**2
    rr = np.arange(3*n)
    embed = sp.csr_matrix((np.ones(3*n),(4*(rr//3)+1+rr%3,rr)),shape=(4*n,3*n))
    for cap,base,op in zip(caps,(velocity,acceleration,diff@acceleration/step),(eye,accel_op,jerk_op)):
        blocks.append(-embed @ right(kron(op)))
        bs.append(np.column_stack([np.full(n,cap),base]).ravel())
        cones.extend(clarabel.SecondOrderConeT(4) for _ in range(n))
    penalty = sp.diags(np.r_[np.ones(3*n),np.full(3*n,.02)],format='csc')
    settings = clarabel.DefaultSettings()
    settings.verbose = False
    settings.max_threads = 1
    sol = clarabel.DefaultSolver(penalty,np.r_[-target,np.zeros(3*n)],sp.vstack(blocks,format='csc'),
                                 np.concatenate(bs),cones,settings).solve()
    if str(sol.status) not in ('Solved','AlmostSolved'):
        return None
    p,v = np.array(sol.x).reshape(2,n,3)
    return p,v,accel_op@v
