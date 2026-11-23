import numpy as np

def angles(p):
    p = np.asarray(p)
    return np.stack([np.arctan2(p[..., 1], p[..., 0]), np.arctan2(p[..., 2], np.linalg.norm(p[..., :2], axis=-1))], axis=-1)

def angle_error(a, b):
    return (np.asarray(a) - np.asarray(b) + np.pi) % (2*np.pi) - np.pi

def unit(x):
    return np.asarray(x) / max(np.linalg.norm(x), 1e-12)

def clip_norm(x, limit):
    return np.asarray(x) * min(1.0, limit/max(np.linalg.norm(x), 1e-12))

def spherical_to_xyz(z):
    r,a,e = z[:3]
    return r*np.array([np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)])

def position_covariance(z, std):
    r,a,e = z[:3]
    J = np.array([[np.cos(e)*np.cos(a),-r*np.cos(e)*np.sin(a),-r*np.sin(e)*np.cos(a)],
                  [np.cos(e)*np.sin(a),r*np.cos(e)*np.cos(a),-r*np.sin(e)*np.sin(a)],
                  [np.sin(e),0,r*np.cos(e)]])
    return J @ np.diag(np.square(std[:3])) @ J.T

def angular_jacobian(p):
    x,y,z=p; r2=x*x+y*y; r=np.sqrt(max(r2,1e-12)); d=max(r2+z*z,1e-12)
    return np.array([[-y/max(r2,1e-12), x/max(r2,1e-12),0],[-x*z/(r*d),-y*z/(r*d),r/d]])
