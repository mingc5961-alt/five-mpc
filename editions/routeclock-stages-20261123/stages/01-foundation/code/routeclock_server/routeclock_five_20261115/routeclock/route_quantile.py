"""Calibrated interval tube from the gated experts' coordinate marginal quantiles."""
import numpy as np
from scipy.special import ndtr, ndtri

from .geometry import angular_jacobian
from .validity import EPS, QUERY_H, ScaleHead


def mixture_quantile(probability, weight, route_mean, imm_mean, route_std, imm_std):
    """Invert the two-Gaussian marginal CDF simultaneously across coordinates and horizons."""
    z = ndtri(probability)
    a = route_mean + z * route_std
    b = imm_mean + z * imm_std
    low, high = np.minimum(a, b), np.maximum(a, b)
    for _ in range(48):
        mid = (low + high) / 2
        cdf = weight * ndtr((mid-route_mean)/route_std) + (1-weight) * ndtr((mid-imm_mean)/imm_std)
        low = np.where(cdf < probability, mid, low)
        high = np.where(cdf < probability, high, mid)
    return (low+high)/2


class RouteQuantileTube(ScaleHead):
    def __init__(self):
        self.factor = None

    def raw(self, package, features):
        w = package.diagnostics['gate_weights']['RC'][:, None]
        p, i = (package.centers[k] for k in ('P','I'))
        ps, ins = (np.sqrt(np.maximum(np.diagonal(package.covariance[k], axis1=1, axis2=2), 1e-14))
                   for k in ('P','I'))
        lower, upper = (mixture_quantile(q,w,p,i,ps,ins) for q in (.1,.9))
        std = (upper-lower)/2.563103
        jac = np.array([angular_jacobian(x) for x in package.centers['RC']])
        scale = np.sqrt(np.einsum('hij,hj->hi', jac**2, std**2))
        return np.maximum.accumulate(np.maximum(scale, EPS), axis=0)[np.searchsorted(package.hs, QUERY_H)]
