import numpy as np
from scaling import CGS, MKS, scaling

def Planck (T, lamb, units=None, method=''):
    """ The Planck function, in dimensionless units, per log nu or log lambda """
    if units== None:
        x = (1.0/lamb)/T
    else:
        c = (units.h_P/units.k_B)*units.c
        x = (c/(lamb))/(T)

    if method=='simple':
        f = T**4 * x**4 * 1/ (np.exp(x)-1.0)
    else:
        e = np.exp(-x)
        em1 = 1.0 - e
        if isinstance(x,float):
            if x < 1e-10:
                em1 = x
        else:
            ii = np.where(x < 1e-10)
            em1[ii] = x[ii]
        f = ((T*x)**2)**2 * e/em1

    if units==None:
        return f
    else:
        # Cf. Eq. (2-3) in Radiation.ipynb
        return f * (units.Stefan/(np.pi**4/15))