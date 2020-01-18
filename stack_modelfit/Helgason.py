import numpy as np

# Helgason table 2 data
class bands_class:
    
    '''
    Helgason et al. 2012 table 2
    https://ui.adsabs.harvard.edu/#abs/2012ApJ...752..113H/abstract
    '''
    
    def __init__(self,idx):
        
        name_arr = ['UV','U','B','V','R','I','z','J','H','K','L','M']
        wleff_arr = [0.15,0.36,0.45,0.55,0.65,0.79,0.91,1.27,1.63,2.20,3.60,4.50]
        zmax_arr = [8.0,4.5,4.5,3.6,3.0,3.0,2.9,3.2,3.2,3.8,0.7,0.7]
        M0_arr = [-19.62,-20.20,-21.35,-22.13,-22.40,-22.80,-22.86,-23.04,-23.41,-22.97,-22.40,-21.84]
        q_arr = [1.1,1.0,0.6,0.5,0.5,0.4,0.4,0.4,0.5,0.4,0.2,0.3]
        phi0_arr = [2.43,5.46,3.41,2.42,2.25,2.05,2.55,2.21,1.91,2.74,3.29,3.29]
        p_arr = [0.2,0.5,0.4,0.5,0.5,0.4,0.4,0.6,0.8,0.8,0.8,0.8]
        alpha0_arr = [-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00]
        r_arr = [0.086,0.076,0.055,0.060,0.070,0.070,0.060,0.035,0.035,0.035,0.035,0.035]
        
        self.name = name_arr[idx]
        self.wleff = wleff_arr[idx]
        self.zmax = zmax_arr[idx]
        self.M0 = M0_arr[idx]
        self.q = q_arr[idx]
        self.phi0 = phi0_arr[idx]
        self.p = p_arr[idx]
        self.alpha0 = alpha0_arr[idx]
        self.r = r_arr[idx]
        
        self._name_arr = np.asarray(name_arr)
        self._wleff_arr = np.asarray(wleff_arr)
        self._zmax_arr = np.asarray(zmax_arr)
        self._M0_arr = np.asarray(M0_arr)
        self._q_arr = np.asarray(q_arr)
        self._phi0_arr = np.asarray(phi0_arr)
        self._p_arr = np.asarray(p_arr)
        self._alpha0_arr = np.asarray(alpha0_arr)
        self._r_arr = np.asarray(r_arr)

def Helgason_LF(z, M_arr, bandidx):
    '''
    Helgason + 2012
    rest frame luminosity function [#/mag/Mpc^3].
    Helgason eq 1, eq 3~5, table 2
    '''
    params = bands_class(bandidx)
    Mstr = params.M0 - 2.5 * np.log10((1 + (z - 0.8))**params.q)
    phistr = params.phi0 * np.exp(-params.p * (z - 0.8)) * 1e-3
    alpha = params.alpha0 * (z / 0.01)**params.r
    phi_arr = 0.4 * np.log(10) * phistr * (10**(0.4 * (Mstr - M_arr)))**(alpha + 1) \
            * np.exp(-10**(0.4 * (Mstr - M_arr)))
    return phi_arr