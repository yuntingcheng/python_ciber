import numpy as np
import scipy

def wgI_zm_approx(z, theta_arr, bg=1, bI=1, dIdz=1):
    '''
    Galaxy Intensity Correlation function: w_GI(theta)
    The is an approximation by assuming the galaxies are in a thing redshift shell,
    
    w_{gI} (\theta) = \frac{H(z)}{c} b_g(z_m) b_I(z_m)\frac{dI}{dz}\bigg\rvert_{z_m}
    \int_{0}^{\infty}\frac{dk}{2\pi}kP_{\delta\delta}(k, z_m)J_0(k\theta\chi(z_m))
    
    Inputs:
    =======
    z: redshift zm
    theta_arr: arr of angle [arcsec]
    bg: gal bias at z
    bI: intensity bias at z
    dIdz: dI/dz @ z [nW/m^2/sr]
    
    Outputs:
    ========
    w_arr: correlation function w_{gI} at theta_arr [arcsec]
    '''
    
    import nbodykit.lab
    from astropy import units as u
    from astropy import cosmology
    from astropy import constants as const
    cosmo = cosmology.Planck15
    
    theta_arr = (np.array(theta_arr) * u.arcsec).to(u.rad).value
    chi = (cosmo.comoving_distance(z)*cosmo.h).value
    Plin = nbodykit.lab.cosmology.LinearPower(nbodykit.lab.cosmology.Planck15,
                                              redshift=z, transfer='CLASS')
    kbinedges = np.logspace(-3, 3, 1000) #[h/Mpc]
    kbins = np.sqrt(kbinedges[:-1] * kbinedges[1:])
    dk = kbinedges[1:] - kbinedges[:-1]
    Plin = Plin(kbins) #[h^-3/Mpc^3]
    
    w_arr = np.zeros_like(theta_arr)
    for i,th in enumerate(theta_arr):
        w_arr[i] = np.sum(dk/2/np.pi*kbins*Plin*scipy.special.jv(0, chi*kbins*th))
    
    # H0/c = 100/3e5 [h/Mpc]
    w_arr *= (100/3e5) * bg * bI * dIdz
    
    return w_arr