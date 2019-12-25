import numpy as np
import scipy
from astroML.correlation import bootstrap_two_point_angular, two_point_angular

def get_angular_2pt_func(ra, dec, bins, nboot=1):
    ra, dec = np.array(ra), np.array(dec)
    coords = np.array([ra, dec]).transpose()
    if nboot==1:
        corr = two_point_angular(coords[:,0], coords[:,1], bins)
        return corr
    else:
        corr, dcorr, boot = bootstrap_two_point_angular\
        (coords[:,0], coords[:,1], bins, method='landy-szalay', Nbootstraps=nboot)
        return corr, dcorr, boot
    
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
    
    dzdchi = (cosmo.H(z)/const.c/cosmo.h).to(1/u.Mpc).value # H0/c[h/Mpc]
    w_arr *= dzdchi * bg * bI * dIdz
    
    return w_arr

def wgg(zin, theta_arr, bg=1, zbinedges=None):
    '''
    Galaxy-Galaxy Correlation function: w_gg(theta)
    
    w_{gg} (\theta) = b_g(z_m)\int_{0}^{\infty}\frac{dk}{2\pi}k
    \left [ \sum_{i\in zbins} P_{\delta\delta}(k, z_i)J_0(k\theta\chi(z_i))
    \frac{1}{\Delta z} \frac{H(z_i)}{c} \left ( \frac{N_i}{N_{tot}} \right )^2\right ]
    
    Inputs:
    =======
    zin: list of redshift of the sources
    theta_arr: arr of angle [arcsec]
    bg: gal bias at z
    
    Outputs:
    ========
    w_arr: correlation function w_{gg} at theta_arr [arcsec]
    '''
    zin = np.array(zin)
    
    import nbodykit.lab
    from astropy import units as u
    from astropy import cosmology
    from astropy import constants as const
    cosmo = cosmology.Planck15

    theta_bins_arcsec = theta_arr
    theta_bins_rad = (theta_bins_arcsec * u.arcsec).to(u.rad).value # deg

    if zbinedges is None:
        zbinedges = np.arange(0,1.1,0.1)
    zbins = (zbinedges[1:] + zbinedges[:-1]) / 2
    dz = zbinedges[1] - zbinedges[0]

    kbinedges = np.logspace(-3, 3, 1000) #[h/Mpc]
    kbins = np.sqrt(kbinedges[:-1] * kbinedges[1:])
    dk = kbinedges[1:] - kbinedges[:-1]

    chis = (cosmo.comoving_distance(zbins)*cosmo.h).value # [Mpc/h]
    dzdchis = (cosmo.H(zbins)/const.c/cosmo.h).to(1/u.Mpc).value # [h/Mpc]
    Plins = []
    for z in zbins:
        Plin = nbodykit.lab.cosmology.LinearPower(nbodykit.lab.cosmology.Planck15,
                                                  redshift=z, transfer='CLASS')
        Plin = Plin(kbins) #[h^-3/Mpc^3]
        Plins.append(Plin)
    Plins = np.array(Plins)

    Nis,_ = np.histogram(zin, bins=zbinedges)
    fis = Nis / np.sum(Nis)

    w_arr = np.zeros_like(theta_bins_arcsec)
    for ith, th in enumerate(theta_bins_rad):
        zints = []
        for ik, k in enumerate(kbins):
            zint = np.sum((1/dz) * dzdchis * fis**2 * Plins[:,ik] * scipy.special.jv(0, chis*k*th))
            zints.append(zint)
        zints = np.array(zints)

        w_arr[ith] = np.sum(dk * kbins * zints) / 2 / np.pi
    
    return bg**2*w_arr

def wgI(zin, theta_arr, bg=1, bI=1, dIdz=1, zbinedges=None):
    '''
    Galaxy-Intensity Correlation function: w_gI(theta)
    
    w_{gI} (\theta) = b_g(z_m)b_I(z_m)\frac{dI}{dz}\bigg\rvert_{z_m}
    \int_{0}^{\infty}\frac{dk}{2\pi}k\left [ \sum_{i\in zbins} P_{\delta\delta}(k, z_i)
    J_0(k\theta\chi(z_i))\frac{H(z_i)}{c}\frac{N_i}{N_{tot}}\right ]
    
    Inputs:
    =======
    zin: list of redshift of the sources
    theta_arr: arr of angle [arcsec]
    bg: gal bias at z
    
    Outputs:
    ========
    w_arr: correlation function w_{gg} at theta_arr [arcsec]
    '''
    zin = np.array(zin)
    
    import nbodykit.lab
    from astropy import units as u
    from astropy import cosmology
    from astropy import constants as const
    cosmo = cosmology.Planck15

    theta_bins_arcsec = theta_arr
    theta_bins_rad = (theta_bins_arcsec * u.arcsec).to(u.rad).value # deg

    if zbinedges is None:
        zbinedges = np.arange(0,1.1,0.1)
    zbins = (zbinedges[1:] + zbinedges[:-1]) / 2
    dz = zbinedges[1] - zbinedges[0]

    kbinedges = np.logspace(-3, 3, 1000) #[h/Mpc]
    kbins = np.sqrt(kbinedges[:-1] * kbinedges[1:])
    dk = kbinedges[1:] - kbinedges[:-1]

    chis = (cosmo.comoving_distance(zbins)*cosmo.h).value # [Mpc/h]
    dzdchis = (cosmo.H(zbins)/const.c/cosmo.h).to(1/u.Mpc).value # [h/Mpc]
    Plins = []
    for z in zbins:
        Plin = nbodykit.lab.cosmology.LinearPower(nbodykit.lab.cosmology.Planck15,
                                                  redshift=z, transfer='CLASS')
        Plin = Plin(kbins) #[h^-3/Mpc^3]
        Plins.append(Plin)
    Plins = np.array(Plins)

    Nis,_ = np.histogram(zin, bins=zbinedges)
    fis = Nis / np.sum(Nis)

    w_arr = np.zeros_like(theta_bins_arcsec)
    for ith, th in enumerate(theta_bins_rad):
        zints = []
        for ik, k in enumerate(kbins):
            zint = np.sum(dzdchis * fis * Plins[:,ik] * scipy.special.jv(0, chis*k*th))
            zints.append(zint)
        zints = np.array(zints)

        w_arr[ith] = np.sum(dk * kbins * zints) / 2 / np.pi
    
    return bg*bI*dIdz*w_arr
