import numpy as np
import scipy
import os
from scipy import interpolate
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

class cosmology_power_spectrum:
    """
    Get the power spectrum from nbodykit
    Don't really need this hmf is much better
    """
    def __init__(self, z=0, k=None):
        self.z = z
        self.zbins_data = np.arange(0.01,1,0.05)
        self.kbins_data = np.logspace(-3, 3, 100) # h/Mpc
        self.savename_linear = './linear_power'
        self.savename_nonlinear = './nonlinear_power'
        self.update(k)
        if not os.path.exists(self.savename_linear + '.npy') \
        and not os.path.exists(self.savename_nonlinear + '.npy'):
            self._calc_power()
        
    def update(self, k, z=None):
        if k is None:
            k = self.kbins_data
        if z is None:
            z = self.z
        self.z = z
        self.k = k
        self.linear_power = self.interpolate_linear_power(k=k)
        self.nonlinear_power = self.interpolate_nonlinear_power(k=k)
    
    def interpolate_linear_power(self, k):
        data = np.load(self.savename_linear + '.npy')
        logdata = np.log10(data)
        logP = np.zeros(len(self.kbins_data))
        
        for i in range(len(self.kbins_data)):            
            spline_z = interpolate.InterpolatedUnivariateSpline\
            (self.zbins_data, logdata[:,i])
            logP[i] = spline_z(self.z)
        
        spline_k = interpolate.InterpolatedUnivariateSpline\
        (np.log10(self.kbins_data), logP)
        P = 10**spline_k(np.log10(k))
        
        return P
         
    def interpolate_nonlinear_power(self, k):
        data = np.load(self.savename_nonlinear + '.npy')
        logdata = np.log10(data)
        logP = np.zeros(len(self.kbins_data))
        
        for i in range(len(self.kbins_data)):            
            spline_z = interpolate.InterpolatedUnivariateSpline\
            (self.zbins_data, logdata[:,i])
            logP[i] = spline_z(self.z)
        
        spline_k = interpolate.InterpolatedUnivariateSpline\
        (np.log10(self.kbins_data), logP)
        P = 10**spline_k(np.log10(k))

        return P

    def _calc_power(self):
        import nbodykit.lab
        from astropy import units as u
        from astropy import cosmology
        from astropy import constants as const
        cosmo = cosmology.Planck15       

        zbins = self.zbins_data
        kbins = self.kbins_data

        cosmonbodykit = nbodykit.lab.cosmology.Planck15
        cosmonbodykit = cosmonbodykit.clone(P_k_max = np.max(kbins), nonlinear=True)

        data_l = np.zeros([len(zbins), len(kbins)])
        data_nl = np.zeros([len(zbins), len(kbins)])
        for i, z in enumerate(zbins):
            print('calculate  P(k) at z=%.3f (%d/%d z bin)'%(z, i, len(zbins)))
            Plin = nbodykit.lab.cosmology.LinearPower(cosmonbodykit,
                                              redshift=z, transfer='CLASS')
            Pm = nbodykit.lab.cosmology.HalofitPower(cosmonbodykit, redshift=z)
            data_l[i,:] = Plin(kbins)
            data_nl[i,:] = Pm(kbins)
        
        np.save(self.savename_linear, data_l)
        np.save(self.savename_nonlinear, data_nl)

        return
    
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
    
    from astropy import units as u
    from astropy import cosmology
    from astropy import constants as const
    cosmo = cosmology.Planck15
    
    theta_arr = (np.array(theta_arr) * u.arcsec).to(u.rad).value
    chi = (cosmo.comoving_distance(z)*cosmo.h).value

    kbinedges = np.logspace(-3, 3, 1000) #[h/Mpc]
    kbins = np.sqrt(kbinedges[:-1] * kbinedges[1:])
    dk = kbinedges[1:] - kbinedges[:-1]
    cosmo_power = cosmology_power_spectrum(z=z,k=kbins)
    Plin = cosmo_power.linear_power #[h^-3/Mpc^3]
    
    w_arr = np.zeros_like(theta_arr)
    for i,th in enumerate(theta_arr):
        w_arr[i] = np.sum(dk/2/np.pi*kbins*Plin*scipy.special.jv(0, chi*kbins*th))
    
    dzdchi = (cosmo.H(z)/const.c/cosmo.h).to(1/u.Mpc).value # H0/c[h/Mpc]
    w_arr *= dzdchi * bg * bI * dIdz
    
    return w_arr


def wgg(zin, theta_arr, bg=1, zbinedges=None, linear=True):
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
        cosmo_power = cosmology_power_spectrum(z=z,k=kbins)
        if linear:
            Plin = cosmo_power.linear_power #[h^-3/Mpc^3]
        else:
            Plin = cosmo_power.nonlinear_power #[h^-3/Mpc^3]
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

def wgI(zin, theta_arr, bg=1, bI=1, dIdz=1, zbinedges=None,linear=True):
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
        cosmo_power = cosmology_power_spectrum(z=z,k=kbins)
        if linear:
            Plin = cosmo_power.linear_power #[h^-3/Mpc^3]
        else:
            Plin = cosmo_power.nonlinear_power #[h^-3/Mpc^3]
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


# adopted from astroML.correlation 
# https://www.astroml.org/_modules/astroML/correlation.html#two_point_angular
from sklearn.neighbors import BallTree
from sklearn.utils import check_random_state
from astroML.correlation import ra_dec_to_xyz, angular_dist_to_euclidean_dist,uniform_sphere

def two_point_angular_window(coords_D, coords_R, bins, D_idx=None, R_idx=None, 
                             random_state=None):
    """Two-point correlation function

    Parameters
    ----------
    data_D: data ra, dec in deg, shape = [2, n_samples]
    data_R: random ra, dec in deg, shape = [2, n_samples]
    D_idx: idx of data in field, None if all in field
    R_idx: idx of random in field, None if all in field
    random_state : integer, np.random.RandomState, or None
        specify the random state to use for generating background

    Returns
    -------
    corr : ndarray
        the estimate of the correlation function within each bin
        shape = Nbins
    corr_in:
    
    """

    rng = check_random_state(random_state)
    
    coords_D, coords_R = np.asanyarray(coords_D), np.asanyarray(coords_R)
    
    if D_idx is None:
        D_idx = np.arange(coords_D.shape[1], dtype=int)
    
    if R_idx is None:
        R_idx = np.arange(coords_R.shape[1], dtype=int)
        
    if bins.ndim != 1:
        raise ValueError("bins must be a 1D array")

    data = np.asarray(ra_dec_to_xyz(coords_D[0], coords_D[1]), order='F').T
    data_R = np.asarray(ra_dec_to_xyz(coords_R[0], coords_R[1]), order='F').T

    bins = angular_dist_to_euclidean_dist(bins)    
    Nbins = len(bins) - 1

    factor = len(data_R) * 1. / len(data)
    factor_in = len(R_idx) * 1. / len(D_idx)

    BT_D = BallTree(data)
    BT_R = BallTree(data_R)

    counts_DD = np.zeros(Nbins + 1, dtype=int)
    counts_RR = np.zeros(Nbins + 1, dtype=int)
    counts_DD_in = np.zeros(Nbins + 1, dtype=int)
    counts_RR_in = np.zeros(Nbins + 1, dtype=int)
    for i in range(Nbins + 1):
        count_listD = BT_D.query_radius(data, bins[i])
        count_listR = BT_R.query_radius(data_R, bins[i])
        countD = np.sum([len(count) for count in count_listD])
        countR = np.sum([len(count) for count in count_listR])
        countD_in = np.sum([len(count) for count in count_listD[D_idx]])
        countR_in = np.sum([len(count) for count in count_listR[R_idx]])
        counts_DD[i], counts_RR[i] = countD, countR
        counts_DD_in[i], counts_RR_in[i] = countD_in, countR_in

    DD = np.diff(counts_DD)
    RR = np.diff(counts_RR)
    DD_in = np.diff(counts_DD_in)
    RR_in = np.diff(counts_RR_in)

    # check for zero in the denominator
    RR_zero = np.where(RR==0)[0]
    RR_in_zero = np.where(RR_in==0)[0]
    RR[RR_zero] = 1
    RR_in[RR_in_zero] = 1
    corr = factor ** 2 * DD / RR - 1
    corr_in = factor_in ** 2 * DD_in / RR_in - 1

    corr[RR_zero] = np.nan
    corr_in[RR_in_zero] = np.nan    
    
    return corr, corr_in


def bootsrap_two_point_angular_window(coords_D, coords_R, bins, D_idx=None, R_idx=None, 
                             random_state=None, Nbootstrap=None):

    coords_D, coords_R = np.asanyarray(coords_D), np.asanyarray(coords_R)

    if Nbootstrap is None:
        return two_point_angular_window(coords_D, coords_R, bins, D_idx=D_idx, R_idx=R_idx, 
                             random_state=random_state)
    
    N_D, N_R = coords_D.shape[1], coords_R.shape[1]
    multihot_D, multihot_R = np.zeros(N_D), np.zeros(N_R)
    multihot_D[D_idx] = 1
    multihot_R[R_idx] = 1

    bootstraps = []
    for i in range(Nbootstrap):
        ind_D = np.random.randint(0, N_D, N_D)
        ind_R = np.random.randint(0, N_R, N_R)
        D_idx_b = np.where(multihot_D[ind_D]==1)[0]
        R_idx_b = np.where(multihot_R[ind_R]==1)[0]
        corr_b, corr_in_b = two_point_angular_window(coords_D[:,ind_D], coords_R[:,ind_R], bins,
                            D_idx=D_idx_b, R_idx=R_idx_b, random_state=random_state)
        bootstraps.append([corr_b, corr_in_b])
        
    bootstraps = np.asarray(bootstraps)
    corr, corr_in = np.nanmean(bootstraps, 0)
    corr_err, corr_in_err = np.nanstd(bootstraps, 0, ddof=1)

    return corr, corr_in, corr_err, corr_in_err, bootstraps