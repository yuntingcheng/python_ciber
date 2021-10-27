import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
from astropy import units as u
from astropy import constants as const
from astropy import cosmology
import pickle
from hmf import MassFunction


# cosmo params
cosmo = cosmology.Planck15
cosmo.hmf = MassFunction(Mmin = np.log10(1e6), Mmax = np.log10(1e15), cosmo_model=cosmo, hmf_model = 'SMT')

class HMFz:
    '''
    HMF, P(k), and T(k) at given redshift.
    '''
    def __init__(self, z, cosmo = cosmo):
        hmf = copy.deepcopy(cosmo.hmf)
        hmf.update(z=z)
        self.z = z
        self.hmf = hmf

    def sigma(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        mass variance
        output:
        =======
        dndm_arr []
        m_arr [Msun h^-1]
        '''
        
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        logm_arr = np.log(m_arr)
        logm_dat_arr = np.log(self.hmf.m)
        sigma_dat_arr = self.hmf.sigma
        
        sigma_arr = np.interp(logm_arr, logm_dat_arr, sigma_dat_arr)
        return sigma_arr, m_arr

    def dndm(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        halo mass funciton
        output:
        =======
        dndm_arr [h^4 Mpc^-3 Msun^-1]
        m_arr [Msun h^-1]
        '''
        
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        logm_arr = np.log(m_arr)
        logm_dat_arr = np.log(self.hmf.m)
        logdndm_dat_arr = np.log(self.hmf.dndm)
        
        logdndm_arr = np.interp(logm_arr, logm_dat_arr, logdndm_dat_arr)
        dndm_arr = np.exp(logdndm_arr)
        return dndm_arr, m_arr
    
    def dndlnm(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):        
        '''
        halo mass funciton
        output:
        =======
        dndlnm_arr [h^3 Mpc^-3]
        m_arr [Msun h^-1]
        '''
        dndm_arr, m_arr = self.dndm(Mmin = Mmin, Mmax = Mmax, dlog10m = dlog10m, m_arr = m_arr)
        
        return dndm_arr*m_arr, m_arr
    
    def dndlog10m(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        halo mass funciton
        output:
        =======
        dndlog10m_arr [h^3 Mpc^-3]
        m_arr [Msun h^-1]
        '''
        
        dndm_arr, m_arr = self.dndm(Mmin = Mmin, Mmax = Mmax, dlog10m = dlog10m, m_arr = m_arr)
        
        return dndm_arr*m_arr*np.log(10), m_arr
    
    def P(self, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr=None):
        '''
        Power spectrum
        output:
        =======
        P_arr [Mpc^3 h^-3]
        k_arr [h/Mpc]
        '''
        
        if k_arr is None:
            k_arr = 10**np.arange(np.log10(kmin), np.log10(kmax), dlog10k)
        
        logk_arr = np.log(k_arr)

        logk_dat_arr = np.log(self.hmf.k)
        logP_dat_arr = np.log(self.hmf.power)
        
        logP_arr = np.interp(logk_arr, logk_dat_arr, logP_dat_arr)
        P_arr = np.exp(logP_arr)
        
        return P_arr, k_arr
    
    def Del2(self, kmin = 1e-3, kmax = 1e1, dlog10k = 0.01, k_arr = None):
        '''
        Dimensionless power spectrum
        output:
        =======
        Del2_arr []
        k_arr [h/Mpc]
        '''
        
        P_arr, k_arr = self.P(kmin = kmin, kmax = kmax, dlog10k = dlog10k, k_arr = k_arr)
        Del2_arr = P_arr*k_arr**3/2/np.pi**2
        
        return Del2_arr, k_arr
    
    def bias(self, Mmin = 1e8, Mmax = 1e15, dlog10m = 0.01, m_arr = []):
        '''
        Halo bias
        Sheth, Mo, Tormen 2001 eq 8
        
        Input:
        ======
        m_arr [Msun h^-1]
        '''
        if len(m_arr)==0:
            m_arr = 10**np.arange(np.log10(Mmin), np.log10(Mmax), dlog10m)
        
        del_sc = self.delta_sc(self.z)
        v = del_sc / self.sigma(m_arr = m_arr)[0]
        
        a = 0.707
        b = 0.5
        c = 0.6
        b_Lag = 1/np.sqrt(a)/del_sc *(np.sqrt(a)*a*v**2 + np.sqrt(a)*b*(a*v**2)**(1-c)\
                - (a*v**2)**c/((a*v**2)**c + b*(1-c)*(1-c/2)))
        b = 1 + b_Lag
        return b, m_arr
    
    def delta_sc(self, z):
        # Kitayama & Suto 1996 eq A6
        z = np.array(z)
        Omf = cosmo.Om0 * (1+z)**3 / (cosmo.Om0 * (1+z)**3 + cosmo.Ode0)
        deltasc = 3*(12*np.pi)**(2./3)/20 * (1 + 0.0123 * np.log10(Omf))
        return deltasc
    
    def sigma8(self,R=8):
        P_arr, k_arr = self.P()
        dlnk = np.log(k_arr[1]) - np.log(k_arr[0])
        Del = k_arr**3*P_arr/2/np.pi**2
        W = (3/(k_arr*R)**3)*(np.sin(k_arr*R) - (k_arr*R)*np.cos(k_arr*R))
        s8 = np.sqrt(np.sum(Del*W**2*dlnk))
        return s8

class cosmo_dist:
    '''
    cosmo distance at z, in unit Mpc/h (kpc/h)
    '''
    def __init__(self, z, cosmo = cosmo):
        self.z = z
        self.h = cosmo.h
        self.hubble_distance = self._hdist(cosmo.hubble_distance)
        self.H = self._hdist_inv(cosmo.H(z))
        self.comoving_distance = self._hdist(cosmo.comoving_distance(z))
        self.angular_diameter_distance = self._hdist(cosmo.angular_diameter_distance(z))
        self.luminosity_distance = self._hdist(cosmo.luminosity_distance(z))
        self.comoving_transverse_distance = self._hdist(cosmo.comoving_transverse_distance(z))
        self.kpc_comoving_per_arcmin = self._hdist(cosmo.kpc_comoving_per_arcmin(z))
        self.kpc_proper_per_arcmin = self._hdist(cosmo.kpc_proper_per_arcmin(z))
        if 0 not in np.array(z):
            self.arcsec_per_kpc_comoving = self._hdist_inv(cosmo.arcsec_per_kpc_comoving(z))
            self.arcsec_per_kpc_proper = self._hdist_inv(cosmo.arcsec_per_kpc_proper(z))
    
    def _hdist(self, d):
        return d * self.h / u.h
    
    def _hdist_inv(self,dinv):
        return dinv / self.h * u.h
    
# spectral lines
class spec_lines():
    def __init__(self, line_name=None):
        self.CII = 157.7409 * u.um
        self.CO = self._CO
        self.Lya = 0.12157 * u.um
        self.Lyb = 0.10257 * u.um
        self.Ha = 0.6563 * u.um
        self.Hb = 0.4861 * u.um
        self.HI = 211061.140542 * u.um
        self.OII = 0.3727 * u.um
        self. OIII = 0.5007 * u.um
        
        if line_name is not None:
            wl_um_dict = {'Lya':self.Lya, 'Lyb':self.Lyb,
                  'Ha':self.Ha, 'Hb':self.Hb,
                  'OII':self.OII, 'OIII':self.OIII, 'CII':self.CII}
            wl_um = wl_um_dict[line_name]
            self.wl_um = wl_um
            
    def _CO(self, J = 1):
        return 2610./J * u.um
    
    def z_prj(self, j_prj, z_targ, j_targ):
        '''
        CII, CO prjected redshift. Return the projected z z_prj of line j_prj to 
        rest frame z_targ of line j_targ.
        '''
        wl_prj = self.CO(j_prj) if j_prj !=0 else self.CII
        wl_targ = self.CO(j_targ) if j_targ !=0 else self.CII
        zprj = (wl_targ / wl_prj)*(1 + z_targ) - 1
        return zprj.value

def get_Plin_fast(z, k_arr=None):
    '''
    Precomuting P(k,z) in to a data, file, and do interpolation upon calling
    '''
    k_data = np.logspace(-5,4,1000)
    z_data = np.arange(0.01,10,0.01)
    try:
        Plin_data = np.load('./Plin_data.npy',allow_pickle=True)
    except:
        print('pre-compute Plin for interpolation ...')
        Plin_data = np.zeros([len(z_data),len(k_data)])
        for i,zz in enumerate(z_data):
            if (i%20) == 19:
                print('z=%.1f'%zz)
            Plin_data[i] = HMFz(zz).P(k_arr = k_data)[0]
        np.save('./Plin_data',Plin_data) 
    
    k_arr = k_data if k_arr is None else k_arr
    zidx = np.argmin(np.abs(z_data-z))
    P_data = Plin_data[zidx]
    logP_arr = np.interp(np.log(k_arr), np.log(k_data), np.log(P_data))
    P_arr = np.exp(logP_arr)
    return P_arr, k_arr


def get_dndlnM_fast(z, Mh_arr=None):
    '''
    Precomuting dndlnM(z, Mh_arr[Msun/h]) in to a data,
    file, and do interpolation upon calling
    '''
    Mh_data = np.logspace(8,15,1000)
    z_data = np.arange(0.01,10,0.01)
    try:
        dndlnM_data = np.load('./dndlnM_data.npy',allow_pickle=True)
    except:
        print('pre-compute dndlnM for interpolation ...')
        dndlnM_data = np.zeros([len(z_data),len(Mh_data)])
        for i,zz in enumerate(z_data):
            if (i%20) == 19:
                print('z=%.1f'%zz)
            dndlnM_data[i] = HMFz(zz).dndlnm(m_arr=Mh_data)[0]
        np.save('./dndlnM_data',dndlnM_data) 
    
    Mh_arr = Mh_data if Mh_arr is None else Mh_arr
    zidx = np.argmin(np.abs(z_data-z))

    n_data = dndlnM_data[zidx]
    logn_arr = np.interp(np.log(Mh_arr), np.log(Mh_data), np.log(n_data))
    dndlnM_arr = np.exp(logn_arr)
    return dndlnM_arr, Mh_arr

def get_bias_fast(z, Mh_arr=None):
    '''
    Precomuting bias(z, Mh_arr[Msun/h]) in to a data,
    file, and do interpolation upon calling
    '''
    Mh_data = np.logspace(8,15,1000)
    z_data = np.arange(0.01,10,0.01)
    try:
        bhalo_data = np.load('./bhalo_data.npy',allow_pickle=True)
    except:
        print('pre-compute b(z,M) for interpolation ...')
        bhalo_data = np.zeros([len(z_data),len(Mh_data)])
        for i,zz in enumerate(z_data):
            if (i%20) == 19:
                print('z=%.1f'%zz)
            bhalo_data[i] = HMFz(zz).bias(m_arr=Mh_data)[0]
        np.save('./bhalo_data',bhalo_data) 
    
    Mh_arr = Mh_data if Mh_arr is None else Mh_arr
    zidx = np.argmin(np.abs(z_data-z))

    b_data = bhalo_data[zidx]
    b_arr = np.interp(np.log(Mh_arr), np.log(Mh_data), b_data)
    return b_arr, Mh_arr

def get_Mstr_fast(z):
    '''
    Precomuting Mstr(z) in to a data,
    file, and do interpolation upon calling
    Mstr is defined s.t.
    delta_sc(z) = sigma(Mh=Mstr) 
    --> nu=1 Eq 57 Cooray & Sheth 2002
    '''
    z_data = np.arange(0.01,8,0.01)
    Mh_arr = np.logspace(2,15,1000)
    try:
        Mstr_data = np.load('./Mstr_data.npy',allow_pickle=True)
    except:
        print('pre-compute Mstr(z) for interpolation ...')
        Mstr_data = np.zeros(len(z_data))
        for i,zz in enumerate(z_data):
            if (i%20) == 19:
                print('z=%.1f'%zz)
            delta_sc = HMFz(zz).delta_sc(zz)
            sigma_arr = HMFz(zz).sigma(m_arr = Mh_arr)
            idx = np.argmin(np.abs(delta_sc/sigma_arr - 1))
            Mstr_data[i] = Mh_arr[idx]
        np.save('./Mstr_data',Mstr_data) 
    
    Mstr = np.exp(np.interp(z, z_data, np.log(Mstr_data)))
    return Mstr

class NFW_proile:
    
    def __init__(self):
        
        self.NFW_2d_scaled_calc()
    
    def Rvir(self, z, Mh):
        '''
        Mh [Msun/h]
        Rvir [Mpc/h]
        '''
        if isinstance(z, list):
            z = np.array(z)
            Mh = np.array(Mh)
        
        rhoc = cosmo.critical_density(z).to(u.M_sun / u.Mpc**3).value
        rvir = ((3 * Mh / cosmo.h) / (4 * np.pi * 200 * rhoc))**(1./3)
        rvir *= cosmo.h
        
        return rvir
    
    def conc(self, z, Mh):
        
        if isinstance(z, list):
            z = np.array(z)
            Mh = np.array(Mh)
            
        Mstr = get_Mstr_fast(z)
        
        conc = (9/(1+z)) * (Mh / Mstr)**-0.13
        
        return conc
    
    def r_scaled(self, r, z, Mh, r_units='arcsec'):

        if isinstance(z, list):
            z = np.array(z)
            Mh = np.array(Mh)
        
        if isinstance(r, list):
            r = np.array(r)
            
        R_vir = self.Rvir(z, Mh)
        conc = self.conc(z, Mh)
        if r_units == 'arcsec':
            DA = cosmo.comoving_distance(z).value # [Mpc]
            r = r * (u.arcsec.to(u.rad)) * DA * cosmo.h
        if r_units == 'arcmin':
            DA = cosmo.comoving_distance(z).value # [Mpc]
            r = r * (u.arcmin.to(u.rad)) * DA * cosmo.h
        if r_units == 'deg':
            DA = cosmo.comoving_distance(z).value # [Mpc]
            r = r * (u.deg.to(u.rad)) * DA * cosmo.h
        if r_units == 'rad':
            DA = cosmo.comoving_distance(z).value # [Mpc]
            r = r * DA * cosmo.h
            
        r_scaled = r * conc / R_vir
        
        return r_scaled
    
    def NFW_3d_scaled(self, r):

        if isinstance(r, list):
            r = np.array(r)
            profile = np.zeros_like(r)
            profile[r!=0] = 1 / (r[r!=0]) / (1 + r[r!=0])**2
            return profile

        elif isinstance(r,  np.ndarray):
            profile = np.zeros_like(r)
            profile[r!=0] = 1 / (r[r!=0]) / (1 + r[r!=0])**2
            return profile

        else:
            if r == 0:
                return 0
            return 1 / (r) / (1 + r)**2 
                        
    
    def NFW_3d(self, r, z, Mh):
        
        r_scaled = self.r_scaled(r, z, Mh)
        rho = self.NFW_3d_scaled(r_scaled)
        
        return rho
    
    def NFW_2d_scaled_calc(self):
        
        # populate a 3D quadrant and integrate to 2D
        # save this somewhere
        
        r = np.arange(0.001,10,0.001)
        xx, yy = np.meshgrid(r, r)
        rr = np.sqrt(xx**2 + yy**2)
        rho_map = self.NFW_3d_scaled(rr)
        
        self.r_2d = r
        self.rho_2d = np.sum(rho_map,axis=0)
        self.rho_2d_poly = np.polyfit(np.log10(self.r_2d),
                                      np.log10(self.rho_2d), deg=3)
        
        return
        
    def NFW_2d(self, r, z, Mh, r_units='arcsec'):
        

        if isinstance(r, list):
            r = np.array(r)
            r_scaled = self.r_scaled(r, z, Mh, r_units='arcsec')
            rho_2d = np.zeros_like(r_scaled)
            rho_2d[r_scaled!=0] = 10**np.polyval(self.rho_2d_poly,
                                     np.log10(r_scaled[r_scaled!=0]))
            rho_2d /= np.sum(rho_2d)
            
            return rho_2d

        elif isinstance(r,  np.ndarray):
            r_scaled = self.r_scaled(r, z, Mh, r_units='arcsec')
            rho_2d = np.zeros_like(r_scaled)
            rho_2d[r_scaled!=0] = 10**np.polyval(self.rho_2d_poly,
                                     np.log10(r_scaled[r_scaled!=0]))
            rho_2d /= np.sum(rho_2d)
            
            return rho_2d

        else:
            if r == 0:
                return 0
            r_scaled = self.r_scaled(r, z, Mh, r_units='arcsec')
            return 10**np.polyval(self.rho_2d_poly, np.log10(r_scaled))