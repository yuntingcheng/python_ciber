from utils import *

class Helgason_model:
    
    '''
    Helgason+12 model
    https://ui.adsabs.harvard.edu/#abs/2012ApJ...752..113H/abstract
    '''        

    def __init__(self):
        self.get_params()
        
    def get_params(self):
        '''
        Helgason+12 table 2
        '''
        name_arr = ['UV','U','B','V','R','I','z','J','H','K','L','M']
        wleff_arr = [0.15,0.36,0.45,0.55,0.65,0.79,0.91,1.27,1.63,2.20,3.60,4.50]
        zmax_arr = [8.0,4.5,4.5,3.6,3.0,3.0,2.9,3.2,3.2,3.8,0.7,0.7]
        M0_arr = [-19.62,-20.20,-21.35,-22.13,-22.40,-22.80,
                  -22.86,-23.04,-23.41,-22.97,-22.40,-21.84]
        q_arr = [1.1,1.0,0.6,0.5,0.5,0.4,0.4,0.4,0.5,0.4,0.2,0.3]
        phi0_arr = [2.43,5.46,3.41,2.42,2.25,2.05,2.55,2.21,1.91,2.74,3.29,3.29]
        p_arr = [0.2,0.5,0.4,0.5,0.5,0.4,0.4,0.6,0.8,0.8,0.8,0.8]
        alpha0_arr = [-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,
                      -1.00,-1.00,-1.00,-1.00,-1.00,-1.00]
        r_arr = [0.086,0.076,0.055,0.060,0.070,0.070,0.060,
                 0.035,0.035,0.035,0.035,0.035]
        
        self._name_arr = np.asarray(name_arr)
        self._wleff_arr = np.asarray(wleff_arr)
        self._zmax_arr = np.asarray(zmax_arr)
        self._M0_arr = np.asarray(M0_arr)
        self._q_arr = np.asarray(q_arr)
        self._phi0_arr = np.asarray(phi0_arr)
        self._p_arr = np.asarray(p_arr)
        self._alpha0_arr = np.asarray(alpha0_arr)
        self._r_arr = np.asarray(r_arr)
        
        return
    
    def LF_band(self, z, bandidx, M_arr = np.arange(-25,-15,0.01)):
        '''
        rest frame luminosity function [#/mag/Mpc^3] at band # bandidx
        Helgason eq 1, eq 3~5, table 2
        '''   
        self.z = z
        self.name = self._name_arr[bandidx]
        self.wleff = self._wleff_arr[bandidx]
        self.zmax = self._zmax_arr[bandidx]
        self.M0 = self._M0_arr[bandidx]
        self.q = self._q_arr[bandidx]
        self.phi0 = self._phi0_arr[bandidx]
        self.p = self._p_arr[bandidx]
        self.alpha0 = self._alpha0_arr[bandidx]
        self.r = self._r_arr[bandidx]
        Mstr = self.M0 - 2.5 * np.log10((1 + (z - 0.8))**self.q)
        phistr = self.phi0 * np.exp(-self.p * (z - 0.8)) * 1e-3
        alpha = self.alpha0 * (z / 0.01)**self.r
        phi_arr = 0.4 * np.log(10) * phistr * (10**(0.4 * (Mstr - M_arr)))**(alpha + 1) \
                * np.exp(-10**(0.4 * (Mstr - M_arr)))
        
        return M_arr, phi_arr
        
    def LF_wl(self, z, wl_rf, M_arr):
        '''
        rest frame luminosity function [#/mag/Mpc^3] at (rest frame wl) wl_rf [um]
        interpolate Helgason LF to given wl in um
        '''
        
        wl_eff_arr = self._wleff_arr

        if len(np.where(wl_eff_arr==wl_rf)[0])==1:
            phi_arr = self.LF_band(z, np.where(wl_eff_arr==wl_rf)[0][0], M_arr)
            return phi_arr

        if wl_rf < wl_eff_arr[0]:
            idx0, idx1 = 0, 1
        elif wl_rf > wl_eff_arr[-1]:
            idx0, idx1 = len(wl_eff_arr)-2, len(wl_eff_arr)-1
        else:
            idx0, idx1 = np.where(wl_eff_arr>wl_rf)[0][0]-1, np.where(wl_eff_arr>wl_rf)[0][0]

        wl0, wl1 = wl_eff_arr[idx0], wl_eff_arr[idx1]
        
        phi0 = np.log(self.LF_band(z, idx0, M_arr=M_arr)[1])
        phi1 = np.log(self.LF_band(z, idx1, M_arr=M_arr)[1])
        
        phi_arr = phi0 + (wl_rf - wl0) * (phi1 - phi0) / (wl1 - wl0)
        phi_arr = np.exp(phi_arr)

        return M_arr, phi_arr    

    def dN_dsr_dz(self, z, wl_obs, m_min=0, m_max=15):
        '''
        Mean number density between AB mag [m_min, m_max], dN/dz/dsr
        '''
        m_arr = np.linspace(m_min, m_max, 1000)
        dm = m_arr[1] - m_arr[0]
        DM = 5 * np.log10((cosmo.luminosity_distance(z) / (10 * u.pc)).decompose()).value
        M_arr = m_arr - DM + (2.5 * np.log10(1+z))
        m_arr = m_arr[M_arr > -26]
        M_arr = M_arr[M_arr > -26]
        if len(m_arr)==0:
            return 0
        
        dchi_dz = (const.c / cosmo.H(z)).to(u.Mpc).value
        dMpc2_dsr = (cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.rad).value)**2

        phi_arr = self.LF_wl(z, wl_obs/(1+z), M_arr)[1] # [#/mag/Mpc^3]
        dn_dz = np.sum(phi_arr) * dm * dchi_dz * dMpc2_dsr # [1/sr]

        return dn_dz
    
    def dN_dsr(self, wl_obs, m_min=0, m_max=15, zmin=0, zmax=6):
        z_arr = np.linspace(zmin, zmax, 100)
        dz = z_arr[1] - z_arr[0]
        if zmin==0:
            z_arr[0] = 1e-4
        n_tot = 0
        for z in z_arr:
            n_tot += self.dN_dsr_dz(z, wl_obs, m_min=m_min, m_max=m_max)
        n_tot *= dz
        
        return n_tot
    
    def dN_dm_dsr(self, wl_obs, m_arr):
        
        try:
            len(np.array(m_arr))
        except TypeError:
            m_arr = np.array([m_arr])
        m_arr = np.array(m_arr)
        
        dm = 0.1
        dN_dm_dsr_arr = []
        for m in m_arr:
            dN_dsr = self.dN_dsr(wl_obs, m_min=m-dm/2, m_max=m+dm/2)
            dN_dm_dsr_arr.append(dN_dsr)
        dN_dm_dsr_arr = np.array(dN_dm_dsr_arr) / dm
        
        if len(m_arr)==1:
            return dN_dm_dsr_arr[0]
        else:
            return dN_dm_dsr_arr
        
    def dN_dm_ddeg2(self, wl_obs, m_arr):
        return self.dN_dm_dsr(wl_obs,m_arr) / (u.sr.to(u.deg**2))        
        
        
    def dnuInu_dz(self, z, wl_obs, m_min = -np.inf, m_max = np.inf, M_arr = np.arange(-25,0,0.01)):
        '''
        integrate Helgason LF to get d(nuInu) / dz [nW/m2/sr] at obs wavelength wl_obs [um]
        \frac{d\nu I\nu}{dz}=
        \int dm \left [ \frac{dN}{dmdV^{CMV}}(z,\mu_{rf}=\mu/(1+z)) \right ]
         \nu F\nu\frac{d\chi}{dz} \left ( D_A^{CMV}(z) \right )^2
        '''
        
        # abs mag -> apparent AB mag. Helgason+12 Eq 8
        DM = 5 * np.log10((cosmo.luminosity_distance(z) / (10 * u.pc)).decompose()).value
        m_arr = M_arr + DM - (2.5 * np.log10(1+z))
        dm = m_arr[1] - m_arr[0]
        sp = np.where((m_arr>m_min) & (m_arr<m_max))[0]
        if len(sp)==0:
            return 0
        m_arr = m_arr[sp]
        M_arr = M_arr[sp]
        dchi_dz = (const.c / cosmo.H(z)).to(u.Mpc).value
        dMpc2_dsr = (cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.rad).value)**2

        nu_obs = const.c.value / (wl_obs*1e-6)
        Fnu_arr = 3631 * 10**(-m_arr / 2.5) # [Jy]
        phi_arr = self.LF_wl(z, wl_obs/(1+z), M_arr)[1] # [#/mag/Mpc^3]
        dInu_dz = np.sum(phi_arr * Fnu_arr) * dm * dchi_dz * dMpc2_dsr # [Jy/sr]
        dnuInu_dz = (nu_obs * dInu_dz)*(u.Hz * u.Jy / u.sr).to(u.nW/u.m**2/u.sr)

        return dnuInu_dz
    
    def dnuInu_dz_filter_weighted(self, z, filt_wls, filt_trans, verbose=False, **kwargs):
        filt_wls, filt_trans = np.array(filt_wls), np.array(filt_trans)
        dwls = np.diff(filt_wls)
        wls = ((filt_wls[1:] + filt_wls[:-1]) / 2)
        Rs = (filt_trans[1:] + filt_trans[:-1]) / 2
        dnuInu_dz_w = 0
        for i,(wl, R, dwl) in enumerate(zip(wls, Rs, dwls)):
            if verbose:
                print('Filter weighted dnuInu/dz %d/%d'%(i,len(wls)))
            dnuInu_dz = self.dnuInu_dz(z, wl, **kwargs)
            dnuInu_dz_w += dnuInu_dz * R * dwl
        dnuInu_dz_w /= np.sum(Rs * dwls)
        
        return dnuInu_dz_w
    
    def nuInu(self, wl_obs, zmin=0, zmax=6, **kwargs):
        '''
        Total EBL nuInu[nW/m2/sr] at obs wl wl_obs[um] from redshift range (zmin, zmax)
        '''
        z_arr = np.linspace(zmin, zmax, 100)
        dz = z_arr[1] - z_arr[0]
        if zmin==0:
            z_arr[0] = 1e-4
        nuInu_tot = 0
        for z in z_arr:
            nuInu_tot += self.dnuInu_dz(z, wl_obs, **kwargs)
        nuInu_tot *= dz
        
        return nuInu_tot
    
    
    def dnuInu_dm(self, wl_obs, m_arr):
        '''
        nW/m2/sr
        '''
        try:
            len(np.array(m_arr))
        except TypeError:
            m_arr = np.array([m_arr])
        m_arr = np.array(m_arr)
        
        dm = 0.1
        dnuInu_dm_arr = []
        for m in m_arr:
            dnuInu_dm_arr.append(self.nuInu(wl_obs,m_min=m-dm/2, m_max=m+dm/2))
        dnuInu_dm_arr = np.array(dnuInu_dm_arr) / dm

        if len(m_arr)==1:
            return dnuInu_dm_arr[0]
        else:
            return dnuInu_dm_arr
        
    def dClsh_dz(self, z, wl_obs, m_min = 13.5, M_arr = np.arange(-25,0,0.01)):
        '''
        integrate 2nd moment of Helgason LF to get dCl_sh / dz [(nW/m2/sr)^2 sr].
        Only integrate sources below AB mag m_min.
        at obs wavelength wl_obs [um]
        \frac{dC_{\ell,sh}}{dz}=
        \int_{m_{lim}}^\infty dm \left [ \frac{dN}{dmdV^{CMV}}(z,\mu_{rf}=\mu/(1+z)) \right ]
         \left ( \nu F\nu \right )^2\frac{d\chi}{dz} \left ( D_A^{CMV}(z) \right )^2
        '''
        
        # abs mag -> apparent AB mag. Helgason+12 Eq 8
        DM = 5 * np.log10((cosmo.luminosity_distance(z) / (10 * u.pc)).decompose()).value
        m_arr = M_arr + DM - (2.5 * np.log10(1+z))
        dm = m_arr[1] - m_arr[0]
        sp = np.where((m_arr>m_min))[0]
        if len(sp)==0:
            return 0
        m_arr = m_arr[sp]
        M_arr = M_arr[sp]
        
        dchi_dz = (const.c / cosmo.H(z)).to(u.Mpc).value
        dMpc2_dsr = (cosmo.kpc_comoving_per_arcmin(z).to(u.Mpc/u.rad).value)**2

        nu_obs = const.c.value / (wl_obs*1e-6)
        Fnu_arr = 3631 * 10**(-m_arr / 2.5) # [Jy]
        phi_arr = self.LF_wl(z, wl_obs/(1+z), M_arr)[1] # [#/mag/Mpc^3]
        dClsh_dz = np.sum(phi_arr * Fnu_arr**2 * nu_obs**2) * dm * dchi_dz * dMpc2_dsr # [Jy/sr]
        dClsh_dz = (dClsh_dz)*((u.Hz * u.Jy)**2 / u.sr).to((u.nW/u.m**2)**2/u.sr)

        return dClsh_dz
    
    def Clsh(self, wl_obs, m_min = 13.5, zmin=0, zmax=6, **kwargs):
        '''
        Total Cl_shot [(nW/m2/sr)^2 sr] at obs wl wl_obs[um] from redshift range (zmin, zmax)
        '''
        z_arr = np.linspace(zmin, zmax, 1000)
        dz = z_arr[1] - z_arr[0]
        if zmin==0:
            z_arr[0] = 1e-4
        Clsh_tot = 0
        for z in z_arr:
            Clsh_tot += self.dClsh_dz(z, wl_obs, m_min=m_min, **kwargs)
        Clsh_tot *= dz
        
        return Clsh_tot
    
    def Clsh_filter_weighted(self, filt_wls, filt_trans, verbose=False, **kwargs):
        filt_wls, filt_trans = np.array(filt_wls), np.array(filt_trans)
        dwls = np.diff(filt_wls)
        wls = ((filt_wls[1:] + filt_wls[:-1]) / 2)
        Rsqs = ((filt_trans[1:] + filt_trans[:-1]) / 2)**2
        Clsh_w = 0
        for i,(wl, Rsq, dwl) in enumerate(zip(wls, Rsqs, dwls)):
            Clsh = self.Clsh(wl, **kwargs)
            Clsh_w += Clsh * Rsq * dwl
            if verbose:
                print('Filter weighted Clsh %d/%d (%.3f um, Clsh=%.2e)'%(i,len(wls), wl, Clsh))

        Clsh_w /= np.sum(Rsqs * dwls)
        
        return Clsh_w














# Helgason table 2 data
# class bands_class:
    
#     '''
#     Helgason et al. 2012 table 2
#     https://ui.adsabs.harvard.edu/#abs/2012ApJ...752..113H/abstract
#     '''
    
#     def __init__(self,idx):
        
#         name_arr = ['UV','U','B','V','R','I','z','J','H','K','L','M']
#         wleff_arr = [0.15,0.36,0.45,0.55,0.65,0.79,0.91,1.27,1.63,2.20,3.60,4.50]
#         zmax_arr = [8.0,4.5,4.5,3.6,3.0,3.0,2.9,3.2,3.2,3.8,0.7,0.7]
#         M0_arr = [-19.62,-20.20,-21.35,-22.13,-22.40,-22.80,-22.86,-23.04,-23.41,-22.97,-22.40,-21.84]
#         q_arr = [1.1,1.0,0.6,0.5,0.5,0.4,0.4,0.4,0.5,0.4,0.2,0.3]
#         phi0_arr = [2.43,5.46,3.41,2.42,2.25,2.05,2.55,2.21,1.91,2.74,3.29,3.29]
#         p_arr = [0.2,0.5,0.4,0.5,0.5,0.4,0.4,0.6,0.8,0.8,0.8,0.8]
#         alpha0_arr = [-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00,-1.00]
#         r_arr = [0.086,0.076,0.055,0.060,0.070,0.070,0.060,0.035,0.035,0.035,0.035,0.035]
        
#         self.name = name_arr[idx]
#         self.wleff = wleff_arr[idx]
#         self.zmax = zmax_arr[idx]
#         self.M0 = M0_arr[idx]
#         self.q = q_arr[idx]
#         self.phi0 = phi0_arr[idx]
#         self.p = p_arr[idx]
#         self.alpha0 = alpha0_arr[idx]
#         self.r = r_arr[idx]
        
#         self._name_arr = np.asarray(name_arr)
#         self._wleff_arr = np.asarray(wleff_arr)
#         self._zmax_arr = np.asarray(zmax_arr)
#         self._M0_arr = np.asarray(M0_arr)
#         self._q_arr = np.asarray(q_arr)
#         self._phi0_arr = np.asarray(phi0_arr)
#         self._p_arr = np.asarray(p_arr)
#         self._alpha0_arr = np.asarray(alpha0_arr)
#         self._r_arr = np.asarray(r_arr)

# def Helgason_LF(z, M_arr, bandidx):
#     '''
#     Helgason + 2012
#     rest frame luminosity function [#/mag/Mpc^3].
#     Helgason eq 1, eq 3~5, table 2
#     '''
#     params = bands_class(bandidx)
#     Mstr = params.M0 - 2.5 * np.log10((1 + (z - 0.8))**params.q)
#     phistr = params.phi0 * np.exp(-params.p * (z - 0.8)) * 1e-3
#     alpha = params.alpha0 * (z / 0.01)**params.r
#     phi_arr = 0.4 * np.log(10) * phistr * (10**(0.4 * (Mstr - M_arr)))**(alpha + 1) \
#             * np.exp(-10**(0.4 * (Mstr - M_arr)))
#     return phi_arr