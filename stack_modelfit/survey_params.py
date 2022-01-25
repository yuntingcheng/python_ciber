from cosmo_tools import *
from Helgason import *

class dirbe_param:
    '''
    DIRBE params
    https://ui.adsabs.harvard.edu/abs/1998ApJ...508...25H/abstract
    DIRBE filters
    http://svo2.cab.inta-csic.es/svo/theory//fps3/index.php?mode=browse&gname=COBE
    '''
    def __init__(self):
        
        self.dth = 0.7 #[deg]
        self.fsky = 1.
        self.lmax = np.pi / (self.dth * np.pi / 180)
        self.wl_name = np.array([1.25, 2.2, 3.5, 4.9, 12, 25, 60, 100, 140, 240]) # [um]        
        self.dnu = [5.95e13, 2.24e13, 2.20e13, 8.19e12, 1.33e13,
                    4.13e12, 2.32e12,9.74e11, 6.05e11, 4.95e11] # [Hz]
        self.dnu = 1e-9 * np.array(self.dnu)
        self.sign = np.array([2.4, 1.6, 0.9, 0.8, 0.9, 
                     0.9, 0.9, 0.5, 32.8, 10.7]) # nuInu [nW/m^2/sr]
        self.beam = np.array([1.198, 1.420, 1.285, 1.463, 1.427,
                     1.456, 1.512, 1.425, 1.385, 1.323])*1e-4 # [sr]
        self.Cln = self.sign**2 * self.beam # Cln = sigma_n^2 * Omega_beam
        self._get_params()
    
    def _get_params(self):
        
        wl = []
        for i in range(len(self.wl_name)):
            wls, T = self._filter_trans(i)
            wl.append(np.sum(wls*T)/np.sum(T))
        self.wl = np.array(wl)
        
        um2GHz = 1* u.um.to(u.GHz, equivalencies=u.spectral())
        self.nu = um2GHz / self.wl # [GHz]
        self.nu_max = self.nu + self.dnu / 2
        self.nu_min = self.nu - self.dnu / 2
        
        GHz2um = 1* u.GHz.to(u.um, equivalencies=u.spectral())
        self.wl_max = GHz2um / self.nu_min
        self.wl_min = GHz2um / self.nu_max
        
        self.dwl = self.wl_max - self.wl_min
        self.R = self.nu / self.dnu
        
        return
    
    def _filter_trans(self, iband, return_type='wl'):
        wl = self.wl_name[iband]
        fname = '{}'.format(wl).replace('.','p')
        if fname[-1] == '0':
            fname = fname[:-2]
        fname = 'data_external/DIRBE_filters/COBE_DIRBE.' +\
        fname + 'm.dat'
        
        data = np.loadtxt(fname)
        wls, T = data[:,0]*1e-4, data[:,1]
        um2GHz = 1* u.um.to(u.GHz, equivalencies=u.spectral())
        nus = um2GHz / wls # [GHz] 

        if return_type == 'wl':
            return wls, T
        
        elif return_type == 'freq':
            return nus, T

class eBOSS_param:
    '''
    https://ui.adsabs.harvard.edu/abs/2016AJ....151...44D/abstract
    deltaz = sigma_z/(1+z) ~ 1e-3 (p.9)
    ELG:
    https://ui.adsabs.harvard.edu/abs/2017MNRAS.471.3955R/abstract
    '''
    def __init__(self, z=0, tracer_name='LRG',
                 field_name=None, ns_field_name=None, get_bias=True):
        
        self.tracer_names = ['CMASS', 'LRG', 'ELG', 'QSO']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'eBOSS ' + tracer_name
        self.print_name_survey = '(e)BOSS'
        if tracer_name == 'CMASS':
            self.print_name = self.print_name[1:]
        self._calc_params(get_bias=get_bias)

    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self, field_name=None):
        
        self.survey_area_eBOSS = 7500 # [deg^2]
        self.survey_area_early = 3193 # obs in fisrt year[deg^2]
        self.survey_area_ELG_SGC = 620 # [deg^2]
        self.survey_area_ELG_NGC = 600 # [deg^2]
        self.survey_area_CMASS = 10000 # [deg^2]
        
        if self.tracer_name != 'ELG':
            Adeg = self.survey_area_eBOSS
            if field_name == 'early':
                Adeg = self.survey_area_early
            elif self.tracer_name == 'CMASS':
                Adeg = self.survey_area_CMASS
        else:
            Adeg = self.survey_area_ELG_SGC
            if field_name == 'NGC':
                Adeg = self.survey_area_ELG_NGC
        
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self, ns_field_name=None):
        
        if self.tracer_name == 'LRG':
            table = self.LRG_density_table()
            ns_field_name = 'ns_zconf1' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'QSO':
            table = self.QSO_density_table()
            ns_field_name = 'ns' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'ELG':
            table = self.ELG_density_table()
            ns_field_name = 'ns_SGC' if ns_field_name is None else ns_field_name
        elif self.tracer_name == 'CMASS':
            table = self.CMASS_density_table()
            ns_field_name = 'ns'
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_z_deg2 = 0
            self.n_zbin_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(deg^2)/(dzbin)]
        self.n_zbin_deg2 = table[ns_field_name][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]

        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)
        
        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)
        
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = self.n_zbin_deg2 / dV
                
            
    def LRG_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
                    2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_zconf0 = np.zeros_like(zbins)
        ns_zconf0[:7]= [0.6, 6.2, 15.2, 15.3, 9.4, 3.2, 0.6]
        
        ns_zconf1 = np.zeros_like(zbins)
        ns_zconf1[:7]= [0.6, 5.9, 14.8, 14.7, 8.7, 2.7, 0.5]
        
        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns_zconf0': ns_zconf0, 'ns_zconf1': ns_zconf1}
        
        return LRGtable
    
    def QSO_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8,
                    2.0, 2.1, 2.2, 2.4, 2.6, 2.8, 3.0, 3.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_new = np.array([1.0, 1.1, 1.4, 1.4, 2.2, 3.6, 8.4, 10.3, 10.3,
                  9.9, 9.2, 4.0, 2.2, 1.8, 1.1, 0.7, 0.3, 0.4])
        
        ns_known= np.array([0.4, 0.4, 0.7, 1.3, 1.5, 1.0, 1.8, 1.8, 2.1, 2.0,
                   1.9, 1.0, 1.6, 4.5, 3.1, 1.4, 0.8, 1.2])
        
        
        QSOtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns_new + ns_known, 'ns_new': ns_new, 'ns_known': ns_known}
        
        return QSOtable
    
    def ELG_density_table(self):
        '''
        Raichoor table 4
        ns [deg^-2]
        '''
        zbinedges = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 
                              0.8, 0.9, 1. , 1.1, 1.2, 1.3, 1.4, 1.5])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns_SGC = np.array([0.2, 1.1, 2.0, 1.9, 1.1, 1.4, 9.2, 56.6,
                 61.6, 31.6, 13.4, 6.4, 2.9, 1.5, 0.7])
        
        ns_NGC = np.array([0.3, 1.1, 2.6, 2.6, 1.7, 2.2, 10.3, 42.0,
                           48.5, 26.3, 12.0, 5.4, 2.5, 0.9, 0.4])
        
        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns_SGC': ns_SGC, 'ns_NGC': ns_SGC}
        
        return ELGtable
    
    def CMASS_density_table(self):
        '''
        Dawson+16 table 1
        ns [deg^-2]
        '''
        zbinedges = np.array([0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2
        
        ns = np.array([27.3, 45.7, 19.4, 3.5, 0.2, 0.03])

        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LRGtable

    def tracer_bias(self, sigma8_interp=True):
        
        if not sigma8_interp:
            if self.tracer_name == 'LRG':
                bias = 1.7 * HMFz(0).sigma8() / HMFz(self.z).sigma8()
            elif self.tracer_name == 'QSO':
                bias = 0.53 + 0.29 * (1+self.z)**2
            elif self.tracer_name == 'ELG':
                bias = 1.0* HMFz(0).sigma8() / HMFz(self.z).sigma8()
        else:
            # pre-computed sigma8 value for fast interpolation
#             z_arr = np.arange(0,10,0.05)
#             sigma8_arr = []
#             for z in z_arr:
#                 sigma8_arr.append(HMFz(z).sigma8())

            z_arr = np.arange(0,10,0.05)
            sigma8_arr = [0.81583, 0.79475, 0.77410, 0.75394, 0.73430, 0.71522,
                          0.69671, 0.67879, 0.66146, 0.64472, 0.62857, 0.61299,
                          0.59799, 0.58353, 0.56962, 0.55622, 0.54333, 0.53092,
                          0.51898, 0.50749, 0.49642, 0.48577, 0.47551, 0.46562,
                          0.45610, 0.44692, 0.43806, 0.42952, 0.42128, 0.41333,
                          0.40564, 0.39822, 0.39105, 0.38411, 0.37740, 0.37091,
                          0.36462, 0.35854, 0.35264, 0.34692, 0.34138, 0.33601,
                          0.33079, 0.32573, 0.32082, 0.31604, 0.31140, 0.30689,
                          0.30251, 0.29825, 0.29410, 0.29006, 0.28612, 0.28229,
                          0.27856, 0.27492, 0.27138, 0.26792, 0.26455, 0.26126,
                          0.25805, 0.25491, 0.25185, 0.24886, 0.24594, 0.24309,
                          0.24030, 0.23757, 0.23490, 0.23229, 0.22974, 0.22724,
                          0.22480, 0.22240, 0.22006, 0.21776, 0.21552, 0.21331,
                          0.21115, 0.20904, 0.20696, 0.20492, 0.20293, 0.20097,
                          0.19905, 0.19717, 0.19532, 0.19350, 0.19172, 0.18997,
                          0.18825, 0.18656, 0.18490, 0.18327, 0.18167, 0.18009,
                          0.17855, 0.17702, 0.17553, 0.17406, 0.17261, 0.17119,
                          0.16979, 0.16841, 0.16705, 0.16572, 0.16441, 0.16312,
                          0.16184, 0.16059, 0.15936, 0.15814, 0.15694, 0.15576,
                          0.15460, 0.15346, 0.15233, 0.15122, 0.15012, 0.14904,
                          0.14798, 0.14693, 0.14589, 0.14487, 0.14387, 0.14287,
                          0.14189, 0.14093, 0.13998, 0.13903, 0.13811, 0.13719,
                          0.13629, 0.13540, 0.13452, 0.13365, 0.13279, 0.13194,
                          0.13111, 0.13028, 0.12946, 0.12866, 0.12786, 0.12708,
                          0.12630, 0.12553, 0.12477, 0.12403, 0.12329, 0.12255,
                          0.12183, 0.12112, 0.12041, 0.11971, 0.11902, 0.11834,
                          0.11767, 0.11700, 0.11634, 0.11569, 0.11505, 0.11441,
                          0.11378, 0.11315, 0.11254, 0.11193, 0.11132, 0.11073,
                          0.11013, 0.10955, 0.10897, 0.10840, 0.10783, 0.10727,
                          0.10672, 0.10617, 0.10562, 0.10508, 0.10455, 0.10402,
                          0.10350, 0.10299, 0.10247, 0.10197, 0.10146, 0.10097,
                          0.10048, 0.09999, 0.09951, 0.09903, 0.09855, 0.09808,
                          0.09762, 0.09716, 0.09670, 0.09625, 0.09580, 0.09536,
                          0.09492, 0.09448]
            sigma8_0 = sigma8_arr[0]
            sigma8_z = np.interp(self.z, z_arr, sigma8_arr)
            
            if self.tracer_name == 'LRG':
                bias = 1.7 * sigma8_0 / sigma8_z
            elif self.tracer_name == 'QSO':
                bias = 0.53 + 0.29 * (1+self.z)**2
            elif self.tracer_name == 'ELG':
                bias = 1.0* sigma8_0 / sigma8_z
            elif self.tracer_name == 'CMASS':
                # CMASS bias at z=0.53 https://arxiv.org/pdf/1202.6057.pdf
                bias = 2
        return bias
    
    
class DESI_param:
    '''
    https://arxiv.org/pdf/1611.00036.pdf
    table 2.3 & 2.6
    deltaz~5e-4 p.8
    '''
    def __init__(self, z=0, tracer_name='LRG', get_bias=True):
        
    
        self.tracer_names = ['BGS', 'LRG', 'ELG', 'QSO', 'LAF']        
        self.z = z
        self.tracer_name = tracer_name
        self.print_name = 'DESI ' + tracer_name
        if tracer_name == 'LAF':
            self.print_name = 'DESI QSO'
        self.print_name_survey = 'DESI'
        self._calc_params(get_bias=get_bias)
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    
    def _get_area(self):
        
        Adeg = 14000
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        if self.tracer_name == 'LRG':
            table = self.LRG_density_table()
        elif self.tracer_name == 'QSO':
            table = self.QSO_density_table()
        elif self.tracer_name == 'ELG':
            table = self.ELG_density_table()
        elif self.tracer_name == 'BGS':
            table = self.BGS_density_table()
        elif self.tracer_name == 'LAF':
            table = self.LAF_density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(deg^2)/(dzbin)]
        self.n_zbin_deg2 = table['ns'][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)
        
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = self.n_zbin_deg2 / dV
                
            
    def LRG_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.3
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0.6,2,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.zeros_like(zbins)
        ns[:6] = [832, 986, 662, 272, 51, 17]
        ns *= dz
        
        LRGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LRGtable
    
    def QSO_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.3
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0.6,2,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.zeros_like(zbins)
        ns[:] = [47,55,61,67,72,76,80,83,85,87,87,87,86]
        ns *= dz
        
        QSOtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return QSOtable
     
    def ELG_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.3
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0.6,2,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.zeros_like(zbins)
        ns[:-2] = [309,2269,1923,2094,1441,1353,1337,523,466,329,126]
        ns *= dz
        
        ELGtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return ELGtable
    
    def BGS_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.5
        
        ns/zbin/deg^2 [deg^-2]
        '''
        zbinedges = np.arange(0,0.6,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.array([1165, 3074, 1909,732,120], dtype=float)
        ns *= dz
        
        BGStable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return BGStable

    def LAF_density_table(self):
        '''
        https://arxiv.org/pdf/1611.00036.pdf
        table 2.7
        NOTE: these numbers are for the background QSO of LAF
        ns/zbin/deg^2 [deg^-2]
        '''
        zbins = np.array([1.96,2.12,2.28,2.43,2.59,2.75,2.91,
                          3.07,3.23,3.39,3.55, 3.70, 3.86, 4.02])
        zbinedges = (zbins[1:] + zbins[:-1])/2
        zbinedges = np.concatenate(([zbinedges[0]-0.16], zbinedges, [zbinedges[-1]+0.16]))
        zbinedges[0] = 1.9
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        dz = np.diff(zbinedges)
        ns = np.array([82, 69, 53, 43, 37, 31, 26, 21, 16, 13, 9 ,7, 5, 3], dtype=float)
        ns *= dz
        
        LAFtable = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return LAFtable

    def tracer_bias(self, sigma8_interp=True):
        
        if not sigma8_interp:
            if self.tracer_name == 'LRG':
                bias = 1.7 * HMFz(0).sigma8() / HMFz(self.z).sigma8()
            elif self.tracer_name == 'QSO':
                bias = 0.53 + 0.29 * (1+self.z)**2
            elif self.tracer_name == 'ELG':
                bias = 1.0* HMFz(0).sigma8() / HMFz(self.z).sigma8()
        else:
            # pre-computed sigma8 value for fast interpolation
#             z_arr = np.arange(0,10,0.05)
#             sigma8_arr = []
#             for z in z_arr:
#                 sigma8_arr.append(HMFz(z).sigma8())

            z_arr = np.arange(0,10,0.05)
            sigma8_arr = [0.81583, 0.79475, 0.77410, 0.75394, 0.73430, 0.71522,
                          0.69671, 0.67879, 0.66146, 0.64472, 0.62857, 0.61299,
                          0.59799, 0.58353, 0.56962, 0.55622, 0.54333, 0.53092,
                          0.51898, 0.50749, 0.49642, 0.48577, 0.47551, 0.46562,
                          0.45610, 0.44692, 0.43806, 0.42952, 0.42128, 0.41333,
                          0.40564, 0.39822, 0.39105, 0.38411, 0.37740, 0.37091,
                          0.36462, 0.35854, 0.35264, 0.34692, 0.34138, 0.33601,
                          0.33079, 0.32573, 0.32082, 0.31604, 0.31140, 0.30689,
                          0.30251, 0.29825, 0.29410, 0.29006, 0.28612, 0.28229,
                          0.27856, 0.27492, 0.27138, 0.26792, 0.26455, 0.26126,
                          0.25805, 0.25491, 0.25185, 0.24886, 0.24594, 0.24309,
                          0.24030, 0.23757, 0.23490, 0.23229, 0.22974, 0.22724,
                          0.22480, 0.22240, 0.22006, 0.21776, 0.21552, 0.21331,
                          0.21115, 0.20904, 0.20696, 0.20492, 0.20293, 0.20097,
                          0.19905, 0.19717, 0.19532, 0.19350, 0.19172, 0.18997,
                          0.18825, 0.18656, 0.18490, 0.18327, 0.18167, 0.18009,
                          0.17855, 0.17702, 0.17553, 0.17406, 0.17261, 0.17119,
                          0.16979, 0.16841, 0.16705, 0.16572, 0.16441, 0.16312,
                          0.16184, 0.16059, 0.15936, 0.15814, 0.15694, 0.15576,
                          0.15460, 0.15346, 0.15233, 0.15122, 0.15012, 0.14904,
                          0.14798, 0.14693, 0.14589, 0.14487, 0.14387, 0.14287,
                          0.14189, 0.14093, 0.13998, 0.13903, 0.13811, 0.13719,
                          0.13629, 0.13540, 0.13452, 0.13365, 0.13279, 0.13194,
                          0.13111, 0.13028, 0.12946, 0.12866, 0.12786, 0.12708,
                          0.12630, 0.12553, 0.12477, 0.12403, 0.12329, 0.12255,
                          0.12183, 0.12112, 0.12041, 0.11971, 0.11902, 0.11834,
                          0.11767, 0.11700, 0.11634, 0.11569, 0.11505, 0.11441,
                          0.11378, 0.11315, 0.11254, 0.11193, 0.11132, 0.11073,
                          0.11013, 0.10955, 0.10897, 0.10840, 0.10783, 0.10727,
                          0.10672, 0.10617, 0.10562, 0.10508, 0.10455, 0.10402,
                          0.10350, 0.10299, 0.10247, 0.10197, 0.10146, 0.10097,
                          0.10048, 0.09999, 0.09951, 0.09903, 0.09855, 0.09808,
                          0.09762, 0.09716, 0.09670, 0.09625, 0.09580, 0.09536,
                          0.09492, 0.09448]
            sigma8_0 = sigma8_arr[0]
            sigma8_z = np.interp(self.z, z_arr, sigma8_arr)
            
            if self.tracer_name == 'LRG':
                bias = 1.7 * sigma8_0 / sigma8_z
            elif self.tracer_name == 'QSO':
                bias = 0.53 + 0.29 * (1+self.z)**2
            elif self.tracer_name == 'ELG':
                bias = 1.0* sigma8_0 / sigma8_z
            elif self.tracer_name == 'BGS':
                # CMASS bias at z=0.53 https://arxiv.org/pdf/1202.6057.pdf
                bias = 2
            elif self.tracer_name == 'LAF':
                bias = 0.53 + 0.29 * (1+self.z)**2

        return bias

class Euclid_param:
    '''
    https://link.springer.com/content/pdf/10.1007/s41114-017-0010-3.pdf
    table 3, n_2 case
    '''
    def __init__(self, z=0, tracer_name='ELG', get_bias=True):
        
        self.z = z
        self.tracer_names = ['ELG'] 
        self.tracer_name = tracer_name
        self.print_name = 'Euclid ' + tracer_name
        self.print_name_survey = 'Euclid'
        self._calc_params(get_bias=get_bias)
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        
        Adeg = 15000
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        table = self.density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = table['ns'][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]

        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        self.n_zbin_deg2 = self.n_Mpc3 * dV
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)                
            
    def density_table(self):
        '''
        https://link.springer.com/content/pdf/10.1007/s41114-017-0010-3.pdf
        table 3, n_2 case
        
        ns: [dN/d(Mpc/h)^3]
        '''
        zbinedges = np.arange(0.65,2.06,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.array([1.25, 1.92, 1.83, 1.68, 1.51, 1.35, 1.20, 1.00, 0.80,
                      0.58, 0.38, 0.35, 0.21, 0.11]) * 1e-3
        
        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return table
  
    def tracer_bias(self):
        '''
        Euclid detect galaxy through Ha, so use Ha bias fit.
        Ha bias, Merson+19 https://arxiv.org/abs/1903.02030
        '''
        return 0.7 * self.z + 0.7

class Euclid_deep_param:
    def __init__(self, z=0, tracer_name='ELG', get_bias=True):
        
        self.z = z
        self.tracer_names = ['ELG'] 
        self.tracer_name = tracer_name
        self.print_name = 'Euclid deep ' + tracer_name
        self.print_name_survey = 'Euclid deep'
        self._calc_params(get_bias=get_bias)
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        
        Adeg = 10
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        table = self.density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = table['ns'][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]

        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        self.n_zbin_deg2 = self.n_Mpc3 * dV
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)                
            
    def density_table(self):
        '''        
        ns: [dN/d(Mpc/h)^3]
        '''
        zbinedges = np.arange(0.65,2.06,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.ones_like(zbins) * 4e-3
        
        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return table
  
    def tracer_bias(self):
        return 0.7 * self.z + 0.7

class WFIRST_param:
    '''
    https://arxiv.org/pdf/1907.09680.pdf
    Use "Dust model fit at high redshifts"
    z < 2: Ha table 1, 2 < z < 3: OIII table 
    '''
    def __init__(self, z=0, tracer_name='ELG', get_bias=True):
        
        self.z = z
        self.tracer_names = ['ELG'] 
        self.tracer_name = tracer_name
        self.print_name = 'Roman ' + tracer_name
        self.print_name_survey = 'Roman'
        self._calc_params(get_bias=get_bias)
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        '''
        https://arxiv.org/pdf/1710.00833.pdf
        p.2
        '''
        Adeg = 2200
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        table = self.density_table()
            
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(deg^2)/(dzbin)]
        self.n_zbin_deg2 = table['ns'][zidx[0]]
        self.zbin_min = zbins_min[zidx[0]]
        self.zbin_max = zbins_max[zidx[0]]

        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)
 
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = self.n_zbin_deg2 / dV
            
    def density_table(self):
        '''
        https://arxiv.org/pdf/1907.09680.pdf
        table 2, Dust model fit at high redshifts, flux limit 1e-16
        
        ns: [dN/d(deg)^2]
        '''
        zbinedges = np.arange(0.5,3.01,0.1)
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = np.array([19912,23110,18567,17942,15570,16235,14312,
                       10377,11615,9270,7190,6075,4705,2792,2232,
                      875,597,715,820,520,477,312,295,337,222]) # [dN/dz/d(deg)^2]  
        
        ns = ns * dz # [dN/d(deg)^2]

        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns}
        
        return table
  
    def tracer_bias(self):
        '''
        Ha bias, Merson+19 https://arxiv.org/abs/1903.02030
        '''
        return 0.7 * self.z + 0.7

class WFIRSTphot_param:
    '''
    Y(1.060um), J(1.293), H(1.577) < 26.5
    https://arxiv.org/pdf/2004.05271.pdf
    (use the band that gives min # density)
    depth also see Fig 1, https://arxiv.org/pdf/1808.10458.pdf

    sigma_z/(1+z) ~ 0.04 (avg), ~0.2 on boundaries
    Fig 7
    https://arxiv.org/pdf/1808.10458.pdf

    filters:
    https://wfirst.ipac.caltech.edu/sims/Param_db.html#wfi_filters
    '''
    def __init__(self, z=0, tracer_name='Phot', get_bias=True):
        
        self.z = z
        self.tracer_names = ['Phot'] 
        self.tracer_name = tracer_name
        self.print_name = 'Roman ' +' (phot.)'
        
        if self.tracer_name == 'highz':
            self.print_name = 'Roman (phot.) \n Lyman Break'

        self.print_name_survey = 'Roman'
        self._calc_params(get_bias=get_bias)
        self._get_highz_bins_info()
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        '''
        https://arxiv.org/pdf/1710.00833.pdf
        p.2
        '''
        Adeg = 2200
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        if self.tracer_name == 'Phot':
            table = self.density_table()
        elif self.tracer_name == 'highz':
            table = self.density_table_highz()
            if table['ns']!=0:
                zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
                zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]
                self.highz_z = table['zbins'][zidx[0]]
                self.highz_zmin = table['zbins_min'][zidx[0]]
                self.highz_zmax = table['zbins_max'][zidx[0]]
        
        ns = table['ns']
        
        if ns==0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = ns
        self.zbin_min = self.z - 0.02*(1+self.z)/2
        self.zbin_max = self.z + 0.02*(1+self.z)/2
        if self.tracer_name == 'highz':
            self.zbin_min = self.highz_zmin
            self.zbin_max = self.highz_zmax
        if self.zbin_min < 0:
            self.zbin_min = 0

        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        self.n_zbin_deg2 = self.n_Mpc3 * dV
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)                
            
    def density_table(self):
        '''
        Y(1.060um), J(1.293), H(1.577) < 26.5, sigma_z/(1+z) < 0.03
        https://arxiv.org/pdf/2004.05271.pdf

        https://wfirst.ipac.caltech.edu/sims/Param_db.html#wfi_filters
        ns: [dN/d(Mpc/h)^3]
        '''
        z = self.z
        if self.z < 0.05:
            z = 0.05
        if self.z<=6.001:
            _, ns_y = Helgason_model().get_mask_th(z, 1.060, 26.5, 1.060)
            _, ns_j = Helgason_model().get_mask_th(z, 1.293, 26.5, 1.293)
            _, ns_h = Helgason_model().get_mask_th(z, 1.577, 26.5, 1.577)
            ns = np.min((ns_y,ns_j,ns_h))
        else:
            ns = 0.

        table = {'ns': ns}
        
        return table

    def density_table_highz(self):
        '''
        High-z Lyman break
        redshift defined by Lyman break across bands
        use lambda_min, lambda_max as the boundaries
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=WFIRST&asttype=
        band_arr = ['u','g','r','i','z','y']
        use the z bins from Finkelstein table
        six redshift slices centered at: [6, 7, 8, 9, 10]
        with deltaz = (Delta z)/ (1+z) = [0.143, 0.125, 0.111, 0.100, 0.091]
        
        consider the same depth of all samples:
        r < 27.5, 0 < z < 4

        ns: [dN/d(Mpc/h)^3]
        '''
        z = self.z
        
        zbinedges = np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        zbins = (zbinedges[1:]+zbinedges[:-1])/2
        delz = np.diff(zbinedges) / (zbins+1)

        if self.z < zbinedges[0]:
            ns = 0.
            table = {'ns': ns, 'zbins':zbins, 'zbins_min': zbinedges[:-1], 'zbins_max': zbinedges[1:]}
            return table

        elif self.z > zbinedges[-1]:
            ns = 0.
            table = {'ns': ns, 'zbins':zbins, 'zbins_min': zbinedges[:-1], 'zbins_max': zbinedges[1:]}
            return table
        else:
            
            # dN/dz/FoV
            if z<6.5:
                ns = 3300000
                zbin_min, zbin_max = 5.5, 6.5
            elif z<7.5:
                ns = 530000
                zbin_min, zbin_max = 6.5, 7.5
            elif z<8.5:
                ns = 280000
                zbin_min, zbin_max = 7.5, 8.5
            elif z<9.5:
                ns = 75000
                zbin_min, zbin_max = 8.5, 9.5
            elif z<10.5:
                ns = 19000
                zbin_min, zbin_max = 9.5, 10.5
                
        chi1 = cosmo.comoving_distance(zbin_max)
        chi2 = cosmo.comoving_distance(zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        ns = ns / self.survey_area_deg / dV
        
        table = {'ns': ns, 'zbins':zbins, 'zbins_min': zbinedges[:-1], 'zbins_max': zbinedges[1:]}
        return table


    def _get_highz_bins_info(self):
        zbinedges = np.array([5.5, 6.5, 7.5, 8.5, 9.5, 10.5])
        zbins = (zbinedges[1:]+zbinedges[:-1])/2
        delz = np.diff(zbinedges) / (zbins+1)
        self.Lybreak_info_zbins = zbins
        self.Lybreak_info_zbinedges = zbinedges
        self.Lybreak_info_zmin = zbinedges[:-1]
        self.Lybreak_info_zmax = zbinedges[1:]
        self.Lybreak_info_delz = delz
    
    def tracer_bias(self):
        '''
        Use Helgason's bias
        '''
        
        savename='./Helgason_bias_HOD.csv'
        df = pd.read_csv(savename)
        z_arr = np.array(df['z'])
        b_arr = np.array(df['bias'])
        b = np.interp(self.z, z_arr, b_arr)
        return b

class LSSTphot_param:
    '''
    LSST wl_eff:
    http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=LSST

    Depth:
    https://www.lsst.org/scientists/keynumbers
    u-26.1, g-27.4, r-27.5, i-26.8, z-26.1, y=24.9
    
    all:
    https://arxiv.org/pdf/0912.0201.pdf (p.75)
    r < 27.5, 0 < z < 4

    Gold sample
    https://arxiv.org/pdf/0912.0201.pdf (p.75)
    i < 25.3, 0 < z < 3,  sigma_z/(1+z) < 0.05  (goal 0.02)
    
    High-z Lyman break
    redshift defined by Lyman break across bands
    use lambda_min, lambda_max as the boundaries
    http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=LSST
    band_arr = ['u','g','r','i','z','y']
    wl_min_arr = np.array([3205.54, 3873.01, 5375.74, 6765.00, 8034.98, 9088.94])
    wl_max_arr = np.array([4081.24, 5665.07, 7054.95, 8324.70, 9374.71, 10897.23])
    
    this gives the redshift boundaries: [2.36, 3.66, 4.80, 5.85, 6.71]
    six redshift slices centered at: [3.01, 4.23, 5.32, 6.28]
    with deltaz = (Delta z)/ (1+z) = [0.33,0.22,0.17,0.12]

    # this gives the redshift boundaries: [2.51, 3.4, 5.1, 6.6, 8.0, 9.1, 10.9]
    # six redshift slices centered at: [2.9, 4.2, 5.8, 7.3, 8.5, 10.0]
    # with deltaz = (Delta z)/ (1+z) = [0.21,0.32,0.22,0.17,0.12,0.17]
    '''
    def __init__(self, z=0, tracer_name='Phot', get_bias=True):
        
        self.z = z
        self.tracer_names = ['Phot','Gold'] 
        self.tracer_name = tracer_name
        if self.tracer_name == 'Phot':
            self.print_name = 'Rubin (phot.)'
        if self.tracer_name == 'Gold':
            self.print_name = 'Rubin (phot.) \n Gold Sample'
        elif self.tracer_name == 'Lybreak':
            self.print_name = 'Rubin (phot.) \n Lyman Break'

        self.print_name_survey = 'Rubin'
        self._calc_params(get_bias=get_bias)
        self._get_Lybreak_bins_info()
    
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        '''
        https://en.wikipedia.org/wiki/Vera_C._Rubin_Observatory
        18,000 deg^2
        '''
        Adeg = 18000
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     
        
    def _get_density(self):
        
        if self.tracer_name == 'Gold':
            table = self.density_table_gold()
        elif self.tracer_name == 'Phot':
            table = self.density_table()
        elif self.tracer_name == 'Lybreak':
            table = self.density_table_Lybreak()
            if table['ns']!=0:
                zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
                zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]
                self.Lybreak_z = table['zbins'][zidx[0]]
                self.Lybreak_zmin = table['zbins_min'][zidx[0]]
                self.Lybreak_zmax = table['zbins_max'][zidx[0]]

        ns = table['ns']
        
        if ns==0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        
        # [dN/d(Mpc/h)^3]
        self.n_Mpc3 = ns
        self.zbin_min = self.z - 0.02*(1+self.z)/2
        self.zbin_max = self.z + 0.02*(1+self.z)/2
        if self.tracer_name == 'highz':
            self.zbin_min = self.highz_zmin
            self.zbin_max = self.highz_zmax
        if self.zbin_min < 0:
            self.zbin_min = 0

        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        self.n_zbin_deg2 = self.n_Mpc3 * dV
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)

        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min)                
            
    def density_table(self):
        '''
        all:
        https://arxiv.org/pdf/0912.0201.pdf (p.75)
        r < 27.5, 0 < z < 4
        ns: [#/(Mpc/h)^3]
        '''
        z = self.z
        if self.z < 0.05:
            z = 0.05
        if self.z<=4.01:
            wl_obs_th_ref = 0.617323
            m_th_ref = 27.5
            _, ns = Helgason_model().get_mask_th(z, wl_obs_th_ref,
                                                       m_th_ref, wl_obs_th_ref)
        else:
            ns = 0.

        table = {'ns': ns}
        
        return table
    
    def density_table_gold(self):
        '''
        Gold sample
        https://arxiv.org/pdf/0912.0201.pdf (p.75)
        i < 25.3, 0 < z < 3,  sigma_z/(1+z) < 0.05  (goal 0.02)
        ns: [#/(Mpc/h)^3]
        '''
        z = self.z
        if self.z < 0.05:
            z = 0.05
        if self.z<=3.01:
            wl_obs_th_ref = 0.750162
            m_th_ref = 25.3
            _, ns = Helgason_model().get_mask_th(z, wl_obs_th_ref,
                                                       m_th_ref, wl_obs_th_ref)
        else:
            ns = 0.

        table = {'ns': ns}
        
        return table

    def density_table_Lybreak(self):
        '''
        High-z Lyman break
        redshift defined by Lyman break across bands
        use lambda_min, lambda_max as the boundaries
        http://svo2.cab.inta-csic.es/theory/fps/index.php?mode=browse&gname=LSST
        band_arr = ['u','g','r','i','z','y']
        wl_min_arr = np.array([3205.54, 3873.01, 5375.74, 6765.00, 8034.98, 9088.94])
        wl_max_arr = np.array([4081.24, 5665.07, 7054.95, 8324.70, 9374.71, 10897.23])
        this gives the redshift boundaries: [2.51, 3.4, 5.1, 6.6, 8.0, 9.1, 10.9]
        six redshift slices centered at: [3.01, 4.23, 5.32, 6.28]
        with deltaz = (Delta z)/ (1+z) = [0.33, 0.22, 0.17, 0.12]
        
        consider the same depth of all samples:
        r < 27.5, 0 < z < 4

        ns: [#/(Mpc/h)^3]
        '''
        z = self.z
        
        wl_min_arr = np.array([3205.54, 3873.01, 5375.74, 6765.00, 8034.98, 9088.94])
        wl_max_arr = np.array([4081.24, 5665.07, 7054.95, 8324.70, 9374.71, 10897.23])
        
        # wl_bound = (wl_min_arr[1:] + wl_max_arr[:-1])/2
        # wl_bound = np.concatenate(([wl_min_arr[0]],wl_bound,[wl_max_arr[-1]]))
        # zbinedges = wl_bound/912-1
        wl_bound = wl_max_arr[:-1]
        zbinedges = wl_bound/1216-1

        zbins = (zbinedges[1:]+zbinedges[:-1])/2
        delz = np.diff(zbinedges) / (zbins+1)

        if self.z < zbinedges[0]:
            ns = 0.
        elif self.z > zbinedges[-1]:
            ns = 0.
        else:
            wl_obs_th_ref = 0.617323
            m_th_ref = 27.5
            _, ns = Helgason_model().get_mask_th(z, wl_obs_th_ref,
                                                       m_th_ref, wl_obs_th_ref)

        table = {'ns': ns, 'zbins':zbins, 'zbins_min': zbinedges[:-1], 'zbins_max': zbinedges[1:]}
        
        return table


    def _get_Lybreak_bins_info(self):
        wl_min_arr = np.array([3205.54, 3873.01, 5375.74, 6765.00, 8034.98, 9088.94])
        wl_max_arr = np.array([4081.24, 5665.07, 7054.95, 8324.70, 9374.71, 10897.23])
        
        # wl_bound = (wl_min_arr[1:] + wl_max_arr[:-1])/2
        # wl_bound = np.concatenate(([wl_min_arr[0]],wl_bound,[wl_max_arr[-1]]))
        # zbinedges = wl_bound/912-1
        wl_bound = wl_max_arr[:-1]
        zbinedges = wl_bound/1216-1

        zbins = (zbinedges[1:]+zbinedges[:-1])/2
        delz = np.diff(zbinedges) / (zbins+1)
        self.Lybreak_info_zbins = zbins
        self.Lybreak_info_zbinedges = zbinedges
        self.Lybreak_info_zmin = zbinedges[:-1]
        self.Lybreak_info_zmax = zbinedges[1:]
        self.Lybreak_info_delz = delz
        
    def tracer_bias(self):
        '''
        Use Helgason's bias
        '''
        
        savename='./Helgason_bias_HOD.csv'
        df = pd.read_csv(savename)
        z_arr = np.array(df['z'])
        b_arr = np.array(df['bias'])
        b = np.interp(self.z, z_arr, b_arr)
        return b


class spherex_param:
    '''
    SPHEREx bands:
    0.75 - 1.11 um R = 41
    1.11 - 1.64 um R = 41
    1.64 - 2.42 um R = 41
    2.42 - 3.82 um R = 35
    3.82 - 4.42 um R = 110
    4.42 - 5.00 um R = 130
    '''
    def __init__(self, line_name=None):

        self.dth = 6.2 / 60. # arcmin
        self.beam = ((self.dth * u.arcmin).to(u.rad).value)**2 # sr
        self.lmax = np.pi / (self.dth * np.pi / 180 / 60)
        self.fsky_all = 0.75
        self.fsky_deep = 100 * (np.pi/180)**2 / 4 / np.pi # deep field 100 sq deg
        self.band_wlmin = [0.75, 1.11, 1.64, 2.42, 3.82, 4.42]
        self.band_wlmax = [1.11, 1.64, 2.42, 3.82, 4.42, 5.00]
        self.band_R = [41, 41, 41, 35, 110, 130]
        self.channel_per_array = 16
        self.line_names = ['Ha','OIII','Hb','OII','Lya']
        self._get_bins()
        self._get_Neff()
        self._get_NEI()
        if line_name is not None:
            self.cmv_config(line_name)
        
    def _get_bins(self):

        wl_binedges = np.array([])
        for band in range(len(self.band_R)):
            wl_binedges_band = self.band_wlmin[band] * (self.band_wlmax[band] / self.band_wlmin[band])\
            **(np.arange(self.channel_per_array + 1) / self.channel_per_array)
            wl_binedges = np.concatenate((wl_binedges, wl_binedges_band[:-1]))
        wl_binedges = np.concatenate((wl_binedges, [wl_binedges_band[-1]]))
        broad_band_idx = np.array(([0] * self.channel_per_array + [1] * self.channel_per_array + \
                         [2] * self.channel_per_array + [3] * self.channel_per_array + \
                         [4] * self.channel_per_array + [5] * self.channel_per_array))

        um2GHz = 1 * u.um.to(u.GHz, equivalencies=u.spectral())
        nu_binedges = um2GHz / wl_binedges

        self.nu_min = nu_binedges[-1]
        self.nu_max = nu_binedges[0]
        self.nu_binedges = nu_binedges
        self.nu_bins = (self.nu_binedges[1:] + self.nu_binedges[:-1]) / 2
        self.Nnu = len(self.nu_bins)
        self.dnus = np.abs(np.diff(self.nu_binedges))
        self.broad_band_idx = broad_band_idx
        self.wl_binedges = um2GHz / self.nu_binedges
        self.wl_bins = um2GHz / self.nu_bins
        self.dwls = np.abs(np.diff(wl_binedges))

        return

    def _get_Neff(self):

        D = 0.2
        rmsspot = 1.7
        point1sig = 0.8
        diff = (self.wl_bins * 1e-6 / D) * (180. / np.pi) * 3600
        fwhm = np.sqrt(diff**2 + (point1sig * 2.35)**2 + (rmsspot * 2.35)**2)
        Neff = 0.8 + 0.27 * fwhm + 0.0425 * fwhm**2
        self.Neff = Neff

        return

    def _get_NEI(self):
        '''
        NEI in unit Jy/sr
        data from 
        https://github.com/SPHEREx/Public-products/blob/master/Surface_Brightness_v28_base_cbe.txt
        '''
        
        neidata = 'data_external/spherex_public_product/Surface_Brightness_v28_base_cbe.txt'
        data = np.loadtxt(neidata, skiprows=1)
        wls, NEI_all, NEI_deep = data[:,0], data[:,1], data[:,2]
        nus = const.c / (wls*u.um)
        NEI_all = ((NEI_all * u.nW/u.m**2/u.sr)/ nus).to(u.Jy/u.sr)
        NEI_deep = ((NEI_deep * u.nW/u.m**2/u.sr)/ nus).to(u.Jy/u.sr)
        self.NEI_all, self.NEI_deep = NEI_all.value, NEI_deep.value
        
        return

    def _get_NEI_fixed_mag(self, mag_all=19.5, mag_deep=22):
        '''
        NEI in unit Jy/sr
        5 sigma sensitivity: m_AB = 19.5(all sky), 22(deep)
        NEI = Fn*sqrt(Neff) / Omega_pix

        https://github.com/SPHEREx/Public-products/blob/master/Surface_Brightness_v28_base_cbe.txt
        '''
        Om_pix = (self.dth * u.arcmin.to(u.rad))**2

        F_all = (3631. * 10**(-mag_all / 2.5)) / 5.
        self.NEI_all = F_all / Om_pix / np.sqrt(self.Neff)
        F_deep = (3631. * 10**(-mag_deep / 2.5)) / 5.
        self.NEI_deep = F_deep / Om_pix / np.sqrt(self.Neff)

        return

    def cmv_config(self, line_name):
        _get_cmv_config(self, line_name)
        return
    

def _get_cmv_config(survey, line_name, jco = 1):
    
    if line_name =='CII':
        mu_rest = spec_lines().CII
        name = 'CII'
    elif line_name =='CO':
        if jco not in np.arange(1,9,1,dtype=int):
            print('jco data not exist! (jco best be in [1,2,...,8])')
            return
        else:
            mu_rest = spec_lines().CO(jco)
            name = 'CO(' + str(jco) + '-' + str(jco-1) + ')'
    elif line_name == 'Lya':
        mu_rest = spec_lines().Lya
        name = 'Lya'
    elif line_name == 'Ha':
        mu_rest = spec_lines().Ha
        name = 'Ha'
    elif line_name == 'Hb':
        mu_rest = spec_lines().Hb
        name = 'Hb'
    elif line_name == 'OII':
        mu_rest = spec_lines().OII
        name = 'OII'
    elif line_name == 'OIII':
        mu_rest = spec_lines().OIII
        name = 'OIII'
    else:
        print('line name has to be "CII", "CO", "Lya", "Ha", "Hb", "OII", "OIII".')
        return
    
    survey.mu_rest = mu_rest.value # um
    survey.nu_rest = mu_rest.to(u.GHz, equivalencies=u.spectral()).value # GHz
    survey.z_bins = (survey.nu_rest/survey.nu_bins) - 1
    survey.z_binedges = (survey.nu_rest/survey.nu_binedges) - 1
    survey.dzs = np.abs(np.diff(survey.z_binedges))

    xDcmv_vec = np.zeros_like(survey.z_bins)
    for i in range(len(xDcmv_vec)):
        xDcmv_vec[i] = cosmo_dist(survey.z_bins[i]).comoving_distance.value
    xDcmv_vec *= (survey.dth * u.arcmin).to(u.rad).value
    survey.xDcmv_vec = xDcmv_vec

    zDcmv_edges_vec = np.zeros_like(survey.z_binedges)
    for i in range(len(zDcmv_edges_vec)):
        zDcmv_edges_vec[i] = cosmo_dist(survey.z_binedges[i]).comoving_distance.value
    survey.zDcmv_vec = zDcmv_edges_vec[1:] - zDcmv_edges_vec[:-1]

    survey.k_p_min = 2 * np.pi / np.sum(survey.xDcmv_vec)
    survey.k_p_max = np.pi / np.mean(survey.xDcmv_vec)
    survey.k_l_min = 2 * np.pi / np.sum(survey.zDcmv_vec)
    survey.k_l_max = np.pi / np.mean(survey.zDcmv_vec)

    return

class spherex_gal_param:
    def __init__(self, z=0, tracer_name='galaxy', get_bias=True, photz_err_class=2):

        self.tracer_names = ['galaxy']        
        self.z = z
        self.tracer_name = tracer_name
        self.photz_err_class = photz_err_class
        self.print_name = 'SPHEREx' + ' (spec.)'
        self.print_name_survey = 'SPHEREx'
        self._calc_params(get_bias=get_bias)
        
        photz_err_binedges = [0,0.003,0.01,0.03,0.1,0.2]
        self.photz_err_min = photz_err_binedges[photz_err_class]
        self.photz_err_max = photz_err_binedges[photz_err_class+1]
        
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()
    
    def _get_area(self):
        
        Asr = 0.75*4*np.pi
        self.survey_area_sr = Asr
        
        Adeg = Asr * (u.sr).to(u.deg**2)
        self.survey_area_deg = Adeg
            
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     

    def _get_density(self):
        
        table = self.density_table()
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        # [dN/d(Mpc/h)^3]
        # if len(zidx) == 0:
        #     self.n_Mpc3 = self._get_n_high_z(z_scale=4.6)
        #     self.zbin_min = self.z * 0.95
        #     self.zbin_max = self.z * 1.05
        if len(zidx) == 0:
            self.n_zbin_deg2 = 0
            self.n_z_deg2 = 0
            self.n_z_sr = 0
            self.n_Mpc3 = 0
            return
        else:
            self.n_Mpc3 = table['ns'][zidx[0]]
            self.zbin_min = zbins_min[zidx[0]]
            self.zbin_max = zbins_max[zidx[0]]
        
        chi1 = cosmo.comoving_distance(self.zbin_max)
        chi2 = cosmo.comoving_distance(self.zbin_min)
        
        dchi = (chi1 - chi2)
        dA = (((chi1 + chi2) / 2) / (u.rad).to(u.deg))**2
        dV = (dchi * dA).value * cosmo.h**3 # [(Mpc/h)**3]
        
        self.n_zbin_deg2 = self.n_Mpc3 * dV
        
        # [dN/dz/d(deg^2)]
        self.n_z_deg2 = self.n_zbin_deg2 / (self.zbin_max - self.zbin_min)
        
        # [dN/dz/dsr]
        self.n_z_sr = self.n_zbin_deg2 * (u.deg**-2).to(u.sr**-1) \
                        / (self.zbin_max - self.zbin_min) 

    def tracer_bias(self):
        
        table = self.bias_table()
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((self.z >= zbins_min) & (self.z < zbins_max))[0]

        if len(zidx) == 0:
            return table['bias'][-1]
        
        return table['bias'][zidx[0]]

    def density_table(self):
        '''
        https://github.com/SPHEREx/Public-products/blob/master/galaxy_density_v28_base_cbe.txt
        '''
        
        zbinedges = np.array([0.,0.2,0.4,0.6,0.8,1.,1.6,2.2,2.8,3.4,4.0,4.6001])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)
        
        ns = [] # [h/Mpc]^3
        ns.append(np.array([0.00997, 0.00411, 0.000501, 7.05e-05, 3.16e-05, 1.64e-05,
                        3.59e-06, 8.07e-07, 1.84e-06, 1.5e-06, 1.13e-06]))
        ns.append(np.array([0.0123, 0.00856, 0.00282, 0.000937, 0.00043, 5e-05, 8.03e-06,
                        3.83e-06, 3.28e-06, 1.07e-06, 6.79e-07]))
        ns.append(np.array([0.0134, 0.00857, 0.00362, 0.00294, 0.00204, 0.000212, 6.97e-06,
                        2.02e-06, 1.43e-06, 1.93e-06, 6.79e-07]))
        ns.append(np.array([0.0229, 0.0129, 0.00535, 0.00495, 0.00415, 0.000796, 7.75e-05,
                        7.87e-06, 2.46e-06, 1.93e-06, 1.36e-06]))
        ns.append(np.array([0.0149, 0.00752, 0.00327, 0.0025, 0.00183, 0.000734, 0.000253, 5.41e-05,
                        2.99e-05, 9.41e-06, 2.04e-06]))

        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'ns': ns[self.photz_err_class]}
        
        return table
    
    def bias_table(self):
        bias = []
        bias.append(np.array([1.3, 1.5, 1.8, 2.3, 2.1, 2.7, 3.6, 2.3, 3.2, 2.7, 3.8]))
        bias.append(np.array([1.2, 1.4, 1.6, 1.9, 2.3, 2.6, 3.4, 4.2, 4.3, 3.7, 4.6]))
        bias.append(np.array([1.0, 1.3, 1.5, 1.7, 1.9, 2.6, 3.0, 3.2, 3.5, 4.1, 5.0]))
        bias.append(np.array([0.98, 1.3, 1.4, 1.5, 1.7, 2.2, 3.6, 3.7, 2.7, 2.9, 5.0]))
        bias.append(np.array([0.83, 1.2, 1.3, 1.4, 1.6, 2.1, 3.2, 4.2, 4.1, 4.5, 5.0]))
        
        zbinedges = np.array([0.,0.2,0.4,0.6,0.8,1.,1.6,2.2,2.8,3.4,4.0, 4.6001])
        zbins_min = zbinedges[:-1]
        zbins_max = zbinedges[1:]
        zbins = (zbins_min + zbins_max) / 2        
        dz = np.diff(zbinedges)

        table = {'zbins': zbins, 'zbins_min': zbins_min, 'zbins_max': zbins_max,
                   'bias': bias[self.photz_err_class]}
        
        return table

    def _get_n_high_z(self, z_scale=None):
        savename='./SPHEREx_deep_all_ratio.csv'
        try:
            df = pd.read_csv(savename)
        except:
            spherex_gal_deep_param()._write_deep_all_ratio()
            df = pd.read_csv(savename)
        
        n = np.exp(np.interp(self.z, df['z'], np.log(df['n_Mpc'])))
        if z_scale is None:
            return n
        
        n_scale = np.exp(np.interp(z_scale, df['z'], np.log(df['n_Mpc'])))
        table = self.density_table()
        zbins_min, zbins_max = table['zbins_min'], table['zbins_max']
        zidx = np.where((z_scale >= zbins_min) & (z_scale < zbins_max))[0]
        n_scale_table = 0 if len(zidx) == 0 else table['ns'][zidx[0]]
       
        return n * n_scale_table / n_scale
            
class spherex_gal_deep_param:
    def __init__(self, z=0, tracer_name='galaxy', get_bias=True, photz_err_class=2):

        self.tracer_names = ['galaxy']        
        self.z = z
        self.tracer_name = tracer_name
        self.photz_err_class = photz_err_class
        self.print_name = 'SPHEREx deep' + ' (phot.)'
        self.print_name_survey = 'SPHEREx deep'
        self._calc_params(get_bias=get_bias)
        
        photz_err_binedges = [0,0.003,0.01,0.03,0.1,0.2]
        self.photz_err_min = photz_err_binedges[photz_err_class]
        self.photz_err_max = photz_err_binedges[photz_err_class+1]
        
    def _calc_params(self, get_bias=True):
        self._get_area()
        self._get_density()
        if get_bias:
            self.bias = self.tracer_bias()

    def _get_area(self):
        
        Adeg = 2*100
        self.survey_area_deg = Adeg
        
        Asr = Adeg * (u.deg**2).to(u.sr)
        self.survey_area_sr = Asr
        
        self.fsky = Asr / 4 / np.pi
        self.lmin = np.pi / np.sqrt(Asr)
        
        return     

    def _get_density(self):
        spherex_all_params = spherex_gal_param(z=self.z, 
                                              tracer_name=self.tracer_name,
                                              get_bias=True,
                                              photz_err_class=self.photz_err_class)
        ratio = self._get_deep_all_ratio(use_zbins=True)[0]
        self.n_zbin_deg2 = spherex_all_params.n_zbin_deg2 * ratio
        self.n_z_deg2 = spherex_all_params.n_z_deg2 * ratio
        self.n_z_sr = spherex_all_params.n_z_sr * ratio
        self.n_Mpc3 = spherex_all_params.n_Mpc3 * ratio

    def tracer_bias(self):
        b = spherex_gal_param(z=self.z,tracer_name=self.tracer_name,
                get_bias=True, photz_err_class=self.photz_err_class).tracer_bias()
        return b

    def _get_deep_all_ratio(self, use_zbins=False):
        savename='./SPHEREx_deep_all_ratio.csv'
        try:
            df = pd.read_csv(savename)
        except:
            self._write_deep_all_ratio()
            df = pd.read_csv(savename)
        
        zuse = self.z
        if use_zbins and self.z <= 4.6:
            zbinedges = np.array([0.,0.2,0.4,0.6,0.8,1.,1.6,2.2,2.8,3.4,4.0, 4.6001])
            zbins_min = zbinedges[:-1]
            zbins_max = zbinedges[1:]
            zbins = (zbins_min + zbins_max) / 2
            zidx = np.where((zuse - zbinedges)>=0)[0][-1]
            zidx = 6 if zidx>6 else zidx
            zuse = zbins[zidx]
        if use_zbins and self.z > 4.6:
            zbinedges = np.array([0.,0.2,0.4,0.6,0.8,1.,1.6,2.2,2.8,3.4,4.0, 4.6001])
            zbins_min = zbinedges[:-1]
            zbins_max = zbinedges[1:]
            zbins = (zbins_min + zbins_max) / 2
            zuse = zbins[6]

        ratio = np.interp(zuse, df['z'], df['ratio'])
        n = np.exp(np.interp(zuse, df['z'], np.log(df['n_Mpc'])))
        n_deep = np.exp(np.interp(zuse, df['z'], np.log(df['n_deep_Mpc'])))
        
        return ratio, n, n_deep

    def _calc_deep_all_ratio(self, z):
        
        low_z = False
        if z < 3e-2:
            z = 3e-2
            low_z = True
        
        z = float(z)
        sigma_detection = 5
        spherex_par = spherex_param()

        fname = 'data_external/spherex_public_product/Point_Source_Sensitivity_v28_base_cbe.txt'
        data = np.loadtxt(fname)
        wl, m, m_deep = data[:,0],data[:,1],data[:,2]
        NEP = 3631 * 10**(-m/2.5) / 5
        NEP_deep = 3631 * 10**(-m_deep/2.5) / 5
        
        wl_obs_th_ref = 1
        
        m_th_ref_arr = np.arange(20,25,0.05)
        SNRtot_arr = np.zeros_like(m_th_ref_arr)
        SNRtot_deep_arr = np.zeros_like(m_th_ref_arr)
        for im, m_th_ref in enumerate(m_th_ref_arr):
            SNRtot = 0
            SNRtot_deep = 0
            for iwl,wl_obs in enumerate(spherex_par.wl_bins):
                m_th = Helgason_model().get_mask_th(z, wl_obs, m_th_ref, wl_obs_th_ref)[0]
                F_th = 3631 * 10**(-m_th/2.5)
                SNRtot += (F_th/NEP[iwl])**2
                SNRtot_deep += (F_th/NEP_deep[iwl])**2
            SNRtot_arr[im] = np.sqrt(SNRtot)
            SNRtot_deep_arr[im] = np.sqrt(SNRtot_deep)

        m_th_ref_z_idx = np.argmin(np.abs(SNRtot_arr - sigma_detection))
        m_th_ref_z = m_th_ref_arr[m_th_ref_z_idx]
        n_th_ref_z = Helgason_model().get_mask_th\
        (z, wl_obs_th_ref, m_th_ref_z, wl_obs_th_ref)[1]

        m_th_ref_z_idx_deep = np.argmin(np.abs(SNRtot_deep_arr - sigma_detection))
        m_th_ref_z_deep = m_th_ref_arr[m_th_ref_z_idx_deep]
        n_th_ref_z_deep = Helgason_model().get_mask_th\
        (z, wl_obs_th_ref, m_th_ref_z_deep, wl_obs_th_ref)[1]
        
        if low_z:
            return 1, n_th_ref_z_deep, n_th_ref_z
        
        return n_th_ref_z_deep / n_th_ref_z, n_th_ref_z_deep, n_th_ref_z
    
    def _write_deep_all_ratio(self):
        savename='./SPHEREx_deep_all_ratio.csv'
        z_arr = np.arange(0,11,0.1)
        df = pd.DataFrame()
        df['z'] = z_arr
        ratio_arr = []
        n_deep_arr = []
        n_arr = []
        for z in z_arr:
            ratio, n_deep, n = self._calc_deep_all_ratio(z)
            ratio_arr.append(ratio)
            n_deep_arr.append(n_deep)
            n_arr.append(n)
        df['ratio'] = ratio_arr
        df['n_Mpc'] = n_arr
        df['n_deep_Mpc'] = n_deep_arr
        df.to_csv(savename, index=False)

# gal_par_dict = {'eBOSS':eBOSS_param, 'DESI':DESI_param,
#                 'Euclid':Euclid_param, 'Euclid_deep':Euclid_deep_param,
#                 'WFIRST':WFIRST_param}
# gal_par_dict_phot = {'LSSTphot':LSSTphot_param, 'WFIRSTphot':WFIRSTphot_param,
#                     'SPHERExphot':spherex_gal_param, 
#                     'SPHERExphot_deep':spherex_gal_deep_param}
gal_par_dict = {'eBOSS':eBOSS_param, 'DESI':DESI_param,
                'Euclid':Euclid_param, 'Euclid_deep':Euclid_deep_param,
                'WFIRST':WFIRST_param, 'SPHEREx':spherex_gal_param}
gal_par_dict_phot = {'LSSTphot':LSSTphot_param, 'WFIRSTphot':WFIRSTphot_param}