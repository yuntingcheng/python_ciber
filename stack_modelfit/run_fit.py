from utils import * 
from scipy import interpolate
from scipy.signal import fftconvolve
import emcee
from multiprocessing import Pool

class fit_stacking_mcmc:
    
    def __init__(self, inst, ifield, im):

        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        self.im = im
        self.m_min = im + 16
        self.m_max = im + 17
        
        self._fit_data_preprocess()

    def _load_data(self):
        import json
        loaddir = mypaths['alldat']+'TM' + str(self.inst) + '/'
        with open(loaddir + self.field + '_datafit.json') as json_file:
            data_all = json.load(json_file)
        
        return data_all
    
    def _fit_data_preprocess(self, clus_rcut=10):
        data_all = self._load_data()
        data =data_all[self.im]
        dx = 1200
        
        # Cov
        profd_arr = np.array(data['profex'])
        Cov = np.array(data['cov'])
        Covi = np.linalg.inv(Cov)
        profd_err_diag = np.sqrt(np.diag(Cov))

        # clustering
        r_arr = np.array(data['r_arr'])
        profclus_arr = np.array(data['profclus'])
        profclus_arr[r_arr<clus_rcut] = 0

        # PSF
        radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]), dx, dx)*0.7
        tck = interpolate.splrep(np.array(data['rfull_arr'])[:18], 
                                       np.array(data['profpsf'])[:18])
        psfwin_map = interpolate.splev(radmap,tck)
        psfwin_map[radmap > data['rfull_arr'][18]] = 0
        psfwin_map[psfwin_map < 0] = 0
        profpsf = radial_prof(psfwin_map, dx, dx)
        profpsf_arr = np.array(profpsf['prof'])
        profpsf_arr /= profpsf_arr[0]
        
        self.r_arr = r_arr
        self.profstackg_arr = np.array(data['profg'])
        self.profd_arr = profd_arr
        self.Cov = Cov
        self.Covi = Covi
        self.profd_err_diag = profd_err_diag
        self.profclus_arr = profclus_arr
        self.profpsf_arr = profpsf_arr
        self.psfwin_map = psfwin_map
        self.r_weight = np.array(data['r_weight'])
        
    def get_profexcess_model(self, **kwargs):
        dx = 1200
        # model
        radmap = make_radius_map(np.zeros([201,201]), 100, 100)*0.7
        modeldat = gal_profile_model().Wang19_profile(radmap, self.im, **kwargs)
        modeldat['I_arr'] = np.pad(modeldat['I_arr'], 1100, 'constant')
        
        # conv model
        modconv_map = fftconvolve(self.psfwin_map, modeldat['I_arr'], 'same')
        profmodconv = radial_prof(modconv_map, dx, dx)
        profmod_arr = profmodconv['prof']/profmodconv['prof'][0]
        
        # excess profile
        profexfull_arr = self.profstackg_arr[0]*(profmod_arr - self.profpsf_arr)
        profex_arr = profile_radial_binning(profexfull_arr, self.r_weight)
        
        return profex_arr
        
    
    def get_chi2(self, **kwargs):
        profex_arr = self.get_profexcess_model(**kwargs)
        if 'Aclus' in kwargs.keys(): 
            Aclus = kwargs['Aclus']
        else:
            Aclus = 1
        D = profex_arr + Aclus*self.profclus_arr - self.profd_arr
        Covi = self.Covi
        D = D[np.newaxis,...]
        chi2 = D@Covi@D.T
        return chi2[0,0]
    
    def _log_likelihood(self, theta, d, Covi):
        xe2, Aclus = theta
        profex_arr = self.get_profexcess_model(xe2 = xe2)
        D = profex_arr + Aclus*self.profclus_arr - d
        D = D[np.newaxis,...]
        chi2 = D@Covi@D.T
        return -chi2/2

    def _log_prior(self, theta):
        xe2, Aclus = theta
        if 0.0001 < xe2 < 1 and 0.0 < Aclus < 100:
            return 0.
        return -np.inf

    def _log_prob(self, theta, d, Covi):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta, d, Covi)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                save_chain=True, savedir = None, savename=None):
        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 100, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob, 
                                            args=(self.profd_arr, self.Covi), pool=pool)
            sampler.run_mcmc(p0, steps, progress=True)
        
        if save_chain:
            if savedir is None:
                savedir = mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_2par_' + self.field + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename
        
        if return_chain:
            return sampler.get_chain()
        else:
            return

def run_mcmc_fit(inst, ifield, im, **kwargs):
    
    param_fit = fit_stacking_mcmc(inst, ifield, im)
    
    print('MCMC fit 2 params for TM%d %s %d < m < %d'\
          %(param_fit.inst, param_fit.field, param_fit.m_min, param_fit.m_max))
    
    param_fit.run_mcmc(**kwargs)
    return param_fit