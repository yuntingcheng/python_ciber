from ciber_info import *
from utils import * 
from scipy import interpolate
from scipy.signal import fftconvolve
import emcee
from multiprocessing import Pool
from ciber_info import *
from clustering import *

class fit_stacking_mcmc:
    
    def __init__(self, inst, ifield, im):

        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        self.im = im
        self.m_min = magbindict['m_min'][im]
        self.m_max = magbindict['m_max'][im]
        self.z = magbindict['zmean'][im]
        
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
        r_arr = np.array(data['r_arr'])
        profd_arr = np.array(data['profex'])
        Cov = np.array(data['cov'])
        Covi = np.linalg.inv(Cov)
        profd_err_diag = np.sqrt(np.diag(Cov))

        # clustering
        theta_arr = np.logspace(-1,3.2,100)
        # w_arr = wgI_zm_approx(self.z, theta_arr)
        cat_data = get_catalog(self.inst, self.ifield, self.im)
        w_arr = wgI(cat_data['z'], theta_arr)

        radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]), dx, dx)*0.7
        tck = interpolate.splrep(np.log(theta_arr), w_arr, k=1)
        radmap[dx,dx] = radmap[dx,dx+1]
        w_map = interpolate.splev(np.log(radmap),tck)
        # self.w_arr = wgI_zm_approx(self.z, data['rfull_arr'])
        self.w_arr = w_arr = wgI(cat_data['z'], data['rfull_arr'])
        
        profclus_arr = np.array(data['profclus'])
        profclus_arr[r_arr<clus_rcut] = 0
        
        # PSF
        radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]), dx, dx)*0.7
        tck = interpolate.splrep(np.log(np.array(data['rfull_arr'])[:16]),
                                 np.log(np.array(data['profpsf'])[:16]), k=1)
        radmap[dx,dx] = radmap[dx,dx+1]
        psfwin_map = np.exp(interpolate.splev(np.log(radmap),tck))
        psfwin_map[psfwin_map < 0] = 0
        profpsf = radial_prof(psfwin_map, dx, dx)
        profpsf_arr = np.array(profpsf['prof'])
        profpsf_arr /= profpsf_arr[0]
        
        self.r_arr = r_arr
        self.rfull_arr = np.array(data['rfull_arr'])
        self.profstackg_arr = np.array(data['profg'])
        self.profd_arr = profd_arr
        self.Cov = Cov
        self.Covi = Covi
        self.profd_err_diag = profd_err_diag
        self.profclus_arr = profclus_arr
        self.profpsf_arr = profpsf_arr
        self.psfwin_map = psfwin_map
        self.w_map = w_map
        self.r_weight = np.array(data['r_weight'])
        self.dof_data = len(profd_arr)
        self.dx = 1200
        
    def get_profgal_model(self, **kwargs):
        
        dx = self.dx
        # model
        radmap = make_radius_map(np.zeros([201,201]), 100, 100)*0.7
        modeldat = gal_profile_model().Wang19_profile(radmap, self.im, **kwargs)
        modeldat['I_arr'] = np.pad(modeldat['I_arr'], 1100, 'constant')
        
        # conv model
        modconv_map = fftconvolve(self.psfwin_map, modeldat['I_arr'], 'same')
        modconv_map /= np.sum(modconv_map)
        profmodconv = radial_prof(modconv_map, dx, dx)
        profgal_arr = profmodconv['prof']/profmodconv['prof'][0]
        profgal_arr[profgal_arr<0] = 0
        
        #self.modconv_map = modconv_map
        return profgal_arr
    
    def get_profclus_model_exact(self, **kwargs):
        
        dx = self.dx
        clusconv_map = fftconvolve(self.modconv_map, self.w_map, 'same')
        profclus_arr = radial_prof(clusconv_map, dx, dx)
        profclus_arr = profclus_arr['prof']
        return profclus_arr
        
    def get_profexcess_model(self, **kwargs):
        profgal_arr = self.get_profgal_model(**kwargs)
        # profclus_arr = self.get_profclus_model(**kwargs)
        profclus_arr = self.w_arr
        
        if 'Aclus' in kwargs.keys(): 
            Aclus = kwargs['Aclus']
        else:
            Aclus = 1

        # excess profile
        normg = self.profstackg_arr[0] - Aclus*profclus_arr[0]
        norms = self.profstackg_arr[0]
        profexfull_arr = normg*profgal_arr - norms*self.profpsf_arr
        
        if 'return_full' in kwargs:
            galfull = normg*profgal_arr
            psffull = norms*self.profpsf_arr
            clusfull = Aclus*profclus_arr
            return galfull, psffull, clusfull, profexfull_arr

        profex_arr = profile_radial_binning(profexfull_arr, self.r_weight)
        profclus_arr = profile_radial_binning(profclus_arr, self.r_weight)
  
        return profex_arr, Aclus*profclus_arr    
    
    def get_chi2(self, **kwargs):
        profex_arr, profclus_arr = self.get_profexcess_model(**kwargs)
        D = profex_arr + profclus_arr - self.profd_arr
        Covi = self.Covi
        D = D[np.newaxis,...]
        chi2 = D@Covi@D.T
        
        return chi2[0,0]
    
    def _log_likelihood(self, theta):
        xe2, Aclus = theta
        chi2 = self.get_chi2(xe2=xe2, Aclus=Aclus)
        return np.array([[-chi2/2]])

    def _log_prior(self, theta):
        xe2, Aclus = theta
        if 0.0001 < xe2 < 1 and 0.0 < Aclus < 200:
            return 0.
        return -np.inf

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                save_chain=True, savedir = None, savename=None):
        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob, pool=pool)
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



class joint_fit_mcmc:
    '''
    joint fit the params 
    '''
    def __init__(self, inst, im, ifield_list = [4,5,6,7,8]):
        self.inst = inst
        self.ifield_list = ifield_list
        self.Nfields = len(ifield_list)
        self.field_list = [fieldnamedict[i] for i in ifield_list]
        self.im = im
        self.m_min = magbindict['m_min'][im]
        self.m_max = magbindict['m_max'][im]
        self.param_fits = [fit_stacking_mcmc(inst, i, im) for i in ifield_list]
        
        self.dof_data = 0
        for i in range(self.Nfields):
            self.dof_data += self.param_fits[i].dof_data

    def get_chi2_fields(self, **kwargs):
        chi2_fields = []
        for i in range(self.Nfields):
            chi2_fields.append(self.param_fits[i].get_chi2(**kwargs))
        return chi2_fields
        
    def get_chi2(self, **kwargs):
        chi2tot = 0
        for i in range(len(self.ifield_list)):
            chi2tot += self.param_fits[i].get_chi2(**kwargs)
        return chi2tot
        
    def _log_likelihood(self, theta):
        xe2, Aclus = theta
        chi2 = self.get_chi2(xe2=xe2, Aclus=Aclus)
        return np.array([[-chi2/2]])

    def _log_prior(self, theta):
        xe2, Aclus = theta
        if 0.0001 < xe2 < 1 and 0.0 < Aclus < 200:
            return 0.
        return -np.inf

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                save_chain=True, savedir = None, savename=None):
        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob, pool=pool)
            sampler.run_mcmc(p0, steps, progress=True)
        
        if save_chain:
            if savedir is None:
                savedir = mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_2par_joint' + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename
        
        if return_chain:
            return sampler.get_chain()
        else:
            return

def run_mcmc_fit_joint(inst, im, **kwargs):
    param_fit = joint_fit_mcmc(inst, im)
    
    print('MCMC fit 2 params for TM%d %d < m < %d'\
          %(param_fit.inst, param_fit.m_min, param_fit.m_max))
    
    param_fit.run_mcmc(**kwargs)
    return param_fit


def get_mcmc_fit_params(inst, im, ifield=None):

    R200 = gal_profile_model().Wang19_profile(0,im)['params']['R200']
    
    if ifield in [4,5,6,7,8]:
        param_fit = fit_stacking_mcmc(inst, ifield, im)
        savename = 'mcmc_2par_' + param_fit.field + \
        '_m' + str(param_fit.m_min) + '_' + str(param_fit.m_max) + '.npy'
    elif ifield is None:
        param_fit = fit_stacking_mcmc(inst, 4, im)
        savename = 'mcmc_2par_joint'+ \
        '_m' + str(param_fit.m_min) + '_' + str(param_fit.m_max) + '.npy'
    
    savedir = mypaths['alldat'] + 'TM' + str(param_fit.inst) + '/'
    samples = np.load(savedir + savename)
    flatsamps = samples.copy()
    flatsamps[:,:,0] = flatsamps[:,:,0]
    flatsamps = flatsamps[100:,:,:].reshape((-1,2))
    xe2, Aclus = np.median(flatsamps, axis=0)
    xe2_low, Aclus_low = np.percentile(flatsamps, 16, axis=0)
    xe2_high, Aclus_high = np.percentile(flatsamps, 84, axis=0)
    
    fitparamdat = {'R200': R200, 'xe2': xe2, 'xe2_low': xe2_low, 'xe2_high': xe2_high,
                  'Re2': xe2*R200, 'Re2_low': xe2_low*R200, 'Re2_high': xe2_high*R200,
                  'Aclus': Aclus, 'Aclus_low': Aclus_low, 'Aclus_high': Aclus_high}
    return fitparamdat


'''
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
        tck = interpolate.splrep(np.log(np.array(data['rfull_arr'])[:16]),
                                 np.log(np.array(data['profpsf'])[:16]), k=1)
        radmap[dx,dx] = radmap[dx,dx+1]
        psfwin_map = np.exp(interpolate.splev(np.log(radmap),tck))
        psfwin_map[psfwin_map < 0] = 0
        profpsf = radial_prof(psfwin_map, dx, dx)
        profpsf_arr = np.array(profpsf['prof'])
        profpsf_arr /= profpsf_arr[0]
        
        self.r_arr = r_arr
        self.rfull_arr = np.array(data['rfull_arr'])
        self.profstackg_arr = np.array(data['profg'])
        self.profd_arr = profd_arr
        self.Cov = Cov
        self.Covi = Covi
        self.profd_err_diag = profd_err_diag
        self.profclus_arr = profclus_arr
        self.profpsf_arr = profpsf_arr
        self.psfwin_map = psfwin_map
        self.r_weight = np.array(data['r_weight'])
        self.dof_data = len(profd_arr)
        
    def get_profexcess_model(self, **kwargs):
        dx = 1200
        # model
        radmap = make_radius_map(np.zeros([201,201]), 100, 100)*0.7
        modeldat = gal_profile_model().Wang19_profile(radmap, self.im, **kwargs)
        modeldat['I_arr'] = np.pad(modeldat['I_arr'], 1100, 'constant')
        
        # conv model
        modconv_map = fftconvolve(self.psfwin_map, modeldat['I_arr'], 'same')
        self.modconv_map = modconv_map
        profmodconv = radial_prof(modconv_map, dx, dx)
        profmod_arr = profmodconv['prof']/profmodconv['prof'][0]
        profmod_arr[profmod_arr<0] = 0
        # excess profile
        profexfull_arr = self.profstackg_arr[0]*(profmod_arr - self.profpsf_arr)
        profex_arr = profile_radial_binning(profexfull_arr, self.r_weight)
        
        if 'return_full' in kwargs:
            galfull = self.profstackg_arr[0]*profmod_arr
            psffull = self.profstackg_arr[0]*self.profpsf_arr
            return galfull, psffull, profexfull_arr

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
    
    def _log_likelihood(self, theta):
        xe2, Aclus = theta
        chi2 = self.get_chi2(xe2=xe2, Aclus=Aclus)
        return np.array([[-chi2/2]])

    def _log_prior(self, theta):
        xe2, Aclus = theta
        if 0.0001 < xe2 < 1 and 0.0 < Aclus < 100:
            return 0.
        return -np.inf

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                save_chain=True, savedir = None, savename=None):
        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 100, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob, pool=pool)
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
'''