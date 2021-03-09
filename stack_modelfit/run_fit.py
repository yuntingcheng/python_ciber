from scipy import interpolate
from scipy.signal import fftconvolve
import pyfftw
import emcee
from multiprocessing import Pool
from utils import *
from psfsynth import * 
from stack import *
from clustering import *
from micecat import *
from micecat_auto import *

class fit_stacking_mcmc:
    
    def __init__(self, inst, ifield, im, m_min=None, m_max=None, filt_order=None,
     data_maps=None, loaddir=None, modify_cov=False, subsub=False, 
     cov_method='jackknife', **kwargs):

        self.inst = inst
        self.ifield = ifield
        self.field = fieldnamedict[ifield]
        if im == 'all':
            self.im = 3
            self.m_min = 17
            self.m_max = 20
            self.allmag = True
        else:
            self.im = im
            self.m_min = m_min if m_min is not None else magbindict['m_min'][im]
            self.m_max = m_max if m_max is not None else magbindict['m_max'][im]
            self.allmag = False
        self.dx = 1200
        self.data_maps = data_maps
        self.filt_order = filt_order if filt_order is not None \
                                        else filt_order_dict[inst]
        self.modify_cov = modify_cov
        self.subsub = subsub
        self.cov_method = cov_method
        
        self._fit_data_preprocess(loaddir,**kwargs)
        
    def _fit_data_preprocess(self,loaddir,**kwargs):
        self._load_data(loaddir,**kwargs)
        self._get_model_1h()
        self._get_model_2h()
        self._get_model_psf()

    def _load_data(self, loaddir, **kwargs):
        if self.allmag:
            stackdat = stackdat_combine_mags(self.inst, self.ifield, **kwargs)
            self.Nsrc_arr = stackdat['Nsrc_arr']
        else:
            stackdat = stacking(self.inst, self.ifield,
                                self.m_min, self.m_max,
                                filt_order=self.filt_order,loaddir=loaddir, 
                                load_from_file=True, BGsub=False,
                                subsub=self.subsub, cov_method=self.cov_method,
                                **kwargs).stackdat
            self.Nsrc = stackdat['Nsrc']

        self.Njk = stackdat['Nsub']
        self.rbins = stackdat['rbins']
        self.rbinedges = stackdat['rbinedges']
        self.rsubbins = stackdat['rsubbins']
        self.rsubbinedges = stackdat['rsubbinedges']
        self.profcb = stackdat['profcb']
        self.profcb_cov = stackdat['cov']['profcb']
        self.profcb_covsub = stackdat['cov']['profcbsub']
        self.profcb_err = np.sqrt(np.diag(stackdat['cov']['profcb']))
        self.profcb_sub = stackdat['profcbsub']
        self.profcb_sub_err = np.sqrt(np.diag(stackdat['cov']['profcbsub']))
        self.profex = stackdat['ex']['profcb']
        self.profex_sub = stackdat['ex']['profcbsub']
        self.cov = stackdat['excov']['profcb']
        self.covsub = stackdat['excov']['profcbsub']

        # https://arxiv.org/pdf/astro-ph/0608064.pdf Eq 17
        n, p = self.Njk, len(self.rbins)
        self.cov_inv_debias = (n - p - 2) / (n - 1)
        n, p = self.Njk, len(self.rsubbins)
        self.covsub_inv_debias = (n - p - 2) / (n - 1)

        # self.cov_inv = np.linalg.inv(self.cov)
        self.covsub_inv, self.covsub_inv_Nmode \
        = self._get_covsub_inv(self.covsub)
        self.dof_data = len(self.profcb_sub)
        
        # get modified cov
        if self.modify_cov:
            covg = stackdat['cov']['profcbsub'].copy()
            covpsf = stackdat['PSFcov']['profcbsub'].copy()
            covg1 = covg.copy()
            scale = stackdat['PSF']['Nsrc'] / stackdat['Nsrc']
            covg1[:4,:] = covpsf[:4,:] * scale
            covg1[:,:4] = covpsf[:,:4] * scale
            covex1 = covg1 + covpsf
            self.covsub1 = covex1
            self.covsub_inv, self.covsub_inv_Nmode = \
            self._get_covsub_inv(self.covsub1)

        return stackdat
    
    def _get_covsub_inv(self, Cov):
        
        if self.inst ==1 and self.im == 0:
            # Nmodes = {4:15, 5:15, 6:12, 7:15, 8:15}
            Nmodes = {4:15, 5:15, 6:15, 7:15, 8:15}
        else:
            Nmodes = {4:15, 5:15, 6:15, 7:15, 8:15}
        
        Nmode = Nmodes[self.ifield]
        U, s, VT = np.linalg.svd(Cov)
        V = VT.T
        UT = U.T
        S = np.diag(s)
        Sinv  = np.diag(1/s)
        Cov_svd = U[:,:Nmode]@S[:Nmode,:Nmode]@VT[:Nmode,:]
        Covi_svd = V[:,:Nmode]@Sinv[:Nmode,:Nmode]@UT[:Nmode,:]

        return Covi_svd, Nmode

    def _get_model_1h(self):
        if self.allmag:
            prof1h, prof1h_sub = micecat_1h_combine_mags(self.inst, self.ifield,
             wi_arr=self.Nsrc_arr)
            self.prof1h = prof1h
            self.prof1h_sub = prof1h_sub
        else:
            _, mc_avg, mc_std, _ = get_micecat_sim_1h(self.inst, self.im, 
                Mhcut=1e14, R200cut=0, zcut=0.15, sub=False)
            _, mc_avg_sub, mc_std_sub, _ = get_micecat_sim_1h(self.inst, self.im,
                Mhcut=1e14, R200cut=0, zcut=0.15, sub=True, 
                subsub=self.subsub, Nrebin=9)#!!!
            self.prof1h = mc_avg
            self.prof1h_sub = mc_avg_sub
        return
    
    def _get_model_2h_analytic(self):

        if self.data_maps is None:
            data_maps = {1: image_reduction(1), 2: image_reduction(2)}
        else:
            data_maps = self.data_maps

        mask_inst1, mask_inst2 = \
        load_processed_images(data_maps,
                              return_names=[(1,self.ifield,'mask_inst'),
                                            (2,self.ifield,'mask_inst')])
        
        srcdat = ps_src_select(self.inst, self.ifield, self.m_min, self.m_max, 
                [mask_inst1, mask_inst2], sample_type='all')
        
        dx = self.dx
        theta_arr = np.logspace(-1,3.2,100)
        w_arr = wgI(srcdat['zg_arr'], theta_arr)
        radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]), dx, dx)*0.7
        tck = interpolate.splrep(np.log(theta_arr), w_arr, k=1)
        radmap[dx,dx] = radmap[dx,dx+1]
        w_map = interpolate.splev(np.log(radmap),tck)
        
        self.prof2h = radial_prof(w_map, rbinedges=self.rbinedges/0.7,
                                  return_full=False)
        self.prof2h_sub = radial_prof(w_map, rbinedges=self.rsubbinedges/0.7,
                                      return_full=False)
        
        return

    def _get_model_2h(self):
        if self.allmag:
            prof2h, prof2h_sub = micecat_2h_combine_mags(self.inst, self.ifield,
             wi_arr=self.Nsrc_arr)
            self.prof2h = prof2h
            self.prof2h_sub = prof2h_sub
        else:
            rbins, mc_avg, mc_avg_fit, rsubbins, mc_avgsub, mc_avgsub_fit = \
            micecat_profile_fit(self.inst, self.im, filt_order=self.filt_order,
             return_full=True, subsub=self.subsub, Nrebin=9)#!!!

            self.prof2h = mc_avg_fit
            self.prof2h_sub = mc_avgsub_fit
        
        return
     
    def _get_model_psf(self):
        
        # fname = mypaths['alldat'] + 'TM'+ str(self.inst) + '/psfdata.pkl'
        # with open(fname,"rb") as f:
        #     psfdata = pickle.load(f)

        # spr = np.where(self.rbins<30)[0]
        # tck = interpolate.splrep(np.log(self.rbins[spr]),
        #                          np.log(psfdata[self.ifield]['prof'][spr]), k=1)
        # radmap[dx,dx] = radmap[dx,dx+1]
        # psfwin_map = np.exp(interpolate.splev(np.log(radmap),tck))
        # psfwin_map[psfwin_map < 0] = 0
        
        fname = mypaths['alldat'] + 'TM'+ str(self.inst) +\
         '/psfdata_synth_%s.pkl'%(fieldnamedict[self.ifield])
        with open(fname,"rb") as f:
            psfdata = pickle.load(f)
            
        dx = self.dx
        radmap = make_radius_map(np.zeros([2*dx+1, 2*dx+1]), dx, dx)*0.7
        psfwin_map = psf_comb_interpolate(self.inst, self.ifield, self.im, radmap)

        profpsf_arr = radial_prof(psfwin_map, return_full=False)
        profpsf_arr /= profpsf_arr[0]

        profpsf = radial_prof(psfwin_map, rbinedges=self.rbinedges/0.7,
                                  return_full=False)
        profpsf_sub = radial_prof(psfwin_map, rbinedges=self.rsubbinedges/0.7,
                                      return_full=False)
        
        self.profpsf = profpsf / profpsf[0]
        self.profpsf_sub = profpsf_sub / profpsf[0]
        self.psfwin_map = psfwin_map
        
        a = pyfftw.empty_aligned((2401,2401), dtype='complex64') 
        fftpsf = pyfftw.empty_aligned((2401,2401), dtype='complex64') 
        psffft_obj = pyfftw.FFTW(a,fftpsf, axes=(0,1),threads=1, 
                         direction='FFTW_FORWARD',flags=('FFTW_MEASURE',))
        psffft_obj(psfwin_map)
        self.fft_psfwin_map = fftpsf

        return

    def get_profgal_model(self, subbin=True, return_all=False, **kwargs):
        
        dx = self.dx
        
        # model
        radmap = make_radius_map(np.zeros([201,201]), 100, 100)*0.7
        modeldat = gal_profile_model().Wang19_profile(radmap, self.im, **kwargs)
        modeldat['I_arr'] = np.pad(modeldat['I_arr'], 1100, 'constant')
        
        # conv model (pyFFTw)
        a = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        fftmod = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        modfft_obj = pyfftw.FFTW(a,fftmod, axes=(0,1),threads=1, 
                                 direction='FFTW_FORWARD',flags=('FFTW_MEASURE',))
        modfft_obj(modeldat['I_arr'])
        b = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        modconv_map = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        convifft_obj = pyfftw.FFTW(b,modconv_map, axes=(0,1),threads=1, 
                                   direction='FFTW_BACKWARD',flags=('FFTW_MEASURE',))
        convifft_obj(np.conj(self.fft_psfwin_map)*fftmod)
        modconv_map = np.real(np.fft.fftshift(modconv_map))
        modconv_map[modconv_map<0] = 0
        modconv_map /= np.sum(modconv_map)


        profgal_sub = radial_prof(modconv_map, rbinedges=self.rsubbinedges/0.7,
                                      return_full=False)
        profgal = radial_prof(modconv_map, rbinedges=self.rbinedges/0.7,
                                  return_full=False)
        
        if return_all:
            return profgal / profgal[0], profgal_sub / profgal[0]

        if subbin:
            return profgal_sub / profgal[0]
        else:
            return profgal / profgal[0]

    def get_gal_profile_norm(self,**kwargs):
        # Find the 1st point amplitude of intrinsic gal prof, if galxPSF is 
        # normalized to 1 at the 1st point
        radmap = make_radius_map(np.zeros([201,201]), 100, 100)*0.7
        modeldat = gal_profile_model().Wang19_profile(radmap, self.im, **kwargs)
        modeldat['I_arr'] = np.pad(modeldat['I_arr'], 1100, 'constant')

        # conv model (pyFFTw)
        a = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        fftmod = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        modfft_obj = pyfftw.FFTW(a,fftmod, axes=(0,1),threads=1, 
                                 direction='FFTW_FORWARD',flags=('FFTW_MEASURE',))
        modfft_obj(modeldat['I_arr'])
        b = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        modconv_map = pyfftw.empty_aligned((2401,2401), dtype='complex64')
        convifft_obj = pyfftw.FFTW(b,modconv_map, axes=(0,1),threads=1, 
                                   direction='FFTW_BACKWARD',flags=('FFTW_MEASURE',))
        convifft_obj(np.conj(self.fft_psfwin_map)*fftmod)
        modconv_map = np.real(np.fft.fftshift(modconv_map))
        modconv_map[modconv_map<0] = 0
        modconv_map = modconv_map / np.sum(modconv_map) * np.sum(modeldat['I_arr'])
        
        profgal = radial_prof(modconv_map, rbinedges=self.rbinedges/0.7,
                                  return_full=False)
        profgal0 = radial_prof(modeldat['I_arr'], rbinedges=self.rbinedges/0.7,
                                  return_full=False)

        return profgal0[0]/profgal[0]

    def get_profgal_model_interp(self):

        print('Pre-computing model profiles for interpolation...')
        xe2_arr = np.logspace(np.log10(0.0001),np.log10(1),100)
        profgal_arr = []
        profgal_sub_arr = []
        for xe2 in xe2_arr:
            profgal, profgal_sub = self.get_profgal_model(return_all=True, xe2=xe2)
            profgal_arr.append(profgal)
            profgal_sub_arr.append(profgal_sub)

        profgal_arr = np.array(profgal_arr)
        profgal_sub_arr = np.array(profgal_sub_arr)
        tcks = [interpolate.splrep(xe2_arr, np.log10(profgal_arr[:,i])) \
                for i in range(profgal_arr.shape[1])]
        tcks_sub = [interpolate.splrep(xe2_arr, np.log10(profgal_sub_arr[:,i])) \
                    for i in range(profgal_sub_arr.shape[1])]
        
        self.logprofgal_tcks = tcks
        self.logprofgal_sub_tcks = tcks_sub
        return

    def get_profexcess_model(self, fast=False, **kwargs):
        
        if fast and 'xe2' in kwargs.keys():
            if 'logprofgal_tcks' not in dir(self):
                self.get_profgal_model_interp()
            xe2 = kwargs['xe2']
            profgal = 10**np.array([interpolate.splev([xe2], tck)[0]\
             for tck in self.logprofgal_tcks])
            profgal_sub = 10**np.array([interpolate.splev([xe2], tck)[0]\
             for tck in self.logprofgal_sub_tcks])

        else:
            profgal, profgal_sub = self.get_profgal_model(return_all=True,**kwargs)

        if 'A1h' in kwargs.keys(): 
            A1h = kwargs['A1h']
        else:
            A1h = 1
            
        if 'A2h' in kwargs.keys(): 
            A2h = kwargs['A2h']
        else:
            A2h = 1

        # excess profile
        normg = self.profcb[0] - A1h*self.prof1h[0] - A2h*self.prof2h[0]
        norms = self.profcb[0]
        
        profex = normg*profgal - norms*self.profpsf
        profex_sub = normg*profgal_sub - norms*self.profpsf_sub
        
        modelprof = {}
        modelprof['rbins'] = self.rbins
        modelprof['rbinedges'] = self.rbinedges
        modelprof['rsubbins'] = self.rsubbins
        modelprof['rsubbinedges'] = self.rsubbinedges
        modelprof['profgal'] = normg*profgal
        modelprof['profgal_sub'] = normg*profgal_sub
        modelprof['profpsf'] = norms*self.profpsf
        modelprof['profpsf_sub'] = norms*self.profpsf_sub
        modelprof['prof1h'] = A1h*self.prof1h
        modelprof['prof1h_sub'] = A1h*self.prof1h_sub
        modelprof['prof2h'] = A2h*self.prof2h
        modelprof['prof2h_sub'] = A2h*self.prof2h_sub
        modelprof['profex'] = profex
        modelprof['profex_sub'] = profex_sub
        modelprof['normg'] = normg
        modelprof['norms'] = norms

        return modelprof
    
    def get_chi2(self, **kwargs):
        modelprof = self.get_profexcess_model(fast=True, **kwargs)
        D = modelprof['profex_sub'] + modelprof['prof1h_sub'] +\
        modelprof['prof2h_sub'] - self.profex_sub
        Covi = self.covsub_inv
        D = D[np.newaxis,...]
        chi2 = D@Covi@D.T
        
        return chi2[0,0]
    
    def _log_likelihood(self, theta):
        xe2, A1h, A2h = theta
        chi2 = self.get_chi2(xe2=xe2, A1h=A1h, A2h=A2h)
        return np.array([[-chi2/2]])

    def _log_prior(self, theta):
        xe2, A1h, A2h = theta
        if 0.0001 < xe2 < 1 and 0.0 < A1h < 50 and 0.0 < A2h < 200:
            return 0.
        return -np.inf

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                return_sampler=False, save_chain=True, savedir = None, savename=None,
                moves=None):
        if 'logprofgal_tcks' not in dir(self):
            self.get_profgal_model_interp()
        
        ndim = 3
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 50, nwalkers)
        p03 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02, p03), axis=1)
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, 
                self._log_prob, pool=pool, moves=moves)
            sampler.run_mcmc(p0, steps, progress=progress)
        
        if save_chain:
            if savedir is None:
                savedir = './mcmc_data/'+ 'TM' + str(self.inst) + '/'#mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_3par_' + self.field + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                if self.subsub:
                    savename = 'mcmc_3par_' + self.field + \
                    '_m' + str(self.m_min) + '_' + str(self.m_max) + '_sub.npy'
                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename
        
        if return_sampler:
            return sampler
        if return_chain:
            return sampler.get_chain()
        else:
            return

    def _log_likelihood_2par(self, theta):
        xe2, A2h = theta
        chi2 = self.get_chi2(xe2=xe2, A1h=0, A2h=A2h)
        return np.array([[-chi2/2]])

    def _log_prior_2par(self, theta):
        xe2, A2h = theta
        if 0.0001 < xe2 < 1 and 0.0 < A2h < 200:
            return 0.
        return -np.inf

    def _log_prob_2par(self, theta):
        lp = self._log_prior_2par(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood_2par(theta)

    def run_mcmc_2par(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                return_sampler=False, save_chain=True, savedir = None, savename=None,
                moves=None):

        if 'logprofgal_tcks' not in dir(self):
            self.get_profgal_model_interp()

        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob_2par,
             pool=pool, moves=moves)
            sampler.run_mcmc(p0, steps, progress=progress)
        
        if save_chain:
            if savedir is None:
                savedir = './mcmc_data/'+ 'TM' + str(self.inst) + '/'#mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_2par_' + self.field + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                if self.subsub:
                    savename = 'mcmc_2par_' + self.field + \
                    '_m' + str(self.m_min) + '_' + str(self.m_max) + '_sub.npy'

                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename
        
        if return_sampler:
            return sampler
        if return_chain:
            return sampler.get_chain()
        else:
            return

    def get_chi2_pte(self, Npar=3, **kwargs):
        # reuturn debiased chi2, pte, dof, original chi2

        if 'chi2' not in kwargs.keys(): 
            chi2 = self.get_chi2(**kwargs)
        else:
            chi2 = kwargs['chi2']

        dof = self.dof_data - Npar

        debias_scaling = self.covsub_inv_debias
        if debias_scaling <= 0:
            print('Warning: Inv Cov not well-defined.')
            return 0, 1, dof, chi2
        
        pte = scipy.stats.distributions.chi2
        pte = pte.sf(chi2 * debias_scaling, dof)
        return chi2 * debias_scaling, pte, dof, chi2

class joint_fit_mcmc:
    
    # joint fit the params 
    
    def __init__(self, inst, im, m_min=None, m_max=None, 
        filt_order=None, fast=True, ifield_list = [4,5,6,7,8],
         subsub=False, **kwargs):
        self.inst = inst
        self.ifield_list = ifield_list
        self.Nfields = len(ifield_list)
        self.field_list = [fieldnamedict[i] for i in ifield_list]
        if im == 'all':
            self.im = 'all'
            self.m_min = 17
            self.m_max = 20
            self.allmag = True
        else:
            self.im = im
            self.m_min = m_min if m_min is not None else magbindict['m_min'][im]
            self.m_max = m_max if m_max is not None else magbindict['m_max'][im]
            self.allmag = False
        self.filt_order = filt_order if filt_order is not None \
                                        else filt_order_dict[inst]
        self.subsub = subsub
        
        self.param_fits = []
        for ifield in ifield_list:
            fit_params = fit_stacking_mcmc(inst, ifield, im, 
                m_min=self.m_min, m_max=self.m_max, 
                filt_order=self.filt_order,subsub=subsub,**kwargs)
            if fast:
                fit_params.get_profgal_model_interp()
            self.param_fits.append(fit_params)
        
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

        use_fields = []
        for i in range(len(self.ifield_list)):
            db_scale = self.param_fits[i].covsub_inv_debias
            if db_scale > 0:
                use_fields.append(i)

        for i in use_fields:
            db_scale = self.param_fits[i].covsub_inv_debias
            chi2tot += self.param_fits[i].get_chi2(**kwargs) * db_scale
        return chi2tot
        
    def _log_likelihood(self, theta):
        xe2, A1h, A2h = theta
        chi2 = self.get_chi2(xe2=xe2, A1h=A1h, A2h=A2h)
        return np.array([[-chi2/2]])

    def _log_prior(self, theta):
        xe2, A1h, A2h = theta
        if 0.0001 < xe2 < 1 and 0.0 < A1h < 50 and 0.0 < A2h < 200:
            return 0.
        return -np.inf

    def _log_prob(self, theta):
        lp = self._log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood(theta)

    def run_mcmc(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                return_sampler=False, save_chain=True, savedir = None, savename=None):
        ndim = 3
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 50, nwalkers)
        p03 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02, p03), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob, pool=pool)
            sampler.run_mcmc(p0, steps, progress=True)
        
        if save_chain:
            if savedir is None:
                savedir = './mcmc_data/'+ 'TM' + str(self.inst) + '/'#mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_3par_joint' + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                if self.subsub:
                    savename = 'mcmc_3par_joint' + \
                    '_m' + str(self.m_min) + '_' + str(self.m_max) + '_sub.npy'
                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename

        if return_sampler:
            return sampler
        if return_chain:
            return sampler.get_chain()
        else:
            return

    def _log_likelihood_2par(self, theta):
        xe2, A2h = theta
        chi2 = self.get_chi2(xe2=xe2, A1h=0, A2h=A2h)
        return np.array([[-chi2/2]])

    def _log_prior_2par(self, theta):
        xe2, A2h = theta
        if 0.0001 < xe2 < 1 and 0.0 < A2h < 200:
            return 0.
        return -np.inf

    def _log_prob_2par(self, theta):
        lp = self._log_prior_2par(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self._log_likelihood_2par(theta)

    def run_mcmc_2par(self, nwalkers=100, steps=500, progress=True, return_chain=False, 
                return_sampler=False, save_chain=True, savedir = None, savename=None):
        ndim = 2
        p01 = np.random.uniform(0.0001, 1, nwalkers)
        p02 = np.random.uniform(0.0, 200, nwalkers)
        p0 = np.stack((p01, p02), axis=1)
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self._log_prob_2par, pool=pool)
            sampler.run_mcmc(p0, steps, progress=True)
        
        if save_chain:
            if savedir is None:
                savedir = './mcmc_data/'+ 'TM' + str(self.inst) + '/'#mypaths['alldat'] + 'TM' + str(self.inst) + '/'
            if savename is None:
                savename = 'mcmc_2par_joint' + \
                '_m' + str(self.m_min) + '_' + str(self.m_max) + '.npy'
                if self.subsub:
                    savename = 'mcmc_2par_joint' + \
                    '_m' + str(self.m_min) + '_' + str(self.m_max) + '_sub.npy'

                
            np.save(savedir + savename, sampler.get_chain(), sampler)
            self.mcmc_savename = savedir + savename

        if return_sampler:
            return sampler
        if return_chain:
            return sampler.get_chain()
        else:
            return

    def get_chi2_pte(self, Npar=3, **kwargs):
        chi2tot = 0
        chi2tot_debias = 0
        dof = 0
        use_fields = []
        for i in range(len(self.ifield_list)):
            db_scale = self.param_fits[i].covsub_inv_debias
            if db_scale > 0:
                chi2field = self.param_fits[i].get_chi2(**kwargs) 
                chi2tot += chi2field
                chi2tot_debias += chi2field * db_scale
                dof += self.param_fits[i].dof_data
        dof -= Npar
        pte = scipy.stats.distributions.chi2
        pte = pte.sf(chi2tot_debias, dof)

        return chi2tot_debias, pte, dof, chi2tot

def get_mcmc_chains(inst, im, ifield=None, Npar=3, subsub=False, burn_in=0):
    chaindir = './mcmc_data/'+ 'TM' + str(inst) + '/'#mypaths['alldat'] + 'TM' + str(inst) + '/'
    if ifield in [4,5,6,7,8]:
        savename = 'mcmc_' + str(Npar) + 'par_' + fieldnamedict[ifield] + \
        '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'
    elif ifield is None:
        savename = 'mcmc_' + str(Npar) + 'par_joint' + \
        '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'

    if subsub:
        savename = savename[:-4] + '_sub.npy'
    chains = np.load(chaindir + savename)
    
    return chains[burn_in:,...]


def get_posterior_interval(samples, ci=68, return_hist=False):
    Nsamps = len(samples)
    N68 = Nsamps * ci/100
    hist, binedges = np.histogram(samples,bins=np.linspace(0,np.max(samples),Nsamps//500))
    bins = (binedges[1:] + binedges[:-1]) / 2
    
    idx_in = []
    idx_low = []
    idx_high = []
    Nint = 0

    idx_in.append(np.argmax(hist))
    idx_low = [i for i in np.arange(len(bins)) if i < np.min(idx_in)]
    idx_high = [i for i in np.arange(len(bins))[::-1] if i > np.max(idx_in)]

    param_mid = bins[idx_in[0]]
    Nint += hist[idx_in[0]]
    while Nint < N68:
        if len(idx_low) == 0:
            idx_add = idx_high.pop()
            idx_in.append(idx_add)

        elif len(idx_high) == 0:
            idx_add = idx_low.pop()
            idx_in.append(idx_add)

        elif np.max(hist[idx_low]) <= np.max(hist[idx_high]):
            idx_add = idx_high.pop()
            idx_in.append(idx_add)

        elif np.max(hist[idx_low]) > np.max(hist[idx_high]):
            idx_add = idx_low.pop()
            idx_in.append(idx_add)

        Nint += hist[idx_add] 

    param_low = binedges[np.min(idx_in)]
    param_high = binedges[np.max(idx_in)+1]
    
    if return_hist:
        return param_mid, param_low, param_high, hist, bins, binedges
    return param_mid, param_low, param_high

def get_mcmc_fit_params_3par(inst, im, ifield=None,burn_in=150,
    chaindir=None, subsub=False, savename=None, return_samples=False, **kwargs):

    R200 = gal_profile_model().Wang19_profile(0,im)['params']['R200']

    if savename is None:
        if ifield in [4,5,6,7,8]:
            savename = 'mcmc_3par_' + fieldnamedict[ifield] + \
            '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'
        elif ifield is None:
            savename = 'mcmc_3par_joint' + \
            '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'

        if subsub:
            savename = savename[:-4] + '_sub.npy'

    if chaindir is None:
        chaindir = './mcmc_data/'+ 'TM' + str(inst) + '/'#mypaths['alldat'] + 'TM' + str(inst) + '/'
    samples = np.load(chaindir + savename)
    flatsamps = samples.copy()

    # chain rejection
    chain_use_idx = []
    Nchain = flatsamps.shape[1]
    for i in range(Nchain):
        if not np.any(flatsamps[100:,i,1]>100):
            chain_use_idx.append(i)
    flatsamps = flatsamps[burn_in:,chain_use_idx,:].reshape((-1,3))

    # get 68 C.I.
    xe2, xe2_low, xe2_high = get_posterior_interval(flatsamps[:,0], **kwargs)
    A1h, A1h_low, A1h_high = get_posterior_interval(flatsamps[:,1], **kwargs)
    A2h, A2h_low, A2h_high = get_posterior_interval(flatsamps[:,2], **kwargs)

    fitparamdat = {'R200': R200, 'xe2': xe2, 'xe2_low': xe2_low, 'xe2_high': xe2_high,
                  'Re2': xe2*R200, 'Re2_low': xe2_low*R200, 'Re2_high': xe2_high*R200,
                  'A1h': A1h, 'A1h_low': A1h_low, 'A1h_high': A1h_high,
                  'A2h': A2h, 'A2h_low': A2h_low, 'A2h_high': A2h_high}

    if return_samples:
        return fitparamdat, flatsamps
    return fitparamdat

def get_mcmc_fit_params_2par(inst, im, ifield=None,
    burn_in=150,chaindir=None, subsub=False, savename=None):

    R200 = gal_profile_model().Wang19_profile(0,im)['params']['R200']

    if savename is None:
        if ifield in [4,5,6,7,8]:
            savename = 'mcmc_2par_' + fieldnamedict[ifield] + \
            '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'
        elif ifield is None:
            savename = 'mcmc_2par_joint' + \
            '_m' + str(magbindict['m_min'][im]) + '_' + str(magbindict['m_max'][im]) + '.npy'

        if subsub:
            savename = savename[:-4] + '_sub.npy'

    if chaindir is None:
        chaindir = mypaths['alldat'] + 'TM' + str(inst) + '/'
    samples = np.load(chaindir + savename)
    flatsamps = samples.copy()
    flatsamps = flatsamps[burn_in:,:,:].reshape((-1,2))
    
    # get 68 C.I.
    xe2, xe2_low, xe2_high = get_posterior_interval(flatsamps[:,0])
    A2h, A2h_low, A2h_high = get_posterior_interval(flatsamps[:,1])

    fitparamdat = {'R200': R200, 'xe2': xe2, 'xe2_low': xe2_low, 'xe2_high': xe2_high,
                  'Re2': xe2*R200, 'Re2_low': xe2_low*R200, 'Re2_high': xe2_high*R200,
                  'A2h': A2h, 'A2h_low': A2h_low, 'A2h_high': A2h_high}
    return fitparamdat