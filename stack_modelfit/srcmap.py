from run_fit import *
from skimage import restoration
import pandas as pd

class make_srcmap:
    def __init__(self, inst, m_min=None, m_max=None, srctype='g', catname = 'PanSTARRS',
                   PSmatch=False, ifield = 8, psf_ifield=None, Re2=2, normalize_model=True):
        self.inst = inst
        self.m_min = -5 if m_min is None else m_min
        self.m_max = 40 if m_max is None else m_max
        
        if catname=='PanSTARRS':
            self.catname = catname
            self.catdir = mypaths['PScatdat']
            if ifield in [4,5,6,7,8]:
                self.ifield = ifield
                self.field = fieldnamedict[ifield]
            else:
                raise Exception('ifield invalid (must be int between 4-8)')
        
        elif catname=='2MASS':
            self.catname = catname
            self.catdir = mypaths['2Mcatdat']
            if ifield in [4,5,6,7,8]:
                self.ifield = ifield
                self.field = fieldnamedict[ifield]
                self.PSmatch = PSmatch
            else:
                raise Exception('ifield invalid (must be int between 4-8)')

        elif catname=='HSC':
            self.catname = catname
            self.catdir = mypaths['HSCcatdat']
            if ifield in range(12):
                self.ifield = ifield
                self.field = 'W05_%d'%ifield
            else:
                raise Exception('ifield invalid (must be int between 0-11)')
        else:
            raise Exception('catname invalid (enter PanSTARRS or HSC)')
        
        if psf_ifield in [4,5,6,7,8]:
            self.psf_ifield = psf_ifield
        elif psf_ifield is None:
            self.psf_ifield = 'combined'
        else:
            raise Exception('psf_field invalid (must be int between 4-8 or 0)')
        
        if srctype in [None, 's', 'g', 'u']:
            self.srctype = srctype
        else:
            raise Exception('srctype invalid (must be None, "s", "g", or "u")')
        
        self.Re2 = Re2
        self.Npix_cb = 1024
        self.Nsub = 10
        
        self._get_psf()
        self.xls, self.yls, self.ms, self.Is, self.xss, self.yss, self.ms_inband = self._load_catalog()
        self._get_model()
        
        self.normalize_model = True
        if normalize_model:
            self._normalize_modmap()
            
    def _get_psf(self):
        pix_map = self._pix_func_substack()
        
        fitpsfdat = fitpsfdat=loadmat(mypaths['ciberdir'] + \
                'doc/20170617_Stacking/psf_analytic/TM'\
              + str(self.inst) + '/fitpsfdat.mat')['fitpsfdat'][0]
        if self.psf_ifield in [4,5,6,7,8]:
            im = 1 # use im = 1 PSF for all
            param_fit = fit_stacking_mcmc(self.inst, self.psf_ifield, im)
            psfwin_map = param_fit.psfwin_map/np.sum(param_fit.psfwin_map)
            self.dx = psfwin_map.shape[0]//2
            
            psfparams = fitpsfdat[self.ifield-1][7][0][0]
            beta, rc, norm  = float(psfparams[0]), float(psfparams[1]), float(psfparams[4])
            radmap = make_radius_map(np.zeros([2*self.dx+1, 2*self.dx+1]),
                                     self.dx, self.dx)*0.7
            psfbeta_map = norm * (1 + (radmap/rc)**2)**(-3*beta/2)

        else:
            im = 1
            psfwin_map = 0
            psfbeta_map = 0
            for ifield in [4,5,6,7,8]:
                param_fit = fit_stacking_mcmc(self.inst, ifield, im)
                psfwin_map += param_fit.psfwin_map/np.sum(param_fit.psfwin_map)
                self.dx = psfwin_map.shape[0]//2
                
                psfparams = fitpsfdat[self.ifield-1][7][0][0]
                beta, rc, norm  = float(psfparams[0]), float(psfparams[1]), float(psfparams[4])
                radmap = make_radius_map(np.zeros([2*self.dx+1, 2*self.dx+1]),
                                         self.dx, self.dx)*0.7
                psfbeta_map += norm * (1 + (radmap/rc)**2)**(-3*beta/2)

            psfwin_map /= 5
            psfbeta_map /= 5
            
        psfwin_map /= np.sum(psfwin_map)
        psfbeta_map /= np.sum(psfbeta_map)
        
        psf_map = restoration.richardson_lucy(psfwin_map, pix_map, 5)
        psf_map /= np.sum(psf_map) 

        
        self.pix_map = pix_map
        self.psfwin_map = psfwin_map
        self.psf_map = psf_map
        self.psfbeta_map = psfbeta_map
        
    def _load_catalog(self):
        dx = self.dx
        Npix_cb = self.Npix_cb
        Nsub = self.Nsub
        
        df = pd.read_csv(self.catdir + self.field + '.csv')
        if self.catname == 'PanSTARRS':
            xls = df['y'+str(self.inst)].values
            yls = df['x'+str(self.inst)].values
            ms = df['I_comb'].values
            clss = df['sdssClass'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==6))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==3))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-99))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            

        elif self.catname == '2MASS':
            xls = df['y'+str(self.inst)].values
            yls = df['x'+str(self.inst)].values
            ms = df['I'].values
            PSmatch = df['ps_match'].values
            if self.PSmatch is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]
            elif self.PSmatch:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (PSmatch==1))[0]
            elif not self.PSmatch:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (PSmatch==0))[0]
                
        elif self.catname == 'HSC':
            xls = df['x'].values - 2.5
            yls = df['y'].values - 2.5
            ms = df['Imag'].values
            clss = df['cls'].values
            if self.srctype == 's':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==-1))[0]
            elif self.srctype == 'g':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==1))[0]
            elif self.srctype == 'u':
                sp = np.where((ms>=self.m_min) & (ms<self.m_max) & (clss==2))[0]
            elif self.srctype is None:
                sp = np.where((ms>=self.m_min) & (ms<self.m_max))[0]            
            
        xls = xls[sp]
        yls = yls[sp]
        ms = ms[sp]
        
        ms_inband = ms.copy()
        Is = self._ABmag2Iciber(ms)
        if self.inst == 2:
            if self.catname == 'PanSTARRS':
                ms_inband = df['H_comb'].values[sp]
            elif self.catname == '2MASS':
                ms_inband = df['H'].values[sp]
            elif self.catname == 'HSC':
                ms_inband = df['Hmag'].values[sp]
            Is = self._ABmag2Iciber(ms_inband)
            
        sp = np.where((xls > -dx/Nsub) & (yls > -dx/Nsub) \
                      & (xls < Npix_cb+dx/Nsub) & (yls < Npix_cb+dx/Nsub))[0]
        xls = xls[sp]
        yls = yls[sp]
        ms = ms[sp]
        Is = Is[sp]
        ms_inband = ms_inband[sp]
        xss = np.round(xls * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        yss = np.round(yls * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        return xls, yls, ms, Is, xss, yss, ms_inband
        
    
    def _pix_func_substack(self, dx = 50, Nsub = 10):
        xx,yy = np.meshgrid(np.arange(2 * dx + 1), np.arange(2 * dx + 1))
        xx, yy = abs(xx - dx), abs(yy - dx)
        psf_pix = (Nsub - xx)*(Nsub - yy)
        psf_pix[(xx >= Nsub)] = 0
        psf_pix[(yy >= Nsub)] = 0
        psf_pix = psf_pix / np.sum(psf_pix)
        return psf_pix

    def _rebin_map_coarse(self, original_map, Nsub):
        m, n = np.array(original_map.shape)//(Nsub, Nsub)
        return original_map.reshape(m, Nsub, n, Nsub).mean((1,3))

    def _ABmag2Iciber(self, m):
        '''
        Convert AB mag to I [nW/m2/sr] on CIBER pixel (7'')
        '''
        sr = ((7./3600.0)*(np.pi/180.0))**2
        wl = band_info(self.inst).wl
        I = 3631. * 10**(-m / 2.5) * (3 / wl) * 1e6 / (sr*1e9)
        return I
   
    def _get_model(self):
        radmap = make_radius_map(self.psfwin_map, self.dx, self.dx)*0.7
        im = 1
        modeldat = gal_profile_model().Wang19_profile(radmap, im, Re2 = self.Re2)
        modconv_map = fftconvolve(self.psf_map, modeldat['I_arr'], 'same')
        modconv_map /= np.sum(modconv_map)
        self.modconv_map = modconv_map
        
        modconvwin_map = fftconvolve(self.psfwin_map, modeldat['I_arr'], 'same')
        modconvwin_map /= np.sum(modconvwin_map)
        self.modconvwin_map = modconvwin_map
    
    def _normalize_modmap(self):
        '''
        normalize the modmap -- match the first point of 
        model profile to PSF profile. This is meant to be 
        consistent with the excess definition
        '''
        dx = self.modconvwin_map.shape[0]//2
        profmod = radial_prof(self.modconvwin_map, dx, dx)
        profpsf = radial_prof(self.psfwin_map, dx, dx)
        norm = profpsf['prof'][0]/profmod['prof'][0]
        self.modconv_map = self.modconv_map * norm
        self.modconvwin_map = self.modconvwin_map * norm
        
    def run_srcmap(self, ptsrc=False, psf_beta_model=False, verbose=True):
        dx = self.dx
        Npix_cb = self.Npix_cb
        Nsub = self.Nsub
        
        # make sure this is updated
        self.xss = np.round(self.xls * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        self.yss = np.round(self.yls * Nsub + (Nsub/2 - 0.5) + 2 * dx).astype(np.int32)
        self.Is = self._ABmag2Iciber(self.ms_inband)
        
        srcmap_large = np.zeros([Npix_cb * Nsub + 4 * dx, Npix_cb * Nsub + 4 * dx])
        subpix_srcmap = self.psf_map if ptsrc else self.modconv_map
        if ptsrc and psf_beta_model:
            subpix_srcmap = self.psfbeta_map
        
        sp = np.where((self.xss-dx > 0) & (self.xss+dx < srcmap_large.shape[0]) & \
                     (self.yss-dx > 0) & (self.yss+dx < srcmap_large.shape[0]))[0]
        xss, yss, Is = self.xss[sp], self.yss[sp],self.Is[sp]
        
        for i,(xs,ys,I) in enumerate(zip(xss, yss, Is)):
            
            if len(Is)>20:
                if verbose and i%(len(Is)//20)==0:
                    print('run srcmap %d / %d (%.1f %%)'\
                          %(i, len(Is), i/len(Is)*100))
                    
            srcmap_large[xs-dx : xs+dx+1, ys-dx : ys+dx+1] += (subpix_srcmap*I)
        
        srcmap = self._rebin_map_coarse(srcmap_large, Nsub)*Nsub**2
        srcmap = srcmap[2*dx//Nsub : 2*dx//Nsub+Npix_cb,\
                        2*dx//Nsub : 2*dx//Nsub+Npix_cb]
        return srcmap

    def run_srcmap_nopsf(self):
        self.Is = self._ABmag2Iciber(self.ms_inband)
        srcmap = np.histogram2d(self.xls,self.yls,
                                np.arange(self.Npix_cb+1)-0.5,
                                weights=self.Is)[0]
        return srcmap