import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
from scipy.io import loadmat
import pickle
from ciber_info import *
from utils_plotting import *
from scipy.spatial import cKDTree
from IPython.display import clear_output
import astropy.io.fits as fits
import astropy.wcs as wcs
from astropy.coordinates import SkyCoord
from astropy.coordinates import match_coordinates_sky
from astropy import units as u
from astropy import constants as const
from astropy import cosmology
cosmo = cosmology.Planck15


def ABmag2Fjy(m):
    '''
    Convert AB mag to flux in Jy
    m_AB = -2.5 log(F/3631Jy)
    '''
    
    F = 3631. * 10 ** (-m / 2.5)
    return F

def ABmag2Iciber(inst, m):
    '''
    Convert AB mag to I [nW/m2/sr] on CIBER pixel (7'')
    '''
    sr = ((7./3600.0)*(np.pi/180.0))**2
    wl = band_info(inst).wl
    I = 3631. * 10**(-m / 2.5) * (3 / wl) * 1e6 / (sr*1e9)
    return I

def Iciber2FJy(inst, I):
    '''
    Convert CIBER vIv [nW/m2/sr] to F[Jy] (assuming point source in CIBER pixel)
    '''
    sr = ((7./3600.0)*(np.pi/180.0))**2
    wl = band_info(inst).wl
    nu = 3e14/wl
    
    F = I * sr / nu #[nW/m^2/Hz]
    F *= 1e17 #[Jy]
    return F

def make_radius_map(mapin, cenx, ceny):
    '''
    return radmap of size mapin.shape. 
    radmap[i,j] = distance between (i,j) and (cenx, ceny)
    '''
    Nx, Ny = mapin.shape
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    radmap = np.sqrt((xx - cenx)**2 + (yy - ceny)**2)
    return radmap

def sigma_clip_mask(mapin, maskin, iter_clip=3, sig=5):
    b = mapin[maskin!=0]
    for i in range(iter_clip):
        clipmax = np.nanmedian(b) + sig * np.nanstd(b)
        clipmin = np.nanmedian(b) - sig * np.nanstd(b)
        b = b[(b>clipmin) & (b<clipmax)]
    maskout = maskin.copy()
    maskout[np.where((mapin>clipmax) | (mapin<clipmin))]=0
    return maskout

def radial_prof(mapin, cenx=None, ceny=None, log=True, nbins=25, maskin=None,
                weight=None, rbinedges=None, return_full=True):

    if cenx is None:
        cenx = mapin.shape[0]//2
    if ceny is None:
        ceny = mapin.shape[1]//2

    radmap = make_radius_map(mapin,  cenx, ceny)

    if weight is None:
        weight = np.ones(mapin.shape)

    if maskin is None:
        maskin = np.ones(mapin.shape)

    if rbinedges is None:

        if log:
            rbinedges = np.logspace(0, np.log10(np.max(radmap)), nbins+1)
            rbinedges[-1] *= 1.01
        else:
            rbinedges = np.linspace(0, np.max(radmap), nbins+1)

        rbinedges[-1] *= 1.01
    else:
        nbins = len(rbinedges) - 1


    radmap = radmap.flatten()
    weight = weight.flatten()
    maskin = maskin.flatten()
    mapin = mapin.flatten()

    histw = np.histogram(radmap, bins=rbinedges, weights=weight*maskin)[0]
    histmapw = np.histogram(radmap, bins=rbinedges, weights=mapin*weight*maskin)[0]
    prof = histmapw / histw
    
    if not return_full:
        return prof
    else:
        
        histN = np.histogram(radmap, bins=rbinedges, weights=maskin)[0]
        histrw = np.histogram(radmap, bins=rbinedges, weights=radmap*weight*maskin)[0]
        histmap = np.histogram(radmap, bins=rbinedges, weights=mapin*maskin)[0]
        histmap2 = np.histogram(radmap, bins=rbinedges, weights=mapin**2*maskin)[0]
        
        rbins = histrw / histw
        var = histmap2/histN - (histmap/histN)**2
        err = np.zeros_like(prof)
        sppos = np.where(var>0)[0]
        err[sppos] = var[sppos] / np.sqrt(histN[sppos])
        Npix = histN.astype(int)
        
        profdat = {}
        profdat['rbinedges'] = rbinedges
        profdat['rbins'] = rbins
        profdat['prof'] = prof
        profdat['err'] = err
        profdat['Npix'] = Npix
        
        return profdat

def profile_radial_binning(prof, w):
    prof15 = np.zeros(15)
    prof25 = prof
    prof15[1:-1] = prof25[6:19]
    prof15[0] = np.sum(w[:6]*prof25[:6])/np.sum(w[:6])
    prof15[-1] = np.sum(w[-6:]*prof25[-6:])/np.sum(w[-6:])
    return prof15 

def profile_rbinedges(r_arr):
    rbinedges = np.zeros(len(r_arr)+1)
    rbinedges[2:-2] = np.sqrt(r_arr[1:-2]*r_arr[2:-1])
    rbinedges[-2] = rbinedges[-3]**2/rbinedges[-4]
    rbinedges[1] = rbinedges[2]**2/rbinedges[3]
    rbinedges[-1] = rbinedges[-2]*(rbinedges[3]/rbinedges[2])**6
    return rbinedges

def get_virial_radius(z_arr, Mh_arr, units='arcsec'):
    '''
    Given a list of halo at redshift z_arr and halo mass Mh_arr,
    calculate their virial radius
    
    Inputs:
    =======
    z_arr: z of the halos
    Mh_arr: halo mass of the halos [M_sun]
    units: 'arcsec' or 'Mpc'
    
    Outputs:
    ========
    rvir_arr: virial radius of the desired units
    '''
    rhoc_arr = np.array(cosmo.critical_density(z_arr).to(u.M_sun / u.Mpc**3))
    rvir_arr = ((3 * Mh_arr) / (4 * np.pi * 200 * rhoc_arr))**(1./3)
    if units == 'Mpc':
        return rvir_arr
    DA_arr = np.array(cosmo.comoving_distance(z_arr))
    rvir_arr = (rvir_arr / DA_arr) * u.rad.to(u.arcsec)
    return rvir_arr

class gal_profile_model:
    from astropy import units as u

    def __init__(self):
        pass
    
    def sersic(self, x_arr, Ie, n, xe):
        try:
            from scipy.special import gammaincinv
        except ValueError:
            raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn =  gammaincinv(2. * n, 0.5)
        
        I_arr = (10**Ie) * np.exp(-bn*((x_arr/xe)**(1/n)-1))
        return I_arr
    
    def Wang19_params(self, im, extendedness=False):
        '''
        MNRAS 487, 1580 
        table 3
        '''

        # from MICECAT
        R200_arr = [101.99, 70.55, 48.21, 34.70] #[arcsec]
        R200_err = [23.97, 22.78, 21.36, 20.27] #[arcsec]
        R200_Mpc_arr = [0.442, 0.346, 0.295, 0.260] #[Mpc]
        R200_Mpc_err = [0.126, 0.133, 0.136, 0.135] #[Mpc]
        # from SIDES abundance matching
        # R200_arr = [98.90,62.83,42.48,29.34] #[arcsec]
        W19params = {}
        W19params['m_min'] = im + 16
        W19params['m_max'] = im + 17
        W19params['R200'] = R200_arr[im]
        W19params['R200_err'] = R200_err[im]
        W19params['R200_unit'] = self.u.arcsec
        W19params['R200_Mpc'] = R200_Mpc_arr[im]
        W19params['R200_Mpc_err'] = R200_Mpc_err[im]
        W19params['sersic_params_def'] = '[Ie, n, xe]'
        
        if not extendedness:
            W19params['extendedness'] = False
            W19params['sersic1'] = [-8.471,1.5320,0.0056]
            W19params['sersic2'] = [-8.9330,2.6190,0.0165]
            W19params['sersic1_err'] = [0.0741,0.2115,0.0002]
            W19params['sersic2_err'] = [0.0423,0.0804,0.0008]
        else:
            W19params['extendedness'] = True
            W19params['sersic1'] = [-7.3500,0.0101,0.0015]
            W19params['sersic2'] = [-8.7930,1.143,0.0231]
            W19params['sersic3'] = [-9.9220,3.691,0.0001]
            W19params['sersic1_err'] = [0.0577,0.0707,0.0012]
            W19params['sersic2_err'] = [0.0153,0.0372,0.0012]
            W19params['sersic3_err'] = [0.1447,0.1859,0.0019]
        return W19params
        
    def Wang19_profile(self, r_arr, im, extendedness=False, **kwargs):
        '''
        Given r_arr (any dimension) in unit arcsec, return the gal profile.
        '''
        params = self.Wang19_params(im, extendedness)
        
        if len(kwargs)==0:
            params['use_Wang19_default'] = True
        else:
            params['use_Wang19_default'] = False
            
        for key, value in kwargs.items():
            if key == 'R200':
                params['R200'] = value
                
            if key == 'Ie1':
                params['sersic1'][0] = value
            elif key == 'n1':
                params['sersic1'][1] = value
            elif key == 'xe1':
                params['sersic1'][2] = value
            elif key == 'Ie2':
                params['sersic2'][0] = value
            elif key == 'n2':
                params['sersic2'][1] = value
            elif key == 'xe2':
                params['sersic2'][2] = value
            elif key == 'Re1':
                params['sersic1'][2] = value / params['R200']
            elif key == 'Re2':
                params['sersic2'][2] = value / params['R200']
                
            
            if extendedness:
                if key == 'Ie3':
                    params['sersic3'][0] = value
                elif key == 'n3':
                    params['sersic3'][1] = value
                elif key == 'xe3':
                    params['sersic3'][2] = value
                elif key == 'Re3':
                    params['sersic3'][2] = value / params['R200']
        
        I1_arr = self.sersic(r_arr/params['R200'], params['sersic1'][0],params['sersic1'][1],
                             params['sersic1'][2])
        
        I2_arr = self.sersic(r_arr/params['R200'], params['sersic2'][0], params['sersic2'][1],
                             params['sersic1'][2] + params['sersic2'][2])
        
        profdat={}
        profdat['params'] = params
        profdat['I1_arr'] = I1_arr
        profdat['I2_arr'] = I2_arr
        profdat['I_arr'] = I1_arr + I2_arr
        
        if extendedness:
            I3_arr = self.sersic(r_arr/params['R200'], params['sersic3'][0], params['sersic3'][1],
                                 params['sersic1'][2] + params['sersic2'][2] + params['sersic3'][2])
            profdat['I3_arr'] = I3_arr
            profdat['I_arr'] = I1_arr + I2_arr + I3_arr

        
        return profdat
        
def gal_num_counts(mag, band):
    
    if band == 'Helgason125':
        helgfname = '/Users/ytcheng/ciber/doc/20170904_External/helgason/Helgason125.txt'
        data_helgason = np.loadtxt(helgfname, delimiter=',')
        dN_dm_ddeg2 = 10**(np.interp(mag,data_helgason[:,0], np.log10(data_helgason[:,1])))
    elif band == 'Helgason163':
        helgfname = '/Users/ytcheng/ciber/doc/20170904_External/helgason/Helgason163.txt'
        data_helgason = np.loadtxt(helgfname, delimiter=',')
        dN_dm_ddeg2 = 10**(np.interp(mag,data_helgason[:,0], np.log10(data_helgason[:,1])))
    elif band == 'Y':
        mags = np.array([16.5, 17.5, 18.5, 19.5])
        counts = np.array([96, 300, 960, 2300])
        dN_dm_ddeg2 = 10**(np.interp(mag,mags, np.log10(counts)))
        
    return dN_dm_ddeg2

def get_catalog(inst, ifield, im, src_type='g', return_cols=None):
    
    catcoorddir = mypaths['PScatdat']
    field = fieldnamedict[ifield]
    fname=catcoorddir+ field + '.csv'
    df = pd.read_csv(fname)

    m_min = magbindict['m_min'][im]
    m_max = magbindict['m_max'][im]

    dfi = df.loc[(df['sdssClass']==3) \
                & (df['x'+str(inst)]>-0.5) & (df['x'+str(inst)]<1023.5)\
                & (df['y'+str(inst)]>-0.5) & (df['y'+str(inst)]<1023.5)\
                & (df['x'+str(inst)]>-0.5) & (df['x'+str(inst)]<1023.5)\
                & (df['I_comb']>=m_min) & (df['I_comb']<m_max)]
    Nall = len(dfi)
    dfi = dfi.loc[dfi['Photz']>0]
    f = len(dfi)/Nall
    z = np.array(dfi['Photz'])
    ra = np.array(dfi['ra'])
    dec = np.array(dfi['dec'])
    x = np.array(dfi['x1'])
    y = np.array(dfi['y1'])
    
    cat_data = {'f' : f,
               'z' : z,
               'ra' : ra,
               'dec' : dec,
               'x' : x,
               'y' : y
               }

    return cat_data


def load_processed_images(data_maps, 
    return_names=[(1,4,'cbmap'), (1,4,'psmap')],
    rotate_TM2=False):
    '''
    get the images processed from image_reduction
    
    valid map_names:
    'name': name of the field
    'cf_G1': G1 cal factor, (e-/s) / (ADU/fr)
    'cf_G2': G1 cal factor, (nW/m2/sr) / (e-/s)
    'cf': cal factor G1G2, (nW/m2/sr) / (ADU/fr)
    'Nfr': number of frame used in this field
    'rawmap': line fit w/o linearization [ADU/fr]
    'rawmask': DC inst mask * negative pixel mask * ts mask
    'DCmap': DC map [ADU/fr]
    'DCsubmap': linearized map - DC template [ADU/fr]
    'FFpix': FF from stacking off fields. It takes NaNs values at pix w/o FF info
    'FFsm': FFpix smoothed with 3pix Gaussian kernel to fill in NaNs
    'FF': final FF estimator. FFpix with NaNs filled with FFsm 
    'mask_inst': final instrument mask. rawmask * crmask * sigma clip mask
    'strmask': source mask (m < 20)
    'strnum': source num
    'map': final image after FF corr [ADU/fr]
    'srcmap': source map from PanSTARRS & 2MASS bright sources [nW/m2/sr]
    'cbmap': mean sub 'map' with mask_inst * strmask [nW/m2/sr]
    'psmap': mean sub 'srcmap' with mask_inst * strmask [nW/m2/sr]
    'mean_cb': mean intensity in CIBER image after masking with strmask and mask_inst [nW/m2/sr]
    'mean_ps': mean intensity in srcmap image after masking with strmask and mask_inst [nW/m2/sr]
    'cbgradmap': a 2D gradient (1st order polynominal) fit to CIBER image
    'kelsall': kelsall ZL template [nW/m2/sr]
    
    Input:
    =======
    data_maps: data dict obtained by --
        from reduction import *
        data_maps = {1: image_reduction(1), 2: image_reduction(2)}

    return_names: list of items (inst, ifield, map_name)
    rotate_TM2: if True, rotate TM2 90 deg 
        s.t. it's aligned with TM1 for visualization

    Ouput:
    =======
    return_maps: list of map of the input return_names
    
    '''

    return_maps = []
    for inst,ifield,name in return_names:
        if name == 'name':
            mapi = fieldnamedict[ifield]
        if name == 'cf':
            mapi = cal_factor_dict['apf2nWpm2psr'][inst][ifield]
        elif name =='cf_G1':
            mapi = cal_factor_dict['apf2eps'][inst]
        elif name == 'cf_G2':
            G1G2 = cal_factor_dict['apf2nWpm2psr'][inst][ifield]
            G1 = cal_factor_dict['apf2eps'][inst]
            mapi =  G1G2 / G1
        elif name == 'Nfr':
            mapi = data_maps[inst].stackmapdat[ifield]['Nfr']
        elif name == 'cbmap':
            cf = cal_factor_dict['apf2nWpm2psr'][inst][ifield]
            mapi = data_maps[inst].stackmapdat[ifield]['map'].copy()
            mask_inst = data_maps[inst].stackmapdat[ifield]['mask_inst'].copy()
            strmask = data_maps[inst].stackmapdat[ifield]['strmask'].copy()
            mapi = cf * (mapi - np.mean(mapi[mask_inst*strmask==1]))
            
        elif name == 'psmap':
            mapi = data_maps[inst].stackmapdat[ifield]['srcmap'].copy()
            mask_inst = data_maps[inst].stackmapdat[ifield]['mask_inst'].copy()
            strmask = data_maps[inst].stackmapdat[ifield]['strmask'].copy()
            mapi = mapi - np.mean(mapi[mask_inst*strmask==1])

        elif name == 'DCmap':
            mapi = data_maps[inst].DCtemplate
            
        elif name == 'mean_cb':
            cf = cal_factor_dict['apf2nWpm2psr'][inst][ifield]
            mapi = data_maps[inst].stackmapdat[ifield]['map'].copy()
            mask_inst = data_maps[inst].stackmapdat[ifield]['mask_inst'].copy()
            strmask = data_maps[inst].stackmapdat[ifield]['strmask'].copy()
            mapi = cf * np.mean(mapi[mask_inst*strmask==1])

        elif name == 'mean_ps':
            mapi = data_maps[inst].stackmapdat[ifield]['srcmap'].copy()
            mask_inst = data_maps[inst].stackmapdat[ifield]['mask_inst'].copy()
            strmask = data_maps[inst].stackmapdat[ifield]['strmask'].copy()
            mapi = np.mean(mapi[mask_inst*strmask==1])

        elif name == 'cbgradmap':
            cf = cal_factor_dict['apf2nWpm2psr'][inst][ifield]
            mapi = data_maps[inst].stackmapdat[ifield]['map'].copy()
            mask_inst = data_maps[inst].stackmapdat[ifield]['mask_inst'].copy()
            strmask = data_maps[inst].stackmapdat[ifield]['strmask'].copy()
            _, grad = image_poly_filter(cf*mapi, strmask*mask_inst, degree=1, return_bg=True)
            mapi = grad

        elif name == 'kelsall':
            datadir = mypaths['ciberdir'] + 'data/'
            fname = datadir + 'kelsall/TM' + str(inst) + '/' \
            + fieldnamedict[ifield] + '_band' + str(inst) + '.fits'
            with fits.open(fname) as hdul:
                mapi = hdul[0].data

        else:
            mapi = data_maps[inst].stackmapdat[ifield][name].copy() 
            
        if rotate_TM2 and inst==2:
            mapi = np.rot90(mapi, k=3)

        return_maps.append(mapi)

    return return_maps

def pix_func_substack(dx = 50, Nsub = 10):
    xx,yy = np.meshgrid(np.arange(2 * dx + 1), np.arange(2 * dx + 1))
    xx, yy = abs(xx - dx), abs(yy - dx)
    psf_pix = (Nsub - xx)*(Nsub - yy)
    psf_pix[(xx >= Nsub)] = 0
    psf_pix[(yy >= Nsub)] = 0
    psf_pix = psf_pix / np.sum(psf_pix)
    return psf_pix

def image_poly_filter(image, mask=None, degree=2, return_bg=False):
    '''
    polynominal filter the image
    '''
    if degree is None:
        return image
    
    import warnings
    from astropy.modeling import models, fitting
    
    if mask is None:
        mask = np.ones(image.shape)
        mask[image==0] = 0  
    sp = np.where(mask!=0)

    Nx, Ny = image.shape
    x,y = np.meshgrid(np.arange(Nx), np.arange(Ny))

    p_init = models.Polynomial2D(degree=degree)
    fit_p = fitting.LevMarLSQFitter()

    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        p = fit_p(p_init, x[sp], y[sp], image[sp])
    
    bgmap = p(x, y)
    image_filt = image - bgmap
    
    if return_bg:
        return image_filt, bgmap

    return image_filt

def image_smooth_gauss(image, mask=None, stddev=5, return_unmasked=False):
    '''
    Gaussian smooth the image
    '''
    
    if mask is None:
        mask = np.ones(image.shape)
        mask[image==0] = 0  

    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve

    kernel = Gaussian2DKernel(stddev)
    im = image.copy()
    im[mask==0] = np.nan
    
    im_conv = convolve(im, kernel)

    if return_unmasked:
        return im_conv
        
    im_conv[mask==0] = 0
    
    return im_conv

def normalize_cov(cov):
    cov_rho = np.zeros_like(cov)
    for i in range(cov_rho.shape[0]):
        for j in range(cov_rho.shape[0]):
            if cov[i,i]==0 or cov[j,j]==0:
                cov_rho[i,j] = cov[i,j]
            else:
                cov_rho[i,j] = cov[i,j] / np.sqrt(cov[i,i]*cov[j,j])
    return cov_rho

def profile_cov_normalize(prof, cov, hit, Nsubbin_head=6, Nsubbin_tail=6):
    p = prof / prof[0]
    c = np.zeros_like(cov)
    
    for i in range(c.shape[0]):
        for j in range(c.shape[0]):
            c[i][j] = prof[i]*prof[j]/prof[0]**2*(cov[i][j]/prof[i]/prof[j] \
                                                  - cov[i][0]/prof[i]/prof[0] \
                                        - cov[j][0]/prof[j]/prof[0] + cov[0][0]/prof[0]**2)
    c[0][0] = 0. # sometimes numerical issue make c[0][0] slightly != 0
    Nsub = len(p) - Nsubbin_head - Nsubbin_tail + 2
    psub = np.zeros([Nsub])
    csub = np.zeros([Nsub,Nsub])
    
    pin = p[:Nsubbin_head]
    hitin = hit[:Nsubbin_head]
    win = hitin / np.sum(hitin)
    pout = p[-Nsubbin_tail:]
    hitout = hit[-Nsubbin_tail:]
    wout = hitout / np.sum(hitout)
    
    psub[0] = np.sum(pin * win)
    psub[-1] = np.sum(pout * wout)
    psub[1:-1] = p[Nsubbin_head:-Nsubbin_tail]
    
    w = np.ones_like(p)
    w[:Nsubbin_head] = win
    w[-Nsubbin_tail:] = wout
    wmat = w.reshape(-1,1)@w.reshape(1,-1)
    cw = c * wmat
    csub[0,0] = np.sum(cw[:Nsubbin_head,:Nsubbin_head])
    csub[-1,-1] = np.sum(cw[-Nsubbin_tail:,-Nsubbin_tail:])
    csub[0,-1] = np.sum(cw[:Nsubbin_head,-Nsubbin_tail:])
    csub[-1,0] = csub[0,-1]
    csub[0,1:-1]= np.sum(cw[:Nsubbin_head,Nsubbin_head:-Nsubbin_tail],axis=0)
    csub[-1,1:-1]= np.sum(cw[-Nsubbin_tail:,Nsubbin_head:-Nsubbin_tail],axis=0)
    csub[1:-1,0]= csub[0,1:-1]
    csub[1:-1,-1]= csub[-1,1:-1]
    csub[1:-1,1:-1] = cw[Nsubbin_head:-Nsubbin_tail,Nsubbin_head:-Nsubbin_tail]
    
    return p, c, psub, csub

def plot_atcr(listsamp, title=None, plot=False):
    numbsamp = listsamp.shape[0]
    four = scipy.fftpack.fft(listsamp - np.mean(listsamp, axis=0), axis=0)
    atcr = scipy.fftpack.ifft(four * np.conjugate(four), axis=0).real
    atcr /= np.amax(atcr, 0)
    autocorr = atcr[:int(numbsamp/2), ...]
    indexatcr = np.where(autocorr > 0.2)
    timeatcr = np.argmax(indexatcr[0], axis=0)
    numbsampatcr = autocorr.size
    if plot:
        figr, axis = plt.subplots(figsize=(6,4))
        plt.title(title, fontsize=16)
        axis.plot(np.arange(numbsampatcr), autocorr)
        axis.set_xlabel(r'$\tau$', fontsize=16)
        axis.set_ylabel(r'$\xi(\tau)$', fontsize=16)
        axis.text(0.8, 0.8, r'$\tau_{exp} = %.3g$' % timeatcr, ha='center',
                  va='center', transform=axis.transAxes, fontsize=16)
        axis.axhline(0., ls='--', alpha=0.5)
        plt.tight_layout()
    return np.arange(numbsampatcr), autocorr

def catalog_add_xy_from_radec(field, df):
    order = [c for c in df.columns]
    # find the x, y solution with all quad
    for inst in [1,2]:
        hdrdir = mypaths['ciberdir'] + 'doc/20170617_Stacking/maps/astroutputs/inst' + str(inst) + '/'
        xoff = [0,0,512,512]
        yoff = [0,512,0,512]
        for iquad,quad in enumerate(['A','B','C','D']):
            hdulist = fits.open(hdrdir + field + '_' + quad + '_astr.fits')
            wcs_hdr=wcs.WCS(hdulist[('primary',1)].header, hdulist)
            hdulist.close()
            src_coord = SkyCoord(ra=df['ra']*u.degree, dec=df['dec']*u.degree, frame='icrs')

            x_arr, y_arr = wcs_hdr.all_world2pix(df['ra'],df['dec'],0)
            df['x' + quad] = x_arr + xoff[iquad]
            df['y' + quad] = y_arr + yoff[iquad]

        df['meanx'] = (df['xA'] + df['xB'] + df['xC'] + df['xD']) / 4
        df['meany'] = (df['yA'] + df['yB'] + df['yC'] + df['yD']) / 4

        # assign the x, y with the nearest quad solution
        df['x'+str(inst)] = df['xA'].copy()
        df['y'+str(inst)] = df['yA'].copy()
        bound = 511.5
        df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'x'+str(inst)] = df['xB']
        df.loc[ (df['meanx'] < bound) & (df['meany'] > bound),'y'+str(inst)] = df['yB']
        
        df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'x'+str(inst)] = df['xC']
        df.loc[ (df['meanx'] > bound) & (df['meany'] < bound),'y'+str(inst)] = df['yC']

        df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'x'+str(inst)] = df['xD']
        df.loc[ (df['meanx'] > bound) & (df['meany'] > bound),'y'+str(inst)] = df['yD']

    # write x, y to df
    order = order[:2] + ['x1','y1','x2','y2'] + order[2:]
    dfout = df[order].copy()
    
    return dfout

def match_catalog_by_coord(df1, df2, match_dist=0.7, return_full=False):
    '''
    Findh the df2 counter part of all the df1 sources that matches within
    match_dist[arcsec]
    '''
    catalog1 = (df1[['ra','dec']].values * np.pi/180).tolist()
    ps1 = [[item[0], item[1]] for item in catalog1]
    catalog2 = (df2[['ra','dec']].values * np.pi/180).tolist()
    ps2 = [[item[0], item[1]] for item in catalog2]
    kdt = cKDTree(ps2)
    obj = kdt.query_ball_point(ps1, (match_dist * u.arcsec).to(u.rad).value)
    Nmatch = np.array([len(obj_i) for obj_i in obj])
    idxmatch2 = np.array([obj_i[0] for obj_i in obj if len(obj_i)>0])
    idxmatch1 = np.where(Nmatch>0)[0]
    
    match_dict = {'match_dist':match_dist,'match_obj':obj,'Nmatch':Nmatch,
                 'idxmatch1':idxmatch1, 'idxmatch2':idxmatch2}
    
    if return_full:
        return df1.iloc[idxmatch1], df2.iloc[idxmatch2], match_dict
    return df1.iloc[idxmatch1], df2.iloc[idxmatch2]


# Gonzalez+07 table 1 https://iopscience.iop.org/article/10.1086/519729/pdf
# Gonzalez+05 table 4 https://iopscience.iop.org/article/10.1086/425896/pdf
Gonzalez_dict = {
#      Gonzelez+07 (f_BCG+ICL=L_BCGICL/(L_BCGICL+L_gal))
#     ['z','M500','M_BCG+ICL','M_BCG+ICL_err','Mgal','f_BCG+ICL','f_BCG+ICL_err',
#     Gonzelez+05
#     'Mtot','Mtot_ep','Mtot_em','Mext','Mext_e_stat','Mext_ep_sys','Mext_em_sys',
#     'Mcor','Mcor_e_stat','Mcor_ep_sys','Mcor_em_sys'],
'A0122':[0.1134,14.31,-25.60,0.04,-26.11,0.38,0.04,-25.66,0.06,0.06,-25.56,0.01,0.06,0.06,-22.97,0.03,0.05,0.05],
'A1651':[0.0847,14.87,-25.63,0.10,-26.88,0.24,0.04,-25.84,0.17,0.27,-24.95,0.07,0.48,0.53,-25.21,0.01,0.00,0.02],
'A2400':[0.0880,14.26,-25.08,0.06,-26.10,0.28,0.06,-25.17,0.18,0.32,-24.61,0.04,0.19,0.44,-24.19,0.08,0.17,0.11],
'A2571':[0.1091,14.30,-25.37,0.10,-25.53,0.46,0.10,-25.42,0.15,0.19,-25.11,0.01,0.17,0.22,-23.89,0.02,0.08,0.09],
'A2721':[0.1144,14.63,-25.19,0.02,-26.90,0.17,0.02,-25.20,0.02,0.02,-25.02,0.03,0.04,0.04,-23.14,0.14,0.08,0.08],
'A2730':[0.1201,14.91,-26.04,0.11,-26.49,0.40,0.11,-26.14,0.24,0.30,-26.00,0.03,0.25,0.32,-23.85,0.03,0.19,0.16],
'A2811':[0.1079,14.66,-25.64,0.08,-26.40,0.33,0.08,-25.70,0.22,0.22,-25.53,0.03,0.22,0.23,-23.59,0.03,0.22,0.15],
'A2955':[0.0943,13.19,-25.16,0.06,-24.96,0.55,0.06,-25.31,0.08,0.08,-25.22,0.01,0.08,0.08,-22.61,0.03,0.09,0.08],
'A2969':[0.1259,14.85,-25.58,0.08,-26.62,0.28,0.08,-25.58,0.26,0.44,-25.35,0.03,0.27,0.49,-23.80,0.08,0.21,0.18],
'A2984':[0.1042,13.84,-25.63,0.05,-25.51,0.53,0.05,-25.87,0.01,0.09,-25.79,0.01,0.01,0.09,-23.04,0.02,0.01,0.04],
'A3112':[0.0753,14.79,-25.76,0.03,-26.96,0.25,0.03,-25.82,0.07,0.07,-25.74,0.01,0.06,0.06,-22.95,0.04,0.16,0.17],
'A3166':[0.1174,12.56,-24.55,0.06,-23.99,0.63,0.06,-24.73,0.07,0.08,-24.65,0.02,0.07,0.08,-21.92,0.04,0.08,0.08],
'A3705':[0.0895,14.89,-24.37,0.02,-26.53,0.12,0.02,-24.38,0.04,0.08,-24.27,0.01,0.03,0.00,-21.85,0.07,0.18,0.62],
'A3727':[0.1159,14.09,-24.97,0.06,-25.83,0.31,0.06,-25.14,0.17,0.19,-24.97,0.05,0.19,0.21,-23.07,0.03,0.05,0.04],
'A3809':[0.0623,13.99,-24.69,0.04,-25.38,0.35,0.04,-24.76,0.07,0.08,-23.85,0.03,0.13,0.16,-24.14,0.03,0.02,0.02],
'A3920':[0.1268,13.63,-25.06,0.04,-25.27,0.45,0.04,-25.08,0.02,0.04,-24.80,0.01,0.01,0.04,-23.48,0.01,0.07,0.05],
'A4010':[0.0955,14.21,-25.56,0.08,-26.25,0.35,0.08,-25.71,0.21,0.28,-25.56,0.02,0.19,0.28,-23.52,0.06,0.37,0.28],
'AS0084':[0.1080,13.93,-25.38,0.04,-26.16,0.33,0.04,-25.38,0.03,0.05,-25.33,0.01,0.03,0.05,-21.94,0.04,0.08,0.10],
'AS0296':[0.0696,13.80,-25.19,0.04,-25.17,0.51,0.04,-25.05,0.03,0.07,-24.82,0.04,0.16,0.02,-23.25,0.22,0.26,0.72]
                }

# Burke+05 https://academic.oup.com/mnras/article/449/3/2353/1123740
# ICL: light below surface brightness 25 mag/arcsec^2
Burke_dict = {
    'z':[0.187,0.296,0.213,0.224,0.234,0.288,0.313,0.345,0.348,0.352,0.391,0.396,0.399],
'f_ICL':np.array([23.05,17.11,17.97,16.64,12.89,13.02,7.58,7.44,6.44,6.11,3.16,2.69,2.36])*1e-2,
'f_ICL_err':np.array([0.73,0.77,0.73,0.78,1.11,0.23,0.65,0.17,0.10,0.29,0.06,0.10,0.13])*1e-2,   
'M200':np.array([1.04,1.17,1.20,1.76,0.73,1.03,1.26,0.64,1.40,1.13,0.88,2.50,0.96])*1e15,   
'M200_err':np.array([0.07,0.07,0.59,0.18,0.18,0.07,0.06,0.09,0.12,0.10,0.08,0.50,0.14])*1e15,   
}