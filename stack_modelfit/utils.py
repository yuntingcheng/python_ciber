import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat
import pickle
from ciber_info import *
from utils_plotting import *
from IPython.display import clear_output
from astropy import units as u
from astropy import cosmology
cosmo = cosmology.Planck15


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

def radial_prof(mapin, cenx, ceny, log=True, nbins=25, maskin=None,
                weight=None, rbinedges=None, return_full=True):
    
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
    
    rbins = np.zeros(nbins)
    prof = np.zeros(nbins)
    err = np.zeros(nbins)
    Npix = np.zeros(nbins)
    
    for i in range(nbins):
        sp = np.where((radmap < rbinedges[i+1]) & (radmap >= rbinedges[i]) &\
                      (maskin != 0) & (~np.isnan(mapin)) & (~np.isnan(weight)))
        Npixi = np.sum(sp)
        
        if Npixi==0:
            rbins[i] = np.mean(rbinedges[i:i+2])
            prof[i] = 0
            err[i] = 0
            Npix[i] = 0
            continue
            
        mapi = mapin[sp]
        wi = weight[sp]
        wi = wi/np.nansum(wi)
        ri = radmap[sp]
        rbins[i] = np.nansum(ri*wi) / np.nansum(wi)
        prof[i] = np.nansum(mapi*wi) / np.nansum(wi)
        err[i] = np.nanstd(mapi) / np.sqrt(Npixi)
        Npix[i] = Npixi
        
    if not return_full:
        return prof
    else:
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
    if units is 'Mpc':
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
    
    def Wang19_params(self, im, extendedness=True):
        '''
        MNRAS 487, 1580 
        table 3
        '''

        # from SIDES abundance matching
        R200_arr = [98.90,62.83,42.48,29.34] #[arcsec]
        W19params = {}
        W19params['m_min'] = im + 16
        W19params['m_max'] = im + 17
        W19params['R200'] = R200_arr[im]
        W19params['R200_unit'] = self.u.arcsec
        W19params['sersic_params_def'] = '[Ie, n, xe]'
        
        if extendedness:
            W19params['extendedness'] = True
            W19params['sersic1'] = [-8.471,1.5320,0.0056]
            W19params['sersic2'] = [-8.9330,2.6190,0.0165]
        else:
            W19params['extendedness'] = False
            W19params['sersic1'] = [-7.3500,0.0101,0.0015]
            W19params['sersic2'] = [-8.7930,1.143,0.0231]
            W19params['sersic3'] = [-9.9220,3.691,0.0001]
        return W19params
        
    def Wang19_profile(self, r_arr, im, extendedness=True, **kwargs):
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
                
            
            if not extendedness:
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
        
        if not extendedness:
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

def load_processed_images(return_names=[(1,4,'cbmap'), (1,4,'psmap')]):
    '''
    get the images processed by stack_preprocess.m
    
    Input:
    =======
    return_names: list of items (inst, ifield, map name)
    
    Ouput:
    =======
    return_maps: list of map of the input return_names
    
    '''
    img_names = {'rawmap':0, 'rawmask':1, 'DCsubmap':2, 'FF':3, 'FFunholy':4,
                'map':5, 'cbmap':6, 'psmap':7, 'mask_inst':8, 'strmask':9, 'strnum':10}
    data = {}
    data[1] = loadmat(mypaths['alldat'] + 'TM' + str(1) + '/stackmapdatarr.mat')['data']
    data[2] = loadmat(mypaths['alldat'] + 'TM' + str(2) + '/stackmapdatarr.mat')['data']
    
    return_maps = []
    for inst,ifield,name in return_names:
        mapi = data[inst][ifield-4][img_names[name]]
        return_maps.append(mapi)
    return return_maps

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

def image_smooth_gauss(image, mask=None, stddev=5):
    '''
    Gaussian smooth the image
    '''
    
    if mask is None:
        mask = np.ones(image.shape)
        mask[image==0] = 0  

    from astropy.convolution import Gaussian2DKernel
    from astropy.convolution import convolve

    kernel = Gaussian2DKernel(stddev=5)
    im = image.copy()
    im[mask==0] = np.nan
    
    im_conv = convolve(im, kernel)
    im_conv[mask==0] = 0
    
    return im_conv
