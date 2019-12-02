from ciber_info import *
from utils_plotting import *
import numpy as np

def make_radius_map(mapin, cenx, ceny):
    '''
    return radmap of size mapin.shape. 
    radmap[i,j] = distance between (i,j) and (cenx, ceny)
    '''
    Nx, Ny = mapin.shape
    xx, yy = np.meshgrid(np.arange(Nx), np.arange(Ny), indexing='ij')
    radmap = np.sqrt((xx - cenx)**2 + (yy - ceny)**2)
    return radmap


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
        sp = [(radmap < rbinedges[i+1]) & (radmap >= rbinedges[i]) & (maskin != 0) & 
              (~np.isnan(mapin)) & (~np.isnan(weight))]
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
        R200_arr = [98.90,62.83,42.48,29.34]; # [arcsec]
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
        
