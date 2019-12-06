import numpy as np
from utils import *
from power_spec import *


def MZ14_mask(inst, xs, ys, ms, verbose=True):    
    
    if inst==1:
        ms_vega = np.array(ms) + 2.5*np.log10(1594./3631.)
    else:
        ms_vega = np.array(ms) + 2.5*np.log10(1024./3631.)

    alpha = -6.25
    beta = 110
    rs = alpha * ms_vega + beta # arcsec
    
    m_max_vega = 17
    sp = np.where(ms_vega < m_max_vega)
    xs = xs[sp]
    ys = ys[sp]
    rs = rs[sp]
    
    mask = np.ones([1024,1024])
    for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
        if verbose and i%(len(xs)//20)==0 and len(xs)>20:
            print('run MZ14_mask %d / %d (%.1f %%)'\
              %(i, len(xs), i/len(xs)*100))
        radmap = make_radius_map(mask, x, y)
        mask[radmap < r/7.] = 0
    return mask

class mask_Mkk:
    def __init__(self, mask, **kwargs):
        
        self.mask = mask
        
        lbins, Cl, Clerr, Nmodes, lbinedges, l2d, ps2d \
        = get_power_spec(mask, return_full=True, **kwargs)
        
        self.lbins = lbins
        self.lbinedges = lbinedges
        
        if 'pixsize' in kwargs:
            self.pixsize = kwargs['pixsize']
        else:
            self.pixsize = 7
        
    def get_Mkk_sim(self, Nsims=100, verbose=True):
        
        mask = self.mask
        lbins = self.lbins
        lbinedges = self.lbinedges
        pixsize = self.pixsize
        
        Nbins = len(lbins)
        Mkk = np.zeros([Nbins, Nbins])

        for ibin in range(Nbins):

            if verbose:
                print('run Mkk sim for %d/%d ell bin'%(ibin, Nbins))
            Clin = np.zeros(Nbins)
            Clin[ibin] = 1

            Clouts = 0
            for isim in range(Nsims):
                mapi = map_from_Cl(lbins, lbinedges, Clin, pixsize=pixsize)
                _, Clout, _ = get_power_spec(mapi)
                
                mapi /= np.sqrt(Clout[ibin])
                _, Clout, _ = get_power_spec(mapi, mask=mask, pixsize=pixsize)
                Clouts += Clout
                
            Clouts /= Nsims
            Mkk[:,ibin] = Clouts

        invMkk = np.linalg.inv(Mkk)        
        
        self.Mkk = Mkk
        self.invMkk = invMkk
        self.NsimsMkk = Nsims
    
    def Mkk_correction(self, Cl, Clerr=None):
        if Clerr is None:
            return self.invMkk@Cl
        else:
            return self.invMkk@Cl, self.invMkk@Clerr