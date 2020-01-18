from scipy.io import loadmat
from utils import *
from power_spec import *

def get_mask_radius_th(ifield, m_arr, inst=1, Ith=0.5):
    '''
    r_arr: arcsec
    '''
    m_arr = np.array(m_arr)
    fitpsfdat=loadmat(mypaths['ciberdir'] + 'doc/20170617_Stacking/psf_analytic/TM'\
              + str(inst) + '/fitpsfdat.mat')['fitpsfdat'][0][ifield-1][7][0][0]
    beta, rc, norm  = float(fitpsfdat[0]), float(fitpsfdat[1]), float(fitpsfdat[4])
    
    Nlarge = 100
    radmap = make_radius_map(np.zeros([2*Nlarge+1, 2*Nlarge+1]), Nlarge, Nlarge)*0.7
    Imap_large = norm * (1 + (radmap/rc)**2)**(-3*beta/2)
    
    lambdaeff = band_info(inst).wl
    sr = ((7./3600.0)*(np.pi/180.0))**2
    I_arr=3631*10**(-m_arr/2.5)*(3/lambdaeff)*1e6/(sr*1e9)
    r_arr = np.zeros_like(m_arr, dtype=float)
    for i, I in enumerate(I_arr):
        sp = np.where(Imap_large*I > (Ith/100))
        if len(sp[0])>0:
            r_arr[i] = np.max(radmap[sp])
    
    return r_arr

def MZ14_mask(inst, xs, ys, ms, return_radius=False, verbose=True):    
    
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
    
    if return_radius:
        return rs
    
    mask = np.ones([1024,1024])
    num = np.zeros([1024,1024])
    for i,(x,y,r) in enumerate(zip(xs, ys, rs)):
        if len(xs)>20:
            if verbose and i%(len(xs)//20)==0:
                print('run MZ14_mask %d / %d (%.1f %%)'\
                  %(i, len(xs), i/len(xs)*100))
        radmap = make_radius_map(mask, x, y)
        mask[radmap < r/7.] = 0
        num[radmap < r/7.] += 1
    return mask, num

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