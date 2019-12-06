import numpy as np

def get_power_spectrum_2d(map_a, map_b=None, pixsize=7.):
    '''
    calculate 2d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    pixsize:[arcsec]
    
    Outputs:
    ========
    l2d: corresponding ell modes
    ps2d: 2D Cl
    '''
    
    if map_b is None:
        map_b = map_a.copy()
        
    dimx, dimy = map_a.shape
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix
    
    ffta = np.fft.fftn(map_a*sterad_per_pix)
    fftb = np.fft.fftn(map_b*sterad_per_pix)
    ps2d = np.real(ffta * np.conj(fftb)) / V 
    ps2d = np.fft.ifftshift(ps2d)
    
    lx = np.fft.fftfreq(dimx)*2
    ly = np.fft.fftfreq(dimy)*2
    lx = np.fft.ifftshift(lx)*(180*3600./pixsize)
    ly = np.fft.ifftshift(ly)*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    return l2d, ps2d


def get_power_spec(map_a, map_b=None, mask=None, pixsize=7., 
                   lbinedges=None, lbins=None, nbins=29, 
                   logbin=True, weights=None, return_full=False, return_Dl=False):
    '''
    calculate 1d cross power spectrum Cl
    
    Inputs:
    =======
    map_a: 
    map_b:map_b=map_a if no input, i.e. map_a auto
    mask: common mask for both map
    pixsize:[arcsec]
    lbinedges: predefined lbinedges
    lbins: predefined lbinedges
    nbins: number of ell bins
    logbin: use log or linear ell bin
    weights: Fourier weight
    return_full: return full output or not
    return_Dl: return Dl=Cl*l*(l+1)/2pi or Cl
    
    Outputs:
    ========
    lbins: 1d ell bins
    ps2d: 2D Cl
    Clerr: Cl error, calculate from std(Cl2d(bins))/sqrt(Nmode)
    Nmodes: # of ell modes per ell bin
    lbinedges: 1d ell binedges
    l2d: 2D ell modes
    ps2d: 2D Cl before radial binning
    '''

    if map_b is None:
        map_b = map_a.copy()

    if mask is not None:
        map_a = map_a*mask - np.mean(map_a[mask==1])
        map_b = map_b*mask - np.mean(map_b[mask==1])
    else:
        map_a = map_a - np.mean(map_a)
        map_b = map_b - np.mean(map_b)
        
    l2d, ps2d = get_power_spectrum_2d(map_a, map_b=map_b, pixsize=pixsize)
    
    if lbinedges is None:
        lmin = np.min(l2d[l2d!=0])
        lmax = np.max(l2d[l2d!=0])
        if logbin:
            lbinedges = np.logspace(np.log10(lmin), np.log10(lmax), nbins)
            lbins = np.sqrt(lbinedges[:-1] * lbinedges[1:])
        else:
            lbinedges = np.linspace(lmin, lmax, nbins)
            lbins = (lbinedges[:-1] + lbinedges[1:]) / 2

        lbinedges[-1] = lbinedges[-1]*(1.01)
    
    if weights is None:
        weights = np.ones(ps2d.shape)
    
    Cl = np.zeros(len(lbins))
    Clerr = np.zeros(len(lbins))
    Nmodes = np.zeros(len(lbins),dtype=int)
    for i,(lmin, lmax) in enumerate(zip(lbinedges[:-1], lbinedges[1:])):
        sp = np.where((l2d>=lmin) & (l2d<lmax))
        p = ps2d[sp]
        w = weights[sp]
        Cl[i] = np.sum(p*w) / np.sum(w)
        Clerr[i] = np.std(p) / np.sqrt(len(p))
        Nmodes[i] = len(p)
    
    if return_Dl:
        Cl = Cl * lbins * (lbins+1) / 2 / np.pi
        
    if return_full:
        return lbins, Cl, Clerr, Nmodes, lbinedges, l2d, ps2d
    else:
        return lbins, Cl, Clerr
    
def get_bl(psfmap, l, pixsize=0.7):
    '''
    calculate beam bl function
    
    Inputs:
    =======
    psfmap: psf image
    pixsize: pixsize of psfmap [arcsec]
    l: desired array of ell mode to calculate bl
    
    Outputs:
    ========
    bl: bl function in each mode in l
    '''

    ldat,bldat,_ = get_power_spec(psfmap, pixsize=pixsize)
    bldat /= bldat[0]
    
    bl = np.interp(np.log10(l), np.log10(ldat), np.log10(bldat))
    bl = 10**bl
    bl[l < ldat[0]] = 1
    
    return bl
    
def map_from_Cl(lbins, lbinedges, Cl, mapsize=(1024,1024), pixsize=7, Clinterp=False):
    '''
    generate 2D image from input power specum
    
    Inputs:
    =======
    lbins:
    lbinedges:
    Cl;
    mapsize: dim of output map
    pixsize: output map pixsize [arcsec]
    Clinterp: assign 
    
    Outputs:
    ========
    bl: bl function in each mode in l
    '''
    
    dimx, dimy = mapsize
    sterad_per_pix = (pixsize/3600/180*np.pi)**2
    V = dimx * dimy * sterad_per_pix

    lx = np.fft.fftfreq(dimx)*2*(180*3600./pixsize)
    ly = np.fft.fftfreq(dimy)*2*(180*3600./pixsize)
    ly, lx = np.meshgrid(ly, lx)
    l2d = np.sqrt(lx**2 + ly**2)
    
    if Clinterp:
        Cl2d = np.interp(l2d, lbins, Cl)
    else:
        Cl2d = np.zeros(l2d.shape)
        for i,(lmin, lmax) in enumerate(zip(lbinedges[:-1], lbinedges[1:])):
            Cl2d[(l2d>=lmin) & (l2d<lmax)] = Cl[i]
    
    real_part = np.random.normal(size=l2d.shape)
    im_part = np.random.normal(size=l2d.shape)

    ft_map = (real_part + im_part*1.0j) * np.sqrt(abs(Cl2d * V)) / sterad_per_pix
    map2D = np.fft.ifftn(ft_map)
    map2D = np.real(map2D)
    
    return map2D