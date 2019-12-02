import numpy as np
from utils import *

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
    