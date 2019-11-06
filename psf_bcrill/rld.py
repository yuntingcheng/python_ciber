import numpy as np
from scipy.signal import convolve2d

def width(data):
    total = np.abs(data).sum()
    X, Y = np.indices(data.shape)
    x = (X*np.abs(data)).sum()/total
    y = (Y*np.abs(data)).sum()/total
    col = data[:, int(y)]
    width_x = np.sqrt(np.abs((np.arange(col.size)-y)**2*col).sum()/np.abs(col).sum())
    row = data[int(x), :]
    width_y = np.sqrt(np.abs((np.arange(row.size)-x)**2*row).sum()/np.abs(row).sum())
    return ( width_x + width_y ) / 2.

def centroid(data):
    total = np.abs(data).sum()
    X, Y = np.indices(data.shape)
    x = (X*np.abs(data)).sum()/total
    y = (Y*np.abs(data)).sum()/total
    return (x,y)

def rld(d,p,niter=25,verbose=True):
    '''Richardson-Lucy Deconvolution
    d = data
    p = PSF
    niter = number of iterations
    verbose=True will compute an estimate of the width of the image based on the rms
    '''
    u_t = d
    p_mirror = p[::-1, ::-1]
    for iiter in np.arange(niter):
        c = convolve2d(u_t,p,mode='same')
        u_t1 = u_t * convolve2d (d / c , p_mirror, mode='same')
        u_t = u_t1
        if verbose:
            print(width(u_t))
    return u_t
