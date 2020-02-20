import numpy as np
import matplotlib.pyplot as plt
import matplotlib

def imageclip(image, iters=3, vmin=None, vmax=None, ax=None, cbar=True,
              return_objects=False, figsize=(6,5), **kwargs):
    
    if (not vmin is None) and (not vmax is None):
        pass
    elif np.all(image==0):
        vmin, vmax = 0 ,1
    else:
        b = image[(image!=0) & (image!=np.inf) & (image!=-np.inf)]
        if np.min(b) == np.max(b):
            vmin, vmax = 0, 1
        else:
            if len(b) > 5e4:
                b = b[np.random.choice(len(b),int(5e4),replace=False)]
            for i in range(iters):
                if np.std(b) == 0:
                    continue
                clipmin = np.nanmedian(b) - 5*np.std(b)
                clipmax = np.nanmedian(b) + 5*np.std(b)
                b = b[(b<clipmax) & (b>clipmin)]
                
            if np.std(b) == 0:
                pass
            else:
                clipmin = np.nanmedian(b) - 3*np.std(b)
                clipmax = np.nanmedian(b) + 3*np.std(b)
                b = b[(b<clipmax) & (b>clipmin)]
                
            vmin, vmax = np.min(b), np.max(b)
        
    
    objs = {}
    objs['vmin'] = vmin
    objs['vmax'] = vmax
    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)
        objs['fig'] = fig
        objs['ax'] = ax

    p = ax.imshow(image,vmin=vmin, vmax=vmax, cmap='jet', origin='lower', **kwargs)    
    objs['p'] = p
    if cbar:
        cbar = plt.colorbar(p,ax=ax)
        objs['cbar'] = cbar
        
    if return_objects:
        return objs
    else:
        return
    
    
def plot_err_log(x, y, yerr, ax=None, xlog=True, xerr=None, xedges=None, plot_xerr=True, 
                 color='k', capsize=5, markersize=10, figsize=(6,5), alpha=1, label=None):

    if ax is None:
        fig, ax = plt.subplots(1,1, figsize=figsize)

    if plot_xerr and xedges is None:
        if xlog:
            rx = x[1]/x[0]
            xedges = np.concatenate(([x[0]/rx],np.sqrt(x[1:]*x[:-1]),[x[-1]*rx]))
        else:
            rx = x[1]-x[0]
            xedges = np.concatenate(([x[0]-rx],(x[1:]+x[:-1])/2,[x[-1]+rx]))
    
    if plot_xerr:
        x_err_low = x - xedges[:-1]
        x_err_high = xedges[1:] - x
    
    if xerr is not None:
        x_err_low = xerr
        x_err_high = xerr

    spp = np.where(y>=0)[0]
    spn = np.where(y<0)[0]

    if plot_xerr:
        ax.errorbar(x[spp], y[spp], yerr[spp], [x_err_low[spp], x_err_high[spp]],
                    fmt ='.', color=color, capsize=capsize, 
                    markersize=markersize, alpha=alpha, label=label)
        ax.errorbar(x[spn], -y[spn], yerr[spn], [x_err_low[spn], x_err_high[spn]],
                    fmt ='.', mfc='white', color=color, 
                    capsize=capsize, markersize=markersize, alpha=alpha)
    else:
        ax.errorbar(x[spp], y[spp], yerr[spp],
                    fmt ='.', color=color, capsize=capsize, 
                    markersize=markersize, alpha=alpha, label=label)
        ax.errorbar(x[spn], -y[spn], yerr[spn],
                    fmt ='.', mfc='white', color=color, 
                    capsize=capsize, markersize=markersize, alpha=alpha)
    ax.set_yscale('log')
    if xlog:
        ax.set_xscale('log')
    
    return

class OOMFormatter(matplotlib.ticker.ScalarFormatter):
    '''
    Formatting the colorbar tick labels to be in scientific notation.
    https://stackoverflow.com/questions/43324152/python-matplotlib-colorbar-scientific-notation-base?rq=1
    
    cbar = fig.colorbar(plot, format=OOMFormatter(-2, mathText=False))

    '''
    def __init__(self, order=0, fformat="%1.1f", offset=True, mathText=True):
        self.oom = order
        self.fformat = fformat
        matplotlib.ticker.ScalarFormatter.__init__(self,useOffset=offset,useMathText=mathText)
    def _set_orderOfMagnitude(self, nothing):
        self.orderOfMagnitude = self.oom
    def _set_format(self, vmin, vmax):
        self.format = self.fformat
        if self._useMathText:
            self.format = '$%s$' % matplotlib.ticker._mathdefault(self.format)

