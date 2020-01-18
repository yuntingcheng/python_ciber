import numpy as np
import matplotlib.pyplot as plt

def imageclip(image, iters=3, vmin=None, vmax=None, ax=None, cbar=True,
              return_objects=False, figsize=(6,5), **kwargs):
    
    if (not vmin is None) and (not vmax is None):
        pass
    elif np.all(image==0):
        vmin, vmax = 0 ,1
    else:
        b = image[image!=0]
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