import numpy as np
import matplotlib.pyplot as plt
import rld
import pickle

pix_size = 6. # arcseconds

if False:
    # read from the giant database of SPHEREx stacks
    fp = open('SPHEREx_stacks2.pkl','rb')
    Stack = pickle.load(fp)
    fp.close()

    index = '/Users/bcrill/Downloads/Twocases_SPHEREXPSFs/CBE/SPHEREX_band1_x-24mm_y-17mm_lam0750nm_wfe154nm_case039.fits'
    # this is the example stack we'll use
    Example_Stack = Stack[index]['PSF_jitter']
    bin_edge = Stack['bin_edge']
    
    # and save an example
    fp = open("Example_Stack.pkl","wb")
    pickle.dump(Example_Stack,fp)
    pickle.dump(bin_edge,fp)
    fp.close()
else:
    # Or... just read the example
    fp = open('Example_Stack.pkl','rb')
    Example_Stack = pickle.load(fp)
    bin_edge = pickle.load(fp)
    fp.close()
    
# build a pixel shape
pix_size = 6.

apixel = np.zeros(np.shape(Example_Stack))
bin_edge = bin_edge*pix_size
xx,yy = np.meshgrid((bin_edge[1:] + bin_edge[:-1])/2.0,(bin_edge[1:] + bin_edge[:-1])/2.0)

ii = np.where((xx<0.5*pix_size)&(xx>=-0.5*pix_size)&(yy>=-0.5*pix_size)&(yy<0.5*pix_size))
apixel[ii] = 1.0

Deconvolved_Stack = rld.rld(Example_Stack,apixel,niter=25)

plt.pcolormesh(Deconvolved_Stack)
plt.show()
