import numpy as np

# get filters
def ciber_filter(band, interp = False, interp_lim = [0.3,30]):
    '''
    get ciber filter transmittance
    output:
    wl_arr [um]
    T_arr : transmittance
    '''
    filtdir = '/Users/ytcheng/ciber/ciber_analysis/pro/imagers/paperfigs/'
    if band == 'I':
        data = np.loadtxt(filtdir + 'iband_transmittance.txt', skiprows=1)
        wl_arr = np.asarray(data[:,0])*1e-3
        T_arr = data[:,1]
        data = np.genfromtxt(filtdir + 'Iband_trans_digitize.txt', dtype=None, delimiter=',') 
        T_arr = np.asanyarray([T_arr[i]*data[i][1] for i,_ in enumerate(data)])*1e-4
    elif band == 'H':
        data = np.loadtxt(filtdir + 'hband_transmittance.txt', skiprows=1)
        wl_arr = np.asarray(data[:,0])*1e-3
        T_arr = data[:,1]
        data = np.genfromtxt(filtdir + 'Hband_trans_digitize.txt', dtype=None, delimiter=',') 
        T_arr = np.asanyarray([T_arr[i]*data[i][1] for i,_ in enumerate(data)])*1e-4
    else:
        print('input band incorrect')
        
    if interp:
        wl_data = wl_arr
        T_data = T_arr
        wl_arr = np.arange(interp_lim[0],interp_lim[1],0.001)
        T_arr = np.zeros_like(wl_arr)
        sp = np.where((wl_arr > wl_data[0]) & (wl_arr < wl_data[-1]))[0]
        T_arr[sp] = np.interp(wl_arr[sp],wl_data,T_data)        
        
    return wl_arr, T_arr

def wise_filter(band, interp = False, interp_lim = [0.3,30]):
    '''
    get wise filter transmittance
    output:
    wl_arr [um]
    T_arr : transmittance
    '''

    filtdir = '/Users/ytcheng/ciber/lephare/lephare_dev/filt/wise/'
    
    if band not in ['w1','w2','w3','w4']:
        print('input band incorrect')
        return
    data = np.loadtxt(filtdir + band + '.pb', skiprows=1)
    wl_arr = np.asarray(data[:,0])*1e-4
    T_arr = np.asarray(data[:,1])

    if interp:
        wl_data = wl_arr
        T_data = T_arr
        wl_arr = np.arange(interp_lim[0],interp_lim[1],0.001)
        T_arr = np.zeros_like(wl_arr)
        sp = np.where((wl_arr > wl_data[0]) & (wl_arr < wl_data[-1]))[0]
        T_arr[sp] = np.interp(wl_arr[sp],wl_data,T_data)
            
    return wl_arr, T_arr

def panstarrs_filter(band, interp = False, interp_lim = [0.3,30]):
    '''
    get panstarrs filter transmittance
    output:
    wl_arr [um]
    T_arr : transmittance
    '''

    filtdir = '/Users/ytcheng/ciber/lephare/lephare_dev/filt/panstarrs/'
    
    if band not in ['g','r','i','z','y']:
        print('input band incorrect')
        return
    data = np.loadtxt(filtdir + band + '.pb', skiprows=1)
    wl_arr = np.asarray(data[:,0])*1e-4
    T_arr = np.asarray(data[:,1])

    if interp:
        wl_data = wl_arr
        T_data = T_arr
        wl_arr = np.arange(interp_lim[0],interp_lim[1],0.001)
        T_arr = np.zeros_like(wl_arr)
        sp = np.where((wl_arr > wl_data[0]) & (wl_arr < wl_data[-1]))[0]
        T_arr[sp] = np.interp(wl_arr[sp],wl_data,T_data)
            
    return wl_arr, T_arr

# get SED
def load_sed(filename, interp = True, interp_lim = [0.3,30], z = 0):
    '''
    return the SED in filename. The SED has arbitrary normalization. 
    if interp: interpolate to interp_lim um with linear equal spacing 0.001 um.
    output:
    wl_arr[um]
    f: F_lambda [dE/dt/dlambda/dA]
    '''
    wl_arr = np.arange(interp_lim[0],interp_lim[1],0.001)
    f = np.loadtxt(filename)
    wl_data = f[:,0]*1e-4 * (1. + z)
    f_data = f[:,1] / (1. + z)**2
    
    if interp:
        if min(wl_data) > interp_lim[0]:
            print('min wavelength in data is %.2f > %.2f'%(min(wl_data),interp_lim[0]))
        if max(wl_data) < interp_lim[1]:
            print('max wavelength in data is %.2f < %.2f'%(min(wl_data),interp_lim[0]))

        f_arr = np.interp(wl_arr,wl_data,f_data)
    
    else:
        wl_arr = wl_data
        f_arr = f_data
    return wl_arr, f_arr