from lognormal_counts_yunting import *
from clustering import *
import time

def run_window_sim(run_label, n_catalog=30, zmin=0.01, zmax=1.0, counts_per_sqdeg=5000):
    
    # this is the number of redshift bins you want,
    #if you want to generate several GRFs and superpose them
    ng_bins = 1
    
    ell_min = 90
    Npix = 1024
    deg_pix = 2/Npix

    upscale = 4
    size = Npix*upscale # side length of image to place counts in 
    n_square_deg = (2*upscale)**2

    theta_binedges_arcsec = np.logspace(0.3,3.2,15) # arcsec
    theta_binedges_deg = (theta_binedges_arcsec * u.arcsec).to(u.deg).value # deg
    theta_bins = np.sqrt(theta_binedges_arcsec[1:] * theta_binedges_arcsec[:-1])

    corrs = []
    corr_ins = []
    corr_in_onlys = []

    start_time = time.time()
    
    for icat in range(n_catalog):
        elapse_time = (time.time()-start_time)/60
        print('generating %d/%d catalogs from correlated GRF (%.2f min)'%(icat,n_catalog, elapse_time))
        tx, ty = generate_galaxy_clustering([counts_per_sqdeg], size=size, 
                                            ell_min=ell_min/upscale, n_square_deg=n_square_deg, 
                                            n_catalog=1, ng_bins=ng_bins)
        
        tx, ty = np.array(tx[0]), np.array(ty[0])
        ra = (tx - (Npix*upscale-1)/2) * deg_pix
        dec = (ty - (Npix*upscale-1)/2) * deg_pix

        ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                     (min(dec), max(dec)),
                                     2 * len(ra))

        D_idx = np.where((ra > -1) & (ra < 1) & (dec > -1) & (dec < 1))[0]
        R_idx = np.where((ra_R > -1) & (ra_R < 1) & (dec_R > -1) & (dec_R < 1))[0]

        elapse_time = (time.time()-start_time)/60
        print('...calculate frac (%.2f min)'%(elapse_time))
        f_arr = []
        for ibin in range(len(theta_bins)):
            print(ibin)
            Npix_expected = np.pi*(theta_binedges_arcsec[ibin+1]**2 \
                                   - theta_binedges_arcsec[ibin]**2)/7**2
            Ntot, Nin = 0, 0
            for x,y in zip(tx[D_idx]-1536, ty[D_idx]-1536):
                radmap = make_radius_map(np.zeros([1024,1024]), x, y)*7
                sp = np.where((radmap>theta_binedges_arcsec[ibin]) & \
                             (radmap<theta_binedges_arcsec[ibin+1]))
                Ntot += Npix_expected
                Nin += len(sp[0])

            f_arr.append(Nin / Ntot)
        r_arr = np.array(f_arr)

        elapse_time = (time.time()-start_time)/60
        print('...calculate corr func (%.2f min)'%(elapse_time))
        corr, corr_in = bootsrap_two_point_angular_window([ra, dec], [ra_R, dec_R],
                                                            theta_binedges_deg, D_idx, R_idx)

        corr_in_only = get_angular_2pt_func\
                        (ra[D_idx], dec[D_idx], theta_binedges_deg, nboot=1)

        corrs.append(corr)
        corr_ins.append(corr_in)
        corr_in_onlys.append(corr_in_only)
          
        data = np.stack((np.array(corrs), np.array(corr_ins), 
                         np.array(corr_in_onlys), np.array(f_arr)))
        
        np.save('./wfunc_data/wfunc_sim_run_%d'%(run_label),data)
        
    return data