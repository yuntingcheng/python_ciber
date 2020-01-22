from lognormal_counts_yunting import *
from clustering import *
import time

def run_window_sim(run_label, n_catalog=30, zmin=0.01, zmax=1.0, counts_per_sqdeg=5000):
    
    # this is the number of redshift bins you want,
    #if you want to generate several GRFs and superpose them
    ng_bins = 1
    # counts per square degree
    number_counts = [counts_per_sqdeg]*n_catalog 
    
    ell_min = 90
    Npix = 1024
    deg_pix = 2/Npix

    upscale = 4
    size = Npix*upscale # side length of image to place counts in 
    n_square_deg = (2*upscale)**2
    
    start_time = time.time()
    
    # generate the GRFs and catalogs
    print('generating %d catalogs from correlated GRF'%n_catalog)
    tx, ty = generate_galaxy_clustering(number_counts, size=size, 
                                        ell_min=ell_min/upscale, n_square_deg=n_square_deg, 
                                        n_catalog=n_catalog, ng_bins=ng_bins)
    
    theta_binedges_arcsec = np.logspace(0.3,3.2,15) # arcsec
    theta_binedges_deg = (theta_binedges_arcsec * u.arcsec).to(u.deg).value # deg
    theta_bins = np.sqrt(theta_binedges_arcsec[1:] * theta_binedges_arcsec[:-1])

    corrs = []
    corr_ins = []
    corr_in_onlys = []

    for icat in range(n_catalog):
        elapse_time = (time.time()-start_time)/60
        print('calculate correlation function for catalog %d/%d, %.4f min'%(icat, n_catalog, elapse_time))
        ra = (np.array(tx[icat]) - (Npix*upscale-1)/2) * deg_pix
        dec = (np.array(ty[icat]) - (Npix*upscale-1)/2) * deg_pix

        ra_R, dec_R = uniform_sphere((min(ra), max(ra)),
                                     (min(dec), max(dec)),
                                     2 * len(ra))

        D_idx = np.where((ra > -1) & (ra < 1) & (dec > -1) & (dec < 1))[0]
        R_idx = np.where((ra_R > -1) & (ra_R < 1) & (dec_R > -1) & (dec_R < 1))[0]


        corr, corr_in = bootsrap_two_point_angular_window([ra, dec], [ra_R, dec_R],
                                                            theta_binedges_deg, D_idx, R_idx)

        corr_in_only = get_angular_2pt_func\
                        (ra[D_idx], dec[D_idx], theta_binedges_deg, nboot=1)

        corrs.append(corr)
        corr_ins.append(corr_in)
        corr_in_onlys.append(corr_in_only)
    
    corrs = np.array(corrs)
    corr_ins = np.array(corr_ins)
    corr_in_onlys = np.array(corr_in_onlys)
    
    data = np.stack((corrs, corr_ins, corr_in_onlys))
    
    np.save('./wfunc_data/wfunc_sim_run_%d'%(run_label),data)
    
    return data