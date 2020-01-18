import numpy as np
from hankel import SymmetricFourierTransform
from scipy import interpolate
from scipy import stats
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from hmf import MassFunction
from astroML.correlation import bootstrap_two_point_angular, two_point_angular

cosmo = FlatLambdaCDM(H0=70, Om0=0.28)
# cosmo = FlatLambdaCDM(H0=67.74, Om0=0.28)

mass_function = MassFunction(z=0., dlog10m=0.02)

def counts_from_density_2d(density_fields, Ntot = 200000, ell_min=90.):
    counts_map = np.zeros_like(density_fields)
    
    surface_area = (np.pi/ell_min)**2

    # compute the mean number density per cell 
    N_mean = float(Ntot) / float(counts_map.shape[-2]*counts_map.shape[-1])
    
    # calculate the expected number of galaxies per cell
    expectation_ngal = (density_fields+1.)*N_mean 
    # generate Poisson realization of 2D field
    count_maps = np.random.poisson(expectation_ngal)
    
    dcounts = np.array([int(np.sum(ct_map)-Ntot) for ct_map in count_maps])
    # if more counts in the poisson realization than Helgason number counts, subtract counts
    # from non-zero elements of array.
    # The difference in counts is usually at the sub-percent level, so adding counts will only slightly
    # increase shot noise, while subtracting will slightly decrease power at all scales
    for i in np.arange(count_maps.shape[0]):
        if dcounts[i] > 0:
            nonzero_x, nonzero_y = np.nonzero(count_maps[i])
            rand_idxs = np.random.choice(np.arange(len(nonzero_x)), dcounts[i], replace=True)
            count_maps[i][nonzero_x[rand_idxs], nonzero_y[rand_idxs]] -= 1
        elif dcounts[i] < 0:
            randx, randy = np.random.choice(np.arange(count_maps[i].shape[-1]), size=(2, np.abs(dcounts[i])))
            for j in range(np.abs(dcounts[i])):
                counts_map[i][randx[j], randy[j]] += 1

    return count_maps

''' This function generates the indices needed when generating G(k), which gets Fourier transformed to a gaussian
random field with some power spectrum.'''
def fftIndgen(n):
#     a = range(0, n//2+1)
#     b = range(1, n//2)
#     b.reverse()
#     b = [-i for i in b]
    a = np.arange(0, n//2+1)
    b = np.arange(1, n//2)
    b = -np.flip(b,0)
    return np.concatenate((a,b))

def gaussian_random_field_2d(n_samples, alpha=None, size=128, ell_min=90., cl=None, ell_sampled=None, fac=1, plot=False):
    
    up_size = size*fac
    steradperpixel = ((np.pi/ell_min)/up_size)**2
    surface_area = (np.pi/ell_min)**2

    cl_g, spline_cl_g = hankel_spline_lognormal_cl(ell_sampled, cl)
    ls = make_ell_grid(up_size, ell_min=ell_min)
    amplitude = 10**spline_cl_g(np.log10(ls))
    amplitude /= surface_area
    
    amplitude[0,0] = 0.
    
    noise = np.random.normal(size = (n_samples, up_size,up_size)) + 1j * np.random.normal(size = (n_samples, up_size, up_size))
    gfield = np.array([np.fft.ifft2(n * amplitude).real for n in noise])
    gfield /= steradperpixel
    
    # up to this point, the gaussian random fields have mean zero
    return gfield, amplitude, ls

def generate_count_map_2d(n_samples, ell_min=90., size=128, Ntot=2000000, cl=None, ell_sampled=None, plot=False, save=False):
    # we want to generate a GRF twice the size of our final GRF so we include larger scale modes
    realgrf, amp, k = gaussian_random_field_2d(n_samples, size=size, cl=cl, ell_sampled=ell_sampled, ell_min=ell_min/2.,fac=2)
    realgrf = realgrf[:,int(0.5*size):int(1.5*size),int(0.5*size):int(1.5*size)]
    
    density_fields = np.array([np.exp(grf-np.std(grf)**2)-1. for grf in realgrf]) # assuming a log-normal density distribution
    
    counts = counts_from_density_2d(density_fields, Ntot=Ntot)

    return counts, density_fields


''' This function takes an angular power spectrum and computes the corresponding power spectrum for the log 
of the density field with that power spectrum. This involves converting the power spectrum to a correlation function, 
computing the lognormal correlation function, and then transforming back to ell space to get C^G_ell.'''
def hankel_spline_lognormal_cl(ells, cl, plot=False, ell_min=90, ell_max=1e5):
    
    ft = SymmetricFourierTransform(ndim=2, N = 200, h = 0.03)
    # transform to angular correlation function with inverse hankel transform and spline interpolated C_ell
    spline_cl = interpolate.InterpolatedUnivariateSpline(np.log10(ells), np.log10(cl))
    f = lambda ell: 10**(spline_cl(np.log10(ell)))
    thetas = np.pi/ells
    w_theta = ft.transform(f ,thetas, ret_err=False, inverse=True)
    # compute lognormal angular correlation function
    w_theta_g = np.log(1.+w_theta)
    # convert back to multipole space
    spline_w_theta_g = interpolate.InterpolatedUnivariateSpline(np.log10(np.flip(thetas,0)), np.flip(w_theta_g,0))
    g = lambda theta: spline_w_theta_g(np.log10(theta))
    cl_g = ft.transform(g ,ells, ret_err=False)
    spline_cl_g = interpolate.InterpolatedUnivariateSpline(np.log10(ells), np.log10(np.abs(cl_g)))
    
    # plotting is just for validation if ever unsure
    if plot:
        plt.figure()
        plt.loglog(ells, cl, label='$C_{\\ell}$', marker='.')
        plt.loglog(ells, cl_g, label='$C_{\\ell}^G$', marker='.')
        plt.loglog(np.linspace(ell_min, ell_max, 1000), 10**spline_cl_g(np.log10(np.linspace(ell_min, ell_max, 1000))), label='spline of $C_{\\ell}^G$')
        plt.legend(fontsize=14)
        plt.xlabel('$\\ell$', fontsize=16)
        plt.title('Angular power spectra', fontsize=16)
        plt.show()
    
    return cl_g, spline_cl_g



# this is for projecting a 3D power spectrum to a 2D angular power spectrum.

def limber_project(halo_ps, zmin, zmax, ng=None, flux_prod_rate=None, nbin=20, ell_min=90, ell_max=1e5, n_ell_bin=30):
    ''' This currently takes in the dark matter halo power spectrum from the hmf package. The only place where this is
    used is when updating the redshift of the power spectrum. Should add option to use array of power spectra as well'''
    cls = np.zeros((nbin, n_ell_bin))    

    ell_space = 10**(np.linspace(np.log10(ell_min), np.log10(ell_max), n_ell_bin))
    zs = np.linspace(zmin, zmax, nbin+1)
    dz = zs[1]-zs[0]
    central_zs = 0.5*(zs[1:]+zs[:-1])
    
    D_A = cosmo.angular_diameter_distance(central_zs)/cosmo.h # has units h^-1 Mpc
    H_z = cosmo.H(central_zs) # has units km/s/Mpc (or km/s/(h^-1 Mpc)?)
    inv_product = (cosmo.h*dz*H_z)/(const.c.to('km/s')*(1.+central_zs)*D_A**2)
    
    for i in range(nbin):
        halo_ps.update(z=central_zs[i])
        ks = ell_space / (cosmo.comoving_distance(central_zs[i]))
        log_spline = interpolate.InterpolatedUnivariateSpline(np.log10(halo_ps.k), np.log10(halo_ps.nonlinear_power))    
        ps = 10**(log_spline(np.log10(ks.value)))
        cls[i] = ps

    ''' One can compute the auto and cross power spectrum for intensity maps/tracer catalogs 
    by specifying flux_prod_rate and/or ng (average number of galaxies per steradian)''' 
    
    if flux_prod_rate is not None:
        if ng is not None:
            cls = np.array([flux_prod_rate[i]*ng**(-1)*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
        else:
            cls = np.array([flux_prod_rate[i]**2*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
    else:
        if ng is not None:
            cls = np.array([ng**(-2)*inv_product[i]*cls[i] for i in range(inv_product.shape[0])])
        else:
            cls = np.array([inv_product[i]*cls[i] for i in range(inv_product.shape[0])])

    integral_cl = np.sum(cls, axis=0)

    return ell_space, integral_cl

''' This is used when making gaussian random fields '''
def make_ell_grid(size, ell_min=90.):
    ell_grid = np.zeros(shape=(size,size))
    size_ind = np.array(fftIndgen(size))
    for i, sx in enumerate(size_ind):
        ell_grid[i,:] = np.sqrt(sx**2+size_ind**2)*ell_min
    return ell_grid

''' Given a counts map, generate source catalog positions consistent with those counts. 
Not doing any subpixel position assignment or anything like that.''' 
def positions_from_counts(counts_map, cat_len=None):
    thetax, thetay = [], []
    for i in np.arange(np.max(counts_map)):
        pos = np.where(counts_map > i)
        thetax.extend(pos[0].astype(np.float))
        thetay.extend(pos[1].astype(np.float))

    if cat_len is not None:
        idxs = np.random.choice(np.arange(len(thetax)), cat_len)
        thetax = np.array(thetax)[idxs]
        thetay = np.array(thetay)[idxs]

    return np.array(thetax), np.array(thetay)




def generate_positions(Nsrc, size, nmaps, ell_min=90., random_positions=False,
                       cl=None, ells=None, return_grf=False):
    if random_positions:
        txs = np.random.uniform(0, size, (nmaps, Nsrc))
        tys = np.random.uniform(0, size, (nmaps, Nsrc))

    else:
        txs, tys = [], []
        # draw galaxy positions from GRF with given power spectrum, number_counts in deg**-2
        counts, grfs = generate_count_map_2d(nmaps, cl=cl, size=size, 
                                             ell_min=ell_min, Ntot=Nsrc, ell_sampled=ells)

        # uncomment these lines to look at the density fields generated
#         for i in range(counts.shape[0]):
#             plt.figure()
#             plt.imshow(grfs[i])
#             plt.colorbar()
#             plt.show()
        
        for i in range(counts.shape[0]):
            tx, ty = positions_from_counts(counts[i])
            txs.append(tx)
            tys.append(ty)
    
    if return_grf:
        return txs, tys, grfs
    else:
        return txs, tys

def make_galaxy_cts_map(cat, refmap_shape, normalize=True):
    # this function is for visualizing the galaxy counts map
    gal_map = np.zeros(shape=refmap_shape)

    cat = np.array([src for src in cat if src[0]<refmap_shape[0] and src[1]<refmap_shape[1]])
   
    for src in cat:
        gal_map[int(src[0]),int(src[1])] +=1.

    if normalize:
        gal_map /= np.mean(gal_map)
        gal_map -= 1.
    
    return gal_map

def generate_galaxy_clustering(number_counts, size=1024, \
                               # number counts should be an array corresponding to the desired redshift bins
                               zmin=0.01, zmax=1.0, ng_bins=3, \
                               # set redshift range 
                               n_square_deg=4.0, n_catalog=1, \
                               # set FOV in square degrees and number of clustering realizations
                               random_positions=False, \
                               # set to True if you want just shot noise
                               cl = None, \
                                # leave this as None and the function will compute C_ell from the halo model
                                ell_min=90.,\
                              return_grf=False):
                              # size is just the side length of the simulated image, but 
    zrange_grf = np.linspace(zmin, zmax, ng_bins+1) # ng_bins+1 since we take midpoints
    midzs = 0.5*(zrange_grf[:-1]+zrange_grf[1:])
    dzs = zrange_grf[1:]-zrange_grf[:-1] 
    thetax_list = [[] for x in range(n_catalog)]
    thetay_list = [[] for x in range(n_catalog)]
    grfs_list = []
    
    for i, z in enumerate(midzs):

        if cl is None:
            # ell_min and ell_max don't need to be the exact range of the FOV, just need to cover the range
            # the limber project ell_min/ell_max are separate from the one used to generate positions
            ells, cl = limber_project(mass_function, zrange_grf[i], zrange_grf[i+1],
                                      ell_min=22.5, ell_max=3e5)
        if return_grf:
            txs, tys, grfs = generate_positions(number_counts[i]*n_square_deg, size, n_catalog, \
                                             ell_min=ell_min, random_positions=random_positions, \
                                            cl=cl, ells=ells, return_grf=return_grf)
            
            grfs_list.append(grfs)
        else:
            txs, tys = generate_positions(number_counts[i]*n_square_deg, size, n_catalog, \
                                             ell_min=ell_min, random_positions=random_positions, \
                                            cl=cl, ells=ells, return_grf=return_grf)
            
            

        for cat in range(n_catalog):
            thetax_list[cat].extend(txs[cat])
            thetay_list[cat].extend(tys[cat])
    
    if return_grf:  
        return thetax_list, thetay_list, grfs_list
    else:
        return thetax_list, thetay_list
    
# def get_angular_2pt_func(cat_celestial, bins, nboot=1):

#     print(cat_celestial.shape)
#     if nboot==1:
#         print('No bootstrapping')
#         corr = two_point_angular(cat_celestial[:,0], cat_celestial[:,1], bins)
#         return corr, [], []
#     else:
#         corr, dcorr, boot = bootstrap_two_point_angular(cat_celestial[:,0], 
#                                                         cat_celestial[:,1], bins, 
#                                                         method='landy-szalay', Nbootstraps=nboot)
#         return corr, dcorr, boot

def linear_ang2pt(thetas, theta0=0.01666, A=0.8):
    return A*(thetas/theta0)**(-0.8)
