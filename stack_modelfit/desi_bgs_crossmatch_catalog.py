%run desi_bgs_target_selection_functions.py
%matplotlib inline 
from __future__ import division


# --------------------- SDSS ---------------------------------
sdss_data_all = pd.read_csv(str(string) + '/' + str(string) + '_all.txt')
sdss_all_sort = sdss_data_all.sort_values(by='objID', ascending=[True])
sdss_positions = (sdss_all_sort[["ra", "dec"]].values)*(np.pi/180)
sdss_all = (sdss_all_sort).values
sdss_all_gen = np.array(sdss_all).transpose() #for general cross match


# ------------------------- Gaia ------------------------------------
gaia_data_all = pd.read_csv(str(string) + '/gaia_witherrs_astroexcess.txt')
gaia_all = gaia_data_all.values
gaia_pos = (gaia_data_all[["ra", "dec"]].values)*(np.pi/180)
gaia_all_gen = np.array(gaia_all).transpose()

# ------------------------------ WISE ----------------------------------------- 
# this isn't actually going to be necessary, since Wise W1 and W2 are in the DECaLS data releases already. 

wise_cat = pd.read_csv(str(string) + '/' + 'wise_catalog.csv', header=None, delim_whitespace=True)
wise_cat = wise_cat.values
wise_cat_new = wise_cat

wise_cat = np.array(wise_cat).transpose()
wise_ra = np.array(wise_cat[0])*np.pi/180
wise_dec = np.array(wise_cat[1])*np.pi/180
wise_sigradec = wise_cat[2]
wise_w1mag = wise_cat[3]
wise_w1unc = wise_cat[4]
wise_w2mag = wise_cat[5]
wise_w2unc = wise_cat[6]
wise_psf_flag = wise_cat[7]
wise_radec = np.array([wise_ra, wise_dec]).transpose()
wise_cat2 = np.array([wise_ra, wise_dec, wise_sigradec]).transpose()



# constructing catalog with positions in radians for gaia/wise/sdss
catalog_gaia = []
for source in (gaia_all):
    vector1 = [source[1]*np.pi/180, source[2]*np.pi/180]
    vector2 = [source[i] for i in xrange(len(source)) if i>2]
    vector3 = vector1 + vector2
    catalog_gaia.append(vector3)

catalog_wise = []
for source in wise_cat_new:
    vector1 = [source[0]*np.pi/180, source[1]*np.pi/180]
    vector2 = [source[i] for i in xrange(len(source)) if i>1]
    vector3 = vector1 + vector2
    catalog_wise.append(vector3)
    
catalog_sdss = []
for source in sdss_all:
    catalog_sdss.append([source[0]*np.pi/180, source[1]*np.pi/180, source[2]])


# DECaLS data from sweep file, 
decals_sweep = fits.open('./sweep-000m005-010p000.fits', mmap=True) # 340 < RA < 350, 0 < DEC < 5
decals_data = decals_sweep[1].data
decals_sweep.close()

ras = np.array(decals_data[:]['RA'])
decs = np.array(decals_data[:]['DEC'])
source_type = np.array(decals_data[:]['TYPE'])
flux = np.array(decals_data[:]['DECAM_FLUX'])
decam_mw = np.array(decals_data[:]['DECAM_MW_TRANSMISSION'])
wise_flux = np.array(decals_data[:]['WISE_FLUX'])
wise_mw = np.array(decals_data[:]['WISE_MW_TRANSMISSION'])
tycho_in_blob = np.array(decals_data[:]['TYCHO2INBLOB'])
dchisq = np.array(decals_data[:]['DCHISQ'])


decals_list = [ras*np.pi/180, decs*np.pi/180, source_type, flux, decam_mw, wise_flux, wise_mw, tycho_in_blob, dchisq]
print('Creating restricted area catalog...')
decals_all = restricted_area_catalog(ra_min*np.pi/180,ra_max*np.pi/180,dec_min*np.pi/180,dec_max*np.pi/180,decals_list)


decals_1620_all = filter_decals_mags(decals_all, 16, 20)

# ------------- deredden fluxes and convert to magnitudes
all_mags = []
for i in range(len(decals_1620_all[3])):
    r = decals_1620_all[3][i][2]*decals_1620_all[4][i][2]
    g = decals_1620_all[3][i][1]*decals_1620_all[4][i][1]
    z = decals_1620_all[3][i][4]*decals_1620_all[4][i][4]
    u = decals_1620_all[3][i][0]*decals_1620_all[4][i][0]
    w1 = decals_1620_all[5][i][0]*decals_1620_all[6][i][0]
    w2 = decals_1620_all[5][i][1]*decals_1620_all[6][i][1]
    
    fluxes = [u,g,r,z,w1,w2]
    mags = []
    for j in fluxes:
        mag = nanomaggy_to_mag(j)
        mags.append(mag)
    all_mags.append(mags)


catalog_bright = []
# --------------Gaia--------------- Bright Sources ----------------------------
gaia_bright = pd.read_csv(str(string) + '/gaia_lt_11_witherrors.csv')
gaia_bright = gaia_bright.values

for source in gaia_bright:
    catalog_bright.append([source[1]*np.pi/180, source[2]*np.pi/180, exclusion_radius_gaia(source[5])*(np.pi/180/3600)])

tychos = pd.read_csv(str(string) + '/' + str(string) + '_tycho_stars.csv')
tycho_data = tychos.values
for source in tycho_data:
    catalog_bright.append([source[0]*np.pi/180, source[1]*np.pi/180, exclusion_radius_elg(source[3])*(np.pi/180/3600)])


catalog = []
indices = [0, 1, 2, 7]
indices2 = [2,7]
for i in range(len(decals_1620_all[3])):
    vector = []
    for j in indices:
        vector.append(decals_1620_all[j][i])
    for k in range(5):
        vector.append(decals_1620_all[8][i][k])
    for j in all_mags[i]:
        vector.append(float(j))
    catalog.append(vector)    
# ---- initialize bits for gaia match, wise match, sdss match as False
for i in catalog:
    for j in range(3):
        i.append(False)

plt.figure()
plt.hist([len(x) for x in catalog])
plt.yscale('log')
plt.xlabel('Length of catalog source')
plt.show()

# run cross matching routine for d (DECaLS), g (Gaia), w (WISE), and s (SDSS) respectively. 
# crossmatches are based off of DECaLS positions wrt Gaia/WISE/SDSS in each case (takes 0, 1 indices of source parameter vector)
# bit specifies the index for crossmatch flag to be set to True if there is a cross match. 
catalog_dg = general_cross_match(max_radius, catalog, catalog_gaia, bit=15, nfalse=4)
catalog_dgw = general_cross_match(max_radius, catalog_dg, wise_cat2, bit=16, nfalse=1)
catalog_dgws = general_cross_match(max_radius, catalog_dgw, catalog_sdss, bit=17, nfalse=1)

# new catalog with bit flag for proximity to bright star
cat = flag_near_brightstar(catalog_dgws, catalog_bright)

mags = [10,11,12,13]
cate = deepcopy(cat)
for i in mags:
    cate = [x for x in cate if np.isnan(x[i])==False and np.isinf(x[i])==False]


# specify catalog of dereddened DECaLS 'PSF' sources with r-band magnitude between 17 and 19 and WISE counterpart
rwisecatalog = [x for x in cate if x[11] < 19 and x[11] > 17 and x[16]==True and x[2][0]=='P' and x[24]==False]
rwises = np.array([x[11]-x[13] for x in rwisecatalog])
gzs = np.array([x[10]-x[12] for x in rwisecatalog])

# similar as above but for Gaia and r-band
gaiarcat = [x for x in cate if x[11] < 19 and x[11] > 17 and x[15]==True and x[2][0]=='P']
gaiar = np.array([x[19]-x[11] for x in gaiarcat])
gzgaiar = np.array([x[10]-x[12] for x in gaiarcat])

#iterative sigma clipping for Delta(Gaia) and Delta(W1)
rwise_gz_2sig_fit = rwise_gz_fit(gzs, rwises, nsig=2, plot='yes', save='no') 
gaiar_gz_2sig_fit = gaiar_gz_fit(gzgaiar, gaiar, nsig=2, plot='yes', save='no')


cata = deepcopy(cate)
gaiapred = piecewise_linear([i[10]-i[12] for i in cata], *gaiar_gz_2sig_fit) + np.array([i[11] for i in cata])
rwisepred = rwise_gz_2sig_fit3([i[10] - i[12] for i in cata])
delgaia, delrwise = [[],[]]

for i in range(len(cata)):
    if cata[i][19]==False:
        cata[i].append(20.7 - gaiapred[i]) #if there's no Gaia match then set lower bound
        kop = False
    else:
        cata[i].append(cata[i][19]-gaiapred[i])
        kop = astrometric_excess_cut(cata[i][19])
            
    cata[i].append((cata[i][11]-cata[i][13])-rwisepred[i])
    if kop==False:
        cata[i].append(nan)
    else:
        if cata[i][21] > kop:
            cata[i].append(False)
        else:
            cata[i].append(True)
            
